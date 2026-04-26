"""A1 sim2sim (IsaacGym policy -> MuJoCo) 运行脚本。

主要目标：
1) 保持训练时观测结构与缩放一致；
2) policy 低频更新（50Hz），PD 高频闭环（每个 mj_step）；
3) 自动处理 MuJoCo 与 policy 的关节顺序不一致问题；
4) 支持 /joy 实时控制，必要时可 fallback 固定命令。
"""

import os
import time
from typing import Optional, Tuple

import mujoco
import mujoco_viewer
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import ExternalShutdownException
    from sensor_msgs.msg import Joy
    ROS2_AVAILABLE = True
except ImportError:
    rclpy = None
    Node = object
    ExternalShutdownException = Exception
    Joy = None
    ROS2_AVAILABLE = False

try:
    import rospy
    ROS1_AVAILABLE = True
except ImportError:
    rospy = None
    ROS1_AVAILABLE = False

# 训练策略期望的关节顺序（来自你训练时打印的顺序）。
# 注意：若与 MuJoCo actuator 顺序不同，会自动做重排映射。
POLICY_JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

# 与训练配置保持一致（a1_config.py 的 default_joint_angles）
DEFAULT_DOF_POS = np.array([
    0.1, 0.8, -1.5,   # FL
    -0.1, 0.8, -1.5,  # FR
    0.1, 1.0, -1.5,   # RL
    -0.1, 1.0, -1.5,  # RR
], dtype=np.float64)

# A1RoughCfg.control.action_scale = 0.25
ACTION_SCALE = 0.25
CLIP_ACTIONS = 100.0

# 全局命令缓存：[vx, vy, yaw]
joy_cmd = np.zeros(3, dtype=np.float64)
last_joy_update_time = 0.0


class ObsScales:
    lin_vel = 2.0
    ang_vel = 0.25
    dof_pos = 1.0
    dof_vel = 0.05


class SimCfg:
    """仿真与模型路径配置。"""
    mujoco_model_path = '/home/qyw/rl_project/legged_gym/resources/robots/a1/xml/scene.xml'
    policy_model_path = '/home/qyw/rl_project/legged_gym/onnx/legged.onnx'
    sim_duration = 60.0
    dt = 0.005
    decimation = 4  # 50Hz policy 更新
    clip_observations = 100.0
    enable_realtime_sync = True


class CommandCfg:
    """命令输入配置：优先 /joy，必要时 fallback。"""
    enable_fallback_cmd = False
    joy_timeout_s = 0.5
    fallback_cmd = np.array([0.6, 0.0, 0.0], dtype=np.float64)
    joy_deadband = 0.05


class RobotCfg:
    """控制器参数。"""
    kp = 20.0
    kd = 0.5
    # 与 URDF effort limit 对齐：hip=20, thigh=55, calf=55
    tau_limit = np.array([
        20.0, 55.0, 55.0,
        20.0, 55.0, 55.0,
        20.0, 55.0, 55.0,
        20.0, 55.0, 55.0,
    ], dtype=np.float64)

def joy_callback(joy_msg):
    """/joy 回调：将摇杆输入映射到 [vx, vy, yaw]。"""
    global joy_cmd, last_joy_update_time
    joy_cmd[0] = joy_msg.axes[1]
    joy_cmd[1] = joy_msg.axes[0]
    joy_cmd[2] = joy_msg.axes[3]
    last_joy_update_time = time.perf_counter()


class JoySubscriberNode(Node):
    def __init__(self):
        super().__init__('sim2sim_joy_subscriber')
        self.create_subscription(Joy, '/joy', joy_callback, 10)

def quat_rotate_inverse(q, v):
    """使用四元数逆旋转向量。

    参数:
      q: [x,y,z,w]
      v: [x,y,z]
    """
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c

def get_imu_obs(data):
    """读取 IMU 相关观测并返回 (quat_xyzw, omega, lin_vel)。"""
    # MuJoCo framequat: [w, x, y, z] -> [x, y, z, w]
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.float64)
    omega = data.sensor('angular-velocity').data.astype(np.float64)
    lin_vel = data.sensor('linear-velocity').data.astype(np.float64)
    return quat, omega, lin_vel

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD 控制器：tau = kp*(q* - q) + kd*(dq* - dq)。"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def build_joint_mapping(model, policy_joint_names):
    """构建 MuJoCo actuator 顺序 <-> policy 顺序 映射。"""
    mj_joint_ids = []
    mj_joint_names = []

    for act_id in range(model.nu):
        j_id = int(model.actuator_trnid[act_id, 0])
        if j_id < 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
        mj_joint_ids.append(j_id)
        mj_joint_names.append(name)

    if len(mj_joint_names) != 12:
        raise RuntimeError(f"期望12个actuated joints，实际得到{len(mj_joint_names)}: {mj_joint_names}")

    policy_name_to_idx = {n: i for i, n in enumerate(policy_joint_names)}
    missing = [n for n in mj_joint_names if n not in policy_name_to_idx]
    if missing:
        raise RuntimeError(f"以下 MuJoCo joints 不在 POLICY_JOINT_NAMES 中: {missing}")

    # mj_to_policy_idx[mj_idx] = policy_idx
    mj_to_policy_idx = np.array([policy_name_to_idx[n] for n in mj_joint_names], dtype=np.int64)
    qpos_adr = np.array([int(model.jnt_qposadr[j]) for j in mj_joint_ids], dtype=np.int64)
    qvel_adr = np.array([int(model.jnt_dofadr[j]) for j in mj_joint_ids], dtype=np.int64)

    return mj_joint_names, mj_to_policy_idx, qpos_adr, qvel_adr


def reorder_mj_to_policy(vec_mj, mj_to_policy_idx):
    vec_policy = np.zeros_like(vec_mj)
    vec_policy[mj_to_policy_idx] = vec_mj
    return vec_policy


def reorder_policy_to_mj(vec_policy, mj_to_policy_idx):
    return vec_policy[mj_to_policy_idx]


def init_joy_backend():
    """初始化输入后端（ROS2 > ROS1 > none）。"""
    if ROS2_AVAILABLE:
        rclpy.init(args=None)
        node = JoySubscriberNode()
        print("[INFO] 使用 ROS2 (/joy) 控制。")
        return "ros2", node

    if ROS1_AVAILABLE:
        rospy.init_node('sim2sim_joy_subscriber', anonymous=True)
        rospy.Subscriber('/joy', Joy, joy_callback, queue_size=10)
        print("[INFO] 使用 ROS1 (/joy) 控制。")
        return "ros1", None

    print("[WARN] 未检测到 ROS1/ROS2，/joy 不可用。")
    return "none", None


def spin_joy_once(backend, node):
    if backend == "ros2":
        try:
            rclpy.spin_once(node, timeout_sec=0.0)
        except (ExternalShutdownException, Exception):
            pass


def shutdown_joy_backend(backend, node):
    if backend == "ros2":
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def resolve_command(now: float) -> Tuple[np.ndarray, bool]:
    """获取当前命令向量，返回 (cmd, use_fallback)。"""
    joy_vec = np.array(joy_cmd, dtype=np.float64)
    joy_vec[np.abs(joy_vec) < CommandCfg.joy_deadband] = 0.0
    joy_stale = (now - last_joy_update_time) > CommandCfg.joy_timeout_s
    use_fallback = CommandCfg.enable_fallback_cmd and (joy_stale or np.all(np.abs(joy_vec) < 1e-6))
    cmd_vec = CommandCfg.fallback_cmd if use_fallback else joy_vec
    return cmd_vec, bool(use_fallback)


def build_policy_obs(quat, omega, cmd_vec, q_policy, dq_policy, last_actions_policy):
    """构建与训练一致的 45 维本体观测。"""
    obs = np.zeros([1, 45], dtype=np.float32)
    gravity_vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    proj_gravity = quat_rotate_inverse(quat, gravity_vec)

    obs[0, 0] = omega[0] * ObsScales.ang_vel
    obs[0, 1] = omega[1] * ObsScales.ang_vel
    obs[0, 2] = omega[2] * ObsScales.ang_vel
    obs[0, 3] = proj_gravity[0]
    obs[0, 4] = proj_gravity[1]
    obs[0, 5] = proj_gravity[2]
    obs[0, 6] = cmd_vec[0] * ObsScales.lin_vel
    obs[0, 7] = cmd_vec[1] * ObsScales.lin_vel
    obs[0, 8] = cmd_vec[2] * ObsScales.ang_vel
    obs[0, 9:21] = (q_policy - DEFAULT_DOF_POS) * ObsScales.dof_pos
    obs[0, 21:33] = dq_policy * ObsScales.dof_vel
    obs[0, 33:45] = last_actions_policy
    return np.clip(obs, -SimCfg.clip_observations, SimCfg.clip_observations)


def main():
    """主仿真流程。"""
    if not os.environ.get("DISPLAY"):
        raise RuntimeError("未检测到 DISPLAY，无法创建可视化窗口。请在桌面会话或正确的 X11 转发环境中运行。")

    joy_backend, joy_node = init_joy_backend()

    policy = ort.InferenceSession(SimCfg.policy_model_path, providers=['CPUExecutionProvider'])
    policy_input_name = policy.get_inputs()[0].name
    policy_output_name = policy.get_outputs()[0].name

    model = mujoco.MjModel.from_xml_path(SimCfg.mujoco_model_path)
    model.opt.timestep = SimCfg.dt
    model.opt.gravity = (0, 0, -9.81)
    mj_joint_names, mj_to_policy_idx, qpos_adr, qvel_adr = build_joint_mapping(model, POLICY_JOINT_NAMES)

    print("\n[DEBUG] 关节顺序审计：")
    print(f"  POLICY_JOINT_NAMES: {POLICY_JOINT_NAMES}")
    print(f"  MuJoCo actuator joint顺序: {mj_joint_names}")
    print(f"  mj_to_policy_idx: {mj_to_policy_idx.tolist()}")
    print(f"  DEFAULT_DOF_POS: {DEFAULT_DOF_POS.tolist()}")
    if mj_joint_names != POLICY_JOINT_NAMES:
        print("  [WARN] 检测到 joint 顺序不一致，已启用自动重排映射。")
    else:
        print("  [INFO] joint 顺序一致，无需重排。")

    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q_policy = DEFAULT_DOF_POS.copy()
    action_policy = np.zeros((12,), dtype=np.float64)
    last_actions_policy = np.zeros((12,), dtype=np.float64)

    total_steps = int(SimCfg.sim_duration / SimCfg.dt)
    next_step_time = time.perf_counter()
    last_cmd_warn_ts = time.perf_counter()
    cmd_vec = np.zeros(3, dtype=np.float64)
    step_count = 0
    debug_counter = 0

    if CommandCfg.enable_fallback_cmd:
        print(f"[INFO] 已启用 fallback_cmd={CommandCfg.fallback_cmd.tolist()} (joy超时={CommandCfg.joy_timeout_s}s)")

    try:
        for _ in tqdm(range(total_steps), desc="Simulating..."):
            spin_joy_once(joy_backend, joy_node)

            # 1) 读取当前关节状态（MuJoCo 顺序）
            q_mj = data.qpos[qpos_adr].astype(np.float64)
            dq_mj = data.qvel[qvel_adr].astype(np.float64)

            # 2) 重排到 policy 顺序
            q_policy = reorder_mj_to_policy(q_mj, mj_to_policy_idx)
            dq_policy = reorder_mj_to_policy(dq_mj, mj_to_policy_idx)

            # 3) policy 低频更新（50Hz）
            if step_count % SimCfg.decimation == 0:
                now = time.perf_counter()
                cmd_vec, use_fallback = resolve_command(now)
                if now - last_cmd_warn_ts > 1.0 and use_fallback:
                    last_cmd_warn_ts = now
                    print("[WARN] /joy 超时或接近0，正在使用 fallback_cmd 驱动。")

                quat, omega, _ = get_imu_obs(data)
                obs = build_policy_obs(quat, omega, cmd_vec, q_policy, dq_policy, last_actions_policy)

                policy_out = policy.run([policy_output_name], {policy_input_name: obs})[0]
                action_policy = np.asarray(policy_out, dtype=np.float64).reshape(-1)
                if action_policy.shape[0] != 12:
                    raise RuntimeError(f"ONNX policy 输出维度异常: {action_policy.shape}")

                action_policy = np.clip(action_policy, -CLIP_ACTIONS, CLIP_ACTIONS)
                last_actions_policy = action_policy.copy()
                target_q_policy = action_policy * ACTION_SCALE + DEFAULT_DOF_POS

            # 4) PD 高频更新（每个 mj_step）
            target_q_mj = reorder_policy_to_mj(target_q_policy, mj_to_policy_idx)
            target_dq_mj = np.zeros((12,), dtype=np.float64)

            tau_mj = pd_control(target_q_mj, q_mj, RobotCfg.kp, target_dq_mj, dq_mj, RobotCfg.kd)
            tau_mj = np.clip(tau_mj, -RobotCfg.tau_limit, RobotCfg.tau_limit)
            data.ctrl[:] = tau_mj

            debug_counter += 1
            if debug_counter % 100 == 0:
                lin_vel = data.sensor('linear-velocity').data.astype(np.float64)
                print(
                    f"\n[DEBUG step={step_count}] cmd={cmd_vec} "
                    f"action[:3]={action_policy[:3]} q[:3]={q_policy[:3]} "
                    f"tau[:3]={tau_mj[:3]} lin_vel={lin_vel}"
                )

            mujoco.mj_step(model, data)

            # 5) 实时同步（可关闭）
            if SimCfg.enable_realtime_sync:
                next_step_time += SimCfg.dt
                sleep_time = next_step_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)

            try:
                viewer.render()
            except Exception as e:
                print(f"[INFO] 可视化窗口已关闭，结束仿真: {e}")
                break

            step_count += 1
    finally:
        viewer.close()
        shutdown_joy_backend(joy_backend, joy_node)

if __name__ == '__main__':
    main()
