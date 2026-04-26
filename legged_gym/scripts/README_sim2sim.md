# A1 Sim2Sim 使用说明（IsaacGym -> MuJoCo）

## 1. 目标
本脚本用于将 `legged_gym` 训练得到的 A1 策略（ONNX）部署到 MuJoCo 中运行。

核心设计：
- 策略网络低频更新（50Hz）
- PD 力矩高频闭环（每个 `mj_step`）
- 自动处理 MuJoCo 与策略关节顺序差异
- 支持 `/joy` 实时速度指令；可选 fallback 命令

---

## 2. 文件与入口
- 脚本： [legged_gym/legged_gym/scripts/sim2sim.py](legged_gym/legged_gym/scripts/sim2sim.py)
- 入口函数：`main()`

---

## 3. 运行流程
1. 初始化输入后端（ROS2 / ROS1 / none）
2. 加载 ONNX 策略
3. 加载 MuJoCo 模型并建立关节映射
4. 主循环：
   - 每步读取关节状态、IMU
   - 每 `decimation=4` 步更新一次策略输出
   - 每步都进行 PD 力矩计算并 `mj_step`
   - 可选实时同步与可视化

---

## 4. 关键一致性参数
在脚本顶部常量区统一维护：

- `POLICY_JOINT_NAMES`：策略关节顺序
- `DEFAULT_DOF_POS`：默认关节角
- `ACTION_SCALE = 0.25`：与 A1 训练配置一致
- `CLIP_ACTIONS = 100.0`：与训练归一化配置一致

观测缩放：
- `ObsScales.lin_vel = 2.0`
- `ObsScales.ang_vel = 0.25`
- `ObsScales.dof_pos = 1.0`
- `ObsScales.dof_vel = 0.05`

---

## 5. /joy 与 fallback 逻辑
命令来源优先级：
1. `/joy`（实时手柄）
2. fallback（当 `/joy` 超时或接近零）

配置位于 `CommandCfg`：
- `enable_fallback_cmd`
- `joy_timeout_s`
- `joy_deadband`
- `fallback_cmd`

如需完全由手柄控制，可将 `enable_fallback_cmd=False`。

### 5.1 如何发布 `/joy` 指令
`sim2sim.py` 订阅的是 `sensor_msgs/msg/Joy`，消息里最关键的是 `axes` 数组。

当前脚本的映射关系是：
- `axes[1]` -> 前进/后退速度 `vx`
- `axes[0]` -> 横向速度 `vy`
- `axes[3]` -> 偏航角速度 `yaw`

如果你有手柄节点，它会持续发布 `/joy`，脚本就会直接使用这些输入。

如果你想手动测试，可直接发布一个 Joy 消息：

ROS2：
```bash
ros2 topic pub /joy sensor_msgs/msg/Joy "{axes: [0.0, 0.6, 0.0, 0.0], buttons: []}"
```

ROS1：
```bash
rostopic pub /joy sensor_msgs/Joy "axes: [0.0, 0.6, 0.0, 0.0]\nbuttons: []" -r 10
```

说明：
- 上面的例子会给出一个持续向前的 `vx` 指令。
- 如果你想让机器人原地转向，可以改 `axes[3]`。
- 如果你使用真实手柄，通常需要先启动对应的 `joy_node`，让它自动向 `/joy` 发布数据。

---

## 6. 调试输出含义
脚本每 100 步打印一次：
- `action[:3]`：策略输出片段
- `q[:3]`：当前关节角片段（策略顺序）
- `tau[:3]`：PD 力矩片段
- `lin_vel`：IMU 线速度

用于判断：
- 策略是否在更新
- 关节是否在响应
- 机器人是否有前进速度

---

## 7. 常见问题
### Q1: 机器人不动
- 检查 `/joy` 是否发布；若无，确认 fallback 是否开启。
- 查看调试行里 `action` 是否变化。

### Q2: 动一下就停
- 常见原因是命令接近零。
- 检查 `joy_deadband` 是否过大，或手柄轴映射是否正确。

### Q3: 行为和训练差异大
- 先核对 `ACTION_SCALE`、`DEFAULT_DOF_POS`、关节顺序映射。
- 再核对模型路径是否是最新导出的 ONNX。

---

## 8. 运行
在脚本目录执行：

`python3 sim2sim.py`

如果可视化窗口关闭，脚本会正常退出。