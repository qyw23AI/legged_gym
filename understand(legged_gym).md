# legged_gym 体系总览（从配置到策略更新）

> 目标：先建立“全局地图”，明确 **配置如何进入环境**、**训练如何采样与更新策略**、**文件之间如何关联**。
> 后续你可以按本文件中的“预留深入解读”逐节补充细节。

---

## 1. 总体架构（一句话）

`legged_gym` 负责 **任务定义 + 物理仿真 + 奖励/观测构建**，`rsl_rl` 负责 **PPO 采样与更新**。两者通过 `VecEnv` 接口对接。

- 环境侧入口： [legged_gym/scripts/train.py](legged_gym/scripts/train.py)
- 任务注册中心： [legged_gym/utils/task_registry.py](legged_gym/utils/task_registry.py)
- 环境实现核心： [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)
- 算法执行核心： [../rsl_rl/rsl_rl/runners/on_policy_runner.py](../rsl_rl/rsl_rl/runners/on_policy_runner.py)
- PPO 更新核心： [../rsl_rl/rsl_rl/algorithms/ppo.py](../rsl_rl/rsl_rl/algorithms/ppo.py)

### 注解：看成“环境层 vs 算法层”的标准解耦。

在强化学习里，它们分工如下：

#### legged_gym：负责 定义问题本身（MDP）

任务/场景：机器人、地形、目标、终止条件、随机化
物理仿真：用 Isaac Gym 并行推进动力学
step() 内部逻辑：执行动作 → 仿真若干子步 → 计算观测、奖励、done、info
本质上回答：“智能体所处世界是什么，动作会导致什么后果，什么叫做好/坏”


#### rsl_rl：负责 求解这个问题（优化策略）

采样：用当前策略与 VecEnv 交互，收集轨迹（obs, action, reward, done）
估计优势：如 GAE、回报计算
PPO 更新：策略损失、价值损失、熵正则、裁剪、mini-batch 多轮优化
本质上回答：“如何利用数据把策略参数更新得更好”
通过 VecEnv 对接时的核心意义
VecEnv 是统一协议，屏蔽了环境细节，让算法只依赖标准接口：

* reset()：返回初始批量观测
  step(actions)：返回下一批 obs, rewards, dones, infos

#### 这样带来三点价值：

解耦：换任务（legged_gym）不必改 PPO 主体（rsl_rl）。
并行高吞吐：上千环境同时滚动，提升 on-policy 数据效率。
可复用：同一算法可接多个环境，同一环境可接不同算法。
训练循环里各自位置（简化）
1.rsl_rl 用当前策略输出动作
2.legged_gym 接收动作并仿真，返回奖励与新观测
3.rsl_rl 累积 rollout，做 PPO 更新
4.重复直到收敛

一句话总结：
legged_gym 决定“学什么”与“数据怎么产生”，rsl_rl 决定“怎么学”与“参数怎么更新”。



### 核心代码位置：
[on_policy_runner.py**:83-143**](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)

---

#### 1) 训练前初始化（一次性或每次 `learn()` 开始）

* 创建 `SummaryWriter`：用于 TensorBoard 记录。
* `init_at_random_ep_len`：把 `episode_length_buf` 随机化，避免所有并行环境同一时刻同时结束，降低采样相关性。
* `obs = self.env.get_observations()`，`privileged_obs = self.env.get_privileged_observations()`：
  * actor 用普通观测；
  * critic 可用特权观测（teacher-style 训练），即 asymmetric actor-critic。
* `self.alg.actor_critic.train()`：进入训练模式（如 dropout、norm 的训练行为）。

**RL 背后知识**
这里定义了“谁看什么信息”。策略网络（actor）受部署约束，价值网络（critic）可用更多状态信息来稳定优势估计，常见于机器人任务。

#### 2) 统计缓存初始化

对应 [on_policy_runner.py**:102-107**](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)

* `rewbuffer`、`lenbuffer`：滑动窗口统计最近 100 个 episode。
* `cur_reward_sum`、`cur_episode_length`：每个并行环境当前回合累积奖励与长度。

**RL 背后知识**
这些不是训练必需，但用于监控学习是否正常（回报上升、回合长度变化等）。

#### 3）外层迭代：一次迭代 = 一次 rollout + 一次 PPO 更新

for it in ...：第 it 次策略迭代。
每次分两阶段：
收集 on-policy 数据（rollout）
用这批新数据更新策略（update）

#### 4) Rollout 采样阶段（最关键）

对应 [on_policy_runner.py**:113-136**](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)

* `with torch.inference_mode()`：采样时不构建反向图，省显存提速。
* 内层 `for i in range(self.num_steps_per_env)`：

  1. `actions = self.alg.act(obs, critic_obs)`
     * 当前策略采样动作，同时通常会缓存 `log_prob`、`value` 等（供 PPO 用）。
  2. `obs, privileged_obs, rewards, dones, infos = self.env.step(actions)`
     * `VecEnv` 并行推进所有环境一步，返回批量结果。
  3. `self.alg.process_env_step(rewards, dones, infos)`
     * 把一步 transition 写入 rollout buffer。
  4. 日志 bookkeeping：按 `dones` 找到结束回合，提取该环境的累积奖励和长度，写入滑动窗口。

  **RL 背后知识**

  * 这是标准 on-policy 采样：更新前必须用“当前策略”重新采样。
  * `dones` 决定轨迹切分和 bootstrap 边界。
  * 并行 `VecEnv` 提供高吞吐，降低样本方差。

  ### 5) 计算回报与优势（更新前）

  对应 [on_policy_runner.py**:137-140**](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)


  * `self.alg.compute_returns(critic_obs)`：用最后时刻 `critic_obs` 的 value 做 bootstrap，计算 return/advantage（通常是 GAE）。

  ### 6) PPO 参数更新

  对应 [on_policy_runner.py**:142-143**](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)

  * `mean_value_loss, mean_surrogate_loss = self.alg.update()`
  * `update()` 通常做：多 epoch、mini-batch、clip surrogate objective、value loss、entropy bonus。

  ### 7) 日志、保存、迭代推进

  对应 [on_policy_runner.py**:145-151**](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)

  * `self.log(locals())`：记录损失、速度、奖励等。
  * `save_interval` 周期保存模型。
  * 最后更新 `self.current_learning_iteration` 并再保存一次。

## 2. 目录分层与职责

### 2.1 任务/环境层（legged_gym）

- 任务注册与装配：
  - [legged_gym/envs/__init__.py](legged_gym/envs/__init__.py)
  - [legged_gym/utils/task_registry.py](legged_gym/utils/task_registry.py)
- 基类与主环境：
  - [legged_gym/envs/base/base_task.py](legged_gym/envs/base/base_task.py)
  - [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)
- 配置系统：
  - [legged_gym/envs/base/base_config.py](legged_gym/envs/base/base_config.py)
  - [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)
  - 示例任务配置：
    - [legged_gym/envs/a1/a1_config.py](legged_gym/envs/a1/a1_config.py)
    - [legged_gym/envs/anymal_c/mixed_terrains/anymal_c_rough_config.py](legged_gym/envs/anymal_c/mixed_terrains/anymal_c_rough_config.py)
    - [legged_gym/envs/anymal_c/flat/anymal_c_flat_config.py](legged_gym/envs/anymal_c/flat/anymal_c_flat_config.py)
- 特化机器人逻辑：
  - [legged_gym/envs/anymal_c/anymal.py](legged_gym/envs/anymal_c/anymal.py)（执行器网络）
  - [legged_gym/envs/cassie/cassie.py](legged_gym/envs/cassie/cassie.py)
- 地形/工具：
  - [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)
  - [legged_gym/utils/helpers.py](legged_gym/utils/helpers.py)
  - [legged_gym/utils/logger.py](legged_gym/utils/logger.py)

### 2.2 算法层（rsl_rl）

- 环境抽象接口： [../rsl_rl/rsl_rl/env/vec_env.py](../rsl_rl/rsl_rl/env/vec_env.py)
- Runner（训练总循环）： [../rsl_rl/rsl_rl/runners/on_policy_runner.py](../rsl_rl/rsl_rl/runners/on_policy_runner.py)
- PPO 算法： [../rsl_rl/rsl_rl/algorithms/ppo.py](../rsl_rl/rsl_rl/algorithms/ppo.py)
- 策略/价值网络： [../rsl_rl/rsl_rl/modules/actor_critic.py](../rsl_rl/rsl_rl/modules/actor_critic.py)
- 采样缓存（rollout buffer）： [../rsl_rl/rsl_rl/storage/rollout_storage.py](../rsl_rl/rsl_rl/storage/rollout_storage.py)

---

## 3. 从“配置”到“训练”的完整链路

## 3.1 第 0 步：启动入口与参数

入口脚本： [legged_gym/scripts/train.py](legged_gym/scripts/train.py)

核心调用：

1. `task_registry.make_env(...)` 创建环境
2. `task_registry.make_alg_runner(...)` 创建 PPO runner
3. `ppo_runner.learn(...)` 进入训练循环

命令行参数解析在： [legged_gym/utils/helpers.py](legged_gym/utils/helpers.py) 的 `get_args()`。

**预留深入解读**：

- [ ] train.py 的参数覆盖优先级
- [ ] 多设备参数（`sim_device`/`rl_device`）含义

---

## 3.2 第 1 步：任务注册与配置装配

注册发生在： [legged_gym/envs/__init__.py](legged_gym/envs/__init__.py)

每个任务都绑定四元组：

- `task_name`
- `EnvClass`
- `EnvCfg`
- `TrainCfg`

例如 `a1` 任务绑定到：

- 环境类 `LeggedRobot`
- 环境配置 `A1RoughCfg`
- 训练配置 `A1RoughCfgPPO`

配置获取/覆写逻辑在： [legged_gym/utils/task_registry.py](legged_gym/utils/task_registry.py)

关键点：

- `get_cfgs()` 会把 `train_cfg.seed` 复制到 `env_cfg.seed`
- `update_cfg_from_args()` 会用 CLI 覆盖配置

**预留深入解读**：

- [ ] `task_registry` 的扩展方式（外部自定义任务）
- [ ] `seed` 传递路径

---

## 3.3 第 2 步：环境初始化（Isaac Gym 批量并行）

环境类主干： [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)

初始化流程：

1. `_parse_cfg()`：解析 `dt`、reward scales、命令范围、episode 长度
2. `BaseTask.__init__()`：创建 sim / viewer / 基础 buffer
3. `create_sim()`：创建地形、加载机器人资产、批量创建 `num_envs` 个环境
4. `_init_buffers()`：绑定 Isaac GPU Tensor（root state、dof、contact force）
5. `_prepare_reward_function()`：把非零奖励项映射到 `_reward_xxx()` 函数列表

地形生成由： [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py)

**预留深入解读**：

- [ ] `_create_envs()` 中 URDF 资产与属性注入
- [ ] `terrain curriculum` 的地图组织方式

---

## 3.4 第 3 步：采样时每一步发生什么（环境 `step`）

在 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py) 的 `step()` 与 `post_physics_step()`：

1. 策略给出动作 `actions`
2. 动作裁剪后经 `_compute_torques()` 转成力矩（或目标位姿 PD）
3. 进行 `decimation` 次物理子步（控制频率低于仿真频率）
4. 刷新状态后：
   - `check_termination()` 判终止/超时
   - `compute_reward()` 累加奖励
   - `reset_idx(...)` 对终止环境重置
   - `compute_observations()` 生成新观测
5. 返回 `(obs, privileged_obs, reward, done, extras)`

说明：这正是 `VecEnv.step()` 约定接口（见 [../rsl_rl/rsl_rl/env/vec_env.py](../rsl_rl/rsl_rl/env/vec_env.py)）。

**预留深入解读**：

- [ ] `decimation` 与控制频率/仿真频率关系
- [ ] `extras["time_outs"]` 在 PPO 中的作用

---

## 3.5 第 4 步：Runner 组织采样与更新

主循环在： [../rsl_rl/rsl_rl/runners/on_policy_runner.py](../rsl_rl/rsl_rl/runners/on_policy_runner.py)

每个 iteration：

1. 采样阶段：循环 `num_steps_per_env`
   - `alg.act(...)`：策略采样动作 + 价值估计
   - `env.step(...)`
   - `alg.process_env_step(...)`：写入 rollout storage
2. 更新阶段：
   - `alg.compute_returns(...)`（GAE）
   - `alg.update()`（PPO 多 epoch、小批量优化）
3. 日志与保存：TensorBoard + `model_x.pt`

**预留深入解读**：

- [ ] runner 中 episode 统计与日志字段
- [ ] checkpoint 恢复逻辑

---

## 3.6 第 5 步：PPO 参数更新（核心强化学习过程）

实现文件： [../rsl_rl/rsl_rl/algorithms/ppo.py](../rsl_rl/rsl_rl/algorithms/ppo.py)

### 关键步骤

1. **行为策略采样**：
   - actor 输出均值，方差由可学习 `std` 给出
   - 采样动作 $a_t \sim \pi_\theta(a_t|s_t)$
2. **收集轨迹**：存入 [../rsl_rl/rsl_rl/storage/rollout_storage.py](../rsl_rl/rsl_rl/storage/rollout_storage.py)
3. **优势估计（GAE）**：
   - 使用 $\gamma$ 与 $\lambda$ 反向计算 returns/advantages
4. **PPO 剪切目标**：
   - 概率比 $r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$
   - 目标：$\min\big(r_tA_t,\ \text{clip}(r_t,1-\epsilon,1+\epsilon)A_t\big)$
5. **值函数损失 + 熵正则 + 梯度裁剪**：
   - 总损失 = surrogate + value loss - entropy bonus
6. **可选自适应学习率**：依据 KL 与 `desired_kl` 调整 lr

**预留深入解读**：

- [ ] PPO 剪切项在本项目中的具体数值影响
- [ ] `desired_kl` 自适应策略的稳定性分析

---

## 4. 配置体系（EnvCfg 与 TrainCfg）如何影响训练

核心定义： [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)

- `LeggedRobotCfg`：环境与仿真配置
  - `env/terrain/commands/init_state/control/asset/domain_rand/rewards/normalization/noise/sim`
- `LeggedRobotCfgPPO`：训练配置
  - `policy/algorithm/runner`

任务配置通过继承修改差异：

- [legged_gym/envs/a1/a1_config.py](legged_gym/envs/a1/a1_config.py)
- [legged_gym/envs/anymal_c/mixed_terrains/anymal_c_rough_config.py](legged_gym/envs/anymal_c/mixed_terrains/anymal_c_rough_config.py)
- [legged_gym/envs/anymal_c/flat/anymal_c_flat_config.py](legged_gym/envs/anymal_c/flat/anymal_c_flat_config.py)

### 一个非常关键的机制：奖励自动装配

`rewards.scales` 中非零项会自动对应到 `LeggedRobot` 的 `_reward_xxx()`。

例如：

- `tracking_lin_vel` -> `_reward_tracking_lin_vel()`
- `torques` -> `_reward_torques()`

这使得“奖励开关/调权重”主要靠配置，而不是改训练主循环。

**预留深入解读**：

- [ ] 各 reward 项的梯度信号强弱
- [ ] `only_positive_rewards` 对 early training 的影响

---

## 5. 观测、动作、奖励的语义（RL 视角）

### 5.1 状态/观测（Observation）

由 `compute_observations()` 拼接：

- 基座角速度
- 重力投影
- 速度命令
- 关节位置/速度
- 上一步动作
- （可选）地形高度采样

意义：

- 本体状态 + 任务条件（命令）+ 记忆痕迹（last action）
- 若加入高度测量，会提高崎岖地形可感知性

### 5.2 动作（Action）

策略输出连续动作（通常 $[-1,1]$ 附近高斯采样），再映射到：

- PD 目标位姿/速度，或
- 直接力矩（`control_type='T'`）

### 5.3 奖励（Reward）

本质是多目标加权和：

- 跟踪命令（前进/转向）
- 稳定性（姿态、垂向速度）
- 能耗/平滑（torque、action_rate）
- 安全约束（碰撞、关节极限）

可理解为：

$$
r_t = \sum_i w_i r_t^{(i)}
$$

其中权重 $w_i$ 来自配置。

---

## 6. 域随机化与课程学习在这里怎么落地

### 6.1 域随机化（Domain Randomization）

对应配置：`domain_rand`，落地代码在 `LeggedRobot`：

- 摩擦随机化：`_process_rigid_shape_props()`
- 机体质量随机化：`_process_rigid_body_props()`
- 随机推扰：`_push_robots()`

目标：让策略面对参数扰动，提高 sim-to-real 鲁棒性。

### 6.2 课程学习（Curriculum）

- 地形课程：`_update_terrain_curriculum()`
- 命令课程：`update_command_curriculum()`

目标：先学简单，再逐步提升难度，改善收敛稳定性。

---

## 7. Actor-Critic 网络结构与概率策略

网络文件： [../rsl_rl/rsl_rl/modules/actor_critic.py](../rsl_rl/rsl_rl/modules/actor_critic.py)

- Actor：MLP 输出动作均值
- Critic：MLP 输出状态价值 $V(s)$
- 动作分布：对角高斯 `Normal(mean, std)`，`std` 可学习

术语解释：

- **Actor-Critic**：一个网络（或两头）负责“做动作”，另一个负责“评估好坏”。
- **Entropy bonus**：鼓励探索，防止过早收敛到过窄分布。
- **GAE**：平衡方差与偏差的优势估计方法。

---

## 8. 训练产物、恢复与部署相关文件

- 模型保存（训练中）：`logs/<experiment>/<run>/model_x.pt`
- 推理脚本： [legged_gym/scripts/play.py](legged_gym/scripts/play.py)
- 导出：
  - JIT / ONNX 导出在 [legged_gym/utils/helpers.py](legged_gym/utils/helpers.py)
  - Sim2Sim（MuJoCo）脚本在 [legged_gym/scripts/sim2sim.py](legged_gym/scripts/sim2sim.py)

**预留深入解读**：

- [ ] `model_state_dict` 结构与加载路径
- [ ] ONNX 输入输出张量与部署侧接口约定

---

## 9. 文件关系图（文字版）

1. [legged_gym/scripts/train.py](legged_gym/scripts/train.py)-> 调 [legged_gym/utils/task_registry.py](legged_gym/utils/task_registry.py)
2. `task_registry.make_env()`-> 从 [legged_gym/envs/__init__.py](legged_gym/envs/__init__.py) 找任务注册-> 创建 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)（或子类）
3. `task_registry.make_alg_runner()`-> 创建 [../rsl_rl/rsl_rl/runners/on_policy_runner.py](../rsl_rl/rsl_rl/runners/on_policy_runner.py)
4. Runner 内部
   -> 用 [../rsl_rl/rsl_rl/modules/actor_critic.py](../rsl_rl/rsl_rl/modules/actor_critic.py) 建网
   -> 用 [../rsl_rl/rsl_rl/algorithms/ppo.py](../rsl_rl/rsl_rl/algorithms/ppo.py) 更新
   -> 用 [../rsl_rl/rsl_rl/storage/rollout_storage.py](../rsl_rl/rsl_rl/storage/rollout_storage.py) 存轨迹

---

## 10. 重要名词速查（先给简版）

- **VecEnv**：并行环境接口，一次 step 同时推进大量子环境。
- **Rollout**：按当前策略采样的一段轨迹数据。
- **On-policy**：更新时只能使用“当前（或最近）策略”采样的数据。
- **PPO Clip**：限制策略更新幅度，防止性能崩塌。
- **Advantage ($A_t$)**：某动作相对基线价值的“超额收益”。
- **Return ($R_t$)**：从当前时刻开始的折扣累计回报。
- **Domain Randomization**：训练中随机化物理参数提升泛化。
- **Curriculum Learning**：逐步提高任务难度。
- **Privileged Observation**：仅给 critic 的额外信息（可选）。
- **Decimation**：每次策略动作对应多个物理仿真子步。

---

## 11. 后续可逐章补充的“占位目录”

1. 配置字段全解（`env/terrain/commands/...`）
2. `LeggedRobot.step()` 全流程逐行图解
3. 奖励项逐项推导与调参建议
4. PPO 数学细节与本工程默认超参数经验
5. 域随机化与 sim2real 经验总结
6. 模型导出（JIT/ONNX）与 sim2sim/real 接入流程

---

## 12. 你当前仓库中的一条建议（非必须）

你仓库里的 [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py) 中 `num_observations` 被写成 `235 -187 -3`（结果为 45），这是可行的，但建议后续改为显式常量并在注释中同步观测组成，减少维护歧义。
