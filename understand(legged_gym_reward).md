# legged_gym 奖励函数调整实战指南（训练前必读）

> 目标：你后续要改奖励函数时，知道先看哪里、怎么改、何时改、如何定位问题。

---

## 1. 先建立地图：奖励相关核心文件

## 1.1 奖励配置（权重开关与大小）

- 基础奖励配置定义：
  - [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)
- 具体机器人任务覆盖（A1/ANYmal/Cassie）：
  - [legged_gym/envs/a1/a1_config.py](legged_gym/envs/a1/a1_config.py)
  - [legged_gym/envs/anymal_c/mixed_terrains/anymal_c_rough_config.py](legged_gym/envs/anymal_c/mixed_terrains/anymal_c_rough_config.py)
  - [legged_gym/envs/anymal_c/flat/anymal_c_flat_config.py](legged_gym/envs/anymal_c/flat/anymal_c_flat_config.py)
  - [legged_gym/envs/cassie/cassie_config.py](legged_gym/envs/cassie/cassie_config.py)

## 1.2 奖励计算逻辑（函数本体）

- 核心环境类：
  - [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)
- 子类额外奖励（如 Cassie）：
  - [legged_gym/envs/cassie/cassie.py](legged_gym/envs/cassie/cassie.py)

## 1.3 训练与日志（看奖励是否健康）

- 训练入口：
  - [legged_gym/scripts/train.py](legged_gym/scripts/train.py)
- 任务装配：
  - [legged_gym/utils/task_registry.py](legged_gym/utils/task_registry.py)
- TensorBoard 记录奖励分项：
  - [../rsl_rl/rsl_rl/runners/on_policy_runner.py](../rsl_rl/rsl_rl/runners/on_policy_runner.py)
- 本地 play 调试日志：
  - [legged_gym/scripts/play.py](legged_gym/scripts/play.py)
  - [legged_gym/utils/logger.py](legged_gym/utils/logger.py)

---

## 2. 奖励在代码里是如何“自动接线”的

关键在 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)：

1. `LeggedRobot._prepare_reward_function()` 读取 `cfg.rewards.scales`
2. 所有非零权重项自动映射到同名函数：`_reward_<name>()`
3. `LeggedRobot.compute_reward()` 每步求和：
   $$r_t = \sum_i w_i r_t^{(i)}$$
4. 分项累计到 `episode_sums`，最终通过 `infos["episode"]` 写入训练日志

这意味着你改奖励通常有两种方式：
- **只改权重**：改配置文件（最安全）
- **改奖励形状**：改 `_reward_xxx()` 函数（影响更大）

---

## 3. 你要重点解析的函数（按优先级）

## A. 总控函数（必须读懂）

- `LeggedRobot.compute_reward()`
- `LeggedRobot._prepare_reward_function()`
- `LeggedRobot.compute_observations()`
- `LeggedRobot.check_termination()`
- `LeggedRobot.reset_idx()`

原因：奖励并不孤立，和观测、终止条件强耦合。

## B. 常用奖励项函数（高频会改）

- 跟踪类：`_reward_tracking_lin_vel()`、`_reward_tracking_ang_vel()`
- 稳定类：`_reward_orientation()`、`_reward_lin_vel_z()`、`_reward_base_height()`
- 能耗类：`_reward_torques()`、`_reward_action_rate()`
- 安全类：`_reward_collision()`、`_reward_dof_pos_limits()`、`_reward_torque_limits()`
- 步态类：`_reward_feet_air_time()`、`_reward_stumble()`

都在 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)。

---

## 4. 如何知道“该改哪个奖励”

## 4.1 先按症状定位（推荐表）

1. **走不起来/速度跟不上命令**
   - 增强：`tracking_lin_vel`、`tracking_ang_vel`
   - 降低：`torques`、`action_rate`（可能抑制了动作幅度）

2. **能走但抖动明显/动作很碎**
   - 增强：`action_rate`、`dof_acc`
   - 适度增强：`torques`

3. **容易摔/姿态不稳**
   - 增强：`orientation`、`ang_vel_xy`、`lin_vel_z`
   - 检查：`termination` 与终止条件是否过松

4. **拖腿/碰撞多**
   - 增强：`collision`、`feet_stumble`
   - 适度增强：`feet_air_time`（促进抬脚）

5. **站不住或静止指令下乱动**
   - 增强：`stand_still`
   - 降低：过强的速度跟踪项（避免“必须动起来”）

## 4.2 什么时候改“权重” vs 改“函数形状”

- 优先改权重（`scales`）：
  - 目标行为方向没错，但强弱不对
- 才改函数形状（`_reward_xxx`）：
  - 奖励曲线本身有问题，例如：
    - 梯度区间太窄（学不到）
    - 对异常值过敏（训练不稳定）
    - 与任务目标不一致（奖励漏洞）

---

## 5. 一套可执行的调参流程（建议固定使用）

1. **只改一个因素**：一次只改 1~2 个权重，避免耦合混乱。  
2. **先大方向，后精修**：先确保能完成任务，再压能耗/提平滑。  
3. **看分项曲线而不是只看总回报**：总回报升高可能是“钻漏洞”。  
4. **固定随机种子做 A/B**：同配置跑 baseline 与 candidate。  
5. **固定评测脚本回放**：用 [legged_gym/scripts/play.py](legged_gym/scripts/play.py) 对比动作质量。

---

## 6. 调试与排障：如何快速发现奖励问题

## 6.1 训练中要看的关键日志

来自 [../rsl_rl/rsl_rl/runners/on_policy_runner.py](../rsl_rl/rsl_rl/runners/on_policy_runner.py)：

- `Episode/rew_tracking_lin_vel` 等分项
- `Train/mean_reward`
- `Train/mean_episode_length`
- `Loss/value_function`
- `Loss/surrogate`

判断建议：
- 若 `mean_reward` 上升但 `episode_length` 下降，常见是策略冒险取巧。
- 若 `tracking` 长期很低，通常奖励冲突或惩罚过重。

## 6.2 在环境侧加“诊断量”

你可以在 `reset_idx()` 的 `self.extras["episode"]` 里增加自定义统计（均值速度误差、碰撞次数等），便于 TensorBoard 直观看到问题根因。

文件位置： [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)

## 6.3 常见失败模式

1. **奖励尺度失衡**：某个项量级过大，其他项形同虚设。  
2. **奖励冲突**：比如既强惩罚动作又要求高跟踪速度。  
3. **奖励稀疏**：早期几乎拿不到正反馈。  
4. **奖励漏洞**：学会“奇怪姿态/抖动”刷分。  
5. **终止条件与奖励不一致**：终止太晚导致危险行为被“容忍”。

---

## 7. 数学与工程直觉（简版）

PPO 学的是期望回报最大化：
$$
J(\pi)=\mathbb{E}_{\tau\sim\pi}\left[\sum_t \gamma^t r_t\right]
$$

所以奖励设计本质是“目标函数设计”。

- **奖励项 = 软约束**
- **终止条件 = 硬约束**

经验上：
- 先靠“跟踪/存活”建立可行行为；
- 再逐步提高“平滑/能耗/安全”权重，打磨动作质量。

---

## 8. 针对你后续改奖励的建议起步顺序

1. 先在目标任务配置里只改 `scales`（例如 A1 用 [legged_gym/envs/a1/a1_config.py](legged_gym/envs/a1/a1_config.py)）。
2. 确认行为方向正确后，再改 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py) 的 `_reward_xxx()` 形状。
3. 每次改动配套记录：
   - 改了哪个权重/函数
   - 预期影响
   - 实际现象（分项曲线 + 视频回放）
4. 对关键版本保留 checkpoint 与配置快照，避免“回不去”。

---

## 9. 后续逐章补充占位（你后面可继续加）

- [ ] 每个奖励项的物理意义、典型范围、相互冲突关系
- [ ] A1/ANYmal/Cassie 的奖励参数对照表
- [ ] 常见任务目标（快走、稳走、抗扰）对应奖励模板
- [ ] 奖励漏洞案例库与修复策略
- [ ] 训练失败排查 checklist
