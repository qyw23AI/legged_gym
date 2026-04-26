# legged_gym 随机化训练专项解读（新增项导向）

> 目标：你要加的 3 类随机化分别落到哪些文件、哪些函数、在训练流程哪个时机生效。

---

## 1. 先看总流程：随机化插入点在哪

训练入口是 [legged_gym/scripts/train.py](legged_gym/scripts/train.py)，主链路是：

1. `task_registry.make_env()` 创建环境（[legged_gym/utils/task_registry.py](legged_gym/utils/task_registry.py)）
2. `LeggedRobot.__init__()` + `create_sim()` 建立并行 env（[legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)）
3. 进入 `step()` 循环，周期执行 `post_physics_step()` / `reset_idx()`

随机化通常分两类：

- **创建时随机化**（每个 env 创建 actor 时生效一次）
- **重置时随机化**（每次 episode reset 或命令重采样时重新采样）

---

## 2. 你这 3 项需求对应的“重点文件 + 重点函数”

## 2.1 电机偏置 + Kp/Kd + 力矩随机化

### 必看文件

- 配置定义： [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)
- 执行逻辑： [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)
- 任务覆盖配置（A1/ANYmal/Cassie）：
  - [legged_gym/envs/a1/a1_config.py](legged_gym/envs/a1/a1_config.py)
  - [legged_gym/envs/anymal_c/mixed_terrains/anymal_c_rough_config.py](legged_gym/envs/anymal_c/mixed_terrains/anymal_c_rough_config.py)

### 必看函数

- `LeggedRobot._init_buffers()`：初始化随机化 buffer（如 `motor_offsets`、`Kp_factors`、`Kd_factors`、`motor_strengths`）
- `LeggedRobot._compute_torques()`：把随机化参数真正作用到控制律
- `LeggedRobot.reset_idx()` 或 `LeggedRobot._post_physics_step_callback()`：触发每次重采样
- 建议新增 `LeggedRobot._randomize_dof_props(env_ids)`：专门管理上述采样

### 你应采用的控制律形态

对 `P` 控制建议改为：

$$
\tau = s_{motor}\left(K_p\odot f_{kp}\odot(q^* + b_{motor} - q) - K_d\odot f_{kd}\odot\dot q\right)
$$

其中：

- $b_{motor}$：电机偏置随机量
- $f_{kp}, f_{kd}$：Kp/Kd 缩放因子
- $s_{motor}$：电机强度（力矩幅值）随机因子

并继续经过 `torch.clip(..., -torque_limits, torque_limits)`。

### 现成参考实现（可直接对照迁移）

你的兄弟目录已有近似完整版本：

- [../legged_gym1/legged_gym/envs/base/legged_robot.py](../legged_gym1/legged_gym/envs/base/legged_robot.py)
- [../legged_gym1/legged_gym/envs/base/legged_robot_config.py](../legged_gym1/legged_gym/envs/base/legged_robot_config.py)

重点参考 `._randomize_dof_props()` 与 `._compute_torques()`。

---

## 2.2 质心 + 摩擦 + 恢复系数（restitution）随机化

### 必看文件

- [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)
- [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)

### 必看函数

- `LeggedRobot._process_rigid_shape_props(props, env_id)`
  - 当前已有摩擦随机化入口
  - 你应在这里加入 `props[s].restitution = ...`
- `LeggedRobot._process_rigid_body_props(props, env_id)`
  - 当前已有 base mass 随机入口
  - 你应在这里加入 `props[0].com += Vec3(...)`（base COM）

### 时机说明（很关键）

这两个回调在 `_create_envs()` 中调用，属于**创建时随机化**。
如果你希望“每个 episode 都变”，需要在 `reset_idx()` 里额外重新下发刚体属性（代价更高）。

---

## 2.3 全身质量随机 + 全身质心随机

### 必看文件

- [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)
- [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py)

### 必看函数

- `LeggedRobot._process_rigid_body_props(props, env_id)`

### 实现策略

当前主仓版本只处理 base mass。你需要扩展为：

1. **全身质量随机**：循环 `for i in range(len(props))`，对每个 link 的 `mass` 乘随机因子或加偏移。
2. **全身 COM 随机**：循环每个 link，对 `com` 加随机三维偏移。
3. 对 base 和 limbs 使用不同分布范围（更稳定）。

同样可参考：

- [../legged_gym1/legged_gym/envs/base/legged_robot.py](../legged_gym1/legged_gym/envs/base/legged_robot.py)

---

## 3. 配置层建议：`domain_rand` 最少要加哪些字段

修改 [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py) 的 `domain_rand`，建议至少补齐：

- 电机与控制：
  - `randomize_motor_offset`, `motor_offset_range`
  - `randomize_Kp_factor`, `Kp_factor_range`
  - `randomize_Kd_factor`, `Kd_factor_range`
  - `randomize_motor_strength`, `motor_strength_range`
- 接触：
  - `restitution_range`
- 质量与质心：
  - `randomize_base_com`, `added_com_range`
  - `randomize_leg_mass`, `factor_leg_mass_range`
  - `randomize_leg_com`, `added_leg_com_range`

再在任务配置中按机器人单独覆盖（A1 和 ANYmal 通常范围不同）。

---

## 4. 你应重点阅读的函数调用关系（按执行顺序）

1. `_parse_cfg()`
2. `_create_envs()`
   - `_process_rigid_shape_props()`
   - `_process_dof_props()`
   - `_process_rigid_body_props()`
3. `_init_buffers()`
4. `step()`
   - `_compute_torques()`
5. `post_physics_step()`
6. `reset_idx()`（建议在这里触发一次随机化重采样）

都在 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py)。

---

## 5. 与强化学习稳定性直接相关的注意点

1. **随机化强度不要一步拉满**：先小范围，观察回报曲线。
2. **控制相关随机化比接触随机化更“敏感”**：`Kp/Kd/motor_strength` 过大易发散。
3. **训练前期建议课程化随机化**：迭代早期缩窄范围，后期放宽。
4. **确保观测中有可补偿信息**：如关节速度、重力投影；否则策略难适应隐藏扰动。
5. **关注 reward 分项日志**：重点看 `tracking_*`、`torques`、`collision` 是否异常波动。

修改奖励函数---

## 6. 推荐你的实施优先级（最稳妥）

1. 先加 **motor_offset + motor_strength**（改动小，收益直接）
2. 再加 **Kp/Kd factor**（与控制器耦合更强，需小范围）
3. 再加 **restitution + base COM**
4. 最后加 **全身 mass/com**（最容易引入不稳定）

---

## 7. 逐项补充解读占位

- [ ] `domain_rand` 每个字段的物理意义与推荐范围
- [ ] `._compute_torques()` 数学形式与数值稳定性
- [ ] `._process_rigid_body_props()` 的 link 级随机化策略
- [ ] A1 与 ANYmal 的差异化随机化模板
- [ ] 训练日志诊断模板（判断是“欠随机化”还是“过随机化”）
