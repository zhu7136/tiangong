## Purpose

定义 HU_D04 人形机器人的奔跑训练任务，包括环境配置、reward 函数、观测空间和 AMP 参数，使其可在 IsaacLab 中进行强化学习训练。

## ADDED Requirements

### Requirement: 任务注册
系统 SHALL 注册 `hu_d04_run` 任务，可通过 `--task=hu_d04_run` 参数启动训练。

#### Scenario: 启动训练
- **WHEN** 用户执行 `python train.py --task=hu_d04_run`
- **THEN** 系统加载 HU_D04 配置并开始奔跑训练

### Requirement: 速度指令范围
任务 SHALL 配置与 TienKung 相同的速度指令范围：前进 -0.6~1.0 m/s、横向 -0.5~0.5 m/s、转向 ±1.57 rad/s。

#### Scenario: 速度指令采样
- **WHEN** 训练过程中指令重采样
- **THEN** 速度指令在配置范围内随机采样

### Requirement: Reward 函数适配
Reward 函数 SHALL 使用 HU_D04 的关节命名规范（`left_/right_` 前缀）进行正则匹配。

#### Scenario: Reward 计算正确
- **WHEN** 每个训练步
- **THEN** Reward 函数使用正确的关节名计算各项奖励

### Requirement: 终止条件配置
终止条件 SHALL 检测膝盖、肩膀、肘部、腰部的非期望接触。

#### Scenario: 非期望接触触发终止
- **WHEN** 膝盖、肩膀、肘部或腰部发生接触
- **THEN** 当前 episode 被标记为终止

### Requirement: AMP 参考动作
任务 SHALL 使用 HU_D04 的 AMP 参考动作数据进行运动风格学习。

#### Scenario: AMP 数据加载
- **WHEN** 训练开始
- **THEN** 系统加载 HU_D04 的 AMP 参考动作文件

### Requirement: 脚部 reward 计算
Reward 函数 SHALL 使用 HU_D04 的脚部 link 名称（`contact_foot_center_{L,R}`）计算脚部相关奖励。

#### Scenario: 脚部 reward 生效
- **WHEN** 每个训练步
- **THEN** 脚部滑动、脚部力等 reward 使用正确的脚部 link 名称计算
