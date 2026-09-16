## Why

TienKung-Lab 当前仅支持 TienKung2 Lite 人形机器人（20 DOF）。需要适配新的 HU_D04 机器人（33 DOF）进行奔跑训练，验证框架对不同关节结构和命名规范的泛化能力。

## What Changes

- 新增 HU_D04 机器人资产配置（USD 模型、关节参数、电机参数）
- 修改环境配置中的关节名正则匹配，适配 HU_D04 的 `left_/right_` 前缀命名
- 修改 `tienkung_env.py` 中硬编码的关节名（ankle_roll_* → left/right_ankle_roll）
- 修改 reward 函数中的 body_names 和 joint_names 正则
- 注册新的 HU_D04 run 训练任务

## Capabilities

### New Capabilities
- `hu-d04/robot-asset`: HU_D04 机器人资产配置（USD 路径、关节定义、电机参数、初始姿态）
- `hu-d04/run-task`: HU_D04 奔跑训练任务（环境配置、reward 函数、AMP 参数）

### Modified Capabilities
<!-- 无现有 capability 需要修改 -->

## Impact

- `legged_lab/assets/` — 新增 `hu_d04/` 目录
- `legged_lab/envs/tienkung/tienkung_env.py` — 修改硬编码关节名
- `legged_lab/envs/tienkung/run_cfg.py` — 新增 HU_D04 配置类
- `legged_lab/envs/__init__.py` — 注册新任务
- `legged_lab/envs/tienkung/datasets/` — 需要 GMR 生成 AMP 参考动作数据
