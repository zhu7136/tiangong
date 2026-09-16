## Purpose

定义 HU_D04 人形机器人的资产配置，包括 USD 模型路径、关节参数、电机参数和初始姿态，使其可在 IsaacLab 仿真环境中加载和控制。

## ADDED Requirements

### Requirement: HU_D04 USD 模型加载
系统 SHALL 支持通过 `ArticulationCfg` 加载 HU_D04 的 USD 模型文件。

#### Scenario: 加载 USD 模型
- **WHEN** 训练任务指定 HU_D04 机器人配置
- **THEN** IsaacLab 从 `humanoid-description/HU_D04_description/usd/HU_D04_01.usd` 加载模型

### Requirement: 关节定义完整性
配置 SHALL 包含 HU_D04 所有 33 个可动关节的定义：腿部 12 DOF、手臂 14 DOF、腰部 3 DOF、头部 2 DOF。

#### Scenario: 关节数量匹配
- **WHEN** HU_D04 配置被加载
- **THEN** 机器人拥有 33 个可动关节，关节命名与 URDF 一致（`left_/right_` 前缀）

### Requirement: 电机参数配置
配置 SHALL 为腿部、手臂、腰部、头部关节分别设置力矩限制、速度限制、刚度和阻尼。

#### Scenario: 电机参数生效
- **WHEN** 仿真开始运行
- **THEN** 各关节的力矩限制、速度限制、刚度、阻尼参数生效

### Requirement: 初始姿态设置
配置 SHALL 定义 HU_D04 的初始站立姿态，使机器人在仿真开始时处于稳定站立状态。

#### Scenario: 初始姿态正确
- **WHEN** 仿真环境重置
- **THEN** 机器人恢复到预定义的初始站立姿态

### Requirement: 接触传感器配置
配置 SHALL 为脚部启用接触传感器，用于检测地面接触和 reward 计算。

#### Scenario: 接触传感器激活
- **WHEN** 仿真开始运行
- **THEN** 脚部接触传感器处于激活状态，可检测地面接触力
