## Context

TienKung-Lab 当前仅支持 TienKung2 Lite 机器人（20 DOF，`_l/_r` 后缀命名）。HU_D04（33 DOF，`left_/right_` 前缀命名）需要适配。核心挑战：关节命名规范差异、reward 函数中的硬编码关节名、脚部 link 名称差异。

## Goals / Non-Goals

**Goals:**
- 支持 HU_D04 机器人进行奔跑训练
- 保持现有 TienKung 功能不受影响
- 最小化代码改动，复用现有 reward 函数框架

**Non-Goals:**
- 不修改现有 TienKung 机器人配置
- 不实现 waist、wrist、head 关节的主动控制（训练跑步时固定）
- 不实现 AMP 参考动作数据生成（需要 GMR 工具单独处理）

## Decisions

### 1. 机器人配置放置位置
**决定：** 在 `legged_lab/assets/hu_d04/` 下新建配置文件，与 TienKung 结构一致。

**理由：** 保持项目结构一致性，便于维护和扩展。

### 2. 关节命名映射策略
**决定：** 在 `tienkung_env.py` 中添加关节名查找逻辑，支持动态关节名，而非硬编码。

**理由：** 原代码硬编码了 `ankle_roll_*_joint`，需要改为通过配置传入关节名。这样可以复用同一个环境类。

### 3. 训练时控制的 DOF 范围
**决定：** 训练时仅控制腿部 12 DOF + 手臂 8 DOF，waist 和 head 关节固定在初始姿态。

**理由：** 跑步任务不需要 waist 和 head 的主动控制，减少动作空间维度，加速训练收敛。

### 4. 脚部 link 名称处理
**决定：** 在机器人配置中定义 `feet_body_names` 和 `terminate_contacts_body_names`，reward 函数通过配置获取。

**理由：** 原代码通过正则匹配 `ankle_roll.*`，HU_D04 的脚部 link 是 `contact_foot_center_{L,R}`，需要配置化。

### 5. AMP 数据复用
**决定：** 暂不生成 HU_D04 的 AMP 数据，先用 TienKung 的 walk/run 数据作为初始参考，后续再用 GMR 生成专用数据。

**理由：** AMP 对参考动作的依赖是软约束，初期可以用相近的动作数据启动训练。

## Risks / Trade-offs

- [Risk] HU_D04 和 TienKung 身体比例差异较大，TienKung 的 AMP 数据可能不适合 → Mitigation: 先训练无 AMP 的版本，验证基本运动能力
- [Risk] 手臂 DOF 差异（4 vs 7）可能导致观测空间维度不匹配 → Mitigation: 固定 wrist 关节，仅控制 shoulder + elbow
- [Trade-off] 复用 TienKungEnv 类 vs 新建 HU_D04Env → 选择复用，通过配置差异化工厂模式
