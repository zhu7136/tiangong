## 1. 机器人资产配置

- [x] 1.1 创建 `legged_lab/assets/hu_d04/__init__.py` 和 `hu_d04.py`，定义 `HU_D04_CFG` 配置（USD 路径、33 关节初始姿态、电机参数），验证配置可被 Python 导入
- [x] 1.2 在 `legged_lab/assets/__init__.py` 中导出 `HU_D04_CFG`，验证 `from legged_lab.assets.hu_d04 import HU_D04_CFG` 成功

## 2. 环境配置适配

- [x] 2.1 创建 `legged_lab/envs/tienkung/run_hu_d04_cfg.py`，基于 `run_cfg.py` 修改 `robot=HU_D04_CFG`，更新 `terminate_contacts_body_names` 为 `["left_knee_link", "right_knee_link", "left_shoulder_roll_link", "right_shoulder_roll_link", "left_elbow_link", "right_elbow_link", "pelvis"]`，`feet_body_names` 为 `["contact_foot_center_L", "contact_foot_center_R"]`，验证配置可导入
- [x] 2.2 更新 reward 函数中的关节名正则：`hip_roll_.*_joint` → `left_hip_roll_joint|right_hip_roll_joint`，`hip_pitch_.*_joint` → `left_hip_pitch_joint|right_hip_pitch_joint`，`knee_pitch_.*_joint` → `left_knee_joint|right_knee_joint`，`ankle_pitch_.*_joint` → `left_ankle_pitch_joint|right_ankle_pitch_joint`，`ankle_roll_.*_joint` → `left_ankle_roll_joint|right_ankle_roll_joint`，`shoulder_pitch_.*_joint` → `left_shoulder_pitch_joint|right_shoulder_pitch_joint`，`shoulder_roll_.*_joint` → `left_shoulder_roll_joint|right_shoulder_roll_joint`，`elbow_pitch_.*_joint` → `left_elbow_joint|right_elbow_joint`，验证配置语法正确

## 3. 环境类修改

- [x] 3.1 修改 `tienkung_env.py` 中硬编码的关节名查找：将 `ankle_pitch_l_joint` 等改为从配置动态获取，添加 `self.left_ankle_pitch_id, self.right_ankle_pitch_id` 等变量，验证代码无语法错误
- [x] 3.2 修改 `tienkung_env.py` 中脚部位置获取逻辑：将 `ankle_roll_l_link` 等改为从配置动态获取的 link 名称，验证代码无语法错误

## 4. 任务注册

- [x] 4.1 在 `legged_lab/envs/__init__.py` 中导入 `HU_D04` 配置并注册 `hu_d04_run` 任务，验证 `task_registry` 包含新任务

## 5. 验证

- [x] 5.1 运行 `python -c "from legged_lab.envs import task_registry; print(task_registry.task_classes)"` 验证新任务已注册
- [x] 5.2 运行 `python legged_lab/scripts/train.py --task=hu_d04_run --headless --logger=tensorboard --num_envs=4` 验证训练可启动（不验证收敛）
