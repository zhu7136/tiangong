# HU_D04 续训入口修复（2026-11-19）

## 当前执行路径：Git 输入 + Flux 官方 SDK 回传

本次入口使用 `--flux-native --git-inputs --run-id <公开的8位执行标识>`，不需要私有配置、
账号密钥、签名 URL、个人存储或 W&B 全局 Key 下发到 worker。
已审查官方 `gradmotion 1.0.12` 的 checkpoint 发现和收尾逻辑：`gm-run` 在训练启动前
记录监控基线，在监控根目录内发现 `.pt/.pth/.ckpt`，并在入口退出后等待上传收尾。
入口会核对监控模块源码哈希、平台 TASK_ID 及输出目录是否位于初始 cwd 的监控树内。
最终 checkpoint 的 `infos` 同时保存非敏感环境配置、agent 配置与训练摘要，以便重新下载验收。

训练只改变前进指令范围至 `(1.2, 2.8) m/s`，不宣称实际速度已达到 2.8 m/s。
保持完整模型、critic、优化器、AMP、归一化和迭代号恢复；单卡、最多 2000 次 PPO 更新、
worker 看门狗 3600 秒，不自动重试。worker 上限不是平台账单上限，镜像启动与官方 SDK
上传收尾另计；尚未发生的上传不能当作已经持久化。
必须在任务结束后从服务器读取 `final_checkpoint.pt` 记录、重新下载校验哈希和恢复结构，
再做匹配条件的独立速度评估。训练 batch 均速不能证明稳定最高速度。

本次无 GPU 检查：入口合成测试 39 项、官方模型监控测试 27 项；真实 Git 分片合并
及 SHA-256 校验独立执行。以上均不等于 GPU 运行验证。

## 后续纠正：本工作流使用普通 Git 分片，不要求个人存储

用户确认本 Flux 工作流采用手册 §5.2 的 Git 分片，并于 2026-11-19 明确允许
在现有公开 tiangong 仓库发布机器人资产及完整 checkpoint。旧文中的私有挂载仅是
可选交付分支，不是必须条件；此前让用户反复上传个人存储的建议不适用于本工作流。

新增 `data/hud04_resume_15500/` 的两个普通 Git 分片（48 MiB 与 19,385,261 bytes），
启动入口 `--git-inputs` 选择固定清单，逐片及整体校验后安全解压，不进行输入 HTTPS 下载。
CPU 检查可运行 `tools/hud04_speed_resume/verify_git_inputs.py --output-dir outputs/git-input-preflight`。
输出目录必须新建，不能覆盖已有产物。输出现已改为上文的官方 SDK 回传路径；
密钥、签名 URL 和私有配置绝不因本次公开数据授权而提交 Git。

以下是上一次修复的历史说明；涉及“必须私有挂载”的表述不再作为 Git 交付的前置条件。

## 范围与状态

依据 `HuD04_WBT/flux-training-playbook:v0` 的交付原则和 `:v3` 的完整日志取证流程修复。
本分支是代码修复，不是一次新 GPU 任务。原失败任务、原部署 commit 和主分支均不覆盖。
CPU 回归测试通过不等于 IsaacSim 启动、完整 checkpoint 恢复或速度提升已经通过。

## 修复内容

1. **明确安装错误的归属。** 完整日志显示失败包为 `evdev`，依赖链为
   `LeggedLab -> pynput -> evdev`；不能把其 `project.license` 错误归到机器人配置或根包。
   `setup.py` 默认仅保留 IsaacLab 无头训练的依赖，MuJoCo/键盘回放转为 `sim2sim` extra。
   需要原有交互回放的环境改用 `pip install -e '.[sim2sim]'`，其依赖仍应在匹配 ABI 的环境预构建与验证。
   不升级镜像 Torch/CUDA，不在付费 worker 中临时升级全局 setuptools。
2. **私有持久输入优先，未知远程路径默认拒绝。** 配置增加 `input_archive`：指向平台实际挂载的
   私有输入包绝对路径，校验大小及 SHA-256 后再安全解压；不把私有机器人资产或 checkpoint 提交到公开 Git。
   这里没有创建挂载，也没有证明当前 GPU 资源支持该挂载。路径、权限、资源兼容性仍须核实。
   缺少本地输入时默认报告 `local_input_required`，不再自动进行收费的外网下载尝试。
3. **远程输入只保留显式兼容分支。** 只有 `allow_remote_input: true` 才使用原先限定的
   `storage.googleapis.com` 单对象能力；仍禁止重定向、保持 240 秒总下载限制和大小/哈希限制。
   64 KiB 分块、每约 5 秒进度、30 秒后低吞吐提前停止，记录结构化原因和已下载字节。
   这不是已验证的加速方案，也不代表原任务已确定由限时终止。
4. **敏感配置不放到 argv。** 原平台 SDK 会打印 `gm-run` 参数，因此删去
   `--private-config-b64` 接口。Base64 不是保密。
   新入口使用 `--private-config-file`，文件必须为实际存在的绝对路径、非符号链接、最多 64 KiB，
   且不能给予 group/other 访问权限（0600 或更严格）。必须通过平台受支持的私有配置交付，不能提交到 Git。
   平台账号、Flux Key、GitHub Token、W&B 全局 Key 均不下发到训练子进程。
5. **失败可定位。** 启动失败事件含稳定原因、阶段、代码文件名/函数/行号及受控数字，
   不直接记录任意异常字符串、签名 URL 或原始私有参数。已创建输出目录时同时保留小型 `report.json`；
   启动前失败不保证能回传结果包，以事件和平台正式日志导出取证，不能谎称报告已上传。

## CPU 检查

在仓库根运行：

```bash
python tools/hud04_speed_resume/test_bootstrap.py
python -m py_compile tools/hud04_speed_resume/train_resume.py tools/hud04_speed_resume/input_delivery.py
```

合成测试覆盖私有本地输入、默认拒绝外网、哈希/长度、慢速与超时、传输异常不泄漏 URL、
私有配置权限/类型/路径穿越、失败定位、无头安装依赖拆分。不会加载真实 checkpoint，不访问网络，不启动 CUDA。

## 重启前仍需满足

- 确认修复分支固定 commit 及镜像/原有依赖兼容。
- 通过平台受支持方式私有交付原始输入包和配置，核实容器路径、权限、大小和哈希。
  `flux task storage list` 只能列举，不能据此编造上传命令或宣称挂载成功。
- 刷新单对象输出上传能力并重新核对排队/全周期有效期；旧签名不能直接复用。
- 构建新的明确执行意图，读回草稿后单次启动；不得重跑旧提交脚本绕过 one-shot 防重护栏。
- 保持单 GPU、2000 次更新/3600 秒上限；本次修复没有扩大预算或自动重试授权。
- 实际重启后以 `inputs_verified`、`checkpoint_loaded` 和 `update` 事件分别验收输入、完整恢复与 PPO。
  速度是否超过原基线只能由匹配的速度评估证明，不能拿训练 batch 均速替代。

历史续训方案为 `hu_d04_run`、4096 环境、指令范围 `(1.2, 2.0)`；本次按已确认的
2.8 m/s 目标将范围改为 `(1.2, 2.8)`。奖励、AMP、步态、执行器上限及 checkpoint 身份不变。