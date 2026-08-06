# 04. PHC 物理修复

## 目标

用 Isaac Gym 中训练好的 PHC policy 对人体身体轨迹进行物理可行性修复，并输出修复结果和可视化证据。PHC 只修身体，不负责 MANO 手指，也不负责 GMR 机器人 IK。

## 环境边界

PHC 使用独立的 Python 3.8 环境：

```text
phc Python 3.8 / torch 2.4.1 + cu118 / Isaac Gym Preview 4
SMPLSim / poselib / PHC package
```

GVHMR、Locomotion、GMR 使用 `locomotion` Python 3.10 环境。不要把两个环境合并，尤其不要在 PHC 进程中先导入 torch 再导入 Isaac Gym。

## 工作流

```text
001_smoothed.npz / 当前选定身体来源
  -> convert_zitai_to_phc.py / batch_repair_zitai.py
  -> Isaac Gym Humanoid policy inference
  -> export_repaired_smpl_npz.py
  -> PHC smoothing/grounding
  -> 001_phc*.npz + PHC MP4/报告
```

当前总入口在 `run_pipeline.sh` 内部调用 PHC；对接方不应自己拼一套不同的 Isaac Gym 环境变量。

## 启动方法

正式启动：

```bash
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage human --clip-filter chairwood
```

只检查 PHC 环境：

```bash
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh" \
PHC_ENV_NAME=phc bash scripts/setup_phc_server_env.sh "$PWD"
```

## 主要输出

```text
<clip>/001_phc.npz 或 001_phc_smoothed*.npz
<clip>/phc_renderings/ 或 videos/*phc*.mp4
<clip>/*phc*report*.json
```

具体文件名由当前 PHC wrapper 和 smoothing 配置决定；GMR `source: final` 按完成状态选择可信 PHC 文件，否则回退到 `001_smoothed.npz`。wrapper 失败会写入 `<output.root>/human_stage_failures.json`，不会伪装成通过。

## 验收和故障

- 能 import Isaac Gym、poselib、SMPLSim、smplx；policy 文件存在。
- 输出长度、FPS、`root_orient/trans/pose_body` 与输入可追踪。
- `PyTorch was imported before isaacgym`：调整 import 顺序或使用 Docker 预检。
- EGL/X11 报错：人体-only 默认 EGL；服务器需要 NVIDIA Container Toolkit 与 `--gpus all`。
- 显存不足：先单 clip、减少并行，不要同时启动多个 Isaac Gym repair 进程。
