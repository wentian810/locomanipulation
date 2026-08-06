# 01. 输入、转码与全身准入

## 目标

把原始视频变成后续模块共同使用的稳定 RGB 流，并在昂贵推理前排除不满足“单主体、全身证据足够”的视频。

## 工作流

```text
input.dataset_dir -> 扩展名/clip_filter/时长筛选
  -> 1280×960、H.264、30 FPS 工作视频
  -> YOLO-Pose 均匀抽帧
  -> 头、左右腕、左右踝、主体比例和多人竞争检查
  -> 通过的视频交给 GVHMR；拒绝的视频只记录报告
```

配置位置：`configs/default.yaml` 的 `input`；正式人体配置覆盖输入目录和输出目录。

## 启动方法

```bash
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage all --check

python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage all --clip-filter chairwood
```

底层脚本为 `scripts/preflight_fullbody_gate.py`，实际参数以 YAML 为准；不要由对接方复制一套阈值。

## 输入要求

| 项目 | 当前约定 |
|---|---|
| 支持扩展名 | `.mp4 .avi .mov .mkv .m4v` |
| 工作尺寸 | 1280×960 |
| 工作帧率 | 30 FPS；PHC/GMR 共享该值 |
| 主体 | 每个 clip 只允许一个主要人物 |
| 时长 | 默认至少 8 秒 |
| 视频目录 | `input.dataset_dir`，不由管线删除 |

## 输出

```text
<project>/dataset_new6_work_1280/<clip>.mp4
<project>/<output.root>/fullbody_preflight.csv
<project>/<output.root>/fullbody_preflight.jsonl
```

`mode: gate` 会阻断不合格视频；`mode: report` 只记录、不阻断。当前生产配置使用 `gate`。
如果没有任何合格 clip，runner 会明确打印并跳过后续昂贵阶段。

## 常见问题

- `model missing`：挂载/放置 `models/yolo11n-pose.pt`，或在纯环境检查时关闭准入检查。
- 视频能播放但 OpenCV 读不到：先用 FFmpeg 转成 H.264、固定尺寸和 30 FPS。
- 通过准入但 GVHMR 失败：检查第二个人、身体边缘裁切和主体尺度。
- `fullbody_preflight.csv` 只是“是否允许进入推理”的门禁，不是人体结果。
