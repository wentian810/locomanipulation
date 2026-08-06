# 06. PHC 物理修复与最终人体轨迹选择

## 模块职责

PHC 将 Locomotion 平滑后的 SMPL-H 身体轨迹放入 Isaac Gym 相关运行链，尝试进行物理可执行性修复，并输出修复、后平滑和落地对齐版本。它不重新估计手部：双手始终沿用模块 03 产生的 001_smplx_hands.npz。

human-only 正式配置中：

~~~yaml
phc:
  enabled: true
gmr:
  source: final
~~~

这表示 PHC 是人体阶段的一部分，GMR 消费每次运行明确发布的 001_final.npz 选择点。PHC 失败不会让旧 PHC 结果被误用；本次运行会显式回退到 001_smoothed.npz 并记录原因。

## 输入与运行环境

| 输入 | 位置 | 说明 |
|---|---|---|
| 平滑身体轨迹 | <clip>/001_smoothed.npz | 模块 05 输出，30 FPS |
| PHC 工作输入 | <clip>/phc_in/001/001_optimized.npz | 运行脚本复制生成 |
| PHC 代码与 Isaac Gym | phc-dev-felix-pipeline | Docker 镜像/服务器运行时提供 |
| PHC 策略权重 | PHC output 下的 Humanoid.pth | primitive 与 composer 两个策略 |
| SMPL 模型 | PHC data/smpl | 用于地面修复 |
| PHC sample-data | `phc-dev-felix-pipeline/sample_data/` | AMASS 性别/形状与站立样本；发布时来自 S3 的 `human-only-phc-sample-data_20260806.tar.zst` |
| Python 环境 | Conda phc | Python 3.8 + Isaac Gym；不同于 locomotion 环境 |

底层包装函数会为 PHC 注入独立的 LD_LIBRARY_PATH、PYTHONPATH、ISAACGYM_PATH 和 phc Conda bin 路径。不要在 locomotion 环境中直接 import Isaac Gym，也不要用 phc 环境执行 GVHMR 或 GMR。

## 实际工作流

~~~text
001_smoothed.npz
  -> phc_in/001/001_optimized.npz
  -> PHC batch_repair_zitai.py
  -> phc_repaired 中的 repaired/validated NPZ
  -> 001_phc_grounded.npz：SMPL mesh floor alignment
  -> 001_phc_smoothed.npz：PHC 输出尖峰平滑
  -> 001_phc_smoothed_grounded.npz：最终落地对齐
  -> 001_final.npz 符号链接 + final_motion_selection.json
~~~

PHC 运行时记录 Isaac 视频到 phc_renderings。之后的平滑仍使用四元数连续化与 Savitzky-Golay 逻辑；地面修复根据 SMPL mesh 和当前重力方向校正，不是简单把 root 平移的一个分量取最小值。

## 最终轨迹选择契约

run_pipeline.sh 每次都会写：

~~~text
<clip>/001_final.npz
<clip>/final_motion_selection.json
~~~

final_motion_selection.json 至少包含 selected_stage、selection_reason 和 selected_file。正常成功优先为：

~~~text
phc_smoothed_grounded
  -> phc_smoothed
  -> phc_grounded
  -> phc
~~~

若 PHC 被关闭、没有修复输出或本次 PHC 调用失败，selected_stage 为 smoothed，selection_reason 必须说明 phc_disabled、phc_no_output 或 phc_runtime_failed 等回退原因。GMR source=final 时必须读取该 JSON；不能扫目录中“最新的 001_phc 文件”，因为旧文件可能来自另一次运行。

## 启动、跳过与重算

对接方的正式启动方式仍是总入口：

~~~bash
PYTHONUNBUFFERED=1 /opt/conda/envs/locomotion/bin/python \
  scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage all \
  --dataset-dir /data/raw_videos \
  --output-root /data/human_output \
  --set input.work_video.directory=/data/human_work \
  --clip-filter <clip>
~~~

需要重算 PHC 时保留前序缓存并设置：

~~~bash
... --set resume.force_phc=true --clip-filter <clip>
~~~

只做一次 PHC 开关 A/B 时可使用 --skip-phc。该参数会覆盖 YAML，最终 selection 文件会清楚显示 smoothed 回退。它适用于比较或环境排障，不应与“PHC 已通过”的交付包混用。

没有单独的 --stage phc。工程选择让 PHC 依赖于与当前视频、手部、Locomotion 和缓存指纹一致的同一条 human 流程；若只重跑 PHC，使用 --stage all 加上述强制开关，前序阶段由 skip_existing 复用。

## 输出与成功判定

| 输出 | 必需性 | 用途 |
|---|---|---|
| phc_repaired/ | 工作缓存 | 追溯 PHC 原始 repaired/validated 轨迹 |
| 001_phc_grounded.npz | 正常成功时应有 | 原始 PHC 结果的落地版本 |
| 001_phc_smoothed.npz | 正常成功时应有 | 后平滑版本 |
| 001_phc_smoothed_grounded.npz | 首选 | 后平滑再落地的身体轨迹 |
| 001_phc_smoothed_report.json | 有后平滑时 | 异常帧、平滑前后统计 |
| phc_renderings/*.mp4 | 正常成功时应有 | Isaac 人工审核视频 |
| 001_final.npz 与 final_motion_selection.json | 总是需要 | 下游唯一可信选择点 |

成功不仅是 PHC 进程返回 0。还应确认 final_motion_selection.json 指向本次预期文件、final 与手部/camera 的帧数一致、PHC 预览可读、身体没有明显爆炸、漂浮或地面穿透。

## 常见失败与交接处理

| 现象 | 原因边界 | 处理与交接结论 |
|---|---|---|
| 找不到 phc Python 或 Isaac binding | 环境/镜像不完整 | 运行 Docker preflight；不能用 locomotion Python 顶替 |
| 找不到 `sample_data/amass_isaac_gender_betas_unique.pkl` | PHC sample-data 运行时包未下载或未挂载 | 重新执行 bootstrap；不得接受 smoothed fallback 作为 PHC 成功 |
| PHC 进程失败 | 驱动、策略、序列稳定性或 Isaac 环境 | 保留 phc 日志和工作区；final 会安全回退，质量报告将标记原因 |
| 没有 repaired NPZ | PHC 未产生可用结果 | 记录 phc_no_output 回退，禁止把旧 phc 文件作为本次输出 |
| 后平滑失败 | 数值/依赖异常 | 可保留 grounded 原结果，但交付状态至少为 warn |
| 地面修复失败 | 模型路径/网格或序列异常 | 保留原 PHC 结果并在 selection/日志记录；人工检查后再决定是否导出 |
| Isaac 视频缺失 | 渲染配置或运行时失败 | PHC 数值结果仍需单独检查；若预览是产品必需项，质量/导出会拒绝 |

PHC 只负责人体身体轨迹，不改变 SMPL-X/MANO sidecar。若身体帧数改变或手部被重新估计，应视为管线契约破坏，而不是正常 PHC 行为。
