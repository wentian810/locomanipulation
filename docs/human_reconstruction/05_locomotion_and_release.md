# 05. Locomotion 身高优化与预 PHC 时序平滑

## 模块职责

本模块只处理人体身体轨迹：以 001_converted.npz 为输入，进行基于 SMPL-H 资产的身高/重力方向优化，并对根平移和身体旋转做鲁棒时序修复与平滑。其输出 001_smoothed.npz 是 PHC 的输入和 GMR 的保底人体源。

这里的 Locomotion 指人体运动后处理，不是机器人控制。本模块不调用 Isaac Gym、物理策略、机器人 IK 或 Sharpa 手链；它完成后由模块 06 决定是否进入 PHC。

## 工作流

~~~text
001_converted.npz
  -> 复制到 Locomotion 的单序列输入
  -> optimizer_v2.py：SMPL-H 高度/重力方向优化
  -> 001_optimized.npz
  -> smooth_motion.py：异常帧修复 + 四元数连续化 + Savitzky-Golay 平滑
  -> 001_smoothed.npz
  -> export_gvhmr_camera.py
  -> gvhmr_camera.npz
~~~

优化器入口是 locomotion_pipeline-main/code/optim/optimizer_v2.py，使用该子项目的 config.yaml 与 SMPL-H/参考动作资产。运行时调用中固定使用 --optim_height 1 和 --gravity_axis y-。如果可选的穿透/速度/重力检查过滤掉结果，内部脚本会退化重试为仅高度优化，并在 stage_status.json 写入降级原因。

## 时序平滑的行为

smooth_motion.py 不对 axis-angle 逐分量硬滤。它会：

1. 用根平移加速度和 MAD 阈值识别异常帧，并对异常段插值。
2. 将根和身体关节轴角转为四元数，统一四元数符号，避免 q 与 -q 的伪跳变。
3. 对四元数与平移应用合法窗口长度的 Savitzky-Golay 平滑。
4. 写回原 NPZ 的兼容字段，同时生成根加速度和异常帧数统计。

正常人体链调用的保护参数为 pose window 11、translation window 15、根加速度阈值 10.0、关节加速度阈值 300.0。窗口并非越大越好；过大将抹掉快速肢体动作。若修改它们，必须作为独立版本重新验收。

## 启动方式

总入口会自动运行本模块：

~~~bash
/opt/conda/envs/locomotion/bin/python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage all \
  --dataset-dir /data/raw_videos \
  --output-root /data/human_output \
  --set input.work_video.directory=/data/human_work \
  --clip-filter <clip>
~~~

若仅需要重新执行高度优化和平滑，可在保证上游 GVHMR 与 sidecar 已存在的前提下使用：

~~~bash
... --set resume.force_locomotion=true --set resume.force_smoothing=true \
    --clip-filter <clip>
~~~

手动运行平滑器只适合恢复与研究，不会重新生成相机或更新缓存指纹：

~~~bash
PY_LOCO=/opt/conda/envs/locomotion/bin/python
$PY_LOCO GVHMR-hand/GVHMR-main/tools/pipeline/smooth_motion.py \
  --input /data/human_output/<clip>/locomotion/optimizer/results_filter/001/001_optimized.npz \
  --output /data/human_output/<clip>/001_smoothed.npz \
  --report /data/human_output/<clip>/human_smoothing_report.json \
  --pose_window 11 --trans_window 15 \
  --acc_threshold 10.0 --joint_acc_threshold 300.0
~~~

手动命令完成后仍需重新导出 gvhmr_camera.npz，并以模块 06 的结构检查验证帧对齐。

## 输出与交接边界

~~~text
<out>/<clip>/
  001_converted.npz
  locomotion/
    .../results_filter/001/001_optimized.npz
  001_smoothed.npz
  gvhmr_camera.npz
  stage_status.json（仅降级/异常时）
~~~

001_smoothed.npz 是 PHC 之前的身体主文件。001_optimized.npz 用于定位高度优化问题，001_converted.npz 用于追溯 GVHMR 原值，它们都是中间产物，不应用于新的对接代码。

下一模块会在 PHC 后发布 001_final.npz 选择点，原因是所有下游消费者统一读取 final。PHC 成功时它选择 grounded 结果；PHC 被关闭或本次失败时选择 001_smoothed.npz，并记录明确原因。不能仅凭 final 文件存在判断 PHC 成功。

## 成功判定

| 检查 | 期望 |
|---|---|
| 文件 | 001_smoothed.npz 存在且可读取 |
| shape | root_orient 为 (T,3)，pose_body 为 (T,63)，trans 为 (T,3) |
| 时间 | T 与 001_smplx_hands.npz 及 gvhmr_camera.npz 相同；fps=30 |
| 数值 | 所有核心 float 字段有限；不存在 NaN、Inf |
| 运动 | 根平移无孤立大跳，身体朝向无一帧 180° 翻转 |
| 几何 | 预览中人物整体高度合理，不因优化漂浮或沉入地面 |

## 常见问题

| 现象 | 原因边界 | 处理 |
|---|---|---|
| 找不到 SMPL-H 资产 | Locomotion 模型挂载不完整 | 按 Docker 文档挂载 locomotion assets；不要从其他版本随意拷贝模型 |
| optimizer 没有结果 | 输入 NPZ 损坏或可选检查过滤掉序列 | 查看 clip log 与 stage_status.json；保留高度-only 降级记录 |
| 平滑失败后仍有结果 | 脚本会退回使用优化器输出 | 视为 warn，人工审核并在交付清单标注；不要假装已平滑 |
| 人物落地不理想 | 单目尺度、姿态或输入全身证据不足 | 先回查 GVHMR/输入；高度优化不能替代 PHC 物理修复 |
| 下游想要 Z-up | 坐标系统不同 | 在下游做显式变换，保留原 human 文件不改写 |

本模块至此结束人体预处理。下一模块 PHC 负责物理修复，之后才由 GMR/Sharpa 消费选出的 final 轨迹。场景和物体始终不属于本组 human-only 文档。
