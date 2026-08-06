# 02. GVHMR 与 Hand4Whole++

## 目标

从统一工作视频估计人体 SMPL-H/相机和双手 MANO，形成后续 Locomotion、PHC、GMR 共用的身体轨迹和手部 sidecar。
当前生产后端是 `hand4wholepp`。

## 工作流

```text
工作视频 -> GVHMR 人物轨迹 / ViTPose wholebody
  -> GVHMR SMPL-H 全局人体 -> Hand4Whole++ MANO 双手
  -> direct-MANO joint 一致性重算
  -> 短缺失 crop tracking -> temporal/finger filter
  -> 高证据帧的可见手 2D refinement -> 身体 NPZ + sidecar + 相机/预览
```

2D refinement 是证据门控的可选后处理，不改变 Hand4Whole++ 主估计器边界，也不能单独判定手掌正反面语义。

## 启动方法

正式入口：

```bash
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage human --clip-filter chairwood
```

直接调试底层 wrapper（仅开发者使用）：

```bash
PIPELINE_ROOT=/workspace/locomotion \
DATASET=/data/input OUTPUT_BASE=/data/output \
bash GVHMR-hand/GVHMR-main/tools/pipeline/run_batch_dataset6_hand4wholepp.sh
```

## 关键输入

| 输入 | 用途 |
|---|---|
| GVHMR checkpoint | 人体姿态和相机 |
| HMR2 checkpoint | GVHMR 底层人体模块 |
| ViTPose wholebody | 人体/手部关键点 |
| Hand4Whole++ `snapshot_6.pth` | 双手 MANO |
| SMPL/SMPL-X/MANO | 参数模型和关节计算 |
| YOLO/人物轨迹 | 预检和 crop 约束 |

## 主要输出

```text
gvhmr_out/                    # GVHMR 缓存、人体和相机相关结果
001_converted.npz             # 供 Locomotion 使用的转换结果
001_smplx_hands.npz           # 左右手 MANO sidecar，T 与身体一致
gvhmr_camera.npz              # 相机/坐标/对齐信息
videos/*gvhmr*.mp4            # 人体可视化
hamer_diagnostics/            # 当前后端仍可能保留诊断命名
```

内部缓存如 `hmr4d_results.pt`、ViTPose feature 和原始 checkpoint 不进入最终商品资产。

## 手部 sidecar 契约

至少包含：

```text
left/right_hand_pose       (T, 45)
left/right_hand_valid      (T,)
left/right_hand_quality    (T,)
left/right repaired flags  (T,)
bbox / reprojection diagnostics
```

它必须和身体的 `T`、FPS 对齐。GMR/Sharpa 只使用 sidecar 中经过有效性与可靠度门控后的轨迹。

## 资源缺失和故障

- 缺 GVHMR/HMR2/ViTPose/SMPL/MANO：不能进入 human stage。
- 缺 Hand4Whole++ snapshot：不能使用当前生产 backend，不要静默切换 backend。
- GPU OOM：先把 `human.hand_batch_size` 调为 1、保持 `low_memory: true`，再检查旧进程。
- 预览正常但手部质量低：看 `quality_report.json` 的 hands 阶段，不要只看 GMR MP4。
