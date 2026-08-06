# 03. Locomotion 高度优化与时序平滑

## 目标

把 GVHMR 人体轨迹整理为物理后端可以消费的 SMPL-H 序列，修正高度/脚部接触造成的漂浮和穿地，再输出稳定的 GMR 身体来源。

## 工作流

```text
001_converted.npz
  -> SMPL-H 资产/分段/ACCAD 资源加载
  -> optimizer_v2.py 高度与接触优化
  -> root/body 轨迹清洗
  -> temporal smoothing（默认 smooth_window=9）
  -> 001_smoothed.npz
```

该阶段不是 PHC，也不是 GMR 的机器人 IK；它只在人体坐标/SMPL-H 轨迹层面工作。

## 启动方法

推荐通过总入口启动：

```bash
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage human --clip-filter chairwood
```

已有 GVHMR 结果、只想重算时，使用配置中的 `resume.force_locomotion` 和 `resume.force_smoothing`，
不要直接删除整个 clip 工作区。底层实现由 `GVHMR-hand/GVHMR-main/tools/pipeline/run_pipeline.sh` 调用，
Locomotion 代码在 `locomotion_pipeline-main/code/`。

## 关键输入和资源

| 输入/资源 | 用途 |
|---|---|
| `001_converted.npz` | GVHMR 到 Locomotion 的身体接口 |
| `locomotion_pipeline-main/code/configs/config.yaml` | 优化器参数 |
| `assets/smplh-seg/*.json` | 身体部位/穿透检测分段 |
| `assets/smplh/*.pkl` | SMPL-H 模型 |
| `assets/ACCAD/` | 高度优化参考动作 |

当前仓库保留分段 JSON；SMPL-H 与 ACCAD 在服务器/Docker 的 `/models/locomotion_assets` 外部挂载中提供。

## 输出

```text
001_converted.npz
001_contact_stabilized.npz（若启用对应后处理）
001_smoothed.npz
locomotion*/                 # 调试缓存、日志和中间统计
```

身体核心字段是 `root_orient (T,3)`、`pose_body (T,63)`、`trans (T,3)`、`betas (16,)`、
`mocap_frame_rate=30`；旋转为 axis-angle，平移单位为米。

## 验收和常见问题

1. NPZ 可以 `np.load(..., allow_pickle=False)`，所有时间数组首维相同且数值有限。
2. 帧率仍为 30 FPS；PHC 成功时 GMR 使用最终候选，失败时必须有 `001_smoothed.npz` 保底。
3. 缺 `SMPLH_*` 或 `ACCAD` 会在优化器启动前失败，先检查挂载路径。
4. 轨迹整体高度异常时，先核对坐标和配置，不要在 GMR 阶段重复做全局高度修正。
