# 06. 质量门禁与最终资产

## 目标

在不依赖 GT 的情况下，用文件完整性、帧数/FPS、手部诊断、根速度、GMR 四元数和视频可读性对每个 clip 分层；只有通过门禁的结果才导出为对接资产。

## 启动方法

```bash
# 只读重算质量，不重新运行模型
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage quality --clip-filter chairwood

# 读取已有工作结果，执行人体-only 商品导出
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage product --clip-filter chairwood
```

人体配置已设置 `object.enabled: false` 和 `product.object_policy: exclude`，不会调用场景重建或物体导出。

## 检查内容

| 类别 | 代表检查 |
|---|---|
| files | 必需文件、NPZ `allow_pickle=False` 读取、数值有限 |
| frame | human/hand/robot/camera/preview 帧数和 FPS 一致 |
| human | 根速度 P95、身体轨迹有效性 |
| hands | valid ratio、重投影 P90、spike/repaired ratio |
| GMR | 根四元数范数、关节越限率、dof 列名 |
| visualization | GMR 与 2×2 视频可读性和时长 |

`fail` 不能导出；`warn` 是否导出由 `product.minimum_quality_status` 决定，当前默认为允许 `warn`，但对接时应将报告一起交付。

## 输出

工作区：

```text
<output.root>/<clip>/quality_report.json
<output.root>/quality_overview.csv
<output.root>/quality_overview.json
```

人体-only 商品目录：

```text
<product.root>/<clip>/
  human_motion.npz
  human_phc_motion.npz        # PHC 启用且有有效结果时
  robot_motion.npz
  robot_hand_motion.npz
  camera.npz
  preview_2x2.mp4
  quality_report.json
  manifest.json
  pipeline_config.yaml
  rights.json
  checksums.sha256
<product.root>/catalog.jsonl
```

商品 NPZ 必须是压缩格式、无 `dtype=object`、可用 `allow_pickle=False` 重开，且不暴露服务器绝对路径。

## 清理和验收

`human_sharpa_batch.yaml` 的 `prune_workspace_after_export: true` 只在商品 NPZ、manifest、checksums 和质量门禁全部成功后删除对应 scratch clip。调试 YAML 默认不清理。

- [ ] 每条资产都有 manifest、rights、checksums。
- [ ] `quality_report.json` 的 status/verdict 已被接收方记录。
- [ ] 根四元数字段使用 `root_quat_xyzw`。
- [ ] 预览和 NPZ 来自同一运行及 config snapshot。
- [ ] 商品包不含 checkpoint、pickle、原始 feature、密钥或未授权输入。
