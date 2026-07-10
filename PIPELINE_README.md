# 视频到机器人运行手册

更新日期：2026-07-03

本手册只描述当前实际入口。底层 Bash 脚本仍可调试，但工程化运行统一通过
`scripts/run_pipeline_from_config.py + YAML`。

## 1. 选择一份 YAML

| 配置 | 物体 | 工作区 | 适用场景 |
| --- | --- | --- | --- |
| `human_sharpa.yaml` | 无 | 保留 | 原始 GVHMR/Locomotion/PHC/GMR 流程检查 |
| `human_sharpa_batch.yaml` | 无 | 导出成功后清理 | 几千条人体动作批处理 |
| `rgb_monocular_sharpa.yaml` | 有 | 保留 | 单条物体重建调试 |
| `rgb_monocular_sharpa_batch.yaml` | 有 | 导出成功后清理 | 已确认参数后的物体批处理 |

每次先检查最终配置：

```bash
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage all \
  --clip-filter chairwood \
  --check \
  --print-config
```

`--check` 不运行推理。`--dry-run` 会继续展开各阶段命令，但不执行。

## 2. 执行命令

### 2.1 原始人体-only 全流程

```bash
cd <repo-root>
PYTHONUNBUFFERED=1 python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage all \
  --clip-filter chairwood
```

执行顺序：

```text
GVHMR/Hand4Whole++ -> Locomotion -> 平滑 -> PHC -> GMR/Sharpa -> 2×2
-> assets/dataset_new6_human_sharpa/<clip>
```

`object.enabled: false`，因此不需要腾讯云密钥，也不会打开点选窗口。

### 2.2 在已有的人体结果上追加物体

这正是旧输出接入物体时使用的方式：

```bash
source configs/secrets.env
PYTHONUNBUFFERED=1 python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/rgb_monocular_sharpa.yaml \
  --stage object \
  --output-root output_dir/dataset_new6_hand4wholepp_directmano_gmr_sharpa_aligned \
  --clip-filter chairwood
```

`--stage object` 要求对应 clip 已存在 `robot_motion.pkl`、`gvhmr_camera.npz` 和人体
渲染结果。当前 `segmentation_prompt_mode: auto`，YOLO 会在扩大的人体框内选择目标，
SAM-HQ/Cutie 自动分割，不需要逐条点击。

### 2.3 从零运行人体 + 物体

```bash
source configs/secrets.env
PYTHONUNBUFFERED=1 python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/rgb_monocular_sharpa.yaml \
  --stage all \
  --clip-filter chairwood
```

物体只有通过尺寸、有效率、投影等检查后才进入 GVHMR 与 GMR 面板。当前 2×2 布局：

| 左上 | 右上 |
| --- | --- |
| 原视频 | GVHMR（人体 + 物体） |
| PHC / Isaac Gym | GMR（机器人 + 物体） |

### 2.4 批处理

人体-only：

```bash
PYTHONUNBUFFERED=1 python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa_batch.yaml \
  --stage all
```

含物体：

```bash
source configs/secrets.env
PYTHONUNBUFFERED=1 python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/rgb_monocular_sharpa_batch.yaml \
  --stage all
```

示例物体批处理 YAML 的 `input.clip_filter: chairwood` 与当前唯一的
`object.clips` 条目一致。扩批时必须先为目标视频补齐 clip 配置，再修改/清空过滤器；
否则 Human 会为未配置视频产生工作缓存，而物体商品门禁不会接受它们。

批处理 YAML 使用独立 `scratch/`。每个资产包完成以下检查后，才删除该 clip 的工作区：

1. 人体、机器人、手、相机帧数和 FPS 一致；
2. 所有数值有限，四元数为单位四元数；
3. NPZ 可用 `allow_pickle=False` 重开；
4. 商品清单与全部 SHA-256 校验通过；
5. 自动质量结论至少达到 `product.minimum_quality_status`；
6. 物体批处理还要求物体适配和轨迹 QA 成功。

不要把现有调试输出目录手工改成批处理 scratch。先用保留工作区的 YAML 验证一条，再
切换批处理配置。

### 2.5 只导出已有结果

```bash
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage product \
  --clip-filter chairwood
```

`--stage product` 不重跑模型。它读取可信工作区内的 `.pkl/.pt`，但只把安全压缩 NPZ
写入资产包。若 `quality_evaluation.require_for_product: true`，会先为已有结果生成质量
报告，再执行商品门禁。

### 2.6 只重算质量报告

```bash
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage quality \
  --clip-filter chairwood
```

该命令不运行模型、不修改动作，只读取已有结果并更新
`output.root/<clip>/quality_report.json` 和根目录的 CSV/JSONL 汇总。

## 3. 数据流程

| 阶段 | 初始数据状态 | 成功输出数据状态 | 是否进入资产包 |
| --- | --- | --- | --- |
| 工作视频 | 原始 MP4/MOV 等 | 1280×960 H.264 视频 | 否 |
| GVHMR | 工作视频 | `hmr4d_results.pt`、incam/global 视频 | 仅最终参数与预览 |
| 手部 | 人物框、ViTPose | MANO 参数、`001_smplx_hands.npz` | 精简进 `human_motion.npz` |
| 转换 | GVHMR + MANO | `001_converted.npz` | 否 |
| Locomotion | converted NPZ | 高度/接触优化结果 | 通过后续选择间接进入 |
| 平滑 | Locomotion 结果 | `001_smoothed.npz` | 默认作为人体资产来源 |
| PHC | smoothed NPZ | repaired/grounded NPZ、Isaac 视频 | `human_phc_motion.npz` + 2×2 |
| GMR | `gmr.source` 指定 NPZ | `robot_motion.pkl`、机器人视频 | 转为 `robot_motion.npz` |
| Sharpa | 手部 sidecar | `001_sharpa_chain_hands.npz` | `robot_hand_motion.npz` |
| 相机 | GVHMR + 对齐偏移 | `gvhmr_camera.npz` | `camera.npz` |
| 物体分割 | 参考帧、人体框、目标名 | mask 与跟踪诊断 | 否 |
| 物体重建 | mask + RGB | 米制网格、6D 位姿 | 有效网格与 `object_motion.npz` |
| 2×2 | 四路视频 | `composite_2x2.mp4` | `preview_2x2.mp4` |

注意：默认 `gmr.source: smoothed`，所以 PHC 会生成左下角视频和修复结果，但 GMR
仍读取 `001_smoothed.npz`。若要让 GMR 使用 PHC 结果，显式设置
`gmr.source: phc_smoothed`，然后重新核验动作与物体对齐。

## 4. 工作输出与最终资产

工作目录允许存在大量缓存，便于续跑：

```text
output.root/<clip>/
  gvhmr_out/                 # 模型缓存和人体视频
  locomotion*/ phc*/        # 优化过程
  001_*.npz                 # 各阶段格式
  robot_motion.pkl          # 可信工作文件
  object_reconstruction/    # 物体适配与 QA
  composite_2x2.mp4
```

最终目录是稳定交付接口：

```text
product.root/<clip>/
  human_motion.npz
  human_phc_motion.npz
  robot_motion.npz
  robot_hand_motion.npz
  camera.npz
  object_motion.npz         # 可选
  object/mesh/              # 可选
  object/collision/         # 可选
  quality/object/           # 可选
  quality_report.json
  preview_2x2.mp4
  manifest.json
  pipeline_config.yaml
  rights.json
  checksums.sha256
```

`product.root/catalog.jsonl` 是全数据集索引，每条资产一行，不含本机绝对路径。

不交付：

- `hmr4d_results.pt`、MANO `.pt`、ViTPose feature；
- `robot_motion.pkl` 和 PHC pickle；
- 分割逐帧 mask、MoGe depth、跟踪临时文件；
- checkpoint、SMPL/MANO 模型、密钥和输入视频。

以当前 chairwood 为例，人体-only 商品包的主要体积来自约 16.7 MB 的 2×2 视频，
全部 NPZ 远小于原工作区。若不需要预览，可设 `product.include_preview: false`。

## 5. 自动质量分层

每条 clip 的主结论是 `pass | warn | fail`，分数只用于排序，不能覆盖关键失败：

```json
{
  "clip": "xxx",
  "mode": "human_only",
  "status": "warn",
  "stage_status": {
    "files": "pass",
    "human": "pass",
    "hands": "warn",
    "gmr": "pass",
    "visualization": "pass"
  },
  "overall_score": 86.4,
  "critical_fail_reasons": [],
  "warn_reasons": ["right_hand_reproj_error_relative_p90=0.4200"]
}
```

第一版用于自动发现坏片段，检查以下 12 组无 GT 指标：

1. 必需文件与安全可读性；
2. 人体、手、相机、GMR、预览和物体帧数一致率；
3. 左右手有效率；
4. 左右手相对重投影误差 P90；
5. 左右手时序尖峰率；
6. 左右手修复率；
7. 人体根速度 P95；
8. GMR 关节限位违反率和根四元数；
9. 物体有效率；
10. 物体平移/旋转跳变数；
11. 网格投影框与分割框 IoU 均值；
12. 手腕到物体 OBB 的接触代理比例，以及可选 MuJoCo 抬起验证。

人体-only 权重为文件 15、身体 25、手 35、GMR 15、可视化 10；含物体权重为
文件 10、身体 15、手 20、物体 20、接触 20、动态 15。文件/人体/手/GMR/物体的
关键失败直接判 `fail`；接触或非强制动态验证失败判 `warn`。接触指标是无 GT
几何代理，不等同于真实接触力，应与少量人工抽检和动态验证配合使用。

批量汇总：

```text
output.root/quality_summary.csv
output.root/quality_summary.jsonl
```

阈值、权重及动态验证强制策略均由 YAML 控制，详见
[configs/README.md](configs/README.md)。

## 6. 物体选择与质量约束

自动模式按以下顺序工作：

```text
object_name / detector_labels
  -> YOLO 候选
  -> 候选中心必须位于扩大的人体框
  -> 候选框与人体 ROI 重叠率检查
  -> SAM-HQ 中心点 + box prompt
  -> mask 至少有指定比例位于人体 ROI
  -> Cutie 全视频传播
```

关键 YAML：

```yaml
object:
  monocular:
    segmentation_prompt_mode: auto
    person_roi:
      bbox_scale: 2.0
      min_mask_inside_ratio: 0.60
    auto_detector:
      confidence: 0.05
      min_box_inside_person_roi: 0.50
      box_padding_ratio: 0.08
  clips:
    <clip-name>:
      object_name: chair
      reference_frame: 60
      expected_size_m: [0.45, 1.60]
      allow_no_object: false
```

`allow_no_object: false` 表示分割、尺寸或桥接失败时整条任务失败，不允许把错误物体悄悄
作为商品输出。更完整的参数中文说明见 [configs/README.md](configs/README.md)。

## 7. 续跑与重算

常用开关位于 YAML：

```yaml
resume:
  skip_existing: true
  force_hand_preprocess: false
  force_locomotion: false
  force_smoothing: false
  force_phc: false
  force_gmr: true
```

只调 GMR 时保持前面缓存，设置 `force_gmr: true`。分割目标、参考帧或物体名变化时，
应清除该 clip 对应的 `object_work/.../<clip>` 或使用新的 scratch/output root，避免
旧 mask/mesh 被误复用。

## 8. 常见失败

| 现象 | 原因与处理 |
| --- | --- |
| 缺少 `TENCENTCLOUD_*` | 当前 shell 未执行 `source configs/secrets.env`；只影响物体重建 |
| 自动分到墙 | 检查检测预览、`object_name`、参考帧和人体 ROI；不要降低 ROI 约束掩盖问题 |
| `--stage object` 缺文件 | `--output-root` 未指向已有完整人体/GMR 输出 |
| 商品导出失败 | 先看具体帧数/FPS/四元数/NPZ 报错；失败时不会清理工作区 |
| 质量结论为 `fail` | 查看 `quality_report.json` 的 `critical_fail_reasons` 和对应 metric details |
| 商品无物体 | `object_unavailable.json` 或物体 QA 未通过；调试 YAML 会保留现场 |
| GMR 与物体整体旋转 180° | 核对 `gmr.human_yaw_offset_deg`，人体和物体必须共享同一个值 |
| 磁盘继续增长 | 确认使用 `_batch.yaml`，并检查 `product.retention` 和 `_exported/*.json` |

## 9. 商业交付

商品包提供技术完整性，不自动授予版权或商业许可。销售前至少确认：

- 输入视频商业使用权；
- 画面人物/受试者授权；
- GVHMR 单独商业许可；
- 手部模型、checkpoint、机器人描述和物体网格许可；
- 隐私、肖像、数据跨境与目标平台规则。

对应状态写在 YAML 的 `product.rights`，并固化到每条资产的 `rights.json`。默认全部
为 `false`，这是刻意的安全默认值。
