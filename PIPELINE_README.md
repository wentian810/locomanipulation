# 视频到 G1 + Sharpa 机器人：人体-only 运行手册

更新日期：2026-08-06

当前正式范围只有：

```text
原视频 -> GVHMR/Hand4Whole++ -> Locomotion -> PHC -> GMR/G1 + Sharpa -> 2×2/质量/资产
```

场景重建和物体重建不属于本交付入口。

## 最快启动

```bash
# 配置/依赖检查，不运行推理
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage all --check --print-config

# 运行一条视频；clip-filter 是输入文件名子串
PYTHONUNBUFFERED=1 python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage all --clip-filter chairwood
```

工程化主入口是 `scripts/run_pipeline_from_config.py`；底层 Bash 文件是实现细节。

## 配置选择

| 配置 | 用途 | 工作区清理 |
|---|---|---|
| `configs/pipelines/human_sharpa.yaml` | 单条/小批量检查，保留中间结果 | 否 |
| `configs/pipelines/human_sharpa_batch.yaml` | 批处理和正式资产导出 | 成功导出后按配置清理 |

人体配置的关键值：`human.backend: hand4wholepp`、`phc.enabled: true`、
`gmr.hand_model: sharpa`、`object.enabled: false`、帧率 30 FPS。

## 阶段和输出

| 阶段 | 输入 | 输出 |
|---|---|---|
| 输入/准入 | MP4/MOV 等 | 工作视频、`fullbody_preflight.*` |
| GVHMR/手部 | 工作视频 + 模型 | `hmr4d_results.pt`、`001_converted.npz`、手部 sidecar、相机/人体视频 |
| Locomotion/平滑 | converted NPZ | 高度/接触优化、`001_smoothed.npz` |
| PHC | smoothed NPZ + Isaac Gym | `001_phc*.npz`、PHC 修复视频/报告 |
| GMR/Sharpa | 身体 + 手部 sidecar | `robot_motion.pkl`、`001_sharpa_chain_hands.npz`、GMR MP4 |
| 2×2/质量/导出 | 完整工作目录 | 2×2、质量报告、安全 NPZ、manifest/checksums |

细节见 `docs/PIPELINE_INTEGRATION_GUIDE.md` 和 `docs/modules/01`–`07`。

## 已有结果的追加操作

```bash
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml --stage quality --clip-filter chairwood
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml --stage product --clip-filter chairwood
```

`product` 不重跑模型，必须已有可信 `robot_motion.pkl` 和相关人体/手部输出。

## Docker

```bash
docker/prepare_phc_context.sh
docker/prepare_conda_packs.sh
sudo -n docker buildx build --load --progress=plain \
  --build-context phc=/home/jixingyu/.codex_phc_context_20260806 \
  --build-context envs=/home/jixingyu/.codex_conda_packs_20260806 \
  -t locomotion-human-only:20260806 .
```

镜像固定代码和两个 Conda 环境；大模型和原视频从 `/models`、`/data/input` 挂载。
完整挂载、自检、运行和 UCloud S3 物料说明见 `docs/modules/07_docker_and_delivery.md`。

本次交付目录还包含 `human-only-model-assets_20260806.tar.zst` 和 `dataset_new6_20260806.tar.zst`：前者是 GVHMR/Hand4Whole++/WiLoR/SMPL-X/locomotion 权重，后者是原始视频。两者不写入 Docker layer，解压后按 Docker 交付文档挂载。

## GitHub + UCloud US3 恢复

源码、配置和启动脚本从 GitHub 获取；镜像、两个已固化的 Conda 环境、PHC runtime、模型和示例视频从私有 US3 获取。接收方不需要在本机重新求解 Conda 依赖。完整命令、SHA-256 门禁、只读凭证边界和空目录验收步骤见 [docs/RELEASE_FROM_GITHUB_AND_S3.md](docs/RELEASE_FROM_GITHUB_AND_S3.md)；每个底层 CLI 的职责与直接调用限制见 [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md)。

服务器侧自检使用已经打好的本地交付目录，而不从 US3 回下载：

```bash
bash release/bootstrap_from_ucloud.sh \
  --artifact-dir /home/jixingyu/locomotion_human_only_delivery_20260806
bash release/run_human_only_docker.sh --check-only
```

恢复脚本只会把解压物放到 checkout 的 `.release/`；原始压缩包保持不变，且 `.release/`、模型、视频和所有凭证均被 Git 忽略。

## 重要语义

- GVHMR world 重力轴为 `-Y`，GMR/MuJoCo 为 `+Z` up。
- GMR 根四元数导出字段是 `root_quat_xyzw`。
- `source: final` 优先采用 PHC 成功轨迹；失败时必须明确回退并记录。
- 最终包不包含 pickle、checkpoint、SMPL/MANO、密钥或本机绝对路径。
