# Loco-manipulation human-only pipeline

这是当前对外对接版本：从 RGB 原视频生成 G1 + Sharpa 的人体动作结果。

```text
原视频 -> GVHMR/Hand4Whole++ -> Locomotion -> PHC -> GMR/G1 + Sharpa -> 2×2/质量/资产
```

当前入口不包含场景重建和物体重建。

## 快速开始

```bash
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage all --check --print-config

python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage all --clip-filter chairwood
```

详细对接说明从 [PIPELINE_README.md](PIPELINE_README.md) 开始：

- [全流程对接总览](docs/PIPELINE_INTEGRATION_GUIDE.md)
- [输入与全身准入](docs/modules/01_input_and_preflight.md)
- [GVHMR 与 Hand4Whole++](docs/modules/02_gvhmr_and_hands.md)
- [Locomotion 与平滑](docs/modules/03_locomotion_and_smoothing.md)
- [PHC](docs/modules/04_phc.md)
- [GMR 与 Sharpa](docs/modules/05_gmr_and_sharpa.md)
- [质量与最终资产](docs/modules/06_quality_and_product.md)
- [Docker、挂载与 UCloud 交付](docs/modules/07_docker_and_delivery.md)
- [每个 CLI 的职责、输入输出与启动边界](docs/CLI_REFERENCE.md)
- [从 GitHub + UCloud US3 恢复可运行 release](docs/RELEASE_FROM_GITHUB_AND_S3.md)

## 代码结构

```text
configs/                         # 默认配置和 human_sharpa pipeline
scripts/run_pipeline_from_config.py
GVHMR-hand/GVHMR-main/           # GVHMR/Hand4Whole++ 调度
locomotion_pipeline-main/        # 高度/接触优化
GMR-master/                      # G1 重定向、Sharpa IK、MuJoCo
tests/                           # 自动测试
Dockerfile                       # 人体-only Docker runtime
docker/preflight.sh              # 容器内环境/资源/测试自检
```

模型权重、原视频和工作输出不进入 Git。PHC 运行代码由服务器的 `server-pipeline` 工作树提供给 Docker 构建上下文；GVHMR/Hand4Whole++/SMPL/ViTPose 等大权重在运行时挂载。

## 资产接口

长期交付目录包含：

```text
human_motion.npz
human_phc_motion.npz
robot_motion.npz
robot_hand_motion.npz
camera.npz
preview_2x2.mp4
quality_report.json
manifest.json
checksums.sha256
```

商品包不包含 pickle、checkpoint、模型权重、密钥或服务器绝对路径。Docker 镜像构建、权重挂载、预检和 S3 物料见 [Docker 交付说明](docs/modules/07_docker_and_delivery.md)。接收方从空目录恢复时，应遵循 [GitHub + UCloud US3 release 说明](docs/RELEASE_FROM_GITHUB_AND_S3.md)；其中的脚本会校验归档哈希后再加载镜像和挂载模型。
