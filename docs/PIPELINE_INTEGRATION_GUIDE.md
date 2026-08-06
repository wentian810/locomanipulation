# Human-only Locomotion 对接总览

更新日期：2026-08-06

本文是当前 `human-pipeline-support-contacts` 分支的对接入口。范围固定为：

```text
原视频 -> 全身准入 -> GVHMR/Hand4Whole++ -> Locomotion -> 时序平滑
        -> PHC 物理修复 -> GMR/G1 -> Sharpa 手部 IK -> 2x2/质量/资产导出
```

不包含场景重建、物体分割、Hunyuan3D、MoGe、MegaPose、FoundationPose，也不要求腾讯云密钥。

## 1. 对接方先看什么

| 目标 | 阅读/执行 |
|---|---|
| 了解阶段边界和文件 | 本文第 2–5 节 |
| 了解某个模块如何启动 | `docs/modules/01`–`06` |
| 使用 Docker | `docs/modules/07_docker_and_delivery.md` |
| 只跑一条视频 | `configs/pipelines/human_sharpa.yaml` |
| 批量跑视频 | `configs/pipelines/human_sharpa_batch.yaml` |

正式入口只有一个：

```bash
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage all
```

底层 Bash 文件是模块内部实现，不建议由对接方手工拼接阶段命令。

## 2. 运行时分层

### 2.1 代码层

```text
scripts/                         # 配置合并、准入、质量、资产导出
configs/                         # default + human pipeline YAML
GVHMR-hand/GVHMR-main/           # GVHMR、Hand4Whole++ 调度和后处理
locomotion_pipeline-main/        # 身高/接触优化
GMR-master/                      # G1 重定向、Sharpa IK、MuJoCo 渲染
tests/                           # 合同和质量测试
```

PHC 代码不在该 Git 分支中，而是从服务器上的 `server-pipeline` 工作树提供给 Docker 构建上下文；
Dockerfile 只复制 PHC 运行所需目录，不复制场景重建工作区。

### 2.2 外部模型层

以下文件不进 Git，也默认不进 Docker 镜像：

| 外部目录 | 用途 |
|---|---|
| `GVHMR-hand/GVHMR-main/inputs/checkpoints/` | GVHMR、HMR2、ViTPose、SMPL/SMPL-X、YOLO |
| `GVHMR-hand/GVHMR-main/third-party/Hand4Whole-plus-plus_RELEASE/` | Hand4Whole++ 代码和 snapshot |
| `GVHMR-hand/GVHMR-main/third-party/WiLoR/` | 备用手部后端 |
| `locomotion_pipeline-main/assets/smplh/`、`ACCAD/` | Locomotion 的 SMPL-H 与参考动作 |
| PHC 的 SMPL/Isaac Gym/策略权重 | PHC 运行资源；当前 Docker 将 PHC 权重纳入镜像 |

权重通常有许可和几十 GB 体积。对接方只需要把它们挂载到文档规定的 `/models` 位置，
不要把 `.ckpt/.pth/.pkl` 上传到代码仓库或写进配置文件。

## 3. 阶段契约

| 阶段 | 主要输入 | 主要输出 | 运行环境 |
|---|---|---|---|
| Input/Preflight | 原始视频 | 统一 1280×960 工作视频、准入报告 | locomotion |
| GVHMR + Hand4Whole++ | 工作视频、人物轨迹、模型权重 | `hmr4d_results.pt`、人体/手部渲染和手部 sidecar | locomotion |
| Locomotion | `001_converted.npz` | 高度/接触优化结果、`001_smoothed.npz` | locomotion |
| PHC | 平滑后的 SMPL-H 轨迹、PHC 权重 | `001_phc*.npz`、Isaac Gym 修复视频/报告 | phc/Python 3.8 |
| GMR | `final` 或 `smoothed` 身体轨迹、SMPL-X、机器人 XML | `robot_motion.pkl` | locomotion |
| Sharpa | `001_smplx_hands.npz`、Sharpa XML/mesh | `001_sharpa_chain_hands.npz` | locomotion |
| Render | 原视频、GVHMR/PHC/GMR 视频 | 2×2 MP4 | locomotion |
| Quality/Product | 工作目录完整输出 | 质量报告、压缩 NPZ、manifest、SHA-256 | locomotion |

每个 clip 的工作目录类似：

```text
output.root/<clip>/
  gvhmr_out/                    # GVHMR 缓存和人体视频
  001_converted.npz             # GVHMR -> Locomotion 接口
  001_smoothed.npz              # GMR 的身体保底来源
  001_phc*.npz                  # PHC 修复/平滑/接地结果
  001_smplx_hands.npz           # 手部 sidecar
  robot_motion.pkl              # GMR 内部可信工作文件
  001_sharpa_chain_hands.npz    # Sharpa 22-DoF 手部轨迹
  videos/                       # 稳定的 gvhmr/phc/gmr/2x2 别名
  quality_report.json
```

## 4. 配置和续跑

配置优先级为：

```text
CLI --set > 进程环境变量 > pipeline YAML > configs/default.yaml
```

推荐只修改 pipeline YAML；密钥永远用进程环境变量，不写 YAML。

```bash
# 只检查，不推理
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage all --check --print-config

# 展开阶段命令，但不执行
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage all --dry-run

# 选一条视频
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage all --clip-filter chairwood

# 已有结果只做质量/导出
python scripts/run_pipeline_from_config.py --config_dir configs/pipelines/human_sharpa.yaml --stage quality
python scripts/run_pipeline_from_config.py --config_dir configs/pipelines/human_sharpa.yaml --stage product
```

`resume.skip_existing: true` 允许续跑；强制重算时使用 YAML 中的 `force_*` 开关。
批处理使用 `scratch/`，导出成功后才允许清理 clip 工作区。

## 5. 坐标和数据语义

人体和相机来自 GVHMR world，当前重力约定为 `-Y`；GMR/MuJoCo 使用 `+Z` 向上。
GMR 根四元数为 `xyzw`。身体 NPZ 核心字段为 `root_orient (T,3)`、`pose_body (T,63)`、
`trans (T,3)`、`betas (16,)`、`mocap_frame_rate=30`，旋转是弧度 axis-angle、平移单位是米。

## 6. 对接方必须保证

1. 每条视频是单主体，并尽量包含头、双腕、双踝；准入门禁会拒绝长期裁切或多人视频。
2. 输入视频能被 FFmpeg/OpenCV 解码，最终帧率固定为 30 FPS。
3. 模型权重按 `docs/modules/07_docker_and_delivery.md` 挂载。
4. Docker 使用 `--gpus all`，NVIDIA Container Toolkit 和驱动可用。
5. 输出目录可写；输入目录建议只读挂载。
6. 不把 pickle、checkpoint、SMPL/MANO、密钥和原视频写入最终商品包。

## 7. 故障定位顺序

```text
docker/preflight.sh -> 配置 --check -> locomotion import -> PHC/Isaac import
                    -> 模型挂载 -> 单元测试 -> 单条视频 -> quality_report.json
```

先看 clip 日志和 `quality_report.json`，再看根目录汇总；不要只根据 2×2 视频判断 NPZ 合格。
