# GitHub + UCloud US3：human-only 可复现交付

更新时间：2026-08-06

本项目把“小而可审计的源码”和“大而受许可约束的运行时物料”分开发布：源码、配置、文档和启动脚本在 GitHub；Docker 镜像、两个 Conda 环境的固化结果、PHC 运行时、模型资产和示例视频在 UCloud US3。接收方不需要复制服务器的 Conda 环境，也不应从项目目录手工拼装依赖。

运行脚本会把 GitHub checkout 中的 `GVHMR-hand/GVHMR-main/hmr4d/`、`tools/pipeline/` 和容器 preflight 脚本以只读方式覆盖挂载到镜像同一路径；这保证 GVHMR Python 源码、平滑/地面修正脚本与发布 commit 完全一致，同时复用镜像内已验证的 CUDA、Conda、PHC 和 Isaac Gym runtime。脚本也为 conda-pack 中失效的 editable `chumpy` 路径补充镜像内只读源码路径，并把模型包的 `assets/` 目录只读挂载为 GMR 的 `GMR_BODY_MODEL_PATH`，避免旧镜像内的兼容链接覆盖 GVHMR 的 SMPL-X 挂载。不要只下载 image 后直接执行，而要通过仓库的 `release/run_human_only_docker.sh` 启动。

这一说明对应 human-only 全链路：`视频 -> GVHMR/Hand4Whole++ -> Locomotion -> PHC -> GMR/Sharpa -> 质量/资产`。不包含场景、物体、S3 视频采集或对象重建功能。

## 1. 发布边界和信任关系

| 位置 | 包含 | 不包含 | 目的 |
|---|---|---|---|
| GitHub `wentian810/locomanipulation` 的 `human-pipeline-support-contacts` 分支（发布时附同名 tag） | 所有公开脚本、YAML、Dockerfile、交接文档、恢复脚本 | 权重、人体模型、原视频、UCloud 凭证 | 可审计的源代码入口 |
| `us3://robotic-docker-images/locomotion/` | Docker 镜像归档、模型资产归档、示例数据、`SHA256SUMS` | AccessKey、SecretKey、日志、产出目录 | 可下载的运行时依赖 |
| 接收方机器 | Docker、NVIDIA 驱动、有效 US3 只读配置 | 服务器 SSH 权限 | 独立复现环境 |

UCloud 控制台账号密码不能直接用于 US3 CLI。接收方需要由发布方创建最小权限的 US3 **只读**公私钥，或获得等价的、带到期时间的下载链接。公私钥只能保存在接收方本地 US3CLI profile，绝不能提交 Git、写到 YAML、通过 Docker `-e` 传入或放进镜像。

## 2. 固定物料

Release `20260806` 的对象名如下。以远端 `SHA256SUMS` 为唯一完整性依据；文件大小只是上传恢复时的辅助判断。

| 对象 | 约定解压位置 | 用途 |
|---|---|---|
| `human-pipeline-support-contacts_20260806.tar.zst` | 可选，历史源码审计快照 | 仅供归档核验；不能替代 GitHub checkout，也不被 release runner 使用 |
| `locomotion-human-only_20260806.docker.tar.zst` | Docker image store | 固化的 CUDA 11.8、`locomotion` / `phc` Conda 环境、PHC/Isaac Gym 和运行时代码；tag 为 `locomotion-human-only:20260806` |
| `human-only-model-assets_20260806.tar.zst` | `<repo>/.release/20260806/extracted/model-assets` | GVHMR、HMR2、ViTPose、Hand4Whole++、WiLoR、SMPL/SMPL-X、Locomotion 的 SMPL-H/ACCAD 资产 |
| `dataset_new6_20260806.tar.zst` | `<repo>/.release/20260806/extracted/dataset/dataset_new6` | 七条示例原视频；用于自检和回归，不是生产输入的唯一来源 |
| `SHA256SUMS` | `<repo>/.release/20260806` | 上述归档的 SHA-256 清单 |

模型包不自动解压到 Git checkout，因此不会把受限权重或第三方树误加入 Git。Docker 运行脚本会按只读 volume 挂载这些路径。

模型包内部同时有 `GVHMR-hand/GVHMR-main/inputs/checkpoints/`（GVHMR/HMR2/ViTPose 与 SMPL-X `.npz`）和 `GVHMR-main/inputs/checkpoints/body_models/`（供 GMR 使用的 SMPL-X `.pkl`）。这是刻意保留的上游格式差异；请使用 release runner，不要把两个目录手工合并。

`models/yolo11n-pose.pt` 也随模型包发布，用于输入全身准入 gate；release runner 会把它只读挂载到镜像内配置所引用的 `/workspace/locomotion/models/`。因此真正的推理与 `--check-only` 的配置检查使用同一份默认 YAML 路径。

## 3. 接收方前置条件

1. Linux x86_64 主机，NVIDIA 驱动能通过 `nvidia-smi` 访问 GPU。
2. Docker Engine 已安装，并已配置 NVIDIA Container Toolkit；先确认 `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi` 成功。
3. `zstd`、`sha256sum`、`tar`、Bash 和 Git 可用。
4. US3CLI 已安装，并由发布方提供只读的 bucket/prefix 授权。UCloud endpoint、region、公钥/私钥都写在接收方本地 profile 中。
5. 至少预留约 90 GB 空闲磁盘：约 17 GB 压缩 Docker 镜像、约 17 GB 压缩模型包、两者解压/导入后的空间，以及输出和示例数据。实际容量取决于 Docker layer 去重与日志数量。

不要在 macOS/Windows 原生 Docker Desktop 上宣称该发布已通过 PHC/Isaac Gym 验证；推荐在原生 Linux 或 Linux GPU 节点执行。

接收方应优先加入 Docker group 并重新登录，使 `docker info` 无需 sudo。若运维策略要求 `sudo -n docker`，两个 release 脚本都可显式传 `--docker-sudo`；它只调用非交互式 sudo，未配置权限时会立即失败而不会索要密码。

## 4. 从空目录恢复的标准命令

```bash
git clone --branch human-pipeline-support-contacts \
  https://github.com/wentian810/locomanipulation.git locomotion-human-only
cd locomotion-human-only

# 只在本机创建 US3 read-only profile；命令会交互式索取 endpoint/region/公私钥。
us3cli-linux64 config --config locomotion-download

# 下载 -> SHA-256 验证 -> 解压模型/示例视频 -> docker load。
bash release/bootstrap_from_ucloud.sh --us3-config locomotion-download

# 不跑推理的完整门禁：GPU、两个 Conda 环境、PHC/Isaac、模型挂载、YAML 与单元测试。
bash release/run_human_only_docker.sh --check-only

# 运行一个示例 clip；输出不写入镜像或模型包。
bash release/run_human_only_docker.sh --clip-filter chairwood
```

如果该主机只能通过 `sudo -n docker` 访问 Docker daemon，在上述两个 Docker 命令后均加入 `--docker-sudo`。

`bootstrap_from_ucloud.sh` 支持重复执行：已经通过哈希验证的归档会复用。只重新验证本地文件时用 `--verify-only`；只下载不解压/导入镜像时用 `--download-only`；`--with-source-archive` 只额外校验历史归档，不参与运行。完整参数见 `bash release/bootstrap_from_ucloud.sh --help`。

服务器发布自检不得再次从 S3 回下载。应直接复用打包目录并只把解压物写进 Git checkout 的 `.release/`：

```bash
bash release/bootstrap_from_ucloud.sh \
  --artifact-dir /home/jixingyu/locomotion_human_only_delivery_20260806
```

`--artifact-dir` 会原地读取 `SHA256SUMS` 和归档、验证哈希、解压到测试位置并加载 Docker image；不会网络下载、复制或修改原始交付归档。这样 S3 上传和服务器验收相互独立。

生产视频不需要重新打镜像，只要将 `run_human_only_docker.sh` 中的 `/data/input` 映射改为自己的单人视频目录，并保持模型挂载为只读。当前脚本默认使用 S3 包内的回归数据，以便交接验收有确定输入。

## 5. “跑通”的可验证定义

发布完成不以“Docker build 成功”作为标准，而以如下顺序验收：

1. GitHub checkout 的 `git status --short` 为空，且 `git rev-parse HEAD` 已记录在交接单。
2. bootstrap 对每个选中的归档给出 `verified`；任何 hash mismatch 都必须删除该本地归档后重新下载。
3. `--check-only` 返回 `docker-preflight PASS`，其中包括 `locomotion` 与 `phc` 两套 Python import、Isaac Gym、PHC 权重、模型挂载、配置检查和单元测试。
4. 对示例视频至少跑一条 `--clip-filter`，确认生成 `001_final.npz`、`robot_motion.pkl`、`001_sharpa_chain_hands.npz` 和审核视频。
5. 最后运行 `--stage quality` 与 `--stage product`，按 [输出契约与验收](human_reconstruction/09_output_contract_and_acceptance.md) 检查 `manifest.json`、`checksums.sha256`、帧数、有限数值和人工预览。

只有步骤 1--5 全部通过，才称该接收方机器“可复现运行”。第 1--3 是安装门禁，第 4--5 才是端到端功能门禁。

## 6. 发布者的上传前门禁

发布者应在服务器上执行以下不包含密钥的检查，再创建 tag / push：

```bash
git diff --check
git status --short
docker run --rm --gpus all ... locomotion-human-only:20260806 check
bash release/bootstrap_from_ucloud.sh --verify-only
```

上传脚本 `docker/upload_delivery_to_ucloud.sh` 仅从本地 `us3cli` profile 读取凭证，并在每个对象上传后比对远端 `Content-Length`。它可安全重复运行，已存在且同大小的对象会跳过。对象全部可见并与 `SHA256SUMS` 一致后，才允许对外宣布 release 可下载。

## 7. 常见问题和界限

| 现象 | 处理 |
|---|---|
| `us3cli` 报认证或 endpoint 错误 | 向发布方索要当前 region 对应的只读 US3 公私钥和 endpoint；账号密码本身无效。不要把 key 发到工单或 GitHub Issue。 |
| `docker: could not select device driver` 或无 GPU | 安装/修复 NVIDIA Container Toolkit，先通过前置条件中的 CUDA `nvidia-smi` 容器测试。 |
| 模型挂载缺失 | 重新执行 bootstrap 的模型部分；不要把模型复制进仓库，也不要用空目录替代。 |
| `--check-only` 通过而推理失败 | 这是模型/输入或算法级问题。保留有效 YAML、`fullbody_preflight`、clip 名称、容器 tag 和日志，再按模块文档排障。 |
| 只想运行人体上游 | 在容器内显式传 `--stage human`；完整交付默认 `all`，不要误把上游调试结果当作 PHC/GMR 成品。 |

发布版本的脚本、环境边界和每个 CLI 的职责见 [CLI 参考](CLI_REFERENCE.md)；人体链路的逐模块输入输出见 [人体重建交接总览](HUMAN_RECONSTRUCTION_HANDOVER.md)。
