# 07. Docker、权重挂载与交付

## 1. 镜像边界

根目录 `Dockerfile` 构建人体-only runtime image。

镜像内：Ubuntu 22.04 + CUDA 11.8 libraries、`locomotion` Python 3.10、`phc` Python 3.8、当前分支代码、GMR 依赖和 Sharpa XML/mesh、PHC/Isaac Gym runtime 及 PHC policy weights、预检脚本。

镜像外：GVHMR/HMR2/ViTPose/SMPL-X/MANO/Hand4Whole++/WiLoR 大模型权重、原始视频、工作目录、最终资产、密钥。

模型许可和体积是分离权重的原因；不是 Docker 无法打包模型。

## 2. 构建

```bash
cd /home/jixingyu/Loco-manipulation-human-pipeline-support-contacts
docker/prepare_phc_context.sh
docker/prepare_conda_packs.sh
sudo -n docker buildx build --load --progress=plain \
  --build-context phc=/home/jixingyu/.codex_phc_context_20260806 \
  --build-context envs=/home/jixingyu/.codex_conda_packs_20260806 \
  -t locomotion-human-only:20260806 .
```

`prepare_phc_context.sh` 只用硬链接构造一个约 3GB 的最小上下文，不会修改源工作树；`prepare_conda_packs.sh` 只读取服务器上已经验证的 `locomotion` 和 `phc` 环境并生成归档。两个 `--build-context` 都必须存在，因为当前分支只提交编排层，PHC 代码/运行权重和两个 Conda 环境归档在工作树之外。Dockerfile 不会读取旧工作树中的场景重建工作区。

## 3. 权重挂载

| 宿主机 | 容器 |
|---|---|
| GVHMR `inputs/checkpoints` | `/models/gvhmr/checkpoints` |
| GMR/SMPL-X body models (`SMPLX_NEUTRAL.pkl`) | `/models/gvhmr/body_models` |
| Hand4Whole++ 完整目录 | `/models/hand4whole` |
| WiLoR 完整目录 | `/models/wilor` |
| Locomotion `assets` | `/models/locomotion_assets` |
| 输入视频目录 | `/data/input` |
| 输出目录 | `/data/output` |
| 可选工作缓存 | `/data/work` |

服务器现有路径示例：

```text
/home/jixingyu/Loco-manipulation-human-only-release/GVHMR-hand/GVHMR-main/inputs/checkpoints
/home/jixingyu/Loco-manipulation-human-only-release/GVHMR-hand/GVHMR-main/third-party/Hand4Whole-plus-plus_RELEASE
/home/jixingyu/Loco-manipulation-human-only-release/GVHMR-hand/GVHMR-main/third-party/WiLoR
/home/jixingyu/Loco-manipulation-human-only-release/locomotion_pipeline-main/assets
/home/jixingyu/Loco-manipulation-human-only-release/models
```

## 4. 自检

不挂模型也可以检查镜像依赖、配置、单元测试：

```bash
sudo -n docker run --gpus all --rm -e RUN_TESTS=1 \
  locomotion-human-only:20260806 check
```

挂载服务器权重做完整门禁：

```bash
sudo -n docker run --gpus all --rm -e REQUIRE_MODELS=1 \
  -v /home/jixingyu/Loco-manipulation-human-only-release/GVHMR-hand/GVHMR-main/inputs/checkpoints:/models/gvhmr/checkpoints:ro \
  -v /home/jixingyu/Loco-manipulation-human-only-release/GVHMR-main/inputs/checkpoints/body_models:/models/gvhmr/body_models:ro \
  -v /home/jixingyu/Loco-manipulation-human-only-release/GVHMR-hand/GVHMR-main/third-party/Hand4Whole-plus-plus_RELEASE:/models/hand4whole:ro \
  -v /home/jixingyu/Loco-manipulation-human-only-release/GVHMR-hand/GVHMR-main/third-party/WiLoR:/models/wilor:ro \
  -v /home/jixingyu/Loco-manipulation-human-only-release/locomotion_pipeline-main/assets:/models/locomotion_assets:ro \
  -v /home/jixingyu/Loco-manipulation-human-only-release/models:/workspace/locomotion/models:ro \
  locomotion-human-only:20260806 check
```

预检顺序是 import → PHC/Isaac bindings → 资源存在 → YAML `--check` → pytest → shell 语法；它不默认运行完整视频。

## 5. 运行

```bash
sudo -n docker run --gpus all --rm \
  -v /home/jixingyu/Loco-manipulation-human-only-release/GVHMR-hand/GVHMR-main/inputs/checkpoints:/models/gvhmr/checkpoints:ro \
  -v /home/jixingyu/Loco-manipulation-human-only-release/GVHMR-main/inputs/checkpoints/body_models:/models/gvhmr/body_models:ro \
  -v /home/jixingyu/Loco-manipulation-human-only-release/GVHMR-hand/GVHMR-main/third-party/Hand4Whole-plus-plus_RELEASE:/models/hand4whole:ro \
  -v /home/jixingyu/Loco-manipulation-human-only-release/GVHMR-hand/GVHMR-main/third-party/WiLoR:/models/wilor:ro \
  -v /home/jixingyu/Loco-manipulation-human-only-release/locomotion_pipeline-main/assets:/models/locomotion_assets:ro \
  -v /home/jixingyu/Loco-manipulation-human-only-release/models:/workspace/locomotion/models:ro \
  -v /path/to/input:/data/input:ro -v /path/to/output:/data/output -v /path/to/work:/data/work \
  -e DATASET=/data/input -e OUTPUT_BASE=/data/output -e WORK_DATASET=/data/work \
  locomotion-human-only:20260806 run --clip-filter chairwood
```

## 6. UCloud S3 交付物

推荐上传两个对象，而不是把输入、权重和环境混成一个不可审计的大包：

```text
human-only/<version>/locomotion-human-only_<version>.docker.tar.zst
human-only/<version>/human-pipeline-support-contacts_<version>.tar.zst
human-only/<version>/human-only-model-assets_<version>.tar.zst
human-only/<version>/dataset_new6_<version>.tar.zst
```

导出镜像：

```bash
sudo -n docker save locomotion-human-only:20260806 | zstd -T0 -19 -o locomotion-human-only_20260806.docker.tar.zst
sha256sum locomotion-human-only_20260806.docker.tar.zst > locomotion-human-only_20260806.sha256
```

本次服务器上已额外准备模型权重归档和原视频归档。模型归档保留旧 release 的目录结构，解压后可以直接按本文件前面的挂载表提供 `/models`；原视频归档解压后得到 `dataset_new6/`，可直接作为 `/data/input`。模型权重包含第三方依赖的 checkpoint，上传前应确认项目内部的使用和再分发许可。

本次目标 UFile bucket 是 `robotic-docker-images`，目标前缀是 `locomotion/`。服务器已安装 UCloud 官方 `us3cli-linux64` 到 `/home/jixingyu/.local/bin/us3cli-linux64`。配置好权限后，使用 US3CLI 的分片上传：

```bash
US3=/home/jixingyu/.local/bin/us3cli-linux64
D=/home/jixingyu/locomotion_human_only_delivery_20260806
for f in "$D"/*.tar.zst; do
  name=$(basename "$f")
  "$US3" cp "$f" "us3://robotic-docker-images/locomotion/$name" --parallel 8
done
```

仓库的 `docker/upload_delivery_to_ucloud.sh` 是本次交付的可恢复上传器：它会先核对远端对象大小、跳过已完成文件，再顺序上传大文件并再次核对大小。默认使用 profile `locomotion-upload`、目标前缀 `robotic-docker-images/locomotion/`，可直接后台运行并将日志保存到交付目录。

不要把 AccessKey/SecretKey 写入仓库、Dockerfile 或命令历史；建议先用 `us3cli config` 建立本地 profile，或通过一次性受保护的运行环境提供凭证。上传前后可用 `us3cli ls us3://robotic-docker-images/locomotion` 和 `us3cli stat` 核对对象大小。

代码归档只包含 Git 跟踪内容、Dockerfile、docs、配置和测试；不要打包 `scene_work`、物体 archive、原视频、checkpoint 和 secret。UCloud S3 的 endpoint、bucket、prefix 和凭证应通过官方 CLI/SDK 的环境变量或 profile 提供，不要写进 Dockerfile、YAML、README 或镜像环境层。

当前工作区未发现已配置的 UCloud S3 CLI/凭证，因此本文件定义上传物料和审计边界，待 bucket/endpoint/凭证可用后再执行实际上传。
