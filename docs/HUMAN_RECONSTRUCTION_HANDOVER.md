# 人体重建交接总览

更新时间：2026-08-06

这组文档交接当前项目的 human-only 运动链：原始单目视频、人体三维运动、PHC 物理修复、GMR/Sharpa 机器人重定向、质量检查和安全资产导出。这里的 human-only 意味着不进行物体或场景重建，不需要分割、Hunyuan3D、MoGe、MegaPose、FoundationPose 或云端密钥；它不意味着跳过 PHC 和 GMR。

## 1. 范围与最终产物

~~~text
原始视频
  └─ 输入规范化 + 全身准入
       └─ GVHMR 身体轨迹
            └─ Hand4Whole++ 双手 MANO
                 └─ 手部时序修复 / 可见手细化
                      └─ SMPL-H/SMPL-X 转换
                           └─ Locomotion 身高优化 + 时序平滑
                                └─ PHC 物理修复与落地
                                     └─ GMR 身体重定向 + Sharpa 手部 IK
                                          └─ 视频、质量和资产交付
~~~

human-only 交付的工作区与资产包分别保留。<out> 是 output.root，<asset> 是 product.root，<clip> 是不含扩展名的视频名。

~~~text
<out>/<clip>/
  001_smoothed.npz          # Locomotion 后的身体轨迹
  001_final.npz             # 本次运行选出的身体轨迹；PHC 成功则为 PHC-grounded
  001_smplx_hands.npz       # 同帧 MANO/SMPL-X 手部 sidecar
  gvhmr_camera.npz          # 与人体世界坐标对齐的相机轨迹
  robot_motion.pkl          # 仅可信工作区使用的 GMR 结果
  001_sharpa_chain_hands.npz # Sharpa 双手 22-DoF 轨迹
  phc_renderings/           # PHC/Isaac 审核视频
  videos/                   # 稳定名称的 gvhmr/phc/gmr/2x2 视频别名
  gvhmr_out/<clip>/
    hmr4d_results.pt         # GVHMR 原始结果，供排障/重转码使用
    1_incam.mp4              # 人体重建审核视频（文件名可能因版本略有不同）
  hamer_diagnostics/         # 手部诊断视频与 JSON；启用 diagnostics 时生成

<asset>/<clip>/
  human_motion.npz          # 对外安全人体 + MANO 运动接口
  human_phc_motion.npz      # 可选：PHC 修复的身体轨迹
  robot_motion.npz          # 对外安全 GMR 机器人运动接口
  robot_hand_motion.npz     # Sharpa 22-DoF 手部接口
  camera.npz                # 相机与坐标元数据
  quality_report.json       # pass/warn/fail 和指标细节
  preview_2x2.mp4           # 审核预览
  manifest.json
  pipeline_config.yaml
  rights.json
  checksums.sha256
~~~

001_converted.npz 是未经过高度优化与平滑的中间身体轨迹。001_final.npz 是本次运行的选择点：PHC 成功时选择 PHC-grounded 结果；PHC 被关闭或本次失败时回退到 001_smoothed.npz，并通过 stage_status.json 或 final 选择记录说明原因。

## 2. 面向对接方的最快启动方式

当前实际 human-only 全流程配置是 [human_sharpa.yaml](../configs/pipelines/human_sharpa.yaml)：它关闭 object，但保留 PHC、GMR/Sharpa、质量和产品导出。对接交付应以它为默认入口。

~~~bash
cd /path/to/Loco-manipulation-human-pipeline-support-contacts

# 仅验证依赖、配置、输入目录和全身准入模型是否齐全；不推理
/opt/conda/envs/locomotion/bin/python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage all \
  --dataset-dir /data/raw_videos \
  --output-root /data/human_output \
  --set input.work_video.directory=/data/human_work \
  --check

# 单条视频。clip_filter 是文件名的子串；同批可用逗号分隔多个子串。
/opt/conda/envs/locomotion/bin/python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage all \
  --dataset-dir /data/raw_videos \
  --output-root /data/human_output \
  --set input.work_video.directory=/data/human_work \
  --clip-filter chairwood
~~~

不要直接调用 run_batch_dataset6.sh；它是内部后端，缺少配置入口冻结的环境变量时会拒绝运行或得到不可复现的结果。

Docker 镜像的 run/human 入口默认使用 human_sharpa.yaml 并执行 --stage all，与该交接主线一致。完整、可直接执行的模型挂载命令在 [07_docker_and_delivery.md](modules/07_docker_and_delivery.md) 第 5 节；在其镜像命令末尾传入 --dataset-dir、--output-root、--set input.work_video.directory 和可选 --clip-filter 即可。

若需要只验证原视频到 SMPL-H/MANO 的前半段、故意不启动 PHC/GMR，才使用 [human_reconstruction.yaml](../configs/pipelines/human_reconstruction.yaml) 加 --stage human。它是上游调试配置，不是 human-only 完整交付配置。

## 3. 配置优先级与运行边界

配置优先级为：

~~~text
CLI 参数与 --set  > 进程环境变量 > pipeline YAML > configs/default.yaml
~~~

对接时优先使用 --dataset-dir、--output-root、--clip-filter 和少量 --set 覆盖；不要复制并手改底层 Bash 命令。--print-config 会打印合并后的有效配置，可用于将一次交付的实际参数写入工单。

每个原视频必须只含一个主要人物。工程在推理前使用 YOLO-Pose 抽样检查头、双腕、双踝、人物尺度和竞争人物；不满足条件的 clip 会写入报告并在 mode: gate 下跳过，属于输入不合格而非模型运行失败。

## 4. 模块文档索引

| 顺序 | 模块 | 输入 | 责任输出 | 详细说明 |
|---|---|---|---|---|
| 01 | 输入与准入 | 原视频 | 工作视频、准入报告 | [01_input_and_admission.md](human_reconstruction/01_input_and_admission.md) |
| 02 | GVHMR 身体 | 工作视频、人体/姿态模型 | hmr4d_results.pt、GVHMR 预览 | [02_gvhmr_body.md](human_reconstruction/02_gvhmr_body.md) |
| 03 | Hand4Whole++ 手部 | GVHMR 人体轨迹、全身关键点、手部模型 | 最终 MANO 参数、手部诊断 | [03_hand4wholepp_mano.md](human_reconstruction/03_hand4wholepp_mano.md) |
| 04 | 转换与相机 | GVHMR 结果、最终 MANO | 001_converted.npz、手部 sidecar、相机 | [04_conversion_and_camera.md](human_reconstruction/04_conversion_and_camera.md) |
| 05 | Locomotion 与预 PHC 平滑 | 转换后的身体 NPZ | 001_smoothed.npz | [05_locomotion_and_release.md](human_reconstruction/05_locomotion_and_release.md) |
| 06 | PHC 物理修复 | smoothed 身体轨迹、PHC 权重 | 001_final.npz、Isaac 视频 | [06_phc_repair.md](human_reconstruction/06_phc_repair.md) |
| 07 | GMR 与 Sharpa | final 身体、手部 sidecar、相机 | 机器人/22-DoF 手部、GMR 视频 | [07_gmr_and_sharpa.md](human_reconstruction/07_gmr_and_sharpa.md) |
| 08 | 质量与产品导出 | 完整可信工作区 | 报告、压缩 NPZ、manifest、SHA-256 | [08_quality_and_product.md](human_reconstruction/08_quality_and_product.md) |
| 09 | 输出契约与验收 | human-only 资产包 | 接收判定、问题单信息 | [09_output_contract_and_acceptance.md](human_reconstruction/09_output_contract_and_acceptance.md) |

## 5. 环境和模型所有权

人体部分依赖两个 Conda 环境：

| 环境 | 主要责任 | 不应由对接方自行替换的原因 |
|---|---|---|
| locomotion | 配置入口、YOLO-Pose 准入、GVHMR 调度、转换、Locomotion、GMR/Sharpa、质量和导出 | PyTorch、CUDA、SMPL-H/SMPL-X 与现有脚本版本已配套验证 |
| phc | Isaac Gym、PHC 策略与物理修复、PHC 后平滑 | Python 3.8、Isaac Gym 和策略依赖必须与服务器已验证版本一致 |

运行时还需要合法取得并挂载 GVHMR/HMR2/ViTPose、SMPL/SMPL-X/MANO、Hand4Whole++ checkpoint 与 Locomotion 的 SMPL-H 资产。模型目录、Docker 挂载和镜像自检请看 [07_docker_and_delivery.md](modules/07_docker_and_delivery.md)。模型权重、原视频和密钥都不应提交到 Git，也不应写进 Dockerfile 或 YAML。

## 6. 交付前后分别做什么

发送方按顺序完成：

1. 运行 --check，保存有效配置和控制台日志。
2. 对每条视频先看 fullbody_preflight.csv，确认被准入。
3. 完成推理后按照第 09 节检查人体、手、相机、机器人和预览的帧数/FPS/有限数值。
4. 审看 1_incam.mp4、PHC/Isaac、GMR 与 2×2 视频；重点检查腕部翻转、手指突跳、物理落地和机器人 IK。
5. 运行 --stage quality 与 --stage product，交付资产包、原始输入文件名/哈希、有效 YAML、软件版本和验收结果。

接收方不应从视频渲染效果单独判断数据正确性。应以资产包内的 human_motion.npz、robot_motion.npz、robot_hand_motion.npz 和 camera.npz 的帧对齐、坐标约定和有效性字段为机器接口，以 preview_2x2.mp4 作为人工审核证据。
