# CLI 参考：human-only 主线、辅助管线与保留的对象工具

更新时间：2026-08-06

本页描述仓库中可见命令的责任边界。它的目的不是鼓励逐个脚本手工拼接，而是让对接方知道某一项输出由谁产生、在什么 Conda 环境启动，以及何时只能作为工程内部实现使用。

正式 human-only 对外交付只建议从 `scripts/run_pipeline_from_config.py` 启动。它串起 GVHMR、Hand4Whole++、Locomotion、PHC、GMR/Sharpa、质量和产品导出；不启用 object 分支。其余命令用于准入、诊断、修复、重跑某一模块或维护旧的物体管线。

## 1. 共同约定

| 术语 | 含义 |
|---|---|
| `locomotion` 环境 | 配置入口、全身准入、GVHMR 调度、格式转换、Locomotion、GMR/Sharpa、质量/导出。Docker 内是 `/opt/conda/envs/locomotion/bin/python`。 |
| `phc` 环境 | Isaac Gym、PHC 策略、物理修复以及与 `smpl_sim` 有关的后处理。Docker 内是 `/opt/conda/envs/phc/bin/python`。 |
| `human_sharpa.yaml` | 默认完整交付配置：PHC + GMR + Sharpa 开启，object 关闭。 |
| `human_reconstruction.yaml` | 只到人体/手/相机/Locomotion 的上游调试配置；不能替代完整交付。 |
| `<clip>` | 原视频不含扩展名的 slug；每个 clip 在 `output.root/<clip>/` 下独立保存中间物。 |

所有参数以实际 `--help` 为准。配置覆盖遵循 `CLI 参数/--set > 环境变量 > pipeline YAML > configs/default.yaml`；一份交付必须保存最终 YAML 或 `--print-config` 输出。

## 2. 对接方的主入口

### `scripts/run_pipeline_from_config.py`

**职责**：读取 pipeline YAML，验证路径与依赖，并按 `human/object/quality/product/all` 选择阶段。human-only 的生产运行使用 `--stage all`，内部会选择 `hand4wholepp` 与 `sharpa` 后端；`object.enabled: false` 使场景/物体部分不启动。

**启动环境**：`locomotion`。Docker 用 `release/run_human_only_docker.sh`，不要把宿主机 Conda 路径写入新脚本。

**关键输入**：`--config_dir`、视频目录（`--dataset-dir`）、输出根目录（`--output-root`），可选单/多 clip 筛选 `--clip-filter`。工作视频目录通常通过 `--set input.work_video.directory=...` 覆盖。

**关键输出**：每条 clip 的 `gvhmr_out/`、`001_converted.npz`、`001_smoothed.npz`、`001_final.npz`、手部/相机 sidecar、PHC/GMR 视频、`robot_motion.pkl`、Sharpa 22-DoF 手轨迹和产品包。阶段状态与回退理由写入各 clip 工作目录。

**最小示例**：

```bash
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage all --dataset-dir /data/raw_videos --output-root /data/human_output \
  --set input.work_video.directory=/data/human_work --clip-filter chairwood
```

**常用控制项**：`--check` 只做环境/配置/输入检查；`--dry-run` 展示计划；`--print-config` 输出合并后的配置；`--enable-phc` / `--skip-phc` 临时覆盖物理阶段；`--human-backend {hamer,hand4wholepp,wilor}`、`--gmr-hand-model {sharpa,g1,brainco}` 用于专家调试。对外交付不要随意改变后两项，否则输出手部定义会改变。

### `scripts/preflight_fullbody_gate.py`

**职责**：在昂贵推理前对原视频抽帧跑 YOLO-Pose，检查主人体尺度、头、双腕、双踝、竞争人物与置信度。它不产生三维人体，也不修复视频。

**启动环境**：`locomotion`。

**输入/输出**：`--video` 或 `--video-list`；必须给 `--report`（JSON）和 `--csv`（逐视频结果）以及 `--model`（YOLO pose 权重）。可用 `--samples`、`--imgsz` 和阈值调整严格度，`--no-cache` 强制重新计算。

**何时调用**：主入口的 `input.fullbody_preflight` 已经集成它。只有要单独评估一批候选视频或排查 gate 拒绝原因时才直接调用。

### `scripts/export_dataset_product.py`

**职责**：将已经可信的工作区压缩为对外资产包；不重新执行 GVHMR、PHC 或 GMR。

**启动环境**：`locomotion`。

**输入/输出**：读取 pipeline 配置、输出工作区和可选 `--clip-filter`，写出 `human_motion.npz`、可选 `human_phc_motion.npz`、`robot_motion.npz`、`robot_hand_motion.npz`、`camera.npz`、质量报告、预览、`manifest.json`、权利声明和 `checksums.sha256`。

**安全边界**：若缺少可信的 `robot_motion.pkl`、人体/手部 sidecar 或质量结论，产品导出应失败，而不是制造看似完整的包。可先加 `--dry-run` 确认会处理哪些 clip。

### `scripts/evaluate_clip_quality.py`

**职责**：对已有中间/最终结果计算时序、有限数值、帧数对齐等质量指标，产出供 `quality` 阶段和产品导出使用的报告。

**启动环境**：`locomotion`。

**何时直接调用**：复查单个 clip、修改阈值后的回归或不想重跑上游时。完整交付优先经 `run_pipeline_from_config.py --stage quality` 调用，避免报告位置和配置与产品阶段脱节。

### `scripts/collect_s3_videos.py`

**职责**：整理已经下载到本地的展示/重定向视频为数据集目录，统一文件命名/slug，并可在 `hardlink`、`copy`、`symlink` 之间选择连接方式。

**启动环境**：普通 Python 或 `locomotion`。

**输入/输出**：`--show-root`、`--gmr-root`、`--output-root`；用 `--link-mode` 和 `--max-slug-len` 控制行为，可加 `--dry-run`。它不访问 UCloud、不会下载对象，也不会重建人体；名称里的 S3 是历史来源命名。

## 3. GVHMR、手部与人体格式工具

这些脚本位于 `GVHMR-hand/GVHMR-main/tools/pipeline/`。完整流程由主入口调用，直接运行仅用于定位某一环失败的位置。

| CLI / 脚本 | 启动环境 | 输入 -> 输出 | 使用边界 |
|---|---|---|---|
| `run_pipeline.sh` | `locomotion` | 单条原视频及环境变量 -> GVHMR 原始结果、渲染、手部与转换中间物 | 内部单 clip 后端；由 batch wrapper 设置路径和模型变量。不要把它当公开稳定 API。 |
| `run_batch_dataset6_hand4wholepp.sh` | `locomotion` | 工作视频目录 -> Hand4Whole++ MANO、人体转换结果 | human-only 主线实际选择的后端 wrapper。必须由 YAML 主入口提供冻结的 `PIPELINE_*` 路径/参数。 |
| `run_batch_dataset6_hamer.sh` | `locomotion` | 同上 -> HaMeR 手部结果 | 备选旧后端，手参数品质和依赖不同，不与正式 Hand4Whole++ 成品混用。 |
| `run_batch_dataset6_wilor.sh` | `locomotion` | 同上 -> WiLoR 手部结果 | 备选研究后端；需完整 WiLoR 外部树和权重挂载。 |
| `run_batch_dataset6.sh` | `locomotion` | 数据集批量调度 -> 后端中间物 | 基础内部 wrapper，不保证独立配置充分；对接方不要直接调用。 |
| `configure_hand_constraints.sh` | Bash，`source` 到当前 shell | 设置手部可靠性、时序和约束环境变量 | 不是独立命令；必须用 `source`，由 wrapper 负责加载。 |

### `convert_to_npz.py`

**职责**：把 GVHMR 结果和最终手部参数转为下游统一 NPZ。主要产物是 `001_converted.npz`；它保留人体 root、姿态、形状、FPS/坐标约定，以及与身体逐帧对齐的手部信息所需字段。

**启动环境**：`locomotion`，需要 SMPL/SMPL-H/MANO 与 GVHMR 输出。输入与输出路径由 wrapper 传入；不要对不同帧率的 body 和 mano 文件强行配对。

**下游**：Locomotion 读取身体轨迹，手部 sidecar 继续传给 Sharpa；转换不是最终落地或机器人结果。

### `export_gvhmr_camera.py`

**职责**：从 GVHMR 原始结果导出与人体世界坐标对齐的相机 NPZ，供机器人视频合成、产品包和对接方坐标解释使用。

**启动环境**：`locomotion`。输入是 `hmr4d_results.pt`（及 clip 上下文），输出是 `gvhmr_camera.npz`。

**验收点**：相机帧数必须与最终选中身体轨迹相同；此脚本不推断新的相机，也不能修复上游相机漂移。

### `diagnose_hamer_hand.py` 与 `export_hand_review.py`

**职责**：二者均为诊断/审核工具，不改变正式轨迹。前者汇总 HaMeR 手部、ViTPose、MANO 的可靠性和异常；后者把原视频、渲染视频、MANO 与关键点合成为可人工查看的手部审核视频。

**启动环境**：`locomotion`。`diagnose_hamer_hand.py` 至少要有 `--output`，可给 clip 目录/视频、ViTPose、sidecar、MANO、`--summary`、`--person-idx`、抽帧控制。`export_hand_review.py` 要求原视频、渲染视频、MANO、ViTPose 和输出路径，可设置 crop、上下文帧、skip/max-frame。

**使用边界**：Hand4Whole++ 是当前正式后端，所以这两个工具主要服务旧 HaMeR 回归或问题定位；不可把诊断视频当作手部数据接口。

### `smooth_motion.py`

**职责**：对转换后的身体运动执行时间平滑，控制 root/关节突跳，写出平滑后的 NPZ。它服务于 Locomotion/PHC 前的稳定输入，正式输出名约定为 `001_smoothed.npz`。

**启动环境**：`locomotion`。必须保留输入 FPS 和逐帧语义；不允许用平滑改变 clip 长度却不更新相机/手部对齐。

### `fix_motion_floor.py`

**职责**：根据 PHC/SMPLSim 语义修正最终人体轨迹的地面高度或脚底接触参考。它只处理 PHC 后或与 PHC 相关的轨迹，不替代 Locomotion 高度优化。

**启动环境**：`phc`，因为它依赖 `smpl_sim`；在 `locomotion` 环境直接启动会缺依赖。输入是 PHC 选中的身体 NPZ，输出纳入 `001_final.npz`/PHC 结果链。对接方应通过主入口的 PHC 阶段调用。

## 4. Locomotion 与 PHC 的阶段接口

Locomotion 本体在 `locomotion_pipeline-main/`，PHC 在 `phc-dev-felix-pipeline/`。当前 release 不把它们暴露成额外的“任意输入单文件 CLI”，而是把输入输出契约冻结在主 YAML：

1. `001_converted.npz` 进入 Locomotion，进行人体尺度/接触/时序处理。
2. `smooth_motion.py` 写 `001_smoothed.npz`，是 PHC 的候选输入。
3. PHC 使用 Isaac Gym 与策略权重修复动态可行性和落地，成功时产生物理可信候选。
4. `fix_motion_floor.py` 在 `phc` 环境执行最终地面修正，主入口把成功结果选为 `001_final.npz`；失败或显式禁用时，必须记录回退至 `001_smoothed.npz`。

PHC 的训练、策略实验和 Isaac Gym 原生工具不是此交付的公开 CLI。接收方需要的只是 Docker image 中已固化的 PHC runtime、策略权重和 `--stage all` 调度。

## 5. GMR、Sharpa 与机器人可视化工具

这些工具位于 `GMR-master/`，均在 `locomotion` 环境中运行，且依赖 MuJoCo、机器人资产、SMPL-X/SMPL-H/ACCAD。正式顺序是“身体重定向 -> 手部 IK -> 渲染/合成”，由 `run_show_gmr_batch.sh` 统一调度。

| CLI / 脚本 | 输入 -> 输出 | 何时使用 |
|---|---|---|
| `run_show_gmr_batch.sh` | `001_final.npz`、相机/手 sidecar -> `robot_motion.pkl`、GMR 视频及后续调用 | 正式内部 GMR wrapper；由主 YAML 调用。 |
| `run_sharpa_hand_batch.sh` | 可信身体/手部工作区 -> `001_sharpa_chain_hands.npz` | 正式 Sharpa 双手 batch wrapper。 |
| `run_brainco_hand_batch.sh` | 同类输入 -> BrainCo 手轨迹 | 备选硬件分支，不是当前 Sharpa 交付格式。 |
| `configure_hand_model.sh` | 设定手模型/资产环境变量 | 必须 `source`；不作为独立稳定 CLI。 |
| `pipeline_defaults.sh` | 统一 GMR 路径、帧率和默认参数 | 供 wrapper source；不要直接运行。 |

### `scripts/smpl_npz_to_robot_headless.py`

**职责**：无界面 MuJoCo 人体到机器人重定向的底层命令。读取源身体 NPZ，使用目标机器人、SMPL body model、源坐标空间、fps、优化/接触/平滑参数，写出机器人运动（通常由 wrapper 规范化为 `robot_motion.pkl`）。

**直接调用适用场景**：研究调参、隔离 GMR 问题、重跑已确认的 `001_final.npz`。必须明确 `--src_root`、`--tgt_root`、`--pattern`、机器人及 body-model 路径；不要把未经 PHC 选择的轨迹作为正式输入。

### `scripts/sharpa_hand_retarget.py`

**职责**：将双手 MANO/sidecar 重定向成 Sharpa 22-DoF 手关节轨迹，结合腕部与身体链路做 IK、可靠性权重、时间连续性、关节角限制和可选平滑。

**输入/输出**：`--hand_npz` -> 输出 NPZ；需要 Sharpa 根目录、左右手信息和 solver/步长/成本/最大帧等设置。正式输出 `001_sharpa_chain_hands.npz` 必须与 `robot_motion.pkl` 同帧对齐。

### `scripts/brainco_hand_retarget.py`

**职责**：输出 BrainCo 兼容手部重定向。它与 Sharpa 的关节定义、资产和 IK 约束不同，不能用文件改名的方法替代 Sharpa 结果。

### `scripts/render_robot_motion_headless.py`

**职责**：无显示服务器渲染 GMR 机器人动作视频。要求机器人类型、运动文件、输出视频；支持 XML、EGL/GL、分辨率/FPS、相机和不同手部模型设置。

**用途**：CI/服务器审看和单模块可视化；渲染成功不代表运动动力学或手部契约已验收。

### `scripts/render_composite_2x2.py`

**职责**：合成原视频、GVHMR、GMR 和可选 Isaac/手部诊断为四宫格审核视频。核心必需输入是原视频、GVHMR 视频、GMR 视频和输出；可设置布局尺寸、FPS、标签和 CRF。

**用途**：人工审核证据。数值接口仍以各 NPZ、`robot_motion.pkl` 和质量报告为准。

## 6. 物体/场景工具：保留但不随 human-only 调用

下列 `GMR-master/scripts/` 命令服务历史对象适配或接触仿真，不属于当前 human-only 镜像的默认路径。文档保留其职责以避免对接方误用。

| CLI | 输入 -> 输出 | 当前状态 |
|---|---|---|
| `build_object_proxy_from_robot_motion.py` | 机器人运动 -> 对象代理 | 物体管线专用；human-only 不执行。 |
| `build_object_proxy_from_smpl_motion.py` | SMPL 运动 -> 对象代理 | 物体管线专用；human-only 不执行。 |
| `validate_object_adapter.py` | 对象适配输入 -> 验证报告 | 物体接口校验；human-only 不执行。 |
| `simulate_robot_object_contacts.py` | 机器人+对象代理 -> 接触仿真/报告 | 物理对象阶段；human-only 不执行。 |
| `setup_robot_hand_assets.sh` | 下载/布置机器人手部资产 | 仅需维护/开发；release 中应从受控资产包挂载。 |

`GMR-master/scripts_legacy/` 下的命令为历史实验/原项目兼容脚本：没有稳定的交接输入输出契约，也不纳入 Docker preflight 或 human-only 发布验收。若必须启用，应单独冻结 commit、模型权利、配置和验证用例，不能在本 release 上直接承诺可复现。

## 7. 运行选择速查

| 目标 | 推荐命令 | 不要做什么 |
|---|---|---|
| 完整交付 | `run_pipeline_from_config.py --config_dir human_sharpa.yaml --stage all` 或 Docker runner | 手工按目录执行多个 wrapper。 |
| 只查配置/模型/路径 | 主入口 `--check` 或 Docker `--check-only` | 用一次真实推理代替环境验证。 |
| 只重建上游人体 | 主入口 `--stage human` + `human_reconstruction.yaml` | 将该结果直接标为机器人/PHC 成品。 |
| 仅补质量/产品 | `--stage quality` / `--stage product` | 在缺失人体、机器人或手部数据时强行导出。 |
| 定位一只手异常 | `diagnose_hamer_hand.py`、`export_hand_review.py` | 用审核 MP4 作为下游数值数据。 |
| 隔离 GMR | `smpl_npz_to_robot_headless.py`、`render_robot_motion_headless.py` | 改变坐标/FPS 后不更新相机与手部对齐。 |
| 对象研究 | 单独的 object release | 在 `human_sharpa.yaml` 的 human-only 交接上打开 object。 |

每一模块的算法工作流、输出内容与验收条件见 [人体重建交接总览](HUMAN_RECONSTRUCTION_HANDOVER.md) 及其 `human_reconstruction/01`--`09` 子文档；接收方从 GitHub/S3 恢复环境的命令见 [GitHub + UCloud US3 交付说明](RELEASE_FROM_GITHUB_AND_S3.md)。
