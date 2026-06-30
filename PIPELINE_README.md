# 视频到机器人全流程说明

更新日期：2026-06-30

本文档描述当前实际使用的端到端流程：

```text
输入视频
  -> 工作视频与人物跟踪
  -> ViTPose WholeBody 2D 关键点
  -> GVHMR 身体估计
  -> HaMeR / Hand4Whole++ / WiLoR 手部估计
  -> MANO 时序与手指清洗
  -> SMPL-H 身体轨迹 + 独立手部 sidecar
  -> Locomotion 高度优化
  -> PHC 物理修复与 Isaac Gym 渲染
  -> GMR 身体重定向
  -> Sharpa / G1 Dex3 / BrainCo Revo2 手部重定向
  -> 机器人渲染与 2x2 对比视频
```

当前代码入口位于：

```text
GVHMR-hand/GVHMR-main/tools/pipeline/
```

旧的 `GVHMR-main/tools/pipeline/` 仍提供部分通用转换和渲染工具，但不再是当前带手部流程的入口。

工程化配置、模块职责和 `.pt/.npz/.pkl` 字段字典见
`PIPELINE_ENGINEERING.md`。配置驱动入口：

```bash
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/rgb_monocular_sharpa.yaml \
  --stage human \
  --check
```

配置优先级为：CLI > 环境变量 > 管线专属 YAML > `configs/default.yaml`。

## 1. 最常用运行命令

进入项目根目录：

```bash
cd <repo-root>
```

当前推荐的 Hand4Whole++、保守约束、H1 + Sharpa 配置：

```bash
GMR_HAND_MODEL=sharpa \
GVHMR_HAND_CONSTRAINT_PROFILE=conservative \
GVHMR_HAND4WHOLEPP_BATCH_SIZE=1 \
bash GVHMR-hand/GVHMR-main/tools/pipeline/run_batch_dataset6_hand4wholepp.sh
```

`GVHMR_HAND4WHOLEPP_BATCH_SIZE=1` 是 16GB 主存机器首次运行时最稳妥的设置。脚本默认值为 2；确认稳定后可以省略这一行。

三条独立的手部估计流程：

```bash
# HaMeR
bash GVHMR-hand/GVHMR-main/tools/pipeline/run_batch_dataset6_hamer.sh

# Hand4Whole++
bash GVHMR-hand/GVHMR-main/tools/pipeline/run_batch_dataset6_hand4wholepp.sh

# WiLoR
bash GVHMR-hand/GVHMR-main/tools/pipeline/run_batch_dataset6_wilor.sh
```

三条流程都会继续执行 Locomotion、PHC、GMR 和 2x2 渲染，不只是生成手部参数。

## 2. 两个彼此独立的选择

必须区分“从视频估计哪一种人手”与“最终映射到哪一种机器人手”。

### 2.1 人手估计后端

| 后端 | 入口脚本 | 主要特点 |
| --- | --- | --- |
| HaMeR | `run_batch_dataset6_hamer.sh` | 单手 MANO，使用多尺度手部 crop、2D 重投影选择和测试时优化 |
| Hand4Whole++ | `run_batch_dataset6_hand4wholepp.sh` | 全身人物 crop 内联合估计，内部使用 DWPose 和 WiLoR，并输出直接 MANO 关节 |
| WiLoR | `run_batch_dataset6_wilor.sh` | 独立 WiLoR MANO 估计，使用接近官方设置的 1.8/2.0/2.2 crop 候选 |

三种后端最后都转换成同一种 MANO/SMPL-H 手部 sidecar 格式，因此可以公平地接入同一套滤波、GMR 和机器人手 IK。

### 2.2 GMR 机器人手目标

用 `GMR_HAND_MODEL` 选择：

| 值 | 身体与手 | 手部输出 |
| --- | --- | --- |
| `sharpa` | Unitree H1 身体 + Sharpa Wave 五指手 | 22-DoF morphology-matched chain IK |
| `g1` | Unitree G1 身体 + 原生 Dex3-1 三指手 | G1 XML 内置手关节映射 |
| `brainco` | Unitree G1 身体 + BrainCo Revo2 五指手 | 每手 6 个主动电机，并应用官方 mimic 关系 |

例如，同一 Hand4Whole++ 后端可以分别运行到三种机器人手：

```bash
GMR_HAND_MODEL=sharpa \
bash GVHMR-hand/GVHMR-main/tools/pipeline/run_batch_dataset6_hand4wholepp.sh

GMR_HAND_MODEL=g1 \
bash GVHMR-hand/GVHMR-main/tools/pipeline/run_batch_dataset6_hand4wholepp.sh

GMR_HAND_MODEL=brainco \
bash GVHMR-hand/GVHMR-main/tools/pipeline/run_batch_dataset6_hand4wholepp.sh
```

默认输出目录包含估计后端和机器人手名称，所以这三组结果不会互相覆盖。

## 3. 输入视频与人物裁剪

批处理默认读取：

```text
dataset_new6/*.mp4
dataset_new6/*.mov
```

默认先生成工作视频：

```text
dataset_new6_work_1280/
```

主要参数：

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `DATASET` | `<root>/dataset_new6` | 输入视频目录 |
| `USE_WORK_VIDEO` | `1` | 是否生成统一尺寸工作视频 |
| `WORK_WIDTH` | `1280` | 最大工作宽度 |
| `WORK_HEIGHT` | `960` | 最大工作高度 |
| `WORK_CRF` | `18` | H.264 工作视频质量 |
| `CLIP_FILTER` | 空 | 只运行文件名包含该字符串的视频 |
| `OUTPUT_BASE` | 由后端脚本决定 | 输出根目录 |

只运行一个片段：

```bash
CLIP_FILTER=chairwood_hand \
GMR_HAND_MODEL=sharpa \
GVHMR_HAND_CONSTRAINT_PROFILE=conservative \
bash GVHMR-hand/GVHMR-main/tools/pipeline/run_batch_dataset6_hand4wholepp.sh
```

使用自定义数据和新输出目录：

```bash
DATASET=/path/to/videos \
OUTPUT_BASE="$PWD/output_dir/my_hand4wholepp_conservative_v2" \
GMR_HAND_MODEL=sharpa \
GVHMR_HAND_CONSTRAINT_PROFILE=conservative \
bash GVHMR-hand/GVHMR-main/tools/pipeline/run_batch_dataset6_hand4wholepp.sh
```

约束 profile 名称不会自动写进默认输出路径。比较 `balanced` 和 `conservative` 时，应显式设置不同的 `OUTPUT_BASE`。

### 3.1 YOLO 和 ViTPose 实际做了什么

正常流程不是直接把整张画面逐帧送进手部模型：

1. YOLO 先跟踪画面中的人物，生成每帧人物 bbox。
2. bbox 被转换成平滑的人物 crop，并保存到 `preprocess/bbx.pt`。
3. ViTPose WholeBody 在人物 crop 上以流式 batch 方式运行，输出身体、脸、脚和左右手共 133 个 2D 关键点。
4. HaMeR 和独立 WiLoR 使用 ViTPose 手关键点构造左右手 crop。
5. Hand4Whole++ 使用 GVHMR 人物 bbox 截取整个人，再由其内部 DWPose 定位手、WiLoR 回归 MANO。
6. ViTPose 手关键点还用于计算 MANO 的 2D 重投影误差和每帧可信度。

因此，YOLO 主要负责“找到并跟踪人”，ViTPose 负责“在人的区域内找身体和手关键点”。Hand4Whole++ 正常运行时不会再用自己的 YOLO fallback；只有缺少 `bbx.pt` 时才会回退到 `yolo11n.pt`。

## 4. GVHMR-hand 阶段

单片段入口是：

```text
GVHMR-hand/GVHMR-main/tools/pipeline/run_pipeline.sh
```

批处理入口会为每个视频调用一次该脚本。

### 4.1 身体估计

GVHMR/HMR2 根据人物跟踪、ViTPose 身体关键点和视频特征恢复：

- SMPL-X/SMPL 身体姿态；
- 根节点平移与全局运动；
- 相机轨迹；
- `hmr4d_results.pt`；
- 带手部网格的 `1_incam.mp4` 和 `2_global.mp4`。

### 4.2 HaMeR

当前 HaMeR 配置会：

- 从 ViTPose 左右手 2D 点生成手部 bbox；
- 对 3.0、3.5、4.0 三种 crop 尺度分别估计；
- 用重投影误差和时序切换代价选择逐帧候选；
- 对高可信帧执行少量 2D 关键点测试时优化；
- 保留低可信估计，同时用 valid、重投影误差和后续滤波决定可信度，不再直接清零为张手。

### 4.3 WiLoR

独立 WiLoR 流程与 HaMeR 共用人物跟踪、ViTPose crop、候选选择、重投影和后处理，但使用 WiLoR 网络。其 crop 候选为 1.8、2.0、2.2，更接近 WiLoR 官方 2.0 设置。

`GVHMR_WILOR_FAST=0` 为默认全精度模式；设置为 1 可以加速，但应先完成精度对比。

### 4.4 Hand4Whole++

Hand4Whole++ 逐人物、逐帧流式读取视频，不把整段视频一次性解码进内存。当前设置：

- 使用 GVHMR 人物 bbox；
- 内部 DWPose 生成手 bbox；
- 内部 WiLoR 估计左右 MANO；
- `GVHMR_HAND4WHOLEPP_JOINT_SOURCE=direct_mano`；
- 手部 pose 和 21 个 3D joints 来自同一套 WiLoR/MANO 结果；
- ViTPose 用于独立重投影评分；
- checkpoint 采用内存映射加载，只加载推理需要的网络权重。

`direct_mano` 避免旧的 `fused_smplx` 关节与 WiLoR 手姿态不一致。

## 5. MANO 手部清洗

三种估计后端共用同一条后处理链：

```text
raw mano_params.pt
  -> 可选 wrist candidate filter
  -> temporal outlier filter
  -> finger articulation filter
  -> 用最终 MANO 重新渲染 GVHMR
  -> 导出同一轨迹到 001_smplx_hands.npz
```

当前三个 wrapper 的默认值：

```text
wrist filter    = 0
temporal filter = 1
finger filter   = 1
```

### 5.1 为什么 wrist candidate filter 默认关闭

该滤波器会在 MANO、翻转 MANO、身体腕部和混合方向之间选候选。候选分支切换本身可能产生 90 度或 180 度跳变，因此默认关闭。

目前采用职责分离：

- GVHMR/SMPL 身体负责肩、肘、腕和手臂到手掌的稳定连接；
- MANO 主要负责手指弯曲和局部掌形；
- H1 的单轴手腕 roll 从 SMPL 掌腕方向估计；
- G1/Dex3 和 G1/BrainCo 保留 GMR 求得的完整三轴腕部方向。

### 5.2 Temporal filter

`filter_mano_temporal.py` 综合使用：

- ViTPose 手关键点置信度；
- MANO 到 2D 手关键点的重投影误差；
- 手 bbox 相对前臂的尺寸；
- bbox 缩小、突跳、左右手重叠和时间连续性；
- 最大插值缺口与边缘 hold 长度。

绝对像素误差仍作为缺少 bbox 时的 fallback；正常情况下主要使用：

```text
relative_error = reprojection_error / hand_bbox_diagonal
```

这使远近尺度不同的视频使用同一判断标准。

### 5.3 Finger filter

`filter_mano_fingers.py` 处理：

- 可靠帧与弱证据帧使用不同平滑权重；
- 短缺口插值和有限边缘 hold；
- MANO joint rotation/position 单帧最大变化；
- 手腕四元数时序平滑；
- bbox/手尺寸异常；
- “2D 手已经张开但 MANO 仍卷曲”的 open-hand rescue。

它的目标是清除闪烁、孤立尖峰和错误卷曲，不负责凭空恢复长时间不可见的真实手势。

### 5.4 三种约束 profile

统一通过：

```bash
GVHMR_HAND_CONSTRAINT_PROFILE=responsive|balanced|conservative
```

| profile | 使用场景 | 行为 |
| --- | --- | --- |
| `responsive` | 手很清晰、快速动作 | 最短窗口、较大速度/加速度范围 |
| `balanced` | 普通 30 FPS 视频 | 默认折中 |
| `conservative` | 小手、远景、遮挡、抖动较多 | 最强修复和阻尼 |

profile 同时协调 MANO 源信号清洗和机器人手目标空间的速度/加速度约束，避免两端各自叠加一套过强平滑。

### 5.5 统一手部质量分数

`convert_to_npz.py` 为左右手生成连续的：

```text
left_hand_quality
right_hand_quality
```

质量分数综合：

- 归一化重投影误差；
- 原始 valid；
- 当前帧是直接观测、插值、hold 还是修复结果；
- bbox 和关键点证据。

可靠可见帧更依赖图像目标；低可信帧在机器人 IK 中降低位置约束、提高时序约束。它是软权重，不会再次改写 GVHMR 已经显示的手势。

## 6. 转换、Locomotion 和 PHC

### 6.1 GVHMR 转换

`convert_to_npz.py` 把结果拆成两个同步文件：

| 文件 | 内容 |
| --- | --- |
| `001_converted.npz` | SMPL-H 身体姿态、根节点和形状 |
| `001_smplx_hands.npz` | 左右 MANO pose、21 joints、valid、bbox、重投影误差、quality 和 wrist frame |

wrapper 使用 `GVHMR_HAND_REFINE_MODE=raw`，表示 sidecar 直接保存最终已滤波、已用于 GVHMR 渲染的 MANO 轨迹。不会只为 GMR 再做第二遍隐藏插值，因此 GVHMR 面板和 GMR 输入保持一致。

### 6.2 Locomotion

Locomotion 使用身体 NPZ 做高度/地面优化，输出：

```text
locomotion/optimizer/results_filter/001/001_optimized.npz
001_smoothed.npz
```

`001_smoothed.npz` 是当前 GMR 默认身体来源。手部不塞回这一身体文件，而是继续通过 `001_smplx_hands.npz` 独立传递，防止 PHC/SMPL 格式丢失手指。

### 6.3 PHC

PHC 对身体动作做物理修复并通过 Isaac Gym 渲染：

```text
phc_repaired/
phc_renderings/*.mp4
001_phc_grounded.npz
001_phc_smoothed.npz
001_phc_smoothed_grounded.npz
```

默认会：

- 用 PHC primitive/composer 模型修复身体；
- 以 30 FPS 输出；
- 使用 GVHMR 相机参数；
- 对 PHC 导出的尖峰再平滑；
- 修正地面高度；
- 保留原始 Isaac 视频供 2x2 对比。

可用 `SKIP_PHC=1` 跳过 PHC。GMR 默认仍使用 `001_smoothed.npz`，而不是 PHC 输出。

## 7. GMR 与机器人手接入

批处理默认 `RUN_GMR=1`，完成所有视频的 GVHMR/Locomotion/PHC 后会自动调用：

```text
GMR-master/run_show_gmr_batch.sh
```

默认身体来源：

```text
GMR_SOURCE=smoothed -> 001_smoothed.npz
```

可选：

| `GMR_SOURCE` | 文件 |
| --- | --- |
| `converted` | `001_converted.npz` |
| `smoothed` | `001_smoothed.npz` |
| `phc_smoothed` | `001_phc_smoothed.npz` |
| `auto` | 依次寻找 smoothed、converted、phc_smoothed |

GMR 同时自动读取同目录下的：

```text
001_smplx_hands.npz
```

### 7.1 身体与腕部

GMR 首先把 SMPL-H 身体重定向到目标机器人，得到 `robot_motion.pkl`。

腕部处理按机器人结构区分：

- H1 + Sharpa：H1 本体只有单轴手部 roll，肩/肘/腕由身体 GMR 驱动，掌心 roll 使用稳定的 SMPL 来源并限制到合理范围。
- G1 + Dex3：保留 G1 完整三轴腕部方向，不再把 pitch/yaw 强制置零。
- G1 + BrainCo：BrainCo 安装在 G1 腕部后，同样保留完整三轴腕部方向。

MANO crop-camera 的 global wrist orientation 仅用于诊断或 wrist-frame 构造，不再直接覆盖机器人手臂方向。这是避免 GVHMR 与 GMR 后几秒逐渐相差约 90 度的关键。

### 7.2 Sharpa Wave

`sharpa_hand_retarget.py` 根据 MANO 21 joints 构建 morphology-normalized MCP/PIP/DIP/tip 目标，并求解 Sharpa 22-DoF 手链。

现有约束包括：

- 左右手独立正确 mount quaternion；
- PIP/DIP 最大生理解剖弯曲；
- source bone direction 单帧突变检测；
- source bone length 漂移检测；
- 短异常段修复；
- 每帧 quality 自适应位置 cost；
- 低可信帧增强 temporal cost；
- 关节限位、姿态先验；
- 速度与置信度自适应加速度限制。

输出：

```text
001_sharpa_chain_hands.npz
```

### 7.3 G1 Dex3

选择 `GMR_HAND_MODEL=g1` 后使用：

```text
GMR-master/assets/unitree_g1/g1_mocap_29dof_with_hands.xml
```

G1 身体和 Dex3 手在同一 MuJoCo 模型中。Dex3 是宇树三指手，每侧由独立的左右手资产/关节组成，因此模型或目录中常以 left/right 两份出现；不是重复模型。

### 7.4 BrainCo Revo2

选择 `GMR_HAND_MODEL=brainco` 后使用 Unitree G1 身体并外接 BrainCo Revo2。当前选择 Revo2，而不是 Revo3，是因为已有官方 G1 适配路径指向 Revo2。

IK 输出每手 6 个主动电机，并在模型内应用五指官方 mimic 关系。

输出：

```text
001_brainco_revo2_hands.npz
```

## 8. 渲染与 2x2 对比

GMR 渲染默认：

```text
GMR_RENDER=1
GMR_COMPOSITE=1
GMR_CAMERA_SOURCE=gvhmr
GMR_MUJOCO_GL=osmesa
```

2x2 视频包含：

1. 原始输入视频；
2. 最终滤波后的 GVHMR 人体与手；
3. PHC/Isaac Gym 身体动作；
4. GMR 机器人身体与目标机器人手。

输出文件：

```text
composite_2x2.mp4
```

GMR 渲染名称随目标手变化，例如：

```text
unitree_h1_with_hand_sharpa_gvhmr.mp4
unitree_g1_with_hands_retarget_gvhmr.mp4
unitree_g1_brainco_revo2_gvhmr.mp4
```

### 8.1 物体重建、6D 位姿和两块面板接入

物体链路的职责划分如下：

```text
FoundationPose++:
  RGB-D + mask -> Hunyuan3D 米制 mesh -> T_camera_object

GVHMR pipeline:
  T_camera_object + T_world_to_camera -> GVHMR 世界轨迹
  在 1_incam.mp4 合成真实物体 mesh

GMR:
  映射到机器人 z-up 世界 -> GMR mesh 渲染
  CoACD/凸包碰撞件 -> free-joint 物体接触仿真
```

重要数据前提：FoundationPose 的 6D 注册和 `compute_real_dims.py` 都需要与 RGB
逐像素对齐的深度。当前 `dataset_new6/*.mp4` 只有 RGB，不能仅凭这些 MP4
直接得到“真实尺度”的 FoundationPose 结果。需要配套 RealSense `.db3`、
`self_demo` 的 stereo `depth.h5`，或先生成并标定可信的米制深度。Hunyuan3D
只从单张 RGBA 生成外观网格，本身不提供视频 6D 轨迹或可靠米制尺度。

当前 RGB-only 推荐改走 `do-as-i-do`：

```text
SAM3 mask -> Hunyuan3D mesh -> Fast-SAM3D 6D
-> MoGe pointmap + HaWoR 手部尺度锚定
-> 统一 T_camera_object/米制 mesh 接口
```

配置入口和首次安装命令见 `PIPELINE_ENGINEERING.md` 第 10 节。

先在 `foundationpose-plus-plus-main` 中完成原仓库流程，期望得到：

```text
<foundation_dir>/
  cam_K.json
  color/
  depth/
  mesh/mesh.obj
  pose.npy                 # (T, 4, 4), T_camera_object
```

腾讯云后端只从环境变量读取凭证：

```bash
export TENCENTCLOUD_SECRET_ID=...
export TENCENTCLOUD_SECRET_KEY=...
```

源码中曾出现过的旧凭证必须在腾讯云控制台轮换。调用生成接口可能计费，不会由
总管线隐式触发。

将已经完成的 FoundationPose 结果接到某个片段：

```bash
PY_LOCO="$HOME/miniconda3/envs/locomotion/bin/python"
CLIP_DIR="$PWD/output_dir/dataset_new6_hand4wholepp_directmano_gmr_sharpa_aligned/Date02_Sub02_chairwood_hand.0.color"
FOUNDATION_DIR="$PWD/foundationpose-plus-plus-main/test_data/chairwood"

"$PY_LOCO" \
  GVHMR-hand/GVHMR-main/tools/pipeline/run_object_reconstruction_bridge.py \
  --foundation_dir "$FOUNDATION_DIR" \
  --clip_dir "$CLIP_DIR" \
  --collision_method auto
```

该入口依次调用：

- `GMR-master/scripts/build_object_collision_asset.py`：CoACD 凸分解；未安装
  `coacd` 时回退为单凸包；
- `GMR-master/scripts/import_foundationpose_object.py`：完成
  `T_camera_object -> GVHMR world -> GMR z-up world`；
- `GVHMR-hand/GVHMR-main/tools/pipeline/render_foundationpose_object.py`：
  生成 `1_incam_object.mp4`。

输出：

```text
<clip>/object_reconstruction/
  object_motion_gvhmr.npz
  object_motion_gmr.npz
  object_manifest.json
  collision/collision_*.obj
  collision/collision_manifest.json
<clip>/gvhmr_out/<clip>/1_incam_object.mp4
```

`run_show_gmr_batch.sh` 会自动寻找
`object_reconstruction/object_motion_gmr.npz`，GMR 侧用真实 mesh 渲染；2x2
合成会优先选择 `1_incam_object.mp4`，因此 GVHMR 和 GMR 两块都会出现重建物体。
重新生成 GMR 与 2x2：

```bash
SHOW_ROOT="$(dirname "$CLIP_DIR")" \
OUT_ROOT="$(dirname "$CLIP_DIR")" \
GMR_OVERRIDE=1 \
bash GMR-master/run_show_gmr_batch.sh
```

### 8.2 MuJoCo 动态抓取验证

普通 GMR 视频中的物体按观测轨迹作 mocap 显示，只证明坐标和视觉对齐，不能证明
抓取成功。下面的脚本把物体改为带 free joint、质量、摩擦、凸碰撞件的动态刚体；
机器人仍按 GMR qpos 做运动学回放，物体在 `release_frame` 后只能通过接触被带走：

```bash
"$PY_LOCO" GMR-master/scripts/simulate_robot_object_contacts.py \
  --robot_motion_path "$CLIP_DIR/robot_motion.pkl" \
  --object_motion_path "$CLIP_DIR/object_reconstruction/object_motion_gmr.npz" \
  --sharpa_hand_npz "$CLIP_DIR/001_sharpa_chain_hands.npz" \
  --release_frame 300 \
  --substeps 16 \
  --object_floor_collision \
  --output "$CLIP_DIR/object_reconstruction/object_dynamic_sim.npz" \
  --video_path "$CLIP_DIR/object_reconstruction/object_dynamic_sim.mp4"
```

`release_frame` 应设在手指闭合前若干帧。对应 JSON 报告中的
`dynamic_lift_success=true` 才表示“自由物体在有效手部接触期间被抬高”，而不是
物体被轨迹强制跟随。若失败，需要继续做接触感知手臂/手指 IK 或 MPC 优化；
仅调高摩擦不能修复错误的物体位姿、尺度或穿模。

## 9. 输出目录结构

以 Hand4Whole++ + Sharpa 为例：

```text
output_dir/dataset_new6_hand4wholepp_directmano_gmr_sharpa_aligned/
  batch.log
  summary.csv
  <clip>.pipeline.log
  <clip>/
    gvhmr_out/<clip>/
      0_input_video.mp4
      preprocess/bbx.pt
      vitpose_wholebody.pt
      mano_params.pt
      hmr4d_results.pt
      1_incam.mp4
      1_incam_object.mp4
      2_global.mp4

    mano_params_temporal_fixed.pt
    mano_params_temporal_fixed.json
    mano_params_finger_fixed.pt
    mano_params_finger_fixed.json

    hamer_diagnostics/
      <clip>_hamer_diag_*.mp4
      <clip>_hamer_diag_*.json

    001_converted.npz
    001_smplx_hands.npz
    locomotion/
    001_smoothed.npz
    gvhmr_camera.npz

    phc_repaired/
    phc_renderings/
    001_phc_grounded.npz
    001_phc_smoothed.npz
    001_phc_smoothed_grounded.npz

    robot_motion.pkl
    001_sharpa_chain_hands.npz
    object_reconstruction/
      object_motion_gvhmr.npz
      object_motion_gmr.npz
      object_manifest.json
      collision/
    unitree_h1_with_hand_sharpa_gvhmr.mp4
    composite_2x2.mp4

    .hand_backend
    .hand_config
    .filter_config
    .filtered_render_config
```

`hamer_diagnostics` 是历史目录名，HaMeR、Hand4Whole++、WiLoR 都会使用它；目录名不代表当前后端一定是 HaMeR。真实后端记录在 `.hand_backend`。

## 10. 缓存、续跑与强制重算

默认：

```text
SKIP_EXISTING=1
GMR_OVERRIDE=1
```

流程会缓存：

- 工作视频；
- YOLO bbox；
- ViTPose 关键点；
- MANO；
- GVHMR；
- 滤波结果；
- Locomotion；
- PHC；
- GMR 和渲染。

`.hand_config` 和 `.filter_config` 保存代码、模型和关键参数 fingerprint。后端、checkpoint 或滤波配置变化时，相关阶段会自动重算。

如果运行中断或系统重启，使用完全相同的命令重新运行即可。已经生成的 `bbx.pt`、`vitpose_wholebody.pt` 等会被复用。

常用强制开关：

| 变量 | 作用 |
| --- | --- |
| `FORCE_WORK_VIDEO=1` | 重建工作视频 |
| `GVHMR_FORCE_HAND_PREPROCESS=1` | 重跑手部估计并重新渲染 GVHMR |
| `FORCE_LOCO=1` | 重跑 Locomotion |
| `FORCE_SMOOTH=1` | 重跑身体平滑 |
| `FORCE_PHC=1` | 重跑 PHC |
| `GMR_OVERRIDE=1` | 覆盖 GMR、机器人手 IK 和渲染 |
| `RUN_GMR=0` | 只运行到 GVHMR/Locomotion/PHC |

不要用 `SKIP_EXISTING=0` 作为日常“保险”，它会导致大量已完成阶段无条件重算。

## 11. 只重跑 GMR

已有 `001_smoothed.npz` 和 `001_smplx_hands.npz` 时，可以不重跑手部估计：

```bash
cd <repo-root>

export PIPELINE_ROOT="$PWD"
export GMR_HAND_MODEL=sharpa
source GMR-master/configure_hand_model.sh

SHOW_ROOT="$PWD/output_dir/dataset_new6_hand4wholepp_directmano_gmr_sharpa_aligned" \
OUT_ROOT="$PWD/output_dir/dataset_new6_hand4wholepp_directmano_gmr_sharpa_aligned" \
GMR_AUTO_HAND_NPZ=1 \
GMR_OVERRIDE=1 \
bash GMR-master/run_show_gmr_batch.sh
```

切换到 G1 或 BrainCo 时修改 `GMR_HAND_MODEL`，并建议使用新的 `OUT_ROOT`，避免覆盖原机器人结果。

## 12. 模型与机器人资产

### 12.1 HaMeR

```bash
bash GVHMR-hand/GVHMR-main/scripts/setup_hamer_assets.sh
```

### 12.2 WiLoR

```bash
bash GVHMR-hand/GVHMR-main/scripts/setup_wilor_assets.sh
```

MANO 模型需要从 MANO 官方站点注册下载，setup 脚本会提示缺失路径。

### 12.3 Hand4Whole++

```bash
DOWNLOAD_H4WPP=1 \
bash GVHMR-hand/GVHMR-main/scripts/setup_hand4wholepp_assets.sh
```

该脚本准备：

- Hand4Whole++ 仓库；
- `snapshot_6.pth`；
- WiLoR；
- DWPose/mmpose；
- SMPL、SMPL-X、MANO；
- MANO/SMPL-X 映射文件。

部分官方模型可能仍需要手动下载或接受上游许可。

### 12.4 机器人手资产

检查本地状态：

```bash
bash GMR-master/scripts/setup_robot_hand_assets.sh status
```

只下载 BrainCo Revo2：

```bash
bash GMR-master/scripts/setup_robot_hand_assets.sh brainco
```

下载 Unitree、BrainCo 和 LimX/逐际候选资产：

```bash
bash GMR-master/scripts/setup_robot_hand_assets.sh download
```

外部厂商资产保存在：

```text
GMR-master/third_party/robot_hands/
```

重新分发前应检查各上游仓库许可证。

## 13. Hand4Whole++ 内存保护

Hand4Whole++ 初始化时需要同时处理约 2.4GB 的 WiLoR checkpoint 和约 3.0GB 的 whole-body checkpoint。当前已经加入：

- checkpoint `mmap` 文件映射；
- 不加载推理不需要的 optimizer state；
- 权重写入模型后立即释放 checkpoint 字典；
- Hand4Whole++ 前回收 ViTPose CUDA cache；
- 手部预处理与 HMR2/GVHMR 使用独立 OS 进程；
- `GVHMR_LOW_MEMORY=1`；
- 默认 Hand4Whole++ batch 从 4 降为 2。

16GB 主存机器首次建议：

```bash
GVHMR_HAND4WHOLEPP_BATCH_SIZE=1 \
GVHMR_ISOLATE_HAND_PREPROCESS=1 \
GVHMR_LOW_MEMORY=1 \
bash GVHMR-hand/GVHMR-main/tools/pipeline/run_batch_dataset6_hand4wholepp.sh
```

batch size 只影响推理吞吐量，不改变模型和滤波精度。

如果仍然是整机硬死机，而不是 Python/CUDA OOM，应检查：

```bash
journalctl -b -1 -k | grep -Ei 'oom|nvrm|xid|mce|hardware error|thermal'
free -h
swapon --show
nvidia-smi
```

内核出现 `Machine check events logged` 时，还需要检查内存、CPU 稳定性、超频、电源和 RTX 4090 供电，不能只继续减 Python batch size。

## 14. 如何定位误差在哪一层

| 现象 | 优先检查 |
| --- | --- |
| 原视频清楚，但手部诊断中的 2D 点就错了 | 人物 crop、ViTPose、左右手身份 |
| 2D 点正确，GVHMR 手网格错误 | 手估计后端、crop 尺度、MANO 重投影 |
| GVHMR 正确但有逐帧闪烁 | temporal/finger filter 和 quality |
| GVHMR 手势正确，GMR 手势不同 | `001_smplx_hands.npz`、机器人 IK、joint limits |
| 两手整体方向差约 90/180 度 | wrist frame、mount quaternion、机器人腕部 DoF |
| 只有手臂动作，没有机器人手指 | `GMR_AUTO_HAND_NPZ`、sidecar、目标手轨迹文件 |
| Sharpa 左手方向反了 | 左右 mount quaternion 是否分别传入 |
| BrainCo 手指活动不合理 | 6 个主动电机输出和 mimic 约束 |

推荐先看：

```text
<clip>/hamer_diagnostics/*.mp4
<clip>/hamer_diagnostics/*.json
<clip>/001_smplx_hands.npz
<clip>/001_sharpa_chain_hands.npz
<clip>/composite_2x2.mp4
<output_root>/<clip>.pipeline.log
<output_root>/batch.log
```

## 15. 相关文档

- `PIPELINE_ENGINEERING.md`：完整代码流程、YAML 优先级和数据文件字段字典。
- `GMR-master/HAND_ACCURACY_ROADMAP.md`：当前手部精度问题、论文方向和后续路线。
- `GMR-master/ROBOT_HAND_ASSETS.md`：机器人手资产及接入状态。
- `GMR-master/DO_AS_I_DO_GMR_ADJUSTMENTS.md`：腕部、掌心 roll 和 do-as-i-do 坐标调整。
- `GVHMR-hand/GVHMR-main/docs/INSTALL.md`：GVHMR-hand 基础环境。
- `GMR-master/README.md`：上游 GMR 通用说明。

日常运行应优先以本文档和三个 `run_batch_dataset6_<backend>.sh` wrapper 为准；wrapper 中的默认参数是当前实验配置的最终来源。
