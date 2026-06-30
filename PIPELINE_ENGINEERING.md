# 人体、手部、物体重建与机器人重定向工程说明

本文描述当前工作区中已经接通的完整数据管线、代码模块职责、执行命令、配置方式和
中间文件协议。日常参数与快速命令仍可查阅 `PIPELINE_README.md`；本文重点回答：

- 输入视频如何转换；
- 每个阶段由哪个脚本负责；
- 每个 `.pt`、`.npz`、`.pkl` 文件保存什么；
- RGB-only 物体路线与 RGB-D 路线有什么区别；
- 如何通过 YAML 配置统一启动。

## 0. 当前落地状态

已经完成：

- 人体/手部/Locomotion/PHC/GMR/2x2 主流程；
- 四级配置优先级与关键 CLI 覆盖；
- MP4/AVI/MOV/MKV/M4V 输入扫描；
- RGB-only 的 Hunyuan mesh + 单目 6D 适配代码；
- RGB-D FoundationPose 适配；
- GVHMR/GMR 两块物体可视化；
- MuJoCo 凸碰撞件和 free-joint 动态抓取验证；
- 中间文件数据字典。

尚未在真实 `chairwood` 视频上执行的部分：

- `do-as-i-do-main/reconstruction/modules/` 当前为空；
- 对应四个 Conda 环境和权重尚未安装/核验；
- Hunyuan 云 API 尚未调用；
- 参考帧 `200` 只是示例，尚未人工确认；
- 当前 RTX 4090 为 24GB，需要先用低显存配置试跑。

因此当前状态是“代码和接口已准备，真实 RGB-only 物体任务等待上游模型安装与凭证
轮换后首次运行”，不是已经产出真实 chair mesh/pose。

## 1. 总体架构

```text
MP4 / AVI / MOV / MKV / M4V
  |
  +-- 工作视频标准化：H.264、偶数尺寸、默认最大 1280x960、无音频
  |
  +-- 人体与手部
  |     YOLO 人物跟踪
  |       -> ViTPose WholeBody 133 点
  |       -> GVHMR 身体/相机
  |       -> HaMeR / Hand4Whole++ / WiLoR 手部 MANO
  |       -> wrist / temporal / finger filters
  |       -> SMPL-H 身体 NPZ + 独立手部 sidecar NPZ
  |       -> Locomotion 高度优化
  |       -> Savitzky-Golay 平滑
  |       -> PHC 物理修复与 Isaac Gym 视频
  |       -> GMR 身体重定向
  |       -> Sharpa / Dex3 / BrainCo 手部重定向
  |
  +-- 物体（二选一）
  |     RGB-only:
  |       SAM3 视频分割
  |       -> SAM3D 或 Hunyuan3D mesh
  |       -> MoGe pointmap
  |       -> HaWoR 手部尺度锚定
  |       -> TAPIR + Fast-SAM3D 6D 跟踪
  |       -> T_camera_object + 米制 mesh
  |
  |     RGB-D:
  |       对齐 RGB/depth + K
  |       -> SAM-HQ mask
  |       -> Hunyuan3D mesh
  |       -> 深度反投影恢复真实尺寸
  |       -> FoundationPose 6D 跟踪
  |       -> T_camera_object + 米制 mesh
  |
  +-- 人体/物体融合
        T_camera_object + GVHMR T_world_to_camera
          -> object_motion_gvhmr.npz
          -> GVHMR 物体合成视频
          -> object_motion_gmr.npz
          -> GMR 真实 mesh 视频
          -> CoACD/凸包碰撞件
          -> MuJoCo free-joint 动态抓取验证
          -> 2x2 对比视频
```

## 2. 统一配置入口

默认配置：

```text
configs/default.yaml
```

RGB-only + H1/Sharpa 管线配置：

```text
configs/pipelines/rgb_monocular_sharpa.yaml
```

配置驱动入口：

```text
scripts/run_pipeline_from_config.py
```

建议只复制管线专属配置。默认 YAML 保持为全工程稳定默认值：

```bash
cp configs/pipelines/rgb_monocular_sharpa.yaml \
  configs/pipelines/my_experiment.yaml
```

实际配置优先级严格为：

```text
CLI > 环境变量 > 管线专属 YAML > configs/default.yaml
```

检查 YAML、输入目录、后端和脚本：

```bash
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/my_experiment.yaml \
  --stage human \
  --check
```

查看 YAML 最终映射出的环境变量：

```bash
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/my_experiment.yaml \
  --stage human \
  --check \
  --print-env
```

查看四级合并后的最终配置：

```bash
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/my_experiment.yaml \
  --stage human \
  --check \
  --print-config
```

只打印将要运行的命令：

```bash
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/my_experiment.yaml \
  --stage human \
  --dry-run
```

执行人体、手部、PHC 和 GMR：

```bash
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/my_experiment.yaml \
  --stage human
```

执行物体阶段：

```bash
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/my_experiment.yaml \
  --stage object
```

执行全部启用阶段：

```bash
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/my_experiment.yaml \
  --stage all
```

常用 CLI 覆盖：

```bash
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/my_experiment.yaml \
  --clip-filter chairwood \
  --backend hand4wholepp \
  --robot sharpa \
  --object-mode monocular \
  --object-name chair \
  --reference-frame 200 \
  --anchor-hand right \
  --set human.hand_batch_size=2 \
  --set object.dynamic_validation.enabled=false \
  --stage all
```

环境变量覆盖示例：

```bash
GVHMR_HAND4WHOLEPP_BATCH_SIZE=2 \
GMR_MUJOCO_GL=osmesa \
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/my_experiment.yaml \
  --stage human
```

如果同时传入 `--set human.hand_batch_size=1`，CLI 的 1 会覆盖环境变量的 2。

### 2.1 YAML 分区

| 分区 | 作用 |
| --- | --- |
| `input` | 数据目录、扩展名、片段过滤、工作视频规格 |
| `output` | 实验输出根目录 |
| `resume` | 缓存和各阶段强制重算 |
| `human` | 手部后端、batch、内存模式、滤波 profile |
| `locomotion` | 高度优化和身体平滑 |
| `phc` | 是否运行 PHC |
| `gmr` | 机器人、手、相机、渲染和 2x2 |
| `object` | RGB-only/RGB-D 路线、物体、碰撞和动态验证 |
| `runtime` | 统一 Python 解释器 |
| `environment` | 直接覆盖高级 Bash 环境变量 |

`environment` 是兼容已有脚本的逃生口，例如：

```yaml
environment:
  GVHMR_FINGER_FILTER_SMOOTH_WINDOW: 15
  GMR_SHARPA_MAX_DELTA: 0.12
```

不要在 YAML 中写入：

```yaml
TENCENTCLOUD_SECRET_ID: ...
TENCENTCLOUD_SECRET_KEY: ...
```

云凭证只能通过当前进程环境注入。

## 3. 初始输入状态

### 3.1 人体主流程

输入目录默认是：

```text
dataset_new6/
  clip_a.mp4
  clip_b.avi
  clip_c.mov
```

支持扩展名：

```text
.mp4 .avi .mov .mkv .m4v
```

一个目录下默认每个文件代表一个独立片段。片段名是“去除最后一个媒体扩展名后的
文件名”；文件名内部的点会保留，例如：

```text
Date02_Sub02_chairwood_hand.0.color.mp4
-> Date02_Sub02_chairwood_hand.0.color
```

### 3.2 RGB-only 物体流程

最低输入：

```text
单目 RGB 视频
参考帧编号
物体名称
锚定手：left 或 right
```

该路线不需要真实深度，但需要通过单目 pointmap 和 HaWoR 手部几何推断尺度。
单目尺度是估计值，进入接触仿真前必须检查。

### 3.3 RGB-D 物体流程

FoundationPose 路线需要：

```text
color/<frame>.png
depth/<frame>.png      # 与 color 对齐，默认 uint16 毫米
cam_K.json
0_mask.png
mesh/mesh.obj
```

仅有 RGB 视频时，不能直接运行 FoundationPose 的注册/跟踪。Hunyuan3D 可以从
RGB 抠图生成形状，但不会提供逐帧 6D 位姿，也不会自动恢复可靠米制尺度。

## 4. 工作视频标准化

生产者：

```text
GVHMR-hand/GVHMR-main/tools/pipeline/run_batch_dataset6.sh
```

默认转换：

```bash
ffmpeg \
  -i input.avi \
  -vf "scale=1280:960:force_original_aspect_ratio=decrease,scale=trunc(iw/2)*2:trunc(ih/2)*2" \
  -c:v libx264 \
  -preset veryfast \
  -crf 18 \
  -pix_fmt yuv420p \
  -an \
  work_video.mp4
```

输入状态：

```text
原始 MP4/AVI/MOV/MKV/M4V，分辨率和编码可不同
```

输出状态：

```text
dataset_new6_work_1280/<clip>.mp4
dataset_new6_work_1280/<clip>.mp4.config
```

`.config` 记录原文件路径、大小/时间、目标尺寸和 CRF，用于判断缓存是否失效。

## 5. 人体与手部模块

### 5.1 批处理入口

推荐入口：

```text
GVHMR-hand/GVHMR-main/tools/pipeline/
  run_batch_dataset6_hamer.sh
  run_batch_dataset6_hand4wholepp.sh
  run_batch_dataset6_wilor.sh
```

单片段入口：

```text
GVHMR-hand/GVHMR-main/tools/pipeline/run_pipeline.sh
```

直接执行单片段：

```bash
bash GVHMR-hand/GVHMR-main/tools/pipeline/run_pipeline.sh \
  /absolute/path/input.mp4 \
  /absolute/path/output_clip
```

### 5.2 人物框

输出：

```text
gvhmr_out/<clip>/preprocess/bbx.pt
```

数据：

| 字段 | 通用形状 | 说明 |
| --- | --- | --- |
| `bbx_xyxy` | `(P,T,4)` | 人物框 `(x1,y1,x2,y2)` |
| `bbx_xys` | `(P,T,3)` | 中心和统一尺度 |
| `bbx_conf` | `(P,T)` | 跟踪置信度 |

`P` 是人物数，当前主流程通常取 `P=1`；`T` 是视频帧数。

### 5.3 ViTPose WholeBody

输出：

```text
gvhmr_out/<clip>/vitpose_wholebody.pt
```

形状：

```text
(P,T,133,3)
```

最后一维是：

```text
x, y, confidence
```

133 点包含身体、脚、脸和左右手。

### 5.4 MANO 手部估计

输出：

```text
gvhmr_out/<clip>/mano_params.pt
```

主要字段：

| 字段 | 通用形状 | 说明 |
| --- | --- | --- |
| `left/right_hand_global_orient` | `(P,T,3,3)` | 手部全局旋转矩阵 |
| `left/right_hand_pose` | `(P,T,15,3,3)` | 15 个 MANO 关节旋转矩阵 |
| `left/right_hand_joints_3d` | `(P,T,21,3)` | 21 个手关节 |
| `left/right_hand_valid` | `(P,T)` | 该帧手部是否有效 |
| `left/right_hand_reproj_error` | `(P,T)` | 2D 重投影误差 |
| `left/right_hand_bbox_xyxy` | `(P,T,4)` | 手部 crop |

三个后端最终都转换成此接口：

```text
HaMeR
Hand4Whole++
WiLoR
```

### 5.5 MANO 后处理

可能生成：

```text
mano_params_wrist_fixed.pt
mano_params_temporal_fixed.pt
mano_params_temporal_fixed.json
mano_params_finger_fixed.pt
mano_params_finger_fixed.json
```

职责：

| 模块 | 逻辑 |
| --- | --- |
| wrist filter | 在候选腕部方向中做时序/身体一致性选择 |
| temporal filter | 检测 bbox、重投影、关节速度异常并插值短缺口 |
| finger filter | 平滑手指与腕部、限制角速度/位移、修复异常张合 |

最终 MANO 会重新用于 GVHMR 渲染，避免“GVHMR 面板”和“GMR 输入”使用两套手势。

### 5.6 GVHMR 身体结果

输出：

```text
gvhmr_out/<clip>/hmr4d_results.pt
```

主要字段：

| 字段 | 通用形状 | 说明 |
| --- | --- | --- |
| `smpl_params_global.body_pose` | `(P,T,21,3,3)` | 世界系身体关节旋转 |
| `smpl_params_global.global_orient` | `(P,T,1,3,3)` | 世界系根方向 |
| `smpl_params_global.transl` | `(P,T,3)` | 世界系根平移 |
| `smpl_params_global.betas` | `(P,T,10)` | 身形 |
| `smpl_params_incam.*` | 同上 | 相机系 SMPL 参数 |
| `K_fullimg` | `(T,3,3)` | 每帧完整图像内参 |
| `net_outputs.static_conf_logits` | `(P,T,6)` | 静态/相机相关置信信息 |
| `fps` | 标量 | 帧率 |
| `width/height/focal_length` | `(T,)` | 图像和焦距 |

`.pt` 使用 PyTorch 序列化，只能从可信来源加载。

### 5.7 GVHMR 视频

```text
gvhmr_out/<clip>/0_input_video.mp4
gvhmr_out/<clip>/1_incam.mp4
gvhmr_out/<clip>/2_global.mp4
```

| 文件 | 说明 |
| --- | --- |
| `0_input_video.mp4` | 当前工作输入视频 |
| `1_incam.mp4` | 人体/手部投回原相机画面 |
| `2_global.mp4` | 世界视角 |

## 6. SMPL-H 转换与 NPZ 协议

生产者：

```text
GVHMR-hand/GVHMR-main/tools/pipeline/convert_to_npz.py
```

### 6.1 `001_converted.npz`

身体主文件：

| 字段 | 通用形状 | 说明 |
| --- | --- | --- |
| `poses` | `(T,24,3)` | 24 个关节 axis-angle |
| `pose_body` | `(T,63)` | 21 个非根身体关节 axis-angle |
| `root_orient` | `(T,3)` | 根 axis-angle |
| `trans` | `(T,3)` | 当前根平移 |
| `trans_original` | `(T,3)` | 转换前根平移 |
| `betas` | `(16,)` | 身形 |
| `gender` | 字符串标量 | SMPL 性别 |
| `mocap_frame_rate` | 标量 | 帧率 |

### 6.2 `001_smplx_hands.npz`

独立手部 sidecar。它同时保存身体同步信息与手部数据，核心字段：

| 字段 | 通用形状 | 说明 |
| --- | --- | --- |
| `full_pose_smplx` | `(T,165)` | SMPL-X 完整 pose 容器 |
| `left/right_hand_pose` | `(T,45)` | MANO 15 关节 axis-angle |
| `left/right_hand_global_orient` | `(T,3)` | 手部全局 axis-angle |
| `left/right_hand_joints_3d` | `(T,21,3)` | 最终手关节 |
| `left/right_hand_valid` | `(T,)` | 最终有效标记 |
| `left/right_hand_*_raw` | 对应形状 | 滤波前值 |
| `left/right_hand_bbox_xyxy` | `(T,4)` | 手框 |
| `left/right_hand_reproj_error` | `(T,)` | 像素误差 |
| `left/right_hand_reproj_error_relative` | `(T,)` | 相对 bbox 误差 |
| `left/right_hand_bad_mask` | `(T,)` | 坏帧 |
| `left/right_hand_spike_mask` | `(T,)` | 时序尖峰 |
| `left/right_hand_wrist_quat` | `(T,4)` | 腕部四元数 |
| `left/right_hand_palm_normal` | `(T,3)` | 掌面法向 |
| `left/right_hand_quality` | `(T,)` | 下游 IK 软权重 |
| `left/right_hand_source_reliable` | `(T,)` | 可靠观测 |
| `left/right_hand_source_repaired` | `(T,)` | 插值/修复来源 |
| `hand_backend` | 字符串 | hamer/hand4wholepp/wilor |

手部不嵌进 Locomotion/PHC 身体文件，避免 SMPL 格式转换丢失手指。

## 7. Locomotion、平滑与 PHC

### 7.1 Locomotion

输入：

```text
001_converted.npz
```

工作拷贝：

```text
locomotion_in/001/001_optimized.npz
```

输出：

```text
locomotion/optimizer/results_filter/001/001_optimized.npz
```

主要修改根高度、地面接触相关平移；身体 pose 和 SMPL 形状协议保持兼容。

### 7.2 身体平滑

输出：

```text
001_smoothed.npz
```

默认使用 Savitzky-Golay 平滑。字段与 `001_converted.npz` 相同。当前 GMR 默认从
此文件读取身体。

### 7.3 GVHMR 相机导出

输出：

```text
gvhmr_camera.npz
```

| 字段 | 通用形状 | 坐标/说明 |
| --- | --- | --- |
| `camera_pos_world` | `(T,3)` | GVHMR 世界系相机位置 |
| `camera_target_world` | `(T,3)` | 世界系观察目标 |
| `camera_pos_isaac` | `(T,3)` | z-up 坐标相机位置 |
| `camera_target_isaac` | `(T,3)` | z-up 目标 |
| `subject_world` | `(T,3)` | 人体根/目标位置 |
| `subject_isaac` | `(T,3)` | z-up 人体位置 |
| `T_w2c` | `(T,4,4)` | `T_camera_world`，世界点到相机 |
| `K_fullimg` | `(T,3,3)` | 图像内参 |
| `world_to_isaac` | `(3,3)` | GVHMR 世界到 z-up |
| `alignment_offset_world` | `(3,)` | Locomotion 对齐偏移 |
| `horizontal_fov_deg` | 标量 | 水平 FOV |

物体桥接使用：

```text
T_world_object = inverse(T_world_to_camera) @ T_camera_object
```

然后加 `alignment_offset_world` 并转换到 z-up。

### 7.4 PHC

输入：

```text
001_smoothed.npz
```

输出：

```text
phc_repaired/
phc_renderings/*.mp4
001_phc_grounded.npz
001_phc_smoothed.npz
001_phc_smoothed_grounded.npz
001_phc_smoothed_report.json
```

这些 NPZ 继续使用：

```text
poses/root_orient/pose_body/trans/trans_original/betas/gender/mocap_frame_rate
```

PHC 视频来自 Isaac Gym。当前 GMR 默认仍使用 `001_smoothed.npz`，除非配置
`gmr.source: phc_smoothed`。

## 8. GMR 与 `robot_motion.pkl`

生产者：

```text
GMR-master/scripts/smpl_npz_to_robot_headless.py
```

批处理：

```text
GMR-master/run_show_gmr_batch.sh
```

输出：

```text
robot_motion.pkl
```

### 8.1 核心运动字段

| 字段 | 通用形状 | 说明 |
| --- | --- | --- |
| `fps` | 标量 | 输出帧率 |
| `root_pos` | `(T,3)` | MuJoCo z-up 世界中的机器人根位置 |
| `root_rot` | `(T,4)` | 根四元数，存储顺序 `xyzw` |
| `dof_pos` | `(T,D)` | 机器人主体关节位置 |
| `local_body_pos` | `(T,B,3)` | FK/渲染用局部 body 位置 |
| `link_body_list` | `(B,)` | `local_body_pos` 的 body 名称 |

实际 H1 + Sharpa 示例：

```text
T=1250
D=45
B=46
```

Sharpa 手指仍在独立的 `001_sharpa_chain_hands.npz` 中，不塞入 `dof_pos`。

### 8.2 来源与求解配置

| 字段 | 说明 |
| --- | --- |
| `source_motion` | 输入 SMPL NPZ 路径 |
| `model_type` | `smpl/smplh/smplx` |
| `retarget_mode` | full/upper 等 |
| `smooth_window/smooth_polyorder` | GMR 平滑参数 |
| `height_adjust_mode` | 地面高度处理 |
| `human_yaw_offset_deg` | 人体到机器人世界的 yaw |
| `relaxed_orientation_bodies` | 放松方向约束的 body |

### 8.3 手腕与掌心字段

| 字段 | 说明 |
| --- | --- |
| `hand_npz` | 手部 sidecar 路径 |
| `hand_wrist_orientation_mode` | 腕部方向使用策略 |
| `palm_roll_mode/source` | H1 单轴掌心 roll 策略 |
| `left/right_palm_roll_*` | roll 范围、分支、饱和、来源诊断 |
| `left/right_hand_wrist_quat` | `(T,4)` 最终腕部四元数 |
| `left/right_hand_palm_normal` | `(T,3)` 掌面法向 |
| `left/right_hand_wrist_frame_valid` | `(T,)` 腕框有效性 |
| `wrist_pitch_yaw_stabilize` | G1 三轴腕部稳定策略 |

`pkl` 是 Python pickle。它可以保存字典、数组和字符串元数据，但加载 pickle 会执行
反序列化逻辑，因此只能加载本工程或可信来源生成的文件。

## 9. 机器人手输出

### 9.1 Sharpa

```text
001_sharpa_chain_hands.npz
```

核心字段：

| 字段 | 通用形状 | 说明 |
| --- | --- | --- |
| `left/right_hand_qpos` | `(T,22)` | 每只 Sharpa 手 22 个关节 |
| `left/right_hand_qpos_names` | `(22,)` | 关节名称 |
| `left/right_qpos` | `(T,28)` | 含腕部/安装链的扩展 qpos |
| `left/right_valid` | `(T,)` | 有效性 |
| `left/right_reliability` | `(T,)` | 可靠度 |
| `left/right_chain_error_*` | 标量 | IK 链误差统计 |
| `left/right_anatomic_bad_frames` | 标量 | 解剖异常帧数 |
| `left/right_repaired_finger_frames` | 标量 | 修复帧数 |

其余标量保存本次 IK 的 cost、平滑、速度限制和解剖约束，便于结果复现。

### 9.2 G1 Dex3 和 BrainCo

根据 `gmr.hand_model` 输出对应手部 NPZ。身体仍保存于 `robot_motion.pkl`，外接手
保持独立文件并在渲染时按名称写入 MuJoCo qpos。

## 10. RGB-only 物体路线

上游：

```text
do-as-i-do-main/reconstruction/run_pipeline.sh
```

参考 mesh 后端由 YAML 控制：

```yaml
object:
  monocular:
    mesh_backend: hunyuan   # hunyuan | sam3d
```

`hunyuan` 路线是：

```text
SAM3 参考帧 mask
-> RGB + mask 合成 RGBA
-> 腾讯云 Hunyuan3D GLB
-> OBJ
-> Fast-SAM3D 单目 6D 跟踪
-> HaWoR/MoGe 尺度优化
```

因此即使没有 RGB-D，也可以使用 Hunyuan 生成参考网格；绝对尺度由后续手部锚定
而不是 Hunyuan 自己提供。

首次需要：

```bash
cd do-as-i-do-main/reconstruction
./setup/00_init_submodules.sh
./setup/01_create_envs.sh
python -m pip install 'huggingface-hub[cli]<1.0'
hf auth login
./setup/02_fetch_weights.sh --download
cd ../..
```

下载前需要先在 Hugging Face 获得 `facebook/sam3` 和
`facebook/sam-3d-objects` 的访问权限。HaWoR 所需的
`MANO_RIGHT.pkl`/`MANO_LEFT.pkl` 受许可证约束，下载脚本不会代取；
目标路径和完整检查项会由 `--stage object --check` 报出。

Hunyuan 后端还需要轮换后的云凭证：

```bash
cp configs/secrets.env.example configs/secrets.env
# 编辑 configs/secrets.env 后：
source configs/secrets.env
```

`configs/secrets.env` 已被 Git 忽略。

注意其模型许可、Hugging Face 权限和显存要求。

当前机器检测为 RTX 4090 24GB，而上游 README 建议至少 32GB。
`rgb_monocular_sharpa.yaml` 已采用较保守的起步值：

```yaml
object:
  monocular:
    num_pose_samples: 8
    euler_steps: 15
    torch_compile: false
```

这降低峰值显存和首次编译开销，但可能降低姿态候选覆盖；确认显存稳定后再逐步增加。

单独执行：

```bash
bash do-as-i-do-main/reconstruction/run_pipeline.sh \
  /absolute/path/clip.mp4 \
  200 \
  chair \
  right
```

通过统一入口执行当前 `chairwood` 片段：

```bash
source configs/secrets.env

python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/rgb_monocular_sharpa.yaml \
  --stage object \
  --output-root output_dir/dataset_new6_hand4wholepp_directmano_gmr_sharpa_aligned \
  --clip-filter chairwood \
  --object-name chair \
  --reference-frame 200 \
  --anchor-hand right \
  --mesh-backend hunyuan \
  --check

python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/rgb_monocular_sharpa.yaml \
  --stage object \
  --output-root output_dir/dataset_new6_hand4wholepp_directmano_gmr_sharpa_aligned \
  --clip-filter chairwood \
  --object-name chair \
  --reference-frame 200 \
  --anchor-hand right \
  --mesh-backend hunyuan
```

参考帧应选择物体轮廓完整、手部遮挡较少的一帧；`200` 只是配置示例。
Stage 1 会在 `SAM3_DISPLAY` 指定的桌面上弹出 OpenCV 点选窗口；左键添加物体
正样本点、右键添加负样本点，按 Enter 确认。当前机器检测到的是 `:0`，
默认配置已经与之对齐；远程运行时需要 X 转发或显式覆盖 `SAM3_DISPLAY`。

阶段：

| 阶段 | 输入 | 输出 |
| --- | --- | --- |
| ffmpeg | RGB 视频 | `all_frames/*.png` |
| SAM3 | 视频、点击/文本 | `video_segmentation/masks/` |
| SAM3D | 参考图和 mask | 物体 OBJ |
| MoGe | RGB 帧 | `*_pointmap.npy`、`*_intrinsics.npy` |
| HaWoR | RGB 视频 | `all_hand_meshes.npz` |
| GeoCalib | RGB 帧 | `gravity.json` |
| TAPIR | 视频和 mask | 速度统计 |
| Fast-SAM3D | mesh、mask、速度 | `layout.json` |
| camera conversion | layout | `layout_camera_frame.json` |
| hand scale optimization | pointmap、手、物体 | `layout_camera_frame_optimized.json` |

优化后的 layout 每帧包含：

```text
frame_idx / frame_index
local_to_scene.translation_camera_frame
local_to_scene.quat_wxyz_camera_frame
local_to_scene.translation_scale_optimized
```

全局摘要包含：

```text
translation_scale_optimization.mesh_scale
translation_scale_optimization.ref_frame
translation_scale_optimization.ref_frame_k
translation_scale_optimization.per_frame
```

适配器：

```text
GMR-master/scripts/adapt_do_as_i_do_object.py
```

它生成：

```text
monocular_adapter/
  pose.npy
  pose_valid.npy
  pose_observed.npy
  cam_K.json
  mesh/mesh.obj
  adapter_manifest.json
```

其中：

```text
pose.npy: (T,4,4), T_camera_object
mesh/mesh.obj: 已应用手部锚定 mesh_scale，单位按米解释
pose_observed.npy: 原跟踪器直接观测帧
pose_valid.npy: 观测或内部插值后可用帧
```

风险边界：

- 单目绝对尺度不是传感器测量值；
- 手部遮挡、反光、旋转对称物体会造成 pose 漂移；
- 接触前必须检查投影、物体尺寸、手物距离；
- Hunyuan 网格不能直接替换 SAM3D 跟踪网格，除非先对齐网格原点、方向和尺度。

## 11. RGB-D / FoundationPose 物体路线

上游：

```text
foundationpose-plus-plus-main/scripts/run_pipeline.py
```

腾讯云 Hunyuan：

```bash
export TENCENTCLOUD_SECRET_ID=...
export TENCENTCLOUD_SECRET_KEY=...

python foundationpose-plus-plus-main/scripts/run_pipeline.py \
  --use_hunyuan3d \
  --hunyuan3d_backend tencent \
  --activate_2d_tracker \
  --activate_kalman_filter
```

本地服务：

```bash
python foundationpose-plus-plus-main/scripts/run_pipeline.py \
  --use_hunyuan3d \
  --hunyuan3d_backend service \
  --hunyuan3d_url http://localhost:18081 \
  --activate_2d_tracker \
  --activate_kalman_filter
```

输出：

```text
<foundation_dir>/
  color/
  depth/
  cam_K.json
  0_mask.png
  0_mask_rgba.png
  real_dims.json
  mesh/mesh.glb
  mesh/mesh.obj
  pose.npy
```

数据含义：

| 文件 | 说明 |
| --- | --- |
| `real_dims.json` | mask 内深度反投影/PCA 得到的米制尺寸 |
| `mesh.obj` | Hunyuan 网格按真实尺寸缩放后的 OBJ |
| `pose.npy` | `(T,4,4)`，`T_camera_object` |

## 12. 物体统一接口

总入口：

```text
GVHMR-hand/GVHMR-main/tools/pipeline/run_object_reconstruction_bridge.py
```

核心导入：

```text
GMR-master/scripts/import_foundationpose_object.py
```

虽然文件名保留 `foundationpose`，它读取的是统一契约，所以也接受 RGB-only 适配器输出。

### 12.1 `object_motion_gvhmr.npz`

| 字段 | 通用形状 | 说明 |
| --- | --- | --- |
| `pose_camera_object` | `(To,4,4)` | 物体到 OpenCV 相机 |
| `pose_gvhmr_world_object` | `(To,4,4)` | 对齐后的 GVHMR 世界位姿 |
| `K` | `(3,3)` 或 `(To,3,3)` | 物体跟踪使用的内参 |
| `valid` | `(To,)` | 位姿有效性 |
| `visual_mesh_path` | 字符串 | 米制视觉网格 |
| `collision_mesh_paths` | `(C,)` 字符串 | 凸碰撞件 |
| `density/friction/solref` | 标量/数组 | MuJoCo 物理参数 |

### 12.2 `object_motion_gmr.npz`

| 字段 | 通用形状 | 说明 |
| --- | --- | --- |
| `position` | `(T,3)` | GMR z-up 世界位置 |
| `quat_wxyz` | `(T,4)` | GMR 世界物体方向 |
| `valid` | `(T,)` | 有效性 |
| `source_frame_index` | `(T,)` | 对应物体源帧 |
| `robot_root` | `(T,3)` | 映射时使用的机器人根 |
| `subject_gvhmr_world` | `(T,3)` | 映射时使用的人体根 |
| `pose_gvhmr_world_object` | `(T,4,4)` | 调试用世界位姿 |

物体映射保留“物体相对人体根”的位置，再放到机器人根周围。它不会把物体绑定到
机器人手，因此可以暴露手臂 IK 没有真正到达物体的问题。

## 13. 碰撞资产和动态抓取

碰撞构建：

```text
GMR-master/scripts/build_object_collision_asset.py
```

输出：

```text
object_reconstruction/collision/
  collision_000.obj
  collision_001.obj
  ...
  collision_manifest.json
```

优先使用 CoACD；未安装时回退单凸包。凹物体、椅子、把手不应长期使用单凸包。

动态验证：

```text
GMR-master/scripts/simulate_robot_object_contacts.py
```

物理模型：

- 机器人按 GMR qpos 做运动学回放；
- 物体有 free joint、质量、惯量、摩擦和凸碰撞件；
- Sharpa 指节/指尖/掌心碰撞几何与物体建立显式 contact pair；
- `release_frame` 前可用观测位姿初始化；
- `release_frame` 后不再覆盖物体位姿；
- 物体只能通过重力和接触移动。

输出：

```text
object_dynamic_sim.npz
object_dynamic_sim.json
object_dynamic_sim.mp4
```

NPZ：

| 字段 | 说明 |
| --- | --- |
| `position` | 仿真的自由物体位置 |
| `quat_wxyz` | 仿真的自由物体方向 |
| `contact_count` | 每帧手物接触数 |
| `max_normal_force` | 每帧最大接触力 |
| `release_frame` | 释放帧 |

JSON：

| 字段 | 说明 |
| --- | --- |
| `contact_frames_after_release` | 释放后接触帧数 |
| `max_contact_force_n` | 最大接触力 |
| `max_lift_m` | 相对释放高度的最大抬升 |
| `dynamic_lift_success` | 接触期间是否超过抬升阈值 |

`dynamic_lift_success=true` 是当前“不是 mocap 强制跟随”的最低物理门槛。机器人本体
仍是运动学回放；若需要力矩级真实控制，应继续接入 contact-aware IK、MPC 或
MuJoCo Warp 优化。

## 14. 可视化与 2x2

GVHMR 物体合成：

```text
GVHMR-hand/GVHMR-main/tools/pipeline/render_foundationpose_object.py
-> gvhmr_out/<clip>/1_incam_object.mp4
```

GMR 真实 mesh：

```text
GMR-master/scripts/render_robot_motion_headless.py
```

2x2：

```text
GMR-master/scripts/render_composite_2x2.py
-> composite_2x2.mp4
```

四块依次为：

```text
原始视频 | GVHMR 人体、手和物体
PHC      | GMR 机器人、机器人手和物体
```

如果 `1_incam_object.mp4` 存在，2x2 自动优先使用它。

## 15. 完整输出目录

```text
output_root/
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
      *.mp4
      *.json

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
      monocular_adapter/
      object_motion_gvhmr.npz
      object_motion_gmr.npz
      object_manifest.json
      integration_summary.json
      collision/
      object_dynamic_sim.npz
      object_dynamic_sim.json
      object_dynamic_sim.mp4

    unitree_h1_with_hand_sharpa_gvhmr.mp4
    composite_2x2.mp4
```

## 16. 缓存、复现和工程约定

### 16.1 缓存

常用开关：

| 参数 | 作用 |
| --- | --- |
| `SKIP_EXISTING=1` | 命中有效缓存时跳过 |
| `FORCE_WORK_VIDEO=1` | 重建标准化视频 |
| `GVHMR_FORCE_HAND_PREPROCESS=1` | 重跑手部和相关渲染 |
| `FORCE_LOCO=1` | 重跑 Locomotion |
| `FORCE_SMOOTH=1` | 重跑身体平滑 |
| `FORCE_PHC=1` | 重跑 PHC |
| `GMR_OVERRIDE=1` | 重跑 GMR/手 IK/视频/2x2 |

### 16.2 每次实验应保留

```text
使用的 YAML
batch.log
summary.csv
每片段 pipeline.log
.hand_backend
.hand_config
.filter_config
object_manifest.json
collision_manifest.json
动态验证 JSON
```

### 16.3 安全

- 不提交云密钥；
- 已在聊天、源码或 Git 历史中出现的密钥应立即轮换；
- 不加载来源不明的 `.pkl`/`.pt`；
- Hunyuan3D 云 API 可能计费，只能显式启用；
- 动态仿真前先检查物体尺寸、坐标轴和投影。

## 17. 当前 RGB-only 条件下的推荐顺序

```text
1. 先运行 human 阶段，得到 robot_motion.pkl 和手部 NPZ
2. 配置 object.mode: monocular
3. 为每个片段填写 object_name/reference_frame/anchor_hand
4. 完成 do-as-i-do 模型与环境安装
5. 运行 object 阶段
6. 检查 1_incam_object.mp4 的投影
7. 检查 GMR 视频中手物相对位置
8. 设置合理 release_frame
9. 开启 dynamic_validation
10. 只有 dynamic_lift_success 通过后再做控制器/MPC
```

在没有 RGB-D 的情况下，管线不是“不能运行”，而是“尺度从传感器测量改成单目模型
和手部锚定估计”。这个区别必须保留在 manifest、可视化和动态验证中。
