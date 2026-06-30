# CLAUDE.md - FoundationPose++

## 项目概述

FoundationPose++ 是一个面向高动态场景的实时 6D 位姿跟踪器。项目基于 NVIDIA 的 [FoundationPose](https://github.com/NVlabs/FoundationPose)（CVPR 2024 Highlight），在其基础上增加了三个核心模块：**2D 跟踪器 + 卡尔曼滤波器 + Amodal Completion**（Amodal Completion 仍在开发中）。

### 核心思路

FoundationPose 原版的跟踪是"伪跟踪"——每帧用上一帧的 6D 位姿作为优化的初始解。在高动态场景中表现不佳。FoundationPose++ 将 6 自由度拆解：

| 自由度 | 方法 |
|--------|------|
| x, y（平移） | 2D 跟踪器（默认 Cutie，也可用 OSTrack/Samurai） |
| z（深度） | 直接从深度图在 (x, y) 处取值 |
| roll, pitch, yaw（旋转） | 卡尔曼滤波器（KalmanFilter6D） |

这是一个简单但有效的工程化改进，在保持实时性（>30 FPS on RTX 3090）的同时大幅提升动态场景下的跟踪鲁棒性。

## 目录结构

```
FoundationPose-plus-plus/
├── FoundationPose/          # 原始 FoundationPose 代码（NVIDIA，已修改）
│   ├── estimater.py         # 核心类：FoundationPose, ScorePredictor, PoseRefinePredictor
│   ├── Utils.py             # 大量工具函数（渲染、几何变换、深度处理等）
│   ├── datareader.py        # 数据读取器（YCB-Video, LINEMOD 等数据集）
│   ├── run_demo.py          # FoundationPose 原始 demo 入口
│   ├── run_ycb_video.py     # YCB-Video 数据集评测脚本
│   ├── run_linemod.py       # LINEMOD 数据集评测脚本
│   ├── offscreen_renderer.py # 离屏渲染器
│   ├── learning/            # 训练相关
│   │   ├── models/          # refine_network, score_network, network_modules
│   │   ├── training/        # predict_pose_refine, predict_score, training_config
│   │   └── datasets/        # pose_dataset, h5_dataset
│   ├── bundlesdf/           # BundleSDF 神经隐式表示（model-free 模式）
│   │   ├── run_nerf.py      # 训练 Neural Object Field
│   │   └── nerf_runner.py   # NeRF 运行器
│   ├── mycpp/               # C++/CUDA 扩展（PyBind11）
│   │   ├── src/Utils.cpp
│   │   ├── src/app/pybind_api.cpp
│   │   ├── include/Utils.h
│   │   └── CMakeLists.txt
│   ├── build_all.sh         # 构建脚本（mycpp + kaolin + mycuda）
│   ├── build_all_conda.sh   # Conda 环境构建脚本
│   └── weights/             # 预训练权重目录（需从 Google Drive 下载）
│
├── src/                     # ★ FoundationPose++ 新增核心代码
│   ├── obj_pose_track.py    # ★ 主入口：6D 位姿跟踪主流程
│   ├── VOT.py               # 2D 跟踪器封装（Tracker_2D 基类 + Cutie 实现）
│   ├── WebAPI/              # 模型推理 Web API（基于 FastAPI）
│   │   ├── qwen2_vl_api.py  # Qwen2-VL API：根据文本描述检测物体边界框
│   │   ├── hq_sam_api.py    # SAM-HQ API：根据边界框生成精细分割 mask
│   │   └── cutie_api.py     # Cutie API：基于 HDF5 的批量视频分割
│   └── utils/
│       ├── __init__.py      # 导出 visualize_mask, visualize_bbox
│       ├── kalman_filter_6d.py  # ★ 6D 卡尔曼滤波器（12维状态空间）
│       ├── obj_bbox.py      # Qwen2-VL 边界框获取工具
│       ├── obj_mask.py      # SAM-HQ mask 生成请求工具
│       └── visualization.py # 可视化工具（mask 叠加、bbox 绘制）
│
├── Cutie/                   # Cutie 2D 视频分割跟踪器子模块
│   ├── cutie/               # 核心代码
│   │   ├── model/           # 模型定义（cutie.py, modules.py, transformer/...）
│   │   ├── inference/       # 推理引擎（inference_core.py, memory_manager.py...）
│   │   ├── dataset/         # 训练数据集
│   │   ├── utils/           # 工具函数
│   │   └── config/          # YAML 配置文件（model, data, train, eval）
│   ├── gui/                 # GUI 交互分割工具
│   ├── scripts/             # 数据处理脚本
│   ├── interactive_demo.py  # 交互式 demo
│   └── scripting_demo.py   # 脚本化 demo
│
├── sam-hq/                  # SAM-HQ 高质量分割子模块
│   ├── segment_anything/    # SAM 核心代码
│   │   ├── modeling/        # 模型结构（image_encoder, mask_decoder_hq, transformer...）
│   │   ├── build_sam.py     # 构建 SAM 模型
│   │   ├── predictor.py     # SAM 预测器
│   │   └── utils/           # 工具函数
│   ├── seginw/              # SegInW 分割测试 + GroundingDINO
│   ├── train/               # SAM 训练代码
│   └── pretrained_checkpoints/  # 预训练权重目录
│
├── Qwen2-VL/                # Qwen2-VL 视觉语言模型子模块
│   └── quick_start.py       # 快速开始脚本
│
├── .figs/                   # README 中的示意图
├── README.md                # 项目说明
├── Install.md               # 环境安装指南
├── CITATION.cff             # 引用信息
└── .gitignore
```

## 核心架构与数据流

### 主流程（`src/obj_pose_track.py` → `pose_track()`）

```
第1帧（初始化）:
  1. 读取初始 mask（init_mask）
  2. FoundationPose.register() → 用 RefineNetwork + ScoreNetwork 求解首帧 6D 位姿
  3. 初始化 2D 跟踪器（Cutie）和卡尔曼滤波器

后续帧（跟踪）:
  1. 2D 跟踪器（Cutie.track()）→ 获取当前帧物体 bbox
  2. adjust_pose_to_image_point() → 用 bbox 中心修正上一帧位姿的 x, y
  3. [可选] KalmanFilter6D → 更新/预测/融合位姿估计
  4. FoundationPose.track_one() → 用修正后的位姿运行一次 RefineNetwork 得到最终位姿
  5. 可视化 + 保存
```

### 卡尔曼滤波器（`src/utils/kalman_filter_6d.py`）

- **状态空间**（12维）：`[tx, ty, tz, rx, ry, rz, v_tx, v_ty, v_tz, v_rx, v_ry, v_rz]`
- **两种观测更新**：
  - `update()`：用 FoundationPose 完整 6D 位姿更新
  - `update_from_xy()`：用 2D 跟踪器的 (x, y) 测量值更新（精度更高）
- **关键参数**：`measurement_noise_scale`（默认 0.05），值越大滤波越强

### 2D 跟踪器封装（`src/VOT.py`）

- `Tracker_2D`：基类，`initialize()` 和 `track()` 返回 `[-1, -1, 0, 0]`（不跟踪）
- `Cutie(Tracker_2D)`：使用 Cutie 模型进行视频分割跟踪
  - `initialize()`：用初始 mask 初始化 Cutie，返回初始 bbox
  - `track()`：逐帧跟踪，对输出 mask 做腐蚀（erosion）后计算 bbox
  - 关键参数：`cutie_seg_threshold`（默认 0.1），`erosion_size`（默认 5）

### Web API 架构

项目使用 **FastAPI** 将大模型封装为 Web 服务，避免主进程中重复加载模型：

| API | 端口 | 功能 |
|-----|------|------|
| `qwen2_vl_api.py` | 9003 | Qwen2-VL-7B：文本描述 → 物体边界框 |
| `hq_sam_api.py` | 9002 | SAM-HQ：边界框 → 精细分割 mask |
| `cutie_api.py` | 8000 | Cutie：视频帧序列 → 分割 mask 序列（HDF5） |

### FoundationPose 核心类（`FoundationPose/estimater.py`）

- **`FoundationPose`**：主类
  - `register()`：首帧位姿初始化——随机生成位姿假设 → Refine → Score → 选最高分
  - `track_one()`：逐帧跟踪——用上一帧位姿运行一次 Refine
  - `pose_last`：存储上一帧的位姿（供跟踪模式使用）
  - `track_one_w_spec_last_pose()`：用指定的初始位姿进行跟踪
- **`ScorePredictor`**：评分网络，对多个位姿假设打分排序
- **`PoseRefinePredictor`**：位姿精炼网络，迭代优化 6D 位姿
- **`dr.RasterizeCudaContext()`**：NVDiffRast 的 CUDA 光栅化上下文

## 常用命令

### 构建环境

```bash
# 使用提供的 Docker 镜像
docker pull shingarey/foundationpose_custom_cuda121:latest

# 构建 FoundationPose（C++/CUDA 扩展）
cd FoundationPose
bash build_all.sh
```

### 下载权重

```bash
# FoundationPose 权重 → FoundationPose/weights/
# (Google Drive: https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i)
#   - refiner: 2023-10-28-18-33-37
#   - scorer: 2024-01-11-20-02-45

# Qwen2-VL 权重 → Qwen2-VL/weights/
# (HuggingFace: Qwen/Qwen2-VL-7B-Instruct)

# SAM-HQ 权重 → sam-hq/pretrained_checkpoints/
# (Google Drive: sam_hq_vit_h.pth)

# Cutie 权重（自动下载）
python Cutie/cutie/utils/download_models.py
```

### 运行 Demo

```bash
export PROJECT_ROOT=/path/to/FoundationPose-plus-plus
export TESTCASE="lego_20fps"

python src/obj_pose_track.py \
  --rgb_seq_path $PROJECT_ROOT/$TESTCASE/color \
  --depth_seq_path $PROJECT_ROOT/$TESTCASE/depth \
  --mesh_path $PROJECT_ROOT/$TESTCASE/mesh/1x4.stl \
  --init_mask_path $PROJECT_ROOT/$TESTCASE/0_mask.png \
  --pose_output_path $PROJECT_ROOT/$TESTCASE/pose.npy \
  --mask_visualization_path $PROJECT_ROOT/$TESTCASE/mask_visualization \
  --bbox_visualization_path $PROJECT_ROOT/$TESTCASE/bbox_visualization \
  --pose_visualization_path $PROJECT_ROOT/$TESTCASE/pose_visualization \
  --cam_K "[[426.8704833984375, 0.0, 423.89471435546875], [0.0, 426.4277648925781, 243.5056915283203], [0.0, 0.0, 1.0]]" \
  --activate_2d_tracker \
  --activate_kalman_filter \
  --apply_scale 0.01
```

### 启动 Web API

```bash
# 启动 Qwen2-VL API（物体检测）
python src/WebAPI/qwen2_vl_api.py --weight_path Qwen2-VL/weights &

# 启动 SAM-HQ API（mask 生成）
python src/WebAPI/hq_sam_api.py --checkpoint_path sam-hq/pretrained_checkpoints/sam_hq_vit_h.pth &

# 获取初始 bbox
python src/utils/obj_bbox.py \
  --frame_path $TESTCASE/color/0.png \
  --visualize_path $TESTCASE/0_bbox.png \
  --object_name "蓝色乐高积木"

# 生成初始 mask
python src/utils/obj_mask.py \
  --frame_path $TESTCASE/color/0.png \
  --bbox_xywh "$BOUNDING_BOX_POSITION" \
  --output_mask_path $TESTCASE/0_mask.png
```

### 数据格式要求

```
$TESTCASE/
├── color/       # RGB 图像序列 (0.png, 1.png, ...)
├── depth/       # 深度图序列 (0.png, 1.png, ...)（单位：毫米）
└── mesh/        # 物体 mesh 文件 (.obj / .stl)
```

## 关键技术细节

### 位姿坐标系
- FoundationPose 内部使用**中心化 mesh**（mesh 顶点减去 model_center）
- 最终输出的位姿通过 `get_tf_to_centered_mesh()` 转换回原始 mesh 坐标系
- `pose_last` 是相对于中心化 mesh 的位姿

### 关键参数说明
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `est_refine_iter` | 10 | 首帧初始化时 Refine 网络迭代次数 |
| `track_refine_iter` | 5 | 跟踪阶段 Refine 网络迭代次数 |
| `apply_scale` | 0.01 | Mesh 缩放因子（米为单位，通常 0.01） |
| `activate_2d_tracker` | False | 启用 2D 跟踪器 |
| `activate_kalman_filter` | False | 启用卡尔曼滤波器 |
| `kf_measurement_noise_scale` | 0.05 | 卡尔曼滤波器测量噪声比例 |

### 2D 跟踪器选择
- **默认**：Cutie（视频分割跟踪器，质量高但稍慢）
- **可替换**：OSTrack（>100 FPS on 3090），Samurai 等
- 切换方式：在 `VOT.py` 中新增 `Tracker_2D` 的子类

### Amodal Completion（开发中）
- 在 `VOT.py` 中预留了 `# TODO: get occluded mask` 的注释
- 目标：提升遮挡场景下的鲁棒性
- 当前存在兼容性和性能问题

## 故障排除参考

1. **新 GPU（如 4090）兼容性问题**：参考 [FoundationPose #27](https://github.com/NVlabs/FoundationPose/issues/27)
2. **不合理的结果**：参考 [FoundationPose #44](https://github.com/NVlabs/FoundationPose/issues/44#issuecomment-2048141043)
3. **Windows 环境搭建**：参考 [FoundationPose #148](https://github.com/NVlabs/FoundationPose/issues/148)
4. **Qwen2-VL 检测结果不理想**：尝试提供参考图片（`--reference_img_path`），或用中文描述物体名称

## 开发注意事项

- Python 版本：3.9+（推荐匹配 Docker 镜像中的版本）
- CUDA 版本：12.1（Docker 镜像预装）
- 大部分模型推理必须是 **GPU（CUDA）**，无法在 CPU 上运行
- FoundationPose 子模块包含 NVIDIA 专有代码（NVIDIA Source Code License），FoundationPose++ 新增代码使用 MIT License
- `FoundationPose/estimater.py` 中的 `FoundationPose.__init__()` 默认使用 `device='cuda'`，`glctx` 使用 `RasterizeCudaContext`
- `FoundationPose/mycpp/` 需要编译（CMake），`bundlesdf/mycuda/` 也需要编译
- 默认情况下 FoundationPose 的 `build_all.sh` 会重新编译 kaolin 和 mycuda
- 如果不需要 model-free 模式，kaolin 是可选的
- `scene_pts` 在 register 中未被使用，`compute_add_err_to_gt_pose` 始终返回 -1（因为在线跟踪无 gt 位姿）
