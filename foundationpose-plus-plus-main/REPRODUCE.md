# FoundationPose++ 完整复现部署流程

> **目标**: 基于 Intel RealSense D435 录制的 .db3 数据，完成 FoundationPose++ 的 6D 位姿估计与跟踪，最终输出带 3D BBox 叠加的 mp4 可视化视频。

---

## Docker 部署

本项目的环境非常复杂：`pytorch3d` 和 `nvdiffrast` 必须从源码编译（不在 PyPI 上），`mycpp` C++/CUDA 扩展也需要 cmake 编译。Docker 镜像预装了所有这些依赖，你只需拉取（或构建）即可。

### 前置条件

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA GPU (≥8GB VRAM) |
| NVIDIA 驱动 | ≥525 (本项目主机: 570.211.01) |
| Docker | ≥19.03 (本项目主机: 29.4.1) |
| nvidia-container-runtime | `nvidia-smi` 在容器内可用 |

### A.1 构建镜像

```bash
cd /FoundationPose-plus-plus

# pytorch3d、nvdiffrast、transformers 这两个库需要从源码编译，因此要将源码提前下载到宿主机，直接 COPY 进镜像编译

1） git clone https://github.com/facebookresearch/pytorch3d

2） git clone https://github.com/NVlabs/nvdiffrast

3） 解压以下 transformers 版本至项目文件中（/FoundationPose-plus-plus）： https://github.com/huggingface/transformers/archive/21fac7abba2a37fae86106f87fcf9974fd1e3830.zip 解压完成后将文件夹重命名为 transformers

# 构建（首次约 30-60 分钟，主要耗时在 pytorch3d 和 nvdiffrast 编译）
docker build -t foundationpose-pp:latest .

# 验证
docker run --gpus all --rm foundationpose-pp:latest \
  python -c "
import torch
import pytorch3d
import nvdiffrast.torch as dr
print(f'PyTorch {torch.__version__}  CUDA {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print('All OK')
"
```

> 可调 `MAX_JOBS` 加速：`docker build --build-arg MAX_JOBS=8 -t foundationpose-pp:latest .`

### A.2 下载权重

权重文件太大（Qwen2-VL-7B 约 14GB），不嵌入镜像，通过 volume 挂载：

```bash
# 1. FoundationPose 权重 → FoundationPose/weights/
#    Google Drive: https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i
#    下载 refiner (2023-10-28-18-33-37) + scorer (2024-01-11-20-02-45) 两个文件夹

# 2. SAM-HQ 权重 → sam-hq/pretrained_checkpoints/
#    https://drive.google.com/file/d/1Uk17tDKX1YAKas5knI4y9ZJCo0lRVL0G/view
#    下载 sam_hq_vit_h.pth

# 3. Qwen2-VL 权重 → Qwen2-VL/weights/
#    https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct

# 4. Cutie 权重 (容器内运行一次自动下载)
```

### A.3 构建并运行容器

```bash
cd /FoundationPose-plus-plus

# 删除旧容器（如果存在）
docker rm -f fpp 2>/dev/null

# 启动容器（后台运行）
# ★ --network host: 容器共享宿主机网络栈，localhost:9002/9003 直接访问宿主机 API
docker run --gpus 1 -itd --name fpp \
  --network host \
  -v $(pwd)/FoundationPose/weights:/workspace/FoundationPose/weights \
  -v $(pwd)/Cutie/weights:/workspace/Cutie/weights \
  -v $(pwd)/realsense_data:/workspace/realsense_data \
  -v $(pwd)/test_data:/workspace/test_data \
  -v $(pwd)/src:/workspace/src \
  -v $(pwd)/scripts:/workspace/scripts \
  foundationpose-pp:latest
```

### A.4 宿主机启动 SAM-HQ 分割 API

物体检测使用远程 Qwen3-VL-30B 服务（`192.168.10.242:12067`），无需本地部署。只需启动 SAM-HQ：

#### A.4.1 安装依赖（只做一次）

```bash
cd /FoundationPose-plus-plus
bash scripts/setup_host_apis.sh  # 会同时创建 hq_sam 和 qwen2_vl 环境（后者已不需要但无害）
```

#### A.4.2 启动服务

```bash
cd /FoundationPose-plus-plus
mkdir -p logs

# 启动 SAM-HQ API (端口 9002)
nohup conda run -n hq_sam --no-capture-output \
  python src/WebAPI/hq_sam_api.py \
  > logs/hq_sam.log 2>&1 &
echo $! > logs/hq_sam.pid

# 等待就绪
while true; do
  if curl -s http://localhost:9002/docs > /dev/null 2>&1; then
    echo "✅ SAM-HQ (9002) 已就绪"; break
  fi
  sleep 2
done
```

> **停止**：`kill $(cat logs/hq_sam.pid)`

### A.5 容器内验证

```bash
docker exec -it fpp bash

# 进入容器后
python -c "
import torch
import pytorch3d
import nvdiffrast.torch as dr
from FoundationPose.estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
print('FoundationPose++ 环境完整!')
"

# 下载 Cutie 权重（首次）
python Cutie/cutie/utils/download_models.py
```

### A.6 从容器内继续复现流程
docker exec -it fpp bash
```

---

## 第 4 步：从 .db3 录制数据提取帧（包括完成深度对齐）

### 数据格式说明

`.db3` 文件是 ROS2 的 SQLite 格式 bag 文件。内部包含两个图像 topic：

| Topic | 分辨率 | 编码 | 内容 |
|-------|--------|------|------|
| `/device_0/sensor_1/Color_0/image/data` | 640×480 | rgb8 | 彩色图像 |
| `/device_0/sensor_0/Depth_0/image/data` | 848×480 | mono16 | 深度图像 (uint16 mm) |

**关键**: D435 的彩色图像与深度图像**原始分辨率不同、视场不同、未对齐**。
彩色图是 640×480, 深度图是 848×480, 两者使用不同的内参且存在视差（基线约 15mm）。

### 深度对齐算法原理

对齐过程模拟 `pyrealsense2` 的 `rs.align(rs.stream.color)` 操作：

```bash
对每个深度像素 (u_d, v_d, z):
  1. 反投影到深度相机 3D 坐标系:
     P_d = z * K_depth^{-1} * [u_d, v_d, 1]^T

  2. 用出厂外参 T_depth→color 变换到彩色相机坐标系:
     P_c = R_depth2color * P_d + t_depth2color

  3. 用彩色内参投影到彩色图像平面:
     [u_c, v_c, w] = K_color * P_c
     u_c /= w, v_c /= w

  4. 将深度值 z 填入输出图像 (v_c, u_c) 位置
     (如果多个深度像素映射到同一彩色像素, 取最近深度)
```

**D435 出厂外参 T_depth→color**：
- 旋转：近似单位矩阵 (红外相机与彩色相机光轴平行)
- 平移：`[~0.015, 0, 0]`（米），深度传感器在彩色传感器右侧约 15mm

### 运行提取脚本

```bash
cd /FoundationPose-plus-plus

# 确保脚本所在目录存在
mkdir -p scripts

# 对每个 .db3 文件运行提取（物体名称自动从文件名解析）
python scripts/extract_realsense_bag.py --db3_path realsense_data/mouse.db3
python scripts/extract_realsense_bag.py --db3_path realsense_data/notebook.db3
python scripts/extract_realsense_bag.py --db3_path realsense_data/phone.db3
```

**输出目录结构**：
```
test_data/mouse/
├── cam_K.json              # 彩色相机内参 (对齐后深度图共用此内参)
├── depth_scale.json         # 深度缩放因子
├── color/                   # RGB 图像序列
│   ├── 0.png
│   ├── 1.png
│   ├── ...
│   └── N.png
└── depth/                   # 对齐后的深度图序列 (uint16 毫米)
    ├── 0.png
    ├── 1.png
    ├── ...
    └── N.png
```

### 提取脚本关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--db3_path` | 必填 | .db3 文件路径 |
| `--output_dir` | 必填 | 输出目录 |
| `--max_frames` | None | 限制提取帧数（调试用），不传则全量提取 |
| `--depth_scale_factor` | 0.001 | 深度值缩放，D435 默认为 0.001（uint16 → 米） |
| `--disable_alignment` | False | 禁用深度对齐（直接输出原始深度，不推荐） |

---

## 第 5 步：目标物体 Mesh 准备

FoundationPose 需要一个 .obj 或 .stl 格式的 3D 模型文件。你有以下几种选择：

### 选项 A：使用已有 CAD 模型

如果你有物体的 CAD 模型（.step/.stl 等），直接放入对应测试目录：

```bash
mkdir -p test_data/mouse/mesh
cp /path/to/mouse_model.obj test_data/mouse/mesh/
```

### 选项 B：使用 Hunyuan3D 自动生成（需要额外配置）

如果使用 foundation_pose 项目的 Hunyuan3D API 服务（端口 18081），可以自动从 RGB 图像生成 mesh。

### 选项 C：手动制作简单代理 Mesh

对于纯跟踪测试，可以用简单的长方体/圆柱体近似：

```python
import trimesh
# 生成长方体代理 mesh
box = trimesh.creation.box(extents=[0.1, 0.04, 0.06])  # 宽高深(米)
box.export("test_data/mouse/mesh/proxy.obj")
```

### Mesh 注意事项
- 文件格式：`.obj` 或 `.stl`
- 单位：建议使用**米 (meters)**
- FoundationPose 内部会自动通过 `--apply_scale` 缩放（默认 0.01）
- 如果 mesh 没有纹理，需要用 `--force_apply_color` 强制添加颜色

---

## 第 6 步：初始帧 Mask 生成

FoundationPose 的 register（初始位姿估计）需要一个初始 mask。这里用 **Qwen2-VL + SAM-HQ** 的二阶段方法。

### 6.1 启动 Web API 服务

API 服务已在宿主机上启动（见 [A.4 节](#a4-在宿主机启动-qwen2-vl--sam-hq-api)）。如果尚未启动或已停止，重新执行 A.4 中的命令即可。

确认它们仍在运行：

```bash
curl -s http://localhost:9002/docs > /dev/null && echo "SAM-HQ OK" || echo "SAM-HQ 未运行!"
curl -s http://localhost:9003/docs > /dev/null && echo "Qwen2-VL OK" || echo "Qwen2-VL 未运行!"
```

### 6.2 用 Qwen2-VL 检测目标物体 BBox

```bash
cd /FoundationPose-plus-plus

# 对每个物体检测 bbox，用中文描述效果更好
BBOX_MOUSE=$(python src/utils/obj_bbox.py \
  --frame_path test_data/mouse/color/0.png \
  --visualize_path test_data/mouse/0_bbox.png \
  --object_name "鼠标")

BBOX_NOTEBOOK=$(python src/utils/obj_bbox.py \
  --frame_path test_data/notebook/color/0.png \
  --visualize_path test_data/notebook/0_bbox.png \
  --object_name "笔记本")

BBOX_PHONE=$(python src/utils/obj_bbox.py \
  --frame_path test_data/phone/color/0.png \
  --visualize_path test_data/phone/0_bbox.png \
  --object_name "手机")
```

> **如果 Qwen2-VL 检测不准**：
> 1. 尝试用中文更详细描述物体（如"银色的苹果鼠标"而非"鼠标"）
> 2. 提供参考图片 `--reference_img_path`
> 3. 直接手动指定 bbox（跳过 Qwen2-VL）：
>    ```bash
>    BBOX_MOUSE="[x, y, w, h]"
>    ```

### 6.3 用 SAM-HQ 生成精细 Mask

```bash
cd /FoundationPose-plus-plus

python src/utils/obj_mask.py \
  --frame_path test_data/mouse/color/0.png \
  --bbox_xywh "$BBOX_MOUSE" \
  --output_mask_path test_data/mouse/0_mask.png

python src/utils/obj_mask.py \
  --frame_path test_data/notebook/color/0.png \
  --bbox_xywh "$BBOX_NOTEBOOK" \
  --output_mask_path test_data/notebook/0_mask.png

python src/utils/obj_mask.py \
  --frame_path test_data/phone/color/0.png \
  --bbox_xywh "$BBOX_PHONE" \
  --output_mask_path test_data/phone/0_mask.png
```

---

## 第 7 步：运行 6D 位姿跟踪推理

### 7.1 读取 cam_K

提取脚本已将相机内参保存为 JSON，读取方式：

```bash
cd /FoundationPose-plus-plus
CAM_K=$(python -c "import json; k=json.load(open('test_data/mouse/cam_K.json')); print(json.dumps(k))")
echo $CAM_K
```

### 7.2 运行跟踪（单个物体）

```bash
cd /FoundationPose-plus-plus

python src/obj_pose_track.py \
  --rgb_seq_path test_data/mouse/color \
  --depth_seq_path test_data/mouse/depth \
  --mesh_path test_data/mouse/mesh/mouse.obj \
  --init_mask_path test_data/mouse/0_mask.png \
  --pose_output_path test_data/mouse/pose.npy \
  --mask_visualization_path test_data/mouse/mask_vis \
  --bbox_visualization_path test_data/mouse/bbox_vis \
  --pose_visualization_path test_data/mouse/pose_vis \
  --cam_K "$CAM_K" \
  --activate_2d_tracker \
  --activate_kalman_filter \
  --kf_measurement_noise_scale 0.05 \
  --apply_scale 0.01 \
  --force_apply_color \
  --est_refine_iter 10 \
  --track_refine_iter 5
```

### 7.3 关键参数说明

| 参数 | 作用 | 推荐值 |
|------|------|--------|
| `--activate_2d_tracker` | 启用 Cutie 2D 跟踪器，修正 x/y 平移 | 开启 |
| `--activate_kalman_filter` | 启用 12D 卡尔曼滤波器，平滑旋转 | 开启 |
| `--kf_measurement_noise_scale` | 滤波强度，越大越平滑但越延迟 | 0.05 |
| `--est_refine_iter` | 首帧 refine 迭代次数，越多越精确 | 10 |
| `--track_refine_iter` | 跟踪每帧 refine 迭代次数 | 3~5 |
| `--apply_scale` | Mesh 单位缩放 (100 units → 1m 则填 0.01) | 0.01 |
| `--force_apply_color` | 为无纹理 mesh 强制添加纯色（跟踪精度提升） | 建议开启 |

### 7.4 批量运行所有物体

```bash
# 使用一键 pipeline 脚本 (自动扫描 realsense_data/ 下所有 .db3 文件)
python scripts/run_pipeline.py \
  --data_dir test_data \
  --activate_2d_tracker \
  --activate_kalman_filter
```

---

## 第 8 步：生成可视化视频

推理完成后，`pose_vis/` 目录下已有逐帧标注图，将其合成为 mp4 视频：

```bash
cd /FoundationPose-plus-plus

python scripts/generate_video.py \
  --frame_dir test_data/mouse/pose_vis \
  --output test_data/mouse/mouse_pose.mp4 \
  --fps 30

python scripts/generate_video.py \
  --frame_dir test_data/notebook/pose_vis \
  --output test_data/notebook/notebook_pose.mp4 \
  --fps 30

python scripts/generate_video.py \
  --frame_dir test_data/phone/pose_vis \
  --output test_data/phone/phone_pose.mp4 \
  --fps 30
```

**生成视频包含**：
- 原始 RGB 图像
- 绿色 3D BBox（物体 6D 位姿投影）
- 红/绿/蓝坐标轴 (X/Y/Z)
- 帧号等信息

---

## 第 9 步：一键运行完整 Pipeline

对于所有物体的端到端处理，使用 `scripts/run_pipeline.py`：

```bash
cd /data2/zhenghao26/FoundationPose-plus-plus

# 完整 pipeline (自动扫描 realsense_data/ 下所有 .db3 文件)
python scripts/run_pipeline.py \
  --data_dir test_data \
  --activate_2d_tracker \
  --activate_kalman_filter
```

---

## 输出文件说明

完成所有步骤后，每个物体的输出目录结构：

```
test_data/mouse/
├── cam_K.json              # 相机内参
├── depth_scale.json         # 深度缩放
├── color/                   # RGB 图像序列
│   ├── 0.png
│   └── ...
├── depth/                   # 对齐后深度图 (uint16 mm)
│   ├── 0.png
│   └── ...
├── mesh/                    # 物体 3D 模型
│   └── mouse.obj
├── 0_mask.png               # 初始帧 mask
├── 0_bbox.png               # 初始帧 bbox 可视化
├── pose.npy                  # 每帧的 6D 位姿 (N×4×4)
├── mask_vis/                # 2D 跟踪 mask 可视化
├── bbox_vis/                # 2D 跟踪 bbox 可视化
├── pose_vis/                # 3D BBox + 坐标轴可视化帧
└── mouse_pose.mp4           # 最终视频
```

---

## 故障排除

### 编译问题

| 问题 | 解决方案 |
|------|----------|
| `nvcc` 版本不匹配 | 确保 CUDA 12.1：`export PATH=/usr/local/cuda-12.1/bin:$PATH` |
| PyTorch CUDA 不匹配 | `python -c "import torch; print(torch.version.cuda)"` 应为 12.1 |
| kaolin 编译失败 | 如不需要 model-free 模式，跳过 `bundlesdf/mycuda` 编译 |
| `RasterizeCudaContext` 错误 | CUDA 驱动版本需 ≥ nvcc 版本 |

### 推理问题

| 问题 | 可能原因 | 解决 |
|------|----------|------|
| 首帧 register 失败 | mask 质量差 / mesh 与实际物体差异大 | 改善 mask / 使用更精确 mesh |
| 跟踪漂移 | 2D 跟踪器丢失 | 检查 bbox_vis 确认 Cutie 跟踪是否正常 |
| 深度缺失导致位姿跳动 | D435 红外在反光/透明表面失效 | 调整物体角度避免反光 |
| `apply_scale` 不合适 | mesh 单位不明确 | 检查 mesh 实际尺寸，调整 scale |

### 提取问题

| 问题 | 可能原因 | 解决 |
|------|----------|------|
| rosbags 安装失败 | Python 版本不兼容 | 使用 Python 3.10-3.12 |
| .db3 解析异常 | 录制中断 / 文件损坏 | 检查 .db3 完整性 |

---

## 依赖版本参考

以下是在 RTX 3090 / RTX 4090 + Ubuntu 20.04/22.04 上验证过的版本组合：

```
python==3.10
torch==2.4.1
torchvision==0.19.1
CUDA==12.1
numpy<2.0
opencv-python==4.10
trimesh==4.4
pyrender==0.1.45
pytorch3d==0.7.7
nvdiffrast==0.3.2
imageio==2.35
imageio-ffmpeg==0.5
fastapi==0.115
uvicorn==0.30
rosbags>=0.9
```

---

## 参考链接

- FoundationPose 原始项目: https://github.com/NVlabs/FoundationPose
- FoundationPose 使用手册: https://github.com/030422Lee/FoundationPose_manual
- FoundationPose++ 仓库: https://github.com/teal024/FoundationPose-plus-plus
- Cutie: https://github.com/hkchengrex/Cutie
- SAM-HQ: https://github.com/SysCV/sam-hq
- Qwen2-VL: https://github.com/QwenLM/Qwen2-VL
