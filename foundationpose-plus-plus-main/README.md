# FoundationPose++

6D 位姿跟踪 pipeline：从 RGB-D 录制数据（`.db3`）到 3D BBox 可视化视频（`.mp4`）的端到端解决方案。

## 架构

```
┌─────────────────────────────────────────────────────────┐
│ 远程服务（共享，无需部署）                                  │
│   Qwen3-VL-30B (192.168.10.242:12067)  物体检测          │
│   Hunyuan3D 腾讯云 / 本地 :18081（可选）    mesh 生成        │
└─────────────────────────────────────────────────────────┘
        ▲                              ▲
        │  OpenAI 兼容 API              │  HTTP
        │                              │
┌───────┴──────────────────────────────┴──────────────────┐
│ 宿主机（需部署）                                           │
│   SAM-HQ API (:9002)                  分割              │
│   Docker 容器 (--network host) → FoundationPose + Cutie  │
└─────────────────────────────────────────────────────────┘
```

**需要用到的 GPU 显存**：仅 FoundationPose + Cutie（容器内，~10GB）。物体检测走远程服务，零本地开销。

## 前置条件

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA GPU（≥8GB VRAM） |
| NVIDIA 驱动 | ≥525 |
| Docker | ≥19.03 + nvidia-container-runtime |

## 快速开始

### 1. 拉取 Docker 镜像

```bash
docker pull <registry>/foundationpose-pp:latest
```

> 如需自行构建（首次约 30-40 min）：
> ```bash
> git clone https://github.com/facebookresearch/pytorch3d
> git clone https://github.com/NVlabs/nvdiffrast
> # transformers 21fac7ab 也需放入项目根目录
> docker build -t foundationpose-pp:latest .
> ```

### 2. 拉取代码仓库

```bash
git clone <repo-url> && cd FoundationPose-plus-plus
```

### 3. 准备权重文件

需手动下载的权重约 **3.5 GB**：

```
FoundationPose-plus-plus/
├── FoundationPose/weights/          # FoundationPose 位姿估计 (~500MB)
│   ├── 2023-10-28-18-33-37/         #   refiner 权重
│   └── 2024-01-11-20-02-45/         #   scorer 权重
├── sam-hq/pretrained_checkpoints/   # SAM-HQ 分割 (~2.5GB)
│   └── sam_hq_vit_l.pth
└── Cutie/weights/                   # Cutie 2D 跟踪 (~1GB, 容器内下载)
    ├── coco_lvis_h18_itermask.pth
    └── cutie-base-mega.pth
```

| 模型 | 下载地址 | 存放路径 | 大小 |
|------|---------|---------|------|
| **FoundationPose** refiner + scorer | [Google Drive](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i) | `FoundationPose/weights/` | ~500MB |
| **SAM-HQ** | [Google Drive](https://drive.google.com/file/d/1Uk17tDKX1YAKas5knI4y9ZJCo0lRVL0G/view) | `sam-hq/pretrained_checkpoints/sam_hq_vit_l.pth` | ~2.5GB |
| **Cutie** | 容器内运行 `python Cutie/cutie/utils/download_models.py` | `Cutie/weights/` | ~1GB |

> **物体检测（Qwen3-VL-30B）已部署为远程共享服务**（`192.168.10.242:12067`），无需本地下载模型或启动服务。

### 4. 准备数据

```bash
# 录制数据放入 realsense_data/，文件名即物体名
realsense_data/
├── mouse.db3
├── notebook.db3
└── phone.db3
```

**Mesh 来源**，二选一：

| 模式 | 做法 | 需要什么 |
|------|------|---------|
| **手动 mesh** | 将 `.obj` / `.stl` 放入 `test_data/<物体>/mesh/` | 提前准备 CAD 模型 |
| **Hunyuan3D 自动生成** | Pipeline 调用腾讯云或本地服务 | 腾讯云凭证，或已部署的 :18081 服务 |

### 5. 启动 SAM-HQ 分割 API

SAM-HQ 需要在宿主机上运行（端口 9002）：

```bash
cd /FoundationPose-plus-plus
mkdir -p logs

# 安装依赖（首次）
bash scripts/setup_host_apis.sh

# 启动服务
nohup conda run -n hq_sam --no-capture-output \
  python src/WebAPI/hq_sam_api.py \
  > logs/hq_sam.log 2>&1 &
echo $! > logs/hq_sam.pid

# 等待就绪（几秒）
curl -s http://localhost:9002/docs > /dev/null && echo "SAM-HQ OK" || echo "SAM-HQ FAIL"
```

> **停止**：`kill $(cat logs/hq_sam.pid)`

### 6. 启动容器

```bash
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

> `--network host` 让容器能通过 `localhost` 访问宿主机 API（SAM-HQ :9002, Hunyuan3D :18081）和远程检测服务。

### 7. 运行 Pipeline

```bash
# 进入容器
docker exec -it fpp bash

# 手动 mesh 模式（你提前放好了 .obj 文件）
python scripts/run_pipeline.py \
  --activate_2d_tracker \
  --activate_kalman_filter

# Hunyuan3D 自动 mesh 模式（无需手动准备 mesh）
export TENCENTCLOUD_SECRET_ID=...
export TENCENTCLOUD_SECRET_KEY=...
python scripts/run_pipeline.py \
  --use_hunyuan3d \
  --hunyuan3d_backend tencent \
  --activate_2d_tracker \
  --activate_kalman_filter \
  --object xxx
```

腾讯云凭证只允许通过环境变量传入，不要写进脚本或提交到 Git。使用本地服务时改为
`--hunyuan3d_backend service --hunyuan3d_url http://localhost:18081`。

自动扫描 `realsense_data/*.db3`，无需手动指定物体名。最终视频：`test_data/<物体>/<物体>_pose.mp4`。

## Pipeline 工作流

```
Step 1  提取帧       extract_realsense_bag.py    .db3 → color/ + depth/ (含深度→彩色对齐)
Step 2  BBox 检测    远程 Qwen3-VL-30B           RGB → (x,y,w,h)
Step 3  Mask 分割    SAM-HQ (:9002)              BBox → 0_mask.png + 0_mask_rgba.png
Step 4  尺寸计算     compute_real_dims.py         mask × depth → (W,H,D) 米
Step 5  Mesh 生成    Hunyuan3D（腾讯云/本地）      RGBA → mesh.obj (含纹理)
Step 6  位姿跟踪     FoundationPose + Cutie      帧序列 → pose.npy + 实时视频
Step 7  视频合成     (跟踪时已实时编码)            pose_vis/ → .mp4
```

**缓存机制**：同一物体 mesh 已生成且相机内参未变化（K 矩阵差异 < 1.0）时，自动跳过 Step 2-5，直接跟踪。

## 分步执行

```bash
# 跳过已完成的提取
python scripts/run_pipeline.py --skip_extraction

# 跳过提取 + mask 生成（用已有 0_mask.png）
python scripts/run_pipeline.py --skip_extraction --skip_mask

# 只跑提取
python scripts/extract_realsense_bag.py --db3_path realsense_data/mouse.db3
```

## 输出产物

```
test_data/<物体>/
├── color/                # RGB 帧序列
├── depth/                # 对齐后深度图序列
├── cam_K.json            # 相机内参
├── mesh/                 # 物体 3D 模型
├── 0_mask.png            # 初始帧分割 mask
├── 0_mask_rgba.png       # RGBA 透明背景抠图
├── pose.npy              # 每帧 6D 位姿 (N×4×4)
└── <物体>_pose.mp4       # 最终可视化视频（跟踪时实时编码）
```

## 适配其他相机

提取脚本 `scripts/extract_realsense_bag.py` **自动从 `.db3` 文件中读取相机内参和外参**。

- **有 camera_info** → 自动适配，无需改动
- **没有 camera_info** → 修改 `read_camera_params()` 的 fallback 分支
- **已对齐的 RGB-D 相机**（Orbbec / Kinect / ZED）→ 手动放入 PNG 序列 + cam_K.json，`--skip_extraction`

> 深度图必须为 **16-bit PNG（uint16，单位 mm）**。

## 参数说明

### Pipeline 步骤控制

| 参数 | 说明 |
|------|------|
| `--skip_extraction` | 跳过 .db3 提取 |
| `--skip_mask` | 跳过 mask 生成 |
| `--skip_mesh` | 跳过 mesh 生成 |
| `--skip_inference` | 跳过跟踪推理 |
| `--skip_video` | 跳过视频合成 |

### 跟踪推理

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--activate_2d_tracker` | off | 启用 Cutie 2D 跟踪器 |
| `--activate_kalman_filter` | off | 启用 12D 卡尔曼滤波器 |
| `--kf_measurement_noise_scale` | 0.05 | KF 滤波强度 |
| `--est_refine_iter` | 10 | 首帧 refine 迭代次数 |
| `--track_refine_iter` | 5 | 跟踪 refine 迭代次数 |
| `--apply_scale` | 0.01 | Mesh 单位缩放因子 |
| `--force_apply_color` | off | 无纹理 mesh 强制涂色 |
| `--no_display` | off | 禁用实时显示窗口（无头服务器） |
| `--keep_pose_frames` | off | 保留逐帧标注图（默认删除省空间） |

### Hunyuan3D

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_hunyuan3d` | off | 启用自动 mesh 生成 |
| `--hunyuan3d_backend` | `tencent` | `tencent` 或 `service` |
| `--hunyuan3d_url` | `http://localhost:18081` | `service` 后端地址 |
| `--hunyuan3d_face_count` | `100000` | 腾讯云输出面数 |
| `--hunyuan3d_timeout` | `900` | 生成/轮询超时秒数 |

### 其他

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max_frames` | None | 限制提取帧数（调试用） |
| `--fps` | 30 | 输出视频帧率 |
| `--object` | all | 选择目标处理数据 |
