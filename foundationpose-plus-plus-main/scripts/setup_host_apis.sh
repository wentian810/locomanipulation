#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# 宿主机 Qwen2-VL + SAM-HQ API 环境安装脚本
#
# 创建两个独立的 conda 环境，各自只装最小依赖。
# 每个约 15-20 min（主要耗时在 PyTorch 下载）。
# ═══════════════════════════════════════════════════════════════
set -e

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

echo "========================================="
echo " 宿主机 API 环境安装"
echo " 项目根目录: $PROJECT_ROOT"
echo "========================================="

# ────────────────────────────────────────
# 环境 1: qwen2_vl (端口 9003)
# ────────────────────────────────────────
echo ""
echo ">>> 创建 qwen2_vl 环境 ..."
conda create -n qwen2_vl python=3.10 -y
conda run -n qwen2_vl conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda run -n qwen2_vl conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

echo ">>> 安装 PyTorch (CUDA 12.1) ..."
conda run -n qwen2_vl pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

echo ">>> 安装 Qwen2-VL 依赖 ..."
conda run -n qwen2_vl pip install \
    fastapi uvicorn pydantic \
    accelerate qwen-vl-utils

echo ">>> 安装 transformers（本地源码） ..."
conda run -n qwen2_vl pip install "$PROJECT_ROOT/transformers"

echo "✅ qwen2_vl 环境完成"

# ────────────────────────────────────────
# 环境 2: hq_sam (端口 9002)
# ────────────────────────────────────────
echo ""
echo ">>> 创建 hq_sam 环境 ..."
conda create -n hq_sam python=3.10 -y
conda run -n hq_sam conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda run -n hq_sam conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

echo ">>> 安装 PyTorch (CUDA 12.1) ..."
conda run -n hq_sam pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

echo ">>> 安装 SAM-HQ 依赖 ..."
conda run -n hq_sam pip install \
    fastapi uvicorn pydantic \
    numpy opencv-python Pillow h5py \
    segment-anything-hq

echo ">>> 安装 segment-anything（本地 sam-hq 源码） ..."
conda run -n hq_sam pip install -e "$PROJECT_ROOT/sam-hq" --no-deps

echo "✅ hq_sam 环境完成"

# ────────────────────────────────────────
# 验证
# ────────────────────────────────────────
echo ""
echo "========================================="
echo " 验证环境"
echo "========================================="

echo ""
echo "--- qwen2_vl ---"
conda run -n qwen2_vl python -c "
import torch; print(f'PyTorch {torch.__version__}  CUDA {torch.version.cuda}')
from fastapi import FastAPI; print('fastapi OK')
from transformers import Qwen2VLForConditionalGeneration; print('transformers OK')
from qwen_vl_utils import process_vision_info; print('qwen-vl-utils OK')
print('[qwen2_vl] 全部依赖 OK')
"

echo ""
echo "--- hq_sam ---"
conda run -n hq_sam python -c "
import torch; print(f'PyTorch {torch.__version__}  CUDA {torch.version.cuda}')
from fastapi import FastAPI; print('fastapi OK')
import cv2; print('opencv OK')
import segment_anything; print('segment-anything OK')
from segment_anything_hq import sam_model_registry; print('segment-anything-hq OK')
print('[hq_sam] 全部依赖 OK')
"

echo ""
echo "========================================="
echo " 安装完成!"
echo ""
echo "启动方式:"
echo "  conda run -n qwen2_vl --no-capture-output python src/WebAPI/qwen2_vl_api.py \\"
echo "    --weight_path $PROJECT_ROOT/Qwen2-VL/weights &"
echo ""
echo "  conda run -n hq_sam --no-capture-output python src/WebAPI/hq_sam_api.py &"
echo "========================================="
