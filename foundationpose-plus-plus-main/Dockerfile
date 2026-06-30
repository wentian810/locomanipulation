# FoundationPose++ Docker 镜像
#
# 此镜像只包含 FoundationPose + Cutie（位姿跟踪核心）。
# Qwen2-VL（检测）和 SAM-HQ（分割）在宿主机独立运行，不在镜像内。
#
# 使用方式:
#   1. 构建: docker build -t foundationpose-pp:latest .
#   2. 运行: 见 REPRODUCE.md

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV FORCE_CUDA=1
ENV MAX_JOBS=4

# ═══════════════════════════════════════════════════════════
# 第 1 层: 系统依赖
# ═══════════════════════════════════════════════════════════

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ninja-build \
    libboost-all-dev libeigen3-dev \
    libgl1-mesa-glx libgl1-mesa-dev libglvnd0 libglvnd-dev \
    libosmesa6-dev libglfw3-dev libglew-dev \
    libjpeg-dev libpng-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libssl-dev libbz2-dev libffi-dev libreadline-dev libsqlite3-dev zlib1g-dev \
    wget curl unzip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/include/eigen3/Eigen /usr/local/include/Eigen \
    && ln -sf /usr/include/eigen3/unsupported /usr/local/include/unsupported 2>/dev/null || true

# ═══════════════════════════════════════════════════════════
# 第 2 层: Miniconda + Python 3.10
# ═══════════════════════════════════════════════════════════

RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda create -n foundation_pose_pp python=3.10 -y \
    && conda clean -afy

SHELL ["/bin/bash", "-c"]
RUN echo 'source /opt/conda/etc/profile.d/conda.sh && conda activate foundation_pose_pp' > /root/.bashrc

# ═══════════════════════════════════════════════════════════
# 第 3 层: PyTorch 2.4.1
# ═══════════════════════════════════════════════════════════

RUN source /opt/conda/etc/profile.d/conda.sh \
    && conda activate foundation_pose_pp \
    && pip install --no-cache-dir \
        torch==2.4.1 torchvision==0.19.1 \
        --index-url https://download.pytorch.org/whl/cu121

# ═══════════════════════════════════════════════════════════
# 第 4 层: pytorch3d + nvdiffrast 源码 COPY
# ═══════════════════════════════════════════════════════════

COPY pytorch3d/ /workspace/pytorch3d/
COPY nvdiffrast/ /workspace/nvdiffrast/

# ═══════════════════════════════════════════════════════════
# 第 5 层: pytorch3d 编译
# ═══════════════════════════════════════════════════════════

RUN source /opt/conda/etc/profile.d/conda.sh \
    && conda activate foundation_pose_pp \
    && pip install --no-cache-dir Cython fvcore iopath \
    && FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="9.0" \
        pip install --no-cache-dir --no-build-isolation \
        /workspace/pytorch3d

# ═══════════════════════════════════════════════════════════
# 第 6 层: nvdiffrast 编译
# ═══════════════════════════════════════════════════════════

RUN source /opt/conda/etc/profile.d/conda.sh \
    && conda activate foundation_pose_pp \
    && pip install --no-cache-dir --no-build-isolation \
        /workspace/nvdiffrast

# ═══════════════════════════════════════════════════════════
# 第 7 层: 纯位姿跟踪所需的 Python 依赖
# ═══════════════════════════════════════════════════════════
#
# 注意: Qwen2-VL (transformers, qwen-vl-utils, accelerate)
#       和 SAM-HQ (segment-anything-hq) 不在此镜像内，
#       它们在宿主机独立运行。

RUN source /opt/conda/etc/profile.d/conda.sh \
    && conda activate foundation_pose_pp \
    && pip install --no-cache-dir \
        numpy opencv-python scipy "imageio[ffmpeg]" trimesh pyrender \
        kornia open3d pybind11 pysdf xatlas h5py \
        wandb timm einops Pillow matplotlib \
        fastapi uvicorn av rosbags requests \
        transformations warp-lang hydra-core omegaconf

# ═══════════════════════════════════════════════════════════
# 第 8 层: FoundationPose C++/CUDA 扩展编译
# ═══════════════════════════════════════════════════════════

WORKDIR /workspace
COPY FoundationPose/mycpp/ /workspace/FoundationPose/mycpp/

RUN sed -i 's|/home/admin01/miniconda3/envs/foundationpose/include/eigen3|/usr/include/eigen3|g' \
        /workspace/FoundationPose/mycpp/CMakeLists.txt \
    && cd /workspace/FoundationPose/mycpp \
    && mkdir -p build && cd build \
    && cmake .. \
        -DPYTHON_EXECUTABLE=/opt/conda/envs/foundation_pose_pp/bin/python \
        -DCMAKE_BUILD_TYPE=Release \
        -Dpybind11_DIR=/opt/conda/envs/foundation_pose_pp/lib/python3.10/site-packages/pybind11/share/cmake/pybind11 \
    && make -j${MAX_JOBS}

# ═══════════════════════════════════════════════════════════
# 第 9 层: 完整项目代码
# ═══════════════════════════════════════════════════════════

COPY . /workspace/

# 安装 Cutie (位姿跟踪所需)
RUN source /opt/conda/etc/profile.d/conda.sh \
    && conda activate foundation_pose_pp \
    && cd /workspace/Cutie \
    && pip install -e . --no-deps 2>&1 | tail -1

# ═══════════════════════════════════════════════════════════
# 最终配置
# ═══════════════════════════════════════════════════════════

ENV PROJECT_ROOT=/workspace
ENV PYTHONPATH=/workspace:/workspace/FoundationPose:/workspace/src

ENTRYPOINT ["/bin/bash", "-c", \
    "source /opt/conda/etc/profile.d/conda.sh && conda activate foundation_pose_pp && exec \"$@\"", "--"]
CMD ["/bin/bash"]
