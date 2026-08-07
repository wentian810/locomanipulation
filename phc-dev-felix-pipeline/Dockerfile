FROM nvcr.io/nvidia/pytorch:23.02-py3

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_EXTRA_INDEX_URL=https://pypi.ngc.nvidia.com

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV PATH=/opt/conda/bin:/home/gymuser/.local/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH}
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
ENV PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    sudo \
    unzip \
    zip \
    make \
    gcc-8 \
    g++-8 \
    libegl1 \
    libgl1-mesa-glx \
    libglfw3 \
    libglib2.0-0 \
    libosmesa6-dev \
    libsm6 \
    libxext6 \
    libx11-6 \
    libxcursor-dev \
    libxinerama-dev \
    libxi-dev \
    libxrandr-dev \
    mesa-common-dev \
    mesa-vulkan-drivers \
    pigz \
    python3-dev \
    patchelf \
    vulkan-utils \
    xserver-xephyr \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

RUN if [ -f /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0 ]; then rm -f /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0; fi && \
    if [ -f /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0.0.0 ]; then rm -f /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0.0.0; fi && \
    if [ -f /usr/share/glvnd/egl_vendor.d/50_mesa.json ]; then rm -f /usr/share/glvnd/egl_vendor.d/50_mesa.json; fi

RUN useradd --create-home --shell /bin/bash gymuser && \
    adduser gymuser sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json

COPY --from=isaacgym_src . /opt/isaacgym
COPY . /workspace/PHC

RUN chown -R gymuser:gymuser /opt/isaacgym /workspace/PHC

USER gymuser
WORKDIR /workspace/PHC

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir --default-timeout=120 --retries 10 \
    gdown \
    mujoco \
    numpy-stl \
    vtk \
    patchelf \
    termcolor \
    torchgeometry \
    scikit-image \
    numpy \
    scipy \
    ipdb \
    'joblib>=1.2.0' \
    opencv-python==4.6.0.66 \
    tqdm \
    pyyaml \
    wandb \
    gym \
    git+https://github.com/ZhengyiLuo/smplx.git@master \
    git+https://github.com/ZhengyiLuo/SMPLSim.git@master \
    lxml \
    human_body_prior \
    autograd \
    scikit-learn \
    git+https://github.com/mattloper/chumpy \
    pyvirtualdisplay \
    chardet \
    cchardet \
    imageio-ffmpeg \
    easydict \
    open3d \
    rl-games==1.1.4 && \
    python -m pip install --no-cache-dir --default-timeout=120 --retries 10 -e /opt/isaacgym/python

ENV PYTHONPATH=/workspace/PHC:/workspace/PHC/isaacgym/python:/opt/isaacgym/python:${PYTHONPATH}
ENV ISAACGYM_PATH=/opt/isaacgym

CMD ["/bin/bash"]
