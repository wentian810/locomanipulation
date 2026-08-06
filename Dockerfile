# syntax=docker/dockerfile:1.7
#
# Build from the human-only support project.  PHC is intentionally supplied as
# a named build context because it is not tracked in this repository:
#
#   docker buildx build --load \
#     --build-context phc=/home/jixingyu/.codex_phc_context_20260806 \
#     --build-context envs=/home/jixingyu/.codex_conda_packs_20260806 \
#     -t locomotion-human-only:20260806 .
#
# The image contains code, system libraries, two Python environments, PHC
# runtime assets and PHC policy weights.  GVHMR/Hand4Whole++/SMPL/ViTPose
# checkpoints and input videos are mounted at runtime; see docs/modules/07.
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PIP_INDEX_URL=https://pypi.org/simple

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    CONDA_BASE=/opt/conda \
    PIPELINE_ROOT=/workspace/locomotion \
    PYTHONNOUSERSITE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PATH=/opt/conda/envs/locomotion/bin:/opt/conda/bin:${PATH} \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates curl git git-lfs \
    build-essential cmake pkg-config patchelf \
    ffmpeg unzip zip pigz \
    libegl1 libgl1 libglvnd0 libglfw3 libglib2.0-0 \
    libosmesa6 libosmesa6-dev libsm6 libxext6 libx11-6 \
    libxcursor1 libxinerama1 libxi6 libxrandr2 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -o /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm -f /tmp/miniforge.sh \
    && /opt/conda/bin/conda config --system --set always_yes true \
    && /opt/conda/bin/conda config --system --set channel_priority strict

WORKDIR /workspace/locomotion
COPY . /workspace/locomotion

# Only the runtime portions of PHC are copied from the named context.  This
# deliberately leaves out PHC logs, sample outputs and unrelated scene work.
COPY --from=phc /phc-dev-felix-pipeline/phc /workspace/locomotion/phc-dev-felix-pipeline/phc
COPY --from=phc /phc-dev-felix-pipeline/poselib /workspace/locomotion/phc-dev-felix-pipeline/poselib
COPY --from=phc /phc-dev-felix-pipeline/isaacgym /workspace/locomotion/phc-dev-felix-pipeline/isaacgym
COPY --from=phc /phc-dev-felix-pipeline/data /workspace/locomotion/phc-dev-felix-pipeline/data
COPY --from=phc /phc-dev-felix-pipeline/assets /workspace/locomotion/phc-dev-felix-pipeline/assets
COPY --from=phc /phc-dev-felix-pipeline/scripts /workspace/locomotion/phc-dev-felix-pipeline/scripts
COPY --from=phc /phc-dev-felix-pipeline/sample_data /workspace/locomotion/phc-dev-felix-pipeline/sample_data
COPY --from=phc /phc-dev-felix-pipeline/output/HumanoidIm /workspace/locomotion/phc-dev-felix-pipeline/output/HumanoidIm
COPY --from=phc /phc-deps /workspace/locomotion/phc-deps

# The validated server environments are packed and supplied as a second named
# build context. This preserves the known-good locomotion/PyTorch3D and
# PHC/Isaac Gym dependency split without re-solving host-specific CUDA pins.
RUN --mount=type=bind,from=envs,target=/mnt/conda-packs,ro \
    mkdir -p /opt/conda/envs/locomotion /opt/conda/envs/phc \
    && tar -xzf /mnt/conda-packs/locomotion.tar.gz -C /opt/conda/envs/locomotion \
    && tar -xzf /mnt/conda-packs/phc.tar.gz -C /opt/conda/envs/phc \
    && /opt/conda/envs/locomotion/bin/conda-unpack \
    && /opt/conda/envs/phc/bin/conda-unpack \
    && rm -rf /tmp/*

RUN /opt/conda/envs/locomotion/bin/python -m pip install --no-build-isolation --no-deps -e /workspace/locomotion/phc-deps/chumpy \
    && /opt/conda/envs/locomotion/bin/python -m pip install --no-deps -e /workspace/locomotion/GMR-master \
    && /opt/conda/envs/phc/bin/python -m pip install --no-build-isolation --no-deps -e /workspace/locomotion/phc-deps/chumpy \
    && if [ -e /workspace/locomotion/phc-deps/smplx_fork/setup.py ]; then \
         /opt/conda/envs/phc/bin/python -m pip install --no-build-isolation --no-deps -e /workspace/locomotion/phc-deps/smplx_fork; \
       fi \
    && /opt/conda/envs/phc/bin/python -m pip install --no-build-isolation --no-deps -e /workspace/locomotion/phc-deps/SMPLSim \
    && /opt/conda/envs/phc/bin/python -m pip install --no-build-isolation --no-deps -e /workspace/locomotion/phc-dev-felix-pipeline/poselib \
    && /opt/conda/envs/phc/bin/python -m pip install --no-build-isolation --no-deps -e /workspace/locomotion/phc-dev-felix-pipeline/isaacgym/python \
    && rm -rf /root/.cache /tmp/*

# Hand4Whole++ and WiLoR contain code as well as checkpoints on the server.
# They are mounted as complete external trees so the image never bakes in
# licensed multi-GB model files.  GMR still needs the Sharpa geometry; the
# dockerignore keeps only that geometry from do-as-i-do-main.
RUN mkdir -p /models \
    /models/gvhmr/body_models \
    /workspace/locomotion/GVHMR-hand/GVHMR-main/inputs \
    /workspace/locomotion/GVHMR-hand/GVHMR-main/third-party \
    /workspace/locomotion/locomotion_pipeline-main/assets \
    /workspace/locomotion/GMR-master/assets \
    && rm -rf /workspace/locomotion/GVHMR-hand/GVHMR-main/inputs/checkpoints \
    && ln -s /models/gvhmr/checkpoints /workspace/locomotion/GVHMR-hand/GVHMR-main/inputs/checkpoints \
    && rm -rf /workspace/locomotion/GVHMR-hand/GVHMR-main/third-party/Hand4Whole-plus-plus_RELEASE \
    && ln -s /models/hand4whole /workspace/locomotion/GVHMR-hand/GVHMR-main/third-party/Hand4Whole-plus-plus_RELEASE \
    && rm -rf /workspace/locomotion/GVHMR-hand/GVHMR-main/third-party/WiLoR \
    && ln -s /models/wilor /workspace/locomotion/GVHMR-hand/GVHMR-main/third-party/WiLoR \
    && ln -s /workspace/locomotion/GVHMR-hand/GVHMR-main /workspace/locomotion/GVHMR-main \
    && ln -s /models/locomotion_assets/smplh /workspace/locomotion/locomotion_pipeline-main/assets/smplh \
    && ln -s /models/locomotion_assets/ACCAD /workspace/locomotion/locomotion_pipeline-main/assets/ACCAD \
    && rm -rf /workspace/locomotion/GMR-master/assets/body_models \
    && mkdir -p /workspace/locomotion/GMR-master/assets/body_models \
    && ln -s /models/gvhmr/body_models/smplx /workspace/locomotion/GMR-master/assets/body_models/smplx \
    && ln -s /models/locomotion_assets/smplh /workspace/locomotion/GMR-master/assets/body_models/smplh \
    && ln -s /models/locomotion_assets/ACCAD /workspace/locomotion/GMR-master/assets/body_models/ACCAD

ENV PY_LOCO=/opt/conda/envs/locomotion/bin/python \
    PY_GVHMR=/opt/conda/envs/locomotion/bin/python \
    PY_PHC=/opt/conda/envs/phc/bin/python \
    PY_GMR=/opt/conda/envs/locomotion/bin/python \
    PIPELINE_PYTHON=/opt/conda/envs/locomotion/bin/python \
    PYTHONPATH=/workspace/locomotion/phc-deps/chumpy:/workspace/locomotion:/workspace/locomotion/GVHMR-hand/GVHMR-main:/workspace/locomotion/locomotion_pipeline-main:/workspace/locomotion/GMR-master:/workspace/locomotion/phc-dev-felix-pipeline

RUN chmod +x /workspace/locomotion/docker/entrypoint.sh /workspace/locomotion/docker/preflight.sh \
    && mkdir -p /tmp/matplotlib /tmp/cache

ENTRYPOINT ["/workspace/locomotion/docker/entrypoint.sh"]
CMD ["check"]
