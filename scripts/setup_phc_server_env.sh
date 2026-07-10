#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
ENV_NAME="${PHC_ENV_NAME:-phc}"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"

cd "$ROOT"

if [ ! -f "$CONDA_SH" ]; then
    echo "conda profile not found: $CONDA_SH" >&2
    echo "Set CONDA_SH=/path/to/conda.sh or install Miniconda first." >&2
    exit 1
fi

for path in \
    "phc-dev-felix-pipeline/isaacgym/python/setup.py" \
    "phc-dev-felix-pipeline/poselib/setup.py" \
    "phc-dev-felix-pipeline/output/HumanoidIm/phc_3/Humanoid.pth" \
    "phc-dev-felix-pipeline/output/HumanoidIm/phc_comp_3/Humanoid.pth" \
    "phc-dev-felix-pipeline/data/smpl/SMPL_NEUTRAL.pkl" \
    "phc-deps/SMPLSim/setup.py" \
    "phc-deps/chumpy/setup.py"; do
    if [ ! -e "$path" ]; then
        echo "missing required PHC asset/source: $ROOT/$path" >&2
        exit 1
    fi
done

source "$CONDA_SH"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" python=3.8
fi
conda activate "$ENV_NAME"

python -m pip install -U "pip<25" setuptools wheel packaging

python -m pip install \
    --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.4.1 torchvision==0.19.1

python -m pip install \
    "numpy==1.24.3" "scipy==1.10.1" "PyOpenGL==3.1.10" \
    mujoco==3.2.3 ninja numpy-stl vtk patchelf termcolor torchgeometry \
    scikit-image==0.21.0 ipdb "joblib>=1.2.0" \
    opencv-python==4.6.0.66 tqdm pyyaml wandb==0.12.21 \
    gym==0.23.1 lxml human_body_prior autograd scikit-learn \
    rl-games==1.1.4 pyvirtualdisplay chardet cchardet imageio-ffmpeg \
    easydict open3d gdown hydra-core==1.3.2 omegaconf==2.3.0 \
    gymnasium==1.1.1 mediapy importlib-resources==3.0.0

python -m pip install -e phc-deps/chumpy --no-build-isolation --no-deps
if [ -e phc-deps/smplx_fork/setup.py ]; then
    python -m pip install -e phc-deps/smplx_fork --no-build-isolation --no-deps
else
    python -m pip install \
        "smplx @ git+https://github.com/ZhengyiLuo/smplx.git@a5b8e4ac14f79f3f33fd2cf2a16e6f507146b813"
fi
python -m pip install -e phc-deps/SMPLSim --no-build-isolation --no-deps
python -m pip install -e phc-dev-felix-pipeline/poselib --no-build-isolation --no-deps
python -m pip install -e phc-dev-felix-pipeline/isaacgym/python --no-build-isolation --no-deps

PHC_ROOT="$ROOT/phc-dev-felix-pipeline"
PHC_BINDINGS="$PHC_ROOT/isaacgym/python/isaacgym/_bindings/linux-x86_64"

LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${PHC_BINDINGS}:${PHC_ROOT}/isaacgym:${LD_LIBRARY_PATH:-}" \
PYTHONPATH="${PHC_ROOT}:${PHC_ROOT}/isaacgym/python:${PYTHONPATH:-}" \
ISAACGYM_PATH="${PHC_ROOT}/isaacgym" \
python - <<'PY'
from isaacgym import gymapi
import torch
import chumpy
import poselib
import smpl_sim
import smplx

print("PHC env ok:", torch.__version__)
PY

echo "PHC environment ready: ${CONDA_PREFIX}"
