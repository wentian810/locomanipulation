#!/bin/bash
# [01] Create the 4 conda envs the pipeline switches between. Names come from config/paths.sh.
# Also see env/README.md.
#
# NOTE: these recreate the envs from original source. For an exact reproduction of a working set up, see env/README.md
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/config/paths.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"

only="${1:-all}"   # optionally: sam3 | sam3d | hawor | tapnet

mk_sam3d () {
  echo "=== [01] sam3d ($ENV_SAM3D) ==="
  # NOTE: default.yml/requirements pin CUDA 12.1 / torch ~2.5; the working env (env/sam3d.yml)
  # runs torch 2.8+cu128. On CUDA 12.8 GPUs prefer:  conda env create -f env/sam3d.yml
  conda env create -n "$ENV_SAM3D" -f "$SAM3D_DIR/environments/default.yml" \
    || conda env update -n "$ENV_SAM3D" -f "$SAM3D_DIR/environments/default.yml"
  conda activate "$ENV_SAM3D"
  conda install -y -c conda-forge ffmpeg   
  pip install -r "$SAM3D_DIR/requirements.txt"
  [ -f "$SAM3D_DIR/requirements.inference.txt" ] && pip install -r "$SAM3D_DIR/requirements.inference.txt" || true
  pip install -e "$SAM3D_DIR" || true     # expose the sam3d_objects package (also used by Fast-SAM3D)
  pip install viser                       # optional Stage 4 viz (scripts/visualize_3d.py)
  pip install "geocalib @ git+https://github.com/cvg/GeoCalib.git"  # Stage 2 gravity (scripts/predict_video_gravity.py)
  conda deactivate
}

mk_hawor () {
  echo "=== [01] hawor ($ENV_HAWOR) — CUDA 11.7 torch (per HaWoR README) ==="
  conda create -y -n "$ENV_HAWOR" python=3.10
  conda activate "$ENV_HAWOR"
  conda install -y -c conda-forge ffmpeg  
  # upstream HaWoR recipe (cu117); the working env (env/hawor.yml) is torch 2.9+cu128 —
  # on CUDA 12.8 GPUs prefer:  conda env create -f env/hawor.yml
  pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
  # NOTE: requirements.txt pins mmcv==1.3.9 (only thirdparty/Metric3D uses it; it won't
  # build on modern torch) — the working env omits it. chumpy comes from the git pin
  # `chumpy@git+...` (resolves to 0.71; do NOT `pip install chumpy==0.71` — PyPI maxes
  # at 0.70) and must be installed with --no-build-isolation (its setup.py needs numpy).
  grep -vE "mmcv==1.3.9|chumpy@" "$HAWOR_DIR/requirements.txt" | pip install -r /dev/stdin
  pip install "chumpy@git+https://github.com/mattloper/chumpy" --no-build-isolation
  pip install "setuptools<81"   # pytorch-lightning 2.2.4 needs pkg_resources (removed in setuptools>=81)
  pip install pytorch-lightning==2.2.4 --no-deps
  pip install lightning-utilities torchmetrics==1.4.0
  # NOTE (Blackwell/torch>=2.x): lietorch's dispatch.h needs a 1-line patch
  # (::detail::scalar_type(the_type) -> the_type.scalarType()) or this build fails.
  ( cd "$HAWOR_DIR/thirdparty/DROID-SLAM" && python setup.py install )
  # torch>=2.6 defaults torch.load(weights_only=True), which rejects HaWoR's checkpoints
  # (they embed an omegaconf DictConfig). These are the official/trusted weights, so restore
  # the pre-2.6 default for this env (no code change needed); applied on every activate.
  conda env config vars set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 -n "$ENV_HAWOR"
  conda deactivate
}

mk_tapnet () {
  echo "=== [01] tapnet ($ENV_TAPNET) — torch-only BootsTAPIR ==="
  conda create -y -n "$ENV_TAPNET" python=3.10
  conda activate "$ENV_TAPNET"
  conda install -y -c conda-forge ffmpeg   
  pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
  pip install -e "$TAPNET_DIR[torch]"     # IMPORTANT: editable install, NOT sys.path (tapnet/torch shadows torch)
  pip install einops tqdm mediapy
  conda deactivate
}

mk_sam3 () {
  echo "=== [01] sam3 ($ENV_SAM3) — Stage 1 segmentation (cu128) ==="
  conda create -y -n "$ENV_SAM3" python=3.12
  conda activate "$ENV_SAM3"
  conda install -y -c conda-forge ffmpeg   # run_pipeline.sh extracts frames with the sam3 env's ffmpeg
  pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
  # NOTE: the bare sam3 package omits cv2/einops/pycocotools/hydra that run_sam3_video.py
  # and sam3.model_builder need — the [dev,notebooks,train] extras are REQUIRED.
  pip install -e "$SAM3_PKG_DIR[dev,notebooks,train]"
  conda deactivate
}

case "$only" in
  all)    mk_sam3; mk_sam3d; mk_hawor; mk_tapnet ;;
  sam3)   mk_sam3 ;;
  sam3d)  mk_sam3d ;;
  hawor)  mk_hawor ;;
  tapnet) mk_tapnet ;;
  *) echo "usage: $0 [all|sam3|sam3d|hawor|tapnet]"; exit 1 ;;
esac

echo "[01] Done."
