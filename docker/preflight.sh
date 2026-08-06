#!/usr/bin/env bash
set -euo pipefail

ROOT="${PIPELINE_ROOT:-/workspace/locomotion}"
CONDA_BASE="${CONDA_BASE:-/opt/conda}"
PY_LOCO="${PY_LOCO:-${CONDA_BASE}/envs/locomotion/bin/python}"
PY_PHC="${PY_PHC:-${CONDA_BASE}/envs/phc/bin/python}"
export PIPELINE_PYTHON="${PIPELINE_PYTHON:-$PY_LOCO}"
REQUIRE_MODELS="${REQUIRE_MODELS:-0}"
RUN_TESTS="${RUN_TESTS:-1}"

cd "$ROOT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT/phc-deps/chumpy:$ROOT:$ROOT/GVHMR-hand/GVHMR-main:$ROOT/locomotion_pipeline-main:$ROOT/GMR-master:$ROOT/phc-dev-felix-pipeline"
export LD_LIBRARY_PATH="${CONDA_BASE}/envs/phc/lib:$ROOT/phc-dev-felix-pipeline/isaacgym/python/isaacgym/_bindings/linux-x86_64:$ROOT/phc-dev-felix-pipeline/isaacgym:${LD_LIBRARY_PATH:-}"

echo "[docker-preflight] root=$ROOT"
echo "[docker-preflight] locomotion=$PY_LOCO"
echo "[docker-preflight] phc=$PY_PHC"

"$PY_LOCO" - <<'PY'
import chumpy, cv2, mujoco, numpy, scipy, torch, yaml
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
import general_motion_retargeting
print("locomotion imports: ok")
print("GVHMR source import: ok (%s)" % DemoPL.__module__)
print("torch=%s cuda=%s available=%s" % (torch.__version__, torch.version.cuda, torch.cuda.is_available()))
print("numpy=%s scipy=%s mujoco=%s opencv=%s" % (numpy.__version__, scipy.__version__, mujoco.__version__, cv2.__version__))
PY

PYTHONPATH="$ROOT/phc-dev-felix-pipeline:$ROOT/phc-dev-felix-pipeline/isaacgym/python:$PYTHONPATH" \
  "$PY_PHC" - <<'PY'
from isaacgym import gymapi
import torch, poselib, smpl_sim, smplx
print("phc imports: ok")
print("torch=%s cuda=%s available=%s" % (torch.__version__, torch.version.cuda, torch.cuda.is_available()))
PY

for required in \
  "$ROOT/GVHMR-hand/GVHMR-main/tools/pipeline/run_batch_dataset6_hand4wholepp.sh" \
  "$ROOT/GMR-master/run_show_gmr_batch.sh" \
  "$ROOT/phc-dev-felix-pipeline/output/HumanoidIm/phc_3/Humanoid.pth" \
  "$ROOT/phc-dev-felix-pipeline/output/HumanoidIm/phc_comp_3/Humanoid.pth"; do
  test -e "$required" || { echo "missing image runtime file: $required" >&2; exit 2; }
done

if [[ "$REQUIRE_MODELS" = "1" ]]; then
  for required in \
    /models/gvhmr/checkpoints/gvhmr/gvhmr_siga24_release.ckpt \
    /models/gvhmr/checkpoints/hmr2/epoch=10-step=25000.ckpt \
    /models/gvhmr/checkpoints/vitpose/vitpose-h-coco-wholebody.pth \
    /models/hand4whole/demo/snapshot_6.pth \
    /models/locomotion_assets/smplh/SMPLH_NEUTRAL.pkl \
    /models/locomotion_assets/ACCAD \
    /models/gvhmr/body_models/smplx/SMPLX_NEUTRAL.pkl; do
    test -e "$required" || { echo "missing mounted model/runtime asset: $required" >&2; exit 3; }
  done
  echo "model mounts: ok"
else
  echo "model mounts: not required (set REQUIRE_MODELS=1 for the full gate)"
fi

CHECK_DATASET="${PIPELINE_CHECK_DATASET:-${ROOT}/dataset_new6}"
if [ -d "$CHECK_DATASET" ] && find "$CHECK_DATASET" -maxdepth 1 -type f \
    \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mov' -o -iname '*.mkv' -o -iname '*.m4v' \) \
    -print -quit | grep -q .; then
  "$PY_LOCO" scripts/run_pipeline_from_config.py \
    --config_dir configs/pipelines/human_sharpa.yaml \
    --stage all --check \
    --set input.dataset_dir="$CHECK_DATASET" \
    --set input.fullbody_preflight.enabled=false \
    --print-config >/tmp/locomotion-config-check.txt
  echo "config check: ok"
else
  echo "config check: skipped (mount a dataset at $CHECK_DATASET to enable it)"
fi

if [[ "$RUN_TESTS" = "1" ]]; then
  "$PY_LOCO" -m pytest -q tests
  echo "unit tests: ok"
fi

bash -n \
  GVHMR-hand/GVHMR-main/tools/pipeline/run_pipeline.sh \
  GVHMR-hand/GVHMR-main/tools/pipeline/run_batch_dataset6.sh \
  GVHMR-hand/GVHMR-main/tools/pipeline/run_batch_dataset6_hand4wholepp.sh \
  GMR-master/run_show_gmr_batch.sh
echo "shell syntax: ok"
echo "[docker-preflight] PASS"
