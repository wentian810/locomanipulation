#!/usr/bin/env bash
# Start the human-only GVHMR -> Locomotion -> PHC -> GMR/Sharpa release image.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RELEASE_ROOT="$REPO_ROOT/.release/20260806"
OUTPUT_ROOT="$REPO_ROOT/output_dir"
WORK_ROOT="$REPO_ROOT/human_work"
IMAGE_NAME="locomotion-human-only:20260806"
IMAGE_OVERRIDE=""
DOCKER_SUDO="${DOCKER_SUDO:-0}"
CLIP_FILTER=""
STAGE="all"
CHECK_ONLY=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash release/run_human_only_docker.sh [options] [-- additional pipeline options]

Prerequisite: run release/bootstrap_from_ucloud.sh once in the same GitHub
checkout.  This runner never mounts credentials into the container.

Options:
  --release-root PATH   Release cache produced by bootstrap (default: .release/20260806)
  --output-root PATH    Host directory for generated output (default: ./output_dir)
  --work-root PATH      Host directory for normalized work videos (default: ./human_work)
  --image NAME          Docker image tag (default: locomotion-human-only:20260806)
  --docker-sudo         Run Docker through non-interactive sudo -n docker
  --clip-filter TEXT    Process matching video names only; comma separates terms
  --stage NAME          Pipeline stage: human|quality|product|all (default: all)
  --check-only          Run the image/model/config self-check, no inference
  -h, --help            Show this help

Examples:
  bash release/run_human_only_docker.sh --check-only
  bash release/run_human_only_docker.sh --clip-filter chairwood
  bash release/run_human_only_docker.sh --stage human --clip-filter chairwood
EOF
}

die() { echo "[human-docker] ERROR: $*" >&2; exit 2; }
note() { echo "[human-docker] $*"; }
docker_cmd() {
  if [[ "$DOCKER_SUDO" == "1" ]]; then
    sudo -n docker "$@"
  else
    docker "$@"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --release-root) RELEASE_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --work-root) WORK_ROOT="$2"; shift 2 ;;
    --image) IMAGE_OVERRIDE="$2"; shift 2 ;;
    --docker-sudo) DOCKER_SUDO=1; shift ;;
    --clip-filter) CLIP_FILTER="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --check-only) CHECK_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) die "Unknown argument: $1 (use -- to pass a raw pipeline option)" ;;
  esac
done

case "$STAGE" in human|quality|product|all) ;; *) die "Invalid --stage: $STAGE" ;; esac
command -v docker >/dev/null || die "docker is not on PATH"
RELEASE_ROOT="$(cd -- "$RELEASE_ROOT" && pwd)"
state="$RELEASE_ROOT/RELEASE_PATHS.env"
[[ -f "$state" ]] || die "Missing bootstrap state: $state"
# This is generated locally by our paired bootstrap script, never from US3.
# shellcheck disable=SC1090
source "$state"

MODEL_ROOT="${MODEL_ROOT:-$RELEASE_ROOT/extracted/model-assets}"
DATASET_DIR="${DATASET_DIR:-$RELEASE_ROOT/extracted/dataset/dataset_new6}"
IMAGE_NAME="${IMAGE_OVERRIDE:-${IMAGE_NAME:-locomotion-human-only:20260806}}"
GVHMR_SOURCE_ROOT="$REPO_ROOT/GVHMR-hand/GVHMR-main/hmr4d"
[[ -f "$GVHMR_SOURCE_ROOT/model/gvhmr/gvhmr_pl_demo.py" ]] \
  || die "GitHub source overlay is incomplete: $GVHMR_SOURCE_ROOT/model/gvhmr/gvhmr_pl_demo.py"
[[ -f "$REPO_ROOT/docker/preflight.sh" ]] \
  || die "GitHub checkout lacks docker/preflight.sh: $REPO_ROOT"
for path in \
  "$MODEL_ROOT/GVHMR-hand/GVHMR-main/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt" \
  "$MODEL_ROOT/GVHMR-hand/GVHMR-main/inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt" \
  "$MODEL_ROOT/GVHMR-hand/GVHMR-main/third-party/Hand4Whole-plus-plus_RELEASE/demo/snapshot_6.pth" \
  "$MODEL_ROOT/locomotion_pipeline-main/assets/smplh/SMPLH_NEUTRAL.pkl" \
  "$MODEL_ROOT/locomotion_pipeline-main/assets/ACCAD" \
  "$MODEL_ROOT/models/yolo11n-pose.pt" \
  "$MODEL_ROOT/GVHMR-main/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.pkl"; do
  [[ -e "$path" ]] || die "Missing extracted runtime asset: $path"
done
[[ -d "$DATASET_DIR" ]] || die "Missing extracted dataset: $DATASET_DIR"
docker_cmd image inspect "$IMAGE_NAME" >/dev/null || die "Docker image not loaded: $IMAGE_NAME"

mkdir -p "$OUTPUT_ROOT" "$WORK_ROOT"
OUTPUT_ROOT="$(cd -- "$OUTPUT_ROOT" && pwd)"
WORK_ROOT="$(cd -- "$WORK_ROOT" && pwd)"

mounts=(
  # The runtime image carries CUDA/Conda/PHC; the checked-out GVHMR source is
  # mounted so GitHub remains the canonical source of every Python module.
  -v "$GVHMR_SOURCE_ROOT:/workspace/locomotion/GVHMR-hand/GVHMR-main/hmr4d:ro"
  -v "$REPO_ROOT/docker/preflight.sh:/workspace/locomotion/docker/preflight.sh:ro"
  -v "$MODEL_ROOT/GVHMR-hand/GVHMR-main/inputs/checkpoints:/models/gvhmr/checkpoints:ro"
  -v "$MODEL_ROOT/GVHMR-main/inputs/checkpoints/body_models:/models/gvhmr/body_models:ro"
  -v "$MODEL_ROOT/GVHMR-hand/GVHMR-main/third-party/Hand4Whole-plus-plus_RELEASE:/models/hand4whole:ro"
  -v "$MODEL_ROOT/locomotion_pipeline-main/assets:/models/locomotion_assets:ro"
  -v "$MODEL_ROOT/models:/workspace/locomotion/models:ro"
  -v "$DATASET_DIR:/data/input:ro"
  -v "$OUTPUT_ROOT:/data/output"
  -v "$WORK_ROOT:/data/work"
)
if [[ -d "$MODEL_ROOT/GVHMR-hand/GVHMR-main/third-party/WiLoR" ]]; then
  mounts+=(-v "$MODEL_ROOT/GVHMR-hand/GVHMR-main/third-party/WiLoR:/models/wilor:ro")
else
  note "WiLoR tree is absent; it is not needed by the default Hand4Whole++ backend"
fi

common=(run --rm --gpus all
  -e REQUIRE_MODELS=1
  -e PIPELINE_CHECK_DATASET=/data/input
  # conda-pack preserved an old editable chumpy .pth path.  Its source is
  # already in the runtime image; make that immutable in-image source importable.
  -e PYTHONPATH=/workspace/locomotion/phc-deps/chumpy
  "${mounts[@]}"
  "$IMAGE_NAME")

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  note "run image, model and configuration self-check"
  docker_cmd "${common[@]}" check
  exit $?
fi

pipeline_args=(human --stage "$STAGE" --dataset-dir /data/input --output-root /data/output \
  --set input.work_video.directory=/data/work)
[[ -z "$CLIP_FILTER" ]] || pipeline_args+=(--clip-filter "$CLIP_FILTER")
pipeline_args+=("${EXTRA_ARGS[@]}")
note "start stage=$STAGE image=$IMAGE_NAME dataset=$DATASET_DIR output=$OUTPUT_ROOT"
docker_cmd "${common[@]}" "${pipeline_args[@]}"
