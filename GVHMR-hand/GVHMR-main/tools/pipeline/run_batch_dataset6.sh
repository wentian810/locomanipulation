#!/bin/bash
# ============================================================================
# Batch pipeline: dataset_new6 videos -> GVHMR-hand -> Locomotion -> PHC -> GMR
# ============================================================================
# Usage:
#   cd <repo-root>
#   bash GVHMR-hand/GVHMR-main/tools/pipeline/run_batch_dataset6.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GVHMR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PIPELINE_ROOT="${PIPELINE_ROOT:-$(cd "${GVHMR}/../.." && pwd)}"

# This is the shared backend, not a reproducible public entrypoint.  It does
# not know which embodiment/profile contract the caller intended unless a
# backend wrapper or the config runner has frozen that contract first.
if [ "${PIPELINE_WRAPPER_CONFIGURED:-0}" != "1" ] && \
   [ "${PIPELINE_ALLOW_DIRECT_BATCH:-0}" != "1" ]; then
    echo "ERROR: run_batch_dataset6.sh is an internal backend and refuses an unconfigured run." >&2
    echo "Use: python scripts/run_pipeline_from_config.py --config_dir configs/pipelines/human_sharpa.yaml --stage human" >&2
    echo "Or:  bash GVHMR-hand/GVHMR-main/tools/pipeline/run_batch_dataset6_hand4wholepp.sh" >&2
    echo "For an explicitly frozen low-level debug environment only, set PIPELINE_ALLOW_DIRECT_BATCH=1." >&2
    exit 2
fi

DATASET="${DATASET:-${PIPELINE_ROOT}/dataset_new6}"
CLIP_FILTER="${CLIP_FILTER:-}"
# Kept overridable for non-inference smoke tests and frozen low-level debug
# environments. Production wrappers leave this unset and always use the
# in-tree pipeline.
PIPELINE_SCRIPT="${PIPELINE_SCRIPT:-${SCRIPT_DIR}/run_pipeline.sh}"
OUTPUT_BASE="${OUTPUT_BASE:-${PIPELINE_ROOT}/output_dir/kungfu_hand}"
USE_WORK_VIDEO="${USE_WORK_VIDEO:-1}"
WORK_DATASET="${WORK_DATASET:-${PIPELINE_ROOT}/dataset_new6_work_1280}"
WORK_WIDTH="${WORK_WIDTH:-1280}"
WORK_HEIGHT="${WORK_HEIGHT:-960}"
WORK_CRF="${WORK_CRF:-18}"
WORK_FPS="${WORK_FPS:-30}"
FORCE_WORK_VIDEO="${FORCE_WORK_VIDEO:-0}"
if [[ ! "$WORK_FPS" =~ ^30([.]0+)?$ ]]; then
    echo "ERROR: WORK_FPS must be 30; PHC, GMR, and motion export currently share a 30 FPS contract." >&2
    exit 2
fi
CONDA_BASE="${CONDA_BASE:-${HOME}/miniconda3}"
if [ -d "${CONDA_BASE}/envs/locomotion/bin" ]; then
    export PATH="${CONDA_BASE}/envs/locomotion/bin:${PATH}"
fi
GVHMR_BATCH_SIZE="${GVHMR_BATCH_SIZE:-1}"
GVHMR_HAMER_BATCH_SIZE="${GVHMR_HAMER_BATCH_SIZE:-2}"
GVHMR_FORCE_HAND_PREPROCESS="${GVHMR_FORCE_HAND_PREPROCESS:-0}"
GVHMR_ISOLATE_HAND_PREPROCESS="${GVHMR_ISOLATE_HAND_PREPROCESS:-auto}"
GVHMR_HAND_BACKEND="${GVHMR_HAND_BACKEND:-hand4wholepp}"
GVHMR_HAND_CONSTRAINT_PROFILE="${GVHMR_HAND_CONSTRAINT_PROFILE:-balanced}"
GVHMR_WILOR_ROOT="${GVHMR_WILOR_ROOT:-${GVHMR}/third-party/WiLoR}"
GVHMR_WILOR_CHECKPOINT="${GVHMR_WILOR_CHECKPOINT:-${GVHMR_WILOR_ROOT}/pretrained_models/wilor_final.ckpt}"
GVHMR_WILOR_CONFIG="${GVHMR_WILOR_CONFIG:-${GVHMR_WILOR_ROOT}/pretrained_models/model_config.yaml}"
GVHMR_WILOR_FAST="${GVHMR_WILOR_FAST:-0}"
GVHMR_HAND4WHOLEPP_ROOT="${GVHMR_HAND4WHOLEPP_ROOT:-${GVHMR}/third-party/Hand4Whole-plus-plus_RELEASE}"
GVHMR_HAND4WHOLEPP_SNAPSHOT="${GVHMR_HAND4WHOLEPP_SNAPSHOT:-${GVHMR_HAND4WHOLEPP_ROOT}/demo/snapshot_6.pth}"
GVHMR_HAND4WHOLEPP_PYTHON="${GVHMR_HAND4WHOLEPP_PYTHON:-}"
GVHMR_HAND4WHOLEPP_BATCH_SIZE="${GVHMR_HAND4WHOLEPP_BATCH_SIZE:-4}"
GVHMR_HAND4WHOLEPP_YOLO_MODEL="${GVHMR_HAND4WHOLEPP_YOLO_MODEL:-yolo11n.pt}"
GVHMR_HAND4WHOLEPP_JOINT_SOURCE="${GVHMR_HAND4WHOLEPP_JOINT_SOURCE:-direct_mano}"
# The generic backend forwards these for a deliberately opt-in crop-tracking
# experiment. The Hand4Whole++ wrapper/config runner owns the default policy.
export GVHMR_HAND4WHOLEPP_CROP_TRACKING="${GVHMR_HAND4WHOLEPP_CROP_TRACKING:-off}"
export GVHMR_HAND4WHOLEPP_CROP_TRACKING_MAX_GAP="${GVHMR_HAND4WHOLEPP_CROP_TRACKING_MAX_GAP:-8}"
export GVHMR_HAND4WHOLEPP_CROP_TRACKING_MAX_PREDICTION_GAP="${GVHMR_HAND4WHOLEPP_CROP_TRACKING_MAX_PREDICTION_GAP:-2}"
export GVHMR_HAND4WHOLEPP_CROP_TRACKING_DIRECT_OBSERVATION_QUALITY="${GVHMR_HAND4WHOLEPP_CROP_TRACKING_DIRECT_OBSERVATION_QUALITY:-0.75}"
GVHMR_VITPOSE_IMG_DS="${GVHMR_VITPOSE_IMG_DS:-1.0}"
GVHMR_HAND_KPT_CONF_THR="${GVHMR_HAND_KPT_CONF_THR:-0.35}"
GVHMR_HAND_KPT_LOW_CONF_THR="${GVHMR_HAND_KPT_LOW_CONF_THR:-0.2}"
GVHMR_HAND_KPT_HI_MIN_KEYPOINTS="${GVHMR_HAND_KPT_HI_MIN_KEYPOINTS:-6}"
GVHMR_HAND_MIN_KEYPOINTS="${GVHMR_HAND_MIN_KEYPOINTS:-3}"
GVHMR_HAMER_BBOX_RESCALE="${GVHMR_HAMER_BBOX_RESCALE:-2.6}"
GVHMR_HAMER_BBOX_RESCALE_CANDIDATES="${GVHMR_HAMER_BBOX_RESCALE_CANDIDATES:-3.0,3.5,4.0}"
GVHMR_HAMER_CANDIDATE_SWITCH_PENALTY="${GVHMR_HAMER_CANDIDATE_SWITCH_PENALTY:-8.0}"
GVHMR_HAND_BBOX_MIN_SIZE="${GVHMR_HAND_BBOX_MIN_SIZE:-96}"
GVHMR_HAND_BBOX_SMOOTHING="${GVHMR_HAND_BBOX_SMOOTHING:-0.55}"
GVHMR_HAND_BBOX_MAX_JUMP="${GVHMR_HAND_BBOX_MAX_JUMP:-90}"
GVHMR_HAND_BBOX_OVERLAP_IOU="${GVHMR_HAND_BBOX_OVERLAP_IOU:-0.55}"
GVHMR_HAND_BBOX_COLLISION_SCORE_RATIO="${GVHMR_HAND_BBOX_COLLISION_SCORE_RATIO:-1.35}"
GVHMR_HAMER_REFINE_STEPS="${GVHMR_HAMER_REFINE_STEPS:-8}"
GVHMR_HAMER_REFINE_LR="${GVHMR_HAMER_REFINE_LR:-0.03}"
GVHMR_HAMER_REFINE_CONF_THR="${GVHMR_HAMER_REFINE_CONF_THR:-0.45}"
GVHMR_HAMER_REFINE_MIN_KEYPOINTS="${GVHMR_HAMER_REFINE_MIN_KEYPOINTS:-6}"
GVHMR_HAMER_REFINE_POSE_PRIOR="${GVHMR_HAMER_REFINE_POSE_PRIOR:-0.005}"
GVHMR_HAMER_REFINE_GLOBAL_PRIOR="${GVHMR_HAMER_REFINE_GLOBAL_PRIOR:-0.01}"
GVHMR_FILTER_MANO_WRIST="${GVHMR_FILTER_MANO_WRIST:-0}"
GVHMR_WRIST_FILTER_W_TEMP="${GVHMR_WRIST_FILTER_W_TEMP:-2.0}"
GVHMR_WRIST_FILTER_W_BODY="${GVHMR_WRIST_FILTER_W_BODY:-0.5}"
GVHMR_WRIST_FILTER_W_TEMP_LOW_CONF="${GVHMR_WRIST_FILTER_W_TEMP_LOW_CONF:-4.0}"
GVHMR_WRIST_FILTER_W_BODY_LOW_CONF="${GVHMR_WRIST_FILTER_W_BODY_LOW_CONF:-1.0}"
GVHMR_WRIST_FILTER_MIX_WEIGHT="${GVHMR_WRIST_FILTER_MIX_WEIGHT:-0.35}"
GVHMR_WRIST_FILTER_HAMER_PENALTY="${GVHMR_WRIST_FILTER_HAMER_PENALTY:-0.0}"
GVHMR_WRIST_FILTER_FLIP_PENALTY="${GVHMR_WRIST_FILTER_FLIP_PENALTY:-0.15}"
GVHMR_WRIST_FILTER_BODY_PENALTY="${GVHMR_WRIST_FILTER_BODY_PENALTY:-0.35}"
GVHMR_WRIST_FILTER_MIX_PENALTY="${GVHMR_WRIST_FILTER_MIX_PENALTY:-0.1}"
GVHMR_WRIST_FILTER_INVALID_HAMER_PENALTY="${GVHMR_WRIST_FILTER_INVALID_HAMER_PENALTY:-1.0}"
GVHMR_WRIST_FILTER_INVALID_FLIP_PENALTY="${GVHMR_WRIST_FILTER_INVALID_FLIP_PENALTY:-1.0}"
GVHMR_WRIST_FILTER_INVALID_MIX_PENALTY="${GVHMR_WRIST_FILTER_INVALID_MIX_PENALTY:-0.2}"
GVHMR_FILTER_MANO_TEMPORAL="${GVHMR_FILTER_MANO_TEMPORAL:-0}"
GVHMR_TEMPORAL_FILTER_REPROJ_ERROR_THR="${GVHMR_TEMPORAL_FILTER_REPROJ_ERROR_THR:-75}"
GVHMR_TEMPORAL_FILTER_BBOX_WINDOW="${GVHMR_TEMPORAL_FILTER_BBOX_WINDOW:-30}"
GVHMR_TEMPORAL_FILTER_BBOX_SHRINK_RATIO="${GVHMR_TEMPORAL_FILTER_BBOX_SHRINK_RATIO:-0.65}"
GVHMR_TEMPORAL_FILTER_MIN_BBOX_FOREARM_RATIO="${GVHMR_TEMPORAL_FILTER_MIN_BBOX_FOREARM_RATIO:-0.45}"
GVHMR_TEMPORAL_FILTER_BBOX_JUMP_RATIO="${GVHMR_TEMPORAL_FILTER_BBOX_JUMP_RATIO:-1.75}"
GVHMR_TEMPORAL_FILTER_BBOX_OVERLAP_IOU="${GVHMR_TEMPORAL_FILTER_BBOX_OVERLAP_IOU:-0.55}"
GVHMR_TEMPORAL_FILTER_BBOX_SIZE_FLOOR_RATIO="${GVHMR_TEMPORAL_FILTER_BBOX_SIZE_FLOOR_RATIO:-0.75}"
GVHMR_TEMPORAL_FILTER_BBOX_SIZE_FLOOR_MIN_FOREARM_RATIO="${GVHMR_TEMPORAL_FILTER_BBOX_SIZE_FLOOR_MIN_FOREARM_RATIO:-0.45}"
GVHMR_TEMPORAL_FILTER_MAX_INTERP_GAP="${GVHMR_TEMPORAL_FILTER_MAX_INTERP_GAP:-60}"
GVHMR_TEMPORAL_FILTER_MAX_EDGE_HOLD="${GVHMR_TEMPORAL_FILTER_MAX_EDGE_HOLD:-15}"
GVHMR_FILTER_MANO_FINGERS="${GVHMR_FILTER_MANO_FINGERS:-0}"
GVHMR_FINGER_FILTER_SMOOTH_WINDOW="${GVHMR_FINGER_FILTER_SMOOTH_WINDOW:-17}"
GVHMR_FINGER_FILTER_RELIABLE_SMOOTH_WEIGHT="${GVHMR_FINGER_FILTER_RELIABLE_SMOOTH_WEIGHT:-0.22}"
GVHMR_FINGER_FILTER_WEAK_SMOOTH_WEIGHT="${GVHMR_FINGER_FILTER_WEAK_SMOOTH_WEIGHT:-0.90}"
GVHMR_FINGER_FILTER_BAD_SMOOTH_WEIGHT="${GVHMR_FINGER_FILTER_BAD_SMOOTH_WEIGHT:-1.0}"
GVHMR_FINGER_FILTER_MAX_INTERP_GAP="${GVHMR_FINGER_FILTER_MAX_INTERP_GAP:-120}"
GVHMR_FINGER_FILTER_MAX_EDGE_HOLD="${GVHMR_FINGER_FILTER_MAX_EDGE_HOLD:-90}"
GVHMR_FINGER_FILTER_MAX_JOINT_ANGLE_DELTA="${GVHMR_FINGER_FILTER_MAX_JOINT_ANGLE_DELTA:-0.25}"
GVHMR_FINGER_FILTER_MAX_JOINT_XYZ_DELTA="${GVHMR_FINGER_FILTER_MAX_JOINT_XYZ_DELTA:-0.04}"
GVHMR_FINGER_FILTER_WRIST_SMOOTH_WINDOW="${GVHMR_FINGER_FILTER_WRIST_SMOOTH_WINDOW:-11}"
GVHMR_FINGER_FILTER_WRIST_RELIABLE_SMOOTH_WEIGHT="${GVHMR_FINGER_FILTER_WRIST_RELIABLE_SMOOTH_WEIGHT:-0.25}"
GVHMR_FINGER_FILTER_WRIST_WEAK_SMOOTH_WEIGHT="${GVHMR_FINGER_FILTER_WRIST_WEAK_SMOOTH_WEIGHT:-0.80}"
GVHMR_FINGER_FILTER_WRIST_BAD_SMOOTH_WEIGHT="${GVHMR_FINGER_FILTER_WRIST_BAD_SMOOTH_WEIGHT:-1.0}"
GVHMR_FINGER_FILTER_MAX_WRIST_ANGLE_DELTA="${GVHMR_FINGER_FILTER_MAX_WRIST_ANGLE_DELTA:-0.25}"
GVHMR_FINGER_FILTER_HAND_SIZE_FLOOR_RATIO="${GVHMR_FINGER_FILTER_HAND_SIZE_FLOOR_RATIO:-0.96}"
GVHMR_FINGER_FILTER_OPEN_RESCUE_ENABLED="${GVHMR_FINGER_FILTER_OPEN_RESCUE_ENABLED:-1}"
GVHMR_FINGER_FILTER_OPEN_RESCUE_WEIGHT="${GVHMR_FINGER_FILTER_OPEN_RESCUE_WEIGHT:-0.30}"
GVHMR_FINGER_FILTER_OPEN_RESCUE_SMOOTH_WEIGHT="${GVHMR_FINGER_FILTER_OPEN_RESCUE_SMOOTH_WEIGHT:-0.70}"
GVHMR_RECOMPUTE_DIRECT_MANO="${GVHMR_RECOMPUTE_DIRECT_MANO:-0}"
GVHMR_RECOMPUTE_DIRECT_MANO_BATCH_SIZE="${GVHMR_RECOMPUTE_DIRECT_MANO_BATCH_SIZE:-256}"
GVHMR_DIAGNOSE_HAND="${GVHMR_DIAGNOSE_HAND:-0}"
GVHMR_DIAGNOSE_HAND_WIDTH="${GVHMR_DIAGNOSE_HAND_WIDTH:-960}"
GVHMR_SKIP_RENDER="${GVHMR_SKIP_RENDER:-0}"
GVHMR_LOW_MEMORY="${GVHMR_LOW_MEMORY:-1}"
SKIP_PHC="${SKIP_PHC:-0}"
FORCE_PHC="${FORCE_PHC:-0}"
GVHMR_HAND_REFINE_MODE="${GVHMR_HAND_REFINE_MODE:-raw}"
GVHMR_HAND_REPROJ_ERROR_THR="${GVHMR_HAND_REPROJ_ERROR_THR:-75}"
GVHMR_HAND_REPROJ_ERROR_RATIO_THR="${GVHMR_HAND_REPROJ_ERROR_RATIO_THR:-0.45}"
GVHMR_HAND_WRIST_OFFSET_MODE="${GVHMR_HAND_WRIST_OFFSET_MODE:-auto}"
GVHMR_HAND_WRIST_OFFSET_SMOOTH_WINDOW="${GVHMR_HAND_WRIST_OFFSET_SMOOTH_WINDOW:-15}"
GVHMR_HAND_SPIKE_MAD_MULTIPLIER="${GVHMR_HAND_SPIKE_MAD_MULTIPLIER:-8}"
GVHMR_HAND_SPIKE_ABS_THRESHOLD="${GVHMR_HAND_SPIKE_ABS_THRESHOLD:-1.2}"
GVHMR_HAND_REFINE_GAP_MERGE="${GVHMR_HAND_REFINE_GAP_MERGE:-2}"
GVHMR_HAND_REFINE_MAX_BURST="${GVHMR_HAND_REFINE_MAX_BURST:-10}"
GVHMR_HAND_REFINE_SMOOTH_WINDOW="${GVHMR_HAND_REFINE_SMOOTH_WINDOW:-0}"
RENDER_COMPARISON="${RENDER_COMPARISON:-0}"

# The inline numeric scorer below is retired.  It cannot safely interpret the
# current flattened Hand4Whole++ schema and also conflates robot-fit residuals
# with human-hand quality.  Formal pass/warn/fail quality reports are emitted
# only by scripts/run_pipeline_from_config.py -> evaluate_clip_quality.py.
if [ "${EVALUATE:-0}" != "0" ]; then
    echo "ERROR: EVALUATE refers to the retired inline evaluator and must remain 0." >&2
    echo "Use scripts/evaluate_clip_quality.py through run_pipeline_from_config.py instead." >&2
    exit 2
fi
EVALUATE=0
export EVALUATE
EVAL_SCORE_WEIGHT_HAND_VALID="${EVAL_SCORE_WEIGHT_HAND_VALID:-25}"
EVAL_SCORE_WEIGHT_HAND_REPROJ="${EVAL_SCORE_WEIGHT_HAND_REPROJ:-20}"
EVAL_SCORE_WEIGHT_CHAIN_ERROR="${EVAL_SCORE_WEIGHT_CHAIN_ERROR:-30}"
EVAL_SCORE_WEIGHT_TEMPORAL="${EVAL_SCORE_WEIGHT_TEMPORAL:-15}"
EVAL_SCORE_WEIGHT_HEIGHT="${EVAL_SCORE_WEIGHT_HEIGHT:-10}"

RUN_GMR="${RUN_GMR:-1}"

# Batch-specific defaults must be set before sourcing the shared defaults.
# Otherwise pipeline_defaults.sh initializes these to 0/hold and the later
# "${VAR:-...}" expressions cannot override those non-empty values.
GMR_AUTO_HAND_NPZ="${GMR_AUTO_HAND_NPZ:-1}"
GMR_HAND_INVALID_MODE="${GMR_HAND_INVALID_MODE:-interp}"

# Source shared GMR defaults while preserving the batch-specific values above.
source "${PIPELINE_ROOT}/GMR-master/pipeline_defaults.sh"

# --- batch-specific GMR overrides ---
GMR_SCRIPT="${GMR_SCRIPT:-${PIPELINE_ROOT}/GMR-master/run_show_gmr_batch.sh}"
GMR_OUT_ROOT="${GMR_OUT_ROOT:-$OUTPUT_BASE}"

G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; B='\033[0;34m'; N='\033[0m'

echo "============================================================"
echo "  Batch pipeline: dataset_new6 -> GVHMR-hand -> GMR"
echo "============================================================"
echo "Dataset: $DATASET"
if [ -n "$CLIP_FILTER" ]; then
    echo "Clip filter: $CLIP_FILTER"
fi
echo "Output:  $OUTPUT_BASE"
echo "Script:  $PIPELINE_SCRIPT"
echo "GMR:     $RUN_GMR ($GMR_EMBODIMENT_LABEL; body_asset=$GMR_ROBOT)"
echo "Work:    ${WORK_WIDTH}x${WORK_HEIGHT}, hand_backend=${GVHMR_HAND_BACKEND}, h4w_joint_source=${GVHMR_HAND4WHOLEPP_JOINT_SOURCE}, isolate_hand=${GVHMR_ISOLATE_HAND_PREPROCESS}, low_memory=${GVHMR_LOW_MEMORY}, vitpose_img_ds=${GVHMR_VITPOSE_IMG_DS}, hand_crop_scales=${GVHMR_HAMER_BBOX_RESCALE_CANDIDATES:-$GVHMR_HAMER_BBOX_RESCALE}, switch_penalty=${GVHMR_HAMER_CANDIDATE_SWITCH_PENALTY}, hand_conf=${GVHMR_HAND_KPT_CONF_THR}/${GVHMR_HAND_KPT_LOW_CONF_THR}, bbox_smooth=${GVHMR_HAND_BBOX_SMOOTHING}, bbox_jump=${GVHMR_HAND_BBOX_MAX_JUMP}, bbox_iou=${GVHMR_HAND_BBOX_OVERLAP_IOU}, bbox_ratio=${GVHMR_HAND_BBOX_COLLISION_SCORE_RATIO}, refine_steps=${GVHMR_HAMER_REFINE_STEPS}"
echo "Wrist:   filter=${GVHMR_FILTER_MANO_WRIST}, w_temp=${GVHMR_WRIST_FILTER_W_TEMP}, w_body=${GVHMR_WRIST_FILTER_W_BODY}"
echo "Hand crop tracking: mode=${GVHMR_HAND4WHOLEPP_CROP_TRACKING}, max_gap=${GVHMR_HAND4WHOLEPP_CROP_TRACKING_MAX_GAP}, max_prediction_gap=${GVHMR_HAND4WHOLEPP_CROP_TRACKING_MAX_PREDICTION_GAP}, direct_quality=${GVHMR_HAND4WHOLEPP_CROP_TRACKING_DIRECT_OBSERVATION_QUALITY}"
echo "Temporal filter: ${GVHMR_FILTER_MANO_TEMPORAL}, window=${GVHMR_TEMPORAL_FILTER_BBOX_WINDOW}, max_gap=${GVHMR_TEMPORAL_FILTER_MAX_INTERP_GAP}"
echo "Finger filter: ${GVHMR_FILTER_MANO_FINGERS}, window=${GVHMR_FINGER_FILTER_SMOOTH_WINDOW}, max_gap=${GVHMR_FINGER_FILTER_MAX_INTERP_GAP}, size_floor=${GVHMR_FINGER_FILTER_HAND_SIZE_FLOOR_RATIO}"
echo "Direct MANO recompute: ${GVHMR_RECOMPUTE_DIRECT_MANO}, batch_size=${GVHMR_RECOMPUTE_DIRECT_MANO_BATCH_SIZE}"
echo "Constraint profile: ${GVHMR_HAND_CONSTRAINT_PROFILE}"
echo "Hand diagnostic: ${GVHMR_DIAGNOSE_HAND}, width=${GVHMR_DIAGNOSE_HAND_WIDTH}"
echo "Palm:    source=${GMR_PALM_ROLL_SOURCE}, gain=${GMR_PALM_ROLL_GAIN}, max_delta=${GMR_PALM_ROLL_MAX_DELTA}, max_abs=${GMR_PALM_ROLL_MAX_ABS}, branch=${GMR_PALM_ROLL_BRANCH_MODE}"
echo "GMR hand: selection=${GMR_HAND_MODEL}, mode=${GMR_HAND_RETARGET_MODE}, sharpa=${GMR_SHARPA_HANDS}, brainco=${GMR_BRAINCO_HANDS}"
echo ""
mkdir -p "$OUTPUT_BASE"

mapfile -t VIDEOS < <(
    find "$DATASET" -maxdepth 1 -type f \
        \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.avi' -o -iname '*.mkv' -o -iname '*.m4v' \) \
        -print0 | sort -z | xargs -0 -r -n1 echo
)
if [ -n "$CLIP_FILTER" ]; then
    FILTERED_VIDEOS=()
    IFS=',' read -r -a CLIP_FILTERS <<< "$CLIP_FILTER"
    for video in "${VIDEOS[@]}"; do
        filename="$(basename "$video")"
        for filter in "${CLIP_FILTERS[@]}"; do
            filter="${filter#"${filter%%[![:space:]]*}"}"
            filter="${filter%"${filter##*[![:space:]]}"}"
            [ -n "$filter" ] || continue
            if [[ "$filename" == *"$filter"* ]]; then
                FILTERED_VIDEOS+=("$video")
                break
            fi
        done
    done
    VIDEOS=("${FILTERED_VIDEOS[@]}")
fi
TOTAL=${#VIDEOS[@]}

if [ "$TOTAL" -eq 0 ]; then
    echo -e "${R}No video files found in $DATASET${N}"
    exit 1
fi

echo "Found $TOTAL videos:"
for video in "${VIDEOS[@]}"; do
    echo "  $(basename "$video")"
done
echo ""

PASS=0
FAIL=0
FAIL_LIST=()
PASS_LIST=()
for i in "${!VIDEOS[@]}"; do
    VIDEO="${VIDEOS[$i]}"
    NAME="$(basename "$VIDEO")"
    NAME="${NAME%.*}"
    NUM=$((i + 1))
    CLIP_LOG="${OUTPUT_BASE}/${NAME}.pipeline.log"

    echo ""
    echo -e "${B}------------------------------------------------------------${N}"
    echo -e "${B}[$NUM/$TOTAL] $NAME${N}"
    echo -e "${B}------------------------------------------------------------${N}"

    INPUT_VIDEO="$VIDEO"
    if [ "$USE_WORK_VIDEO" = "1" ]; then
        mkdir -p "$WORK_DATASET"
        INPUT_VIDEO="${WORK_DATASET}/${NAME}.mp4"
        WORK_CONFIG_MARKER="${INPUT_VIDEO}.config"
        WORK_CONFIG="$(
            printf '%s\n' \
                "schema=2" \
                "source=$(realpath "$VIDEO")" \
                "source_stat=$(stat -c '%s:%Y' "$VIDEO")" \
                "width=$WORK_WIDTH" \
                "height=$WORK_HEIGHT" \
                "crf=$WORK_CRF" \
                "fps=$WORK_FPS"
        )"
        WORK_CONFIG_MATCH=0
        if [ -f "$WORK_CONFIG_MARKER" ] && [ "$(cat "$WORK_CONFIG_MARKER")" = "$WORK_CONFIG" ]; then
            WORK_CONFIG_MATCH=1
        fi
        # The marker validates the requested conversion, but it cannot prove
        # that the MP4 with the same name is intact.  A stale 50-frame
        # 960x720 cache can otherwise masquerade as the requested 1280x960
        # full clip and make the person tracker fail with no detections.
        WORK_MEDIA_MATCH=0
        if [ "$WORK_CONFIG_MATCH" = "1" ] && [ -f "$INPUT_VIDEO" ]; then
            if python - "$VIDEO" "$INPUT_VIDEO" "$WORK_WIDTH" "$WORK_HEIGHT" "$WORK_FPS" <<'PY'
import cv2
import math
import sys

source_path, work_path, max_width, max_height, target_fps = sys.argv[1:]
max_width, max_height, target_fps = int(max_width), int(max_height), float(target_fps)

def metadata(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {path}")
    values = (
        int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
        int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
        int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        float(cap.get(cv2.CAP_PROP_FPS)),
    )
    cap.release()
    return values

try:
    source_frames, _, _, source_fps = metadata(source_path)
    work_frames, work_width, work_height, work_fps = metadata(work_path)
    if min(source_frames, work_frames, work_width, work_height) <= 0:
        raise RuntimeError("missing frame or geometry metadata")
    if source_fps <= 0 or work_fps <= 0:
        raise RuntimeError("missing FPS metadata")
    # force_original_aspect_ratio=decrease must fit inside the target and
    # touch at least one target edge. Both dimensions are even after ffmpeg.
    geometry_ok = (
        work_width <= max_width
        and work_height <= max_height
        and (work_width == max_width or work_height == max_height)
        and work_width % 2 == 0
        and work_height % 2 == 0
    )
    duration_error = abs(work_frames / work_fps - source_frames / source_fps)
    duration_tolerance = max(0.25, 0.01 * (source_frames / source_fps))
    fps_ok = abs(work_fps - target_fps) <= 0.05
    if not geometry_ok or not fps_ok or duration_error > duration_tolerance:
        raise RuntimeError(
            f"source={source_frames}@{source_fps:.3f}; "
            f"work={work_frames}@{work_fps:.3f}, {work_width}x{work_height}; "
            f"expected<= {max_width}x{max_height} at {target_fps:.3f} FPS"
        )
except Exception as exc:
    print(f"work-video media validation failed: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
            then
                WORK_MEDIA_MATCH=1
            else
                echo "Cached work video failed media validation; rebuilding: $INPUT_VIDEO"
            fi
        fi
        if [ "$FORCE_WORK_VIDEO" = "1" ] || [ ! -f "$INPUT_VIDEO" ] || [ "$WORK_CONFIG_MATCH" != "1" ] || [ "$WORK_MEDIA_MATCH" != "1" ]; then
            echo -e "${B}Creating work video:${N} $INPUT_VIDEO"
            ffmpeg -hide_banner -loglevel error -y \
                -i "$VIDEO" \
                -vf "fps=${WORK_FPS},scale=${WORK_WIDTH}:${WORK_HEIGHT}:force_original_aspect_ratio=decrease,scale=trunc(iw/2)*2:trunc(ih/2)*2" \
                -c:v libx264 -preset veryfast -crf "$WORK_CRF" -pix_fmt yuv420p -an \
                "$INPUT_VIDEO"
            printf '%s\n' "$WORK_CONFIG" > "$WORK_CONFIG_MARKER"
        fi
    fi

    if GVHMR_BATCH_SIZE="$GVHMR_BATCH_SIZE" \
       GVHMR_HAMER_BATCH_SIZE="$GVHMR_HAMER_BATCH_SIZE" \
	       GVHMR_FORCE_HAND_PREPROCESS="$GVHMR_FORCE_HAND_PREPROCESS" \
	       GVHMR_ISOLATE_HAND_PREPROCESS="$GVHMR_ISOLATE_HAND_PREPROCESS" \
       GVHMR_HAND_BACKEND="$GVHMR_HAND_BACKEND" \
       GVHMR_WILOR_ROOT="$GVHMR_WILOR_ROOT" \
       GVHMR_WILOR_CHECKPOINT="$GVHMR_WILOR_CHECKPOINT" \
       GVHMR_WILOR_CONFIG="$GVHMR_WILOR_CONFIG" \
       GVHMR_WILOR_FAST="$GVHMR_WILOR_FAST" \
       GVHMR_HAND4WHOLEPP_ROOT="$GVHMR_HAND4WHOLEPP_ROOT" \
       GVHMR_HAND4WHOLEPP_SNAPSHOT="$GVHMR_HAND4WHOLEPP_SNAPSHOT" \
       GVHMR_HAND4WHOLEPP_PYTHON="$GVHMR_HAND4WHOLEPP_PYTHON" \
	       GVHMR_HAND4WHOLEPP_BATCH_SIZE="$GVHMR_HAND4WHOLEPP_BATCH_SIZE" \
	       GVHMR_HAND4WHOLEPP_YOLO_MODEL="$GVHMR_HAND4WHOLEPP_YOLO_MODEL" \
	       GVHMR_HAND4WHOLEPP_JOINT_SOURCE="$GVHMR_HAND4WHOLEPP_JOINT_SOURCE" \
       GVHMR_LOW_MEMORY="$GVHMR_LOW_MEMORY" \
       GVHMR_VITPOSE_IMG_DS="$GVHMR_VITPOSE_IMG_DS" \
       GVHMR_HAND_KPT_CONF_THR="$GVHMR_HAND_KPT_CONF_THR" \
       GVHMR_HAND_KPT_LOW_CONF_THR="$GVHMR_HAND_KPT_LOW_CONF_THR" \
       GVHMR_HAND_KPT_HI_MIN_KEYPOINTS="$GVHMR_HAND_KPT_HI_MIN_KEYPOINTS" \
       GVHMR_HAND_MIN_KEYPOINTS="$GVHMR_HAND_MIN_KEYPOINTS" \
       GVHMR_HAMER_BBOX_RESCALE="$GVHMR_HAMER_BBOX_RESCALE" \
       GVHMR_HAMER_BBOX_RESCALE_CANDIDATES="$GVHMR_HAMER_BBOX_RESCALE_CANDIDATES" \
       GVHMR_HAMER_CANDIDATE_SWITCH_PENALTY="$GVHMR_HAMER_CANDIDATE_SWITCH_PENALTY" \
       GVHMR_HAND_BBOX_MIN_SIZE="$GVHMR_HAND_BBOX_MIN_SIZE" \
       GVHMR_HAND_BBOX_SMOOTHING="$GVHMR_HAND_BBOX_SMOOTHING" \
       GVHMR_HAND_BBOX_MAX_JUMP="$GVHMR_HAND_BBOX_MAX_JUMP" \
       GVHMR_HAND_BBOX_OVERLAP_IOU="$GVHMR_HAND_BBOX_OVERLAP_IOU" \
       GVHMR_HAND_BBOX_COLLISION_SCORE_RATIO="$GVHMR_HAND_BBOX_COLLISION_SCORE_RATIO" \
       GVHMR_HAMER_REFINE_STEPS="$GVHMR_HAMER_REFINE_STEPS" \
       GVHMR_HAMER_REFINE_LR="$GVHMR_HAMER_REFINE_LR" \
	       GVHMR_HAMER_REFINE_CONF_THR="$GVHMR_HAMER_REFINE_CONF_THR" \
	       GVHMR_HAMER_REFINE_MIN_KEYPOINTS="$GVHMR_HAMER_REFINE_MIN_KEYPOINTS" \
	       GVHMR_HAMER_REFINE_POSE_PRIOR="$GVHMR_HAMER_REFINE_POSE_PRIOR" \
	       GVHMR_HAMER_REFINE_GLOBAL_PRIOR="$GVHMR_HAMER_REFINE_GLOBAL_PRIOR" \
	       GVHMR_FILTER_MANO_WRIST="$GVHMR_FILTER_MANO_WRIST" \
	       GVHMR_WRIST_FILTER_W_TEMP="$GVHMR_WRIST_FILTER_W_TEMP" \
	       GVHMR_WRIST_FILTER_W_BODY="$GVHMR_WRIST_FILTER_W_BODY" \
	       GVHMR_WRIST_FILTER_W_TEMP_LOW_CONF="$GVHMR_WRIST_FILTER_W_TEMP_LOW_CONF" \
	       GVHMR_WRIST_FILTER_W_BODY_LOW_CONF="$GVHMR_WRIST_FILTER_W_BODY_LOW_CONF" \
	       GVHMR_WRIST_FILTER_MIX_WEIGHT="$GVHMR_WRIST_FILTER_MIX_WEIGHT" \
	       GVHMR_WRIST_FILTER_HAMER_PENALTY="$GVHMR_WRIST_FILTER_HAMER_PENALTY" \
	       GVHMR_WRIST_FILTER_FLIP_PENALTY="$GVHMR_WRIST_FILTER_FLIP_PENALTY" \
	       GVHMR_WRIST_FILTER_BODY_PENALTY="$GVHMR_WRIST_FILTER_BODY_PENALTY" \
	       GVHMR_WRIST_FILTER_MIX_PENALTY="$GVHMR_WRIST_FILTER_MIX_PENALTY" \
		       GVHMR_WRIST_FILTER_INVALID_HAMER_PENALTY="$GVHMR_WRIST_FILTER_INVALID_HAMER_PENALTY" \
		       GVHMR_WRIST_FILTER_INVALID_FLIP_PENALTY="$GVHMR_WRIST_FILTER_INVALID_FLIP_PENALTY" \
		       GVHMR_WRIST_FILTER_INVALID_MIX_PENALTY="$GVHMR_WRIST_FILTER_INVALID_MIX_PENALTY" \
		       GVHMR_FILTER_MANO_TEMPORAL="$GVHMR_FILTER_MANO_TEMPORAL" \
		       GVHMR_HAND_CONSTRAINT_PROFILE="$GVHMR_HAND_CONSTRAINT_PROFILE" \
		       GVHMR_TEMPORAL_FILTER_REPROJ_ERROR_THR="$GVHMR_TEMPORAL_FILTER_REPROJ_ERROR_THR" \
		       GVHMR_TEMPORAL_FILTER_BBOX_WINDOW="$GVHMR_TEMPORAL_FILTER_BBOX_WINDOW" \
		       GVHMR_TEMPORAL_FILTER_BBOX_SHRINK_RATIO="$GVHMR_TEMPORAL_FILTER_BBOX_SHRINK_RATIO" \
		       GVHMR_TEMPORAL_FILTER_MIN_BBOX_FOREARM_RATIO="$GVHMR_TEMPORAL_FILTER_MIN_BBOX_FOREARM_RATIO" \
		       GVHMR_TEMPORAL_FILTER_BBOX_JUMP_RATIO="$GVHMR_TEMPORAL_FILTER_BBOX_JUMP_RATIO" \
		       GVHMR_TEMPORAL_FILTER_BBOX_OVERLAP_IOU="$GVHMR_TEMPORAL_FILTER_BBOX_OVERLAP_IOU" \
		       GVHMR_TEMPORAL_FILTER_BBOX_SIZE_FLOOR_RATIO="$GVHMR_TEMPORAL_FILTER_BBOX_SIZE_FLOOR_RATIO" \
		       GVHMR_TEMPORAL_FILTER_BBOX_SIZE_FLOOR_MIN_FOREARM_RATIO="$GVHMR_TEMPORAL_FILTER_BBOX_SIZE_FLOOR_MIN_FOREARM_RATIO" \
		       GVHMR_TEMPORAL_FILTER_MAX_INTERP_GAP="$GVHMR_TEMPORAL_FILTER_MAX_INTERP_GAP" \
		       GVHMR_TEMPORAL_FILTER_MAX_EDGE_HOLD="$GVHMR_TEMPORAL_FILTER_MAX_EDGE_HOLD" \
		       GVHMR_FILTER_MANO_FINGERS="$GVHMR_FILTER_MANO_FINGERS" \
		       GVHMR_FINGER_FILTER_SMOOTH_WINDOW="$GVHMR_FINGER_FILTER_SMOOTH_WINDOW" \
		       GVHMR_FINGER_FILTER_RELIABLE_SMOOTH_WEIGHT="$GVHMR_FINGER_FILTER_RELIABLE_SMOOTH_WEIGHT" \
		       GVHMR_FINGER_FILTER_WEAK_SMOOTH_WEIGHT="$GVHMR_FINGER_FILTER_WEAK_SMOOTH_WEIGHT" \
		       GVHMR_FINGER_FILTER_BAD_SMOOTH_WEIGHT="$GVHMR_FINGER_FILTER_BAD_SMOOTH_WEIGHT" \
		       GVHMR_FINGER_FILTER_MAX_INTERP_GAP="$GVHMR_FINGER_FILTER_MAX_INTERP_GAP" \
		       GVHMR_FINGER_FILTER_MAX_EDGE_HOLD="$GVHMR_FINGER_FILTER_MAX_EDGE_HOLD" \
		       GVHMR_FINGER_FILTER_MAX_JOINT_ANGLE_DELTA="$GVHMR_FINGER_FILTER_MAX_JOINT_ANGLE_DELTA" \
		       GVHMR_FINGER_FILTER_MAX_JOINT_XYZ_DELTA="$GVHMR_FINGER_FILTER_MAX_JOINT_XYZ_DELTA" \
		       GVHMR_FINGER_FILTER_WRIST_SMOOTH_WINDOW="$GVHMR_FINGER_FILTER_WRIST_SMOOTH_WINDOW" \
		       GVHMR_FINGER_FILTER_WRIST_RELIABLE_SMOOTH_WEIGHT="$GVHMR_FINGER_FILTER_WRIST_RELIABLE_SMOOTH_WEIGHT" \
		       GVHMR_FINGER_FILTER_WRIST_WEAK_SMOOTH_WEIGHT="$GVHMR_FINGER_FILTER_WRIST_WEAK_SMOOTH_WEIGHT" \
		       GVHMR_FINGER_FILTER_WRIST_BAD_SMOOTH_WEIGHT="$GVHMR_FINGER_FILTER_WRIST_BAD_SMOOTH_WEIGHT" \
		       GVHMR_FINGER_FILTER_MAX_WRIST_ANGLE_DELTA="$GVHMR_FINGER_FILTER_MAX_WRIST_ANGLE_DELTA" \
	       GVHMR_FINGER_FILTER_HAND_SIZE_FLOOR_RATIO="$GVHMR_FINGER_FILTER_HAND_SIZE_FLOOR_RATIO" \
	       GVHMR_FINGER_FILTER_OPEN_RESCUE_ENABLED="$GVHMR_FINGER_FILTER_OPEN_RESCUE_ENABLED" \
	       GVHMR_FINGER_FILTER_OPEN_RESCUE_WEIGHT="$GVHMR_FINGER_FILTER_OPEN_RESCUE_WEIGHT" \
		       GVHMR_FINGER_FILTER_OPEN_RESCUE_SMOOTH_WEIGHT="$GVHMR_FINGER_FILTER_OPEN_RESCUE_SMOOTH_WEIGHT" \
	       GVHMR_RECOMPUTE_DIRECT_MANO="$GVHMR_RECOMPUTE_DIRECT_MANO" \
	       GVHMR_RECOMPUTE_DIRECT_MANO_BATCH_SIZE="$GVHMR_RECOMPUTE_DIRECT_MANO_BATCH_SIZE" \
		       GVHMR_DIAGNOSE_HAND="$GVHMR_DIAGNOSE_HAND" \
		       GVHMR_DIAGNOSE_HAND_WIDTH="$GVHMR_DIAGNOSE_HAND_WIDTH" \
		       GVHMR_SKIP_RENDER="$GVHMR_SKIP_RENDER" \
	       GVHMR_HAND_REFINE_MODE="$GVHMR_HAND_REFINE_MODE" \
	       GVHMR_HAND_REPROJ_ERROR_THR="$GVHMR_HAND_REPROJ_ERROR_THR" \
	       GVHMR_HAND_REPROJ_ERROR_RATIO_THR="$GVHMR_HAND_REPROJ_ERROR_RATIO_THR" \
	       GVHMR_HAND_WRIST_OFFSET_MODE="$GVHMR_HAND_WRIST_OFFSET_MODE" \
	       GVHMR_HAND_WRIST_OFFSET_SMOOTH_WINDOW="$GVHMR_HAND_WRIST_OFFSET_SMOOTH_WINDOW" \
	       GVHMR_HAND_SPIKE_MAD_MULTIPLIER="$GVHMR_HAND_SPIKE_MAD_MULTIPLIER" \
	       GVHMR_HAND_SPIKE_ABS_THRESHOLD="$GVHMR_HAND_SPIKE_ABS_THRESHOLD" \
	       GVHMR_HAND_REFINE_GAP_MERGE="$GVHMR_HAND_REFINE_GAP_MERGE" \
	       GVHMR_HAND_REFINE_MAX_BURST="$GVHMR_HAND_REFINE_MAX_BURST" \
	       GVHMR_HAND_REFINE_SMOOTH_WINDOW="$GVHMR_HAND_REFINE_SMOOTH_WINDOW" \
	       GVHMR_EXPORT_FPS="$WORK_FPS" \
	       SKIP_PHC="$SKIP_PHC" \
	       FORCE_PHC="$FORCE_PHC" \
	       RENDER_COMPARISON="$RENDER_COMPARISON" \
	       bash "$PIPELINE_SCRIPT" "$INPUT_VIDEO" "${OUTPUT_BASE}/${NAME}" 2>&1 | tee "$CLIP_LOG"; then
        echo -e "${G}[$NUM/$TOTAL] PASS: $NAME${N}"
        PASS=$((PASS + 1))
        PASS_LIST+=("$NAME")
    else
        echo -e "${R}[$NUM/$TOTAL] FAIL: $NAME${N}"
        FAIL=$((FAIL + 1))
        FAIL_LIST+=("$NAME")
    fi
done

echo ""
echo "============================================================"
echo "  Batch complete: $PASS passed / $FAIL failed / $TOTAL total"
echo "============================================================"
if [ "$FAIL" -gt 0 ]; then
    echo -e "${Y}Failed clips:${N}"
    for name in "${FAIL_LIST[@]}"; do
        echo "  $name"
    done
fi
echo "Output directory: $OUTPUT_BASE"

if [ "$RUN_GMR" = "1" ] && [ "$PASS" -gt 0 ]; then
    PASS_CSV=""
    for name in "${PASS_LIST[@]}"; do
        PASS_CSV="${PASS_CSV:+$PASS_CSV,}$name"
    done
    RUN_GMR_INCLUDE_CLIPS="${GMR_INCLUDE_CLIPS:-$PASS_CSV}"
    echo ""
    echo "============================================================"
    echo "  GMR batch: $GMR_EMBODIMENT_LABEL"
    echo "============================================================"
    SHOW_ROOT="$OUTPUT_BASE" \
    OUT_ROOT="$GMR_OUT_ROOT" \
    GMR_SOURCE="$GMR_SOURCE" \
    GMR_ROBOT="$GMR_ROBOT" \
    GMR_INCLUDE_CLIPS="$RUN_GMR_INCLUDE_CLIPS" \
    GMR_MODEL_TYPE="$GMR_MODEL_TYPE" \
    GMR_AUTO_HAND_NPZ="$GMR_AUTO_HAND_NPZ" \
    GMR_RENDER="$GMR_RENDER" \
    GMR_COMPOSITE="$GMR_COMPOSITE" \
    GMR_HAND_INVALID_MODE="$GMR_HAND_INVALID_MODE" \
    GMR_HAND_INTERP_MAX_GAP="$GMR_HAND_INTERP_MAX_GAP" \
    GMR_HAND_RETARGET_MODE="$GMR_HAND_RETARGET_MODE" \
    GMR_HAND_ANGLE_SCALE="$GMR_HAND_ANGLE_SCALE" \
    GMR_HAND_CURL_POWER="$GMR_HAND_CURL_POWER" \
    GMR_HAND_WRIST_ORIENTATION_MODE="$GMR_HAND_WRIST_ORIENTATION_MODE" \
    GMR_FORCE_HAND_WRIST_ORIENTATION_OVERRIDE="$GMR_FORCE_HAND_WRIST_ORIENTATION_OVERRIDE" \
    GMR_RELAX_ORIENTATION_BODIES="$GMR_RELAX_ORIENTATION_BODIES" \
    GMR_PALM_ROLL_MODE="$GMR_PALM_ROLL_MODE" \
    GMR_PALM_ROLL_GAIN="$GMR_PALM_ROLL_GAIN" \
    GMR_PALM_ROLL_SOURCE="$GMR_PALM_ROLL_SOURCE" \
    GMR_PALM_ROLL_NORMAL_DOT_MIN="$GMR_PALM_ROLL_NORMAL_DOT_MIN" \
    GMR_PALM_ROLL_BODY_ALIGN="$GMR_PALM_ROLL_BODY_ALIGN" \
    GMR_PALM_ROLL_BODY_ALIGN_DOT_MIN="$GMR_PALM_ROLL_BODY_ALIGN_DOT_MIN" \
    GMR_LEFT_PALM_ROLL_SIGN="$GMR_LEFT_PALM_ROLL_SIGN" \
    GMR_RIGHT_PALM_ROLL_SIGN="$GMR_RIGHT_PALM_ROLL_SIGN" \
    GMR_PALM_ROLL_SMOOTH_WINDOW="$GMR_PALM_ROLL_SMOOTH_WINDOW" \
    GMR_PALM_ROLL_MAX_DELTA="$GMR_PALM_ROLL_MAX_DELTA" \
    GMR_PALM_ROLL_MAX_ABS="$GMR_PALM_ROLL_MAX_ABS" \
    GMR_PALM_ROLL_BRANCH_MODE="$GMR_PALM_ROLL_BRANCH_MODE" \
    GMR_PALM_ROLL_BRANCH_CANDIDATES="$GMR_PALM_ROLL_BRANCH_CANDIDATES" \
    GMR_PALM_ROLL_BRANCH_ANCHOR_FRAMES="$GMR_PALM_ROLL_BRANCH_ANCHOR_FRAMES" \
    GMR_PALM_ROLL_BRANCH_TRANSITION_WEIGHT="$GMR_PALM_ROLL_BRANCH_TRANSITION_WEIGHT" \
    GMR_PALM_ROLL_BRANCH_PENALTY="$GMR_PALM_ROLL_BRANCH_PENALTY" \
    GMR_PALM_ROLL_BRANCH_ANCHOR_PENALTY="$GMR_PALM_ROLL_BRANCH_ANCHOR_PENALTY" \
    GMR_PALM_ROLL_BRANCH_RANGE_PENALTY="$GMR_PALM_ROLL_BRANCH_RANGE_PENALTY" \
    GMR_WRIST_PITCH_YAW_STABILIZE="$GMR_WRIST_PITCH_YAW_STABILIZE" \
    GMR_WRIST_PITCH_NEUTRAL="$GMR_WRIST_PITCH_NEUTRAL" \
    GMR_WRIST_YAW_NEUTRAL="$GMR_WRIST_YAW_NEUTRAL" \
    GMR_WRIST_PITCH_YAW_SOFT_MARGIN_RATIO="$GMR_WRIST_PITCH_YAW_SOFT_MARGIN_RATIO" \
    GMR_SHARPA_HANDS="$GMR_SHARPA_HANDS" \
    GMR_SHARPA_AUTO_RETARGET="$GMR_SHARPA_AUTO_RETARGET" \
    GMR_SHARPA_HAND_NPZ_NAME="$GMR_SHARPA_HAND_NPZ_NAME" \
    GMR_SHARPA_MOUNT_POS="$GMR_SHARPA_MOUNT_POS" \
    GMR_SHARPA_LEFT_MOUNT_POS="$GMR_SHARPA_LEFT_MOUNT_POS" \
    GMR_SHARPA_RIGHT_MOUNT_POS="$GMR_SHARPA_RIGHT_MOUNT_POS" \
    GMR_SHARPA_MOUNT_QUAT="$GMR_SHARPA_MOUNT_QUAT" \
    GMR_SHARPA_LEFT_MOUNT_QUAT="$GMR_SHARPA_LEFT_MOUNT_QUAT" \
    GMR_SHARPA_RIGHT_MOUNT_QUAT="$GMR_SHARPA_RIGHT_MOUNT_QUAT" \
    GMR_SHARPA_STEPS="$GMR_SHARPA_STEPS" \
    GMR_SHARPA_INIT_STEPS="$GMR_SHARPA_INIT_STEPS" \
    GMR_SHARPA_WRIST_POS_COST="$GMR_SHARPA_WRIST_POS_COST" \
    GMR_SHARPA_WRIST_ORI_COST="$GMR_SHARPA_WRIST_ORI_COST" \
    GMR_SHARPA_FINGER_POS_COST="$GMR_SHARPA_FINGER_POS_COST" \
    GMR_SHARPA_JOINT_POS_COST="$GMR_SHARPA_JOINT_POS_COST" \
    GMR_SHARPA_POSTURE_COST="$GMR_SHARPA_POSTURE_COST" \
    GMR_SHARPA_TEMPORAL_COST="$GMR_SHARPA_TEMPORAL_COST" \
    GMR_SHARPA_LOW_CONF_TEMPORAL_GAIN="$GMR_SHARPA_LOW_CONF_TEMPORAL_GAIN" \
    GMR_SHARPA_MIN_TARGET_CONFIDENCE_SCALE="$GMR_SHARPA_MIN_TARGET_CONFIDENCE_SCALE" \
    GMR_SHARPA_RELIABILITY_SMOOTH_WINDOW="$GMR_SHARPA_RELIABILITY_SMOOTH_WINDOW" \
    GMR_SHARPA_LOW_CONF_RELIABILITY_THR="$GMR_SHARPA_LOW_CONF_RELIABILITY_THR" \
    GMR_SHARPA_SMOOTH_WINDOW="$GMR_SHARPA_SMOOTH_WINDOW" \
    GMR_SHARPA_MAX_DELTA="$GMR_SHARPA_MAX_DELTA" \
    GMR_SHARPA_MAX_ACCEL="$GMR_SHARPA_MAX_ACCEL" \
    GMR_SHARPA_LOW_CONF_ACCEL_SCALE="$GMR_SHARPA_LOW_CONF_ACCEL_SCALE" \
    GMR_SHARPA_REPROJ_GOOD_PX="$GMR_SHARPA_REPROJ_GOOD_PX" \
    GMR_SHARPA_REPROJ_BAD_PX="$GMR_SHARPA_REPROJ_BAD_PX" \
    GMR_SHARPA_REPROJ_GOOD_RATIO="$GMR_SHARPA_REPROJ_GOOD_RATIO" \
    GMR_SHARPA_REPROJ_BAD_RATIO="$GMR_SHARPA_REPROJ_BAD_RATIO" \
    GMR_SHARPA_MAX_PIP_BEND_DEG="$GMR_SHARPA_MAX_PIP_BEND_DEG" \
    GMR_SHARPA_MAX_DIP_BEND_DEG="$GMR_SHARPA_MAX_DIP_BEND_DEG" \
    GMR_SHARPA_MAX_SOURCE_BONE_DELTA_DEG="$GMR_SHARPA_MAX_SOURCE_BONE_DELTA_DEG" \
    GMR_SHARPA_MAX_SOURCE_BONE_LENGTH_RATIO="$GMR_SHARPA_MAX_SOURCE_BONE_LENGTH_RATIO" \
    GMR_SHARPA_ANATOMIC_REPAIR_MAX_GAP="$GMR_SHARPA_ANATOMIC_REPAIR_MAX_GAP" \
    GMR_HAND_MODEL="$GMR_HAND_MODEL" \
    GMR_BRAINCO_HANDS="$GMR_BRAINCO_HANDS" \
    GMR_BRAINCO_AUTO_RETARGET="$GMR_BRAINCO_AUTO_RETARGET" \
    GMR_BRAINCO_HAND_NPZ_NAME="$GMR_BRAINCO_HAND_NPZ_NAME" \
    GMR_BRAINCO_ROOT="$GMR_BRAINCO_ROOT" \
    GMR_BRAINCO_LEFT_MOUNT_POS="$GMR_BRAINCO_LEFT_MOUNT_POS" \
    GMR_BRAINCO_RIGHT_MOUNT_POS="$GMR_BRAINCO_RIGHT_MOUNT_POS" \
    GMR_BRAINCO_LEFT_MOUNT_QUAT="$GMR_BRAINCO_LEFT_MOUNT_QUAT" \
    GMR_BRAINCO_RIGHT_MOUNT_QUAT="$GMR_BRAINCO_RIGHT_MOUNT_QUAT" \
    GMR_CAMERA_SOURCE="$GMR_CAMERA_SOURCE" \
    GMR_MUJOCO_GL="$GMR_MUJOCO_GL" \
    GMR_COMPOSITE_PANEL_HEIGHT="$GMR_COMPOSITE_PANEL_HEIGHT" \
    bash "$GMR_SCRIPT"
    echo ""
    echo "All outputs under: $OUTPUT_BASE"
    echo "  2x2 composite:   <clip>/composite_2x2.mp4"
    echo "  GMR robot PKL:   <clip>/robot_motion.pkl"
    echo "  GMR render:      <clip>/${GMR_RENDER_NAME}"
fi

# ============================================================================
# Stage 3: Evaluation & scoring
# ============================================================================
if [ "$EVALUATE" = "1" ] && [ "$PASS" -gt 0 ]; then
    EVAL_CSV="${OUTPUT_BASE}/evaluation_scores.csv"
    EVAL_JSON="${OUTPUT_BASE}/evaluation_summary.json"
    EVAL_PY="${PY_LOCO:-${CONDA_BASE}/envs/locomotion/bin/python}"

    echo ""
    echo "============================================================"
    echo "  Evaluation: extracting precision metrics per clip"
    echo "============================================================"

    # Header
    echo "clip,overall_score,hand_score,chain_score,body_score,temporal_score,left_valid_pct,right_valid_pct,left_reproj_px,right_reproj_px,left_chain_mm,right_chain_mm,mean_speed_ms,root_height_m,grade" > "$EVAL_CSV"

    eval_clip_metrics() {
        local clip_dir="$1"
        local clip_name="$2"
        local eval_work="$clip_dir/.eval_work"
        mkdir -p "$eval_work"

        "$EVAL_PY" -c '
import json, os, sys, pickle, glob
import numpy as np

clip_dir = sys.argv[1]
clip_name = sys.argv[2]
eval_work = sys.argv[3]

def safe_mean(arr, default=0.0):
    a = np.asarray(arr).ravel()
    return float(a.mean()) if len(a) > 0 else default

def safe_median(arr, default=0.0):
    a = np.asarray(arr).ravel()
    return float(np.median(a)) if len(a) > 0 else default

def safe_p95(arr, default=0.0):
    a = np.asarray(arr).ravel()
    return float(np.percentile(a, 95)) if len(a) > 0 else default

metrics = {"clip": clip_name, "errors": []}

# --- 1. GVHMR hand metrics (from mano_params.pt) ---
mano_candidates = [
    os.path.join(clip_dir, "mano_params_finger_fixed.pt"),
    os.path.join(clip_dir, "mano_params_temporal_fixed.pt"),
    os.path.join(clip_dir, "mano_params_wrist_fixed.pt"),
]
gvhmr_out = os.path.join(clip_dir, "gvhmr_out")
if os.path.isdir(gvhmr_out):
    mano_candidates.append(os.path.join(gvhmr_out, "mano_params.pt"))

mano_path = None
for c in mano_candidates:
    if os.path.isfile(c):
        mano_path = c
        break

if mano_path:
    try:
        import torch
        mano = torch.load(mano_path, map_location="cpu", weights_only=False)
        for side in ["left", "right"]:
            if side in mano:
                m = mano[side]
                reproj = np.array(m.get("reprojection_error_px", []))
                valid = np.array(m.get("valid", []), dtype=bool)
                n_valid = int(valid.sum())
                n_total = len(valid)
                metrics[f"{side}_valid_frames"] = f"{n_valid}/{n_total}"
                metrics[f"{side}_valid_ratio"] = n_valid / max(n_total, 1)
                if n_valid > 0:
                    r = reproj[valid]
                    metrics[f"{side}_reproj_mean_px"] = float(r.mean())
                    metrics[f"{side}_reproj_median_px"] = float(np.median(r))
                    metrics[f"{side}_reproj_p95_px"] = float(np.percentile(r, 95))
                    # Quality tiers
                    metrics[f"{side}_reproj_good_pct"] = float((r < 15).sum() / n_valid * 100)
                    metrics[f"{side}_reproj_bad_pct"] = float((r >= 50).sum() / n_valid * 100)
                else:
                    metrics[f"{side}_reproj_mean_px"] = -1.0
    except Exception as e:
        metrics["errors"].append(f"mano_load: {e}")

# --- 2. SMPL body metrics (from NPZ) ---
npz_candidates = [
    os.path.join(clip_dir, "001_phc_smoothed_grounded.npz"),
    os.path.join(clip_dir, "001_phc_grounded.npz"),
    os.path.join(clip_dir, "001_smoothed.npz"),
    os.path.join(clip_dir, "001_converted.npz"),
]
npz_path = None
for c in npz_candidates:
    if os.path.isfile(c):
        npz_path = c
        break

if npz_path:
    try:
        data = np.load(npz_path, allow_pickle=True)
        trans = np.array(data.get("trans", data.get("root_trans", [])))
        metrics["num_frames"] = len(trans)
        if len(trans) > 1:
            vel = np.linalg.norm(np.diff(trans, axis=0), axis=-1) * 30
            metrics["mean_speed_ms"] = float(vel.mean())
            metrics["max_speed_ms"] = float(vel.max())
            # Temporal quality: root jumps
            step = np.linalg.norm(np.diff(trans, axis=0), axis=-1)
            med = np.median(step)
            mad = np.median(np.abs(step - med))
            thr = med + 10 * max(mad, 0.001)
            bad_jumps = int((step > max(thr, 0.25)).sum())
            metrics["root_jump_frames"] = bad_jumps
            metrics["root_jump_ratio"] = bad_jumps / max(len(step), 1)
        if trans.shape[1] >= 3:
            z = trans[:, 2]
            metrics["root_height_mean_m"] = float(z.mean())
            metrics["root_height_min_m"] = float(z.min())
    except Exception as e:
        metrics["errors"].append(f"smp_load: {e}")

# --- 3. GMR robot metrics (from robot_motion.pkl) ---
gmr_out = os.path.join(clip_dir, "gmr_out") if os.path.isdir(os.path.join(clip_dir, "gmr_out")) else clip_dir
pkl_candidates = glob.glob(os.path.join(gmr_out, "**", "robot_motion.pkl"), recursive=True)
if not pkl_candidates:
    pkl_candidates = glob.glob(os.path.join(clip_dir, "**", "robot_motion.pkl"), recursive=True)

if pkl_candidates:
    pkl_path = pkl_candidates[0]
    try:
        with open(pkl_path, "rb") as f:
            robot = pickle.load(f)
        if "dof_pos" in robot and "dof_names" in robot:
            dof = np.array(robot["dof_pos"])
            dof_names = list(robot["dof_names"])
            metrics["robot_dof_count"] = len(dof_names)
            # Wrist DOF analysis
            wrist_dofs = [n for n in dof_names if "wrist" in n.lower()]
            if wrist_dofs:
                idxs = [dof_names.index(n) for n in wrist_dofs]
                w = np.rad2deg(dof[:, idxs])
                metrics["wrist_dof_mean_abs_deg"] = float(np.abs(w).mean())
                metrics["wrist_dof_range_deg"] = float(w.max() - w.min())
    except Exception as e:
        metrics["errors"].append(f"pkl_load: {e}")

# --- 4. Sharpa hand chain_error (from sharpa NPZ) ---
sharpa_candidates = (
    glob.glob(os.path.join(clip_dir, "**", "*sharpa_chain_hands.npz"), recursive=True)
    + glob.glob(os.path.join(clip_dir, "**", "*sharpa_hands.npz"), recursive=True)
)
for side in ["left", "right"]:
    for sp in sharpa_candidates:
        try:
            sd = np.load(sp, allow_pickle=True)
            for key in [f"{side}_chain_error_mean", f"{side}_chain_error_p95"]:
                if key in sd:
                    val = float(np.asarray(sd[key]).ravel()[0])
                    metrics[key] = val  # in meters
        except Exception:
            pass

# --- 5. Compute scores ---
# Weights
W_HAND_VALID = float(os.environ.get("EVAL_SCORE_WEIGHT_HAND_VALID", 25))
W_HAND_REPROJ = float(os.environ.get("EVAL_SCORE_WEIGHT_HAND_REPROJ", 20))
W_CHAIN_ERROR = float(os.environ.get("EVAL_SCORE_WEIGHT_CHAIN_ERROR", 30))
W_TEMPORAL = float(os.environ.get("EVAL_SCORE_WEIGHT_TEMPORAL", 15))
W_HEIGHT = float(os.environ.get("EVAL_SCORE_WEIGHT_HEIGHT", 10))
TOTAL_WEIGHT = W_HAND_VALID + W_HAND_REPROJ + W_CHAIN_ERROR + W_TEMPORAL + W_HEIGHT

subs = {}
score = 0.0

# Hand valid score
hv = 0.0
for side in ["left", "right"]:
    vr = metrics.get(f"{side}_valid_ratio", 0.0)
    hv += vr * (W_HAND_VALID / 2)
subs["hand_valid"] = round(hv, 1)
score += hv

# Hand reprojection score
hr = 0.0
for side in ["left", "right"]:
    rp = metrics.get(f"{side}_reproj_mean_px", -1.0)
    if rp < 0:
        continue
    side_score = W_HAND_REPROJ / 2
    if rp > 15:
        side_score = max(0, side_score - (rp - 15) * (W_HAND_REPROJ / 60))
    hr += side_score
subs["hand_reproj"] = round(hr, 1)
score += hr

# Chain error score
ce = 0.0
for side in ["left", "right"]:
    cem = metrics.get(f"{side}_chain_error_mean", -1.0)
    if cem < 0:
        # No Sharpa data → skip this component, redistribute weight
        ce += W_CHAIN_ERROR / 2
        continue
    cem_mm = cem * 1000.0
    side_score = W_CHAIN_ERROR / 2
    if cem_mm > 10:
        side_score = max(0, side_score - (cem_mm - 10) * (W_CHAIN_ERROR / 60))
    ce += side_score
subs["chain_error"] = round(ce, 1)
score += ce

# Temporal score
ts = W_TEMPORAL
jr = metrics.get("root_jump_ratio", 0.0)
if jr > 0.02:
    ts = max(0, ts - (jr - 0.02) * W_TEMPORAL * 25)
subs["temporal"] = round(ts, 1)
score += ts

# Height score
hs = W_HEIGHT
rh = metrics.get("root_height_mean_m", 0.0)
if rh < 0.1 or rh > 3.0:
    hs = max(0, hs - W_HEIGHT * 0.8)
elif rh < 0.3:
    hs = max(0, hs - (0.3 - rh) * W_HEIGHT * 3)
subs["height"] = round(hs, 1)
score += hs

# Normalize
overall = round(score / TOTAL_WEIGHT * 100, 1)
overall = max(0.0, min(100.0, overall))

# Grade
if overall >= 90:
    grade = "A"
elif overall >= 75:
    grade = "B"
elif overall >= 60:
    grade = "C"
elif overall >= 40:
    grade = "D"
else:
    grade = "F"

metrics["overall_score"] = overall
metrics["grade"] = grade
metrics["sub_scores"] = subs
metrics["score_breakdown"] = (
    f"hand_valid={subs['hand_valid']:.1f}/{W_HAND_VALID:.0f} "
    f"hand_reproj={subs['hand_reproj']:.1f}/{W_HAND_REPROJ:.0f} "
    f"chain_error={subs['chain_error']:.1f}/{W_CHAIN_ERROR:.0f} "
    f"temporal={subs['temporal']:.1f}/{W_TEMPORAL:.0f} "
    f"height={subs['height']:.1f}/{W_HEIGHT:.0f}"
)

# Write per-clip JSON
json_path = os.path.join(eval_work, "metrics.json")
# Convert numpy values for JSON
clean = {}
for k, v in metrics.items():
    if isinstance(v, (np.floating,)):
        clean[k] = float(v)
    elif isinstance(v, (np.integer,)):
        clean[k] = int(v)
    elif isinstance(v, (np.ndarray,)):
        clean[k] = v.tolist()
    else:
        clean[k] = v
with open(json_path, "w") as f:
    json.dump(clean, f, indent=2, ensure_ascii=False, default=str)

# CSV row
csv_path = os.path.join(eval_work, "row.csv")
with open(csv_path, "w") as f:
    f.write(
        f"{clip_name},{overall},"
        f"{subs['hand_valid']:.1f},{subs['chain_error']:.1f},"
        f"{subs['height']:.1f},{subs['temporal']:.1f},"
        f"{metrics.get('left_valid_ratio', 0):.2f},{metrics.get('right_valid_ratio', 0):.2f},"
        f"{metrics.get('left_reproj_mean_px', -1):.1f},{metrics.get('right_reproj_mean_px', -1):.1f},"
        f"{metrics.get('left_chain_error_mean', -1)*1000:.1f},{metrics.get('right_chain_error_mean', -1)*1000:.1f},"
        f"{metrics.get('mean_speed_ms', -1):.2f},{metrics.get('root_height_mean_m', -1):.3f},"
        f"{grade}\n"
    )

print(f"EVAL_OK {clip_name} score={overall} grade={grade}")
' "$clip_dir" "$clip_name" "$eval_work" 2>&1
    }

    ALL_SCORES=()
    ALL_GRADES=()
    EVAL_PASS=0
    EVAL_FAIL=0

    for name in "${PASS_LIST[@]}"; do
        clip_dir="${OUTPUT_BASE}/${name}"
        echo -n "  [$name] "
        result="$(eval_clip_metrics "$clip_dir" "$name" || echo "EVAL_ERR")"

        if echo "$result" | grep -q "EVAL_OK"; then
            score_line="$(echo "$result" | grep "EVAL_OK")"
            echo -e "${G}${score_line}${N}"

            # Append CSV row
            row_file="${clip_dir}/.eval_work/row.csv"
            if [ -f "$row_file" ]; then
                cat "$row_file" >> "$EVAL_CSV"
                csv_score="$(cut -d',' -f2 "$row_file")"
                csv_grade="$(cut -d',' -f16 "$row_file" | tr -d '\n\r')"
                ALL_SCORES+=("$csv_score")
                ALL_GRADES+=("$csv_grade")
            fi
            EVAL_PASS=$((EVAL_PASS + 1))
        else
            echo -e "${Y}metrics extraction failed, see ${clip_dir}/.eval_work${N}"
            EVAL_FAIL=$((EVAL_FAIL + 1))
            echo "${name},ERR,,,,,,,,,,,,,," >> "$EVAL_CSV"
        fi
    done

    # --- Aggregate statistics ---
    echo ""
    echo "============================================================"
    echo "  Evaluation Summary"
    echo "============================================================"

    "$EVAL_PY" -c "
import json, sys, os
import numpy as np

csv_path = '$EVAL_CSV'
if not os.path.isfile(csv_path):
    print('No evaluation CSV found.')
    sys.exit(0)

scores = []
grades = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
left_reprojs = []
right_reprojs = []
left_chains = []
right_chains = []
speeds = []
clip_count = 0

with open(csv_path) as f:
    header = f.readline()
    for line in f:
        parts = line.strip().split(',')
        if len(parts) < 16:
            continue
        clip_count += 1
        try:
            s = float(parts[1])
            if s > 0:
                scores.append(s)
        except: pass
        grade = parts[15].strip()
        if grade in grades:
            grades[grade] += 1
        try:
            lr = float(parts[7])
            if lr > 0: left_reprojs.append(lr)
        except: pass
        try:
            rr = float(parts[8])
            if rr > 0: right_reprojs.append(rr)
        except: pass
        try:
            lc = float(parts[9])
            if lc > 0: left_chains.append(lc)
        except: pass
        try:
            rc = float(parts[10])
            if rc > 0: right_chains.append(rc)
        except: pass
        try:
            sp = float(parts[11])
            if sp > 0: speeds.append(sp)
        except: pass

summary = {
    'clips_evaluated': clip_count,
    'overall_score_mean': round(float(np.mean(scores)), 1) if scores else 'N/A',
    'overall_score_median': round(float(np.median(scores)), 1) if scores else 'N/A',
    'overall_score_min': round(float(np.min(scores)), 1) if scores else 'N/A',
    'overall_score_max': round(float(np.max(scores)), 1) if scores else 'N/A',
    'grade_distribution': grades,
    'left_reproj_mean_px': round(float(np.mean(left_reprojs)), 1) if left_reprojs else 'N/A',
    'right_reproj_mean_px': round(float(np.mean(right_reprojs)), 1) if right_reprojs else 'N/A',
    'left_chain_error_mean_mm': round(float(np.mean(left_chains)), 1) if left_chains else 'N/A',
    'right_chain_error_mean_mm': round(float(np.mean(right_chains)), 1) if right_chains else 'N/A',
}

with open('$EVAL_JSON', 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f'  Clips evaluated:    {clip_count}')
print(f'  Overall Score:       mean={summary[\"overall_score_mean\"]}  median={summary[\"overall_score_median\"]}  range=[{summary[\"overall_score_min\"]}, {summary[\"overall_score_max\"]}]')
print(f'  Grade distribution:  A={grades[\"A\"]}  B={grades[\"B\"]}  C={grades[\"C\"]}  D={grades[\"D\"]}  F={grades[\"F\"]}')
if left_reprojs:
    print(f'  Left reproj (px):    mean={summary[\"left_reproj_mean_px\"]}')
if right_reprojs:
    print(f'  Right reproj (px):   mean={summary[\"right_reproj_mean_px\"]}')
if left_chains:
    print(f'  Left chain (mm):     mean={summary[\"left_chain_error_mean_mm\"]}')
if right_chains:
    print(f'  Right chain (mm):    mean={summary[\"right_chain_error_mean_mm\"]}')
print(f'')
print(f'  Per-clip details:  $EVAL_CSV')
print(f'  Summary JSON:      $EVAL_JSON')
"

    # Update the output listing
    echo ""
    echo "All outputs under: $OUTPUT_BASE"
    echo "  2x2 composite:     <clip>/composite_2x2.mp4"
    echo "  GMR robot PKL:     <clip>/robot_motion.pkl"
    echo "  GMR render:        <clip>/${GMR_RENDER_NAME}"
    echo "  Per-clip metrics:  <clip>/.eval_work/metrics.json"
    echo "  Evaluation CSV:    $EVAL_CSV"
    echo "  Evaluation JSON:   $EVAL_JSON"
fi

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
