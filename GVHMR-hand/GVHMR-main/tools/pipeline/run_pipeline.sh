#!/bin/bash
# ============================================================================
# Hand-aware pipeline: VIDEO -> GVHMR-hand -> Locomotion -> Smooth -> PHC -> MP4
# ============================================================================
# Usage:
#   bash tools/pipeline/run_pipeline.sh your_video.mp4 [output_dir]
#   SKIP_PHC=1 GVHMR_SKIP_RENDER=1 bash tools/pipeline/run_pipeline.sh your_video.mp4
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GVHMR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PIPELINE_ROOT="${PIPELINE_ROOT:-$(cd "${GVHMR}/../.." && pwd)}"
LEGACY_PIPELINE="${LEGACY_PIPELINE:-${PIPELINE_ROOT}/GVHMR-main/tools/pipeline}"
LOCO="${LOCO:-${PIPELINE_ROOT}/locomotion_pipeline-main}"
PHC="${PHC:-${PIPELINE_ROOT}/phc-dev-felix-pipeline}"

CONDA_BASE="${CONDA_BASE:-${HOME}/miniconda3}"
PY_LOCO="${PY_LOCO:-${CONDA_BASE}/envs/locomotion/bin/python}"
PY_GVHMR="${PY_GVHMR:-${PY_LOCO}}"
PY_PHC="${PY_PHC:-${CONDA_BASE}/envs/phc/bin/python}"
PY_PHC_PREFIX="${PY_PHC_PREFIX:-$(cd "$(dirname "$PY_PHC")/.." 2>/dev/null && pwd || true)}"

VIDEO="${1:?Usage: bash run_pipeline.sh <video.mp4> [output_dir]}"
VIDEO="$(realpath "$VIDEO")"
VIDEO_STEM="$(basename "$VIDEO")"
VIDEO_NAME="${VIDEO_STEM%.*}"
OUTPUT_ROOT="${2:-${PIPELINE_ROOT}/output_dir/${VIDEO_NAME}_hand_pipeline}"
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"

SKIP_PHC="${SKIP_PHC:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FORCE_PHC="${FORCE_PHC:-0}"
FORCE_LOCO="${FORCE_LOCO:-0}"
FORCE_SMOOTH="${FORCE_SMOOTH:-0}"
MAX_FRAMES="${MAX_FRAMES:-0}"
COMPARISON_FPS="${COMPARISON_FPS:-30}"
PANEL_SIZE="${PANEL_SIZE:-600}"
RENDER_COMPARISON="${RENDER_COMPARISON:-0}"

GVHMR_STATIC_CAM="${GVHMR_STATIC_CAM:-1}"
GVHMR_SKIP_RENDER="${GVHMR_SKIP_RENDER:-0}"
GVHMR_LOW_MEMORY="${GVHMR_LOW_MEMORY:-1}"
GVHMR_BATCH_SIZE="${GVHMR_BATCH_SIZE:-1}"
GVHMR_PERSON_IDX="${GVHMR_PERSON_IDX:-0}"
GVHMR_EXPORT_FPS="${GVHMR_EXPORT_FPS:-30}"
GVHMR_FORCE_HAND_PREPROCESS="${GVHMR_FORCE_HAND_PREPROCESS:-0}"
GVHMR_ISOLATE_HAND_PREPROCESS="${GVHMR_ISOLATE_HAND_PREPROCESS:-auto}"
GVHMR_HAND_BACKEND="${GVHMR_HAND_BACKEND:-hand4wholepp}"
GVHMR_HAND_CONSTRAINT_PROFILE="${GVHMR_HAND_CONSTRAINT_PROFILE:-custom}"
GVHMR_WILOR_ROOT="${GVHMR_WILOR_ROOT:-${GVHMR}/third-party/WiLoR}"
GVHMR_WILOR_CHECKPOINT="${GVHMR_WILOR_CHECKPOINT:-${GVHMR_WILOR_ROOT}/pretrained_models/wilor_final.ckpt}"
GVHMR_WILOR_CONFIG="${GVHMR_WILOR_CONFIG:-${GVHMR_WILOR_ROOT}/pretrained_models/model_config.yaml}"
GVHMR_WILOR_FAST="${GVHMR_WILOR_FAST:-0}"
GVHMR_HAND4WHOLEPP_ROOT="${GVHMR_HAND4WHOLEPP_ROOT:-${GVHMR}/third-party/Hand4Whole-plus-plus_RELEASE}"
GVHMR_HAND4WHOLEPP_SNAPSHOT="${GVHMR_HAND4WHOLEPP_SNAPSHOT:-${GVHMR_HAND4WHOLEPP_ROOT}/demo/snapshot_6.pth}"
GVHMR_HAND4WHOLEPP_PYTHON="${GVHMR_HAND4WHOLEPP_PYTHON:-$PY_GVHMR}"
GVHMR_HAND4WHOLEPP_BATCH_SIZE="${GVHMR_HAND4WHOLEPP_BATCH_SIZE:-4}"
GVHMR_HAND4WHOLEPP_YOLO_MODEL="${GVHMR_HAND4WHOLEPP_YOLO_MODEL:-yolo11n.pt}"
GVHMR_HAND4WHOLEPP_JOINT_SOURCE="${GVHMR_HAND4WHOLEPP_JOINT_SOURCE:-direct_mano}"
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
GVHMR_HAMER_BATCH_SIZE="${GVHMR_HAMER_BATCH_SIZE:-2}"
GVHMR_HAMER_REFINE_STEPS="${GVHMR_HAMER_REFINE_STEPS:-6}"
GVHMR_HAMER_REFINE_LR="${GVHMR_HAMER_REFINE_LR:-0.03}"
GVHMR_HAMER_REFINE_CONF_THR="${GVHMR_HAMER_REFINE_CONF_THR:-0.45}"
GVHMR_HAMER_REFINE_MIN_KEYPOINTS="${GVHMR_HAMER_REFINE_MIN_KEYPOINTS:-6}"
GVHMR_HAMER_REFINE_POSE_PRIOR="${GVHMR_HAMER_REFINE_POSE_PRIOR:-0.02}"
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
GVHMR_FINGER_FILTER_OPEN_RESCUE_WEIGHT="${GVHMR_FINGER_FILTER_OPEN_RESCUE_WEIGHT:-0.30}"
GVHMR_FINGER_FILTER_OPEN_RESCUE_SMOOTH_WEIGHT="${GVHMR_FINGER_FILTER_OPEN_RESCUE_SMOOTH_WEIGHT:-0.70}"
GVHMR_DIAGNOSE_HAND="${GVHMR_DIAGNOSE_HAND:-0}"
GVHMR_DIAGNOSE_HAND_WIDTH="${GVHMR_DIAGNOSE_HAND_WIDTH:-960}"
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

PHC_GRAVITY_AXIS="${PHC_GRAVITY_AXIS:-neg_y}"
PHC_PRIMITIVE="${PHC}/output/HumanoidIm/phc_3/Humanoid.pth"
PHC_COMPOSER="${PHC}/output/HumanoidIm/phc_comp_3/Humanoid.pth"
PHC_CONTROL_DECIMATION="${PHC_CONTROL_DECIMATION:-2}"
PHC_FIX_HEIGHT="${PHC_FIX_HEIGHT:-ankle_fix}"
PHC_KP_SCALE="${PHC_KP_SCALE:-1.0}"
PHC_EXTRA_OVERRIDES="${PHC_EXTRA_OVERRIDES:-}"
PHC_ZERO_OUT_FAR="${PHC_ZERO_OUT_FAR:-0}"
PHC_ENABLE_EARLY_TERMINATION="${PHC_ENABLE_EARLY_TERMINATION:-0}"
PHC_POST_CHECK_SPEED="${PHC_POST_CHECK_SPEED:-0}"
PHC_SPEED_ACC_THRESHOLD="${PHC_SPEED_ACC_THRESHOLD:-14.7}"
PHC_POST_CHECK_MIN_LENGTH="${PHC_POST_CHECK_MIN_LENGTH:-30}"
PHC_POST_SMOOTH="${PHC_POST_SMOOTH:-1}"
PHC_POST_SMOOTH_POSE_WINDOW="${PHC_POST_SMOOTH_POSE_WINDOW:-7}"
PHC_POST_SMOOTH_TRANS_WINDOW="${PHC_POST_SMOOTH_TRANS_WINDOW:-5}"
PHC_POST_SMOOTH_ACC_THRESHOLD="${PHC_POST_SMOOTH_ACC_THRESHOLD:-14.7}"
PHC_POST_SMOOTH_ROOT_STEP_THRESHOLD="${PHC_POST_SMOOTH_ROOT_STEP_THRESHOLD:-30.0}"
PHC_POST_SMOOTH_JOINT_STEP_THRESHOLD="${PHC_POST_SMOOTH_JOINT_STEP_THRESHOLD:-30.0}"
PHC_POST_SMOOTH_JOINT_ACC_THRESHOLD="${PHC_POST_SMOOTH_JOINT_ACC_THRESHOLD:-250.0}"
PHC_POST_SMOOTH_MAD_MULTIPLIER="${PHC_POST_SMOOTH_MAD_MULTIPLIER:-6.0}"
PHC_POST_SMOOTH_DILATE="${PHC_POST_SMOOTH_DILATE:-1}"
PHC_GROUND_FIX="${PHC_GROUND_FIX:-1}"
PHC_GROUND_FIX_MODE="${PHC_GROUND_FIX_MODE:-global_min}"
PHC_GROUND_CLEARANCE="${PHC_GROUND_CLEARANCE:-0.005}"
PHC_COMPARISON_USE_RAW_ISAAC="${PHC_COMPARISON_USE_RAW_ISAAC:-1}"

LOCO_CHECK_PENETRATION="${LOCO_CHECK_PENETRATION:-0}"
LOCO_CHECK_SPEED="${LOCO_CHECK_SPEED:-0}"
LOCO_GRAVITY_ALIGNMENT="${LOCO_GRAVITY_ALIGNMENT:-0}"
LOCO_OPTIM_HEIGHT="${LOCO_OPTIM_HEIGHT:-1}"
LOCO_GRAVITY_AXIS="${LOCO_GRAVITY_AXIS:-y-}"

ISAAC_RECORD_WIDTH="${ISAAC_RECORD_WIDTH:-960}"
ISAAC_RECORD_HEIGHT="${ISAAC_RECORD_HEIGHT:-720}"
ISAAC_RECORD_FPS="${ISAAC_RECORD_FPS:-30}"
ISAAC_RECORD_FOV="${ISAAC_RECORD_FOV:-70.0}"
ISAAC_CAMERA_DISTANCE="${ISAAC_CAMERA_DISTANCE:-5.0}"
ISAAC_CAMERA_SIDE="${ISAAC_CAMERA_SIDE:-1.2}"
ISAAC_CAMERA_HEIGHT="${ISAAC_CAMERA_HEIGHT:-1.2}"
ISAAC_CAMERA_TARGET_HEIGHT="${ISAAC_CAMERA_TARGET_HEIGHT:-0.45}"
ISAAC_CAMERA_SMOOTHING="${ISAAC_CAMERA_SMOOTHING:-0.25}"
ISAAC_CAMERA_MODE="${ISAAC_CAMERA_MODE:-gvhmr_static}"
ISAAC_CAMERA_PATH="${ISAAC_CAMERA_PATH:-}"
ISAAC_CAMERA_USE_GVHMR_FOV="${ISAAC_CAMERA_USE_GVHMR_FOV:-1}"

R='\033[0;31m'; G='\033[0;32m'; Y='\033[1;33m'; B='\033[0;34m'; N='\033[0m'
log()  { echo -e "${B}[$(date +%H:%M:%S)]${N} $*"; }
ok()   { echo -e "${G}  ok${N} $*"; }
warn() { echo -e "${Y}  !${N} $*"; }
err()  { echo -e "${R}  x${N} $*"; }

WORK="$OUTPUT_ROOT"
GVHMR_OUT="$WORK/gvhmr_out"
GVHMR_CONVERTED="$WORK/001_converted.npz"
GVHMR_HANDS_NPZ="$WORK/001_smplx_hands.npz"
GVHMR_CAMERA_NPZ="$WORK/gvhmr_camera.npz"
HAND_BACKEND_MARKER="$WORK/.hand_backend"
HAND_CONFIG_MARKER="$WORK/.hand_config"
FILTER_CONFIG_MARKER="$WORK/.filter_config"
FILTERED_RENDER_MARKER="$WORK/.filtered_render_config"

HAND_CODE_FINGERPRINT="$(
    sha256sum \
        "$GVHMR/tools/processor/generate_smplxs.py" \
        "$GVHMR/tools/processor/run_hand4wholepp_video.py" 2>/dev/null |
        sha256sum | cut -d' ' -f1
)"
HAND_ASSET_FINGERPRINT="$(
    for asset in \
        "$GVHMR/inputs/checkpoints/vitpose/vitpose-h-coco-wholebody.pth" \
        "$GVHMR/_DATA/hamer_ckpts/checkpoints/hamer.ckpt" \
        "$GVHMR_HAND4WHOLEPP_SNAPSHOT"; do
        if [ -e "$asset" ]; then
            stat -c '%n:%s:%Y' "$asset"
        fi
    done | sha256sum | cut -d' ' -f1
)"
HAND_CONFIG_TEXT="$(
    printf '%s\n' \
        "code=$HAND_CODE_FINGERPRINT" \
        "assets=$HAND_ASSET_FINGERPRINT" \
        "backend=$GVHMR_HAND_BACKEND" \
        "vitpose_img_ds=$GVHMR_VITPOSE_IMG_DS" \
        "hand_kpt_conf_thr=$GVHMR_HAND_KPT_CONF_THR" \
        "hand_kpt_low_conf_thr=$GVHMR_HAND_KPT_LOW_CONF_THR" \
        "hand_kpt_hi_min_keypoints=$GVHMR_HAND_KPT_HI_MIN_KEYPOINTS" \
        "hand_min_keypoints=$GVHMR_HAND_MIN_KEYPOINTS" \
        "bbox_min_size=$GVHMR_HAND_BBOX_MIN_SIZE" \
        "bbox_smoothing=$GVHMR_HAND_BBOX_SMOOTHING" \
        "bbox_max_jump=$GVHMR_HAND_BBOX_MAX_JUMP" \
        "bbox_overlap_iou=$GVHMR_HAND_BBOX_OVERLAP_IOU" \
        "bbox_collision_ratio=$GVHMR_HAND_BBOX_COLLISION_SCORE_RATIO" \
        "hamer_bbox_rescale=$GVHMR_HAMER_BBOX_RESCALE" \
        "hamer_candidates=$GVHMR_HAMER_BBOX_RESCALE_CANDIDATES" \
        "hamer_switch_penalty=$GVHMR_HAMER_CANDIDATE_SWITCH_PENALTY" \
        "hamer_refine_steps=$GVHMR_HAMER_REFINE_STEPS" \
        "hamer_refine_lr=$GVHMR_HAMER_REFINE_LR" \
        "hamer_refine_conf_thr=$GVHMR_HAMER_REFINE_CONF_THR" \
        "hamer_refine_min_keypoints=$GVHMR_HAMER_REFINE_MIN_KEYPOINTS" \
        "hamer_refine_pose_prior=$GVHMR_HAMER_REFINE_POSE_PRIOR" \
        "hamer_refine_global_prior=$GVHMR_HAMER_REFINE_GLOBAL_PRIOR" \
        "hand4wholepp_root=$GVHMR_HAND4WHOLEPP_ROOT" \
        "hand4wholepp_snapshot=$GVHMR_HAND4WHOLEPP_SNAPSHOT" \
        "hand4wholepp_batch_size=$GVHMR_HAND4WHOLEPP_BATCH_SIZE" \
        "hand4wholepp_yolo=$GVHMR_HAND4WHOLEPP_YOLO_MODEL" \
        "hand4wholepp_joint_source=$GVHMR_HAND4WHOLEPP_JOINT_SOURCE"
)"
HAND_CONFIG_FINGERPRINT="$(printf '%s' "$HAND_CONFIG_TEXT" | sha256sum | cut -d' ' -f1)"
USE_GVHMR_CAMERA=0
if [ "$ISAAC_CAMERA_MODE" = "gvhmr" ] || [ "$ISAAC_CAMERA_MODE" = "gvhmr_static" ]; then
    USE_GVHMR_CAMERA=1
fi
if [ -z "$ISAAC_CAMERA_PATH" ] && [ "$USE_GVHMR_CAMERA" = "1" ]; then
    ISAAC_CAMERA_PATH="$GVHMR_CAMERA_NPZ"
fi

LOCO_IN="$WORK/locomotion_in"
LOCO_OUT="$WORK/locomotion"
LOCO_NPZ="$LOCO_OUT/optimizer/results_filter/001/001_optimized.npz"
SMOOTH_NPZ="$WORK/001_smoothed.npz"

PHC_IN="$WORK/phc_in"
PHC_OUT="$WORK/phc_repaired"
PHC_RENDERINGS="$WORK/phc_renderings"
PHC_GROUNDED_NPZ="$WORK/001_phc_grounded.npz"
PHC_SMOOTH_NPZ="$WORK/001_phc_smoothed.npz"
PHC_SMOOTH_GROUNDED_NPZ="$WORK/001_phc_smoothed_grounded.npz"
PHC_SMOOTH_REPORT="$WORK/001_phc_smoothed_report.json"
PHC_RAN=0

COMPARISON_MP4="$WORK/${VIDEO_NAME}_comparison.mp4"

find_first() {
    find "$1" -name "$2" -print -quit 2>/dev/null
}

find_latest_repaired_npz() {
    local root="$1"
    find "$root" -name "*_repaired.npz" -printf '%T@ %p\n' 2>/dev/null
    find "$root" -name "*_validated.npz" -printf '%T@ %p\n' 2>/dev/null
}

require_file() {
    [ -e "$1" ] || { err "$2 not found: $1"; exit 1; }
}

run_phc_python() {
    if [ ! -x "$PY_PHC" ]; then
        err "PHC python not found or not executable: $PY_PHC"
        err "Install the phc conda environment or set PY_PHC=/path/to/envs/phc/bin/python"
        return 127
    fi

    local phc_prefix="${PY_PHC_PREFIX:-$(cd "$(dirname "$PY_PHC")/.." 2>/dev/null && pwd || true)}"
    local phc_bindings="${PHC}/isaacgym/python/isaacgym/_bindings/linux-x86_64"
    local ld_paths="${PHC}/isaacgym:${LD_LIBRARY_PATH:-}"
    [ -n "$phc_prefix" ] && ld_paths="${phc_prefix}/lib:${ld_paths}"
    [ -d "$phc_bindings" ] && ld_paths="${phc_bindings}:${ld_paths}"

    env \
        LD_LIBRARY_PATH="$ld_paths" \
        PATH="${phc_prefix}/bin:${PATH:-}" \
        PYTHONPATH="${PHC}:${PHC}/isaacgym/python:${PYTHONPATH:-}" \
        ISAACGYM_PATH="${PHC}/isaacgym" \
        "$PY_PHC" "$@"
}

ground_fix_npz() {
    local input="$1"
    local output="$2"
    local label="$3"
    local report="${output%.npz}_floor_report.json"

    [ "$PHC_GROUND_FIX" = "1" ] || return 0
    [ -f "$input" ] || return 0

    log "$label: SMPL mesh floor alignment"
    if run_phc_python "${LEGACY_PIPELINE}/fix_motion_floor.py" \
        --input "$input" --output "$output" \
        --report "$report" \
        --model_path "${PHC}/data/smpl" \
        --floor 0.0 \
        --clearance "$PHC_GROUND_CLEARANCE" \
        --mode "$PHC_GROUND_FIX_MODE" \
        --up_axis 1 2>&1; then
        FINAL_POST_NPZ="$output"
        ok "$label floor output: $FINAL_POST_NPZ"
    else
        warn "$label floor alignment failed; keeping $input"
    fi
}

echo ""
echo "============================================================"
echo "  Hand pipeline: Video -> GVHMR-hand -> Locomotion -> PHC"
echo "============================================================"
log "Video:  $VIDEO"
log "Output: $WORK"
log "Root:   $PIPELINE_ROOT"
log "Hand backend: $GVHMR_HAND_BACKEND"
if [ "$GVHMR_HAND_BACKEND" = "wilor" ]; then
    log "WiLoR root: $GVHMR_WILOR_ROOT"
elif [ "$GVHMR_HAND_BACKEND" = "hand4wholepp" ]; then
    log "Hand4Whole++ root: $GVHMR_HAND4WHOLEPP_ROOT"
fi
mkdir -p "$WORK" "$GVHMR_OUT" "$LOCO_IN/001" "$PHC_IN/001" "$PHC_OUT/001"

# ---------------------------------------------------------------------------
# Stage 1: GVHMR-hand inference
# ---------------------------------------------------------------------------
GVHMR_RESULTS="$(find_first "$GVHMR_OUT" "hmr4d_results.pt")"
MANO_PARAMS="$(find_first "$GVHMR_OUT" "mano_params.pt")"
VITPOSE_WHOLEBODY="$(find_first "$GVHMR_OUT" "vitpose_wholebody.pt")"
GVHMR_INCAM="$(find_first "$GVHMR_OUT" "1_incam.mp4")"
GVHMR_RENDER_READY=0
if [ "$GVHMR_SKIP_RENDER" = "1" ] || [ -f "${GVHMR_INCAM:-}" ]; then
    GVHMR_RENDER_READY=1
fi

BACKEND_MATCH=0
if [ -f "$HAND_BACKEND_MARKER" ] && [ "$(cat "$HAND_BACKEND_MARKER")" = "$GVHMR_HAND_BACKEND" ]; then
    BACKEND_MATCH=1
fi
HAND_CONFIG_MATCH=0
if [ -f "$HAND_CONFIG_MARKER" ] && [ "$(cat "$HAND_CONFIG_MARKER")" = "$HAND_CONFIG_FINGERPRINT" ]; then
    HAND_CONFIG_MATCH=1
fi

if [ "$SKIP_EXISTING" = "1" ] && [ "$GVHMR_FORCE_HAND_PREPROCESS" != "1" ] && \
   [ "$BACKEND_MATCH" = "1" ] && [ "$HAND_CONFIG_MATCH" = "1" ] && \
   [ -f "${GVHMR_RESULTS:-}" ] && [ -f "${MANO_PARAMS:-}" ] && \
   [ "$GVHMR_RENDER_READY" = "1" ]; then
    ok "GVHMR-hand results already exist (backend: $GVHMR_HAND_BACKEND)"
    echo "$GVHMR_HAND_BACKEND" > "$HAND_BACKEND_MARKER"
    echo "$HAND_CONFIG_FINGERPRINT" > "$HAND_CONFIG_MARKER"
else
    if [ "$SKIP_EXISTING" = "1" ] && [ -f "${MANO_PARAMS:-}" ] && [ "$BACKEND_MATCH" = "0" ]; then
        warn "Hand backend changed (marker: $(cat "$HAND_BACKEND_MARKER" 2>/dev/null || echo 'none') -> $GVHMR_HAND_BACKEND); forcing re-process"
    elif [ "$SKIP_EXISTING" = "1" ] && [ -f "${MANO_PARAMS:-}" ] && [ "$HAND_CONFIG_MATCH" = "0" ]; then
        warn "Hand inference config changed; forcing hand preprocess"
    fi
    FORCE_HAND_THIS_RUN="$GVHMR_FORCE_HAND_PREPROCESS"
    if [ -f "${MANO_PARAMS:-}" ] && { [ "$BACKEND_MATCH" = "0" ] || [ "$HAND_CONFIG_MATCH" = "0" ]; }; then
        FORCE_HAND_THIS_RUN=1
    fi
    if [ "$FORCE_HAND_THIS_RUN" = "1" ]; then
        find "$GVHMR_OUT" -type f \
            \( -name '1_incam.mp4' -o -name '2_global.mp4' -o -name '*_3_incam_global_horiz.mp4' \) \
            -delete 2>/dev/null || true
    fi
    log ""
    log "[1/5] GVHMR-hand inference"
    require_file "$GVHMR/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt" "GVHMR checkpoint"
    require_file "$GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz" "SMPLX_NEUTRAL.npz"
    require_file "$GVHMR/inputs/checkpoints/vitpose/vitpose-h-coco-wholebody.pth" "ViTPose wholebody checkpoint"
    case "$GVHMR_HAND_BACKEND" in
        hamer)
            require_file "$GVHMR/_DATA/hamer_ckpts/checkpoints/hamer.ckpt" "HaMeR checkpoint"
            require_file "$GVHMR/_DATA/data/mano/MANO_RIGHT.pkl" "MANO_RIGHT.pkl"
            ;;
        wilor)
            require_file "$GVHMR_WILOR_ROOT/wilor/models/__init__.py" "WiLoR source; run scripts/setup_wilor_assets.sh"
            require_file "$GVHMR_WILOR_CHECKPOINT" "WiLoR checkpoint"
            require_file "$GVHMR_WILOR_CONFIG" "WiLoR model config"
            require_file "$GVHMR_WILOR_ROOT/mano_data/MANO_RIGHT.pkl" "WiLoR MANO_RIGHT.pkl link"
            require_file "$GVHMR_WILOR_ROOT/mano_data/mano_mean_params.npz" "WiLoR mano_mean_params.npz"
            ;;
        hand4wholepp)
            require_file "$GVHMR_HAND4WHOLEPP_ROOT/main/model.py" "Hand4Whole++ source; run scripts/setup_hand4wholepp_assets.sh"
            require_file "$GVHMR_HAND4WHOLEPP_SNAPSHOT" "Hand4Whole++ snapshot_6.pth"
            require_file "$GVHMR_HAND4WHOLEPP_ROOT/common/nets/WiLoR/pretrained_models/wilor_final.ckpt" "Hand4Whole++ WiLoR checkpoint link"
            require_file "$GVHMR_HAND4WHOLEPP_ROOT/common/nets/mmpose/dw-ll_ucoco.pth" "Hand4Whole++ DWPose checkpoint"
            require_file "$GVHMR_HAND4WHOLEPP_ROOT/common/utils/human_model_files/smpl/SMPL_NEUTRAL.pkl" "Hand4Whole++ SMPL_NEUTRAL.pkl"
            require_file "$GVHMR_HAND4WHOLEPP_ROOT/common/utils/human_model_files/smplx/SMPLX_NEUTRAL.pkl" "Hand4Whole++ SMPLX_NEUTRAL.pkl"
            require_file "$GVHMR_HAND4WHOLEPP_ROOT/common/utils/human_model_files/smplx/SMPLX_MALE.pkl" "Hand4Whole++ SMPLX_MALE.pkl"
            require_file "$GVHMR_HAND4WHOLEPP_ROOT/common/utils/human_model_files/smplx/SMPLX_FEMALE.pkl" "Hand4Whole++ SMPLX_FEMALE.pkl"
            require_file "$GVHMR_HAND4WHOLEPP_ROOT/common/utils/human_model_files/smplx/SMPLX_NEUTRAL.npz" "Hand4Whole++ SMPLX_NEUTRAL.npz"
            require_file "$GVHMR_HAND4WHOLEPP_ROOT/common/utils/human_model_files/smplx/SMPLX_MALE.npz" "Hand4Whole++ SMPLX_MALE.npz"
            require_file "$GVHMR_HAND4WHOLEPP_ROOT/common/utils/human_model_files/smplx/SMPLX_FEMALE.npz" "Hand4Whole++ SMPLX_FEMALE.npz"
            require_file "$GVHMR_HAND4WHOLEPP_ROOT/common/utils/human_model_files/smplx/MANO_SMPLX_vertex_ids.pkl" "Hand4Whole++ MANO_SMPLX_vertex_ids.pkl"
            require_file "$GVHMR_HAND4WHOLEPP_ROOT/common/utils/human_model_files/smplx/SMPLX_to_J14.pkl" "Hand4Whole++ SMPLX_to_J14.pkl"
            require_file "$GVHMR_HAND4WHOLEPP_ROOT/common/utils/human_model_files/mano/MANO_LEFT.pkl" "Hand4Whole++ MANO_LEFT.pkl"
            require_file "$GVHMR_HAND4WHOLEPP_ROOT/common/utils/human_model_files/mano/MANO_RIGHT.pkl" "Hand4Whole++ MANO_RIGHT.pkl"
            ;;
        *)
            err "Unsupported GVHMR_HAND_BACKEND=$GVHMR_HAND_BACKEND (expected hamer, wilor, or hand4wholepp)"
            exit 1
            ;;
    esac
    if [ "$GVHMR_STATIC_CAM" != "1" ]; then
        require_file "$GVHMR/inputs/checkpoints/dpvo/dpvo.pth" "DPVO checkpoint"
    fi

    GVHMR_CMD=(
        "$PY_GVHMR" -m tools.processor.generate_smplxs
        --video "$VIDEO"
        --video_name "$VIDEO_NAME"
        --output_root "$GVHMR_OUT"
        --batch_size "$GVHMR_BATCH_SIZE"
        --hand_backend "$GVHMR_HAND_BACKEND"
        --wilor_root "$GVHMR_WILOR_ROOT"
        --wilor_checkpoint "$GVHMR_WILOR_CHECKPOINT"
        --wilor_config "$GVHMR_WILOR_CONFIG"
        --hand4wholepp_root "$GVHMR_HAND4WHOLEPP_ROOT"
        --hand4wholepp_snapshot "$GVHMR_HAND4WHOLEPP_SNAPSHOT"
        --hand4wholepp_python "$GVHMR_HAND4WHOLEPP_PYTHON"
        --hand4wholepp_batch_size "$GVHMR_HAND4WHOLEPP_BATCH_SIZE"
        --hand4wholepp_yolo_model "$GVHMR_HAND4WHOLEPP_YOLO_MODEL"
        --hand4wholepp_joint_source "$GVHMR_HAND4WHOLEPP_JOINT_SOURCE"
        --vitpose_img_ds "$GVHMR_VITPOSE_IMG_DS"
        --hand_kpt_conf_thr "$GVHMR_HAND_KPT_CONF_THR"
        --hand_kpt_low_conf_thr "$GVHMR_HAND_KPT_LOW_CONF_THR"
        --hand_kpt_hi_min_keypoints "$GVHMR_HAND_KPT_HI_MIN_KEYPOINTS"
        --hand_min_keypoints "$GVHMR_HAND_MIN_KEYPOINTS"
        --hamer_bbox_rescale "$GVHMR_HAMER_BBOX_RESCALE"
        --hamer_bbox_rescale_candidates "$GVHMR_HAMER_BBOX_RESCALE_CANDIDATES"
        --hamer_candidate_switch_penalty "$GVHMR_HAMER_CANDIDATE_SWITCH_PENALTY"
        --hand_bbox_min_size "$GVHMR_HAND_BBOX_MIN_SIZE"
        --hand_bbox_smoothing "$GVHMR_HAND_BBOX_SMOOTHING"
        --hand_bbox_max_jump "$GVHMR_HAND_BBOX_MAX_JUMP"
        --hand_bbox_overlap_iou "$GVHMR_HAND_BBOX_OVERLAP_IOU"
        --hand_bbox_collision_score_ratio "$GVHMR_HAND_BBOX_COLLISION_SCORE_RATIO"
        --hamer_batch_size "$GVHMR_HAMER_BATCH_SIZE"
        --hamer_refine_steps "$GVHMR_HAMER_REFINE_STEPS"
        --hamer_refine_lr "$GVHMR_HAMER_REFINE_LR"
        --hamer_refine_conf_thr "$GVHMR_HAMER_REFINE_CONF_THR"
        --hamer_refine_min_keypoints "$GVHMR_HAMER_REFINE_MIN_KEYPOINTS"
        --hamer_refine_pose_prior "$GVHMR_HAMER_REFINE_POSE_PRIOR"
        --hamer_refine_global_prior "$GVHMR_HAMER_REFINE_GLOBAL_PRIOR"
    )
    [ "$GVHMR_STATIC_CAM" = "1" ] && GVHMR_CMD+=(-s)
    if [ "$GVHMR_SKIP_RENDER" = "1" ] || \
       [ "$GVHMR_FILTER_MANO_WRIST" = "1" ] || \
       [ "$GVHMR_FILTER_MANO_TEMPORAL" = "1" ] || \
       [ "$GVHMR_FILTER_MANO_FINGERS" = "1" ]; then
        GVHMR_CMD+=(--skip_render)
    fi
    [ "$GVHMR_LOW_MEMORY" = "1" ] && GVHMR_CMD+=(--low_memory)
    [ "$GVHMR_WILOR_FAST" = "1" ] && GVHMR_CMD+=(--wilor_fast)

    cd "$GVHMR"
    ISOLATE_HAND="$GVHMR_ISOLATE_HAND_PREPROCESS"
    if [ "$ISOLATE_HAND" = "auto" ]; then
        if [ "$GVHMR_HAND_BACKEND" = "hamer" ]; then
            ISOLATE_HAND=1
        else
            ISOLATE_HAND=0
        fi
    fi
    if [ "$ISOLATE_HAND" = "1" ]; then
        # HaMeR keeps a large decoded-frame buffer and CUDA allocator state.
        # End that OS process before HMR2 loads its CPU checkpoint/GPU model;
        # Python-level empty_cache alone did not release all host/driver state.
        HAND_ONLY_CMD=("${GVHMR_CMD[@]}" --hand_preprocess_only)
        [ "$FORCE_HAND_THIS_RUN" = "1" ] && HAND_ONLY_CMD+=(--force_hand_preprocess)
        log "[1a] Isolated $GVHMR_HAND_BACKEND hand preprocessing process"
        "${HAND_ONLY_CMD[@]}" 2>&1 || {
            err "$GVHMR_HAND_BACKEND hand preprocessing failed"
            exit 1
        }
        ok "$GVHMR_HAND_BACKEND process exited; host and CUDA memory returned to the OS/driver"
        log "[1a] Fresh process for HMR2/GVHMR body inference"
        "${GVHMR_CMD[@]}" 2>&1 || { err "GVHMR body inference failed"; exit 1; }
    else
        [ "$FORCE_HAND_THIS_RUN" = "1" ] && GVHMR_CMD+=(--force_hand_preprocess)
        "${GVHMR_CMD[@]}" 2>&1 || { err "GVHMR-hand inference failed"; exit 1; }
    fi
    ok "GVHMR-hand inference complete"
    echo "$GVHMR_HAND_BACKEND" > "$HAND_BACKEND_MARKER"
    echo "$HAND_CONFIG_FINGERPRINT" > "$HAND_CONFIG_MARKER"

    GVHMR_RESULTS="$(find_first "$GVHMR_OUT" "hmr4d_results.pt")"
    MANO_PARAMS="$(find_first "$GVHMR_OUT" "mano_params.pt")"
    VITPOSE_WHOLEBODY="$(find_first "$GVHMR_OUT" "vitpose_wholebody.pt")"
    GVHMR_INCAM="$(find_first "$GVHMR_OUT" "1_incam.mp4")"
fi

[ -f "${GVHMR_RESULTS:-}" ] || { err "hmr4d_results.pt not found"; exit 1; }
[ -f "${MANO_PARAMS:-}" ] || warn "mano_params.pt not found; hand sidecar will contain zero hands"

FILTER_CONFIG_TEXT="$(
    FILTER_CODE_FINGERPRINT="$(
        sha256sum \
            "$GVHMR/tools/processor/filter_mano_wrist.py" \
            "$GVHMR/tools/processor/filter_mano_temporal.py" \
            "$GVHMR/tools/processor/filter_mano_fingers.py" \
            "${SCRIPT_DIR}/convert_to_npz.py" 2>/dev/null |
            sha256sum | cut -d' ' -f1
    )"
    printf '%s\n' \
        "code=$FILTER_CODE_FINGERPRINT" \
        "constraint_profile=$GVHMR_HAND_CONSTRAINT_PROFILE" \
        "hand_config=$HAND_CONFIG_FINGERPRINT" \
        "wrist_filter=$GVHMR_FILTER_MANO_WRIST" \
        "wrist_temp=$GVHMR_WRIST_FILTER_W_TEMP/$GVHMR_WRIST_FILTER_W_TEMP_LOW_CONF" \
        "wrist_body=$GVHMR_WRIST_FILTER_W_BODY/$GVHMR_WRIST_FILTER_W_BODY_LOW_CONF" \
        "wrist_mix=$GVHMR_WRIST_FILTER_MIX_WEIGHT" \
        "wrist_penalties=$GVHMR_WRIST_FILTER_HAMER_PENALTY,$GVHMR_WRIST_FILTER_FLIP_PENALTY,$GVHMR_WRIST_FILTER_BODY_PENALTY,$GVHMR_WRIST_FILTER_MIX_PENALTY" \
        "wrist_invalid_penalties=$GVHMR_WRIST_FILTER_INVALID_HAMER_PENALTY,$GVHMR_WRIST_FILTER_INVALID_FLIP_PENALTY,$GVHMR_WRIST_FILTER_INVALID_MIX_PENALTY" \
        "temporal_filter=$GVHMR_FILTER_MANO_TEMPORAL" \
        "temporal_reproj=$GVHMR_TEMPORAL_FILTER_REPROJ_ERROR_THR" \
        "temporal_bbox=$GVHMR_TEMPORAL_FILTER_BBOX_WINDOW,$GVHMR_TEMPORAL_FILTER_BBOX_SHRINK_RATIO,$GVHMR_TEMPORAL_FILTER_MIN_BBOX_FOREARM_RATIO,$GVHMR_TEMPORAL_FILTER_BBOX_JUMP_RATIO,$GVHMR_TEMPORAL_FILTER_BBOX_OVERLAP_IOU" \
        "temporal_floor=$GVHMR_TEMPORAL_FILTER_BBOX_SIZE_FLOOR_RATIO,$GVHMR_TEMPORAL_FILTER_BBOX_SIZE_FLOOR_MIN_FOREARM_RATIO" \
        "temporal_fill=$GVHMR_TEMPORAL_FILTER_MAX_INTERP_GAP,$GVHMR_TEMPORAL_FILTER_MAX_EDGE_HOLD" \
        "finger_filter=$GVHMR_FILTER_MANO_FINGERS" \
        "finger_window=$GVHMR_FINGER_FILTER_SMOOTH_WINDOW" \
        "finger_weights=$GVHMR_FINGER_FILTER_RELIABLE_SMOOTH_WEIGHT,$GVHMR_FINGER_FILTER_WEAK_SMOOTH_WEIGHT,$GVHMR_FINGER_FILTER_BAD_SMOOTH_WEIGHT" \
        "finger_fill=$GVHMR_FINGER_FILTER_MAX_INTERP_GAP,$GVHMR_FINGER_FILTER_MAX_EDGE_HOLD" \
        "finger_limits=$GVHMR_FINGER_FILTER_MAX_JOINT_ANGLE_DELTA,$GVHMR_FINGER_FILTER_MAX_JOINT_XYZ_DELTA" \
        "wrist_smooth=$GVHMR_FINGER_FILTER_WRIST_SMOOTH_WINDOW,$GVHMR_FINGER_FILTER_WRIST_RELIABLE_SMOOTH_WEIGHT,$GVHMR_FINGER_FILTER_WRIST_WEAK_SMOOTH_WEIGHT,$GVHMR_FINGER_FILTER_WRIST_BAD_SMOOTH_WEIGHT,$GVHMR_FINGER_FILTER_MAX_WRIST_ANGLE_DELTA" \
        "finger_floor=$GVHMR_FINGER_FILTER_HAND_SIZE_FLOOR_RATIO" \
        "finger_open=$GVHMR_FINGER_FILTER_OPEN_RESCUE_WEIGHT,$GVHMR_FINGER_FILTER_OPEN_RESCUE_SMOOTH_WEIGHT" \
        "sidecar_refine=$GVHMR_HAND_REFINE_MODE,$GVHMR_HAND_REPROJ_ERROR_THR,$GVHMR_HAND_REPROJ_ERROR_RATIO_THR,$GVHMR_HAND_SPIKE_MAD_MULTIPLIER,$GVHMR_HAND_SPIKE_ABS_THRESHOLD,$GVHMR_HAND_REFINE_GAP_MERGE,$GVHMR_HAND_REFINE_MAX_BURST,$GVHMR_HAND_REFINE_SMOOTH_WINDOW" \
        "wrist_offset=$GVHMR_HAND_BACKEND,$GVHMR_HAND_WRIST_OFFSET_MODE,$GVHMR_HAND_WRIST_OFFSET_SMOOTH_WINDOW" \
        "export=$GVHMR_EXPORT_FPS,$GVHMR_PERSON_IDX" \
        "diagnostic=$GVHMR_DIAGNOSE_HAND,$GVHMR_DIAGNOSE_HAND_WIDTH"
)"
FILTER_FINGERPRINT="$(printf '%s' "$FILTER_CONFIG_TEXT" | sha256sum | cut -d' ' -f1)"
FILTER_CONFIG_OK=0
if [ -f "$FILTER_CONFIG_MARKER" ] && [ "$(cat "$FILTER_CONFIG_MARKER")" = "$FILTER_FINGERPRINT" ]; then
    FILTER_CONFIG_OK=1
fi

MANO_TRACK_LABEL="raw"
if [ "$GVHMR_FILTER_MANO_WRIST" = "1" ] && [ -f "${MANO_PARAMS:-}" ]; then
    MANO_FILTERED="${WORK}/mano_params_wrist_fixed.pt"
    if [ "$SKIP_EXISTING" = "1" ] && [ "$GVHMR_FORCE_HAND_PREPROCESS" != "1" ] && [ "$FILTER_CONFIG_OK" = "1" ] && [ -f "$MANO_FILTERED" ]; then
        ok "Wrist-filtered MANO already exists: $MANO_FILTERED"
    else
        log ""
        log "[1a] Filter MANO wrist candidates"
        (
            cd "$GVHMR"
            "$PY_GVHMR" -m tools.processor.filter_mano_wrist \
                --mano_params "$MANO_PARAMS" \
                --hmr4d_results "$GVHMR_RESULTS" \
                --output "$MANO_FILTERED" \
                --w_temp "$GVHMR_WRIST_FILTER_W_TEMP" \
                --w_body "$GVHMR_WRIST_FILTER_W_BODY" \
                --w_temp_low_conf "$GVHMR_WRIST_FILTER_W_TEMP_LOW_CONF" \
                --w_body_low_conf "$GVHMR_WRIST_FILTER_W_BODY_LOW_CONF" \
                --mix_weight "$GVHMR_WRIST_FILTER_MIX_WEIGHT" \
                --hamer_penalty "$GVHMR_WRIST_FILTER_HAMER_PENALTY" \
                --flip_penalty "$GVHMR_WRIST_FILTER_FLIP_PENALTY" \
                --body_penalty "$GVHMR_WRIST_FILTER_BODY_PENALTY" \
                --mix_penalty "$GVHMR_WRIST_FILTER_MIX_PENALTY" \
                --invalid_hamer_penalty "$GVHMR_WRIST_FILTER_INVALID_HAMER_PENALTY" \
                --invalid_flip_penalty "$GVHMR_WRIST_FILTER_INVALID_FLIP_PENALTY" \
                --invalid_mix_penalty "$GVHMR_WRIST_FILTER_INVALID_MIX_PENALTY"
        )
    fi
    MANO_PARAMS="$MANO_FILTERED"
    MANO_TRACK_LABEL="wrist_fixed"
fi
if [ "$GVHMR_FILTER_MANO_TEMPORAL" = "1" ] && [ -f "${MANO_PARAMS:-}" ]; then
    VITPOSE_WHOLEBODY="$(find_first "$GVHMR_OUT" "vitpose_wholebody.pt")"
    if [ ! -f "${VITPOSE_WHOLEBODY:-}" ]; then
        warn "vitpose_wholebody.pt not found; skipping MANO temporal filter"
    else
        MANO_TEMPORAL_FIXED="${WORK}/mano_params_temporal_fixed.pt"
        MANO_TEMPORAL_SUMMARY="${WORK}/mano_params_temporal_fixed.json"
        if [ "$SKIP_EXISTING" = "1" ] && [ "$GVHMR_FORCE_HAND_PREPROCESS" != "1" ] && [ "$FILTER_CONFIG_OK" = "1" ] && [ -f "$MANO_TEMPORAL_FIXED" ]; then
            ok "Temporal-fixed MANO already exists: $MANO_TEMPORAL_FIXED"
        else
            log ""
            log "[1a] Filter MANO temporal outliers"
            (
                cd "$GVHMR"
                "$PY_GVHMR" -m tools.processor.filter_mano_temporal \
                    --mano_params "$MANO_PARAMS" \
                    --vitpose_wholebody "$VITPOSE_WHOLEBODY" \
                    --output "$MANO_TEMPORAL_FIXED" \
                    --summary "$MANO_TEMPORAL_SUMMARY" \
                    --reproj_error_thr "$GVHMR_TEMPORAL_FILTER_REPROJ_ERROR_THR" \
                    --hand_conf_thr "$GVHMR_HAND_KPT_CONF_THR" \
                    --hand_low_conf_thr "$GVHMR_HAND_KPT_LOW_CONF_THR" \
                    --hand_hi_min_keypoints "$GVHMR_HAND_KPT_HI_MIN_KEYPOINTS" \
                    --hand_min_keypoints "$GVHMR_HAND_MIN_KEYPOINTS" \
                    --bbox_window "$GVHMR_TEMPORAL_FILTER_BBOX_WINDOW" \
                    --bbox_shrink_ratio "$GVHMR_TEMPORAL_FILTER_BBOX_SHRINK_RATIO" \
                    --min_bbox_forearm_ratio "$GVHMR_TEMPORAL_FILTER_MIN_BBOX_FOREARM_RATIO" \
                    --bbox_jump_ratio "$GVHMR_TEMPORAL_FILTER_BBOX_JUMP_RATIO" \
                    --bbox_overlap_iou "$GVHMR_TEMPORAL_FILTER_BBOX_OVERLAP_IOU" \
                    --bbox_size_floor_ratio "$GVHMR_TEMPORAL_FILTER_BBOX_SIZE_FLOOR_RATIO" \
                    --bbox_size_floor_min_forearm_ratio "$GVHMR_TEMPORAL_FILTER_BBOX_SIZE_FLOOR_MIN_FOREARM_RATIO" \
                    --max_interp_gap "$GVHMR_TEMPORAL_FILTER_MAX_INTERP_GAP" \
                    --max_edge_hold "$GVHMR_TEMPORAL_FILTER_MAX_EDGE_HOLD"
            )
        fi
        MANO_PARAMS="$MANO_TEMPORAL_FIXED"
        MANO_TRACK_LABEL="temporal_fixed"
    fi
fi
if [ "$GVHMR_FILTER_MANO_FINGERS" = "1" ] && [ -f "${MANO_PARAMS:-}" ]; then
    VITPOSE_WHOLEBODY="$(find_first "$GVHMR_OUT" "vitpose_wholebody.pt")"
    if [ ! -f "${VITPOSE_WHOLEBODY:-}" ]; then
        warn "vitpose_wholebody.pt not found; skipping MANO finger filter"
    else
        MANO_FINGER_FIXED="${WORK}/mano_params_finger_fixed.pt"
        MANO_FINGER_SUMMARY="${WORK}/mano_params_finger_fixed.json"
        if [ "$SKIP_EXISTING" = "1" ] && [ "$GVHMR_FORCE_HAND_PREPROCESS" != "1" ] && [ "$FILTER_CONFIG_OK" = "1" ] && [ -f "$MANO_FINGER_FIXED" ]; then
            ok "Finger-filtered MANO already exists: $MANO_FINGER_FIXED"
        else
            log ""
            log "[1a] Filter MANO finger articulation"
            (
                cd "$GVHMR"
                "$PY_GVHMR" -m tools.processor.filter_mano_fingers \
                    --mano_params "$MANO_PARAMS" \
                    --vitpose_wholebody "$VITPOSE_WHOLEBODY" \
                    --output "$MANO_FINGER_FIXED" \
                    --summary "$MANO_FINGER_SUMMARY" \
                    --hand_low_conf_thr "$GVHMR_HAND_KPT_LOW_CONF_THR" \
                    --hand_min_keypoints "$GVHMR_HAND_MIN_KEYPOINTS" \
                    --max_interp_gap "$GVHMR_FINGER_FILTER_MAX_INTERP_GAP" \
                    --max_edge_hold "$GVHMR_FINGER_FILTER_MAX_EDGE_HOLD" \
                    --smooth_window "$GVHMR_FINGER_FILTER_SMOOTH_WINDOW" \
                    --reliable_smooth_weight "$GVHMR_FINGER_FILTER_RELIABLE_SMOOTH_WEIGHT" \
                    --weak_smooth_weight "$GVHMR_FINGER_FILTER_WEAK_SMOOTH_WEIGHT" \
                    --bad_smooth_weight "$GVHMR_FINGER_FILTER_BAD_SMOOTH_WEIGHT" \
                    --open_rescue_weight "$GVHMR_FINGER_FILTER_OPEN_RESCUE_WEIGHT" \
                    --open_rescue_smooth_weight "$GVHMR_FINGER_FILTER_OPEN_RESCUE_SMOOTH_WEIGHT" \
                    --max_joint_angle_delta "$GVHMR_FINGER_FILTER_MAX_JOINT_ANGLE_DELTA" \
                    --max_joint_xyz_delta "$GVHMR_FINGER_FILTER_MAX_JOINT_XYZ_DELTA" \
                    --wrist_smooth_window "$GVHMR_FINGER_FILTER_WRIST_SMOOTH_WINDOW" \
                    --wrist_reliable_smooth_weight "$GVHMR_FINGER_FILTER_WRIST_RELIABLE_SMOOTH_WEIGHT" \
                    --wrist_weak_smooth_weight "$GVHMR_FINGER_FILTER_WRIST_WEAK_SMOOTH_WEIGHT" \
                    --wrist_bad_smooth_weight "$GVHMR_FINGER_FILTER_WRIST_BAD_SMOOTH_WEIGHT" \
                    --max_wrist_angle_delta "$GVHMR_FINGER_FILTER_MAX_WRIST_ANGLE_DELTA" \
                    --hand_size_floor_ratio "$GVHMR_FINGER_FILTER_HAND_SIZE_FLOOR_RATIO"
            )
        fi
        MANO_PARAMS="$MANO_FINGER_FIXED"
        MANO_TRACK_LABEL="${MANO_TRACK_LABEL}_finger_fixed"
    fi
fi

if [ "$GVHMR_SKIP_RENDER" != "1" ] && \
   { [ "$GVHMR_FILTER_MANO_WRIST" = "1" ] || [ "$GVHMR_FILTER_MANO_TEMPORAL" = "1" ] || [ "$GVHMR_FILTER_MANO_FINGERS" = "1" ]; } && \
   [ -f "${MANO_PARAMS:-}" ]; then
    GVHMR_INCAM="$(find_first "$GVHMR_OUT" "1_incam.mp4")"
    FILTERED_RENDER_OK=0
    if [ -f "$FILTERED_RENDER_MARKER" ] && \
       [ "$(cat "$FILTERED_RENDER_MARKER")" = "$FILTER_FINGERPRINT" ] && \
       [ -f "${GVHMR_INCAM:-}" ]; then
        FILTERED_RENDER_OK=1
    fi
    if [ "$SKIP_EXISTING" = "1" ] && [ "$FILTERED_RENDER_OK" = "1" ]; then
        ok "Filtered GVHMR render already exists: $GVHMR_INCAM"
    else
        log ""
        log "[1a] Render GVHMR with filtered hand track"
        find "$GVHMR_OUT" -type f \
            \( -name '1_incam.mp4' -o -name '2_global.mp4' -o -name '*_3_incam_global_horiz.mp4' \) \
            -delete 2>/dev/null || true
        FILTERED_RENDER_CMD=(
            "$PY_GVHMR" -m tools.processor.generate_smplxs
            --video "$VIDEO"
            --video_name "$VIDEO_NAME"
            --output_root "$GVHMR_OUT"
            --mano_params_override "$MANO_PARAMS"
            --render_only
        )
        [ "$GVHMR_STATIC_CAM" = "1" ] && FILTERED_RENDER_CMD+=(-s)
        (
            cd "$GVHMR"
            "${FILTERED_RENDER_CMD[@]}"
        )
        echo "$FILTER_FINGERPRINT" > "$FILTERED_RENDER_MARKER"
        GVHMR_INCAM="$(find_first "$GVHMR_OUT" "1_incam.mp4")"
        ok "Filtered GVHMR render: $GVHMR_INCAM"
    fi
fi

if [ "$GVHMR_DIAGNOSE_HAND" = "1" ] && [ -f "${MANO_PARAMS:-}" ]; then
    VITPOSE_WHOLEBODY="$(find_first "$GVHMR_OUT" "vitpose_wholebody.pt")"
    if [ ! -f "${VITPOSE_WHOLEBODY:-}" ]; then
        warn "vitpose_wholebody.pt not found; skipping hand diagnostic video"
    else
        HAND_DIAG_DIR="${WORK}/hamer_diagnostics"
        HAND_DIAG_MP4="${HAND_DIAG_DIR}/${VIDEO_NAME}_hamer_diag_${MANO_TRACK_LABEL}.mp4"
        HAND_DIAG_JSON="${HAND_DIAG_DIR}/${VIDEO_NAME}_hamer_diag_${MANO_TRACK_LABEL}.json"
        if [ "$SKIP_EXISTING" = "1" ] && [ "$GVHMR_FORCE_HAND_PREPROCESS" != "1" ] && [ "$FILTER_CONFIG_OK" = "1" ] && [ -f "$HAND_DIAG_MP4" ] && [ -f "$HAND_DIAG_JSON" ]; then
            ok "Hand diagnostic already exists: $HAND_DIAG_MP4"
        else
            log ""
            log "[1a] Render hand diagnostic video"
            (
                cd "$GVHMR"
                "$PY_GVHMR" "${SCRIPT_DIR}/diagnose_hamer_hand.py" \
                    --video "$VIDEO" \
                    --vitpose_wholebody "$VITPOSE_WHOLEBODY" \
                    --mano_params "$MANO_PARAMS" \
                    --output "$HAND_DIAG_MP4" \
                    --summary "$HAND_DIAG_JSON" \
                    --width "$GVHMR_DIAGNOSE_HAND_WIDTH"
            )
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Stage 1b: Convert body for locomotion/PHC and preserve hands as sidecar
# ---------------------------------------------------------------------------
if [ "$SKIP_EXISTING" = "1" ] && [ "$GVHMR_FORCE_HAND_PREPROCESS" != "1" ] && [ "$FILTER_CONFIG_OK" = "1" ] && \
   [ -f "$GVHMR_CONVERTED" ] && [ -f "$GVHMR_HANDS_NPZ" ]; then
    ok "Converted outputs already exist"
else
    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$GVHMR_CONVERTED" ] && [ "$FILTER_CONFIG_OK" = "0" ]; then
        warn "Filter/gvhmr config changed; re-converting"
    fi
    log ""
    log "[1b] Convert GVHMR-hand output"
    CONVERT_CMD=(
        "$PY_LOCO" "${SCRIPT_DIR}/convert_to_npz.py"
        --gvhmr_results "$GVHMR_RESULTS"
        --output "$GVHMR_CONVERTED"
        --smplx_output "$GVHMR_HANDS_NPZ"
        --fps "$GVHMR_EXPORT_FPS"
        --person_idx "$GVHMR_PERSON_IDX"
        --hand_backend "$GVHMR_HAND_BACKEND"
        --hand_refine_mode "$GVHMR_HAND_REFINE_MODE"
        --hand_reproj_error_thr "$GVHMR_HAND_REPROJ_ERROR_THR"
        --hand_reproj_error_ratio_thr "$GVHMR_HAND_REPROJ_ERROR_RATIO_THR"
        --hand_wrist_offset_mode "$GVHMR_HAND_WRIST_OFFSET_MODE"
        --hand_wrist_offset_smooth_window "$GVHMR_HAND_WRIST_OFFSET_SMOOTH_WINDOW"
        --hand_spike_mad_multiplier "$GVHMR_HAND_SPIKE_MAD_MULTIPLIER"
        --hand_spike_abs_threshold "$GVHMR_HAND_SPIKE_ABS_THRESHOLD"
        --hand_refine_gap_merge "$GVHMR_HAND_REFINE_GAP_MERGE"
        --hand_refine_max_burst "$GVHMR_HAND_REFINE_MAX_BURST"
        --hand_refine_smooth_window "$GVHMR_HAND_REFINE_SMOOTH_WINDOW"
    )
    [ -f "${MANO_PARAMS:-}" ] && CONVERT_CMD+=(--mano_params "$MANO_PARAMS")
    "${CONVERT_CMD[@]}"
    ok "Converted body NPZ: $GVHMR_CONVERTED"
    ok "Preserved hand sidecar: $GVHMR_HANDS_NPZ"
    echo "$FILTER_FINGERPRINT" > "$FILTER_CONFIG_MARKER"
fi

LOCO_SOURCE="$GVHMR_CONVERTED"

# ---------------------------------------------------------------------------
# Stage 2: Locomotion height optimization
# ---------------------------------------------------------------------------
if [ "$SKIP_EXISTING" = "1" ] && [ "$FORCE_LOCO" != "1" ] && [ -f "$LOCO_NPZ" ]; then
    ok "Locomotion already exists: $LOCO_NPZ"
else
    log ""
    log "[2/5] Locomotion height optimization"
    cp "$LOCO_SOURCE" "$LOCO_IN/001/001_optimized.npz"
    cd "$LOCO"

    LOCO_CMD_BASE=(
        "$PY_LOCO" code/optim/optimizer_v2.py
        --config code/configs/config.yaml
        --target_dir "$LOCO_IN"
        --file_pattern "001/*.npz"
        --seqID_index 1
        --prefix optimizer
        --use_timestamp 0
        --output_root "$LOCO_OUT"
        --optim_height "$LOCO_OPTIM_HEIGHT"
        --gravity_axis "$LOCO_GRAVITY_AXIS"
    )

    "${LOCO_CMD_BASE[@]}" \
        --check_penetration "$LOCO_CHECK_PENETRATION" \
        --check_speed "$LOCO_CHECK_SPEED" \
        --gravity_alignment "$LOCO_GRAVITY_ALIGNMENT" 2>&1 || {
        err "Locomotion failed"; exit 1; }

    LOCO_DIAGNOSTICS_ENABLED=0
    if [ "$LOCO_CHECK_PENETRATION" != "0" ] || [ "$LOCO_CHECK_SPEED" != "0" ] || [ "$LOCO_GRAVITY_ALIGNMENT" != "0" ]; then
        LOCO_DIAGNOSTICS_ENABLED=1
    fi
    if [ -z "$(find "$LOCO_OUT/optimizer/results_filter" -name "*optimized.npz" 2>/dev/null | head -1)" ] && \
       [ "$LOCO_DIAGNOSTICS_ENABLED" = "1" ]; then
        warn "Locomotion checks filtered the clip; retrying height-only"
        "${LOCO_CMD_BASE[@]}" --check_penetration 0 --check_speed 0 --gravity_alignment 0 2>&1 || {
            err "Locomotion retry failed"; exit 1; }
    fi
    ok "Locomotion complete"
fi
LOCO_NPZ="$(find "$LOCO_OUT/optimizer/results_filter" -name "*optimized.npz" 2>/dev/null | head -1)"
[ -f "${LOCO_NPZ:-}" ] || { err "Locomotion output not found"; exit 1; }

# ---------------------------------------------------------------------------
# Stage 3: Temporal smoothing
# ---------------------------------------------------------------------------
if [ "$SKIP_EXISTING" = "1" ] && [ "$FORCE_LOCO" != "1" ] && [ "$FORCE_SMOOTH" != "1" ] && [ -f "$SMOOTH_NPZ" ]; then
    ok "Smooth already exists: $SMOOTH_NPZ"
else
    log ""
    log "[3/5] Savitzky-Golay smoothing"
    if run_phc_python "${LEGACY_PIPELINE}/smooth_motion.py" \
        --input "$LOCO_NPZ" --output "$SMOOTH_NPZ" \
        --pose_window 11 --trans_window 15 \
        --acc_threshold 10.0 --joint_acc_threshold 300.0 2>&1; then
        ok "Smooth complete: $SMOOTH_NPZ"
    else
        warn "Smoothing failed; using locomotion output"
        cp "$LOCO_NPZ" "$SMOOTH_NPZ"
    fi
fi
FINAL_PRE_NPZ="$SMOOTH_NPZ"

# ---------------------------------------------------------------------------
# Stage 3b: GVHMR camera export
# ---------------------------------------------------------------------------
if [ "$USE_GVHMR_CAMERA" = "1" ]; then
    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$GVHMR_CAMERA_NPZ" ] && \
       [ "$GVHMR_CAMERA_NPZ" -nt "$GVHMR_RESULTS" ] && [ "$GVHMR_CAMERA_NPZ" -nt "$FINAL_PRE_NPZ" ]; then
        ok "GVHMR camera already exists: $GVHMR_CAMERA_NPZ"
    else
        log ""
        log "[3b] Export GVHMR camera"
        cd "$GVHMR"
        "$PY_LOCO" "${SCRIPT_DIR}/export_gvhmr_camera.py" \
            --gvhmr_results "$GVHMR_RESULTS" \
            --reference_npz "$FINAL_PRE_NPZ" \
            --output "$GVHMR_CAMERA_NPZ" \
            --gravity_axis "$PHC_GRAVITY_AXIS" \
            --person_idx "$GVHMR_PERSON_IDX"
        ok "GVHMR camera: $GVHMR_CAMERA_NPZ"
    fi

    if [ "$ISAAC_CAMERA_USE_GVHMR_FOV" = "1" ] && [ -f "$GVHMR_CAMERA_NPZ" ]; then
        GVHMR_FOV="$("$PY_LOCO" -c 'import sys, numpy as np; d=np.load(sys.argv[1], allow_pickle=True); print(float(np.asarray(d["horizontal_fov_deg"]).reshape(-1)[0]))' "$GVHMR_CAMERA_NPZ")"
        if [ -n "$GVHMR_FOV" ]; then
            ISAAC_RECORD_FOV="$GVHMR_FOV"
            log "Isaac FOV from GVHMR: ${ISAAC_RECORD_FOV} deg"
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Stage 4: PHC repair
# ---------------------------------------------------------------------------
FINAL_POST_NPZ="$FINAL_PRE_NPZ"
if [ "$SKIP_PHC" = "1" ]; then
    warn "Skipping PHC (SKIP_PHC=1)"
elif [ "$SKIP_EXISTING" = "1" ] && [ "$FORCE_PHC" != "1" ] && [ -f "$PHC_SMOOTH_GROUNDED_NPZ" ]; then
    FINAL_POST_NPZ="$PHC_SMOOTH_GROUNDED_NPZ"
    ok "PHC smoothed grounded already exists: $FINAL_POST_NPZ"
elif [ "$SKIP_EXISTING" = "1" ] && [ "$FORCE_PHC" != "1" ] && [ -f "$PHC_GROUNDED_NPZ" ]; then
    FINAL_POST_NPZ="$PHC_GROUNDED_NPZ"
    ok "PHC grounded already exists: $FINAL_POST_NPZ"
else
    log ""
    log "[4/5] PHC physical repair"
    mkdir -p "$PHC_IN/001"
    cp "$FINAL_PRE_NPZ" "$PHC_IN/001/001_optimized.npz"
    cd "$PHC"
    export LD_LIBRARY_PATH="${PHC}/isaacgym:${LD_LIBRARY_PATH:-}"

    PHC_RECOVERY_ARGS=()
    [ "$PHC_ZERO_OUT_FAR" = "1" ] && PHC_RECOVERY_ARGS+=(--zero_out_far)
    [ "$PHC_ENABLE_EARLY_TERMINATION" = "1" ] && PHC_RECOVERY_ARGS+=(--enable_early_termination)

    PHC_POST_CHECK_ARGS=()
    if [ "$PHC_POST_CHECK_SPEED" = "1" ]; then
        PHC_POST_CHECK_ARGS=(--post_check_speed --speed_acc_threshold "$PHC_SPEED_ACC_THRESHOLD" --post_check_min_length "$PHC_POST_CHECK_MIN_LENGTH")
    fi

    PHC_HYDRA_OVERRIDES=()
    [ -n "$PHC_CONTROL_DECIMATION" ] && PHC_HYDRA_OVERRIDES+=("control.decimation=$PHC_CONTROL_DECIMATION")
    [ -n "$PHC_FIX_HEIGHT" ] && PHC_HYDRA_OVERRIDES+=("phc_fix_height=$PHC_FIX_HEIGHT")
    [ -n "$PHC_KP_SCALE" ] && PHC_HYDRA_OVERRIDES+=("env.kp_scale=$PHC_KP_SCALE")
    if [ -n "$PHC_EXTRA_OVERRIDES" ]; then
        read -r -a PHC_EXTRA_OVERRIDE_ITEMS <<< "$PHC_EXTRA_OVERRIDES"
        PHC_HYDRA_OVERRIDES+=("${PHC_EXTRA_OVERRIDE_ITEMS[@]}")
    fi
    PHC_HYDRA_ARGS=()
    [ "${#PHC_HYDRA_OVERRIDES[@]}" -gt 0 ] && PHC_HYDRA_ARGS=(--hydra_overrides "${PHC_HYDRA_OVERRIDES[@]}")

    ISAAC_CAMERA_PATH_ARG=()
    if [ -n "${ISAAC_CAMERA_PATH:-}" ] && [ -f "$ISAAC_CAMERA_PATH" ]; then
        ISAAC_CAMERA_PATH_ARG=(--isaac_camera_path "$ISAAC_CAMERA_PATH")
        log "Isaac camera: $ISAAC_CAMERA_MODE ($ISAAC_CAMERA_PATH)"
    else
        log "Isaac camera: $ISAAC_CAMERA_MODE"
    fi

    if run_phc_python scripts/data_process/batch_repair_zitai.py \
        --input_root "$PHC_IN" --repaired_root "$PHC_OUT" \
        --states_root "${WORK}/phc_states" \
        --rendering_output_root "$PHC_RENDERINGS" \
        --record_isaac_gym \
        --isaac_record_width "$ISAAC_RECORD_WIDTH" \
        --isaac_record_height "$ISAAC_RECORD_HEIGHT" \
        --isaac_record_fps "$ISAAC_RECORD_FPS" \
        --isaac_record_fov "$ISAAC_RECORD_FOV" \
        --isaac_camera_distance "$ISAAC_CAMERA_DISTANCE" \
        --isaac_camera_side "$ISAAC_CAMERA_SIDE" \
        --isaac_camera_height "$ISAAC_CAMERA_HEIGHT" \
        --isaac_camera_target_height "$ISAAC_CAMERA_TARGET_HEIGHT" \
        --isaac_camera_smoothing "$ISAAC_CAMERA_SMOOTHING" \
        --isaac_camera_mode "$ISAAC_CAMERA_MODE" \
        "${ISAAC_CAMERA_PATH_ARG[@]}" \
        --primitive_model_path "$PHC_PRIMITIVE" \
        --composer_checkpoint_path "$PHC_COMPOSER" \
        --gravity_axis "$PHC_GRAVITY_AXIS" --keep_shape \
        --target_fps 30 --episode_length 3000 \
        --no_virtual_display --parallel_workers 1 \
        "${PHC_RECOVERY_ARGS[@]}" "${PHC_POST_CHECK_ARGS[@]}" "${PHC_HYDRA_ARGS[@]}" 2>&1; then
        ok "PHC complete"
        REPAIRED="$(find_latest_repaired_npz "$PHC_OUT" | sort -nr | head -1 | cut -d' ' -f2-)"
        if [ -n "$REPAIRED" ]; then
            FINAL_POST_NPZ="$REPAIRED"
            PHC_RAN=1
            ok "PHC output: $FINAL_POST_NPZ"
            ground_fix_npz "$FINAL_POST_NPZ" "$PHC_GROUNDED_NPZ" "PHC export"
        else
            warn "No repaired output found; using pre-PHC NPZ"
        fi
    else
        warn "PHC failed; using pre-PHC NPZ"
    fi
fi

if [ "$PHC_RAN" = "1" ] && [ "$PHC_POST_SMOOTH" = "1" ] && [ -f "$FINAL_POST_NPZ" ]; then
    log ""
    log "[4b/5] Smooth PHC export spikes"
    if run_phc_python "${LEGACY_PIPELINE}/smooth_motion.py" \
        --input "$FINAL_POST_NPZ" --output "$PHC_SMOOTH_NPZ" \
        --report "$PHC_SMOOTH_REPORT" \
        --pose_window "$PHC_POST_SMOOTH_POSE_WINDOW" \
        --trans_window "$PHC_POST_SMOOTH_TRANS_WINDOW" \
        --acc_threshold "$PHC_POST_SMOOTH_ACC_THRESHOLD" \
        --root_step_threshold "$PHC_POST_SMOOTH_ROOT_STEP_THRESHOLD" \
        --joint_step_threshold "$PHC_POST_SMOOTH_JOINT_STEP_THRESHOLD" \
        --joint_acc_threshold "$PHC_POST_SMOOTH_JOINT_ACC_THRESHOLD" \
        --mad_multiplier "$PHC_POST_SMOOTH_MAD_MULTIPLIER" \
        --dilate "$PHC_POST_SMOOTH_DILATE" 2>&1; then
        FINAL_POST_NPZ="$PHC_SMOOTH_NPZ"
        ok "PHC spike-smooth output: $FINAL_POST_NPZ"
        ground_fix_npz "$FINAL_POST_NPZ" "$PHC_SMOOTH_GROUNDED_NPZ" "PHC spike-smooth"
    else
        warn "PHC spike smoothing failed; keeping original PHC output"
    fi
fi

# ---------------------------------------------------------------------------
# Stage 5: Comparison render
# ---------------------------------------------------------------------------
log ""
if [ "$RENDER_COMPARISON" = "1" ]; then
    log "[5/5] Comparison MP4"
    GVHMR_INCAM="$(find_first "$GVHMR_OUT" "1_incam.mp4")"
    PHC_VIDEO=""
    USE_RAW_ISAAC_VIDEO=0
    if [ "$PHC_COMPARISON_USE_RAW_ISAAC" = "1" ] || [ "$PHC_POST_SMOOTH" != "1" ]; then
        USE_RAW_ISAAC_VIDEO=1
    fi
    if [ "$PHC_RAN" = "1" ] && [ "$USE_RAW_ISAAC_VIDEO" = "1" ]; then
        PHC_VIDEO="$(find "$PHC_RENDERINGS" -name "*.mp4" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
    fi

    INCAM_ARG=()
    [ -f "${GVHMR_INCAM:-}" ] && INCAM_ARG=(--gvhmr_incam "$GVHMR_INCAM")

    PHC_VIDEO_ARG=()
    [ -f "${PHC_VIDEO:-}" ] && PHC_VIDEO_ARG=(--phc_video "$PHC_VIDEO")

    run_phc_python "${LEGACY_PIPELINE}/render_comparison.py" \
        --video "$VIDEO" \
        --gvhmr_npz "$LOCO_SOURCE" --phc_npz "$FINAL_POST_NPZ" \
        --output "$COMPARISON_MP4" \
        --model_path "${PHC}/data/smpl" \
        --max_frames "$MAX_FRAMES" --fps "$COMPARISON_FPS" --panel_size "$PANEL_SIZE" \
        --hide_labels \
        "${INCAM_ARG[@]}" "${PHC_VIDEO_ARG[@]}" 2>&1 || warn "Comparison rendering failed"
else
    warn "Skipping slow Matplotlib comparison MP4 (RENDER_COMPARISON=0)"
fi

echo ""
echo "============================================================"
echo "  Pipeline complete"
echo "============================================================"
echo "  GVHMR results:   ${GVHMR_RESULTS:-missing}"
echo "  Body NPZ:        $GVHMR_CONVERTED"
echo "  Hand sidecar:    $GVHMR_HANDS_NPZ"
echo "  Locomotion NPZ:  ${LOCO_NPZ:-missing}"
echo "  Final NPZ:       ${FINAL_POST_NPZ:-missing}"
if [ "$RENDER_COMPARISON" = "1" ]; then
    echo "  Comparison MP4:  $COMPARISON_MP4"
else
    echo "  Comparison MP4:  skipped (RENDER_COMPARISON=0)"
fi
