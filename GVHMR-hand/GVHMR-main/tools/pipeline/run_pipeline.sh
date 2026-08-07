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
# A scene-aligned continuation supplies 001_smoothed.npz and the matching
# camera explicitly.  It must never revisit GVHMR, conversion, locomotion, or
# temporal smoothing, otherwise those stages overwrite the static-coordinate
# source motion before PHC/GMR consume it.
PIPELINE_POST_ONLY="${PIPELINE_POST_ONLY:-0}"
SMOOTH_RAN=0  # invariant for both full and scene-post-only execution paths
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
GVHMR_HAND4WHOLEPP_CROP_TRACKING="${GVHMR_HAND4WHOLEPP_CROP_TRACKING:-off}"
GVHMR_HAND4WHOLEPP_CROP_TRACKING_MAX_GAP="${GVHMR_HAND4WHOLEPP_CROP_TRACKING_MAX_GAP:-8}"
GVHMR_HAND4WHOLEPP_CROP_TRACKING_MAX_PREDICTION_GAP="${GVHMR_HAND4WHOLEPP_CROP_TRACKING_MAX_PREDICTION_GAP:-2}"
GVHMR_HAND4WHOLEPP_CROP_TRACKING_DIRECT_OBSERVATION_QUALITY="${GVHMR_HAND4WHOLEPP_CROP_TRACKING_DIRECT_OBSERVATION_QUALITY:-0.75}"
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
# Keep orientation policy separate from local finger-articulation filtering so
# a visual A/B can isolate the cause of a palm/back artifact.
GVHMR_TEMPORAL_FILTER_GLOBAL_ORIENT_FILL_MODE="${GVHMR_TEMPORAL_FILTER_GLOBAL_ORIENT_FILL_MODE:-interpolate}"
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
GVHMR_FINGER_FILTER_WRIST_MODE="${GVHMR_FINGER_FILTER_WRIST_MODE:-smooth}"
GVHMR_FINGER_FILTER_HAND_SIZE_FLOOR_RATIO="${GVHMR_FINGER_FILTER_HAND_SIZE_FLOOR_RATIO:-0.96}"
GVHMR_FINGER_FILTER_OPEN_RESCUE_WEIGHT="${GVHMR_FINGER_FILTER_OPEN_RESCUE_WEIGHT:-0.30}"
GVHMR_FINGER_FILTER_OPEN_RESCUE_SMOOTH_WEIGHT="${GVHMR_FINGER_FILTER_OPEN_RESCUE_SMOOTH_WEIGHT:-0.70}"
GVHMR_FINGER_FILTER_OPEN_RESCUE_ENABLED="${GVHMR_FINGER_FILTER_OPEN_RESCUE_ENABLED:-1}"
GVHMR_FINAL_FINGER_SMOOTH_AFTER_VISIBLE_REFINE="${GVHMR_FINAL_FINGER_SMOOTH_AFTER_VISIBLE_REFINE:-1}"
GVHMR_FINAL_FINGER_SMOOTH_WINDOW="${GVHMR_FINAL_FINGER_SMOOTH_WINDOW:-9}"
GVHMR_FINAL_FINGER_RELIABLE_SMOOTH_WEIGHT="${GVHMR_FINAL_FINGER_RELIABLE_SMOOTH_WEIGHT:-0.20}"
GVHMR_FINAL_FINGER_WEAK_SMOOTH_WEIGHT="${GVHMR_FINAL_FINGER_WEAK_SMOOTH_WEIGHT:-0.60}"
GVHMR_FINAL_FINGER_BAD_SMOOTH_WEIGHT="${GVHMR_FINAL_FINGER_BAD_SMOOTH_WEIGHT:-0.85}"
# Experimental correctness repair for Hand4Whole++: filters edit MANO pose
# and joints separately, so recompute the final joints from the final pose
# before converting/retargeting.  It remains globally opt-in because other
# backends do not expose this direct-MANO contract; human_sharpa.yaml enables
# it after the validated Hand4Whole++ A/B.
GVHMR_RECOMPUTE_DIRECT_MANO="${GVHMR_RECOMPUTE_DIRECT_MANO:-0}"
GVHMR_RECOMPUTE_DIRECT_MANO_BATCH_SIZE="${GVHMR_RECOMPUTE_DIRECT_MANO_BATCH_SIZE:-256}"
# Final-render-space, evidence-gated wrist correction. Pipeline configs choose
# whether to enable it; it is never a palm/back flip.
GVHMR_VISIBLE_HAND_REFINE="${GVHMR_VISIBLE_HAND_REFINE:-0}"
GVHMR_VISIBLE_HAND_REFINE_DEVICE="${GVHMR_VISIBLE_HAND_REFINE_DEVICE:-auto}"
GVHMR_VISIBLE_HAND_REFINE_BATCH_SIZE="${GVHMR_VISIBLE_HAND_REFINE_BATCH_SIZE:-16}"
GVHMR_VISIBLE_HAND_REFINE_STEPS="${GVHMR_VISIBLE_HAND_REFINE_STEPS:-8}"
GVHMR_VISIBLE_HAND_REFINE_LR="${GVHMR_VISIBLE_HAND_REFINE_LR:-0.02}"
GVHMR_VISIBLE_HAND_REFINE_PRIOR_WEIGHT="${GVHMR_VISIBLE_HAND_REFINE_PRIOR_WEIGHT:-0.02}"
GVHMR_VISIBLE_HAND_REFINE_FIT_CONFIDENCE="${GVHMR_VISIBLE_HAND_REFINE_FIT_CONFIDENCE:-0.60}"
GVHMR_VISIBLE_HAND_REFINE_FIT_MIN_KEYPOINTS="${GVHMR_VISIBLE_HAND_REFINE_FIT_MIN_KEYPOINTS:-12}"
GVHMR_VISIBLE_HAND_REFINE_FIT_PARTITION="${GVHMR_VISIBLE_HAND_REFINE_FIT_PARTITION:-all}"
GVHMR_VISIBLE_HAND_REFINE_HOLDOUT_MIN_KEYPOINTS="${GVHMR_VISIBLE_HAND_REFINE_HOLDOUT_MIN_KEYPOINTS:-3}"
GVHMR_VISIBLE_HAND_REFINE_HOLDOUT_MAX_RELATIVE_REGRESSION_PX="${GVHMR_VISIBLE_HAND_REFINE_HOLDOUT_MAX_RELATIVE_REGRESSION_PX:-1.0}"
GVHMR_VISIBLE_HAND_REFINE_MAX_DELTA_DEGREES="${GVHMR_VISIBLE_HAND_REFINE_MAX_DELTA_DEGREES:-25}"
GVHMR_VISIBLE_HAND_REFINE_MIN_RELATIVE_IMPROVEMENT="${GVHMR_VISIBLE_HAND_REFINE_MIN_RELATIVE_IMPROVEMENT:-0.25}"
GVHMR_VISIBLE_HAND_REFINE_MIN_RELATIVE_IMPROVEMENT_PX="${GVHMR_VISIBLE_HAND_REFINE_MIN_RELATIVE_IMPROVEMENT_PX:-2.0}"
GVHMR_VISIBLE_HAND_REFINE_MAX_ABSOLUTE_REGRESSION_PX="${GVHMR_VISIBLE_HAND_REFINE_MAX_ABSOLUTE_REGRESSION_PX:-2.0}"
GVHMR_VISIBLE_HAND_REFINE_MAX_ANCHOR_ERROR_PX="${GVHMR_VISIBLE_HAND_REFINE_MAX_ANCHOR_ERROR_PX:-20.0}"
GVHMR_VISIBLE_HAND_REFINE_MAX_ANCHOR_ERROR_BBOX_RATIO="${GVHMR_VISIBLE_HAND_REFINE_MAX_ANCHOR_ERROR_BBOX_RATIO:-0.25}"
GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_HAND_CONFIDENCE="${GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_HAND_CONFIDENCE:-0.45}"
GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_MIN_KEYPOINTS="${GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_MIN_KEYPOINTS:-8}"
GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_MEAN_CONFIDENCE="${GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_MEAN_CONFIDENCE:-0.50}"
GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_WRIST_CONFIDENCE="${GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_WRIST_CONFIDENCE:-0.45}"
GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_MIN_BBOX_DIAGONAL_PX="${GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_MIN_BBOX_DIAGONAL_PX:-96}"
GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_MIN_RUN="${GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_MIN_RUN:-3}"
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
BODY_SMOOTH_POSE_WINDOW="${BODY_SMOOTH_POSE_WINDOW:-11}"
BODY_SMOOTH_TRANS_WINDOW="${BODY_SMOOTH_TRANS_WINDOW:-15}"
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
GVHMR_INPUT_CACHE_MARKER="$WORK/.gvhmr_input_cache_v1"
FILTER_CONFIG_MARKER="$WORK/.filter_config"
FILTERED_RENDER_MARKER="$WORK/.filtered_render_config"
GVHMR_CLIP_CACHE="$GVHMR_OUT/$VIDEO_NAME"

HAND_CODE_FINGERPRINT="$(
    sha256sum \
        "$GVHMR/tools/processor/generate_smplxs.py" \
        "$GVHMR/tools/processor/run_hand4wholepp_video.py" 2>/dev/null |
        sha256sum | cut -d' ' -f1
)"
HAND_AUX_CODE_FINGERPRINT="$(sha256sum "$GVHMR/tools/processor/hand_bbox_tracking.py" "$GVHMR/third-party/Hand4Whole-plus-plus_RELEASE/main/model.py" 2>/dev/null | sha256sum | cut -d' ' -f1)"
HAND_CROP_TRACKING_FINGERPRINT="${GVHMR_HAND4WHOLEPP_CROP_TRACKING}|${GVHMR_HAND4WHOLEPP_CROP_TRACKING_MAX_GAP}|${GVHMR_HAND4WHOLEPP_CROP_TRACKING_MAX_PREDICTION_GAP}|${GVHMR_HAND4WHOLEPP_CROP_TRACKING_DIRECT_OBSERVATION_QUALITY}"
VIDEO_INPUT_FINGERPRINT="$({
    realpath "$VIDEO"
    stat -c '%s:%Y' "$VIDEO"
    if [ -f "${VIDEO}.config" ]; then
        sha256sum "${VIDEO}.config"
    fi
} | sha256sum | cut -d' ' -f1)"
GVHMR_INPUT_CACHE_MARKER_VALUE="schema=1 video=$VIDEO_INPUT_FINGERPRINT"

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
        "video=$VIDEO_INPUT_FINGERPRINT" \
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
HAND_CONFIG_FINGERPRINT="$(printf '%s\n%s\n%s' "$HAND_CONFIG_TEXT" "$HAND_AUX_CODE_FINGERPRINT" "$HAND_CROP_TRACKING_FINGERPRINT" | sha256sum | cut -d' ' -f1)"
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
FINAL_NPZ="$WORK/001_final.npz"
FINAL_SELECTION_JSON="$WORK/final_motion_selection.json"
PHC_ATTEMPTS="$WORK/phc_attempts.jsonl"

PHC_IN="$WORK/phc_in"
PHC_OUT="$WORK/phc_repaired"
PHC_RENDERINGS="$WORK/phc_renderings"
# Recorded-state evidence; a completed process alone is not a PHC success.
PHC_TRACKING_AUDIT="$WORK/phc_states/001/001_optimized_tracking_audit.json"
PHC_GROUNDED_NPZ="$WORK/001_phc_grounded.npz"
PHC_SMOOTH_NPZ="$WORK/001_phc_smoothed.npz"
PHC_SMOOTH_GROUNDED_NPZ="$WORK/001_phc_smoothed_grounded.npz"
PHC_SMOOTH_REPORT="$WORK/001_phc_smoothed_report.json"
PHC_RAN=0
PHC_ATTEMPT_ID=""

COMPARISON_MP4="$WORK/${VIDEO_NAME}_comparison.mp4"

find_first() {
    find "$1" -name "$2" -print -quit 2>/dev/null
}

write_gvhmr_input_cache_marker() {
    local marker_tmp
    marker_tmp="$(mktemp "${GVHMR_INPUT_CACHE_MARKER}.tmp.XXXXXX")" || {
        err "Failed to create GVHMR input-cache marker temporary file"
        exit 1
    }
    if ! printf '%s\n' "$GVHMR_INPUT_CACHE_MARKER_VALUE" > "$marker_tmp"; then
        rm -f -- "$marker_tmp"
        err "Failed to write GVHMR input-cache marker"
        exit 1
    fi
    mv -f -- "$marker_tmp" "$GVHMR_INPUT_CACHE_MARKER"
}

find_latest_repaired_npz() {
    local root="$1"
    find "$root" -name "*_repaired.npz" -printf '%T@ %p\n' 2>/dev/null
    find "$root" -name "*_validated.npz" -printf '%T@ %p\n' 2>/dev/null
}

phc_tracking_pass() {
    local report="$1"
    [ -f "$report" ] && grep -Eq '"status"[[:space:]]*:[[:space:]]*"passed"' "$report"
}

append_phc_attempt() {
    # The selected final motion is a current-state pointer; this append-only
    # ledger preserves every PHC outcome, including an intentional SKIP_PHC
    # rerun after a previous tracking rejection.
    local outcome="$1"
    local tracking_status="missing"
    if [ -f "$PHC_TRACKING_AUDIT" ]; then
        tracking_status="$(grep -Eo '"status"[[:space:]]*:[[:space:]]*"[^"]+"' "$PHC_TRACKING_AUDIT" | head -1 | sed -E 's/.*"([^"]+)"$/\1/' || true)"
        [ -n "$tracking_status" ] || tracking_status="unreadable"
    fi
    if [ -z "$PHC_ATTEMPT_ID" ]; then
        PHC_ATTEMPT_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$-${RANDOM}"
    fi
    PHC_ATTEMPT_ID="$PHC_ATTEMPT_ID" \
    PHC_ATTEMPT_OUTCOME="$outcome" \
    PHC_TRACKING_STATUS="$tracking_status" \
    PHC_ATTEMPTS_PATH="$PHC_ATTEMPTS" \
    PHC_INPUT_PATH="$FINAL_PRE_NPZ" \
    PHC_PRIMITIVE_PATH="$PHC_PRIMITIVE" \
    PHC_COMPOSER_PATH="$PHC_COMPOSER" \
    PHC_TRACKING_PATH="$PHC_TRACKING_AUDIT" \
    PHC_OVERRIDES="$PHC_EXTRA_OVERRIDES" \
    "$PY_LOCO" - <<'PY' || warn "Could not append PHC attempt ledger"
import datetime
import hashlib
import json
import os
from pathlib import Path


def digest(value: str):
    path = Path(value)
    if not path.is_file():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


ledger = Path(os.environ["PHC_ATTEMPTS_PATH"])
ledger.parent.mkdir(parents=True, exist_ok=True)
record = {
    "schema_version": 1,
    "attempt_id": os.environ["PHC_ATTEMPT_ID"],
    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "outcome": os.environ["PHC_ATTEMPT_OUTCOME"],
    "input_sha256": digest(os.environ["PHC_INPUT_PATH"]),
    "primitive_checkpoint_sha256": digest(os.environ["PHC_PRIMITIVE_PATH"]),
    "composer_checkpoint_sha256": digest(os.environ["PHC_COMPOSER_PATH"]),
    "hydra_overrides": os.environ.get("PHC_OVERRIDES", ""),
    "tracking_report": os.environ["PHC_TRACKING_PATH"],
    "tracking_status": os.environ["PHC_TRACKING_STATUS"],
}
with ledger.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
PY
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

    # A caller's conda cross-compiler can hide host system headers from the
    # Isaac Gym JIT extension. Prefer host compilers unless explicitly set.
    local phc_cc="${PHC_CC:-gcc}"
    local phc_cxx="${PHC_CXX:-g++}"
    env \
        LD_LIBRARY_PATH="$ld_paths" \
        PATH="${phc_prefix}/bin:${PATH:-}" \
        CC="$phc_cc" \
        CXX="$phc_cxx" \
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

publish_final_motion() {
    local selected="$1"
    local reason="$2"
    local selected_real selected_sha256 stage link_tmp json_tmp
    [ -f "$selected" ] || { err "final motion source is missing: $selected"; exit 1; }
    selected_real="$(realpath "$selected")"
    selected_sha256="$(sha256sum "$selected_real" | awk '{print $1}')"
    case "$(basename "$selected_real")" in
        001_phc_smoothed_grounded.npz) stage="phc_smoothed_grounded" ;;
        001_phc_smoothed.npz) stage="phc_smoothed" ;;
        001_phc_grounded.npz) stage="phc_grounded" ;;
        001_phc.npz|*_repaired.npz|*_validated.npz) stage="phc" ;;
        001_smoothed.npz) stage="smoothed" ;;
        *) stage="unknown" ;;
    esac
    link_tmp="${FINAL_NPZ}.tmp.$$"
    json_tmp="${FINAL_SELECTION_JSON}.tmp.$$"
    rm -f -- "$link_tmp" "$json_tmp"
    ln -s "$selected_real" "$link_tmp"
    mv -fT "$link_tmp" "$FINAL_NPZ"
    printf '{"schema_version":2,"selected_stage":"%s","selection_reason":"%s","selected_file":"%s","selected_sha256":"%s","phc_attempt_id":"%s"}\n' \
        "$stage" "$reason" "$(basename "$selected_real")" "$selected_sha256" "${PHC_ATTEMPT_ID:-not_attempted}" > "$json_tmp"
    mv -fT "$json_tmp" "$FINAL_SELECTION_JSON"
    ok "Final motion selection: $stage ($reason) -> $FINAL_NPZ"
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

if [ "$PIPELINE_POST_ONLY" = "1" ]; then
    require_file "$SMOOTH_NPZ" "scene-aligned 001_smoothed.npz"
    require_file "$GVHMR_HANDS_NPZ" "preserved hand sidecar"
    if [ "$USE_GVHMR_CAMERA" = "1" ]; then
        require_file "$GVHMR_CAMERA_NPZ" "scene-aligned GVHMR camera"
    fi
    LOCO_SOURCE="$SMOOTH_NPZ"
    LOCO_NPZ="$SMOOTH_NPZ"
    FINAL_PRE_NPZ="$SMOOTH_NPZ"
    log "[SCENE_POST] reusing canonical motion; skipping GVHMR/Locomotion/Smoothing"
else

# generate_smplxs caches ViTPose, ViT features, and MANO sidecars below the
# per-video GVHMR directory.  A work-video re-encode (for example 25 -> 30
# FPS) changes the frame count, so mixing a newly generated MANO file with an
# old ViTPose cache causes temporal filters to fail with incompatible lengths.
GVHMR_INPUT_CACHE_MATCH=0
if [ -f "$GVHMR_INPUT_CACHE_MARKER" ] && [ "$(cat "$GVHMR_INPUT_CACHE_MARKER")" = "$GVHMR_INPUT_CACHE_MARKER_VALUE" ]; then
    GVHMR_INPUT_CACHE_MATCH=1
fi
if [ "$GVHMR_INPUT_CACHE_MATCH" != "1" ] && [ -d "$GVHMR_CLIP_CACHE" ]; then
    GVHMR_OUT_REAL="$(realpath -m "$GVHMR_OUT")"
    GVHMR_CLIP_CACHE_REAL="$(realpath -m "$GVHMR_CLIP_CACHE")"
    if [ "$(dirname "$GVHMR_CLIP_CACHE_REAL")" != "$GVHMR_OUT_REAL" ]; then
        err "Refusing to clear unsafe GVHMR cache path: $GVHMR_CLIP_CACHE_REAL"
        exit 1
    fi
    warn "GVHMR input-cache marker changed; clearing stale GVHMR cache: $GVHMR_CLIP_CACHE_REAL"
    rm -rf -- "$GVHMR_CLIP_CACHE_REAL"
fi

# ---------------------------------------------------------------------------
# Stage 1: GVHMR-hand inference
# ---------------------------------------------------------------------------
CONVERT_RAN=0
LOCO_RAN=0
SMOOTH_RAN=0
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
   [ "$BACKEND_MATCH" = "1" ] && [ "$HAND_CONFIG_MATCH" = "1" ] && [ "$GVHMR_INPUT_CACHE_MATCH" = "1" ] && \
   [ -f "${GVHMR_RESULTS:-}" ] && [ -f "${MANO_PARAMS:-}" ] && \
   [ "$GVHMR_RENDER_READY" = "1" ]; then
    ok "GVHMR-hand results already exist (backend: $GVHMR_HAND_BACKEND)"
    printf '%s\n' "$GVHMR_HAND_BACKEND" > "$HAND_BACKEND_MARKER"
    printf '%s\n' "$HAND_CONFIG_FINGERPRINT" > "$HAND_CONFIG_MARKER"
else
    if [ "$SKIP_EXISTING" = "1" ] && [ -f "${MANO_PARAMS:-}" ] && [ "$BACKEND_MATCH" = "0" ]; then
        warn "Hand backend changed (marker: $(cat "$HAND_BACKEND_MARKER" 2>/dev/null || echo 'none') -> $GVHMR_HAND_BACKEND); forcing re-process"
    elif [ "$SKIP_EXISTING" = "1" ] && [ -f "${MANO_PARAMS:-}" ] && [ "$HAND_CONFIG_MATCH" = "0" ]; then
        warn "Hand inference config changed; forcing hand preprocess"
    fi
    FORCE_HAND_THIS_RUN="$GVHMR_FORCE_HAND_PREPROCESS"
    if [ "$GVHMR_INPUT_CACHE_MATCH" != "1" ]; then
        warn "GVHMR input cache marker is absent or mismatched; forcing hand preprocess"
        FORCE_HAND_THIS_RUN=1
    fi
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
            require_file "$GVHMR_HAND4WHOLEPP_ROOT/common/nets/WiLoR/pretrained_models/detector.pt" "Hand4Whole++ WiLoR detector checkpoint"
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
        --hand4wholepp_crop_tracking "$GVHMR_HAND4WHOLEPP_CROP_TRACKING"
        --hand4wholepp_crop_tracking_max_gap "$GVHMR_HAND4WHOLEPP_CROP_TRACKING_MAX_GAP"
        --hand4wholepp_crop_tracking_max_prediction_gap "$GVHMR_HAND4WHOLEPP_CROP_TRACKING_MAX_PREDICTION_GAP"
        --hand4wholepp_crop_tracking_direct_observation_quality "$GVHMR_HAND4WHOLEPP_CROP_TRACKING_DIRECT_OBSERVATION_QUALITY"
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
       [ "$GVHMR_FILTER_MANO_FINGERS" = "1" ] || \
       [ "$GVHMR_VISIBLE_HAND_REFINE" = "1" ] || \
       [ "$GVHMR_RECOMPUTE_DIRECT_MANO" = "1" ]; then
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

    GVHMR_RESULTS="$(find_first "$GVHMR_OUT" "hmr4d_results.pt")"
    MANO_PARAMS="$(find_first "$GVHMR_OUT" "mano_params.pt")"
    VITPOSE_WHOLEBODY="$(find_first "$GVHMR_OUT" "vitpose_wholebody.pt")"
    GVHMR_INCAM="$(find_first "$GVHMR_OUT" "1_incam.mp4")"
    [ -f "${GVHMR_RESULTS:-}" ] || { err "GVHMR-hand inference returned no hmr4d_results.pt"; exit 1; }
    [ -f "${VITPOSE_WHOLEBODY:-}" ] || { err "GVHMR-hand inference returned no vitpose_wholebody.pt"; exit 1; }
    printf '%s\n' "$GVHMR_HAND_BACKEND" > "$HAND_BACKEND_MARKER"
    printf '%s\n' "$HAND_CONFIG_FINGERPRINT" > "$HAND_CONFIG_MARKER"
    write_gvhmr_input_cache_marker
fi

[ -f "${GVHMR_RESULTS:-}" ] || { err "hmr4d_results.pt not found"; exit 1; }
[ -f "${MANO_PARAMS:-}" ] || warn "mano_params.pt not found; hand sidecar will contain zero hands"

FILTER_CONFIG_TEXT="$(
    FILTER_CODE_FINGERPRINT="$(
        sha256sum \
            "$GVHMR/tools/processor/filter_mano_wrist.py" \
            "$GVHMR/tools/processor/filter_mano_temporal.py" \
            "$GVHMR/tools/processor/filter_mano_fingers.py" \
            "$GVHMR/tools/processor/visible_hand_evidence.py" \
            "$GVHMR/tools/processor/refine_visible_hand_orientation.py" \
            "$GVHMR/tools/processor/recompute_direct_mano_joints.py" \
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
        "temporal_global_orient_fill=$GVHMR_TEMPORAL_FILTER_GLOBAL_ORIENT_FILL_MODE" \
        "finger_filter=$GVHMR_FILTER_MANO_FINGERS" \
        "finger_window=$GVHMR_FINGER_FILTER_SMOOTH_WINDOW" \
        "finger_weights=$GVHMR_FINGER_FILTER_RELIABLE_SMOOTH_WEIGHT,$GVHMR_FINGER_FILTER_WEAK_SMOOTH_WEIGHT,$GVHMR_FINGER_FILTER_BAD_SMOOTH_WEIGHT" \
        "finger_fill=$GVHMR_FINGER_FILTER_MAX_INTERP_GAP,$GVHMR_FINGER_FILTER_MAX_EDGE_HOLD" \
        "finger_limits=$GVHMR_FINGER_FILTER_MAX_JOINT_ANGLE_DELTA,$GVHMR_FINGER_FILTER_MAX_JOINT_XYZ_DELTA" \
        "wrist_smooth=$GVHMR_FINGER_FILTER_WRIST_SMOOTH_WINDOW,$GVHMR_FINGER_FILTER_WRIST_RELIABLE_SMOOTH_WEIGHT,$GVHMR_FINGER_FILTER_WRIST_WEAK_SMOOTH_WEIGHT,$GVHMR_FINGER_FILTER_WRIST_BAD_SMOOTH_WEIGHT,$GVHMR_FINGER_FILTER_MAX_WRIST_ANGLE_DELTA" \
        "wrist_mode=$GVHMR_FINGER_FILTER_WRIST_MODE" \
        "finger_floor=$GVHMR_FINGER_FILTER_HAND_SIZE_FLOOR_RATIO" \
        "finger_open=$GVHMR_FINGER_FILTER_OPEN_RESCUE_ENABLED,$GVHMR_FINGER_FILTER_OPEN_RESCUE_WEIGHT,$GVHMR_FINGER_FILTER_OPEN_RESCUE_SMOOTH_WEIGHT" \
        "final_finger_smooth_after_visible_refine=$GVHMR_FINAL_FINGER_SMOOTH_AFTER_VISIBLE_REFINE,$GVHMR_FINAL_FINGER_SMOOTH_WINDOW,$GVHMR_FINAL_FINGER_RELIABLE_SMOOTH_WEIGHT,$GVHMR_FINAL_FINGER_WEAK_SMOOTH_WEIGHT,$GVHMR_FINAL_FINGER_BAD_SMOOTH_WEIGHT" \
        "visible_refine=$GVHMR_VISIBLE_HAND_REFINE,$GVHMR_VISIBLE_HAND_REFINE_DEVICE,$GVHMR_VISIBLE_HAND_REFINE_BATCH_SIZE,$GVHMR_VISIBLE_HAND_REFINE_STEPS,$GVHMR_VISIBLE_HAND_REFINE_LR,$GVHMR_VISIBLE_HAND_REFINE_PRIOR_WEIGHT" \
        "visible_refine_fit=$GVHMR_VISIBLE_HAND_REFINE_FIT_CONFIDENCE,$GVHMR_VISIBLE_HAND_REFINE_FIT_MIN_KEYPOINTS,$GVHMR_VISIBLE_HAND_REFINE_FIT_PARTITION,$GVHMR_VISIBLE_HAND_REFINE_HOLDOUT_MIN_KEYPOINTS,$GVHMR_VISIBLE_HAND_REFINE_HOLDOUT_MAX_RELATIVE_REGRESSION_PX,$GVHMR_VISIBLE_HAND_REFINE_MAX_DELTA_DEGREES,$GVHMR_VISIBLE_HAND_REFINE_MIN_RELATIVE_IMPROVEMENT,$GVHMR_VISIBLE_HAND_REFINE_MIN_RELATIVE_IMPROVEMENT_PX,$GVHMR_VISIBLE_HAND_REFINE_MAX_ABSOLUTE_REGRESSION_PX" \
        "visible_refine_anchor=$GVHMR_VISIBLE_HAND_REFINE_MAX_ANCHOR_ERROR_PX,$GVHMR_VISIBLE_HAND_REFINE_MAX_ANCHOR_ERROR_BBOX_RATIO" \
        "visible_refine_evidence=$GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_HAND_CONFIDENCE,$GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_MIN_KEYPOINTS,$GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_MEAN_CONFIDENCE,$GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_WRIST_CONFIDENCE,$GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_MIN_BBOX_DIAGONAL_PX,$GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_MIN_RUN" \
        "direct_mano_recompute=$GVHMR_RECOMPUTE_DIRECT_MANO,$GVHMR_RECOMPUTE_DIRECT_MANO_BATCH_SIZE" \
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

case "$GVHMR_RECOMPUTE_DIRECT_MANO" in
    0|1) ;;
    *) err "GVHMR_RECOMPUTE_DIRECT_MANO must be 0 or 1, got: $GVHMR_RECOMPUTE_DIRECT_MANO"; exit 1 ;;
esac

case "$GVHMR_VISIBLE_HAND_REFINE" in
    0|1) ;;
    *) err "GVHMR_VISIBLE_HAND_REFINE must be 0 or 1, got: $GVHMR_VISIBLE_HAND_REFINE"; exit 1 ;;
esac
case "$GVHMR_FINAL_FINGER_SMOOTH_AFTER_VISIBLE_REFINE" in
    0|1) ;;
    *) err "GVHMR_FINAL_FINGER_SMOOTH_AFTER_VISIBLE_REFINE must be 0 or 1, got: $GVHMR_FINAL_FINGER_SMOOTH_AFTER_VISIBLE_REFINE"; exit 1 ;;
esac
if [ "$GVHMR_VISIBLE_HAND_REFINE" = "1" ] && [ "$GVHMR_HAND_BACKEND" != "hand4wholepp" ]; then
    err "GVHMR_VISIBLE_HAND_REFINE=1 currently requires GVHMR_HAND_BACKEND=hand4wholepp"
    exit 1
fi
if [ "$GVHMR_VISIBLE_HAND_REFINE" = "1" ] && [ "$GVHMR_RECOMPUTE_DIRECT_MANO" != "1" ]; then
    err "GVHMR_VISIBLE_HAND_REFINE=1 requires GVHMR_RECOMPUTE_DIRECT_MANO=1"
    exit 1
fi
case "$GVHMR_VISIBLE_HAND_REFINE_FIT_PARTITION" in
    all|non_tip) ;;
    *) err "GVHMR_VISIBLE_HAND_REFINE_FIT_PARTITION must be all or non_tip, got: $GVHMR_VISIBLE_HAND_REFINE_FIT_PARTITION"; exit 1 ;;
esac

case "$GVHMR_TEMPORAL_FILTER_GLOBAL_ORIENT_FILL_MODE" in
    interpolate|preserve) ;;
    *) err "GVHMR_TEMPORAL_FILTER_GLOBAL_ORIENT_FILL_MODE must be interpolate or preserve, got: $GVHMR_TEMPORAL_FILTER_GLOBAL_ORIENT_FILL_MODE"; exit 1 ;;
esac

case "$GVHMR_FINGER_FILTER_WRIST_MODE" in
    smooth|preserve) ;;
    *) err "GVHMR_FINGER_FILTER_WRIST_MODE must be smooth or preserve, got: $GVHMR_FINGER_FILTER_WRIST_MODE"; exit 1 ;;
esac

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
                    --max_edge_hold "$GVHMR_TEMPORAL_FILTER_MAX_EDGE_HOLD" \
                    --global_orient_fill_mode "$GVHMR_TEMPORAL_FILTER_GLOBAL_ORIENT_FILL_MODE"
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
                    --open_rescue_enabled "$GVHMR_FINGER_FILTER_OPEN_RESCUE_ENABLED" \
                    --open_rescue_weight "$GVHMR_FINGER_FILTER_OPEN_RESCUE_WEIGHT" \
                    --open_rescue_smooth_weight "$GVHMR_FINGER_FILTER_OPEN_RESCUE_SMOOTH_WEIGHT" \
                    --max_joint_angle_delta "$GVHMR_FINGER_FILTER_MAX_JOINT_ANGLE_DELTA" \
                    --max_joint_xyz_delta "$GVHMR_FINGER_FILTER_MAX_JOINT_XYZ_DELTA" \
                    --wrist_smooth_window "$GVHMR_FINGER_FILTER_WRIST_SMOOTH_WINDOW" \
                    --wrist_reliable_smooth_weight "$GVHMR_FINGER_FILTER_WRIST_RELIABLE_SMOOTH_WEIGHT" \
                    --wrist_weak_smooth_weight "$GVHMR_FINGER_FILTER_WRIST_WEAK_SMOOTH_WEIGHT" \
                    --wrist_bad_smooth_weight "$GVHMR_FINGER_FILTER_WRIST_BAD_SMOOTH_WEIGHT" \
                    --max_wrist_angle_delta "$GVHMR_FINGER_FILTER_MAX_WRIST_ANGLE_DELTA" \
                    --wrist_mode "$GVHMR_FINGER_FILTER_WRIST_MODE" \
                    --hand_size_floor_ratio "$GVHMR_FINGER_FILTER_HAND_SIZE_FLOOR_RATIO"
            )
        fi
        MANO_PARAMS="$MANO_FINGER_FIXED"
        MANO_TRACK_LABEL="${MANO_TRACK_LABEL}_finger_fixed"
    fi
fi

if [ "$GVHMR_VISIBLE_HAND_REFINE" = "1" ] && [ -f "${MANO_PARAMS:-}" ]; then
    VITPOSE_WHOLEBODY="$(find_first "$GVHMR_OUT" "vitpose_wholebody.pt")"
    if [ ! -f "${VITPOSE_WHOLEBODY:-}" ]; then
        err "GVHMR_VISIBLE_HAND_REFINE=1 requires vitpose_wholebody.pt"
        exit 1
    fi
    MANO_VISIBLE_REFINED="${WORK}/mano_params_visible_refined.pt"
    MANO_VISIBLE_SUMMARY="${WORK}/mano_params_visible_refined.json"
    if [ "$SKIP_EXISTING" = "1" ] && [ "$GVHMR_FORCE_HAND_PREPROCESS" != "1" ] && \
       [ "$FILTER_CONFIG_OK" = "1" ] && [ -f "$MANO_VISIBLE_REFINED" ] && \
       [ -f "$MANO_VISIBLE_SUMMARY" ]; then
        ok "Visible-hand refined MANO already exists: $MANO_VISIBLE_REFINED"
    else
        log ""
        log "[1a] Refine high-evidence wrists in final render camera space"
        (
            cd "$GVHMR"
            "$PY_GVHMR" -m tools.processor.refine_visible_hand_orientation \
                --mano_params "$MANO_PARAMS" \
                --hmr4d_results "$GVHMR_RESULTS" \
                --vitpose_wholebody "$VITPOSE_WHOLEBODY" \
                --output "$MANO_VISIBLE_REFINED" \
                --summary "$MANO_VISIBLE_SUMMARY" \
                --device "$GVHMR_VISIBLE_HAND_REFINE_DEVICE" \
                --batch_size "$GVHMR_VISIBLE_HAND_REFINE_BATCH_SIZE" \
                --steps "$GVHMR_VISIBLE_HAND_REFINE_STEPS" \
                --lr "$GVHMR_VISIBLE_HAND_REFINE_LR" \
                --prior_weight "$GVHMR_VISIBLE_HAND_REFINE_PRIOR_WEIGHT" \
                --fit_confidence "$GVHMR_VISIBLE_HAND_REFINE_FIT_CONFIDENCE" \
                --fit_min_keypoints "$GVHMR_VISIBLE_HAND_REFINE_FIT_MIN_KEYPOINTS" \
                --fit_partition "$GVHMR_VISIBLE_HAND_REFINE_FIT_PARTITION" \
                --holdout_min_keypoints "$GVHMR_VISIBLE_HAND_REFINE_HOLDOUT_MIN_KEYPOINTS" \
                --holdout_max_relative_regression_px "$GVHMR_VISIBLE_HAND_REFINE_HOLDOUT_MAX_RELATIVE_REGRESSION_PX" \
                --max_delta_degrees "$GVHMR_VISIBLE_HAND_REFINE_MAX_DELTA_DEGREES" \
                --min_relative_improvement "$GVHMR_VISIBLE_HAND_REFINE_MIN_RELATIVE_IMPROVEMENT" \
                --min_relative_improvement_px "$GVHMR_VISIBLE_HAND_REFINE_MIN_RELATIVE_IMPROVEMENT_PX" \
                --max_absolute_regression_px "$GVHMR_VISIBLE_HAND_REFINE_MAX_ABSOLUTE_REGRESSION_PX" \
                --max_anchor_error_px "$GVHMR_VISIBLE_HAND_REFINE_MAX_ANCHOR_ERROR_PX" \
                --max_anchor_error_bbox_ratio "$GVHMR_VISIBLE_HAND_REFINE_MAX_ANCHOR_ERROR_BBOX_RATIO" \
                --evidence_hand_confidence "$GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_HAND_CONFIDENCE" \
                --evidence_min_keypoints "$GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_MIN_KEYPOINTS" \
                --evidence_mean_confidence "$GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_MEAN_CONFIDENCE" \
                --evidence_wrist_confidence "$GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_WRIST_CONFIDENCE" \
                --evidence_min_bbox_diagonal_px "$GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_MIN_BBOX_DIAGONAL_PX" \
                --evidence_min_run "$GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_MIN_RUN"
        )
    fi
    [ -f "$MANO_VISIBLE_REFINED" ] || { err "Visible-hand refinement output not found"; exit 1; }
    [ -f "$MANO_VISIBLE_SUMMARY" ] || { err "Visible-hand refinement summary not found"; exit 1; }
    MANO_PARAMS="$MANO_VISIBLE_REFINED"
    MANO_TRACK_LABEL="${MANO_TRACK_LABEL}_visible_refined"
fi

# Visible-hand refinement is an image-space per-frame correction. It runs
# after the first finger filter and can therefore reintroduce tiny temporal
# oscillations. Smooth its accepted MANO output once more before both the
# GVHMR render and the direct-MANO sidecar are generated.
if [ "$GVHMR_FINAL_FINGER_SMOOTH_AFTER_VISIBLE_REFINE" = "1" ] && \
   [ "$GVHMR_VISIBLE_HAND_REFINE" = "1" ] && \
   [ "$GVHMR_FILTER_MANO_FINGERS" = "1" ] && \
   [ -f "${MANO_PARAMS:-}" ]; then
    VITPOSE_WHOLEBODY="$(find_first "$GVHMR_OUT" "vitpose_wholebody.pt")"
    if [ ! -f "${VITPOSE_WHOLEBODY:-}" ]; then
        err "final MANO smoothing requires vitpose_wholebody.pt"
        exit 1
    fi
    MANO_FINAL_FINGER_FIXED="${WORK}/mano_params_final_finger_smoothed.pt"
    MANO_FINAL_FINGER_SUMMARY="${WORK}/mano_params_final_finger_smoothed.json"
    if [ "$SKIP_EXISTING" = "1" ] && [ "$GVHMR_FORCE_HAND_PREPROCESS" != "1" ] && \
       [ "$FILTER_CONFIG_OK" = "1" ] && [ -f "$MANO_FINAL_FINGER_FIXED" ] && \
       [ -f "$MANO_FINAL_FINGER_SUMMARY" ]; then
        ok "Final finger-smoothed MANO already exists: $MANO_FINAL_FINGER_FIXED"
    else
        log ""
        log "[1a] Final temporal smoothing after visible-hand refinement"
        (
            cd "$GVHMR"
            "$PY_GVHMR" -m tools.processor.filter_mano_fingers \
                --mano_params "$MANO_PARAMS" \
                --vitpose_wholebody "$VITPOSE_WHOLEBODY" \
                --output "$MANO_FINAL_FINGER_FIXED" \
                --summary "$MANO_FINAL_FINGER_SUMMARY" \
                --hand_low_conf_thr "$GVHMR_HAND_KPT_LOW_CONF_THR" \
                --hand_min_keypoints "$GVHMR_HAND_MIN_KEYPOINTS" \
                --max_interp_gap "$GVHMR_FINGER_FILTER_MAX_INTERP_GAP" \
                --max_edge_hold "$GVHMR_FINGER_FILTER_MAX_EDGE_HOLD" \
                --smooth_window "$GVHMR_FINAL_FINGER_SMOOTH_WINDOW" \
                --reliable_smooth_weight "$GVHMR_FINAL_FINGER_RELIABLE_SMOOTH_WEIGHT" \
                --weak_smooth_weight "$GVHMR_FINAL_FINGER_WEAK_SMOOTH_WEIGHT" \
                --bad_smooth_weight "$GVHMR_FINAL_FINGER_BAD_SMOOTH_WEIGHT" \
                --open_rescue_enabled "$GVHMR_FINGER_FILTER_OPEN_RESCUE_ENABLED" \
                --open_rescue_weight "$GVHMR_FINGER_FILTER_OPEN_RESCUE_WEIGHT" \
                --open_rescue_smooth_weight "$GVHMR_FINGER_FILTER_OPEN_RESCUE_SMOOTH_WEIGHT" \
                --max_joint_angle_delta "$GVHMR_FINGER_FILTER_MAX_JOINT_ANGLE_DELTA" \
                --max_joint_xyz_delta "$GVHMR_FINGER_FILTER_MAX_JOINT_XYZ_DELTA" \
                --wrist_smooth_window "$GVHMR_FINGER_FILTER_WRIST_SMOOTH_WINDOW" \
                --wrist_reliable_smooth_weight "$GVHMR_FINGER_FILTER_WRIST_RELIABLE_SMOOTH_WEIGHT" \
                --wrist_weak_smooth_weight "$GVHMR_FINGER_FILTER_WRIST_WEAK_SMOOTH_WEIGHT" \
                --wrist_bad_smooth_weight "$GVHMR_FINGER_FILTER_WRIST_BAD_SMOOTH_WEIGHT" \
                --max_wrist_angle_delta "$GVHMR_FINGER_FILTER_MAX_WRIST_ANGLE_DELTA" \
                --wrist_mode "$GVHMR_FINGER_FILTER_WRIST_MODE" \
                --hand_size_floor_ratio "$GVHMR_FINGER_FILTER_HAND_SIZE_FLOOR_RATIO"
        )
    fi
    [ -f "$MANO_FINAL_FINGER_FIXED" ] || { err "Final finger smoothing output not found"; exit 1; }
    [ -f "$MANO_FINAL_FINGER_SUMMARY" ] || { err "Final finger smoothing summary not found"; exit 1; }
    MANO_PARAMS="$MANO_FINAL_FINGER_FIXED"
    MANO_TRACK_LABEL="${MANO_TRACK_LABEL}_final_finger_smoothed"
fi

if [ "$GVHMR_RECOMPUTE_DIRECT_MANO" = "1" ] && [ -f "${MANO_PARAMS:-}" ]; then
    if [ "$GVHMR_HAND_BACKEND" != "hand4wholepp" ]; then
        err "GVHMR_RECOMPUTE_DIRECT_MANO=1 requires GVHMR_HAND_BACKEND=hand4wholepp"
        exit 1
    fi
    MANO_DIRECT_RECOMPUTED="${WORK}/mano_params_direct_mano_recomputed.pt"
    MANO_DIRECT_SUMMARY="${WORK}/mano_params_direct_mano_recomputed.json"
    if [ "$SKIP_EXISTING" = "1" ] && [ "$GVHMR_FORCE_HAND_PREPROCESS" != "1" ] && \
       [ "$FILTER_CONFIG_OK" = "1" ] && [ -f "$MANO_DIRECT_RECOMPUTED" ] && \
       [ -f "$MANO_DIRECT_SUMMARY" ]; then
        ok "Direct-MANO-recomputed joints already exist: $MANO_DIRECT_RECOMPUTED"
    else
        log ""
        log "[1a] Recompute final Hand4Whole++ joints from final MANO pose"
        (
            cd "$GVHMR"
            "$PY_GVHMR" "$GVHMR/tools/processor/recompute_direct_mano_joints.py" \
                --mano_params "$MANO_PARAMS" \
                --output "$MANO_DIRECT_RECOMPUTED" \
                --summary "$MANO_DIRECT_SUMMARY" \
                --hand4wholepp_root "$GVHMR_HAND4WHOLEPP_ROOT" \
                --batch_size "$GVHMR_RECOMPUTE_DIRECT_MANO_BATCH_SIZE"
        )
    fi
    [ -f "$MANO_DIRECT_RECOMPUTED" ] || { err "Direct-MANO recompute output not found"; exit 1; }
    [ -f "$MANO_DIRECT_SUMMARY" ] || { err "Direct-MANO recompute summary not found"; exit 1; }
    MANO_PARAMS="$MANO_DIRECT_RECOMPUTED"
    MANO_TRACK_LABEL="${MANO_TRACK_LABEL}_direct_mano"
fi

if [ "$GVHMR_SKIP_RENDER" != "1" ] && \
   { [ "$GVHMR_FILTER_MANO_WRIST" = "1" ] || [ "$GVHMR_FILTER_MANO_TEMPORAL" = "1" ] || [ "$GVHMR_FILTER_MANO_FINGERS" = "1" ] || [ "$GVHMR_VISIBLE_HAND_REFINE" = "1" ] || [ "$GVHMR_RECOMPUTE_DIRECT_MANO" = "1" ]; } && \
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
    CONVERT_RAN=1
    echo "$FILTER_FINGERPRINT" > "$FILTER_CONFIG_MARKER"
fi

LOCO_SOURCE="$GVHMR_CONVERTED"

# ---------------------------------------------------------------------------
# Stage 2: Locomotion height optimization
# ---------------------------------------------------------------------------
if [ "$SKIP_EXISTING" = "1" ] && [ "$FORCE_LOCO" != "1" ] && \
   [ "$CONVERT_RAN" != "1" ] && [ -f "$LOCO_NPZ" ]; then
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
        printf '{"status":"degraded","stage":"locomotion","actual":"height_only","reason":"full_checks_filtered_clip"}
' > "${WORK}/stage_status.json"
        "${LOCO_CMD_BASE[@]}" --check_penetration 0 --check_speed 0 --gravity_alignment 0 2>&1 || {
            err "Locomotion retry failed"; exit 1; }
    fi
    LOCO_RAN=1
    ok "Locomotion complete"
fi
LOCO_NPZ="$(find "$LOCO_OUT/optimizer/results_filter" -name "*optimized.npz" 2>/dev/null | head -1)"
[ -f "${LOCO_NPZ:-}" ] || { err "Locomotion output not found"; exit 1; }

# ---------------------------------------------------------------------------
# Stage 3: Temporal smoothing
# ---------------------------------------------------------------------------
if [ "$SKIP_EXISTING" = "1" ] && [ "$FORCE_LOCO" != "1" ] && \
   [ "$FORCE_SMOOTH" != "1" ] && [ "$LOCO_RAN" != "1" ] && \
   [ -f "$SMOOTH_NPZ" ]; then
    ok "Smooth already exists: $SMOOTH_NPZ"
else
    log ""
    log "[3/5] Savitzky-Golay smoothing"
    if run_phc_python "${LEGACY_PIPELINE}/smooth_motion.py" \
        --input "$LOCO_NPZ" --output "$SMOOTH_NPZ" \
        --pose_window "$BODY_SMOOTH_POSE_WINDOW" --trans_window "$BODY_SMOOTH_TRANS_WINDOW" \
        --acc_threshold 10.0 --joint_acc_threshold 300.0 2>&1; then
        ok "Smooth complete: $SMOOTH_NPZ"
    else
        warn "Smoothing failed; using locomotion output"
        cp "$LOCO_NPZ" "$SMOOTH_NPZ"
    fi
    SMOOTH_RAN=1
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

fi  # PIPELINE_POST_ONLY

# ---------------------------------------------------------------------------
# Stage 4: PHC repair
# ---------------------------------------------------------------------------
FINAL_POST_NPZ="$FINAL_PRE_NPZ"
FINAL_SELECTION_REASON="phc_not_run_fallback_smoothed"
if [ "$SKIP_PHC" = "1" ]; then
    warn "Skipping PHC (SKIP_PHC=1)"
    FINAL_SELECTION_REASON="phc_disabled_fallback_smoothed"
    append_phc_attempt "phc_skipped"
elif [ "$SKIP_EXISTING" = "1" ] && [ "$FORCE_PHC" != "1" ] && \
     [ -f "$PHC_TRACKING_AUDIT" ] && ! phc_tracking_pass "$PHC_TRACKING_AUDIT"; then
    warn "Cached PHC rollout was rejected by its recorded-state tracking audit"
    FINAL_SELECTION_REASON="phc_cached_tracking_rejected_fallback_smoothed"
    append_phc_attempt "phc_tracking_rejected"
elif [ "$SKIP_EXISTING" = "1" ] && [ "$FORCE_PHC" != "1" ] && \
     [ "$SMOOTH_RAN" != "1" ] && [ -f "$PHC_SMOOTH_GROUNDED_NPZ" ] && \
     phc_tracking_pass "$PHC_TRACKING_AUDIT"; then
    FINAL_POST_NPZ="$PHC_SMOOTH_GROUNDED_NPZ"
    FINAL_SELECTION_REASON="phc_cached"
    ok "PHC smoothed grounded already exists: $FINAL_POST_NPZ"
    append_phc_attempt "phc_accepted"
elif [ "$SKIP_EXISTING" = "1" ] && [ "$FORCE_PHC" != "1" ] && \
     [ "$SMOOTH_RAN" != "1" ] && [ -f "$PHC_GROUNDED_NPZ" ] && \
     phc_tracking_pass "$PHC_TRACKING_AUDIT"; then
    FINAL_POST_NPZ="$PHC_GROUNDED_NPZ"
    FINAL_SELECTION_REASON="phc_cached"
    ok "PHC grounded already exists: $FINAL_POST_NPZ"
    append_phc_attempt "phc_accepted"
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
        PHC_TRACKING_REJECTED=0
        if [ -n "$REPAIRED" ] && ! phc_tracking_pass "$PHC_TRACKING_AUDIT"; then
            warn "PHC output exists but failed/missed the recorded-state tracking gate"
            REPAIRED=""
            PHC_TRACKING_REJECTED=1
            FINAL_SELECTION_REASON="phc_tracking_gate_failed_fallback_smoothed"
            append_phc_attempt "phc_tracking_rejected"
        fi
        if [ -n "$REPAIRED" ]; then
            FINAL_POST_NPZ="$REPAIRED"
            PHC_RAN=1
            FINAL_SELECTION_REASON="phc_repaired"
            ok "PHC output: $FINAL_POST_NPZ"
            ground_fix_npz "$FINAL_POST_NPZ" "$PHC_GROUNDED_NPZ" "PHC export"
            append_phc_attempt "phc_accepted"
        elif [ "$PHC_TRACKING_REJECTED" = "1" ]; then
            printf '{"status":"degraded","stage":"phc","actual":"smoothed","reason":"phc_tracking_rejected"}
' > "${WORK}/stage_status.json"
        else
            warn "No repaired output found; using pre-PHC NPZ"
            FINAL_SELECTION_REASON="phc_no_output_fallback_smoothed"
            printf '{"status":"degraded","stage":"phc","actual":"smoothed","reason":"phc_no_repaired_output"}
' > "${WORK}/stage_status.json"
            append_phc_attempt "phc_no_output"
        fi
    else
        warn "PHC failed; using pre-PHC NPZ"
        if [ -f "$PHC_TRACKING_AUDIT" ] && ! phc_tracking_pass "$PHC_TRACKING_AUDIT"; then
            FINAL_SELECTION_REASON="phc_tracking_rejected_fallback_smoothed"
            printf '{"status":"degraded","stage":"phc","actual":"smoothed","reason":"phc_tracking_rejected"}
' > "${WORK}/stage_status.json"
            append_phc_attempt "phc_tracking_rejected"
        else
            FINAL_SELECTION_REASON="phc_runtime_failed_fallback_smoothed"
            printf '{"status":"degraded","stage":"phc","actual":"smoothed","reason":"phc_runtime_failed"}
' > "${WORK}/stage_status.json"
            append_phc_attempt "phc_runtime_failed"
        fi
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

# ``001_final.npz`` is a zero-copy, per-run selection point for downstream
# consumers. It prevents stale PHC files from a previous attempt being chosen
# when the current PHC invocation fails; GMR then deterministically receives
# the smoothed track.
publish_final_motion "$FINAL_POST_NPZ" "$FINAL_SELECTION_REASON"

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
