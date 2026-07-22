#!/bin/bash
# ============================================================================
# Full pipeline: dataset_new6 -> GVHMR-hand (WiLoR) -> H1 + Sharpa -> 2x2
# ============================================================================
# Usage:
#   cd <repo-root>
#   bash GVHMR-hand/GVHMR-main/tools/pipeline/run_batch_dataset6_wilor.sh
#
# Output: output_dir/dataset_new6_wilor_gmr_<sharpa|g1|brainco>_aligned/<clip>/
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GVHMR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PIPELINE_ROOT="${PIPELINE_ROOT:-$(cd "${GVHMR}/../.." && pwd)}"

export PIPELINE_ROOT GVHMR
export GMR_HAND_MODEL="${GMR_HAND_MODEL:-sharpa}"
source "${PIPELINE_ROOT}/GMR-master/configure_hand_model.sh"

# ---- Independent backend/output -------------------------------------------
export GVHMR_HAND_BACKEND="wilor"
source "${SCRIPT_DIR}/configure_hand_constraints.sh"
export OUTPUT_BASE="${OUTPUT_BASE:-${PIPELINE_ROOT}/output_dir/dataset_new6_wilor_gmr_${GMR_HAND_MODEL}_aligned}"
export GMR_ROBOT="${GMR_ROBOT:-unitree_g1}"
export GMR_AUTO_HAND_NPZ=1
export GMR_HAND_INVALID_MODE="${GMR_HAND_INVALID_MODE:-interp}"

# ---- WiLoR assets and quality mode ----------------------------------------
export GVHMR_WILOR_ROOT="${GVHMR_WILOR_ROOT:-${GVHMR}/third-party/WiLoR}"
export GVHMR_WILOR_CHECKPOINT="${GVHMR_WILOR_CHECKPOINT:-${GVHMR_WILOR_ROOT}/pretrained_models/wilor_final.ckpt}"
export GVHMR_WILOR_CONFIG="${GVHMR_WILOR_CONFIG:-${GVHMR_WILOR_ROOT}/pretrained_models/model_config.yaml}"
# Keep full precision for the first accuracy comparison. Set to 1 later if the
# measured difference is negligible and throughput matters more.
export GVHMR_WILOR_FAST="${GVHMR_WILOR_FAST:-0}"

# WiLoR's official demo uses bbox rescale 2.0.  HaMeR's old 3.0/3.5/4.0
# candidates make the hand too small in WiLoR's 256x256 crop, so use a narrow
# scale search around the model's native setting.
export GVHMR_HAMER_BBOX_RESCALE="${GVHMR_HAMER_BBOX_RESCALE:-2.0}"
export GVHMR_HAMER_BBOX_RESCALE_CANDIDATES="${GVHMR_HAMER_BBOX_RESCALE_CANDIDATES:-1.8,2.0,2.2}"
export GVHMR_HAMER_CANDIDATE_SWITCH_PENALTY="${GVHMR_HAMER_CANDIDATE_SWITCH_PENALTY:-8.0}"
export GVHMR_HAMER_BATCH_SIZE="${GVHMR_HAMER_BATCH_SIZE:-4}"
export GVHMR_HAMER_REFINE_STEPS="${GVHMR_HAMER_REFINE_STEPS:-4}"
export GVHMR_HAMER_REFINE_LR="${GVHMR_HAMER_REFINE_LR:-0.03}"
export GVHMR_HAMER_REFINE_CONF_THR="${GVHMR_HAMER_REFINE_CONF_THR:-0.45}"
export GVHMR_HAMER_REFINE_MIN_KEYPOINTS="${GVHMR_HAMER_REFINE_MIN_KEYPOINTS:-6}"
export GVHMR_HAMER_REFINE_POSE_PRIOR="${GVHMR_HAMER_REFINE_POSE_PRIOR:-0.01}"
export GVHMR_HAMER_REFINE_GLOBAL_PRIOR="${GVHMR_HAMER_REFINE_GLOBAL_PRIOR:-0.01}"

# WiLoR is loaded in-process like HaMeR.  End that process before HMR2 to keep
# host RAM/CUDA peaks separated.
export GVHMR_BATCH_SIZE="${GVHMR_BATCH_SIZE:-4}"
export GVHMR_ISOLATE_HAND_PREPROCESS="${GVHMR_ISOLATE_HAND_PREPROCESS:-1}"
export GVHMR_LOW_MEMORY="${GVHMR_LOW_MEMORY:-1}"

# ---- Shared hand evidence and temporal cleaning ---------------------------
export GVHMR_VITPOSE_IMG_DS="${GVHMR_VITPOSE_IMG_DS:-1.0}"
export GVHMR_HAND_KPT_CONF_THR="${GVHMR_HAND_KPT_CONF_THR:-0.35}"
export GVHMR_HAND_KPT_LOW_CONF_THR="${GVHMR_HAND_KPT_LOW_CONF_THR:-0.2}"
export GVHMR_HAND_KPT_HI_MIN_KEYPOINTS="${GVHMR_HAND_KPT_HI_MIN_KEYPOINTS:-6}"
export GVHMR_HAND_MIN_KEYPOINTS="${GVHMR_HAND_MIN_KEYPOINTS:-3}"
export GVHMR_FILTER_MANO_WRIST="${GVHMR_FILTER_MANO_WRIST:-0}"
export GVHMR_FILTER_MANO_TEMPORAL="${GVHMR_FILTER_MANO_TEMPORAL:-1}"
export GVHMR_FILTER_MANO_FINGERS="${GVHMR_FILTER_MANO_FINGERS:-1}"
export GVHMR_TEMPORAL_FILTER_MAX_INTERP_GAP="${GVHMR_TEMPORAL_FILTER_MAX_INTERP_GAP:-30}"
export GVHMR_TEMPORAL_FILTER_MAX_EDGE_HOLD="${GVHMR_TEMPORAL_FILTER_MAX_EDGE_HOLD:-10}"
export GVHMR_FINGER_FILTER_SMOOTH_WINDOW="${GVHMR_FINGER_FILTER_SMOOTH_WINDOW:-15}"
export GVHMR_FINGER_FILTER_RELIABLE_SMOOTH_WEIGHT="${GVHMR_FINGER_FILTER_RELIABLE_SMOOTH_WEIGHT:-0.25}"
export GVHMR_FINGER_FILTER_WEAK_SMOOTH_WEIGHT="${GVHMR_FINGER_FILTER_WEAK_SMOOTH_WEIGHT:-0.85}"
export GVHMR_FINGER_FILTER_MAX_INTERP_GAP="${GVHMR_FINGER_FILTER_MAX_INTERP_GAP:-30}"
export GVHMR_FINGER_FILTER_MAX_EDGE_HOLD="${GVHMR_FINGER_FILTER_MAX_EDGE_HOLD:-10}"
export GVHMR_FINGER_FILTER_MAX_JOINT_ANGLE_DELTA="${GVHMR_FINGER_FILTER_MAX_JOINT_ANGLE_DELTA:-0.22}"
export GVHMR_FINGER_FILTER_MAX_JOINT_XYZ_DELTA="${GVHMR_FINGER_FILTER_MAX_JOINT_XYZ_DELTA:-0.035}"
export GVHMR_FINGER_FILTER_WRIST_SMOOTH_WINDOW="${GVHMR_FINGER_FILTER_WRIST_SMOOTH_WINDOW:-11}"
export GVHMR_FINGER_FILTER_WRIST_RELIABLE_SMOOTH_WEIGHT="${GVHMR_FINGER_FILTER_WRIST_RELIABLE_SMOOTH_WEIGHT:-0.25}"
export GVHMR_FINGER_FILTER_MAX_WRIST_ANGLE_DELTA="${GVHMR_FINGER_FILTER_MAX_WRIST_ANGLE_DELTA:-0.22}"
export GVHMR_FINGER_FILTER_HAND_SIZE_FLOOR_RATIO="${GVHMR_FINGER_FILTER_HAND_SIZE_FLOOR_RATIO:-0.94}"
export GVHMR_FINGER_FILTER_OPEN_RESCUE_WEIGHT="${GVHMR_FINGER_FILTER_OPEN_RESCUE_WEIGHT:-0.25}"
# The temporal/finger filters above already define the track rendered by
# GVHMR. Do not run a second interpolation pass only for the GMR sidecar.
export GVHMR_HAND_REFINE_MODE="${GVHMR_HAND_REFINE_MODE:-raw}"
export GVHMR_HAND_REPROJ_ERROR_RATIO_THR="${GVHMR_HAND_REPROJ_ERROR_RATIO_THR:-0.45}"
export GVHMR_HAND_WRIST_OFFSET_MODE="${GVHMR_HAND_WRIST_OFFSET_MODE:-temporal}"
export GVHMR_HAND_WRIST_OFFSET_SMOOTH_WINDOW="${GVHMR_HAND_WRIST_OFFSET_SMOOTH_WINDOW:-15}"

# ---- H1 body/wrist + Sharpa visual hand -----------------------------------
export GMR_HAND_WRIST_ORIENTATION_MODE="${GMR_HAND_WRIST_ORIENTATION_MODE:-diagnostic}"
export GMR_PALM_ROLL_MODE="${GMR_PALM_ROLL_MODE:-auto}"
export GMR_PALM_ROLL_SOURCE="${GMR_PALM_ROLL_SOURCE:-smpl}"
export GMR_PALM_ROLL_GAIN="${GMR_PALM_ROLL_GAIN:-0.75}"
export GMR_LEFT_PALM_ROLL_SIGN="${GMR_LEFT_PALM_ROLL_SIGN:-1.0}"
export GMR_RIGHT_PALM_ROLL_SIGN="${GMR_RIGHT_PALM_ROLL_SIGN:-1.0}"
export GMR_PALM_ROLL_SMOOTH_WINDOW="${GMR_PALM_ROLL_SMOOTH_WINDOW:-15}"
export GMR_PALM_ROLL_MAX_DELTA="${GMR_PALM_ROLL_MAX_DELTA:-0.12}"
export GMR_PALM_ROLL_MAX_ABS="${GMR_PALM_ROLL_MAX_ABS:-1.35}"
export GMR_PALM_ROLL_BRANCH_MODE="${GMR_PALM_ROLL_BRANCH_MODE:-off}"
export GMR_HAND_RETARGET_MODE="${GMR_HAND_RETARGET_MODE:-off}"
export GMR_SHARPA_HANDS="${GMR_SHARPA_HANDS:-1}"
export GMR_SHARPA_AUTO_RETARGET="${GMR_SHARPA_AUTO_RETARGET:-1}"
export GMR_SHARPA_HAND_NPZ_NAME="${GMR_SHARPA_HAND_NPZ_NAME:-001_sharpa_chain_hands.npz}"
export GMR_SHARPA_LEFT_MOUNT_QUAT="${GMR_SHARPA_LEFT_MOUNT_QUAT:-0.5,-0.5,0.5,-0.5}"
export GMR_SHARPA_RIGHT_MOUNT_QUAT="${GMR_SHARPA_RIGHT_MOUNT_QUAT:-0.5,0.5,0.5,0.5}"

export GVHMR_DIAGNOSE_HAND="${GVHMR_DIAGNOSE_HAND:-1}"
export GVHMR_DIAGNOSE_HAND_WIDTH="${GVHMR_DIAGNOSE_HAND_WIDTH:-960}"
export RENDER_COMPARISON="${RENDER_COMPARISON:-0}"

echo "============================================================"
echo "  Pipeline: WiLoR hand estimation"
echo "============================================================"
echo "Output:    $OUTPUT_BASE"
echo "Backend:   WiLoR full precision=$((1 - GVHMR_WILOR_FAST))"
echo "Crop:      candidates=$GVHMR_HAMER_BBOX_RESCALE_CANDIDATES refine=$GVHMR_HAMER_REFINE_STEPS"
echo "GMR hand:  $GMR_EMBODIMENT_LABEL (selection=$GMR_HAND_MODEL, body_asset=$GMR_ROBOT)"
echo "Memory:    isolated_wilor=$GVHMR_ISOLATE_HAND_PREPROCESS low_memory=$GVHMR_LOW_MEMORY"
echo "Filters:   wrist=$GVHMR_FILTER_MANO_WRIST temporal=$GVHMR_FILTER_MANO_TEMPORAL fingers=$GVHMR_FILTER_MANO_FINGERS"
echo "Profile:   $GVHMR_HAND_CONSTRAINT_PROFILE"
echo ""

bash "${SCRIPT_DIR}/run_batch_dataset6.sh"
