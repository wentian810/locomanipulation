#!/bin/bash
# ============================================================================
# Full pipeline: dataset_new6 -> GVHMR-hand (Hand4Whole++ backend) -> GMR -> 2x2
# ============================================================================
# Usage:
#   cd <repo-root>
#   bash GVHMR-hand/GVHMR-main/tools/pipeline/run_batch_dataset6_hand4wholepp.sh
#
# Output: output_dir/dataset_new6_hand4wholepp_directmano_gmr_<hand>_aligned/<clip>/
#   ├── gvhmr_out/               GVHMR body + hand (Hand4Whole++) results
#   ├── hamer_diagnostics/        Hand diagnostic MP4 + JSON
#   ├── locomotion/               Height optimization
#   ├── phc_renderings/           Isaac Gym camera videos
#   ├── 001_smplx_hands.npz       Hand sidecar for GMR
#   ├── robot_motion.pkl          GMR robot motion
#   ├── 001_sharpa_chain_hands.npz  Sharpa 22-DoF hand trajectory
#   ├── unitree_h1_with_hand_sharpa_gvhmr.mp4  GMR + Sharpa render
#   └── composite_2x2.mp4         2x2 comparison video
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GVHMR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PIPELINE_ROOT="${PIPELINE_ROOT:-$(cd "${GVHMR}/../.." && pwd)}"

export PIPELINE_ROOT GVHMR
export GMR_HAND_MODEL="${GMR_HAND_MODEL:-sharpa}"
source "${PIPELINE_ROOT}/GMR-master/configure_hand_model.sh"

# ---- Robot ----
export GMR_ROBOT="${GMR_ROBOT:-unitree_h1_with_hand}"
export GMR_AUTO_HAND_NPZ=1
export GMR_HAND_INVALID_MODE="${GMR_HAND_INVALID_MODE:-interp}"

# ---- Hand backend ----
export GVHMR_HAND_BACKEND="hand4wholepp"
source "${SCRIPT_DIR}/configure_hand_constraints.sh"

# ---- Output paths (single directory, GMR output inside GVHMR output) ----
export OUTPUT_BASE="${OUTPUT_BASE:-${PIPELINE_ROOT}/output_dir/dataset_new6_hand4wholepp_directmano_gmr_${GMR_HAND_MODEL}_aligned}"
# GMR_OUT_ROOT defaults to OUTPUT_BASE in run_batch_dataset6.sh

# ---- GVHMR inference ----
export GVHMR_BATCH_SIZE="${GVHMR_BATCH_SIZE:-4}"
# Hand4Whole++ loads a 2.4 GiB WiLoR checkpoint plus a 3.0 GiB whole-body
# checkpoint.  Keep that stage in an expendable process and return all of its
# host/CUDA allocations before HMR2 starts.  This is the safe default for the
# current 16 GiB workstation.
export GVHMR_ISOLATE_HAND_PREPROCESS="${GVHMR_ISOLATE_HAND_PREPROCESS:-1}"
export GVHMR_LOW_MEMORY="${GVHMR_LOW_MEMORY:-1}"

# ---- Hand4Whole++ speed tuning ----
# Hand4Whole++ is a single-pass model: no multi-candidate bboxes, no refinement
# steps.  It is inherently faster than HaMeR.  The main speed knobs are:
#
#   GVHMR_VITPOSE_IMG_DS   – ViTPose input downsampling (shared across backends).
#                            1.0 = full res, 0.5 = half res (~2× faster kpt extraction).
#   GVHMR_HAND4WHOLEPP_BATCH_SIZE – real model batch size; 2 is the stable
#                            default for this 16 GiB RAM / RTX 4090 workstation
#                            and automatically falls back after CUDA OOM.
#   GVHMR_HAND4WHOLEPP_YOLO_MODEL – fallback only; normal pipeline reuses GVHMR
#                            person tracks and does not run YOLO.
#
export GVHMR_HAND4WHOLEPP_ROOT="${GVHMR_HAND4WHOLEPP_ROOT:-${PIPELINE_ROOT}/GVHMR-hand/GVHMR-main/third-party/Hand4Whole-plus-plus_RELEASE}"
export GVHMR_HAND4WHOLEPP_SNAPSHOT="${GVHMR_HAND4WHOLEPP_SNAPSHOT:-${GVHMR_HAND4WHOLEPP_ROOT}/demo/snapshot_6.pth}"
export GVHMR_HAND4WHOLEPP_PYTHON="${GVHMR_HAND4WHOLEPP_PYTHON:-}"
export GVHMR_HAND4WHOLEPP_BATCH_SIZE="${GVHMR_HAND4WHOLEPP_BATCH_SIZE:-2}"
export GVHMR_HAND4WHOLEPP_YOLO_MODEL="${GVHMR_HAND4WHOLEPP_YOLO_MODEL:-yolo11n.pt}"
# Keep the final hand pose and the 21 joints on the same WiLoR/MANO skeleton.
# The legacy fused_smplx joints can disagree with the WiLoR pose consumed by
# GVHMR and were a source of GVHMR-versus-Sharpa visual mismatch.
export GVHMR_HAND4WHOLEPP_JOINT_SOURCE="${GVHMR_HAND4WHOLEPP_JOINT_SOURCE:-direct_mano}"

# ---- ViTPose (shared across all backends) ----
export GVHMR_VITPOSE_IMG_DS="${GVHMR_VITPOSE_IMG_DS:-1.0}"
export GVHMR_HAND_KPT_CONF_THR="${GVHMR_HAND_KPT_CONF_THR:-0.35}"
export GVHMR_HAND_KPT_LOW_CONF_THR="${GVHMR_HAND_KPT_LOW_CONF_THR:-0.2}"
export GVHMR_HAND_KPT_HI_MIN_KEYPOINTS="${GVHMR_HAND_KPT_HI_MIN_KEYPOINTS:-6}"
export GVHMR_HAND_MIN_KEYPOINTS="${GVHMR_HAND_MIN_KEYPOINTS:-3}"

# ---- Hand filters ----
# Keep the body-candidate wrist selector disabled for the same reason as HaMeR;
# use shared temporal rejection and quaternion smoothing for a fair comparison.
export GVHMR_FILTER_MANO_WRIST="${GVHMR_FILTER_MANO_WRIST:-0}"
export GVHMR_FILTER_MANO_TEMPORAL="${GVHMR_FILTER_MANO_TEMPORAL:-1}"
export GVHMR_FILTER_MANO_FINGERS="${GVHMR_FILTER_MANO_FINGERS:-1}"
export GVHMR_TEMPORAL_FILTER_MAX_INTERP_GAP="${GVHMR_TEMPORAL_FILTER_MAX_INTERP_GAP:-30}"
export GVHMR_TEMPORAL_FILTER_MAX_EDGE_HOLD="${GVHMR_TEMPORAL_FILTER_MAX_EDGE_HOLD:-10}"
export GVHMR_FINGER_FILTER_SMOOTH_WINDOW="${GVHMR_FINGER_FILTER_SMOOTH_WINDOW:-15}"
export GVHMR_FINGER_FILTER_RELIABLE_SMOOTH_WEIGHT="${GVHMR_FINGER_FILTER_RELIABLE_SMOOTH_WEIGHT:-0.30}"
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

# ---- H1 wrist/hand input split ----
# The H1 arm IK consumes the stable GVHMR/SMPL shoulder, elbow, and wrist
# frame.  Hand4Whole++ remains responsible for the 21-joint hand geometry,
# but its crop-camera global wrist orientation is not mixed into the
# one-DoF H1 hand joint: that was the source of the delayed ~90-degree drift.
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

# ---- Target-space robot hand retargeting ----
# The H1 built-in 12-hinge hand cannot preserve the full MANO shape and its
# legacy axis-component mapping clips many joints at zero.  Keep the stable H1
# body/wrist, replace only the visual hand with a 22-DoF Sharpa chain solved
# from morphology-normalized PIP/DIP/tip targets.
export GMR_HAND_RETARGET_MODE="${GMR_HAND_RETARGET_MODE:-off}"
export GMR_SHARPA_HANDS="${GMR_SHARPA_HANDS:-1}"
export GMR_SHARPA_AUTO_RETARGET="${GMR_SHARPA_AUTO_RETARGET:-1}"
export GMR_SHARPA_HAND_NPZ_NAME="${GMR_SHARPA_HAND_NPZ_NAME:-001_sharpa_chain_hands.npz}"
export GMR_SHARPA_LEFT_MOUNT_QUAT="${GMR_SHARPA_LEFT_MOUNT_QUAT:-0.5,-0.5,0.5,-0.5}"
export GMR_SHARPA_RIGHT_MOUNT_QUAT="${GMR_SHARPA_RIGHT_MOUNT_QUAT:-0.5,0.5,0.5,0.5}"
export GMR_SHARPA_SCALE="${GMR_SHARPA_SCALE:-1.0}"
export GMR_SHARPA_STEPS="${GMR_SHARPA_STEPS:-4}"
export GMR_SHARPA_INIT_STEPS="${GMR_SHARPA_INIT_STEPS:-50}"
export GMR_SHARPA_WRIST_POS_COST="${GMR_SHARPA_WRIST_POS_COST:-0.3}"
export GMR_SHARPA_WRIST_ORI_COST="${GMR_SHARPA_WRIST_ORI_COST:-0.2}"
export GMR_SHARPA_FINGER_POS_COST="${GMR_SHARPA_FINGER_POS_COST:-5.0}"
export GMR_SHARPA_JOINT_POS_COST="${GMR_SHARPA_JOINT_POS_COST:-3.0}"
export GMR_SHARPA_POSTURE_COST="${GMR_SHARPA_POSTURE_COST:-0.01}"
export GMR_SHARPA_TEMPORAL_COST="${GMR_SHARPA_TEMPORAL_COST:-0.03}"
export GMR_SHARPA_LOW_CONF_TEMPORAL_GAIN="${GMR_SHARPA_LOW_CONF_TEMPORAL_GAIN:-3.0}"
export GMR_SHARPA_SMOOTH_WINDOW="${GMR_SHARPA_SMOOTH_WINDOW:-5}"
export GMR_SHARPA_MAX_DELTA="${GMR_SHARPA_MAX_DELTA:-0.12}"
export GMR_SHARPA_MAX_ACCEL="${GMR_SHARPA_MAX_ACCEL:-0.10}"
export GMR_SHARPA_REPROJ_GOOD_PX="${GMR_SHARPA_REPROJ_GOOD_PX:-30.0}"
export GMR_SHARPA_REPROJ_BAD_PX="${GMR_SHARPA_REPROJ_BAD_PX:-75.0}"
export GMR_SHARPA_MAX_PIP_BEND_DEG="${GMR_SHARPA_MAX_PIP_BEND_DEG:-125.0}"
export GMR_SHARPA_MAX_DIP_BEND_DEG="${GMR_SHARPA_MAX_DIP_BEND_DEG:-105.0}"
export GMR_SHARPA_MAX_SOURCE_BONE_DELTA_DEG="${GMR_SHARPA_MAX_SOURCE_BONE_DELTA_DEG:-45.0}"
export GMR_SHARPA_ANATOMIC_REPAIR_MAX_GAP="${GMR_SHARPA_ANATOMIC_REPAIR_MAX_GAP:-15}"

# ---- Diagnostics ----
export GVHMR_DIAGNOSE_HAND="${GVHMR_DIAGNOSE_HAND:-1}"
export GVHMR_DIAGNOSE_HAND_WIDTH="${GVHMR_DIAGNOSE_HAND_WIDTH:-960}"

# ---- Misc ----
# Keep the sidecar bit-for-bit on the same filtered MANO trajectory used by
# the GVHMR render; confidence still changes IK weights downstream.
export GVHMR_HAND_REFINE_MODE="${GVHMR_HAND_REFINE_MODE:-raw}"
export GVHMR_HAND_REPROJ_ERROR_RATIO_THR="${GVHMR_HAND_REPROJ_ERROR_RATIO_THR:-0.45}"
export GVHMR_HAND_WRIST_OFFSET_MODE="${GVHMR_HAND_WRIST_OFFSET_MODE:-temporal}"
export GVHMR_HAND_WRIST_OFFSET_SMOOTH_WINDOW="${GVHMR_HAND_WRIST_OFFSET_SMOOTH_WINDOW:-15}"
export RENDER_COMPARISON="${RENDER_COMPARISON:-0}"

echo "============================================================"
echo "  Pipeline: Hand4Whole++ hand estimation"
echo "============================================================"
echo "Output:    $OUTPUT_BASE"
echo "Backend:   $GVHMR_HAND_BACKEND (single-pass, no multi-candidate overhead)"
echo "GMR hand:  $GMR_EMBODIMENT_LABEL (selection=$GMR_HAND_MODEL, body_asset=$GMR_ROBOT)"
echo "Speed:     vitpose_ds=$GVHMR_VITPOSE_IMG_DS batch=$GVHMR_HAND4WHOLEPP_BATCH_SIZE yolo_fallback=$GVHMR_HAND4WHOLEPP_YOLO_MODEL"
echo "Memory:    isolated_hand4wholepp=$GVHMR_ISOLATE_HAND_PREPROCESS low_memory=$GVHMR_LOW_MEMORY checkpoint=mmap"
echo "Geometry:  pose/joints source=$GVHMR_HAND4WHOLEPP_JOINT_SOURCE"
echo "Filters:   wrist=$GVHMR_FILTER_MANO_WRIST temporal=$GVHMR_FILTER_MANO_TEMPORAL fingers=$GVHMR_FILTER_MANO_FINGERS"
echo "Profile:   $GVHMR_HAND_CONSTRAINT_PROFILE"
echo "Diagnose:  $GVHMR_DIAGNOSE_HAND"
echo ""

bash "${SCRIPT_DIR}/run_batch_dataset6.sh"
