#!/bin/bash
# ============================================================================
# Full pipeline: dataset_new6 -> GVHMR-hand (HaMeR backend) -> GMR -> 2x2
# ============================================================================
# Usage:
#   cd <repo-root>
#   bash GVHMR-hand/GVHMR-main/tools/pipeline/run_batch_dataset6_hamer.sh
#
# Output: output_dir/dataset_new6_hamer_gmr_<sharpa|g1|brainco>_aligned/<clip>/
#   ├── gvhmr_out/               GVHMR body + hand (HaMeR) results
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
export GVHMR_HAND_BACKEND="hamer"
source "${SCRIPT_DIR}/configure_hand_constraints.sh"

# ---- Output paths (single directory, GMR output inside GVHMR output) ----
export OUTPUT_BASE="${OUTPUT_BASE:-${PIPELINE_ROOT}/output_dir/dataset_new6_hamer_gmr_${GMR_HAND_MODEL}_aligned}"
# GMR_OUT_ROOT defaults to OUTPUT_BASE in run_batch_dataset6.sh

# ---- GVHMR inference ----
export GVHMR_BATCH_SIZE="${GVHMR_BATCH_SIZE:-4}"
# HaMeR and HMR2 are intentionally run in separate OS processes.  This avoids
# carrying HaMeR's decoded-frame buffer, CPU allocator and CUDA context into
# the HMR2 checkpoint load (the transition that previously froze the host).
export GVHMR_ISOLATE_HAND_PREPROCESS="${GVHMR_ISOLATE_HAND_PREPROCESS:-1}"
export GVHMR_LOW_MEMORY="${GVHMR_LOW_MEMORY:-1}"

# ---- HaMeR precision/speed tuning ----
# 3 bbox rescale candidates (3.0/3.5/4.0) → Viterbi selects best per frame.
# This is the primary accuracy knob — keep 3 candidates for hand coverage.
export GVHMR_HAMER_BBOX_RESCALE="${GVHMR_HAMER_BBOX_RESCALE:-2.6}"
export GVHMR_HAMER_BBOX_RESCALE_CANDIDATES="${GVHMR_HAMER_BBOX_RESCALE_CANDIDATES:-3.0,3.5,4.0}"
export GVHMR_HAMER_CANDIDATE_SWITCH_PENALTY="${GVHMR_HAMER_CANDIDATE_SWITCH_PENALTY:-8.0}"

# Batch size: 4 is a safe speedup (2→4 ≈ 2× faster); GPU mem permitting go to 8.
export GVHMR_HAMER_BATCH_SIZE="${GVHMR_HAMER_BATCH_SIZE:-4}"

# Refinement: 4 steps retains >95% of 8-step quality while cutting ~30% time.
# The refinement is a test-time 2D-keypoint optimisation; diminishing returns
# beyond 4–6 steps for most frames.
export GVHMR_HAMER_REFINE_STEPS="${GVHMR_HAMER_REFINE_STEPS:-4}"
export GVHMR_HAMER_REFINE_LR="${GVHMR_HAMER_REFINE_LR:-0.03}"
export GVHMR_HAMER_REFINE_CONF_THR="${GVHMR_HAMER_REFINE_CONF_THR:-0.45}"
export GVHMR_HAMER_REFINE_MIN_KEYPOINTS="${GVHMR_HAMER_REFINE_MIN_KEYPOINTS:-6}"
export GVHMR_HAMER_REFINE_POSE_PRIOR="${GVHMR_HAMER_REFINE_POSE_PRIOR:-0.005}"
export GVHMR_HAMER_REFINE_GLOBAL_PRIOR="${GVHMR_HAMER_REFINE_GLOBAL_PRIOR:-0.01}"

# ---- Hand filters ----
# The body-candidate wrist filter remains off: switching between MANO/body/180°
# branches can itself cause flicker. Temporal rejection plus quaternion wrist
# smoothing in the finger filter is backend-agnostic and safer.
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
# Keep this identical to the Hand4Whole++ wrapper so the GMR comparison only
# differs in finger estimation.  The stable GVHMR/SMPL wrist drives the H1
# one-DoF hand roll; HaMeR drives the articulated finger joints.
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
# Keep this identical to Hand4Whole++: backend differences should affect only
# the source 21-joint hand track, not the robot hand embodiment or IK costs.
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
# The GMR sidecar must be the exact filtered track shown in GVHMR, rather than
# a second independently interpolated version of it.
export GVHMR_HAND_REFINE_MODE="${GVHMR_HAND_REFINE_MODE:-raw}"
export GVHMR_HAND_REPROJ_ERROR_RATIO_THR="${GVHMR_HAND_REPROJ_ERROR_RATIO_THR:-0.45}"
export GVHMR_HAND_WRIST_OFFSET_MODE="${GVHMR_HAND_WRIST_OFFSET_MODE:-auto}"
export RENDER_COMPARISON="${RENDER_COMPARISON:-0}"

echo "============================================================"
echo "  Pipeline: HAMER hand estimation"
echo "============================================================"
echo "Output:    $OUTPUT_BASE"
echo "Backend:   $GVHMR_HAND_BACKEND"
echo "GMR hand:  $GMR_EMBODIMENT_LABEL (selection=$GMR_HAND_MODEL, body_asset=$GMR_ROBOT)"
echo "Memory:    isolated_hamer=$GVHMR_ISOLATE_HAND_PREPROCESS low_memory=$GVHMR_LOW_MEMORY"
echo "Precision: candidates=${GVHMR_HAMER_BBOX_RESCALE_CANDIDATES} refine=${GVHMR_HAMER_REFINE_STEPS} batch=${GVHMR_HAMER_BATCH_SIZE}"
echo "Filters:   wrist=$GVHMR_FILTER_MANO_WRIST temporal=$GVHMR_FILTER_MANO_TEMPORAL fingers=$GVHMR_FILTER_MANO_FINGERS"
echo "Profile:   $GVHMR_HAND_CONSTRAINT_PROFILE"
echo "Diagnose:  $GVHMR_DIAGNOSE_HAND"
echo ""

bash "${SCRIPT_DIR}/run_batch_dataset6.sh"
