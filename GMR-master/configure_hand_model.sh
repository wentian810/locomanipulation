#!/usr/bin/env bash
# Select one complete GMR hand embodiment.
#
# Source this before pipeline_defaults.sh. Supported values:
#   GMR_HAND_MODEL=sharpa     -> Unitree G1 body + external Sharpa Wave hands
#   GMR_HAND_MODEL=sharpa_g1  -> Alias for the G1 + Sharpa default
#   GMR_HAND_MODEL=sharpa_h1  -> Legacy Unitree H1 body + external Sharpa hands
#   GMR_HAND_MODEL=sharpa_h1_2 -> Unitree H1-2 3-DoF wrists + Sharpa hands
#   GMR_HAND_MODEL=g1      -> Unitree G1 body + native Dex3-1 three-finger hands
#   GMR_HAND_MODEL=brainco -> Unitree G1 body + external BrainCo Revo2 hands

if [ -z "${PIPELINE_ROOT:-}" ]; then
    echo "configure_hand_model.sh requires PIPELINE_ROOT" >&2
    return 1 2>/dev/null || exit 1
fi

GMR_HAND_MODEL="${GMR_HAND_MODEL:-sharpa}"
case "$GMR_HAND_MODEL" in
    sharpa_h1)
        GMR_ROBOT="unitree_h1_with_hand"
        GMR_HAND_RETARGET_MODE="off"
        GMR_SHARPA_HANDS="1"
        GMR_SHARPA_AUTO_RETARGET="${GMR_SHARPA_AUTO_RETARGET:-1}"
        GMR_SHARPA_HAND_NPZ_NAME="${GMR_SHARPA_HAND_NPZ_NAME:-001_sharpa_chain_hands.npz}"
        GMR_BRAINCO_HANDS="0"
        GMR_BRAINCO_AUTO_RETARGET="0"
        GMR_EMBODIMENT_LABEL="Unitree H1 body + Sharpa Wave 22-DoF hands"
        ;;
    sharpa|sharpa_g1)
        GMR_ROBOT="unitree_g1"
        GMR_HAND_RETARGET_MODE="off"
        GMR_SHARPA_HANDS="1"
        GMR_SHARPA_AUTO_RETARGET="${GMR_SHARPA_AUTO_RETARGET:-1}"
        GMR_SHARPA_HAND_NPZ_NAME="${GMR_SHARPA_HAND_NPZ_NAME:-001_sharpa_chain_hands.npz}"
        GMR_BRAINCO_HANDS="0"
        GMR_BRAINCO_AUTO_RETARGET="0"
        # Keep the G1 body retargeting on the stable SMPL/GVHMR wrist frame.
        # The Hand4Whole++/MANO wrist frame is excellent for Sharpa's external
        # finger-chain IK, but forcing it into the G1 body IK overconstrains
        # the wrists and pulls the upper body into visually drifting poses.
        GMR_HAND_WRIST_ORIENTATION_MODE="${GMR_HAND_WRIST_ORIENTATION_MODE:-diagnostic}"
        GMR_FORCE_HAND_WRIST_ORIENTATION_OVERRIDE="${GMR_FORCE_HAND_WRIST_ORIENTATION_OVERRIDE:-0}"
        GMR_RELAX_ORIENTATION_BODIES=""
        GMR_WRIST_PITCH_YAW_STABILIZE="soft_limit"
        GMR_PALM_ROLL_MODE="off"
        # PHC grounds the human trajectory, but G1 has different leg/sole
        # geometry. Infer left/right support from G1 sole height and speed;
        # only those frames are corrected, while flight is retained.
        GMR_HEIGHT_ADJUST_MODE="${GMR_HEIGHT_ADJUST_MODE:-support_aware_foot_geom}"
        GMR_SUPPORT_CONTACT_HEIGHT="${GMR_SUPPORT_CONTACT_HEIGHT:-0.08}"
        GMR_SUPPORT_MAX_VERTICAL_SPEED="${GMR_SUPPORT_MAX_VERTICAL_SPEED:-1.20}"
        GMR_SUPPORT_MIN_CONTACT_RUN="${GMR_SUPPORT_MIN_CONTACT_RUN:-3}"
        GMR_SUPPORT_MAX_CONTACT_GAP="${GMR_SUPPORT_MAX_CONTACT_GAP:-1}"
        GMR_SUPPORT_ROOT_STEP_LIMIT="${GMR_SUPPORT_ROOT_STEP_LIMIT:-0.03}"
        GMR_SHARPA_MOUNT_POS="${GMR_SHARPA_MOUNT_POS:-0.0415,0,0}"
        GMR_SHARPA_LEFT_MOUNT_POS="${GMR_SHARPA_LEFT_MOUNT_POS:-0.0415,0.003,0}"
        GMR_SHARPA_RIGHT_MOUNT_POS="${GMR_SHARPA_RIGHT_MOUNT_POS:-0.0415,-0.003,0}"
        GMR_EMBODIMENT_LABEL="Unitree G1 body + 3-DoF wrists + Sharpa Wave 22-DoF hands"
        ;;
    sharpa_h1_2)
        GMR_ROBOT="unitree_h1_2"
        GMR_HAND_RETARGET_MODE="off"
        GMR_SHARPA_HANDS="1"
        GMR_SHARPA_AUTO_RETARGET="${GMR_SHARPA_AUTO_RETARGET:-1}"
        GMR_SHARPA_HAND_NPZ_NAME="${GMR_SHARPA_HAND_NPZ_NAME:-001_sharpa_chain_hands.npz}"
        GMR_BRAINCO_HANDS="0"
        GMR_BRAINCO_AUTO_RETARGET="0"
        # H1-2 has roll/pitch/yaw wrist joints. Feed the aligned Hand4Whole++
        # wrist frame into GMR and let IK solve all three axes directly.
        GMR_HAND_WRIST_ORIENTATION_MODE="override_frames"
        GMR_FORCE_HAND_WRIST_ORIENTATION_OVERRIDE="1"
        GMR_RELAX_ORIENTATION_BODIES=""
        GMR_WRIST_PITCH_YAW_STABILIZE="off"
        GMR_PALM_ROLL_MODE="off"
        GMR_SHARPA_MOUNT_POS="${GMR_SHARPA_MOUNT_POS:-0.054,0,0}"
        GMR_SHARPA_LEFT_MOUNT_POS="${GMR_SHARPA_LEFT_MOUNT_POS:-$GMR_SHARPA_MOUNT_POS}"
        GMR_SHARPA_RIGHT_MOUNT_POS="${GMR_SHARPA_RIGHT_MOUNT_POS:-$GMR_SHARPA_MOUNT_POS}"
        GMR_EMBODIMENT_LABEL="Unitree H1-2 body + 3-DoF wrists + Sharpa Wave 22-DoF hands"
        ;;
    g1)
        GMR_ROBOT="unitree_g1_with_hands"
        GMR_HAND_RETARGET_MODE="${GMR_G1_HAND_RETARGET_MODE:-axis_component}"
        GMR_SHARPA_HANDS="0"
        GMR_SHARPA_AUTO_RETARGET="0"
        GMR_BRAINCO_HANDS="0"
        GMR_BRAINCO_AUTO_RETARGET="0"
        # G1 has a true 3-DoF wrist. Preserve the wrist orientation solved from
        # GVHMR, with a soft pitch/yaw guard near the mechanical limits.
        GMR_RELAX_ORIENTATION_BODIES=""
        GMR_WRIST_PITCH_YAW_STABILIZE="soft_limit"
        GMR_PALM_ROLL_MODE="off"
        GMR_EMBODIMENT_LABEL="Unitree G1 body + native Dex3-1 three-finger hands"
        ;;
    brainco)
        GMR_ROBOT="unitree_g1"
        GMR_HAND_RETARGET_MODE="off"
        GMR_SHARPA_HANDS="0"
        GMR_SHARPA_AUTO_RETARGET="0"
        GMR_BRAINCO_HANDS="1"
        GMR_BRAINCO_AUTO_RETARGET="${GMR_BRAINCO_AUTO_RETARGET:-1}"
        # BrainCo is mounted after G1's yaw wrist, so all three G1 wrist axes
        # must remain active for the hand frame to align with GVHMR. Keep them
        # active with a soft pitch/yaw guard near the mechanical limits.
        GMR_RELAX_ORIENTATION_BODIES=""
        GMR_WRIST_PITCH_YAW_STABILIZE="soft_limit"
        GMR_PALM_ROLL_MODE="off"
        GMR_EMBODIMENT_LABEL="Unitree G1 body + BrainCo Revo2 hands (6 motors/hand)"
        ;;
    *)
        echo "Unsupported GMR_HAND_MODEL=$GMR_HAND_MODEL (use sharpa|sharpa_g1|sharpa_h1|sharpa_h1_2|g1|brainco)" >&2
        return 2 2>/dev/null || exit 2
        ;;
esac

export GMR_HAND_MODEL GMR_ROBOT GMR_HAND_RETARGET_MODE
export GMR_SHARPA_HANDS GMR_SHARPA_AUTO_RETARGET GMR_SHARPA_HAND_NPZ_NAME
export GMR_SHARPA_MOUNT_POS GMR_SHARPA_LEFT_MOUNT_POS GMR_SHARPA_RIGHT_MOUNT_POS
export GMR_BRAINCO_HANDS GMR_BRAINCO_AUTO_RETARGET
export GMR_SUPPORT_CONTACT_HEIGHT GMR_SUPPORT_MAX_VERTICAL_SPEED
export GMR_SUPPORT_MIN_CONTACT_RUN GMR_SUPPORT_MAX_CONTACT_GAP
export GMR_SUPPORT_ROOT_STEP_LIMIT
export GMR_EMBODIMENT_LABEL
export GMR_RELAX_ORIENTATION_BODIES GMR_WRIST_PITCH_YAW_STABILIZE GMR_PALM_ROLL_MODE
export GMR_HEIGHT_ADJUST_MODE
export GMR_HAND_WRIST_ORIENTATION_MODE GMR_FORCE_HAND_WRIST_ORIENTATION_OVERRIDE
export GMR_CAMERA_SOURCE GMR_CAMERA_SUBJECT_ALIGN_XY
