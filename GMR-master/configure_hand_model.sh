#!/usr/bin/env bash
# Select one complete GMR hand embodiment.
#
# Source this before pipeline_defaults.sh. Supported values:
#   GMR_HAND_MODEL=sharpa  -> Unitree H1 body + external Sharpa Wave hands
#   GMR_HAND_MODEL=g1      -> Unitree G1 body + native Dex3-1 three-finger hands
#   GMR_HAND_MODEL=brainco -> Unitree G1 body + external BrainCo Revo2 hands

if [ -z "${PIPELINE_ROOT:-}" ]; then
    echo "configure_hand_model.sh requires PIPELINE_ROOT" >&2
    return 1 2>/dev/null || exit 1
fi

GMR_HAND_MODEL="${GMR_HAND_MODEL:-sharpa}"
case "$GMR_HAND_MODEL" in
    sharpa)
        GMR_ROBOT="unitree_h1_with_hand"
        GMR_HAND_RETARGET_MODE="off"
        GMR_SHARPA_HANDS="1"
        GMR_SHARPA_AUTO_RETARGET="${GMR_SHARPA_AUTO_RETARGET:-1}"
        GMR_BRAINCO_HANDS="0"
        GMR_BRAINCO_AUTO_RETARGET="0"
        GMR_EMBODIMENT_LABEL="Unitree H1 body + Sharpa Wave 22-DoF hands"
        ;;
    g1)
        GMR_ROBOT="unitree_g1_with_hands"
        GMR_HAND_RETARGET_MODE="${GMR_G1_HAND_RETARGET_MODE:-axis_component}"
        GMR_SHARPA_HANDS="0"
        GMR_SHARPA_AUTO_RETARGET="0"
        GMR_BRAINCO_HANDS="0"
        GMR_BRAINCO_AUTO_RETARGET="0"
        # G1 has a true 3-DoF wrist. Preserve the wrist orientation solved from
        # GVHMR instead of zeroing pitch/yaw and replacing only palm roll.
        GMR_RELAX_ORIENTATION_BODIES=""
        GMR_WRIST_PITCH_YAW_STABILIZE="off"
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
        # must remain active for the hand frame to align with GVHMR.
        GMR_RELAX_ORIENTATION_BODIES=""
        GMR_WRIST_PITCH_YAW_STABILIZE="off"
        GMR_PALM_ROLL_MODE="off"
        GMR_EMBODIMENT_LABEL="Unitree G1 body + BrainCo Revo2 hands (6 motors/hand)"
        ;;
    *)
        echo "Unsupported GMR_HAND_MODEL=$GMR_HAND_MODEL (use sharpa|g1|brainco)" >&2
        return 2 2>/dev/null || exit 2
        ;;
esac

export GMR_HAND_MODEL GMR_ROBOT GMR_HAND_RETARGET_MODE
export GMR_SHARPA_HANDS GMR_SHARPA_AUTO_RETARGET
export GMR_BRAINCO_HANDS GMR_BRAINCO_AUTO_RETARGET
export GMR_EMBODIMENT_LABEL
export GMR_RELAX_ORIENTATION_BODIES GMR_WRIST_PITCH_YAW_STABILIZE GMR_PALM_ROLL_MODE
