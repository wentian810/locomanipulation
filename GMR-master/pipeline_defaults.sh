#!/bin/bash
# ============================================================================
# Shared defaults for the GVHMR → GMR retargeting pipeline.
#
# Source this file from any pipeline script to get consistent defaults.
# All variables use the "${VAR:-default}" pattern so that callers can
# override any value by exporting it before sourcing.
#
# Usage:
#   export GMR_ROBOT=unitree_h1_with_hand
#   source "${PIPELINE_ROOT}/GMR-master/pipeline_defaults.sh"
# ============================================================================

# ---- GMR core -----------------------------------------------------------
GMR_SOURCE="${GMR_SOURCE:-smoothed}"
GMR_ROBOT="${GMR_ROBOT:-unitree_g1_with_hands}"
GMR_RETARGET_MODE="${GMR_RETARGET_MODE:-full}"
GMR_UPPER_BODY_ROOT_MODE="${GMR_UPPER_BODY_ROOT_MODE:-fixed}"
GMR_MODEL_TYPE="${GMR_MODEL_TYPE:-smplh}"
GMR_AUTO_HAND_NPZ="${GMR_AUTO_HAND_NPZ:-0}"
GMR_HAND_NPZ_NAME="${GMR_HAND_NPZ_NAME:-001_smplx_hands.npz}"
GMR_TARGET_FPS="${GMR_TARGET_FPS:-30}"
GMR_SOLVER="${GMR_SOLVER:-daqp}"
GMR_HEIGHT_ADJUST_MODE="${GMR_HEIGHT_ADJUST_MODE:-per_frame_geom}"
GMR_GROUND_OFFSET="${GMR_GROUND_OFFSET:-0.0}"
GMR_HUMAN_YAW_OFFSET_DEG="${GMR_HUMAN_YAW_OFFSET_DEG:-180}"

# Body-model auto-discovery
DEFAULT_GMR_BODY_MODEL_PATH="${PIPELINE_ROOT}/GMR-master/assets/body_models"
if [ ! -d "${DEFAULT_GMR_BODY_MODEL_PATH}/${GMR_MODEL_TYPE}" ]; then
    for candidate in \
        "${PIPELINE_ROOT}/gvhmr/assets" \
        "${PIPELINE_ROOT}/GVHMR-main/inputs/checkpoints/body_models" \
        "${PIPELINE_ROOT}/GVHMR-hand/GVHMR-main/inputs/checkpoints/body_models"; do
        if [ -d "${candidate}/${GMR_MODEL_TYPE}" ]; then
            DEFAULT_GMR_BODY_MODEL_PATH="$candidate"
            break
        fi
    done
fi
GMR_BODY_MODEL_PATH="${GMR_BODY_MODEL_PATH:-$DEFAULT_GMR_BODY_MODEL_PATH}"

# ---- GMR wrist orientation ----------------------------------------------
GMR_HAND_WRIST_ORIENTATION_MODE="${GMR_HAND_WRIST_ORIENTATION_MODE:-diagnostic}"
GMR_FORCE_HAND_WRIST_ORIENTATION_OVERRIDE="${GMR_FORCE_HAND_WRIST_ORIENTATION_OVERRIDE:-0}"
if [ -z "${GMR_RELAX_ORIENTATION_BODIES+x}" ]; then
    if [[ "$GMR_HAND_WRIST_ORIENTATION_MODE" = "override_frames" ]] && \
       ([[ "$GMR_FORCE_HAND_WRIST_ORIENTATION_OVERRIDE" = "1" ]] || [[ "$GMR_ROBOT" = "unitree_h1_with_hand_wrist" ]]) && \
       ([[ "$GMR_AUTO_HAND_NPZ" = "1" ]] || [[ -n "${GMR_HAND_NPZ:-}" ]]); then
        GMR_RELAX_ORIENTATION_BODIES=""
    else
        GMR_RELAX_ORIENTATION_BODIES="left_wrist,right_wrist"
    fi
fi

# ---- GMR hand finger retargeting ----------------------------------------
GMR_HAND_INVALID_MODE="${GMR_HAND_INVALID_MODE:-hold}"
GMR_HAND_INTERP_MAX_GAP="${GMR_HAND_INTERP_MAX_GAP:-15}"
GMR_HAND_RETARGET_MODE="${GMR_HAND_RETARGET_MODE:-axis_component}"
GMR_HAND_ANGLE_SCALE="${GMR_HAND_ANGLE_SCALE:-2.0}"
GMR_HAND_CURL_POWER="${GMR_HAND_CURL_POWER:-0.5}"
GMR_HAND_SMOOTH_WINDOW="${GMR_HAND_SMOOTH_WINDOW:-25}"
GMR_HAND_SMOOTH_POLYORDER="${GMR_HAND_SMOOTH_POLYORDER:-2}"
GMR_HAND_MEDIAN_WINDOW="${GMR_HAND_MEDIAN_WINDOW:-5}"
GMR_HAND_MAX_DELTA="${GMR_HAND_MAX_DELTA:-0.08}"
GMR_HAND_DEADBAND="${GMR_HAND_DEADBAND:-0.005}"

# ---- GMR palm roll ------------------------------------------------------
GMR_PALM_ROLL_MODE="${GMR_PALM_ROLL_MODE:-auto}"
if [ -z "${GMR_PALM_ROLL_GAIN+x}" ]; then
    case "$GMR_ROBOT" in
        # H1 has one axial hand joint rather than a full 3-DoF wrist.  A
        # slightly reduced gain tracks pronation/supination without feeding
        # the estimator's residual swing into that single joint.
        unitree_h1_with_hand|unitree_h1_with_hand_wrist)
            GMR_PALM_ROLL_GAIN="0.75" ;;
        *)  GMR_PALM_ROLL_GAIN="1.0" ;;
    esac
fi
if [ -z "${GMR_PALM_ROLL_SOURCE+x}" ]; then
    # Use the body-model wrist/MCP frame for the arm-to-hand connection.
    # HaMeR/Hand4Whole++ global hand frames are crop-camera predictions and
    # can slowly rotate relative to the GVHMR arm frame even when their finger
    # articulation is good.  They remain the source of all finger joints.
    GMR_PALM_ROLL_SOURCE="smpl"
fi
GMR_PALM_ROLL_NORMAL_DOT_MIN="${GMR_PALM_ROLL_NORMAL_DOT_MIN:--0.5}"
# In h1_with_hand.xml both hand joints rotate around the same +X convention.
# Mirroring the left sign creates the observed left/right 90-degree mismatch.
GMR_LEFT_PALM_ROLL_SIGN="${GMR_LEFT_PALM_ROLL_SIGN:-1.0}"
GMR_RIGHT_PALM_ROLL_SIGN="${GMR_RIGHT_PALM_ROLL_SIGN:-1.0}"
GMR_PALM_ROLL_SMOOTH_WINDOW="${GMR_PALM_ROLL_SMOOTH_WINDOW:-15}"
GMR_PALM_ROLL_MAX_DELTA="${GMR_PALM_ROLL_MAX_DELTA:-0.12}"
if [ -z "${GMR_PALM_ROLL_MAX_ABS+x}" ]; then
    case "$GMR_ROBOT" in
        unitree_h1_with_hand|unitree_h1_with_hand_wrist)
            # Human forearm pronation/supination is much narrower than the
            # mechanical +/-175-degree H1 hand-joint range.  Keep the target
            # inside a physically plausible +/-77 degrees.
            GMR_PALM_ROLL_MAX_ABS="1.35" ;;
        *)  GMR_PALM_ROLL_MAX_ABS="0.0" ;;
    esac
fi
if [ -z "${GMR_PALM_ROLL_BRANCH_MODE+x}" ]; then
    # Palm normals are directed vectors and therefore 2*pi-periodic.  The old
    # pi-branch selector treated n and -n as equivalent, changed branch after
    # the opening frames, and left both H1 hands at the wrong angle.
    GMR_PALM_ROLL_BRANCH_MODE="off"
fi
GMR_PALM_ROLL_BRANCH_CANDIDATES="${GMR_PALM_ROLL_BRANCH_CANDIDATES:-2}"
GMR_PALM_ROLL_BRANCH_ANCHOR_FRAMES="${GMR_PALM_ROLL_BRANCH_ANCHOR_FRAMES:-45}"
GMR_PALM_ROLL_BRANCH_TRANSITION_WEIGHT="${GMR_PALM_ROLL_BRANCH_TRANSITION_WEIGHT:-1.0}"
GMR_PALM_ROLL_BRANCH_PENALTY="${GMR_PALM_ROLL_BRANCH_PENALTY:-0.001}"
GMR_PALM_ROLL_BRANCH_ANCHOR_PENALTY="${GMR_PALM_ROLL_BRANCH_ANCHOR_PENALTY:-0.03}"
GMR_PALM_ROLL_BRANCH_RANGE_PENALTY="${GMR_PALM_ROLL_BRANCH_RANGE_PENALTY:-3.0}"

# ---- GMR wrist pitch/yaw stabilization ----------------------------------
GMR_WRIST_PITCH_YAW_STABILIZE="${GMR_WRIST_PITCH_YAW_STABILIZE:-auto}"
GMR_WRIST_PITCH_NEUTRAL="${GMR_WRIST_PITCH_NEUTRAL:-0.0}"
GMR_WRIST_YAW_NEUTRAL="${GMR_WRIST_YAW_NEUTRAL:-0.0}"

# ---- GMR rendering & composite ------------------------------------------
GMR_RENDER="${GMR_RENDER:-1}"
GMR_RENDER_MODE="${GMR_RENDER_MODE:-mesh}"
GMR_CAMERA_SOURCE="${GMR_CAMERA_SOURCE:-gvhmr}"
GMR_RENDER_CAMERA_MODE="${GMR_RENDER_CAMERA_MODE:-threequarter}"
GMR_RENDER_SKIP="${GMR_RENDER_SKIP:-1}"
GMR_RENDER_MAX_FRAMES="${GMR_RENDER_MAX_FRAMES:-0}"
GMR_MUJOCO_GL="${GMR_MUJOCO_GL:-osmesa}"
GMR_RENDER_WIDTH="${GMR_RENDER_WIDTH:-960}"
GMR_RENDER_HEIGHT="${GMR_RENDER_HEIGHT:-720}"
GMR_RENDER_RADIUS="${GMR_RENDER_RADIUS:-2.2}"
GMR_COMPOSITE="${GMR_COMPOSITE:-1}"
GMR_COMPOSITE_NAME="${GMR_COMPOSITE_NAME:-composite_2x2.mp4}"
GMR_COMPOSITE_PANEL_HEIGHT="${GMR_COMPOSITE_PANEL_HEIGHT:-540}"
GMR_COMPOSITE_FPS="${GMR_COMPOSITE_FPS:-30}"

# ---- GMR misc -----------------------------------------------------------
GMR_SMOOTH_WINDOW="${GMR_SMOOTH_WINDOW:-9}"
GMR_HAND_MODEL="${GMR_HAND_MODEL:-sharpa}"
GMR_SHARPA_HANDS="${GMR_SHARPA_HANDS:-0}"
GMR_SHARPA_AUTO_RETARGET="${GMR_SHARPA_AUTO_RETARGET:-0}"
GMR_SHARPA_HAND_NPZ_NAME="${GMR_SHARPA_HAND_NPZ_NAME:-001_sharpa_hands.npz}"
GMR_SHARPA_MOUNT_POS="${GMR_SHARPA_MOUNT_POS:-0.055,0,0}"
GMR_SHARPA_MOUNT_QUAT="${GMR_SHARPA_MOUNT_QUAT:-}"
# Sharpa's left/right meshes are mirrored in local Y, while H1's two hand-link
# frames share the same convention.  The left mount therefore needs its own
# proper rotation (det=+1); reusing the right quaternion reverses its fingers.
if [ -z "${GMR_SHARPA_LEFT_MOUNT_QUAT+x}" ]; then
    GMR_SHARPA_LEFT_MOUNT_QUAT="${GMR_SHARPA_MOUNT_QUAT:-0.5,-0.5,0.5,-0.5}"
fi
if [ -z "${GMR_SHARPA_RIGHT_MOUNT_QUAT+x}" ]; then
    GMR_SHARPA_RIGHT_MOUNT_QUAT="${GMR_SHARPA_MOUNT_QUAT:-0.5,0.5,0.5,0.5}"
fi
GMR_SHARPA_SCALE="${GMR_SHARPA_SCALE:-1.0}"
GMR_SHARPA_STEPS="${GMR_SHARPA_STEPS:-4}"
GMR_SHARPA_INIT_STEPS="${GMR_SHARPA_INIT_STEPS:-50}"
GMR_SHARPA_WRIST_POS_COST="${GMR_SHARPA_WRIST_POS_COST:-0.3}"
GMR_SHARPA_WRIST_ORI_COST="${GMR_SHARPA_WRIST_ORI_COST:-0.2}"
GMR_SHARPA_FINGER_POS_COST="${GMR_SHARPA_FINGER_POS_COST:-5.0}"
GMR_SHARPA_JOINT_POS_COST="${GMR_SHARPA_JOINT_POS_COST:-3.0}"
GMR_SHARPA_POSTURE_COST="${GMR_SHARPA_POSTURE_COST:-0.01}"
GMR_SHARPA_TEMPORAL_COST="${GMR_SHARPA_TEMPORAL_COST:-0.03}"
GMR_SHARPA_LOW_CONF_TEMPORAL_GAIN="${GMR_SHARPA_LOW_CONF_TEMPORAL_GAIN:-3.0}"
GMR_SHARPA_MIN_TARGET_CONFIDENCE_SCALE="${GMR_SHARPA_MIN_TARGET_CONFIDENCE_SCALE:-0.10}"
GMR_SHARPA_SMOOTH_WINDOW="${GMR_SHARPA_SMOOTH_WINDOW:-5}"
GMR_SHARPA_MAX_DELTA="${GMR_SHARPA_MAX_DELTA:-0.12}"
GMR_SHARPA_MAX_ACCEL="${GMR_SHARPA_MAX_ACCEL:-0.10}"
GMR_SHARPA_LOW_CONF_ACCEL_SCALE="${GMR_SHARPA_LOW_CONF_ACCEL_SCALE:-0.50}"
GMR_SHARPA_REPROJ_GOOD_PX="${GMR_SHARPA_REPROJ_GOOD_PX:-30.0}"
GMR_SHARPA_REPROJ_BAD_PX="${GMR_SHARPA_REPROJ_BAD_PX:-75.0}"
GMR_SHARPA_REPROJ_GOOD_RATIO="${GMR_SHARPA_REPROJ_GOOD_RATIO:-0.15}"
GMR_SHARPA_REPROJ_BAD_RATIO="${GMR_SHARPA_REPROJ_BAD_RATIO:-0.45}"
GMR_SHARPA_MAX_PIP_BEND_DEG="${GMR_SHARPA_MAX_PIP_BEND_DEG:-125.0}"
GMR_SHARPA_MAX_DIP_BEND_DEG="${GMR_SHARPA_MAX_DIP_BEND_DEG:-105.0}"
GMR_SHARPA_MAX_SOURCE_BONE_DELTA_DEG="${GMR_SHARPA_MAX_SOURCE_BONE_DELTA_DEG:-45.0}"
GMR_SHARPA_MAX_SOURCE_BONE_LENGTH_RATIO="${GMR_SHARPA_MAX_SOURCE_BONE_LENGTH_RATIO:-0.25}"
GMR_SHARPA_ANATOMIC_REPAIR_MAX_GAP="${GMR_SHARPA_ANATOMIC_REPAIR_MAX_GAP:-15}"
GMR_BRAINCO_HANDS="${GMR_BRAINCO_HANDS:-0}"
GMR_BRAINCO_AUTO_RETARGET="${GMR_BRAINCO_AUTO_RETARGET:-0}"
GMR_BRAINCO_HAND_NPZ_NAME="${GMR_BRAINCO_HAND_NPZ_NAME:-001_brainco_revo2_hands.npz}"
GMR_BRAINCO_ROOT="${GMR_BRAINCO_ROOT:-${PIPELINE_ROOT}/GMR-master/third_party/robot_hands/brainco_description/revo2_system}"
GMR_BRAINCO_LEFT_MOUNT_POS="${GMR_BRAINCO_LEFT_MOUNT_POS:-0.0415,0.003,0}"
GMR_BRAINCO_RIGHT_MOUNT_POS="${GMR_BRAINCO_RIGHT_MOUNT_POS:-0.0415,-0.003,0}"
# Unitree's official Revo2 URDF inserts rpy=(pi/2, pi, 0) between the
# G1-compatible base_link and BrainCo's native hand base.
GMR_BRAINCO_LEFT_MOUNT_QUAT="${GMR_BRAINCO_LEFT_MOUNT_QUAT:-0,0,0.70710678,-0.70710678}"
GMR_BRAINCO_RIGHT_MOUNT_QUAT="${GMR_BRAINCO_RIGHT_MOUNT_QUAT:-0,0,0.70710678,-0.70710678}"
if [ -z "${GMR_EMBODIMENT_LABEL+x}" ]; then
    if [ "$GMR_BRAINCO_HANDS" = "1" ]; then
        GMR_EMBODIMENT_LABEL="Unitree G1 body + BrainCo Revo2 hands (6 motors/hand)"
    elif [ "$GMR_SHARPA_HANDS" = "1" ]; then
        case "$GMR_ROBOT" in
            unitree_h1_with_hand|unitree_h1_with_hand_wrist)
                GMR_EMBODIMENT_LABEL="Unitree H1 body + Sharpa Wave 22-DoF hands"
                ;;
            *)
                GMR_EMBODIMENT_LABEL="${GMR_ROBOT} body + Sharpa Wave 22-DoF hands"
                ;;
        esac
    else
        GMR_EMBODIMENT_LABEL="$GMR_ROBOT"
    fi
fi
GMR_OBJECT_PROXY="${GMR_OBJECT_PROXY:-0}"
GMR_OBJECT_PROXY_SOURCE="${GMR_OBJECT_PROXY_SOURCE:-smpl}"
GMR_OBJECT_TYPE="${GMR_OBJECT_TYPE:-box}"
GMR_OBJECT_GEOM_SIZE="${GMR_OBJECT_GEOM_SIZE:-0.16,0.10,0.07}"
GMR_OBJECT_OFFSET="${GMR_OBJECT_OFFSET:-0,0,0.02}"
GMR_OBJECT_SMOOTH_WINDOW="${GMR_OBJECT_SMOOTH_WINDOW:-15}"
GMR_OVERRIDE="${GMR_OVERRIDE:-1}"
GMR_INCLUDE_CLIPS="${GMR_INCLUDE_CLIPS:-}"
GMR_OUTPUT_NAME="${GMR_OUTPUT_NAME:-robot_motion.pkl}"
if [ -z "${GMR_RENDER_NAME+x}" ]; then
    if [ "$GMR_BRAINCO_HANDS" = "1" ]; then
        GMR_RENDER_NAME="${GMR_ROBOT}_brainco_revo2_${GMR_CAMERA_SOURCE}.mp4"
    elif [ "$GMR_SHARPA_HANDS" = "1" ]; then
        GMR_RENDER_NAME="${GMR_ROBOT}_sharpa_${GMR_CAMERA_SOURCE}.mp4"
    else
        GMR_RENDER_NAME="${GMR_ROBOT}_retarget_${GMR_CAMERA_SOURCE}.mp4"
    fi
fi
if [ -z "${GMR_ROBOT_RGBA+x}" ]; then
    if [ "$GMR_ROBOT" = "unitree_h1_with_hand" ]; then
        GMR_ROBOT_RGBA="0.72,0.74,0.76,1.0"
    else
        GMR_ROBOT_RGBA=""
    fi
fi
if [ -z "${GMR_BACKGROUND_RGB+x}" ]; then
    GMR_BACKGROUND_RGB=""
fi
GMR_BACKGROUND_THRESHOLD="${GMR_BACKGROUND_THRESHOLD:-4}"
GMR_OBJECT_MOTION_NAME="${GMR_OBJECT_MOTION_NAME:-}"
if [ "$GMR_OBJECT_PROXY" = "1" ] && [ -z "$GMR_OBJECT_MOTION_NAME" ]; then
    if [ "$GMR_OBJECT_PROXY_SOURCE" = "robot" ]; then
        GMR_OBJECT_MOTION_NAME="object_proxy_robot.npz"
    else
        GMR_OBJECT_MOTION_NAME="object_proxy_smpl.npz"
    fi
elif [ -z "$GMR_OBJECT_MOTION_NAME" ]; then
    # Written by GVHMR-hand/GVHMR-main/tools/pipeline/
    # run_object_reconstruction_bridge.py.
    # The renderer only enables it when the per-clip file actually exists.
    GMR_OBJECT_MOTION_NAME="object_reconstruction/object_motion_gmr.npz"
fi
