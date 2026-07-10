#!/usr/bin/env bash
# Batch-generate hardware-coupled BrainCo Revo2 trajectories from hand sidecars.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="${PIPELINE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PY_GMR="${PY_GMR:-${HOME}/miniconda3/envs/locomotion/bin/python}"

SHOW_ROOT="${SHOW_ROOT:-${PIPELINE_ROOT}/output_dir/kungfu_hand}"
OUT_ROOT="${OUT_ROOT:-$SHOW_ROOT}"
HAND_NPZ_NAME="${HAND_NPZ_NAME:-001_smplx_hands.npz}"
BRAINCO_OUT_NAME="${BRAINCO_OUT_NAME:-001_brainco_revo2_hands.npz}"
BRAINCO_ROOT="${BRAINCO_ROOT:-${SCRIPT_DIR}/third_party/robot_hands/brainco_description/revo2_system}"
BRAINCO_INCLUDE_CLIPS="${BRAINCO_INCLUDE_CLIPS:-}"
BRAINCO_OVERRIDE="${BRAINCO_OVERRIDE:-0}"
BRAINCO_MAX_FRAMES="${BRAINCO_MAX_FRAMES:-0}"
DRY_RUN="${DRY_RUN:-0}"

clip_in_list() {
    local clip="$1"
    local list="$2"
    [ -z "$list" ] && return 0
    case ",$list," in
        *,"$clip",*) return 0 ;;
        *) return 1 ;;
    esac
}

mapfile -d '' HAND_FILES < <(find "$SHOW_ROOT" -mindepth 2 -maxdepth 2 -name "$HAND_NPZ_NAME" -print0 | sort -z)
if [ "${#HAND_FILES[@]}" -eq 0 ]; then
    echo "FATAL: no hand sidecars found: ${SHOW_ROOT}/*/${HAND_NPZ_NAME}" >&2
    exit 1
fi

echo "BrainCo Revo2 hand batch (6 motors/hand + hardware mimic constraints)"
echo "  show_root:    $SHOW_ROOT"
echo "  output_root:  $OUT_ROOT"
echo "  asset_root:   $BRAINCO_ROOT"
echo "  clips:        ${#HAND_FILES[@]}"

for hand_npz in "${HAND_FILES[@]}"; do
    clip="$(basename "$(dirname "$hand_npz")")"
    clip_in_list "$clip" "$BRAINCO_INCLUDE_CLIPS" || continue
    rel="${hand_npz#$SHOW_ROOT/}"
    out_dir="$OUT_ROOT/$(dirname "$rel")"
    output="$out_dir/$BRAINCO_OUT_NAME"
    mkdir -p "$out_dir"
    if [ "$BRAINCO_OVERRIDE" != "1" ] && [ -f "$output" ]; then
        echo "[$clip] skip existing: $output"
        continue
    fi
    cmd=(
        "$PY_GMR" "$SCRIPT_DIR/scripts/brainco_hand_retarget.py"
        --hand_npz "$hand_npz"
        --output "$output"
        --brainco_root "$BRAINCO_ROOT"
        --max_frames "$BRAINCO_MAX_FRAMES"
        --scale "${BRAINCO_SCALE:-1.0}"
        --steps "${BRAINCO_STEPS:-4}"
        --init_steps "${BRAINCO_INIT_STEPS:-50}"
        --wrist_pos_cost "${BRAINCO_WRIST_POS_COST:-0.3}"
        --wrist_ori_cost "${BRAINCO_WRIST_ORI_COST:-0.2}"
        --finger_pos_cost "${BRAINCO_FINGER_POS_COST:-5.0}"
        --joint_pos_cost "${BRAINCO_JOINT_POS_COST:-3.0}"
        --posture_cost "${BRAINCO_POSTURE_COST:-0.01}"
        --temporal_cost "${BRAINCO_TEMPORAL_COST:-0.03}"
        --low_conf_temporal_gain "${BRAINCO_LOW_CONF_TEMPORAL_GAIN:-3.0}"
        --min_target_confidence_scale "${BRAINCO_MIN_TARGET_CONFIDENCE_SCALE:-0.10}"
        --smooth_window "${BRAINCO_SMOOTH_WINDOW:-5}"
        --max_delta "${BRAINCO_MAX_DELTA:-0.12}"
        --max_accel "${BRAINCO_MAX_ACCEL:-0.10}"
        --low_conf_accel_scale "${BRAINCO_LOW_CONF_ACCEL_SCALE:-0.50}"
        --reproj_good_px "${BRAINCO_REPROJ_GOOD_PX:-30.0}"
        --reproj_bad_px "${BRAINCO_REPROJ_BAD_PX:-75.0}"
        --reproj_good_ratio "${BRAINCO_REPROJ_GOOD_RATIO:-0.15}"
        --reproj_bad_ratio "${BRAINCO_REPROJ_BAD_RATIO:-0.45}"
        --max_pip_bend_deg "${BRAINCO_MAX_PIP_BEND_DEG:-125.0}"
        --max_dip_bend_deg "${BRAINCO_MAX_DIP_BEND_DEG:-105.0}"
        --max_source_bone_delta_deg "${BRAINCO_MAX_SOURCE_BONE_DELTA_DEG:-45.0}"
        --max_source_bone_length_ratio "${BRAINCO_MAX_SOURCE_BONE_LENGTH_RATIO:-0.25}"
        --anatomic_repair_max_gap "${BRAINCO_ANATOMIC_REPAIR_MAX_GAP:-15}"
    )
    echo "[$clip] -> $output"
    if [ "$DRY_RUN" = "1" ]; then
        printf '  DRY_RUN:'
        printf ' %q' "${cmd[@]}"
        printf '\n'
    else
        "${cmd[@]}"
    fi
done
