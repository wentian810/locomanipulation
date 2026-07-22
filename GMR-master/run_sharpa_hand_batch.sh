#!/bin/bash
# Batch-generate 22-DoF Sharpa hand trajectories from GVHMR-hand sidecars.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="${PIPELINE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

CONDA_BASE="${CONDA_BASE:-${HOME}/miniconda3}"
PY_GMR="${PY_GMR:-${CONDA_BASE}/envs/locomotion/bin/python}"

SHOW_ROOT="${SHOW_ROOT:-${PIPELINE_ROOT}/output_dir/kungfu_hand}"
OUT_ROOT="${OUT_ROOT:-$SHOW_ROOT}"
HAND_NPZ_NAME="${HAND_NPZ_NAME:-001_smplx_hands.npz}"
SHARPA_OUT_NAME="${SHARPA_OUT_NAME:-001_sharpa_hands.npz}"
SHARPA_SIDE="${SHARPA_SIDE:-both}"
SHARPA_MAX_FRAMES="${SHARPA_MAX_FRAMES:-0}"
SHARPA_STEPS="${SHARPA_STEPS:-4}"
SHARPA_INIT_STEPS="${SHARPA_INIT_STEPS:-50}"
SHARPA_WRIST_POS_COST="${SHARPA_WRIST_POS_COST:-0.3}"
SHARPA_WRIST_ORI_COST="${SHARPA_WRIST_ORI_COST:-0.2}"
SHARPA_FINGER_POS_COST="${SHARPA_FINGER_POS_COST:-5.0}"
SHARPA_JOINT_POS_COST="${SHARPA_JOINT_POS_COST:-3.0}"
SHARPA_POSTURE_COST="${SHARPA_POSTURE_COST:-0.01}"
SHARPA_TEMPORAL_COST="${SHARPA_TEMPORAL_COST:-0.03}"
SHARPA_LOW_CONF_TEMPORAL_GAIN="${SHARPA_LOW_CONF_TEMPORAL_GAIN:-3.0}"
SHARPA_MIN_TARGET_CONFIDENCE_SCALE="${SHARPA_MIN_TARGET_CONFIDENCE_SCALE:-0.10}"
SHARPA_RELIABILITY_SMOOTH_WINDOW="${SHARPA_RELIABILITY_SMOOTH_WINDOW:-9}"
SHARPA_LOW_CONF_RELIABILITY_THR="${SHARPA_LOW_CONF_RELIABILITY_THR:-0.25}"
SHARPA_SMOOTH_WINDOW="${SHARPA_SMOOTH_WINDOW:-5}"
SHARPA_MAX_DELTA="${SHARPA_MAX_DELTA:-0.12}"
SHARPA_MAX_ACCEL="${SHARPA_MAX_ACCEL:-0.10}"
SHARPA_LOW_CONF_ACCEL_SCALE="${SHARPA_LOW_CONF_ACCEL_SCALE:-0.50}"
SHARPA_REPROJ_GOOD_PX="${SHARPA_REPROJ_GOOD_PX:-30.0}"
SHARPA_REPROJ_BAD_PX="${SHARPA_REPROJ_BAD_PX:-75.0}"
SHARPA_REPROJ_GOOD_RATIO="${SHARPA_REPROJ_GOOD_RATIO:-0.15}"
SHARPA_REPROJ_BAD_RATIO="${SHARPA_REPROJ_BAD_RATIO:-0.45}"
SHARPA_MAX_PIP_BEND_DEG="${SHARPA_MAX_PIP_BEND_DEG:-125.0}"
SHARPA_MAX_DIP_BEND_DEG="${SHARPA_MAX_DIP_BEND_DEG:-105.0}"
SHARPA_MAX_SOURCE_BONE_DELTA_DEG="${SHARPA_MAX_SOURCE_BONE_DELTA_DEG:-45.0}"
SHARPA_MAX_SOURCE_BONE_LENGTH_RATIO="${SHARPA_MAX_SOURCE_BONE_LENGTH_RATIO:-0.25}"
SHARPA_ANATOMIC_REPAIR_MAX_GAP="${SHARPA_ANATOMIC_REPAIR_MAX_GAP:-15}"
SHARPA_INCLUDE_CLIPS="${SHARPA_INCLUDE_CLIPS:-}"
SHARPA_OVERRIDE="${SHARPA_OVERRIDE:-0}"
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

resolve_hand_sidecar_for_clip() {
    local clip_dir="$1"
    local candidate
    for candidate in \
        "$clip_dir/$HAND_NPZ_NAME" \
        "$clip_dir/_intermediate/npz/$HAND_NPZ_NAME"; do
        if [ -f "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

mapfile -d '' CLIP_DIRS < <(find "$SHOW_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
if [ "${#CLIP_DIRS[@]}" -eq 0 ]; then
    echo "FATAL: no clip directories found under: ${SHOW_ROOT}" >&2
    exit 1
fi

echo "Sharpa hand batch"
echo "  show_root:    $SHOW_ROOT"
echo "  output_root:  $OUT_ROOT"
echo "  clips:        ${#CLIP_DIRS[@]}"
echo "  include:      ${SHARPA_INCLUDE_CLIPS:-<all>}"
echo "  python:       $PY_GMR"

FOUND_HAND_FILES=0
MISSING_HAND_FILES=()
for clip_dir in "${CLIP_DIRS[@]}"; do
    clip="$(basename "$clip_dir")"
    if ! clip_in_list "$clip" "$SHARPA_INCLUDE_CLIPS"; then
        continue
    fi
    hand_npz="$(resolve_hand_sidecar_for_clip "$clip_dir" || true)"
    if [ -z "$hand_npz" ]; then
        MISSING_HAND_FILES+=("$clip_dir/$HAND_NPZ_NAME")
        continue
    fi
    FOUND_HAND_FILES=$((FOUND_HAND_FILES + 1))
    out_dir="$OUT_ROOT/$clip"
    output="$out_dir/$SHARPA_OUT_NAME"
    mkdir -p "$out_dir"
    if [ "$SHARPA_OVERRIDE" != "1" ] && [ -f "$output" ]; then
        echo "[$clip] skip existing: $output"
        continue
    fi
    echo "[$clip] -> $output"
    cmd=(
        "$PY_GMR" "$SCRIPT_DIR/scripts/sharpa_hand_retarget.py"
        --hand_npz "$hand_npz"
        --output "$output"
        --side "$SHARPA_SIDE"
        --max_frames "$SHARPA_MAX_FRAMES"
        --steps "$SHARPA_STEPS"
        --init_steps "$SHARPA_INIT_STEPS"
        --wrist_pos_cost "$SHARPA_WRIST_POS_COST"
        --wrist_ori_cost "$SHARPA_WRIST_ORI_COST"
        --finger_pos_cost "$SHARPA_FINGER_POS_COST"
        --joint_pos_cost "$SHARPA_JOINT_POS_COST"
        --posture_cost "$SHARPA_POSTURE_COST"
        --temporal_cost "$SHARPA_TEMPORAL_COST"
        --low_conf_temporal_gain "$SHARPA_LOW_CONF_TEMPORAL_GAIN"
        --min_target_confidence_scale "$SHARPA_MIN_TARGET_CONFIDENCE_SCALE"
        --reliability_smooth_window "$SHARPA_RELIABILITY_SMOOTH_WINDOW"
        --low_conf_reliability_thr "$SHARPA_LOW_CONF_RELIABILITY_THR"
        --smooth_window "$SHARPA_SMOOTH_WINDOW"
        --max_delta "$SHARPA_MAX_DELTA"
        --max_accel "$SHARPA_MAX_ACCEL"
        --low_conf_accel_scale "$SHARPA_LOW_CONF_ACCEL_SCALE"
        --reproj_good_px "$SHARPA_REPROJ_GOOD_PX"
        --reproj_bad_px "$SHARPA_REPROJ_BAD_PX"
        --reproj_good_ratio "$SHARPA_REPROJ_GOOD_RATIO"
        --reproj_bad_ratio "$SHARPA_REPROJ_BAD_RATIO"
        --max_pip_bend_deg "$SHARPA_MAX_PIP_BEND_DEG"
        --max_dip_bend_deg "$SHARPA_MAX_DIP_BEND_DEG"
        --max_source_bone_delta_deg "$SHARPA_MAX_SOURCE_BONE_DELTA_DEG"
        --max_source_bone_length_ratio "$SHARPA_MAX_SOURCE_BONE_LENGTH_RATIO"
        --anatomic_repair_max_gap "$SHARPA_ANATOMIC_REPAIR_MAX_GAP"
    )
    if [ "$DRY_RUN" = "1" ]; then
        printf '  DRY_RUN:'
        printf ' %q' "${cmd[@]}"
        printf '\n'
    else
        "${cmd[@]}"
    fi
done

if [ "$FOUND_HAND_FILES" -eq 0 ]; then
    echo "FATAL: no hand sidecars found for selected clips. Checked:" >&2
    for missing in "${MISSING_HAND_FILES[@]}"; do
        echo "  $missing or ${missing%/*}/_intermediate/npz/$HAND_NPZ_NAME" >&2
    done
    exit 1
fi
