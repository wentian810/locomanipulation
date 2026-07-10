#!/bin/bash
# Batch-retarget selected pipeline results under ../show into GMR robot motions.
#
# Default source is auto, which prefers the final PHC-grounded motion and
# falls back to the best pre-PHC motion. Default height mode uses foot-only
# MuJoCo geometry with a global offset, so non-foot low meshes do not lift
# the robot feet off the floor. Override with:
#   GMR_SOURCE=converted|smoothed|contact_stabilized|phc|phc_smoothed|auto bash run_show_gmr_batch.sh
#   GMR_HEIGHT_ADJUST_MODE=global_foot_geom bash run_show_gmr_batch.sh
#   GMR_CAMERA_SOURCE=fixed bash run_show_gmr_batch.sh
#
# Usage:
#   cd <repo-root>/GMR-master
#   bash run_show_gmr_batch.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="${PIPELINE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SHOW_ROOT="${SHOW_ROOT:-${PIPELINE_ROOT}/show}"
SHOW_ROOT="$(realpath -m "$SHOW_ROOT")"
ORIGINAL_SHOW_ROOT="$SHOW_ROOT"

CONDA_BASE="${CONDA_BASE:-${HOME}/miniconda3}"
PY_GMR="${PY_GMR:-${CONDA_BASE}/envs/gmr/bin/python}"
if [ ! -x "$PY_GMR" ]; then
    PY_GMR="${PY_FALLBACK:-${CONDA_BASE}/envs/locomotion/bin/python}"
fi
if [ -x "$PY_GMR" ]; then
    PY_GMR_BIN_DIR="$(cd "$(dirname "$PY_GMR")" && pwd)"
    export PATH="${PY_GMR_BIN_DIR}:${PATH}"
fi

# Source shared GMR defaults — override any value by exporting it before calling this script
source "${SCRIPT_DIR}/pipeline_defaults.sh"

DRY_RUN="${DRY_RUN:-0}"

case "$GMR_SOURCE" in
    converted)
        SOURCE_FILE="001_converted.npz"
        ;;
    smoothed)
        SOURCE_FILE="001_smoothed.npz"
        ;;
    contact_stabilized)
        SOURCE_FILE="001_contact_stabilized.npz"
        ;;
    phc)
        SOURCE_FILE="001_phc.npz"
        ;;
    phc_smoothed)
        SOURCE_FILE="001_phc_smoothed.npz"
        ;;
    phc_smoothed_grounded)
        SOURCE_FILE="001_phc_smoothed_grounded.npz"
        ;;
    phc_grounded)
        SOURCE_FILE="001_phc_grounded.npz"
        ;;
    auto)
        SOURCE_FILE="001_phc.npz"
        ;;
    *)
        echo "Unsupported GMR_SOURCE=$GMR_SOURCE (use converted|smoothed|contact_stabilized|phc|phc_smoothed|phc_smoothed_grounded|phc_grounded|auto)" >&2
        exit 1
        ;;
esac

case "$GMR_CAMERA_SOURCE" in
    gvhmr|fixed)
        ;;
    *)
        echo "Unsupported GMR_CAMERA_SOURCE=$GMR_CAMERA_SOURCE (use gvhmr|fixed)" >&2
        exit 1
        ;;
esac

OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/output/show_gmr/${GMR_ROBOT}_${GMR_RETARGET_MODE}_${GMR_SOURCE}_${GMR_HEIGHT_ADJUST_MODE}}"
OUT_ROOT="$(realpath -m "$OUT_ROOT")"
LOG_FILE="${OUT_ROOT}/batch.log"
SUMMARY_CSV="${OUT_ROOT}/summary.csv"

mkdir -p "$OUT_ROOT"
: > "$LOG_FILE"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

run_logged() {
    if [ "$DRY_RUN" = "1" ]; then
        printf '[DRY_RUN]' | tee -a "$LOG_FILE"
        printf ' %q' "$@" | tee -a "$LOG_FILE"
        printf '\n' | tee -a "$LOG_FILE"
        return 0
    fi
    PYTHONUNBUFFERED=1 "$@" 2>&1 | tee -a "$LOG_FILE"
}

clip_in_list() {
    local clip="$1"
    local list="$2"
    [ -z "$list" ] && return 0
    case ",$list," in
        *,"$clip",*) return 0 ;;
        *) return 1 ;;
    esac
}

source_clip_name() {
    local src="$1"
    local rel="${src#${SHOW_ROOT}/}"
    printf '%s\n' "${rel%%/*}"
}

clip_slug_name() {
    local raw="$1"
    local slug
    local hash
    slug="$(printf '%s' "$raw" \
        | tr '[:upper:]' '[:lower:]' \
        | sed -E \
            -e 's/clip([0-9]+)/c\1/g' \
            -e 's/form([0-9]+)/f\1/g' \
            -e 's/(demonstration|master|while|the|and|of|a|an)//g' \
            -e 's/[^a-z0-9]+/_/g' \
            -e 's/_+/_/g' \
            -e 's/^_//' \
            -e 's/_$//')"
    [ -n "$slug" ] || slug="clip"
    if [ "${#slug}" -gt 64 ]; then
        hash="$(printf '%s' "$raw" | cksum | awk '{print $1}')"
        slug="${slug:0:54}_${hash}"
        slug="${slug%_}"
    fi
    printf '%s\n' "$slug"
}

clip_video_name() {
    local template="$1"
    local clip="$2"
    local slug
    local name
    local clip_pattern="{clip}"
    local clip_name_pattern="{clip_name}"
    slug="$(clip_slug_name "$clip")"
    name="${template//$clip_name_pattern/$clip}"
    name="${name//$clip_pattern/$slug}"
    printf '%s\n' "$name"
}

resolve_hand_sidecar_for_source() {
    local source="$1"
    local resolved_source
    local source_dir
    local candidate
    resolved_source="$(readlink -f "$source")"
    source_dir="$(dirname "$resolved_source")"

    candidate="${source_dir}/${GMR_HAND_NPZ_NAME}"
    if [ -f "$candidate" ]; then
        printf '%s\n' "$candidate"
        return 0
    fi

    if [ "$(basename "$source_dir")" = "npz" ] && \
       [ "$(basename "$(dirname "$source_dir")")" = "_intermediate" ]; then
        candidate="$(dirname "$(dirname "$source_dir")")/${GMR_HAND_NPZ_NAME}"
        if [ -f "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    fi

    candidate="$(dirname "$source")/${GMR_HAND_NPZ_NAME}"
    if [ -f "$candidate" ]; then
        readlink -f "$candidate"
        return 0
    fi

    printf '%s\n' "${source_dir}/${GMR_HAND_NPZ_NAME}"
    return 1
}

choose_auto_source() {
    local clip_dir="$1"
    GMR_AUTO_SOURCE_QUALITY_GATE="$GMR_AUTO_SOURCE_QUALITY_GATE" \
    GMR_AUTO_SOURCE_MAX_BAD_TRANS_RATIO="$GMR_AUTO_SOURCE_MAX_BAD_TRANS_RATIO" \
    GMR_AUTO_SOURCE_MAX_BAD_JOINT_RATIO="$GMR_AUTO_SOURCE_MAX_BAD_JOINT_RATIO" \
    GMR_AUTO_SOURCE_MAX_TRANS_DELTA_M="$GMR_AUTO_SOURCE_MAX_TRANS_DELTA_M" \
    GMR_AUTO_SOURCE_MAX_ROOT_TILT_P95_DEG="$GMR_AUTO_SOURCE_MAX_ROOT_TILT_P95_DEG" \
    GMR_AUTO_SOURCE_MAX_ROOT_TILT_DEG="$GMR_AUTO_SOURCE_MAX_ROOT_TILT_DEG" \
    GMR_AUTO_SOURCE_MAX_TRANS_FROM_PRE_M="$GMR_AUTO_SOURCE_MAX_TRANS_FROM_PRE_M" \
    "$PY_GMR" - "$clip_dir" <<'PY'
import json
import os
import sys
from pathlib import Path

import numpy as np


clip_dir = Path(sys.argv[1])


def env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def env_bool(name, default=True):
    value = os.environ.get(name)
    if value is None:
        return default
    return value not in {"0", "false", "False", "no", "NO"}


def first_existing(*relative_paths):
    for item in relative_paths:
        path = clip_dir / item
        if path.is_file():
            return path
    return None


def related_npz(name):
    return first_existing(name, f"_intermediate/npz/{name}")


def related_report(name):
    return first_existing(name, f"_intermediate/reports/{name}")


def choose_pre_phc(reason):
    pre = (
        related_npz("001_contact_stabilized.npz")
        or related_npz("001_smoothed.npz")
        or related_npz("001_converted.npz")
    )
    if pre is not None:
        return pre, reason
    phc = choose_phc_candidate()
    if phc is not None:
        return phc, f"{reason}; no_pre_fallback"
    return None, "no_source"


def choose_phc_candidate():
    return (
        related_npz("001_phc.npz")
        or related_npz("001_phc_smoothed_grounded.npz")
        or related_npz("001_phc_grounded.npz")
        or related_npz("001_phc_smoothed.npz")
    )


def rotvec_to_quat(rotvec):
    rv = np.asarray(rotvec, dtype=np.float64).reshape(-1, 3)
    angle = np.linalg.norm(rv, axis=1)
    half = 0.5 * angle
    scale = np.empty_like(angle)
    small = angle < 1e-8
    scale[~small] = np.sin(half[~small]) / angle[~small]
    scale[small] = 0.5 - angle[small] ** 2 / 48.0
    quat = np.empty((rv.shape[0], 4), dtype=np.float64)
    quat[:, :3] = rv * scale[:, None]
    quat[:, 3] = np.cos(half)
    return quat


def quat_apply(quat, vector):
    qv = quat[:, :3]
    qw = quat[:, 3:4]
    vec = np.broadcast_to(np.asarray(vector, dtype=np.float64), qv.shape)
    cross = 2.0 * np.cross(qv, vec)
    return vec + qw * cross + np.cross(qv, cross)


def vector_angle_deg(a, b):
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
    dot = np.clip(np.sum(a * b, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def phc_vs_pre_checks(phc_path, pre_path):
    if phc_path is None or pre_path is None:
        return []
    failures = []
    max_tilt_p95 = env_float("GMR_AUTO_SOURCE_MAX_ROOT_TILT_P95_DEG", 35.0)
    max_tilt = env_float("GMR_AUTO_SOURCE_MAX_ROOT_TILT_DEG", 60.0)
    max_trans = env_float("GMR_AUTO_SOURCE_MAX_TRANS_FROM_PRE_M", 0.75)
    try:
        with np.load(phc_path, allow_pickle=True) as phc, np.load(pre_path, allow_pickle=True) as pre:
            if "root_orient" in phc.files and "root_orient" in pre.files:
                count = min(len(phc["root_orient"]), len(pre["root_orient"]))
                if count > 0:
                    phc_up = quat_apply(rotvec_to_quat(phc["root_orient"][:count]), [0.0, 1.0, 0.0])
                    pre_up = quat_apply(rotvec_to_quat(pre["root_orient"][:count]), [0.0, 1.0, 0.0])
                    tilt = vector_angle_deg(phc_up, pre_up)
                    tilt_p95 = float(np.percentile(tilt, 95))
                    tilt_max = float(np.max(tilt))
                    if tilt_p95 > max_tilt_p95:
                        failures.append(f"root_tilt_p95={tilt_p95:.2f}>{max_tilt_p95:.2f}")
                    if tilt_max > max_tilt:
                        failures.append(f"root_tilt_max={tilt_max:.2f}>{max_tilt:.2f}")
            if "trans" in phc.files and "trans" in pre.files:
                count = min(len(phc["trans"]), len(pre["trans"]))
                if count > 0:
                    delta = np.linalg.norm(
                        np.asarray(phc["trans"][:count], dtype=np.float64)
                        - np.asarray(pre["trans"][:count], dtype=np.float64),
                        axis=1,
                    )
                    delta_max = float(np.max(delta))
                    if delta_max > max_trans:
                        failures.append(f"trans_from_pre={delta_max:.3f}>{max_trans:.3f}")
    except Exception as exc:
        failures.append(f"compare_failed={exc}")
    return failures


def report_checks():
    report = related_report("001_phc_smoothed_report.json")
    if report is None:
        return ["missing_phc_smooth_report"]
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"bad_phc_smooth_report={exc}"]
    checks = [
        (
            "bad_trans",
            float(payload.get("bad_trans_frame_ratio", 1.0)),
            env_float("GMR_AUTO_SOURCE_MAX_BAD_TRANS_RATIO", 0.03),
        ),
        (
            "bad_joint",
            float(payload.get("bad_joint_frame_ratio", 1.0)),
            env_float("GMR_AUTO_SOURCE_MAX_BAD_JOINT_RATIO", 0.18),
        ),
        (
            "trans_delta",
            float(payload.get("trans_delta_m", {}).get("max", 1e9)),
            env_float("GMR_AUTO_SOURCE_MAX_TRANS_DELTA_M", 0.25),
        ),
    ]
    return [f"{name}={value:.4f}>{limit:.4f}" for name, value, limit in checks if value > limit]


phc_path = choose_phc_candidate()
if not env_bool("GMR_AUTO_SOURCE_QUALITY_GATE", True):
    source = phc_path or choose_pre_phc("quality_gate_off")[0]
    if source is None:
        sys.exit(0)
    print(f"{source}\tquality_gate_off")
    sys.exit(0)

if phc_path is None:
    source, reason = choose_pre_phc("missing_phc")
else:
    pre_path, _ = choose_pre_phc("pre_for_comparison")
    failures = report_checks() + phc_vs_pre_checks(phc_path, pre_path)
    if failures:
        source, reason = choose_pre_phc("phc_rejected:" + ",".join(failures[:4]))
    else:
        source, reason = phc_path, "phc_pass"

if source is not None:
    print(f"{source.resolve()}\t{reason}")
PY
}

if [ ! -d "$SHOW_ROOT" ]; then
    log "FATAL: SHOW_ROOT not found: $SHOW_ROOT"
    exit 1
fi
if [ ! -x "$PY_GMR" ]; then
    log "FATAL: PY_GMR is not executable: $PY_GMR"
    log "Set PY_GMR=/path/to/python if your GMR environment is elsewhere."
    exit 1
fi
if [ ! -d "$GMR_BODY_MODEL_PATH" ]; then
    log "FATAL: GMR_BODY_MODEL_PATH not found: $GMR_BODY_MODEL_PATH"
    exit 1
fi

PATTERN="*/${SOURCE_FILE}"
if [ "$GMR_SOURCE" = "auto" ]; then
    AUTO_SRC="${OUT_ROOT}/_auto_src"
    rm -rf "$AUTO_SRC"
    mkdir -p "$AUTO_SRC"
    while IFS= read -r -d '' clip_dir; do
        name="$(basename "$clip_dir")"
        if ! clip_in_list "$name" "$GMR_INCLUDE_CLIPS"; then
            continue
        fi
        mkdir -p "$AUTO_SRC/$name"
        choice="$(choose_auto_source "$clip_dir" || true)"
        src="${choice%%$'\t'*}"
        reason="${choice#*$'\t'}"
        if [ -z "$src" ]; then
            reason="shell_fallback"
            for candidate in \
                001_phc.npz \
                001_phc_smoothed_grounded.npz \
                001_phc_grounded.npz \
                001_phc_smoothed.npz \
                001_contact_stabilized.npz \
                001_smoothed.npz \
                001_converted.npz; do
                if [ -f "$clip_dir/$candidate" ]; then
                    src="$clip_dir/$candidate"
                    break
                fi
                if [ -f "$clip_dir/_intermediate/npz/$candidate" ]; then
                    src="$clip_dir/_intermediate/npz/$candidate"
                    break
                fi
            done
        fi
        if [ -n "$src" ]; then
            src="$(readlink -f "$src")"
            ln -sf "$src" "$AUTO_SRC/$name/selected.npz"
            log "  auto_source ${name}: $(basename "$src") (${reason})"
        fi
    done < <(find "$SHOW_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
    SHOW_ROOT="$AUTO_SRC"
    PATTERN="*/selected.npz"
fi

mapfile -t SOURCES < <(find "$SHOW_ROOT" -mindepth 2 -maxdepth 2 \( -type f -o -type l \) -name "$(basename "$PATTERN")" | sort)
if [ -n "$GMR_INCLUDE_CLIPS" ]; then
    FILTERED_SOURCES=()
    for source in "${SOURCES[@]}"; do
        clip="$(source_clip_name "$source")"
        if clip_in_list "$clip" "$GMR_INCLUDE_CLIPS"; then
            FILTERED_SOURCES+=("$source")
        fi
    done
    SOURCES=("${FILTERED_SOURCES[@]}")
fi
if [ "${#SOURCES[@]}" -eq 0 ]; then
    log "FATAL: no source motions found with pattern ${SHOW_ROOT}/${PATTERN}"
    exit 1
fi

if [ "$GMR_AUTO_HAND_NPZ" = "1" ]; then
    MISSING_HAND_SIDECARS=()
    for source in "${SOURCES[@]}"; do
        hand_sidecar="$(resolve_hand_sidecar_for_source "$source" || true)"
        if [ ! -f "$hand_sidecar" ]; then
            MISSING_HAND_SIDECARS+=("$hand_sidecar")
        fi
    done
    if [ "${#MISSING_HAND_SIDECARS[@]}" -gt 0 ]; then
        log "FATAL: GMR_AUTO_HAND_NPZ=1 but hand sidecars are missing:"
        for hand_sidecar in "${MISSING_HAND_SIDECARS[@]}"; do
            log "  $hand_sidecar"
        done
        exit 1
    fi
fi

GMR_SELECTED_CLIPS=""
for source in "${SOURCES[@]}"; do
    clip="$(source_clip_name "$source")"
    GMR_SELECTED_CLIPS="${GMR_SELECTED_CLIPS:+$GMR_SELECTED_CLIPS,}$clip"
done

log "GMR batch"
log "  show_root:        $SHOW_ROOT"
log "  source:           $GMR_SOURCE ($PATTERN)"
log "  output:           $OUT_ROOT"
log "  embodiment:       $GMR_EMBODIMENT_LABEL"
log "  body_asset:       $GMR_ROBOT"
log "  auto_hand_npz:    $GMR_AUTO_HAND_NPZ"
if [ "$GMR_BRAINCO_HANDS" = "1" ]; then
    log "  hand_model:       BrainCo Revo2 (6 motors/hand, hardware mimic constrained)"
    log "  built_in_hand:    disabled as expected (mode=$GMR_HAND_RETARGET_MODE)"
elif [ "$GMR_SHARPA_HANDS" = "1" ]; then
    log "  hand_model:       Sharpa Wave 22-DoF (morphology-matched chain IK)"
    log "  built_in_hand:    disabled as expected (mode=$GMR_HAND_RETARGET_MODE)"
else
    log "  hand_retarget:    $GMR_HAND_RETARGET_MODE"
fi
log "  hand_invalid:     $GMR_HAND_INVALID_MODE"
log "  hand_interp_gap:  $GMR_HAND_INTERP_MAX_GAP"
log "  hand_wrist_ori:   $GMR_HAND_WRIST_ORIENTATION_MODE"
log "  palm_roll:        $GMR_PALM_ROLL_MODE gain=$GMR_PALM_ROLL_GAIN max_abs=$GMR_PALM_ROLL_MAX_ABS branch=$GMR_PALM_ROLL_BRANCH_MODE body_align=$GMR_PALM_ROLL_BODY_ALIGN"
log "  wrist_mode:       $GMR_WRIST_PITCH_YAW_STABILIZE soft_margin=$GMR_WRIST_PITCH_YAW_SOFT_MARGIN_RATIO"
log "  sharpa_hands:     $GMR_SHARPA_HANDS auto=$GMR_SHARPA_AUTO_RETARGET"
log "  brainco_hands:    $GMR_BRAINCO_HANDS auto=$GMR_BRAINCO_AUTO_RETARGET"
if [ "$GMR_SHARPA_HANDS" = "1" ]; then
    log "  sharpa_mount:     pos=$GMR_SHARPA_MOUNT_POS left_pos=$GMR_SHARPA_LEFT_MOUNT_POS right_pos=$GMR_SHARPA_RIGHT_MOUNT_POS left=$GMR_SHARPA_LEFT_MOUNT_QUAT right=$GMR_SHARPA_RIGHT_MOUNT_QUAT"
    log "  sharpa_confidence: target_floor=$GMR_SHARPA_MIN_TARGET_CONFIDENCE_SCALE reproj_ratio=$GMR_SHARPA_REPROJ_GOOD_RATIO..$GMR_SHARPA_REPROJ_BAD_RATIO"
    log "  sharpa_temporal:  cost=$GMR_SHARPA_TEMPORAL_COST low_conf_gain=$GMR_SHARPA_LOW_CONF_TEMPORAL_GAIN max_delta=$GMR_SHARPA_MAX_DELTA max_accel=$GMR_SHARPA_MAX_ACCEL low_conf_accel=$GMR_SHARPA_LOW_CONF_ACCEL_SCALE"
    log "  sharpa_anatomic:  pip<=$GMR_SHARPA_MAX_PIP_BEND_DEG dip<=$GMR_SHARPA_MAX_DIP_BEND_DEG source_delta<=$GMR_SHARPA_MAX_SOURCE_BONE_DELTA_DEG bone_length_ratio<=$GMR_SHARPA_MAX_SOURCE_BONE_LENGTH_RATIO"
fi
if [ "$GMR_BRAINCO_HANDS" = "1" ]; then
    log "  brainco_asset:    $GMR_BRAINCO_ROOT"
    log "  brainco_mount:    left=$GMR_BRAINCO_LEFT_MOUNT_QUAT right=$GMR_BRAINCO_RIGHT_MOUNT_QUAT"
fi
log "  object_proxy:     $GMR_OBJECT_PROXY source=$GMR_OBJECT_PROXY_SOURCE ${GMR_OBJECT_MOTION_NAME:-<none>}"
log "  height_mode:      $GMR_HEIGHT_ADJUST_MODE"
log "  camera_source:    $GMR_CAMERA_SOURCE"
log "  relax_orient:     ${GMR_RELAX_ORIENTATION_BODIES:-<none>}"
log "  include_clips:    ${GMR_SELECTED_CLIPS:-<all>}"
log "  python:           $PY_GMR"
log "  body_model_path:  $GMR_BODY_MODEL_PATH"
log "  clips:            ${#SOURCES[@]}"

OVERRIDE_ARG=()
[ "$GMR_OVERRIDE" = "1" ] && OVERRIDE_ARG=(--override)

HAND_ARGS=()
if [ "$GMR_AUTO_HAND_NPZ" = "1" ]; then
    HAND_ARGS=(--auto_hand_npz --hand_npz_name "$GMR_HAND_NPZ_NAME")
fi

FORCE_WRIST_OVERRIDE_ARG=()
[ "$GMR_FORCE_HAND_WRIST_ORIENTATION_OVERRIDE" = "1" ] && FORCE_WRIST_OVERRIDE_ARG=(--force_hand_wrist_orientation_override)

(
    cd "$SCRIPT_DIR"
    PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" run_logged "$PY_GMR" scripts/smpl_npz_to_robot_headless.py \
        --src_root "$SHOW_ROOT" \
        --tgt_root "$OUT_ROOT" \
        --pattern "$PATTERN" \
        --include_clips "$GMR_SELECTED_CLIPS" \
        --output_name "$GMR_OUTPUT_NAME" \
        --robot "$GMR_ROBOT" \
        --body_model_path "$GMR_BODY_MODEL_PATH" \
        --model_type "$GMR_MODEL_TYPE" \
        --target_fps "$GMR_TARGET_FPS" \
        --solver "$GMR_SOLVER" \
        --coord_transform gvhmr \
        --human_yaw_offset_deg "$GMR_HUMAN_YAW_OFFSET_DEG" \
        --relax_orientation_bodies "$GMR_RELAX_ORIENTATION_BODIES" \
        --height_adjust_mode "$GMR_HEIGHT_ADJUST_MODE" \
        --ground_offset "$GMR_GROUND_OFFSET" \
        --smooth_window "$GMR_SMOOTH_WINDOW" \
        --hand_smooth_window "$GMR_HAND_SMOOTH_WINDOW" \
        --hand_smooth_polyorder "$GMR_HAND_SMOOTH_POLYORDER" \
        --hand_median_window "$GMR_HAND_MEDIAN_WINDOW" \
        --hand_max_delta "$GMR_HAND_MAX_DELTA" \
        --hand_deadband "$GMR_HAND_DEADBAND" \
        --hand_angle_scale "$GMR_HAND_ANGLE_SCALE" \
        --hand_curl_power "$GMR_HAND_CURL_POWER" \
        --hand_retarget_mode "$GMR_HAND_RETARGET_MODE" \
        --hand_invalid_mode "$GMR_HAND_INVALID_MODE" \
        --hand_interp_max_gap "$GMR_HAND_INTERP_MAX_GAP" \
        --hand_wrist_orientation_mode "$GMR_HAND_WRIST_ORIENTATION_MODE" \
        --palm_roll_mode "$GMR_PALM_ROLL_MODE" \
        --palm_roll_gain "$GMR_PALM_ROLL_GAIN" \
        --palm_roll_source "$GMR_PALM_ROLL_SOURCE" \
        --palm_roll_normal_dot_min "$GMR_PALM_ROLL_NORMAL_DOT_MIN" \
        --palm_roll_body_align "$GMR_PALM_ROLL_BODY_ALIGN" \
        --palm_roll_body_align_dot_min "$GMR_PALM_ROLL_BODY_ALIGN_DOT_MIN" \
        --left_palm_roll_sign "$GMR_LEFT_PALM_ROLL_SIGN" \
        --right_palm_roll_sign "$GMR_RIGHT_PALM_ROLL_SIGN" \
        --palm_roll_smooth_window "$GMR_PALM_ROLL_SMOOTH_WINDOW" \
        --palm_roll_max_delta "$GMR_PALM_ROLL_MAX_DELTA" \
        --palm_roll_max_abs "$GMR_PALM_ROLL_MAX_ABS" \
        --palm_roll_branch_mode "$GMR_PALM_ROLL_BRANCH_MODE" \
        --palm_roll_branch_candidates "$GMR_PALM_ROLL_BRANCH_CANDIDATES" \
        --palm_roll_branch_anchor_frames "$GMR_PALM_ROLL_BRANCH_ANCHOR_FRAMES" \
        --palm_roll_branch_transition_weight "$GMR_PALM_ROLL_BRANCH_TRANSITION_WEIGHT" \
        --palm_roll_branch_penalty "$GMR_PALM_ROLL_BRANCH_PENALTY" \
        --palm_roll_branch_anchor_penalty "$GMR_PALM_ROLL_BRANCH_ANCHOR_PENALTY" \
        --palm_roll_branch_range_penalty "$GMR_PALM_ROLL_BRANCH_RANGE_PENALTY" \
        --wrist_pitch_yaw_stabilize "$GMR_WRIST_PITCH_YAW_STABILIZE" \
        --wrist_pitch_neutral "$GMR_WRIST_PITCH_NEUTRAL" \
        --wrist_yaw_neutral "$GMR_WRIST_YAW_NEUTRAL" \
        --wrist_pitch_yaw_soft_margin_ratio "$GMR_WRIST_PITCH_YAW_SOFT_MARGIN_RATIO" \
        --retarget_mode "$GMR_RETARGET_MODE" \
        --upper_body_root_mode "$GMR_UPPER_BODY_ROOT_MODE" \
        "${HAND_ARGS[@]}" \
        "${FORCE_WRIST_OVERRIDE_ARG[@]}" \
        "${OVERRIDE_ARG[@]}"
)

if [ "$GMR_SHARPA_HANDS" = "1" ] && [ "$GMR_SHARPA_AUTO_RETARGET" = "1" ]; then
    log "Generating morphology-matched Sharpa hand trajectories"
    SHOW_ROOT="$ORIGINAL_SHOW_ROOT" \
    OUT_ROOT="$OUT_ROOT" \
    HAND_NPZ_NAME="$GMR_HAND_NPZ_NAME" \
    SHARPA_OUT_NAME="$GMR_SHARPA_HAND_NPZ_NAME" \
    SHARPA_INCLUDE_CLIPS="$GMR_SELECTED_CLIPS" \
    SHARPA_OVERRIDE="$GMR_OVERRIDE" \
    SHARPA_SCALE="$GMR_SHARPA_SCALE" \
    SHARPA_STEPS="$GMR_SHARPA_STEPS" \
    SHARPA_INIT_STEPS="$GMR_SHARPA_INIT_STEPS" \
    SHARPA_WRIST_POS_COST="$GMR_SHARPA_WRIST_POS_COST" \
    SHARPA_WRIST_ORI_COST="$GMR_SHARPA_WRIST_ORI_COST" \
    SHARPA_FINGER_POS_COST="$GMR_SHARPA_FINGER_POS_COST" \
    SHARPA_JOINT_POS_COST="$GMR_SHARPA_JOINT_POS_COST" \
    SHARPA_POSTURE_COST="$GMR_SHARPA_POSTURE_COST" \
    SHARPA_TEMPORAL_COST="$GMR_SHARPA_TEMPORAL_COST" \
    SHARPA_LOW_CONF_TEMPORAL_GAIN="$GMR_SHARPA_LOW_CONF_TEMPORAL_GAIN" \
    SHARPA_MIN_TARGET_CONFIDENCE_SCALE="$GMR_SHARPA_MIN_TARGET_CONFIDENCE_SCALE" \
    SHARPA_RELIABILITY_SMOOTH_WINDOW="$GMR_SHARPA_RELIABILITY_SMOOTH_WINDOW" \
    SHARPA_LOW_CONF_RELIABILITY_THR="$GMR_SHARPA_LOW_CONF_RELIABILITY_THR" \
    SHARPA_SMOOTH_WINDOW="$GMR_SHARPA_SMOOTH_WINDOW" \
    SHARPA_MAX_DELTA="$GMR_SHARPA_MAX_DELTA" \
    SHARPA_MAX_ACCEL="$GMR_SHARPA_MAX_ACCEL" \
    SHARPA_LOW_CONF_ACCEL_SCALE="$GMR_SHARPA_LOW_CONF_ACCEL_SCALE" \
    SHARPA_REPROJ_GOOD_PX="$GMR_SHARPA_REPROJ_GOOD_PX" \
    SHARPA_REPROJ_BAD_PX="$GMR_SHARPA_REPROJ_BAD_PX" \
    SHARPA_REPROJ_GOOD_RATIO="$GMR_SHARPA_REPROJ_GOOD_RATIO" \
    SHARPA_REPROJ_BAD_RATIO="$GMR_SHARPA_REPROJ_BAD_RATIO" \
    SHARPA_MAX_PIP_BEND_DEG="$GMR_SHARPA_MAX_PIP_BEND_DEG" \
    SHARPA_MAX_DIP_BEND_DEG="$GMR_SHARPA_MAX_DIP_BEND_DEG" \
    SHARPA_MAX_SOURCE_BONE_DELTA_DEG="$GMR_SHARPA_MAX_SOURCE_BONE_DELTA_DEG" \
    SHARPA_MAX_SOURCE_BONE_LENGTH_RATIO="$GMR_SHARPA_MAX_SOURCE_BONE_LENGTH_RATIO" \
    SHARPA_ANATOMIC_REPAIR_MAX_GAP="$GMR_SHARPA_ANATOMIC_REPAIR_MAX_GAP" \
    PY_GMR="$PY_GMR" \
    bash "$SCRIPT_DIR/run_sharpa_hand_batch.sh"
fi

if [ "$GMR_BRAINCO_HANDS" = "1" ] && [ "$GMR_BRAINCO_AUTO_RETARGET" = "1" ]; then
    if [ ! -f "$GMR_BRAINCO_ROOT/mjcf/revo2_left.xml" ] || \
       [ ! -f "$GMR_BRAINCO_ROOT/mjcf/revo2_right.xml" ]; then
        log "FATAL: BrainCo Revo2 assets are missing under $GMR_BRAINCO_ROOT"
        log "Run: bash GMR-master/scripts/setup_robot_hand_assets.sh brainco"
        exit 1
    fi
    log "Generating hardware-coupled BrainCo Revo2 hand trajectories"
    SHOW_ROOT="$ORIGINAL_SHOW_ROOT" \
    OUT_ROOT="$OUT_ROOT" \
    HAND_NPZ_NAME="$GMR_HAND_NPZ_NAME" \
    BRAINCO_OUT_NAME="$GMR_BRAINCO_HAND_NPZ_NAME" \
    BRAINCO_ROOT="$GMR_BRAINCO_ROOT" \
    BRAINCO_INCLUDE_CLIPS="$GMR_SELECTED_CLIPS" \
    BRAINCO_OVERRIDE="$GMR_OVERRIDE" \
    BRAINCO_SCALE="$GMR_SHARPA_SCALE" \
    BRAINCO_STEPS="$GMR_SHARPA_STEPS" \
    BRAINCO_INIT_STEPS="$GMR_SHARPA_INIT_STEPS" \
    BRAINCO_WRIST_POS_COST="$GMR_SHARPA_WRIST_POS_COST" \
    BRAINCO_WRIST_ORI_COST="$GMR_SHARPA_WRIST_ORI_COST" \
    BRAINCO_FINGER_POS_COST="$GMR_SHARPA_FINGER_POS_COST" \
    BRAINCO_JOINT_POS_COST="$GMR_SHARPA_JOINT_POS_COST" \
    BRAINCO_POSTURE_COST="$GMR_SHARPA_POSTURE_COST" \
    BRAINCO_TEMPORAL_COST="$GMR_SHARPA_TEMPORAL_COST" \
    BRAINCO_LOW_CONF_TEMPORAL_GAIN="$GMR_SHARPA_LOW_CONF_TEMPORAL_GAIN" \
    BRAINCO_MIN_TARGET_CONFIDENCE_SCALE="$GMR_SHARPA_MIN_TARGET_CONFIDENCE_SCALE" \
    BRAINCO_SMOOTH_WINDOW="$GMR_SHARPA_SMOOTH_WINDOW" \
    BRAINCO_MAX_DELTA="$GMR_SHARPA_MAX_DELTA" \
    BRAINCO_MAX_ACCEL="$GMR_SHARPA_MAX_ACCEL" \
    BRAINCO_LOW_CONF_ACCEL_SCALE="$GMR_SHARPA_LOW_CONF_ACCEL_SCALE" \
    BRAINCO_REPROJ_GOOD_PX="$GMR_SHARPA_REPROJ_GOOD_PX" \
    BRAINCO_REPROJ_BAD_PX="$GMR_SHARPA_REPROJ_BAD_PX" \
    BRAINCO_REPROJ_GOOD_RATIO="$GMR_SHARPA_REPROJ_GOOD_RATIO" \
    BRAINCO_REPROJ_BAD_RATIO="$GMR_SHARPA_REPROJ_BAD_RATIO" \
    BRAINCO_MAX_PIP_BEND_DEG="$GMR_SHARPA_MAX_PIP_BEND_DEG" \
    BRAINCO_MAX_DIP_BEND_DEG="$GMR_SHARPA_MAX_DIP_BEND_DEG" \
    BRAINCO_MAX_SOURCE_BONE_DELTA_DEG="$GMR_SHARPA_MAX_SOURCE_BONE_DELTA_DEG" \
    BRAINCO_MAX_SOURCE_BONE_LENGTH_RATIO="$GMR_SHARPA_MAX_SOURCE_BONE_LENGTH_RATIO" \
    BRAINCO_ANATOMIC_REPAIR_MAX_GAP="$GMR_SHARPA_ANATOMIC_REPAIR_MAX_GAP" \
    PY_GMR="$PY_GMR" \
    bash "$SCRIPT_DIR/run_brainco_hand_batch.sh"
fi

if [ "$GMR_RENDER" = "1" ]; then
    log "Rendering robot videos"
    while IFS= read -r -d '' motion; do
        clip_root="$(dirname "$motion")"
        clip="$(basename "$clip_root")"
        if ! clip_in_list "$clip" "$GMR_SELECTED_CLIPS"; then
            continue
        fi
        video="${clip_root}/$(clip_video_name "$GMR_RENDER_NAME" "$clip")"
        if [ "$GMR_OVERRIDE" != "1" ] && [ -f "$video" ]; then
            log "[SKIP] render exists: $video"
            continue
        fi
        CAMERA_ARGS=()
        if [ "$GMR_CAMERA_SOURCE" = "gvhmr" ]; then
            CAMERA_PATH="${ORIGINAL_SHOW_ROOT}/${clip}/gvhmr_camera.npz"
            CAMERA_FALLBACK="${ORIGINAL_SHOW_ROOT}/${clip}/_intermediate/npz/gvhmr_camera.npz"
            if [ ! -f "$CAMERA_PATH" ] && [ -f "$CAMERA_FALLBACK" ]; then
                CAMERA_PATH="$CAMERA_FALLBACK"
            fi
            if [ -f "$CAMERA_PATH" ]; then
                CAMERA_ARGS=(
                    --camera_path "$CAMERA_PATH"
                    --camera_path_source isaac
                    --camera_yaw_offset_deg "$GMR_HUMAN_YAW_OFFSET_DEG"
                    --camera_fov_from_path
                    --camera_static
                )
                if [ "$GMR_CAMERA_SUBJECT_ALIGN_XY" = "1" ]; then
                    CAMERA_ARGS+=(--camera_subject_align_xy)
                fi
            else
                log "[WARN] GVHMR camera not found for ${clip}, falling back to fixed camera"
            fi
        fi
        (
            cd "$SCRIPT_DIR"
            ROBOT_RGBA_ARGS=()
            [ -n "$GMR_ROBOT_RGBA" ] && ROBOT_RGBA_ARGS=(--robot_rgba "$GMR_ROBOT_RGBA")
            BACKGROUND_ARGS=()
            [ -n "$GMR_BACKGROUND_RGB" ] && BACKGROUND_ARGS=(--background_rgb "$GMR_BACKGROUND_RGB" --background_threshold "$GMR_BACKGROUND_THRESHOLD")
            SHARPA_ARGS=()
            if [ "$GMR_SHARPA_HANDS" = "1" ]; then
                SHARPA_HAND_NPZ="${clip_root}/${GMR_SHARPA_HAND_NPZ_NAME}"
                if [ ! -f "$SHARPA_HAND_NPZ" ]; then
                    SHARPA_HAND_NPZ="${ORIGINAL_SHOW_ROOT}/${clip}/${GMR_SHARPA_HAND_NPZ_NAME}"
                fi
                if [ -f "$SHARPA_HAND_NPZ" ]; then
                    SHARPA_ARGS=(
                        --sharpa_hand_npz "$SHARPA_HAND_NPZ"
                        --sharpa_scale "$GMR_SHARPA_RENDER_SCALE"
                        --sharpa_mount_pos "$GMR_SHARPA_MOUNT_POS"
                        --sharpa_left_mount_pos "$GMR_SHARPA_LEFT_MOUNT_POS"
                        --sharpa_right_mount_pos "$GMR_SHARPA_RIGHT_MOUNT_POS"
                        --sharpa_mount_quat "$GMR_SHARPA_MOUNT_QUAT"
                        --sharpa_left_mount_quat "$GMR_SHARPA_LEFT_MOUNT_QUAT"
                        --sharpa_right_mount_quat "$GMR_SHARPA_RIGHT_MOUNT_QUAT"
                    )
                else
                    log "[WARN] Sharpa hand NPZ not found for ${clip}; rendering body only"
                fi
            fi
            BRAINCO_ARGS=()
            if [ "$GMR_BRAINCO_HANDS" = "1" ]; then
                BRAINCO_HAND_NPZ="${clip_root}/${GMR_BRAINCO_HAND_NPZ_NAME}"
                if [ ! -f "$BRAINCO_HAND_NPZ" ]; then
                    BRAINCO_HAND_NPZ="${ORIGINAL_SHOW_ROOT}/${clip}/${GMR_BRAINCO_HAND_NPZ_NAME}"
                fi
                if [ -f "$BRAINCO_HAND_NPZ" ]; then
                    BRAINCO_ARGS=(
                        --brainco_hand_npz "$BRAINCO_HAND_NPZ"
                        --brainco_root "$GMR_BRAINCO_ROOT"
                        --brainco_left_mount_pos "$GMR_BRAINCO_LEFT_MOUNT_POS"
                        --brainco_right_mount_pos "$GMR_BRAINCO_RIGHT_MOUNT_POS"
                        --brainco_left_mount_quat "$GMR_BRAINCO_LEFT_MOUNT_QUAT"
                        --brainco_right_mount_quat "$GMR_BRAINCO_RIGHT_MOUNT_QUAT"
                    )
                else
                    log "[WARN] BrainCo hand NPZ not found for ${clip}; rendering body only"
                fi
            fi
            OBJECT_ARGS=()
            OBJECT_MOTION_PATH=""
            if [ "$GMR_OBJECT_PROXY" = "1" ]; then
                OBJECT_MOTION_PATH="${clip_root}/${GMR_OBJECT_MOTION_NAME}"
                if [ "$GMR_OVERRIDE" = "1" ] || [ ! -f "$OBJECT_MOTION_PATH" ]; then
                    if [ "$GMR_OBJECT_PROXY_SOURCE" = "smpl" ]; then
                        OBJECT_SOURCE_NPZ="${ORIGINAL_SHOW_ROOT}/${clip}/${SOURCE_FILE}"
                        if [ ! -f "$OBJECT_SOURCE_NPZ" ]; then
                            for candidate in \
                                001_phc.npz \
                                001_phc_smoothed_grounded.npz \
                                001_phc_grounded.npz \
                                001_phc_smoothed.npz \
                                001_contact_stabilized.npz \
                                001_smoothed.npz \
                                001_converted.npz \
                                selected.npz; do
                                if [ -f "${ORIGINAL_SHOW_ROOT}/${clip}/${candidate}" ]; then
                                    OBJECT_SOURCE_NPZ="${ORIGINAL_SHOW_ROOT}/${clip}/${candidate}"
                                    break
                                fi
                            done
                        fi
                        OBJECT_HAND_NPZ="${ORIGINAL_SHOW_ROOT}/${clip}/${GMR_HAND_NPZ_NAME}"
                        HAND_SOURCE_ARG=()
                        [ -f "$OBJECT_HAND_NPZ" ] && HAND_SOURCE_ARG=(--hand_npz "$OBJECT_HAND_NPZ")
                        if [ -f "$OBJECT_SOURCE_NPZ" ]; then
                            PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" run_logged "$PY_GMR" scripts/build_object_proxy_from_smpl_motion.py \
                                --source_npz "$OBJECT_SOURCE_NPZ" \
                                --robot_motion_path "$motion" \
                                --output "$OBJECT_MOTION_PATH" \
                                --body_model_path "$GMR_BODY_MODEL_PATH" \
                                --model_type "$GMR_MODEL_TYPE" \
                                --coord_transform gvhmr \
                                --human_yaw_offset_deg "$GMR_HUMAN_YAW_OFFSET_DEG" \
                                --object_type "$GMR_OBJECT_TYPE" \
                                --geom_size "$GMR_OBJECT_GEOM_SIZE" \
                                --offset "$GMR_OBJECT_OFFSET" \
                                --smooth_window "$GMR_OBJECT_SMOOTH_WINDOW" \
                                "${HAND_SOURCE_ARG[@]}"
                        else
                            log "[WARN] Source NPZ not found for SMPL object proxy: ${clip}"
                        fi
                    elif [ "$GMR_OBJECT_PROXY_SOURCE" = "robot" ]; then
                        PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" run_logged "$PY_GMR" scripts/build_object_proxy_from_robot_motion.py \
                            --robot_motion_path "$motion" \
                            --output "$OBJECT_MOTION_PATH" \
                            --object_type "$GMR_OBJECT_TYPE" \
                            --geom_size "$GMR_OBJECT_GEOM_SIZE" \
                            --offset "$GMR_OBJECT_OFFSET" \
                            --smooth_window "$GMR_OBJECT_SMOOTH_WINDOW"
                    else
                        log "[WARN] Unsupported GMR_OBJECT_PROXY_SOURCE=$GMR_OBJECT_PROXY_SOURCE"
                    fi
                fi
            elif [ -n "$GMR_OBJECT_MOTION_NAME" ]; then
                OBJECT_CANDIDATES=()
                if [[ "$GMR_OBJECT_MOTION_NAME" = /* ]]; then
                    OBJECT_CANDIDATES=("$GMR_OBJECT_MOTION_NAME")
                else
                    OBJECT_CANDIDATES=(
                        "${ORIGINAL_SHOW_ROOT}/${clip}/${GMR_OBJECT_MOTION_NAME}"
                        "${clip_root}/${GMR_OBJECT_MOTION_NAME}"
                    )
                fi
                for candidate in "${OBJECT_CANDIDATES[@]}"; do
                    if [ -f "$candidate" ]; then
                        OBJECT_MOTION_PATH="$candidate"
                        break
                    fi
                done
            fi
            if [ -n "$OBJECT_MOTION_PATH" ] && [ -f "$OBJECT_MOTION_PATH" ]; then
                OBJECT_ARGS=(--object_motion_path "$OBJECT_MOTION_PATH")
            fi
            PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" run_logged "$PY_GMR" scripts/render_robot_motion_headless.py \
                --robot "$GMR_ROBOT" \
                --robot_motion_path "$motion" \
                --video_path "$video" \
                --mode "$GMR_RENDER_MODE" \
                --mujoco_gl "$GMR_MUJOCO_GL" \
                --width "$GMR_RENDER_WIDTH" \
                --height "$GMR_RENDER_HEIGHT" \
                --radius "$GMR_RENDER_RADIUS" \
                --camera_mode "$GMR_RENDER_CAMERA_MODE" \
                --skip "$GMR_RENDER_SKIP" \
                --max_frames "$GMR_RENDER_MAX_FRAMES" \
                "${ROBOT_RGBA_ARGS[@]}" \
                "${BACKGROUND_ARGS[@]}" \
                "${SHARPA_ARGS[@]}" \
                "${BRAINCO_ARGS[@]}" \
                "${OBJECT_ARGS[@]}" \
                "${CAMERA_ARGS[@]}"
        )
    done < <(find "$OUT_ROOT" -type f -name "$GMR_OUTPUT_NAME" -print0 | sort -z)
fi

if [ "$GMR_COMPOSITE" = "1" ]; then
    COMPOSITE_SCRIPT="${SCRIPT_DIR}/scripts/render_composite_2x2.py"
    if [ ! -f "$COMPOSITE_SCRIPT" ]; then
        log "[WARN] Composite script not found: $COMPOSITE_SCRIPT"
    else
        log "Generating 2x2 composite videos"
        while IFS= read -r -d '' motion; do
            clip_root="$(dirname "$motion")"
            clip="$(basename "$clip_root")"
            if ! clip_in_list "$clip" "$GMR_SELECTED_CLIPS"; then
                continue
            fi
            gmr_video="${clip_root}/$(clip_video_name "$GMR_RENDER_NAME" "$clip")"
            # Name composite video using standard name (clip context is already in the path)
            composite_video="${clip_root}/$(clip_video_name "$GMR_COMPOSITE_NAME" "$clip")"

            if [ "$GMR_OVERRIDE" != "1" ] && [ -f "$composite_video" ]; then
                log "[SKIP] composite exists: $composite_video"
                continue
            fi

            if [ ! -f "$gmr_video" ]; then
                log "[WARN] GMR video missing for ${clip}, skip composite"
                continue
            fi

            # Locate the 3 source videos from original show directory
            SHOW_CLIP_DIR="${ORIGINAL_SHOW_ROOT}/${clip}"
            GVHMR_OUT_DIR="${SHOW_CLIP_DIR}/gvhmr_out/${clip}"
            ORIGINAL_VIDEO="${GVHMR_OUT_DIR}/0_input_video.mp4"
            if [ ! -f "$ORIGINAL_VIDEO" ]; then
                ORIGINAL_VIDEO="${GVHMR_OUT_DIR}/valid_video.mp4"
            fi
            GVHMR_INCAM="${GVHMR_OUT_DIR}/1_incam.mp4"
            GVHMR_OBJECT_INCAM="${GVHMR_OUT_DIR}/1_incam_object.mp4"
            if [ -f "$GVHMR_OBJECT_INCAM" ]; then
                GVHMR_INCAM="$GVHMR_OBJECT_INCAM"
            fi

            # Latest ISSAC GYM video from phc_renderings
            ISAAC_VIDEO=""
            if [ -d "${SHOW_CLIP_DIR}/phc_renderings" ]; then
                ISAAC_VIDEO=$(find "${SHOW_CLIP_DIR}/phc_renderings" -name "*.mp4" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
            fi

            MISSING=()
            [ -f "$ORIGINAL_VIDEO" ] || MISSING+=("original")
            [ -f "$GVHMR_INCAM" ] || MISSING+=("gvhmr_incam")
            [ -f "${ISAAC_VIDEO:-}" ] || MISSING+=("isaac")

            if [ "${#MISSING[@]}" -gt 0 ]; then
                log "[WARN] ${clip}: missing videos: ${MISSING[*]}, skip composite"
                continue
            fi

            log "  composite: ${clip} -> ${composite_video}"
            (
                cd "$SCRIPT_DIR"
                LABEL_ARGS=()
                if [ "$GMR_COMPOSITE_HIDE_LABELS" = "1" ]; then
                    LABEL_ARGS=(--hide_labels)
                fi
                PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" run_logged "$PY_GMR" \
                    "$COMPOSITE_SCRIPT" \
                    --original "$ORIGINAL_VIDEO" \
                    --gvhmr "$GVHMR_INCAM" \
                    --isaac "$ISAAC_VIDEO" \
                    --gmr "$gmr_video" \
                    --output "$composite_video" \
                    --panel_height "$GMR_COMPOSITE_PANEL_HEIGHT" \
                    --fps "$GMR_COMPOSITE_FPS" \
                    "${LABEL_ARGS[@]}"
            )
        done < <(find "$OUT_ROOT" -type f -name "$GMR_OUTPUT_NAME" -print0 | sort -z)
    fi
fi

{
    echo "clip,source,height_mode,camera_source,robot_motion,render_video,composite_video"
    while IFS= read -r -d '' motion; do
        clip="$(basename "$(dirname "$motion")")"
        if ! clip_in_list "$clip" "$GMR_SELECTED_CLIPS"; then
            continue
        fi
        echo "${clip},${GMR_SOURCE},${GMR_HEIGHT_ADJUST_MODE},${GMR_CAMERA_SOURCE},${motion},$(dirname "$motion")/$(clip_video_name "$GMR_RENDER_NAME" "$clip"),$(dirname "$motion")/$(clip_video_name "$GMR_COMPOSITE_NAME" "$clip")"
    done < <(find "$OUT_ROOT" -type f -name "$GMR_OUTPUT_NAME" -print0 | sort -z)
} > "$SUMMARY_CSV"

log "Done"
log "  summary: $SUMMARY_CSV"
