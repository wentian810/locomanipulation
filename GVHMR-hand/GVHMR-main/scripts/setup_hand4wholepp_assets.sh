#!/usr/bin/env bash
# Prepare Hand4Whole++ as an optional GVHMR-hand backend.
#
# This links assets already present in this workspace and reports the official
# files that still need to be downloaded manually.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_ROOT="${PIPELINE_ROOT:-$(cd "${ROOT}/../.." && pwd)}"
H4W_ROOT="${H4W_ROOT:-${ROOT}/third-party/Hand4Whole-plus-plus_RELEASE}"
H4W_COMMIT="${H4W_COMMIT:-f81d35ddd2b74206c40142243eb62b6d64ce0d65}"
H4W_PATCH="${PIPELINE_ROOT}/patches/third_party/hand4wholepp_gvhmr.patch"
WILOR_ROOT="${WILOR_ROOT:-${ROOT}/third-party/WiLoR}"
H4WPP_MODEL_URL="${H4WPP_MODEL_URL:-https://drive.google.com/drive/folders/1sDWjihPLcjJNTzQbGUedJ3zaK0AyalBt?usp=sharing}"
H4WPP_MMPOSE_URL="${H4WPP_MMPOSE_URL:-https://drive.google.com/file/d/1Rxjb9l5m49lhoxfW0ohubl19vVRx9Q_n/view?usp=sharing}"
H4WPP_DWPOSE_URL="${H4WPP_DWPOSE_URL:-https://drive.google.com/file/d/1PHKN3p873dgCSh_YRsYqTZVj-kIbclRS/view?usp=sharing}"
H4WPP_MMPOSE_ID="${H4WPP_MMPOSE_ID:-1Rxjb9l5m49lhoxfW0ohubl19vVRx9Q_n}"
H4WPP_DWPOSE_ID="${H4WPP_DWPOSE_ID:-1PHKN3p873dgCSh_YRsYqTZVj-kIbclRS}"

log() {
    printf '[setup_hand4wholepp_assets] %s\n' "$*"
}

find_first_file() {
    for path in "$@"; do
        if [ -f "$path" ]; then
            printf '%s\n' "$path"
            return 0
        fi
    done
    return 1
}

link_file() {
    local src="$1"
    local dst="$2"
    local label="$3"

    mkdir -p "$(dirname "$dst")"
    if [ -e "$dst" ] || [ -L "$dst" ]; then
        log "$label exists: $dst"
        return 0
    fi
    if [ ! -f "$src" ]; then
        log "missing $label source: $src"
        return 1
    fi
    ln -s "$(realpath "$src")" "$dst"
    log "linked $label: $dst -> $(realpath "$src")"
}

clone_hand4wholepp() {
    if [ -d "$H4W_ROOT/.git" ]; then
        log "Hand4Whole++ repo exists: $H4W_ROOT"
    else
        mkdir -p "$(dirname "$H4W_ROOT")"
        log "cloning Hand4Whole++ into $H4W_ROOT"
        git clone --filter=blob:none \
            https://github.com/mks0601/Hand4Whole-plus-plus_RELEASE.git \
            "$H4W_ROOT"
    fi
    git -C "$H4W_ROOT" checkout --detach "$H4W_COMMIT"
    if git -C "$H4W_ROOT" apply --reverse --check "$H4W_PATCH" >/dev/null 2>&1; then
        log "GVHMR Hand4Whole++ patch already applied"
    else
        git -C "$H4W_ROOT" apply --check "$H4W_PATCH"
        git -C "$H4W_ROOT" apply "$H4W_PATCH"
        log "applied GVHMR Hand4Whole++ patch"
    fi
}

link_wilor() {
    local dst="${H4W_ROOT}/common/nets/WiLoR"
    mkdir -p "$(dirname "$dst")"
    if [ -e "$dst" ] || [ -L "$dst" ]; then
        log "WiLoR target exists: $dst"
        return 0
    fi
    if [ ! -d "$WILOR_ROOT/wilor" ]; then
        log "WiLoR source missing: $WILOR_ROOT"
        log "Run: bash scripts/setup_wilor_assets.sh"
        return 1
    fi
    ln -s "$(realpath "$WILOR_ROOT")" "$dst"
    log "linked WiLoR: $dst -> $(realpath "$WILOR_ROOT")"
}

link_human_models() {
    local human_root="${H4W_ROOT}/common/utils/human_model_files"
    local smpl_neutral smplx_neutral smplx_male smplx_female mano_left mano_right
    local smplx_neutral_npz smplx_male_npz smplx_female_npz
    local missing=0

    smpl_neutral="$(find_first_file \
        "${SMPL_NEUTRAL_PATH:-}" \
        "${PIPELINE_ROOT}/GVHMR-main/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl" \
        "${PIPELINE_ROOT}/phc-dev-felix-pipeline/data/smpl/SMPL_NEUTRAL.pkl" \
        "${PIPELINE_ROOT}/gvhmr/.cache/4DHumans/data/smpl/SMPL_NEUTRAL.pkl" \
        "${PIPELINE_ROOT}/gvhmr/cache/4DHumans/data/smpl/SMPL_NEUTRAL.pkl" || true)"
    smplx_neutral="$(find_first_file \
        "${SMPLX_NEUTRAL_PATH:-}" \
        "${PIPELINE_ROOT}/GVHMR-main/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.pkl" || true)"
    smplx_male="$(find_first_file \
        "${SMPLX_MALE_PATH:-}" \
        "${PIPELINE_ROOT}/GVHMR-main/inputs/checkpoints/body_models/smplx/SMPLX_MALE.pkl" || true)"
    smplx_female="$(find_first_file \
        "${SMPLX_FEMALE_PATH:-}" \
        "${PIPELINE_ROOT}/GVHMR-main/inputs/checkpoints/body_models/smplx/SMPLX_FEMALE.pkl" || true)"
    smplx_neutral_npz="$(find_first_file \
        "${SMPLX_NEUTRAL_NPZ_PATH:-}" \
        "${PIPELINE_ROOT}/GVHMR-main/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz" \
        "${HOME}/下载/models_smplx_v1_1(1)/models/smplx/SMPLX_NEUTRAL.npz" || true)"
    smplx_male_npz="$(find_first_file \
        "${SMPLX_MALE_NPZ_PATH:-}" \
        "${PIPELINE_ROOT}/GVHMR-main/inputs/checkpoints/body_models/smplx/SMPLX_MALE.npz" \
        "${HOME}/下载/models_smplx_v1_1(1)/models/smplx/SMPLX_MALE.npz" || true)"
    smplx_female_npz="$(find_first_file \
        "${SMPLX_FEMALE_NPZ_PATH:-}" \
        "${PIPELINE_ROOT}/GVHMR-main/inputs/checkpoints/body_models/smplx/SMPLX_FEMALE.npz" \
        "${HOME}/下载/models_smplx_v1_1(1)/models/smplx/SMPLX_FEMALE.npz" || true)"
    mano_left="$(find_first_file \
        "${MANO_LEFT_PATH:-}" \
        "${PIPELINE_ROOT}/mano_v1_2/models/MANO_LEFT.pkl" || true)"
    mano_right="$(find_first_file \
        "${MANO_RIGHT_PATH:-}" \
        "${PIPELINE_ROOT}/mano_v1_2/models/MANO_RIGHT.pkl" \
        "${ROOT}/_DATA/data/mano/MANO_RIGHT.pkl" || true)"

    [ -n "$smpl_neutral" ] && link_file "$smpl_neutral" "$human_root/smpl/SMPL_NEUTRAL.pkl" "SMPL_NEUTRAL.pkl" || missing=1
    [ -n "$smplx_neutral" ] && link_file "$smplx_neutral" "$human_root/smplx/SMPLX_NEUTRAL.pkl" "SMPLX_NEUTRAL.pkl" || missing=1
    [ -n "$smplx_male" ] && link_file "$smplx_male" "$human_root/smplx/SMPLX_MALE.pkl" "SMPLX_MALE.pkl" || missing=1
    [ -n "$smplx_female" ] && link_file "$smplx_female" "$human_root/smplx/SMPLX_FEMALE.pkl" "SMPLX_FEMALE.pkl" || missing=1
    [ -n "$smplx_neutral_npz" ] && link_file "$smplx_neutral_npz" "$human_root/smplx/SMPLX_NEUTRAL.npz" "SMPLX_NEUTRAL.npz" || missing=1
    [ -n "$smplx_male_npz" ] && link_file "$smplx_male_npz" "$human_root/smplx/SMPLX_MALE.npz" "SMPLX_MALE.npz" || missing=1
    [ -n "$smplx_female_npz" ] && link_file "$smplx_female_npz" "$human_root/smplx/SMPLX_FEMALE.npz" "SMPLX_FEMALE.npz" || missing=1
    [ -n "$mano_left" ] && link_file "$mano_left" "$human_root/mano/MANO_LEFT.pkl" "MANO_LEFT.pkl" || missing=1
    [ -n "$mano_right" ] && link_file "$mano_right" "$human_root/mano/MANO_RIGHT.pkl" "MANO_RIGHT.pkl" || missing=1

    return "$missing"
}

link_optional_snapshot() {
    if [ -z "${H4WPP_SNAPSHOT:-}" ]; then
        return 0
    fi
    link_file "$H4WPP_SNAPSHOT" "$H4W_ROOT/demo/snapshot_6.pth" "Hand4Whole++ snapshot_6.pth"
}

link_optional_file() {
    local src="${1:-}"
    local dst="$2"
    local label="$3"
    if [ -z "$src" ]; then
        return 0
    fi
    link_file "$src" "$dst" "$label"
}

link_optional_mmpose() {
    local src="${H4WPP_MMPOSE_PATH:-}"
    local dst="${H4W_ROOT}/common/nets/mmpose"
    if [ -z "$src" ]; then
        return 0
    fi
    mkdir -p "$(dirname "$dst")"
    if [ -d "$src" ]; then
        if [ -e "$dst" ] || [ -L "$dst" ]; then
            log "mmpose target exists: $dst"
            return 0
        fi
        ln -s "$(realpath "$src")" "$dst"
        log "linked mmpose: $dst -> $(realpath "$src")"
    elif [ -f "$src" ]; then
        if command -v unzip >/dev/null 2>&1; then
            unzip -q -o "$src" -d "${H4W_ROOT}/common/nets"
            log "unpacked mmpose bundle: $src"
        else
            log "unzip is missing; unpack manually: $src -> $dst"
        fi
    else
        log "missing mmpose source: $src"
        return 1
    fi
}

link_optional_manual_assets() {
    local human_root="${H4W_ROOT}/common/utils/human_model_files"
    link_optional_file "${H4WPP_DWPOSE_PATH:-}" "${H4W_ROOT}/common/nets/mmpose/dw-ll_ucoco.pth" "DWPose checkpoint"
    link_optional_mmpose
    link_optional_file "${H4WPP_MANO_SMPLX_VERTEX_IDS:-}" "$human_root/smplx/MANO_SMPLX_vertex_ids.pkl" "MANO_SMPLX_vertex_ids.pkl"
    link_optional_file "${H4WPP_SMPLX_FLAME_VERTEX_IDS:-}" "$human_root/smplx/SMPL-X__FLAME_vertex_ids.npy" "SMPL-X__FLAME_vertex_ids.npy"
    link_optional_file "${H4WPP_SMPLX_TO_J14:-}" "$human_root/smplx/SMPLX_to_J14.pkl" "SMPLX_to_J14.pkl"
}

gdown_cmd() {
    if command -v gdown >/dev/null 2>&1; then
        printf '%s\n' "gdown"
        return 0
    fi
    if python - <<'PY' >/dev/null 2>&1
import gdown
PY
    then
        printf '%s\n' "python -m gdown"
        return 0
    fi
    return 1
}

download_hand4wholepp_if_requested() {
    if [ "${DOWNLOAD_H4WPP:-0}" != "1" ]; then
        return 0
    fi

    local gd
    if ! gd="$(gdown_cmd)"; then
        cat <<EOF
gdown is required for DOWNLOAD_H4WPP=1.
Install it in the active environment, for example:
  pip install gdown
EOF
        return 1
    fi

    mkdir -p "$H4W_ROOT/demo" "$H4W_ROOT/common/nets/mmpose"

    if [ ! -f "$H4W_ROOT/demo/snapshot_6.pth" ]; then
        log "downloading Hand4Whole++ model folder"
        $gd --folder "$H4WPP_MODEL_URL" -O "$H4W_ROOT/demo" || log "Hand4Whole++ model folder download failed"
        local found_snapshot
        found_snapshot="$(find "$H4W_ROOT/demo" -name 'snapshot_6.pth' -print -quit 2>/dev/null || true)"
        if [ -n "$found_snapshot" ] && [ "$found_snapshot" != "$H4W_ROOT/demo/snapshot_6.pth" ]; then
            ln -sf "$(realpath "$found_snapshot")" "$H4W_ROOT/demo/snapshot_6.pth"
            log "linked snapshot_6.pth -> $(realpath "$found_snapshot")"
        fi
    fi

    if [ ! -f "$H4W_ROOT/common/nets/mmpose/dw-ll_ucoco.pth" ]; then
        log "downloading DWPose checkpoint"
        $gd "$H4WPP_DWPOSE_ID" -O "$H4W_ROOT/common/nets/mmpose/dw-ll_ucoco.pth" || log "DWPose checkpoint download failed"
    fi

    if [ ! -d "$H4W_ROOT/common/nets/mmpose/mmpose" ]; then
        local mmpose_zip="$H4W_ROOT/common/nets/mmpose_download.zip"
        log "downloading mmpose bundle"
        if $gd "$H4WPP_MMPOSE_ID" -O "$mmpose_zip"; then
            if command -v unzip >/dev/null 2>&1; then
                unzip -q -o "$mmpose_zip" -d "$H4W_ROOT/common/nets"
            else
                log "unzip is missing; unpack manually: $mmpose_zip -> $H4W_ROOT/common/nets/mmpose"
            fi
        elif [ -s "$mmpose_zip" ] && command -v unzip >/dev/null 2>&1; then
            unzip -q -o "$mmpose_zip" -d "$H4W_ROOT/common/nets"
        else
            log "mmpose bundle download failed"
        fi
    fi
}

report_missing_official_assets() {
    local human_root="${H4W_ROOT}/common/utils/human_model_files"
    local missing=()

    for path in \
        "$H4W_ROOT/demo/snapshot_6.pth" \
        "$H4W_ROOT/common/nets/mmpose/dw-ll_ucoco.pth" \
        "$H4W_ROOT/common/nets/mmpose/mmpose" \
        "$human_root/smplx/SMPLX_NEUTRAL.npz" \
        "$human_root/smplx/SMPLX_MALE.npz" \
        "$human_root/smplx/SMPLX_FEMALE.npz" \
        "$human_root/smplx/MANO_SMPLX_vertex_ids.pkl" \
        "$human_root/smplx/SMPLX_to_J14.pkl"
    do
        if [ ! -e "$path" ]; then
            missing+=("$path")
        fi
    done

    if [ "${#missing[@]}" -gt 0 ]; then
        cat <<EOF

Hand4Whole++ still needs these official assets:
$(printf '  %s\n' "${missing[@]}")

Download sources are listed in the official README:
  https://github.com/mks0601/Hand4Whole-plus-plus_RELEASE

Direct official links:
  Hand4Whole++ model folder:
    $H4WPP_MODEL_URL
  mmpose bundle:
    $H4WPP_MMPOSE_URL
  DWPose checkpoint:
    $H4WPP_DWPOSE_URL

Useful placement notes:
  - Put the pretrained Hand4Whole++ model at:
      $H4W_ROOT/demo/snapshot_6.pth
    or rerun this script with:
      H4WPP_SNAPSHOT=/path/to/snapshot_6.pth bash scripts/setup_hand4wholepp_assets.sh
  - Put the official mmpose folder at:
      $H4W_ROOT/common/nets/mmpose
    and DWPose at:
      $H4W_ROOT/common/nets/mmpose/dw-ll_ucoco.pth
  - Put SMPL-X mapping files under:
      $human_root/smplx

After manual downloads, you can let this script link/unpack them:
  H4WPP_SNAPSHOT=/path/to/snapshot_6.pth \\
  H4WPP_DWPOSE_PATH=/path/to/dw-ll_ucoco.pth \\
  H4WPP_MMPOSE_PATH=/path/to/mmpose_or_mmpose_zip \\
  H4WPP_MANO_SMPLX_VERTEX_IDS=/path/to/MANO_SMPLX_vertex_ids.pkl \\
  H4WPP_SMPLX_FLAME_VERTEX_IDS=/path/to/SMPL-X__FLAME_vertex_ids.npy \\
  H4WPP_SMPLX_TO_J14=/path/to/SMPLX_to_J14.pkl \\
  bash scripts/setup_hand4wholepp_assets.sh

To try downloading the Google Drive assets with gdown:
  DOWNLOAD_H4WPP=1 bash scripts/setup_hand4wholepp_assets.sh
EOF
    else
        log "all checked Hand4Whole++ assets are present"
    fi
}

clone_hand4wholepp
link_wilor || true
link_human_models || true
link_optional_snapshot || true
link_optional_manual_assets || true
download_hand4wholepp_if_requested || true
report_missing_official_assets
log "done"
