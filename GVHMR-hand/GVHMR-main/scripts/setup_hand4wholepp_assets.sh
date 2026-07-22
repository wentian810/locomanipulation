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
H4WPP_SNAPSHOT_SHA256="${H4WPP_SNAPSHOT_SHA256:-b18c956aa48835659e6dabcbf51e1286987bc192b8d545d77e58327731714ce9}"
H4WPP_DWPOSE_SHA256="${H4WPP_DWPOSE_SHA256:-e9600664e7927229ed594197d552023e3be213f810beb38847a959ec8261e0f7}"
H4WPP_WILOR_SHA256="${H4WPP_WILOR_SHA256:-3e97aafc7dd08d883a4cc5a027df61fdb6fda6136dbd1319405413862ada6bb2}"
H4WPP_DETECTOR_SHA256="${H4WPP_DETECTOR_SHA256:-5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033}"

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
    local fresh_checkout=0
    if [ -d "$H4W_ROOT/.git" ]; then
        log "Hand4Whole++ repo exists: $H4W_ROOT"
    else
        mkdir -p "$(dirname "$H4W_ROOT")"
        log "cloning Hand4Whole++ into $H4W_ROOT"
        git clone --filter=blob:none \
            https://github.com/mks0601/Hand4Whole-plus-plus_RELEASE.git \
            "$H4W_ROOT"
        fresh_checkout=1
    fi
    if [ "$fresh_checkout" != "1" ] && [ "${H4W_REFRESH_SOURCE:-0}" != "1" ]; then
        log "preserving existing Hand4Whole++ source; set H4W_REFRESH_SOURCE=1 to enforce $H4W_COMMIT and the pipeline patch"
        return 0
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

asset_sha256_matches() {
    local path="$1"
    local expected_sha256="$2"
    [ -f "$path" ] && \
        [ "$(sha256sum "$path" | cut -d' ' -f1)" = "$expected_sha256" ]
}

quarantine_invalid_asset() {
    local path="$1"
    local expected_sha256="$2"
    local label="$3"
    if [ -f "$path" ] && ! asset_sha256_matches "$path" "$expected_sha256"; then
        local invalid_path="${path}.invalid.$(date +%Y%m%d%H%M%S)"
        log "$label checksum mismatch; preserving it as $invalid_path"
        mv "$path" "$invalid_path"
    fi
}

sync_wilor_runtime_if_requested() {
    if [ "${H4W_SYNC_WILOR:-0}" != "1" ]; then
        return 0
    fi

    local source_root="$WILOR_ROOT/pretrained_models"
    local target_root="$H4W_ROOT/common/nets/WiLoR/pretrained_models"
    local source target expected label temporary
    local specs=(
        "wilor_final.ckpt:$H4WPP_WILOR_SHA256:WiLoR checkpoint"
        "detector.pt:$H4WPP_DETECTOR_SHA256:WiLoR detector"
    )
    for spec in "${specs[@]}"; do
        IFS=: read -r source expected label <<< "$spec"
        source="$source_root/$source"
        target="$target_root/$(basename "$source")"
        if ! asset_sha256_matches "$source" "$expected"; then
            log "$label is not verified in canonical WiLoR root: $source"
            return 1
        fi
        if asset_sha256_matches "$target" "$expected"; then
            log "$label runtime copy already verified: $target"
            continue
        fi
        mkdir -p "$target_root"
        quarantine_invalid_asset "$target" "$expected" "$label runtime copy"
        temporary="${target}.part.$$"
        rm -f "$temporary"
        cp "$source" "$temporary"
        if ! asset_sha256_matches "$temporary" "$expected"; then
            rm -f "$temporary"
            log "$label runtime copy failed checksum verification"
            return 1
        fi
        mv "$temporary" "$target"
        log "$label runtime copy synchronized and verified: $target"
    done
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

    local snapshot="$H4W_ROOT/demo/snapshot_6.pth"
    local dwpose="$H4W_ROOT/common/nets/mmpose/dw-ll_ucoco.pth"
    quarantine_invalid_asset "$snapshot" "$H4WPP_SNAPSHOT_SHA256" "Hand4Whole++ snapshot"
    if ! asset_sha256_matches "$snapshot" "$H4WPP_SNAPSHOT_SHA256"; then
        log "downloading Hand4Whole++ model folder"
        $gd --folder "$H4WPP_MODEL_URL" -O "$H4W_ROOT/demo" || log "Hand4Whole++ model folder download failed"
        local found_snapshot
        found_snapshot="$(find "$H4W_ROOT/demo" -name 'snapshot_6.pth' -print -quit 2>/dev/null || true)"
        if [ -n "$found_snapshot" ] && [ "$found_snapshot" != "$H4W_ROOT/demo/snapshot_6.pth" ]; then
            ln -sf "$(realpath "$found_snapshot")" "$H4W_ROOT/demo/snapshot_6.pth"
            log "linked snapshot_6.pth -> $(realpath "$found_snapshot")"
        fi
        if ! asset_sha256_matches "$snapshot" "$H4WPP_SNAPSHOT_SHA256"; then
            log "Hand4Whole++ snapshot is still missing or failed checksum verification"
            return 1
        fi
    fi

    quarantine_invalid_asset "$dwpose" "$H4WPP_DWPOSE_SHA256" "DWPose checkpoint"
    if ! asset_sha256_matches "$dwpose" "$H4WPP_DWPOSE_SHA256"; then
        local dwpose_part="${dwpose}.part.$$"
        rm -f "$dwpose_part"
        log "downloading DWPose checkpoint"
        if ! $gd "$H4WPP_DWPOSE_ID" -O "$dwpose_part"; then
            rm -f "$dwpose_part"
            log "DWPose checkpoint download failed"
            return 1
        fi
        if ! asset_sha256_matches "$dwpose_part" "$H4WPP_DWPOSE_SHA256"; then
            rm -f "$dwpose_part"
            log "DWPose checkpoint failed checksum verification"
            return 1
        fi
        mv "$dwpose_part" "$dwpose"
        log "DWPose checkpoint downloaded and verified"
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

    local snapshot="$H4W_ROOT/demo/snapshot_6.pth"
    local dwpose="$H4W_ROOT/common/nets/mmpose/dw-ll_ucoco.pth"
    local wilor="$H4W_ROOT/common/nets/WiLoR/pretrained_models/wilor_final.ckpt"
    local detector="$H4W_ROOT/common/nets/WiLoR/pretrained_models/detector.pt"
    asset_sha256_matches "$snapshot" "$H4WPP_SNAPSHOT_SHA256" || missing+=("$snapshot (missing or checksum mismatch)")
    asset_sha256_matches "$dwpose" "$H4WPP_DWPOSE_SHA256" || missing+=("$dwpose (missing or checksum mismatch)")
    asset_sha256_matches "$wilor" "$H4WPP_WILOR_SHA256" || missing+=("$wilor (missing or checksum mismatch; run scripts/setup_wilor_assets.sh and update the H4W runtime copy)")
    asset_sha256_matches "$detector" "$H4WPP_DETECTOR_SHA256" || missing+=("$detector (missing or checksum mismatch; run scripts/setup_wilor_assets.sh and update the H4W runtime copy)")

    for path in \
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

If WiLoR/detector was downloaded or repaired under third-party/WiLoR, sync the
actual Hand4Whole++ runtime copies afterwards:
  H4W_SYNC_WILOR=1 bash scripts/setup_hand4wholepp_assets.sh
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
sync_wilor_runtime_if_requested
download_hand4wholepp_if_requested || true
report_missing_official_assets
log "done"
