#!/usr/bin/env bash
# Prepare WiLoR as an optional hand backend for GVHMR-hand.
#
# Official WiLoR repo:
#   https://github.com/rolpotamias/WiLoR
#
# Usage:
#   bash scripts/setup_wilor_assets.sh
#
# Optional:
#   INSTALL_WILOR=1 bash scripts/setup_wilor_assets.sh
#   MANO_RIGHT_PATH=/path/to/MANO_RIGHT.pkl bash scripts/setup_wilor_assets.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WILOR_DIR="${WILOR_DIR:-${ROOT}/third-party/WiLoR}"
WILOR_COMMIT="${WILOR_COMMIT:-fcb911312a38fa8badd30d9656a167485d61b8f9}"
WILOR_PATCH="${PIPELINE_ROOT:-$(cd "${ROOT}/../.." && pwd)}/patches/third_party/wilor_gvhmr.patch"
WILOR_CKPT="${WILOR_CKPT:-${WILOR_DIR}/pretrained_models/wilor_final.ckpt}"
WILOR_DETECTOR="${WILOR_DETECTOR:-${WILOR_DIR}/pretrained_models/detector.pt}"
WILOR_CKPT_URL="${WILOR_CKPT_URL:-https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt}"
WILOR_DETECTOR_URL="${WILOR_DETECTOR_URL:-https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt}"
MANO_SRC="${MANO_RIGHT_PATH:-${ROOT}/_DATA/data/mano/MANO_RIGHT.pkl}"
MANO_DST="${WILOR_DIR}/mano_data/MANO_RIGHT.pkl"

log() {
    printf '[setup_wilor_assets] %s\n' "$*"
}

clone_wilor() {
    if [ -d "$WILOR_DIR/.git" ]; then
        log "WiLoR repo exists: $WILOR_DIR"
    else
        mkdir -p "$(dirname "$WILOR_DIR")"
        log "cloning WiLoR into $WILOR_DIR"
        git clone --filter=blob:none https://github.com/rolpotamias/WiLoR.git "$WILOR_DIR"
    fi
    git -C "$WILOR_DIR" checkout --detach "$WILOR_COMMIT"
    if git -C "$WILOR_DIR" apply --reverse --check "$WILOR_PATCH" >/dev/null 2>&1; then
        log "GVHMR WiLoR patch already applied"
    else
        git -C "$WILOR_DIR" apply --check "$WILOR_PATCH"
        git -C "$WILOR_DIR" apply "$WILOR_PATCH"
        log "applied GVHMR WiLoR patch"
    fi
}

download_file() {
    local url="$1"
    local output="$2"
    local label="$3"

    mkdir -p "$(dirname "$output")"
    if [ -s "$output" ]; then
        log "$label exists: $output"
        return 0
    fi

    log "downloading $label to $output"
    if command -v wget >/dev/null 2>&1; then
        wget -q --show-progress -O "$output" "$url"
    elif command -v curl >/dev/null 2>&1; then
        curl -L --progress-bar "$url" -o "$output"
    else
        echo "Need wget or curl to download WiLoR assets." >&2
        exit 1
    fi
}

download_weights() {
    download_file "$WILOR_CKPT_URL" "$WILOR_CKPT" "WiLoR checkpoint"
    download_file "$WILOR_DETECTOR_URL" "$WILOR_DETECTOR" "WiLoR detector"
}

link_mano() {
    mkdir -p "$(dirname "$MANO_DST")"
    if [ -e "$MANO_DST" ] || [ -L "$MANO_DST" ]; then
        log "MANO_RIGHT.pkl target exists: $MANO_DST"
        return 0
    fi
    if [ ! -f "$MANO_SRC" ]; then
        cat <<EOF

MANO_RIGHT.pkl is missing.
Download it manually from the official MANO site after registration:
  https://mano.is.tue.mpg.de/

Then either link it for GVHMR-hand:
  ${ROOT}/_DATA/data/mano/MANO_RIGHT.pkl

or pass it directly:
  MANO_RIGHT_PATH=/path/to/MANO_RIGHT.pkl bash scripts/setup_wilor_assets.sh
EOF
        exit 1
    fi
    ln -s "$(realpath "$MANO_SRC")" "$MANO_DST"
    log "linked MANO_RIGHT.pkl: $MANO_DST -> $(realpath "$MANO_SRC")"
}

install_wilor_if_requested() {
    if [ "${INSTALL_WILOR:-0}" != "1" ]; then
        cat <<EOF

WiLoR code and assets are ready at:
  $WILOR_DIR

The GVHMR pipeline adds this checkout to Python's import path automatically.
If dependencies are missing, install WiLoR requirements in the GVHMR Python env:
  cd "$WILOR_DIR"
  pip install -r requirements.txt
EOF
        return 0
    fi

    log "installing WiLoR requirements into the current Python environment"
    (
        cd "$WILOR_DIR"
        pip install -r requirements.txt
    )
}

clone_wilor
download_weights
link_mano
install_wilor_if_requested

log "done"
log "WiLoR checkpoint: $WILOR_CKPT"
log "WiLoR model config: ${WILOR_DIR}/pretrained_models/model_config.yaml"
