#!/usr/bin/env bash
# Prepare HaMeR code and demo assets for GVHMR-hand.
#
# Official HaMeR repo:
#   https://github.com/geopavlakos/hamer
# Official demo asset archive used by HaMeR:
#   https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz
#
# Usage:
#   bash scripts/setup_hamer_assets.sh
#
# Optional:
#   INSTALL_HAMER=1 bash scripts/setup_hamer_assets.sh
#   MANO_RIGHT_PATH=/path/to/MANO_RIGHT.pkl bash scripts/setup_hamer_assets.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HAMER_DIR="${HAMER_DIR:-${ROOT}/third-party/hamer}"
HAMER_COMMIT="${HAMER_COMMIT:-3a01849f4148352e9260b69bf28b65d1671a4905}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${ROOT}/_downloads}"
ARCHIVE="${DOWNLOAD_DIR}/hamer_demo_data.tar.gz"
ARCHIVE_URL="${ARCHIVE_URL:-https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz}"
GDRIVE_URL="${GDRIVE_URL:-https://drive.google.com/uc?id=1mv7CUAnm73oKsEEG1xE3xH2C_oqcFSzT}"

WHOLEBODY_SRC="${ROOT}/_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth"
WHOLEBODY_DST="${ROOT}/inputs/checkpoints/vitpose/vitpose-h-coco-wholebody.pth"
HAMER_CKPT="${ROOT}/_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
MANO_DST="${ROOT}/_DATA/data/mano/MANO_RIGHT.pkl"

log() {
    printf '[setup_hamer_assets] %s\n' "$*"
}

download_archive() {
    mkdir -p "$DOWNLOAD_DIR"
    if [ -s "$ARCHIVE" ]; then
        log "archive exists: $ARCHIVE"
        return 0
    fi

    log "downloading HaMeR demo assets to $ARCHIVE"
    if command -v wget >/dev/null 2>&1; then
        wget -O "$ARCHIVE" "$ARCHIVE_URL"
    elif command -v curl >/dev/null 2>&1; then
        curl -L "$ARCHIVE_URL" -o "$ARCHIVE"
    elif command -v gdown >/dev/null 2>&1; then
        gdown "$GDRIVE_URL" -O "$ARCHIVE"
    else
        echo "Need wget, curl, or gdown to download HaMeR assets." >&2
        exit 1
    fi
}

clone_hamer() {
    if [ -d "$HAMER_DIR/.git" ]; then
        log "HaMeR repo exists: $HAMER_DIR"
    else
        mkdir -p "$(dirname "$HAMER_DIR")"
        log "cloning HaMeR into $HAMER_DIR"
        git clone --filter=blob:none \
            https://github.com/geopavlakos/hamer.git "$HAMER_DIR"
    fi
    git -C "$HAMER_DIR" checkout --detach "$HAMER_COMMIT"
    git -C "$HAMER_DIR" submodule update --init --recursive
}

extract_assets() {
    if [ -f "$HAMER_CKPT" ] && [ -f "$WHOLEBODY_SRC" ]; then
        log "HaMeR _DATA assets already exist under $ROOT/_DATA"
        return 0
    fi
    download_archive
    log "extracting $ARCHIVE into $ROOT"
    (
        cd "$ROOT"
        tar --warning=no-unknown-keyword --exclude=".*" -xzf "$ARCHIVE"
    )
}

link_wholebody_vitpose() {
    mkdir -p "$(dirname "$WHOLEBODY_DST")"
    if [ -L "$WHOLEBODY_DST" ] || [ -e "$WHOLEBODY_DST" ]; then
        log "wholebody VitPose target exists: $WHOLEBODY_DST"
        return 0
    fi
    if [ ! -f "$WHOLEBODY_SRC" ]; then
        echo "Missing wholebody VitPose source: $WHOLEBODY_SRC" >&2
        exit 1
    fi
    ln -s "$WHOLEBODY_SRC" "$WHOLEBODY_DST"
    log "linked wholebody VitPose: $WHOLEBODY_DST -> $WHOLEBODY_SRC"
}

link_mano_if_provided() {
    if [ -f "$MANO_DST" ]; then
        log "MANO_RIGHT.pkl exists: $MANO_DST"
        return 0
    fi
    if [ -n "${MANO_RIGHT_PATH:-}" ]; then
        mkdir -p "$(dirname "$MANO_DST")"
        ln -s "$(realpath "$MANO_RIGHT_PATH")" "$MANO_DST"
        log "linked MANO_RIGHT.pkl: $MANO_DST -> $(realpath "$MANO_RIGHT_PATH")"
        return 0
    fi

    cat <<EOF

MANO_RIGHT.pkl is still missing.
Download it manually from the official MANO site after registration:
  https://mano.is.tue.mpg.de/

Then either copy or link it here:
  $MANO_DST

Example:
  MANO_RIGHT_PATH=/path/to/MANO_RIGHT.pkl bash scripts/setup_hamer_assets.sh
EOF
}

install_hamer_if_requested() {
    if [ "${INSTALL_HAMER:-0}" != "1" ]; then
        cat <<EOF

HaMeR code is ready at:
  $HAMER_DIR

To make 'import hamer' work, either install it:
  cd "$HAMER_DIR"
  pip install -e .[all]
  pip install -v -e third-party/ViTPose

or run GVHMR-hand with:
  export PYTHONPATH="$HAMER_DIR:\${PYTHONPATH:-}"
EOF
        return 0
    fi

    log "installing HaMeR into the current Python environment"
    (
        cd "$HAMER_DIR"
        pip install -e .[all]
        pip install -v -e third-party/ViTPose
    )
}

clone_hamer
extract_assets
link_wholebody_vitpose
link_mano_if_provided
install_hamer_if_requested

log "done"
log "HaMeR checkpoint: $HAMER_CKPT"
log "WholeBody VitPose: $WHOLEBODY_DST"
