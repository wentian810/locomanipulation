#!/bin/bash
# [02] Populate model weights (gitignored) into weights/ and the submodule checkpoint dirs.
#
# Modes:
#   ./02_fetch_weights.sh --from-local [DIR]   copy from a local monorepo checkout (sam3d-src/, fast-sam3d-src/, hawor-src/, tapnet/) (fast)
#   ./02_fetch_weights.sh --download                 download from HuggingFace / GDrive / URLs
#   ./02_fetch_weights.sh                             auto: --from-local if SRC_ROOT is set, else --download
#
# SAM3D's 12 GB checkpoint set is stored ONCE in weights/sam3d_shared/hf and symlinked into both
# the sam-3d-objects and Fast-SAM3D checkpoint dirs (saves ~12 GB vs duplicating).
# SAM3D weights are GATED: request access at https://huggingface.co/facebook/sam-3d-objects and
# `hf auth login` before using --download.
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/config/paths.sh"

SRC_ROOT="${SRC_ROOT:-}"
case "${1:-}" in
  --from-local) MODE=local; [ -n "${2:-}" ] && SRC_ROOT="$2" ;;
  --download)   MODE=download ;;
  "")           { [ -d "$SRC_ROOT" ] && MODE=local; } || MODE=download ;;
  *) echo "usage: $0 [--from-local [DIR] | --download]"; exit 1 ;;
esac
if [ "$MODE" = local ] && [ -z "$SRC_ROOT" ]; then echo "error: --from-local needs a source dir: $0 --from-local <DIR>" >&2; exit 1; fi
echo "[02] mode=$MODE  SRC_ROOT=$SRC_ROOT"

SHARED="$WEIGHTS_DIR/sam3d_shared/hf"
HEAVY="slat_generator.ckpt slat_encoder.ckpt ss_generator.ckpt ss_encoder.ckpt slat_decoder_mesh.ckpt slat_decoder_mesh.pt slat_decoder_gs.ckpt slat_decoder_gs_4.ckpt ss_decoder.ckpt ss_encoder.safetensors"
mkdir -p "$WEIGHTS_DIR" "$SHARED" "$WEIGHTS_DIR/tapnet"

link_sam3d_into_repos () {   # symlink shared heavy files into both repos' checkpoints/hf
  for name in sam-3d-objects Fast-SAM3D; do
    dst="$ROOT/modules/$name/checkpoints/hf"; mkdir -p "$dst"
    for f in $HEAVY; do [ -f "$SHARED/$f" ] && ln -sf "$SHARED/$f" "$dst/$f"; done
  done
}

# ─────────────────────────────── SAM3D ───────────────────────────────
if [ "$MODE" = local ]; then
  src="$SRC_ROOT/sam3d-src/sam-3d-objects/checkpoints/hf"
  echo "[02] SAM3D: copy shared heavy weights once from $src"
  for f in $HEAVY; do [ -f "$src/$f" ] && cp -n "$src/$f" "$SHARED/$f"; done
  # per-repo small configs (pipeline.yaml, *.yaml) copied real from the local source
  rsync -a --exclude='*.ckpt' --exclude='*.pt' --exclude='*.safetensors' \
        "$SRC_ROOT/sam3d-src/sam-3d-objects/checkpoints/hf/" "$ROOT/modules/sam-3d-objects/checkpoints/hf/"
  rsync -a --exclude='*.ckpt' --exclude='*.pt' --exclude='*.safetensors' \
        "$SRC_ROOT/fast-sam3d-src/Fast-SAM3D/checkpoints/hf/"  "$ROOT/modules/Fast-SAM3D/checkpoints/hf/"
  link_sam3d_into_repos
else
  echo "[02] SAM3D: downloading from HuggingFace (gated: facebook/sam-3d-objects)"
  pip install -q 'huggingface-hub[cli]<1.0' || true
  if hf download facebook/sam-3d-objects --local-dir "$WEIGHTS_DIR/sam3d-download"; then
    dl="$WEIGHTS_DIR/sam3d-download"
    if   [ -d "$dl/checkpoints/hf" ]; then dl="$dl/checkpoints/hf"
    elif [ -d "$dl/checkpoints" ];    then dl="$dl/checkpoints"; fi
    for f in $HEAVY; do [ -f "$dl/$f" ] && cp -n "$dl/$f" "$SHARED/$f"; done
    for name in sam-3d-objects Fast-SAM3D; do
      rsync -a --exclude='*.ckpt' --exclude='*.pt' --exclude='*.safetensors' "$dl/" "$ROOT/modules/$name/checkpoints/hf/"
    done
    link_sam3d_into_repos
  else
    echo "  !! SAM3D download failed — request access + 'hf auth login', then re-run." >&2
  fi
fi

# ─────────────────────────────── HaWoR ───────────────────────────────
mkdir -p "$HAWOR_DIR/weights/hawor/checkpoints" "$HAWOR_DIR/weights/external" \
         "$HAWOR_DIR/thirdparty/Metric3D/weights" \
         "$HAWOR_DIR/_DATA/data/mano" "$HAWOR_DIR/_DATA/data_left/mano_left"
if [ "$MODE" = local ]; then
  hs="$SRC_ROOT/hawor-src/HaWoR"
  echo "[02] HaWoR: copy weights from $hs"
  cp -n "$hs/weights/hawor/checkpoints/hawor.ckpt"   "$HAWOR_DIR/weights/hawor/checkpoints/" 2>/dev/null || true
  cp -n "$hs/weights/hawor/checkpoints/infiller.pt"  "$HAWOR_DIR/weights/hawor/checkpoints/" 2>/dev/null || true
  cp -n "$hs/weights/hawor/model_config.yaml"        "$HAWOR_DIR/weights/hawor/"             2>/dev/null || true
  cp -n "$hs/weights/external/detector.pt"           "$HAWOR_DIR/weights/external/"          2>/dev/null || true
  cp -n "$hs/weights/external/droid.pth"             "$HAWOR_DIR/weights/external/"          2>/dev/null || true
  cp -n "$hs/thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth" "$HAWOR_DIR/thirdparty/Metric3D/weights/" 2>/dev/null || true
  cp -n "$hs/_DATA/data/mano/MANO_RIGHT.pkl"         "$HAWOR_DIR/_DATA/data/mano/"           2>/dev/null || true
  cp -n "$hs/_DATA/data_left/mano_left/MANO_LEFT.pkl" "$HAWOR_DIR/_DATA/data_left/mano_left/" 2>/dev/null || true
else
  echo "[02] HaWoR: download per HaWoR README"
  wget -nc https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -P "$HAWOR_DIR/weights/external/"          || echo "  !! detector.pt failed (continuing)" >&2
  wget -nc https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/hawor.ckpt  -P "$HAWOR_DIR/weights/hawor/checkpoints/"          || echo "  !! hawor.ckpt failed (continuing)" >&2
  wget -nc https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/infiller.pt -P "$HAWOR_DIR/weights/hawor/checkpoints/"          || echo "  !! infiller.pt failed (continuing)" >&2
  wget -nc https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/model_config.yaml       -P "$HAWOR_DIR/weights/hawor/"                      || echo "  !! model_config.yaml failed (continuing)" >&2
  pip install -q gdown || true
  gdown 1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh -O "$HAWOR_DIR/weights/external/droid.pth" || true
  gdown 1eT2gG-kwsVzNy5nJrbm4KC-9DbNKyLnr -O "$HAWOR_DIR/thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth" || true
  echo "  NOTE: MANO models need a manual download (license) from https://mano.is.tue.mpg.de"
  echo "        -> $HAWOR_DIR/_DATA/data/mano/MANO_RIGHT.pkl and _DATA/data_left/mano_left/MANO_LEFT.pkl"
fi

# ─────────────────────────────── TAPIR ───────────────────────────────
if [ "$MODE" = local ]; then
  echo "[02] TAPIR: copy BootsTAPIR checkpoint"
  cp -n "$SRC_ROOT/tapnet/checkpoints/bootstapir_checkpoint_v2.pt" "$WEIGHTS_DIR/tapnet/" 2>/dev/null || true
else
  echo "[02] TAPIR: download BootsTAPIR checkpoint"
  wget -nc https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt -P "$WEIGHTS_DIR/tapnet/"
fi

echo "[02] NOTE: the Stage-1 SAM3 segmentation model (facebook/sam3) is auto-downloaded"
echo "          at runtime by scripts/run_sam3_video.py and is ALSO gated on HuggingFace —"
echo "          request access + 'hf auth login' or Stage 1 will fail to fetch it."
echo "[02] Done."
