#!/usr/bin/env bash
# Upload the prepared human-only delivery bundle to UCloud US3/UFile.
# Credentials remain in a local US3CLI profile; this script never reads or logs them.
set -Eeuo pipefail

US3_BIN="${US3_BIN:-/home/jixingyu/.local/bin/us3cli-linux64}"
US3_CONFIG="${US3_CONFIG:-locomotion-upload}"
DELIVERY_DIR="${DELIVERY_DIR:-/home/jixingyu/locomotion_human_only_delivery_20260806}"
BUCKET="${BUCKET:-robotic-docker-images}"
PREFIX="${PREFIX:-locomotion}"
PARALLEL="${PARALLEL:-16}"
RETRY_COUNT="${RETRY_COUNT:-20}"

files=(
  "human-pipeline-support-contacts_20260806.tar.zst"
  "dataset_new6_20260806.tar.zst"
  "locomotion-human-only_20260806.docker.tar.zst"
  "human-only-model-assets_20260806.tar.zst"
  "SHA256SUMS"
)

[[ -x "$US3_BIN" ]] || { echo "US3CLI not executable: $US3_BIN" >&2; exit 2; }
[[ -d "$DELIVERY_DIR" ]] || { echo "Delivery directory missing: $DELIVERY_DIR" >&2; exit 2; }

remote_size() {
  "$US3_BIN" stat "$1" --config "$US3_CONFIG" 2>/dev/null \
    | awk '/^Content-Length:/ { print $2; exit }'
}

for name in "${files[@]}"; do
  local_file="$DELIVERY_DIR/$name"
  remote_file="us3://$BUCKET/$PREFIX/$name"
  [[ -f "$local_file" ]] || { echo "Missing local artifact: $local_file" >&2; exit 2; }
  local_bytes="$(stat -c %s "$local_file")"
  existing_bytes="$(remote_size "$remote_file" || true)"

  if [[ "$existing_bytes" == "$local_bytes" ]]; then
    echo "[skip] verified existing object: $remote_file ($local_bytes bytes)"
    continue
  fi

  echo "[upload] $local_file -> $remote_file ($local_bytes bytes)"
  "$US3_BIN" cp "$local_file" "$remote_file" \
    --config "$US3_CONFIG" \
    --parallel "$PARALLEL" \
    --retrycount "$RETRY_COUNT"

  uploaded_bytes="$(remote_size "$remote_file")"
  [[ "$uploaded_bytes" == "$local_bytes" ]] || {
    echo "Size mismatch for $remote_file: expected $local_bytes, got ${uploaded_bytes:-missing}" >&2
    exit 1
  }
  echo "[ok] $remote_file"
done

echo "[complete] Remote objects:"
"$US3_BIN" ls "us3://$BUCKET/$PREFIX" --config "$US3_CONFIG" --flat --limit 100
