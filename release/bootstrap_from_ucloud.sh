#!/usr/bin/env bash
# Restore the verified 2026-08-06 human-only runtime release from UCloud US3.
#
# Run this script from a GitHub clone of this repository.  It deliberately
# obtains source code from Git and obtains only the large, untracked runtime
# artifacts from US3.  Credentials stay in a local us3cli profile.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RELEASE="20260806"
DOWNLOAD_ROOT=""
ARTIFACT_DIR=""
US3_BIN="${US3_BIN:-us3cli-linux64}"
US3_CONFIG="${US3_CONFIG:-locomotion-download}"
BUCKET="${BUCKET:-robotic-docker-images}"
PREFIX="${PREFIX:-locomotion}"
PARALLEL="${PARALLEL:-16}"
RETRY_COUNT="${RETRY_COUNT:-20}"
IMAGE_NAME="locomotion-human-only:20260806"
DOCKER_SUDO="${DOCKER_SUDO:-0}"
WITH_IMAGE=1
WITH_MODELS=1
WITH_DATASET=1
WITH_SOURCE_ARCHIVE=0
DOWNLOAD_ONLY=0
VERIFY_ONLY=0

usage() {
  cat <<'EOF'
Usage:
  bash release/bootstrap_from_ucloud.sh [options]

This must be run from a GitHub checkout of the release repository.  It
downloads UCloud US3 artifacts, checks SHA-256, extracts model/data assets and
loads the Docker image.  It never requests, stores or prints an access key.

Options:
  --repo-root PATH        GitHub checkout (default: repository containing script)
  --download-root PATH    Release cache (default: <repo>/.release/<release>)
  --artifact-dir PATH     Use an existing local delivery bundle; do not download it
  --release ID            Release identifier (default: 20260806)
  --us3-bin PATH          us3cli binary (default: us3cli-linux64 on PATH)
  --us3-config NAME       Local us3cli read profile (default: locomotion-download)
  --bucket NAME           US3 bucket (default: robotic-docker-images)
  --prefix PATH           US3 prefix (default: locomotion)
  --parallel N            Multipart workers (default: 16)
  --docker-sudo           Run Docker through non-interactive sudo -n docker
  --no-image              Do not download/load the Docker image archive
  --no-models             Do not download/extract the model asset archive
  --no-dataset            Do not download/extract the example input videos
  --with-source-archive   Also fetch/checksum the archival source snapshot (not runnable)
  --download-only         Fetch and verify only; do not extract or docker load
  --verify-only           Verify existing downloaded files only; no network
  -h, --help              Show this help

Examples:
  # First configure a read-only local profile, then restore the full release.
  us3cli-linux64 config --config locomotion-download
  bash release/bootstrap_from_ucloud.sh --us3-config locomotion-download

  # Recheck a previously downloaded release without contacting US3.
  bash release/bootstrap_from_ucloud.sh --verify-only

  # Server-side validation: reuse the already prepared release bundle.
  bash release/bootstrap_from_ucloud.sh \
    --artifact-dir /home/jixingyu/locomotion_human_only_delivery_20260806
EOF
}

die() { echo "[release-bootstrap] ERROR: $*" >&2; exit 2; }
note() { echo "[release-bootstrap] $*"; }
docker_cmd() {
  if [[ "$DOCKER_SUDO" == "1" ]]; then
    sudo -n docker "$@"
  else
    docker "$@"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) REPO_ROOT="$2"; shift 2 ;;
    --download-root) DOWNLOAD_ROOT="$2"; shift 2 ;;
    --artifact-dir) ARTIFACT_DIR="$2"; shift 2 ;;
    --release) RELEASE="$2"; shift 2 ;;
    --us3-bin) US3_BIN="$2"; shift 2 ;;
    --us3-config) US3_CONFIG="$2"; shift 2 ;;
    --bucket) BUCKET="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --parallel) PARALLEL="$2"; shift 2 ;;
    --docker-sudo) DOCKER_SUDO=1; shift ;;
    --no-image) WITH_IMAGE=0; shift ;;
    --no-models) WITH_MODELS=0; shift ;;
    --no-dataset) WITH_DATASET=0; shift ;;
    --with-source-archive) WITH_SOURCE_ARCHIVE=1; shift ;;
    --download-only) DOWNLOAD_ONLY=1; shift ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

[[ "$RELEASE" =~ ^[0-9]{8}$ ]] || die "Release must use YYYYMMDD, got: $RELEASE"
[[ "$PARALLEL" =~ ^[1-9][0-9]*$ ]] || die "--parallel must be a positive integer"
REPO_ROOT="$(cd -- "$REPO_ROOT" && pwd)"
command -v git >/dev/null || die "git is required to validate --repo-root"
git -C "$REPO_ROOT" rev-parse --is-inside-work-tree 2>/dev/null | grep -qx true \
  || die "--repo-root is not a GitHub checkout: $REPO_ROOT"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-$REPO_ROOT/.release/$RELEASE}"
mkdir -p "$DOWNLOAD_ROOT"
if [[ -n "$ARTIFACT_DIR" ]]; then
  ARTIFACT_DIR="$(cd -- "$ARTIFACT_DIR" && pwd)"
else
  ARTIFACT_DIR="$DOWNLOAD_ROOT"
fi

if [[ "$VERIFY_ONLY" -eq 0 && "$ARTIFACT_DIR" == "$DOWNLOAD_ROOT" ]]; then
  if [[ "$US3_BIN" == */* ]]; then
    [[ -x "$US3_BIN" ]] || die "US3CLI is not executable: $US3_BIN"
  else
    command -v "$US3_BIN" >/dev/null || die "US3CLI is not on PATH: $US3_BIN"
  fi
fi
command -v sha256sum >/dev/null || die "sha256sum is required"

source_name="human-pipeline-support-contacts_${RELEASE}.tar.zst"
dataset_name="dataset_new6_${RELEASE}.tar.zst"
image_name="locomotion-human-only_${RELEASE}.docker.tar.zst"
model_name="human-only-model-assets_${RELEASE}.tar.zst"
manifest="$ARTIFACT_DIR/SHA256SUMS"

fetch_manifest() {
  if [[ "$ARTIFACT_DIR" != "$DOWNLOAD_ROOT" ]]; then
    [[ -f "$manifest" ]] || die "Local artifact directory lacks SHA256SUMS: $ARTIFACT_DIR"
    note "reuse local manifest: $manifest"
    return
  fi
  [[ "$VERIFY_ONLY" -eq 0 ]] || return 0
  local partial="$manifest.partial"
  rm -f -- "$partial"
  note "download SHA256SUMS"
  "$US3_BIN" cp "us3://$BUCKET/$PREFIX/SHA256SUMS" "$partial" \
    --config "$US3_CONFIG" --parallel 1 --retrycount "$RETRY_COUNT"
  mv -f -- "$partial" "$manifest"
}

expected_sha() {
  local name="$1"
  [[ -f "$manifest" ]] || die "Missing manifest: $manifest"
  awk -v wanted="$name" '
    NF >= 2 {
      file=$2
      sub(/^.*\//, "", file)
      if (file == wanted) { print $1; exit }
    }
  ' "$manifest"
}

verify_file() {
  local name="$1" path="$ARTIFACT_DIR/$1" expected actual
  expected="$(expected_sha "$name")"
  [[ "$expected" =~ ^[0-9a-fA-F]{64}$ ]] || die "No SHA-256 entry for $name in $manifest"
  [[ -f "$path" ]] || die "Missing artifact: $path"
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || die "SHA-256 mismatch for $name (delete this local file and rerun)"
  note "verified $name"
}

fetch_file() {
  local name="$1" path="$ARTIFACT_DIR/$1" partial="$DOWNLOAD_ROOT/$1.partial"
  if [[ -f "$path" ]] && verify_file "$name"; then
    note "reuse verified $name"
    return
  fi
  [[ "$VERIFY_ONLY" -eq 0 ]] || die "Cannot verify missing artifact in --verify-only mode: $name"
  [[ "$ARTIFACT_DIR" == "$DOWNLOAD_ROOT" ]] || die "Missing artifact in local --artifact-dir: $name"
  # Keep an interrupted transfer separate from an already verified archive.
  rm -f -- "$partial"
  note "download $name"
  "$US3_BIN" cp "us3://$BUCKET/$PREFIX/$name" "$partial" \
    --config "$US3_CONFIG" --parallel "$PARALLEL" --retrycount "$RETRY_COUNT"
  mv -f -- "$partial" "$path"
  verify_file "$name"
}

extract_archive() {
  local name="$1" destination="$2" marker="$3"
  [[ -f "$destination/$marker" ]] && { note "reuse extracted $name"; return; }
  command -v tar >/dev/null || die "tar is required to extract $name"
  command -v unzstd >/dev/null || die "unzstd is required to extract $name"
  mkdir -p "$destination"
  note "extract $name -> $destination"
  tar --use-compress-program=unzstd -xf "$ARTIFACT_DIR/$name" -C "$destination"
  [[ -e "$destination/$marker" ]] || die "Extraction did not create expected path: $destination/$marker"
}

write_paths_file() {
  local state="$DOWNLOAD_ROOT/RELEASE_PATHS.env"
  {
    echo "# Generated by release/bootstrap_from_ucloud.sh; source from Bash only."
    printf 'RELEASE_ID=%q\n' "$RELEASE"
    printf 'RELEASE_ROOT=%q\n' "$DOWNLOAD_ROOT"
    printf 'MODEL_ROOT=%q\n' "$DOWNLOAD_ROOT/extracted/model-assets"
    printf 'DATA_ROOT=%q\n' "$DOWNLOAD_ROOT/extracted/dataset"
    printf 'DATASET_DIR=%q\n' "$DOWNLOAD_ROOT/extracted/dataset/dataset_new6"
    printf 'IMAGE_ARCHIVE=%q\n' "$ARTIFACT_DIR/$image_name"
    printf 'IMAGE_NAME=%q\n' "$IMAGE_NAME"
  } > "$state"
  note "wrote runtime paths: $state"
}

fetch_manifest
[[ -f "$manifest" ]] || die "Missing manifest. Run without --verify-only once."

for required in "$source_name" "$dataset_name" "$image_name" "$model_name"; do
  [[ -n "$(expected_sha "$required")" ]] || die "Manifest does not describe required release artifact: $required"
done

[[ "$WITH_SOURCE_ARCHIVE" -eq 0 ]] || fetch_file "$source_name"
[[ "$WITH_DATASET" -eq 0 ]] || fetch_file "$dataset_name"
[[ "$WITH_IMAGE" -eq 0 ]] || fetch_file "$image_name"
[[ "$WITH_MODELS" -eq 0 ]] || fetch_file "$model_name"

if [[ "$DOWNLOAD_ONLY" -eq 0 && "$VERIFY_ONLY" -eq 0 ]]; then
  if [[ "$WITH_MODELS" -eq 1 ]]; then
    extract_archive "$model_name" "$DOWNLOAD_ROOT/extracted/model-assets" \
      "GVHMR-hand/GVHMR-main/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt"
  fi
  if [[ "$WITH_DATASET" -eq 1 ]]; then
    extract_archive "$dataset_name" "$DOWNLOAD_ROOT/extracted/dataset" "dataset_new6"
  fi
  if [[ "$WITH_IMAGE" -eq 1 ]]; then
    command -v docker >/dev/null || die "docker is required to load the runtime image"
    command -v unzstd >/dev/null || die "unzstd is required to load the image"
    note "load Docker image $IMAGE_NAME (this can take several minutes)"
    unzstd -c "$ARTIFACT_DIR/$image_name" | docker_cmd load
    docker_cmd image inspect "$IMAGE_NAME" >/dev/null || die "docker load did not provide image: $IMAGE_NAME"
    note "Docker image available: $IMAGE_NAME"
  fi
  write_paths_file
fi

note "SUCCESS: artifacts were downloaded and verified."
if [[ "$DOWNLOAD_ONLY" -eq 0 && "$VERIFY_ONLY" -eq 0 && "$WITH_IMAGE" -eq 1 && "$WITH_MODELS" -eq 1 && "$WITH_DATASET" -eq 1 ]]; then
  note "Next: bash release/run_human_only_docker.sh --release-root '$DOWNLOAD_ROOT' --check-only"
fi
