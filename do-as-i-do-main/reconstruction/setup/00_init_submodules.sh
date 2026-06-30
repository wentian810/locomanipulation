#!/bin/bash
# [00] Initialize third-party modules at pinned fork commits.
# Supports both a normal recursive Git clone and a source archive/company-Git
# export where the parent repository has no .git metadata.
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export GIT_LFS_SKIP_SMUDGE=1   # don't smudge any LFS blobs; weights come from 02_fetch_weights.sh

clone_pinned() {
    local rel_path="$1"
    local url="$2"
    local commit="$3"
    local dst="$ROOT/$rel_path"

    if [ -d "$dst/.git" ]; then
        echo "[00] Reusing $rel_path"
        git -C "$dst" fetch origin
    elif [ -d "$dst" ] && [ -n "$(find "$dst" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
        echo "[00] ERROR: $dst is non-empty but is not a Git checkout." >&2
        echo "Remove or relocate it before rerunning this setup script." >&2
        exit 2
    else
        # Source archives commonly contain empty submodule placeholders.
        # Remove only that verified-empty directory; never delete content.
        [ ! -d "$dst" ] || rmdir "$dst"
        echo "[00] Cloning $url -> $rel_path"
        git clone --filter=blob:none "$url" "$dst"
    fi

    git -C "$dst" checkout --detach "$commit"
    git -C "$dst" submodule update --init --recursive
}

echo "[00] Cloning + checking out pinned module revisions..."
clone_pinned "modules/tapnet" \
    "https://github.com/malik-group/tapnet.git" \
    "f2f8888"
clone_pinned "modules/sam-3d-objects" \
    "https://github.com/malik-group/sam-3d-objects.git" \
    "875b010"
clone_pinned "modules/Fast-SAM3D" \
    "https://github.com/malik-group/Fast-SAM3D.git" \
    "823d478"
clone_pinned "modules/HaWoR" \
    "https://github.com/malik-group/HaWoR.git" \
    "2c3fa0c"
clone_pinned "modules/sam3" \
    "https://github.com/malik-group/sam3.git" \
    "b8e18f5"

echo "[00] Submodule pins:"
for module in modules/tapnet modules/sam-3d-objects modules/Fast-SAM3D modules/HaWoR modules/sam3; do
    printf '  %s  %s\n' "$(git -C "$ROOT/$module" rev-parse --short HEAD)" "$module"
done
echo "[00] Done. Next: ./setup/01_create_envs.sh"
