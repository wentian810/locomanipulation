#!/usr/bin/env bash
# Restore external source repositories without vendoring their Git histories.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-core}"

log() {
    printf '[bootstrap] %s\n' "$*"
}

clone_pinned() {
    local destination="$1"
    local url="$2"
    local commit="$3"
    local recursive="${4:-0}"

    if [ -d "$destination/.git" ]; then
        log "reuse ${destination#$ROOT/}"
    elif [ -e "$destination" ]; then
        if [ -d "$destination" ] &&
            [ -z "$(find "$destination" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
            rmdir "$destination"
        else
            echo "Refusing to replace non-empty path: $destination" >&2
            exit 2
        fi
        git clone --filter=blob:none "$url" "$destination"
    else
        mkdir -p "$(dirname "$destination")"
        git clone --filter=blob:none "$url" "$destination"
    fi

    if ! git -C "$destination" cat-file -e "${commit}^{commit}" 2>/dev/null; then
        git -C "$destination" fetch origin "$commit"
    fi
    git -C "$destination" checkout --detach "$commit"
    if [ "$recursive" = "1" ]; then
        git -C "$destination" submodule update --init --recursive
    fi
}

apply_patch_once() {
    local destination="$1"
    local patch_file="$2"
    if git -C "$destination" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
        log "patch already applied: ${patch_file#$ROOT/}"
        return
    fi
    git -C "$destination" apply --check "$patch_file"
    git -C "$destination" apply "$patch_file"
    log "applied ${patch_file#$ROOT/}"
}

bootstrap_core() {
    local gvhmr="$ROOT/GVHMR-hand/GVHMR-main"

    clone_pinned \
        "$gvhmr/third-party/DPVO" \
        "https://github.com/princeton-vl/DPVO.git" \
        "859bbbfdac6c6185f345003b3c473901fcd13ace" \
        1
    clone_pinned \
        "$gvhmr/third-party/hamer" \
        "https://github.com/geopavlakos/hamer.git" \
        "3a01849f4148352e9260b69bf28b65d1671a4905" \
        1
    clone_pinned \
        "$gvhmr/third-party/WiLoR" \
        "https://github.com/rolpotamias/WiLoR.git" \
        "fcb911312a38fa8badd30d9656a167485d61b8f9"
    apply_patch_once \
        "$gvhmr/third-party/WiLoR" \
        "$ROOT/patches/third_party/wilor_gvhmr.patch"
    clone_pinned \
        "$gvhmr/third-party/Hand4Whole-plus-plus_RELEASE" \
        "https://github.com/mks0601/Hand4Whole-plus-plus_RELEASE.git" \
        "f81d35ddd2b74206c40142243eb62b6d64ce0d65"
    apply_patch_once \
        "$gvhmr/third-party/Hand4Whole-plus-plus_RELEASE" \
        "$ROOT/patches/third_party/hand4wholepp_gvhmr.patch"

    clone_pinned \
        "$ROOT/phc-deps/SMPLSim" \
        "https://github.com/ZhengyiLuo/SMPLSim.git" \
        "b5c08720503ad5fff64050c4d289c42d947fcf8d"
    clone_pinned \
        "$ROOT/phc-deps/chumpy" \
        "https://github.com/mattloper/chumpy.git" \
        "580566eafc9ac68b2614b64d6f7aaa84eebb70da"
    clone_pinned \
        "$ROOT/phc-deps/smplx_fork" \
        "https://github.com/ZhengyiLuo/smplx.git" \
        "a5b8e4ac14f79f3f33fd2cf2a16e6f507146b813"
}

bootstrap_object_modules() {
    (
        cd "$ROOT/do-as-i-do-main/reconstruction"
        ./setup/00_init_submodules.sh
    )
}

bootstrap_robot_hands() {
    clone_pinned \
        "$ROOT/GMR-master/third_party/robot_hands/brainco_description" \
        "https://github.com/BrainCoTech/brainco-description.git" \
        "f332a6f0dc944e26b82976b637074b03f7ee8a2c"
    clone_pinned \
        "$ROOT/GMR-master/third_party/robot_hands/unitree_xr_teleoperate" \
        "https://github.com/unitreerobotics/xr_teleoperate.git" \
        "7dc9aa1a6edbf4a9f4f887d8ab6fc449ea5135f6"
}

case "$MODE" in
    core)
        bootstrap_core
        ;;
    object)
        bootstrap_object_modules
        ;;
    all)
        bootstrap_core
        bootstrap_object_modules
        bootstrap_robot_hands
        ;;
    *)
        echo "Usage: $0 [core|object|all]" >&2
        exit 2
        ;;
esac

log "done ($MODE)"
