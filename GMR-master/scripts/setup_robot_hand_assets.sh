#!/usr/bin/env bash
# Audit/download official robot-hand descriptions used by future GMR targets.
#
# Usage:
#   bash GMR-master/scripts/setup_robot_hand_assets.sh status
#   bash GMR-master/scripts/setup_robot_hand_assets.sh brainco
#   bash GMR-master/scripts/setup_robot_hand_assets.sh download
#
# Downloads are kept under GMR-master/third_party/robot_hands.  They are not
# copied into GMR assets automatically: mounting frames, joint order and
# actuator coupling must be validated per target robot before integration.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GMR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PIPELINE_ROOT="$(cd "${GMR_ROOT}/.." && pwd)"
ASSET_ROOT="${ROBOT_HAND_ASSET_ROOT:-${GMR_ROOT}/third_party/robot_hands}"
ACTION="${1:-status}"

UNITREE_LOCAL="${GMR_ROOT}/assets/unitree_g1/g1_mocap_29dof_with_hands.xml"
UNITREE_XR="${ASSET_ROOT}/unitree_xr_teleoperate"
LIMX_DESCRIPTION="${ASSET_ROOT}/limx_humanoid_description"
BRAINCO_DESCRIPTION="${ASSET_ROOT}/brainco_description"

log() {
    printf '[robot_hand_assets] %s\n' "$*"
}

require_git() {
    command -v git >/dev/null 2>&1 || {
        echo "git is required to download official robot-hand assets." >&2
        exit 1
    }
}

clone_sparse() {
    local label="$1"
    local url="$2"
    local destination="$3"
    shift 3
    local patterns=("$@")

    if [ -d "${destination}/.git" ]; then
        log "$label already exists: $destination"
        git -C "$destination" sparse-checkout add "${patterns[@]}"
        return 0
    fi
    if [ -e "$destination" ]; then
        echo "$label destination exists but is not a git checkout: $destination" >&2
        exit 1
    fi

    mkdir -p "$(dirname "$destination")"
    local attempt
    for attempt in 1 2 3; do
        log "cloning $label (attempt $attempt/3)"
        if git -c http.version=HTTP/1.1 clone \
            --depth 1 --filter=blob:none --sparse \
            "$url" "$destination"; then
            git -C "$destination" sparse-checkout set "${patterns[@]}"
            return 0
        fi
        log "$label clone failed; GitHub/TLS may be temporarily unavailable"
    done
    echo "Failed to clone $label after three attempts: $url" >&2
    exit 1
}

check_file() {
    local label="$1"
    local path="$2"
    if [ -f "$path" ]; then
        log "READY  $label: $path"
    else
        log "MISSING $label: $path"
    fi
}

status() {
    log "workspace: $PIPELINE_ROOT"
    check_file "Unitree G1 Dex3 body model (already local)" "$UNITREE_LOCAL"
    check_file "Unitree standalone Dex3 left" "${UNITREE_XR}/assets/unitree_hand/unitree_dex3_left.urdf"
    check_file "Unitree/因时 Inspire left" "${UNITREE_XR}/assets/inspire_hand/inspire_hand_left.urdf"
    check_file "Unitree BrainCo integration left" "${UNITREE_XR}/assets/brainco_hand/brainco_left.urdf"
    check_file "LimX/逐际 HU_D04 with hand" "${LIMX_DESCRIPTION}/HU_D04_description/urdf/HU_D04_01_with_hand.urdf"
    check_file "BrainCo Revo2 left" "${BRAINCO_DESCRIPTION}/revo2_system/urdf/revo2_left.urdf"
    check_file "BrainCo Revo2 MuJoCo left" "${BRAINCO_DESCRIPTION}/revo2_system/mjcf/revo2_left.xml"
    check_file "BrainCo Revo3 left" "${BRAINCO_DESCRIPTION}/revo3_system/urdf/revo3_left.urdf"
}

case "$ACTION" in
    status)
        status
        ;;
    brainco)
        require_git
        clone_sparse \
            "Unitree BrainCo retargeting configuration" \
            "https://github.com/unitreerobotics/xr_teleoperate.git" \
            "$UNITREE_XR" \
            assets/brainco_hand
        clone_sparse \
            "BrainCo Revo2 description assets" \
            "https://github.com/BrainCoTech/brainco-description.git" \
            "$BRAINCO_DESCRIPTION" \
            revo2_system
        log "BrainCo Revo2 download complete"
        log "NOTICE: the upstream brainco-description repository still says license information is pending."
        status
        ;;
    download)
        require_git
        clone_sparse \
            "Unitree XR hand assets (Dex3/Inspire/BrainCo)" \
            "https://github.com/unitreerobotics/xr_teleoperate.git" \
            "$UNITREE_XR" \
            assets/unitree_hand assets/inspire_hand assets/brainco_hand
        clone_sparse \
            "LimX humanoid hand assets" \
            "https://github.com/limxdynamics/humanoid-description.git" \
            "$LIMX_DESCRIPTION" \
            HU_D04_description
        clone_sparse \
            "BrainCo Revo description assets" \
            "https://github.com/BrainCoTech/brainco-description.git" \
            "$BRAINCO_DESCRIPTION" \
            revo2_system
        log "download complete"
        log "NOTICE: BrainCo currently states that license information will be provided with its public release."
        log "Review each checkout's license before redistribution; keep these vendor trees unmodified."
        status
        ;;
    *)
        echo "Usage: bash $0 status|brainco|download" >&2
        exit 2
        ;;
esac
