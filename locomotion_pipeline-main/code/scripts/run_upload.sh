#!/bin/bash

# 按 sequence_statistics.csv 的合格率阈值筛选序列，并上传通过的序列目录。

set -euo pipefail

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 项目根目录（上两级）
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# ============ 上传配置 ============
# 本地 chunk 目录（相对 PROJECT_ROOT）
LOCAL_CHUNK_REL="output_runs/batch1/zitai/zitai_n10000_idx000"

# 本地 output_runs 根目录（相对 PROJECT_ROOT）
OUTPUT_RUNS_ROOT_REL="output_runs"

# 远端基础路径，LOCAL_CHUNK_REL 相对 OUTPUT_RUNS_ROOT_REL 的子路径会追加到该路径后
# 例如：
#   local:  output_runs/batch1/zitai/zitai_n10000_idx000
#   remote: syna/synadata-output-wlcb/motionx_opt/shenbipai/batch1/zitai/zitai_n10000_idx000
REMOTE_BASE="syna/synadata-output-wlcb/motionx_opt/shenbipai"

# 合格率阈值（0~1）
THRESHOLD="0.95"

# 上传并发数
WORKERS=4

# mc 可执行文件
MC_BIN="mc"

# 是否仅预览（true/false）
DRY_RUN="false"

# 是否要求远端父目录已存在（true/false）
CHECK_PARENT_EXISTS="true"

PYTHON_SCRIPT="${PROJECT_ROOT}/code/scripts/upload.py"
LOCAL_CHUNK="${PROJECT_ROOT}/${LOCAL_CHUNK_REL}"
OUTPUT_RUNS_ROOT="${PROJECT_ROOT}/${OUTPUT_RUNS_ROOT_REL}"

if [[ ! -d "$LOCAL_CHUNK" ]]; then
	echo "ERROR: 本地目录不存在: $LOCAL_CHUNK"
	exit 1
fi

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
	echo "ERROR: 未找到 upload.py: $PYTHON_SCRIPT"
	exit 1
fi

# 计算 remote chunk 与其父目录（用于存在性检查）
LOCAL_REL_PATH="${LOCAL_CHUNK_REL#${OUTPUT_RUNS_ROOT_REL}/}"
if [[ "$LOCAL_REL_PATH" == "$LOCAL_CHUNK_REL" ]]; then
	echo "ERROR: LOCAL_CHUNK_REL 必须位于 OUTPUT_RUNS_ROOT_REL 下"
	echo "       LOCAL_CHUNK_REL=$LOCAL_CHUNK_REL"
	echo "       OUTPUT_RUNS_ROOT_REL=$OUTPUT_RUNS_ROOT_REL"
	exit 1
fi

REMOTE_CHUNK="${REMOTE_BASE%/}/${LOCAL_REL_PATH}"
REMOTE_PARENT="${REMOTE_CHUNK%/*}"

if [[ "$CHECK_PARENT_EXISTS" == "true" ]]; then
	echo "检查远端父目录是否存在: $REMOTE_PARENT"
	if ! "$MC_BIN" ls "$REMOTE_PARENT" >/dev/null 2>&1; then
		echo "ERROR: 远端父目录不存在或不可访问: $REMOTE_PARENT"
		echo "请先创建目录或修正 REMOTE_BASE / LOCAL_CHUNK_REL。"
		exit 1
	fi
fi

echo "==============================================="
echo "本地目录 (Local):       $LOCAL_CHUNK"
echo "本地根目录 (Root):      $OUTPUT_RUNS_ROOT"
echo "远端基路径 (Remote):    $REMOTE_BASE"
echo "目标目录 (RemoteChunk): $REMOTE_CHUNK"
echo "阈值 (Threshold):       $THRESHOLD"
echo "并发 (Workers):         $WORKERS"
echo "Dry-run:                $DRY_RUN"
echo "==============================================="

PYTHON_ARGS=(
	"$LOCAL_CHUNK"
	--threshold "$THRESHOLD"
	--remote-base "$REMOTE_BASE"
	--output-runs-root "$OUTPUT_RUNS_ROOT"
	--mc-bin "$MC_BIN"
	--workers "$WORKERS"
)

if [[ "$DRY_RUN" == "true" ]]; then
	PYTHON_ARGS+=(--dry-run)
fi

python3 "$PYTHON_SCRIPT" "${PYTHON_ARGS[@]}"

