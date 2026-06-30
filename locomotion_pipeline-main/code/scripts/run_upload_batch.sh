#!/bin/bash

# 按 sequence_statistics.csv 的合格率阈值筛选序列，并上传通过的序列目录。

set -euo pipefail

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 项目根目录（上两级）
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# ============ 上传配置 ============
# 本地 zitai 批次根目录（相对 PROJECT_ROOT）
ZITAI_BATCHES_ROOT_REL="output_runs/batch1/zitai"

# 上传顺序：依次上传下列子目录（不存在则跳过并警告）
ZITAI_BATCH_NAMES=(
	zitai_n10000_idx001
	zitai_n10000_idx002
	zitai_n10000_idx003
	zitai_n10000_idx004
	zitai_n10000_idx007
	zitai_n10000_idx008
	zitai_n10000_idx009
	zitai_n10000_idx010
)

# 本地 output_runs 根目录（相对 PROJECT_ROOT）
OUTPUT_RUNS_ROOT_REL="output_runs"

# 远端基础路径：每个批次目录相对 OUTPUT_RUNS_ROOT_REL 的子路径会追加到该路径后
# 例如：output_runs/batch1/zitai/zitai_n10000_idx000 ->
#   syna/synadata-output-wlcb/motionx_opt/shenbipai/batch1/zitai/zitai_n10000_idx000
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
ZITAI_BATCHES_ROOT="${PROJECT_ROOT}/${ZITAI_BATCHES_ROOT_REL}"
OUTPUT_RUNS_ROOT="${PROJECT_ROOT}/${OUTPUT_RUNS_ROOT_REL}"

if [[ ! -d "$ZITAI_BATCHES_ROOT" ]]; then
	echo "ERROR: 本地 zitai 批次根目录不存在: $ZITAI_BATCHES_ROOT"
	exit 1
fi

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
	echo "ERROR: 未找到 upload.py: $PYTHON_SCRIPT"
	exit 1
fi

if [[ "${ZITAI_BATCHES_ROOT_REL}" != "${OUTPUT_RUNS_ROOT_REL}"/* ]]; then
	echo "ERROR: ZITAI_BATCHES_ROOT_REL 必须位于 OUTPUT_RUNS_ROOT_REL 下"
	echo "       ZITAI_BATCHES_ROOT_REL=$ZITAI_BATCHES_ROOT_REL"
	echo "       OUTPUT_RUNS_ROOT_REL=$OUTPUT_RUNS_ROOT_REL"
	exit 1
fi

ZITAI_LOCAL_REL="${ZITAI_BATCHES_ROOT_REL#${OUTPUT_RUNS_ROOT_REL}/}"
REMOTE_ZITAI_PARENT="${REMOTE_BASE%/}/${ZITAI_LOCAL_REL}"

if [[ "$CHECK_PARENT_EXISTS" == "true" ]]; then
	echo "检查远端父目录是否存在: $REMOTE_ZITAI_PARENT"
	if ! "$MC_BIN" ls "$REMOTE_ZITAI_PARENT" >/dev/null 2>&1; then
		echo "ERROR: 远端父目录不存在或不可访问: $REMOTE_ZITAI_PARENT"
		echo "请先创建目录或修正 REMOTE_BASE / ZITAI_BATCHES_ROOT_REL。"
		exit 1
	fi
fi

echo "按指定顺序上传 ${#ZITAI_BATCH_NAMES[@]} 个批次"
echo "==============================================="
echo "本地根目录 (Root):      $OUTPUT_RUNS_ROOT"
echo "远端基路径 (Remote):    $REMOTE_BASE"
echo "阈值 (Threshold):       $THRESHOLD"
echo "并发 (Workers):         $WORKERS"
echo "Dry-run:                $DRY_RUN"
echo "==============================================="

for name in "${ZITAI_BATCH_NAMES[@]}"; do
	LOCAL_CHUNK_REL="${ZITAI_BATCHES_ROOT_REL}/${name}"
	LOCAL_CHUNK="${PROJECT_ROOT}/${LOCAL_CHUNK_REL}"
	if [[ ! -d "$LOCAL_CHUNK" ]]; then
		echo "WARN: 跳过（本地目录不存在）: $LOCAL_CHUNK_REL"
		continue
	fi

	LOCAL_REL_PATH="${LOCAL_CHUNK_REL#${OUTPUT_RUNS_ROOT_REL}/}"
	if [[ "$LOCAL_REL_PATH" == "$LOCAL_CHUNK_REL" ]]; then
		echo "ERROR: 批次路径必须位于 OUTPUT_RUNS_ROOT_REL 下: $LOCAL_CHUNK_REL"
		exit 1
	fi

	REMOTE_CHUNK="${REMOTE_BASE%/}/${LOCAL_REL_PATH}"
	echo ""
	echo "--------------- 批次: $name ---------------"
	echo "本地目录 (Local):       $LOCAL_CHUNK"
	echo "目标目录 (RemoteChunk): $REMOTE_CHUNK"

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
done

echo ""
echo "全部批次已处理完毕。"

