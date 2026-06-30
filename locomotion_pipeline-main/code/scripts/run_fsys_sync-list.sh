#!/bin/bash

# 使用 mc ls 扫描远程路径，在本地创建目录骨架，并统计文件清单和数量
# 支持读取文件夹列表并按规则构造正则进行批量下载

set -e

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 项目根目录（上两级）
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --remote-path <path>        覆盖 REMOTE_PATH
  --local-root <path>         覆盖 LOCAL_ROOT
  --folder-pattern <pattern>  覆盖 FOLDER_PATTERN
  --file-pattern <pattern>    覆盖 FILE_PATTERN
  --folder-list-file <path>   覆盖 FOLDER_LIST_FILE
  --mode <scan|scan-dl|dl>    覆盖 MODE
  --max-workers <number>      覆盖 MAX_WORKERS
  -y, --yes                   自动确认并执行
  -h, --help                  显示帮助
EOF
}

# 远程路径（mc alias）
# ! 请根据实际情况修改为你的远程路径
REMOTE_PATH="syna/synadata-source-wlcb/datasets/shenbipai"
# REMOTE_PATH="syna/synadata-source-wlcb/output/shenbipai"

# 本地输出目录
# ! 请根据实际情况修改为你希望存储数据的本地路径
LOCAL_ROOT="${PROJECT_ROOT}/data/storage/input_struct"
# LOCAL_ROOT="${PROJECT_ROOT}/data/storage/pkl_struct"

# Python 脚本路径
PYTHON_SCRIPT="${PROJECT_ROOT}/code/scripts/mc_scaffold_from_ls.py"

# ============ 操作模式 ============
# 模式说明：
#   scan     - 仅扫描远程路径并创建本地目录骨架，不下载（默认）
#   scan-dl  - 扫描并下载（需配置 folder/file pattern）
#   dl       - 仅下载（跳过扫描，基于现有本地清单，需配置 folder/file pattern）
MODE="dl"

# ============ 列表下载配置 ============
# 最终 regex 将拼接为: FOLDER_PATTERN/<line>/FILE_PATTERN
FOLDER_PATTERN="batch1/姿态转换"
FILE_PATTERN="^"
FOLDER_LIST_FILE="${PROJECT_ROOT}/log/tmp_files/filter_lines.txt"

# 并发下载的最大线程数
MAX_WORKERS=4

# 是否跳过确认提示
AUTO_CONFIRM=0

# ============ 解析输入参数（可覆盖默认配置） ============
while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote-path)
      REMOTE_PATH="$2"
      shift 2
      ;;
    --local-root)
      LOCAL_ROOT="$2"
      shift 2
      ;;
    --folder-pattern)
      FOLDER_PATTERN="$2"
      shift 2
      ;;
    --file-pattern)
      FILE_PATTERN="$2"
      shift 2
      ;;
    --folder-list-file)
      FOLDER_LIST_FILE="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --max-workers)
      MAX_WORKERS="$2"
      shift 2
      ;;
    -y|--yes)
      AUTO_CONFIRM=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: 未知参数 $1"
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$FOLDER_LIST_FILE" ]]; then
  echo "ERROR: 文件夹列表不存在: $FOLDER_LIST_FILE"
  exit 1
fi

if [[ "$MODE" != "scan" ]]; then
  if [[ -z "$FOLDER_PATTERN" ]]; then
    echo "ERROR: MODE=$MODE 时必须提供 FOLDER_PATTERN"
    exit 1
  fi
  if [[ -z "$FILE_PATTERN" ]]; then
    echo "ERROR: MODE=$MODE 时必须提供 FILE_PATTERN"
    exit 1
  fi
fi

echo "==============================================="
echo "批量任务配置 (Batch Config)"
echo "==============================================="
echo "远程路径 (Remote):        $REMOTE_PATH"
echo "本地输出 (Local):         $LOCAL_ROOT"
echo "Python脚本 (Script):      $PYTHON_SCRIPT"
echo "文件夹前缀 (Folder base): $FOLDER_PATTERN"
echo "文件匹配 (File pattern):  $FILE_PATTERN"
echo "列表文件 (List file):     $FOLDER_LIST_FILE"
echo "模式 (Mode):              $MODE"
echo "最大线程数 (Workers):     $MAX_WORKERS"
echo ""
echo "列表预览 (前10行):"
head -n 10 "$FOLDER_LIST_FILE" || true
echo ""

if [[ "$AUTO_CONFIRM" -ne 1 ]]; then
  read -r -p "确认执行以上批量任务？[y/N]: " CONFIRM
  case "$CONFIRM" in
    y|Y|yes|YES)
      ;;
    *)
      echo "已取消执行。"
      exit 0
      ;;
  esac
fi

SUCCESS_COUNT=0
TOTAL_COUNT=0

while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  TOTAL_COUNT=$((TOTAL_COUNT + 1))

  BASE_ARGS=(
    "$REMOTE_PATH"
    "$LOCAL_ROOT"
    --mc-bin mc
    --file-list-name "_mc_files.txt"
    --file-count-name "_mc_file_count.txt"
    --summary-name "_mc_summary.json"
  )

  PYTHON_ARGS=("${BASE_ARGS[@]}")
  if [[ "$FILE_PATTERN" == "^" ]]; then
    REGEX_PATTERN="${FOLDER_PATTERN}/${line}"
  else
    REGEX_PATTERN="${FOLDER_PATTERN}/${line}/${FILE_PATTERN}"
  fi

  case "$MODE" in
    scan)
      PYTHON_ARGS+=(--scan-only)
      ;;
    scan-dl)
      PYTHON_ARGS+=(
        --download
        --regex "$REGEX_PATTERN"
        --max-workers "$MAX_WORKERS"
      )
      ;;
    dl)
      PYTHON_ARGS+=(
        --skip-scan
        --download
        --regex "$REGEX_PATTERN"
        --max-workers "$MAX_WORKERS"
      )
      ;;
    *)
      echo "ERROR: 未知的模式 $MODE"
      echo "Valid modes: scan, scan-dl, dl"
      exit 1
      ;;
  esac

  echo "-----------------------------------------------"
  echo "[$TOTAL_COUNT] 当前条目: $line"
  if [[ "$MODE" != "scan" ]]; then
    echo "正则表达式 (Regex): $REGEX_PATTERN"
  fi

  RUN_CMD=(python3 "$PYTHON_SCRIPT" "${PYTHON_ARGS[@]}")
  printf '执行命令 (Command): %q ' "${RUN_CMD[@]}"
  echo ""

  if "${RUN_CMD[@]}"; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
  else
    echo "ERROR: 条目执行失败: $line"
  fi
done < "$FOLDER_LIST_FILE"

echo ""
echo "==============================================="
echo "批量任务完成 (Batch Done)"
echo "总条目 (Total):   $TOTAL_COUNT"
echo "成功条目 (Success): $SUCCESS_COUNT"
echo "失败条目 (Failed): $((TOTAL_COUNT - SUCCESS_COUNT))"
echo "输出目录 (Output): $LOCAL_ROOT"
echo "==============================================="

if [[ "$SUCCESS_COUNT" -ne "$TOTAL_COUNT" ]]; then
  exit 1
fi
