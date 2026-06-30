#!/bin/bash

# 使用 mc ls 扫描远程路径，在本地创建目录骨架，并统计文件清单和数量
# 支持按正则表达式过滤并下载指定文件

set -e

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 项目根目录（上两级）
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --remote-path <path>     覆盖 REMOTE_PATH
  --local-root <path>      覆盖 LOCAL_ROOT
  --regex-pattern <regex>  覆盖 REGEX_PATTERN
  --mode <scan|scan-dl|dl> 覆盖 MODE
  --max-workers <number>   覆盖 MAX_WORKERS
  -y, --yes                自动确认并执行
  -h, --help               显示帮助
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
#   scan-dl  - 扫描并下载（需配置 REGEX_PATTERN）
#   dl       - 仅下载（跳过扫描，基于现有本地清单，需配置REGEX_PATTERN）
# ! 请根据需要修改 MODE 变量
MODE="dl"

# ============ 下载配置 ============
# 正则表达式：匹配需要下载的文件（MODE 为 scan-dl 或 dl 时生效）
# 示例：
#   "\.zip$"                 - 匹配所有 .zip 文件
#   "batch1.*\.zip$"         - 匹配 batch1 目录下的 .zip 文件
#   "^[^_].*\.(zip|tar)$"    - 匹配非下划线开头的 .zip 或 .tar 文件
# ! 请根据需要修改 REGEX_PATTERN 变量
REGEX_PATTERN="batch1/姿态转换/104010_A_Day_In_The_Life_of_a_860/104010_A_Day_In_The_Life_of_a03.mp4"

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
    --regex-pattern)
      REGEX_PATTERN="$2"
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

# ============ 构建命令行参数 ============
PYTHON_ARGS=(
  "$REMOTE_PATH"
  "$LOCAL_ROOT"
  --mc-bin mc
  --file-list-name "_mc_files.txt"
  --file-count-name "_mc_file_count.txt"
  --summary-name "_mc_summary.json"
)

# 根据模式添加参数
case "$MODE" in
  scan)
    echo "==============================================="
    echo "模式：仅扫描 (Mode: Scan Only)"
    echo "==============================================="
    PYTHON_ARGS+=(--scan-only)
    ;;
  scan-dl)
    echo "==============================================="
    echo "模式：扫描+下载 (Mode: Scan + Download)"
    echo "==============================================="
    echo "正则表达式 (Regex):   $REGEX_PATTERN"
    echo "最大线程数 (Workers): $MAX_WORKERS"
    echo ""
    if [[ -z "$REGEX_PATTERN" ]]; then
      echo "ERROR: MODE=$MODE 时必须提供 REGEX_PATTERN"
      echo "可通过 --regex-pattern 传入"
      exit 1
    fi
    PYTHON_ARGS+=(
      --download
      --regex "$REGEX_PATTERN"
      --max-workers $MAX_WORKERS
    )
    ;;
  dl)
    echo "==============================================="
    echo "模式：仅下载 (Mode: Download Only)"
    echo "==============================================="
    echo "正则表达式 (Regex):   $REGEX_PATTERN"
    echo "最大线程数 (Workers): $MAX_WORKERS"
    echo ""
    if [[ -z "$REGEX_PATTERN" ]]; then
      echo "ERROR: MODE=$MODE 时必须提供 REGEX_PATTERN"
      echo "可通过 --regex-pattern 传入"
      exit 1
    fi
    PYTHON_ARGS+=(
      --skip-scan
      --download
      --regex "$REGEX_PATTERN"
      --max-workers $MAX_WORKERS
    )
    ;;
  *)
    echo "ERROR: 未知的模式 $MODE"
    echo "ERROR: Unknown mode $MODE"
    echo "Valid modes: scan, scan-dl, dl"
    exit 1
    ;;
esac

echo "远程路径 (Remote):   $REMOTE_PATH"
echo "本地输出 (Local):    $LOCAL_ROOT"
echo "Python脚本 (Script): $PYTHON_SCRIPT"
echo ""

RUN_CMD=(python3 "$PYTHON_SCRIPT" "${PYTHON_ARGS[@]}")
echo "即将执行命令 (Command to run):"
printf '  %q ' "${RUN_CMD[@]}"
echo ""

if [[ "$AUTO_CONFIRM" -ne 1 ]]; then
  read -r -p "确认执行？[y/N]: " CONFIRM
  case "$CONFIRM" in
    y|Y|yes|YES)
      ;;
    *)
      echo "已取消执行。"
      exit 0
      ;;
  esac
fi
echo ""

# 调用 Python 脚本
if "${RUN_CMD[@]}"; then
  echo ""
  echo "✓ 任务完成 (Done)!"
  echo ""
  echo "输出目录结构 (Output structure):"
  echo "  $LOCAL_ROOT/"
  echo "    ├─ ... (所有远程目录对应的本地文件夹 / local folders)"
  echo "    ├─ _mc_files.txt (每个目录下的文件清单 / file list)"
  echo "    ├─ _mc_file_count.txt (每个目录中的文件数 / file count)"
  echo "    └─ _mc_summary.json (汇总统计信息 / summary)"
else
  echo ""
  echo "✗ 过程中出错 (Error occurred)!"
  exit 1
fi

