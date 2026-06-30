#!/bin/bash

set -euo pipefail

usage() {
	cat <<EOF
Usage: $(basename "$0") <input_dir> [options]

Description:
	将 input_dir 下的每个一级子文件夹分别压缩为 zip 文件。

Arguments:
	input_dir   需要批量压缩子文件夹的目录。

Options:
	-o, --output-dir <path>     zip 输出目录（可选，默认与 input_dir 相同）。
	-r, --remove-source <bool>  是否在压缩成功后删除对应源文件夹（默认 false）。
	                            支持: true/false, yes/no, 1/0
	-h, --help                  显示帮助

Example:
	$(basename "$0") /data/input --output-dir /data/zips
	$(basename "$0") /data/input -o /data/zips -r true
	$(basename "$0") /data/input
EOF
}

if [[ $# -lt 1 ]]; then
	usage
	exit 1
fi

if ! command -v zip >/dev/null 2>&1; then
	echo "ERROR: 未找到 zip 命令，请先安装。"
	exit 1
fi

INPUT_DIR="$1"
shift

OUTPUT_DIR="$INPUT_DIR"
REMOVE_SOURCE_RAW="false"

while [[ $# -gt 0 ]]; do
	case "$1" in
		-o|--output-dir)
			if [[ -z "${2:-}" ]]; then
				echo "ERROR: --output-dir 需要一个路径参数"
				exit 1
			fi
			OUTPUT_DIR="$2"
			shift 2
			;;
		-r|--remove-source)
			if [[ -z "${2:-}" ]]; then
				echo "ERROR: --remove-source 需要一个布尔参数"
				exit 1
			fi
			REMOVE_SOURCE_RAW="$2"
			shift 2
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			echo "ERROR: 未知参数: $1"
			usage
			exit 1
			;;
	esac
done

case "${REMOVE_SOURCE_RAW,,}" in
	1|true|yes)
		REMOVE_SOURCE=1
		;;
	0|false|no)
		REMOVE_SOURCE=0
		;;
	*)
		echo "ERROR: remove_source 参数无效: $REMOVE_SOURCE_RAW"
		echo "可选值: true/false, yes/no, 1/0"
		exit 1
		;;
esac

if [[ ! -d "$INPUT_DIR" ]]; then
	echo "ERROR: 输入目录不存在: $INPUT_DIR"
	exit 1
fi

mkdir -p "$OUTPUT_DIR"

INPUT_DIR_ABS="$(cd "$INPUT_DIR" && pwd)"
OUTPUT_DIR_ABS="$(cd "$OUTPUT_DIR" && pwd)"

echo "输入目录: $INPUT_DIR_ABS"
echo "输出目录: $OUTPUT_DIR_ABS"
echo "压缩后删除源目录: $REMOVE_SOURCE"

count=0
while IFS= read -r -d '' subdir_path; do
	subdir_name="$(basename "$subdir_path")"
	zip_path="$OUTPUT_DIR_ABS/${subdir_name}.zip"

	echo "压缩: $subdir_name -> $zip_path"
	(
		cd "$INPUT_DIR_ABS"
		zip -r -q "$zip_path" "$subdir_name"
	)
	if [[ $REMOVE_SOURCE -eq 1 ]]; then
		rm -rf "$subdir_path"
		echo "已删除源目录: $subdir_path"
	fi
	count=$((count + 1))
done < <(find "$INPUT_DIR_ABS" -mindepth 1 -maxdepth 1 -type d -print0)

if [[ $count -eq 0 ]]; then
	echo "未发现可压缩的子文件夹。"
else
	echo "完成，共压缩 $count 个子文件夹。"
fi
