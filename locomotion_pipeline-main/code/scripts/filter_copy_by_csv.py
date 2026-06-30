#!/usr/bin/env python3

import argparse
import re
import shutil
import sys
from pathlib import Path


def _setup_import_path() -> None:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_setup_import_path()

from code.utils.statics import SequenceStatistics  # noqa: E402


def get_prefix_without_trailing_digits(name: str) -> str:
    return re.sub(r"\d+$", "", name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "读取源目录下每个子目录中的 CSV，按平均合格率筛选后复制到目标目录，"
            "并按去尾号前缀去重。"
        )
    )
    parser.add_argument("source_dir", type=Path, help="源目录（包含多个子目录）")
    parser.add_argument("target_dir", type=Path, help="目标目录")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="合格率阈值，范围 [0, 1]，默认 0.9",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="sequence_statistics.csv",
        help="每个子目录内要读取的 CSV 文件名",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=90,
        help="最小总帧数阈值，小于该值会被跳过，默认 90",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印结果，不执行复制",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.source_dir.is_dir():
        raise ValueError(f"source_dir 不存在或不是目录: {args.source_dir}")
    if args.threshold < 0 or args.threshold > 1:
        raise ValueError("threshold 必须在 [0, 1] 区间")
    if args.min_frames < 0:
        raise ValueError("min_frames 必须为非负整数")


def load_existing_prefixes(target_dir: Path) -> set[str]:
    prefixes = set()
    if not target_dir.exists():
        return prefixes

    for child in target_dir.iterdir():
        if child.is_dir():
            prefixes.add(get_prefix_without_trailing_digits(child.name))
    return prefixes


def compute_average_rate_and_frames(csv_path: Path) -> tuple[float, int]:
    stats = SequenceStatistics()
    stats.load_from_csv(str(csv_path))
    summary = stats.get_summary()
    return stats.get_average_qualified_rate(), int(summary.get("total_count", 0))


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    source_dir = args.source_dir.resolve()
    target_dir = args.target_dir.resolve()
    csv_name = args.csv_name
    threshold = args.threshold
    min_frames = args.min_frames

    existing_prefixes = load_existing_prefixes(target_dir)
    copied_prefixes = set(existing_prefixes)

    total_dirs = 0
    copied_dirs = 0
    skipped_prefix = 0
    skipped_threshold = 0
    skipped_frames = 0
    skipped_no_csv = 0
    skipped_copy_error = 0

    subdirs = sorted([p for p in source_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    print(f"source_dir: {source_dir}")
    print(f"target_dir: {target_dir}")
    print(f"threshold : {threshold:.4f}")
    print(f"min_frames: {min_frames}")
    print(f"csv_name  : {csv_name}")
    print(f"dry_run   : {args.dry_run}")
    print("-")

    if not args.dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)

    for subdir in subdirs:
        total_dirs += 1
        prefix = get_prefix_without_trailing_digits(subdir.name)

        if prefix in copied_prefixes:
            skipped_prefix += 1
            print(f"[SKIP prefix] {subdir.name}")
            continue

        csv_path = subdir / csv_name
        if not csv_path.is_file():
            skipped_no_csv += 1
            print(f"[SKIP no csv] {subdir.name}")
            continue

        try:
            avg_rate, total_frames = compute_average_rate_and_frames(csv_path)
        except Exception as exc:
            skipped_no_csv += 1
            print(f"[SKIP bad csv] {subdir.name} ({exc})")
            continue

        if total_frames < min_frames:
            skipped_frames += 1
            print(
                f"[SKIP frames] {subdir.name} frames={total_frames} < {min_frames}"
            )
            continue

        if avg_rate < threshold:
            skipped_threshold += 1
            print(
                f"[SKIP rate] {subdir.name} rate={avg_rate:.4f}, frames={total_frames}"
            )
            continue

        dst = target_dir / subdir.name
        if dst.exists():
            skipped_copy_error += 1
            print(f"[SKIP exists] {subdir.name}")
            continue

        print(f"[COPY] {subdir.name} rate={avg_rate:.4f}, frames={total_frames}")
        if not args.dry_run:
            try:
                shutil.copytree(subdir, dst)
            except Exception as exc:
                skipped_copy_error += 1
                print(f"[SKIP copy err] {subdir.name} ({exc})")
                continue

        copied_dirs += 1
        copied_prefixes.add(prefix)

    print("-")
    print("Summary:")
    print(f"  total subdirs   : {total_dirs}")
    print(f"  copied          : {copied_dirs}")
    print(f"  skip prefix     : {skipped_prefix}")
    print(f"  skip threshold  : {skipped_threshold}")
    print(f"  skip frames     : {skipped_frames}")
    print(f"  skip csv issues : {skipped_no_csv}")
    print(f"  skip copy issues: {skipped_copy_error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
