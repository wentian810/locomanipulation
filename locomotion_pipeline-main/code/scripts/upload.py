#!/usr/bin/env python3
"""Upload qualified sequence folders based on per-sequence CSV statistics.

Given a local chunk directory such as:
  output_runs/batch1/zitai/zitai_n10000_idx000

This script will:
1. Scan each sequence folder under the chunk directory.
2. Read `sequence_statistics.csv` inside each sequence folder.
3. Keep only sequences whose qualified rate >= threshold.
4. Upload those sequence folders to the corresponding remote path via `mc cp`.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


DEFAULT_REMOTE_BASE = "syna/synadata-source-wlcb/output/shenbipai"


@dataclass
class SequenceInfo:
	name: str
	local_dir: Path
	csv_path: Path
	qualified_rate: float
	total: int
	qualified: int


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Read per-sequence sequence_statistics.csv under a local output chunk, "
			"filter by qualified rate threshold, and upload passing sequences to remote via mc."
		)
	)
	parser.add_argument(
		"chunk_dir",
		type=Path,
		help="Local chunk folder, e.g. output_runs/batch1/zitai/zitai_n10000_idx000",
	)
	parser.add_argument(
		"--threshold",
		type=float,
		default=0.95,
		help="Minimum qualified rate to pass (0~1). Default: 0.95",
	)
	parser.add_argument(
		"--remote-base",
		type=str,
		default=DEFAULT_REMOTE_BASE,
		help=(
			"Remote base path for uploads (mc alias path). "
			f"Default: {DEFAULT_REMOTE_BASE}"
		),
	)
	parser.add_argument(
		"--output-runs-root",
		type=Path,
		default=Path("output_runs"),
		help=(
			"Local root corresponding to remote base. Relative subpath from this root "
			"to chunk_dir is appended to remote-base. Default: output_runs"
		),
	)
	parser.add_argument(
		"--mc-bin",
		type=str,
		default="mc",
		help="mc executable name/path. Default: mc",
	)
	parser.add_argument(
		"--workers",
		type=int,
		default=4,
		help="Max concurrent uploads. Default: 4",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Only print selected/upload targets, do not execute mc commands",
	)
	return parser.parse_args()


def parse_rate_from_row(row: dict) -> Tuple[float, int, int]:
	total = int(row.get("Total", "0") or 0)
	qualified = int(row.get("Qualified", "0") or 0)

	rate_field = str(row.get("Qualified Rate", "")).strip()
	if rate_field.endswith("%"):
		rate = float(rate_field[:-1]) / 100.0
	elif rate_field:
		rate = float(rate_field)
	else:
		rate = (qualified / total) if total > 0 else 0.0

	return rate, total, qualified


def pick_sequence_row(csv_path: Path, seq_name: str) -> Optional[dict]:
	with csv_path.open("r", newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		rows = [r for r in reader if r.get("Sequence ID") and r.get("Sequence ID") != "总体平均"]

	if not rows:
		return None

	for row in rows:
		if row["Sequence ID"] == seq_name:
			return row

	# Most per-sequence csv files contain exactly one row; fallback to first valid row.
	return rows[0]


def collect_sequence_infos(chunk_dir: Path) -> Tuple[List[SequenceInfo], List[str]]:
	infos: List[SequenceInfo] = []
	errors: List[str] = []

	for seq_dir in sorted(chunk_dir.iterdir()):
		if not seq_dir.is_dir():
			continue
		csv_path = seq_dir / "sequence_statistics.csv"
		if not csv_path.exists():
			continue

		try:
			row = pick_sequence_row(csv_path, seq_dir.name)
			if row is None:
				errors.append(f"No valid data row in csv: {csv_path}")
				continue

			rate, total, qualified = parse_rate_from_row(row)
			infos.append(
				SequenceInfo(
					name=seq_dir.name,
					local_dir=seq_dir,
					csv_path=csv_path,
					qualified_rate=rate,
					total=total,
					qualified=qualified,
				)
			)
		except Exception as exc:
			errors.append(f"Failed parsing {csv_path}: {exc}")

	return infos, errors


def build_remote_chunk_path(chunk_dir: Path, output_runs_root: Path, remote_base: str) -> str:
	chunk_abs = chunk_dir.resolve()
	root_abs = output_runs_root.resolve()

	try:
		rel = chunk_abs.relative_to(root_abs)
		rel_posix = rel.as_posix()
	except ValueError:
		rel_posix = chunk_dir.name

	return f"{remote_base.rstrip('/')}/{rel_posix}" if rel_posix else remote_base.rstrip("/")


def ensure_remote_dir(mc_bin: str, remote_dir: str, dry_run: bool) -> None:
	cmd = [mc_bin, "mb", "--ignore-existing", remote_dir]
	if dry_run:
		print("[dry-run]", " ".join(cmd))
		return

	subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def upload_one(mc_bin: str, seq_dir: Path, remote_chunk_dir: str, dry_run: bool) -> Tuple[bool, str]:
	cmd = [mc_bin, "cp", "--recursive", str(seq_dir), f"{remote_chunk_dir}/"]
	if dry_run:
		return True, "[dry-run] " + " ".join(cmd)

	try:
		subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
		return True, " ".join(cmd)
	except subprocess.CalledProcessError as exc:
		err = exc.stderr.strip() or exc.stdout.strip() or str(exc)
		return False, f"{seq_dir.name}: {err}"


def main() -> int:
	args = parse_args()
	chunk_dir = args.chunk_dir.expanduser().resolve()
	output_runs_root = args.output_runs_root.expanduser().resolve()

	if not chunk_dir.exists() or not chunk_dir.is_dir():
		print(f"ERROR: chunk_dir does not exist or is not a directory: {chunk_dir}", file=sys.stderr)
		return 1

	if not (0.0 <= args.threshold <= 1.0):
		print("ERROR: --threshold must be in [0, 1]", file=sys.stderr)
		return 1

	if args.workers <= 0:
		print("ERROR: --workers must be > 0", file=sys.stderr)
		return 1

	infos, parse_errors = collect_sequence_infos(chunk_dir)
	if parse_errors:
		print("Parse warnings:")
		for err in parse_errors:
			print(f"  - {err}")

	if not infos:
		print("No sequence_statistics.csv found under sequence folders, nothing to upload.")
		return 0

	passed = [x for x in infos if x.qualified_rate >= args.threshold]
	remote_chunk_dir = build_remote_chunk_path(chunk_dir, output_runs_root, args.remote_base)

	print("=" * 60)
	print(f"Local chunk dir:   {chunk_dir}")
	print(f"Remote chunk dir:  {remote_chunk_dir}")
	print(f"Threshold:         {args.threshold:.2%}")
	print(f"Found sequences:   {len(infos)}")
	print(f"Passed sequences:  {len(passed)}")
	print("=" * 60)

	if not passed:
		print("No sequence passes threshold, skip upload.")
		return 0

	try:
		ensure_remote_dir(args.mc_bin, remote_chunk_dir, args.dry_run)
	except Exception as exc:
		print(f"ERROR: failed to create/check remote folder: {exc}", file=sys.stderr)
		return 1

	print("Passing sequences:")
	for info in passed:
		print(
			f"  - {info.name}: {info.qualified_rate:.2%} "
			f"({info.qualified}/{info.total})"
		)

	success = 0
	failed = 0
	failed_msgs: List[str] = []

	with ThreadPoolExecutor(max_workers=args.workers) as executor:
		futures = [
			executor.submit(upload_one, args.mc_bin, info.local_dir, remote_chunk_dir, args.dry_run)
			for info in passed
		]
		for fut in as_completed(futures):
			ok, msg = fut.result()
			if ok:
				success += 1
				print(f"[OK] {msg}")
			else:
				failed += 1
				failed_msgs.append(msg)
				print(f"[FAIL] {msg}", file=sys.stderr)

	print("=" * 60)
	print(f"Upload done: success={success}, failed={failed}")
	print("=" * 60)

	if failed_msgs:
		print("Failed uploads:")
		for msg in failed_msgs:
			print(f"  - {msg}")
		return 1

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
