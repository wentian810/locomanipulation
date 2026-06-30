#!/usr/bin/env python3
"""Fail when a public source checkout is incomplete or contains private assets."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


REQUIRED_PATHS = (
    "README.md",
    "PIPELINE_README.md",
    "PIPELINE_ENGINEERING.md",
    "THIRD_PARTY.md",
    "configs/default.yaml",
    "configs/pipelines/rgb_monocular_sharpa.yaml",
    "configs/secrets.env.example",
    "scripts/run_pipeline_from_config.py",
    "scripts/bootstrap_external_repos.sh",
    "GMR-master/scripts/adapt_do_as_i_do_object.py",
    "GMR-master/scripts/build_object_collision_asset.py",
    "GMR-master/scripts/import_foundationpose_object.py",
    "GMR-master/scripts/simulate_robot_object_contacts.py",
    "GVHMR-hand/GVHMR-main/tools/pipeline/run_pipeline.sh",
    "GVHMR-hand/GVHMR-main/tools/pipeline/run_object_reconstruction_bridge.py",
    "locomotion_pipeline-main/run_optim_v2.sh",
    "phc-dev-felix-pipeline/phc/run_hydra.py",
    "do-as-i-do-main/reconstruction/run_pipeline.sh",
    "foundationpose-plus-plus-main/scripts/hunyuan3d_bridge.py",
)

FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".pth",
    ".pt",
    ".pkl",
    ".npz",
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".tar",
    ".gz",
    ".7z",
}

FORBIDDEN_PARTS = {
    "output_dir",
    "dataset",
    "datasets",
    "weights",
    "checkpoints",
    "mano_v1_2",
    "isaacgym",
    "__pycache__",
}

SECRET_PATTERNS = (
    re.compile(r"AKID[A-Za-z0-9]{16,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"BEGIN (?:RSA|OPENSSH|EC|DSA) PRIVATE KEY"),
)

ALLOWED_PLACEHOLDERS = {
    "REPLACE_WITH_ROTATED_SECRET_ID",
    "REPLACE_WITH_ROTATED_SECRET_KEY",
}

TENCENT_ASSIGNMENT = re.compile(
    r"(?:export\s+)?TENCENT(?:CLOUD)?_SECRET_(?:ID|KEY)"
    r"\s*=\s*['\"]([^'\"]+)['\"]"
)


def git_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [
        root / os.fsdecode(item)
        for item in result.stdout.split(b"\0")
        if item
    ]


def gitlinks(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--stage"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        line.split("\t", 1)[1]
        for line in result.stdout.splitlines()
        if line.startswith("160000 ")
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    root = args.root.resolve()
    errors: list[str] = []

    for relative in REQUIRED_PATHS:
        if not (root / relative).is_file():
            errors.append(f"missing required source: {relative}")

    files = git_files(root)
    links = gitlinks(root)
    if links:
        errors.append(
            "embedded gitlinks are not reproducible; use bootstrap instead: "
            + ", ".join(links)
        )

    for path in files:
        relative = path.relative_to(root)
        lower_parts = {part.lower() for part in relative.parts}
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden generated/restricted file: {relative}")
        if lower_parts & FORBIDDEN_PARTS:
            errors.append(f"forbidden generated/restricted path: {relative}")
        if path.is_file() and path.stat().st_size >= 95 * 1024 * 1024:
            errors.append(f"file is too large for a source repository: {relative}")
        if not path.is_file() or path.stat().st_size > 5 * 1024 * 1024:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                errors.append(f"possible credential in {relative}: {pattern.pattern}")
        for line_number, line in enumerate(text.splitlines(), 1):
            assignment = TENCENT_ASSIGNMENT.search(line)
            if assignment is None:
                continue
            value = assignment.group(1)
            if value == "..." or value in ALLOWED_PLACEHOLDERS:
                continue
            errors.append(
                f"possible Tencent credential assignment: {relative}:{line_number}"
            )

    if errors:
        print("PUBLIC REPOSITORY AUDIT FAILED", file=sys.stderr)
        for item in sorted(set(errors)):
            print(f"- {item}", file=sys.stderr)
        return 1

    total_size = sum(path.stat().st_size for path in files if path.is_file())
    print(
        "PUBLIC REPOSITORY AUDIT PASSED\n"
        f"tracked_files={len(files)}\n"
        f"tracked_bytes={total_size}\n"
        "external repositories and licensed model assets are intentionally "
        "restored after clone"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
