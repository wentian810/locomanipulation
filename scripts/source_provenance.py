#!/usr/bin/env python3
"""Describe the relation between a human evaluation track and GMR input.

The producer's robot_motion.pkl is trusted only inside the pipeline workspace.
It already records source_motion; this module resolves that path (including the
auto-mode selected.npz symlink) without pretending that it is necessarily the
same track used for human/hand evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


MOTION_STAGE_BY_FILENAME = {
    "001_converted.npz": "converted",
    "001_smoothed.npz": "smoothed",
    "001_contact_stabilized.npz": "contact_stabilized",
    "001_phc.npz": "phc",
    "001_phc_smoothed.npz": "phc_smoothed",
    "001_phc_smoothed_grounded.npz": "phc_smoothed_grounded",
    "001_phc_grounded.npz": "phc_grounded",
    "001_final.npz": "final",
}


def _display_path(clip_dir: Path, path: Path) -> str:
    """Return a portable clip-relative path, or only a basename."""
    try:
        return path.resolve(strict=False).relative_to(
            clip_dir.resolve(strict=False)
        ).as_posix()
    except ValueError:
        return path.name


def _source_motion_value(motion: Mapping[str, Any]) -> str | None:
    value = motion.get("source_motion")
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).strip()
    return text or None


def _resolve_robot_source(clip_dir: Path, raw: str) -> tuple[Path, Path]:
    declared = Path(raw).expanduser()
    if not declared.is_absolute():
        declared = clip_dir / declared
    return declared, declared.resolve(strict=False)


def build_source_provenance(
    clip_dir: Path,
    human_path: Path,
    human_stage: str,
    robot_motion: Mapping[str, Any],
) -> dict[str, Any]:
    """Return portable provenance and a conservative track comparison.

    comparison.status is only match when both resolved paths exist and are
    exactly identical. Missing/legacy inputs are unknown rather than guessed
    from a filename.
    """
    human_resolved = human_path.resolve(strict=False)
    human_exists = human_resolved.is_file()
    human = {
        "status": "resolved" if human_exists else "missing",
        "stage": str(human_stage),
        "path": _display_path(clip_dir, human_path),
        "resolved_path": _display_path(clip_dir, human_resolved),
    }

    raw = _source_motion_value(robot_motion)
    if raw is None:
        robot = {
            "status": "legacy_config_fallback",
            "declared_path": None,
            "resolved_path": None,
            "stage": None,
        }
        comparison = {
            "status": "unknown",
            "reason": "robot_motion_source_motion_missing",
        }
        return {
            "human_evaluation": human,
            "robot_input": robot,
            "comparison": comparison,
        }

    declared, resolved = _resolve_robot_source(clip_dir, raw)
    robot_exists = resolved.is_file()
    robot = {
        "status": "resolved" if robot_exists else "missing",
        "declared_path": _display_path(clip_dir, declared),
        "resolved_path": _display_path(clip_dir, resolved),
        "stage": MOTION_STAGE_BY_FILENAME.get(resolved.name, "unknown"),
    }
    if not human_exists:
        comparison = {
            "status": "unknown",
            "reason": "human_evaluation_source_missing",
        }
    elif not robot_exists:
        comparison = {
            "status": "unknown",
            "reason": "robot_motion_source_motion_missing_or_unreadable",
        }
    elif human_resolved == resolved:
        comparison = {
            "status": "match",
            "reason": "resolved_source_paths_match",
        }
    else:
        comparison = {
            "status": "mismatch",
            "reason": "human_evaluation_and_robot_input_are_distinct",
        }
    return {
        "human_evaluation": human,
        "robot_input": robot,
        "comparison": comparison,
    }
