#!/usr/bin/env python3
"""Validate whether VideoMimic H5 foot contacts may supervise a frozen GMR clip.

This is intentionally read-only.  It compares the original, published-model
retarget H5 with an immutable GMR result and writes a transfer admission report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import h5py
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runs(mask: np.ndarray, minimum_frames: int) -> List[List[int]]:
    result: List[List[int]] = []
    start: int | None = None
    for index, active in enumerate(mask.tolist() + [False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            if index - start >= minimum_frames:
                result.append([start, index - 1])
            start = None
    return result


def _summary(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {key: float("nan") for key in ("minimum", "p05", "median", "p95", "maximum")}
    return {
        "minimum": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if len(left) != len(right) or len(left) < 3:
        return None
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator < 1e-10:
        return None
    return float(np.dot(left, right) / denominator)


def _overlap(
    left: np.ndarray, right: np.ndarray, offset: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return arrays where h5[t] is compared with gmr[t + offset]."""
    start = max(0, -offset)
    stop = min(len(left), len(right) - offset)
    if stop <= start:
        return np.empty((0,) + left.shape[1:]), np.empty((0,) + right.shape[1:])
    return left[start:stop], right[start + offset : stop + offset]


def _best_offset(
    h5_joints: np.ndarray,
    h5_names: Iterable[str],
    gmr_joints: np.ndarray,
    gmr_names: Iterable[str],
    search_radius: int,
) -> Dict[str, Any]:
    h5_index = {str(name): index for index, name in enumerate(h5_names)}
    gmr_index = {str(name): index for index, name in enumerate(gmr_names)}
    common = [name for name in h5_index if name in gmr_index]
    if not common:
        raise ValueError("H5 and GMR have no common joint names")

    candidates: List[Dict[str, Any]] = []
    for offset in range(-search_radius, search_radius + 1):
        correlations: List[float] = []
        for name in common:
            h5_value, gmr_value = _overlap(
                np.diff(h5_joints[:, h5_index[name]]),
                np.diff(gmr_joints[:, gmr_index[name]]),
                offset,
            )
            correlation = _correlation(h5_value, gmr_value)
            if correlation is not None:
                correlations.append(correlation)
        candidates.append(
            {
                "gmr_frame_minus_h5_frame": offset,
                "mean_joint_velocity_correlation": float(np.mean(correlations))
                if correlations
                else float("-inf"),
                "joint_count": len(correlations),
            }
        )
    best = max(candidates, key=lambda item: item["mean_joint_velocity_correlation"])
    return {
        "common_joint_names": common,
        "best": best,
        "candidates": candidates,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact-h5", required=True, type=Path)
    parser.add_argument("--robot-motion", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--minimum-contact-run", type=int, default=3)
    parser.add_argument("--offset-search-radius", type=int, default=4)
    parser.add_argument("--max-accepted-offset-frames", type=int, default=1)
    parser.add_argument("--min-joint-velocity-correlation", type=float, default=0.35)
    parser.add_argument("--max-h5-contact-speed-m-s", type=float, default=0.20)
    args = parser.parse_args()
    if args.minimum_contact_run < 2:
        parser.error("--minimum-contact-run must be at least two")
    if args.offset_search_radius < 0 or args.max_accepted_offset_frames < 0:
        parser.error("offset parameters must be non-negative")
    if not -1.0 <= args.min_joint_velocity_correlation <= 1.0:
        parser.error("--min-joint-velocity-correlation must be in [-1, 1]")
    if args.max_h5_contact_speed_m_s <= 0.0:
        parser.error("--max-h5-contact-speed-m-s must be positive")
    return args


def main() -> None:
    args = _parse_args()
    h5_path = args.contact_h5.resolve()
    motion_path = args.robot_motion.resolve()
    if not h5_path.is_file() or not motion_path.is_file():
        raise FileNotFoundError("contact H5 or robot motion is missing")

    with motion_path.open("rb") as stream:
        motion = pickle.load(stream)
    if not isinstance(motion, dict):
        raise ValueError("robot motion must be a dictionary")
    gmr_joints = np.asarray(motion["dof_pos"], dtype=np.float64)
    gmr_names = [str(value) for value in motion["dof_names"]]
    gmr_frames = len(gmr_joints)
    gmr_fps = float(motion["fps"])
    if gmr_joints.ndim != 2 or len(gmr_names) != gmr_joints.shape[1]:
        raise ValueError("GMR dof names and positions are inconsistent")

    with h5py.File(h5_path, "r") as h5:
        required = (
            "joints",
            "link_pos",
            "contacts/left_foot",
            "contacts/right_foot",
        )
        missing = [name for name in required if name not in h5]
        if missing:
            raise ValueError("contact H5 is missing: " + ", ".join(missing))
        h5_joints = np.asarray(h5["joints"], dtype=np.float64)
        h5_link_pos = np.asarray(h5["link_pos"], dtype=np.float64)
        h5_names = [str(value) for value in h5.attrs["joint_names"]]
        h5_link_names = [str(value) for value in h5.attrs["link_names"]]
        h5_fps = float(h5.attrs["fps"])
        contacts = {
            "left": np.asarray(h5["contacts/left_foot"], dtype=bool),
            "right": np.asarray(h5["contacts/right_foot"], dtype=bool),
        }

    h5_frames = len(h5_joints)
    if h5_joints.shape != (h5_frames, len(h5_names)):
        raise ValueError("H5 joint names and positions are inconsistent")
    if h5_link_pos.shape[:2] != (h5_frames, len(h5_link_names)):
        raise ValueError("H5 link names and positions are inconsistent")
    if any(values.shape != (h5_frames,) for values in contacts.values()):
        raise ValueError("H5 contact arrays do not match its frame count")
    if not np.isfinite(h5_fps) or h5_fps <= 0.0 or not np.isfinite(gmr_fps) or gmr_fps <= 0.0:
        raise ValueError("H5 and GMR fps must be finite and positive")

    temporal = _best_offset(
        h5_joints,
        h5_names,
        gmr_joints,
        gmr_names,
        args.offset_search_radius,
    )
    link_index = {name: index for index, name in enumerate(h5_link_names)}
    foot_links = {"left": "left_ankle_roll_link", "right": "right_ankle_roll_link"}
    h5_foot_report: Dict[str, Any] = {}
    for side, link_name in foot_links.items():
        if link_name not in link_index:
            raise ValueError("H5 link is missing: " + link_name)
        position = h5_link_pos[:, link_index[link_name], :2]
        speed = np.zeros(h5_frames, dtype=np.float64)
        if h5_frames > 1:
            speed[1:] = np.linalg.norm(np.diff(position, axis=0), axis=1) * h5_fps
            speed[:-1] = np.maximum(speed[:-1], speed[1:])
        active_speed = speed[contacts[side]]
        h5_foot_report[side] = {
            "h5_link": link_name,
            "contact_frame_count": int(np.count_nonzero(contacts[side])),
            "contact_segments": _runs(contacts[side], args.minimum_contact_run),
            "horizontal_speed_during_h5_contact_m_s": _summary(active_speed),
        }

    reasons: List[str] = []
    if h5_frames != gmr_frames:
        reasons.append("frame_count_mismatch")
    if not np.isclose(h5_fps, gmr_fps, rtol=0.0, atol=1e-6):
        reasons.append("fps_mismatch")
    best = temporal["best"]
    if abs(int(best["gmr_frame_minus_h5_frame"])) > args.max_accepted_offset_frames:
        reasons.append("temporal_offset_exceeds_limit")
    if best["mean_joint_velocity_correlation"] < args.min_joint_velocity_correlation:
        reasons.append("weak_joint_temporal_alignment")
    for side, details in h5_foot_report.items():
        if not details["contact_segments"]:
            reasons.append(side + "_has_no_contact_segment")
        speed_p95 = details["horizontal_speed_during_h5_contact_m_s"]["p95"]
        if np.isfinite(speed_p95) and speed_p95 > args.max_h5_contact_speed_m_s:
            reasons.append(side + "_h5_contact_is_not_stationary")

    report = {
        "schema_version": 1,
        "purpose": "h5_to_frozen_gmr_contact_transfer_admission",
        "contact_h5": str(h5_path),
        "contact_h5_sha256": _sha256(h5_path),
        "robot_motion": str(motion_path),
        "robot_motion_sha256": _sha256(motion_path),
        "robot_motion_modified": False,
        "frame_contract": {
            "h5_frames": h5_frames,
            "gmr_frames": gmr_frames,
            "h5_fps": h5_fps,
            "gmr_fps": gmr_fps,
        },
        "joint_contract": temporal,
        "side_mapping": {
            "h5_left_foot": "GMR left_ankle_roll_link",
            "h5_right_foot": "GMR right_ankle_roll_link",
            "validated_by_shared_joint_names": True,
        },
        "h5_contacts": h5_foot_report,
        "quality_gate": {
            "maximum_accepted_offset_frames": args.max_accepted_offset_frames,
            "minimum_joint_velocity_correlation": args.min_joint_velocity_correlation,
            "maximum_h5_contact_speed_m_s": args.max_h5_contact_speed_m_s,
        },
        "accepted_for_contact_transfer": not reasons,
        "rejection_reasons": sorted(set(reasons)),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"accepted": report["accepted_for_contact_transfer"], "reasons": report["rejection_reasons"]}))


if __name__ == "__main__":
    main()
