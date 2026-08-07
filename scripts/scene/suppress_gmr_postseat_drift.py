#!/usr/bin/env python3
"""Suppress a verified late seated root-depth drift without touching joints.

This is a reviewable, low-dimensional visual-motion correction.  It only
modifies root_pos after an externally detected seat-settle event; all root
orientation and joint trajectories remain byte-for-byte unchanged.  Batch
callers must supply event frames from the upstream contact/state classifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def primitive(payload: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    for item in payload.get("primitives", []):
        if str(item.get("name")) in names:
            return item
    raise ValueError(f"missing any primitive named {names}")


def quintic_segment(p0: float, v0: float, a0: float, p1: float, length: int) -> np.ndarray:
    """Return p[0:length+1] with p/v/a continuity at the start and rest at end."""
    if length < 2:
        raise ValueError("transition needs at least two frames")
    c0, c1, c2 = p0, v0, 0.5 * a0
    T = float(length)
    rhs = np.asarray(
        [
            p1 - (c0 + c1 * T + c2 * T * T),
            -(c1 + 2.0 * c2 * T),
            -2.0 * c2,
        ],
        dtype=np.float64,
    )
    matrix = np.asarray(
        [
            [T**3, T**4, T**5],
            [3.0 * T**2, 4.0 * T**3, 5.0 * T**4],
            [6.0 * T, 12.0 * T**2, 20.0 * T**3],
        ],
        dtype=np.float64,
    )
    c3, c4, c5 = np.linalg.solve(matrix, rhs)
    t = np.arange(length + 1, dtype=np.float64)
    return c0 + c1 * t + c2 * t**2 + c3 * t**3 + c4 * t**4 + c5 * t**5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-motion", required=True, type=Path)
    parser.add_argument("--output-motion", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--chair-primitives", required=True, type=Path)
    parser.add_argument("--transition-start", required=True, type=int)
    parser.add_argument("--anchor-frame", required=True, type=int)
    parser.add_argument("--transition-end", required=True, type=int)
    parser.add_argument("--tail-drift-m", type=float, default=0.015)
    parser.add_argument("--max-correction-m", type=float, default=0.12)
    args = parser.parse_args()

    source_hash = sha256(args.input_motion)
    with args.input_motion.open("rb") as handle:
        motion = pickle.load(handle)
    if not isinstance(motion, dict):
        raise TypeError("motion must be a dictionary with root_pos/root_rot/dof_pos")
    root = np.asarray(motion["root_pos"], dtype=np.float64)
    if root.ndim != 2 or root.shape[1] != 3:
        raise ValueError(f"unexpected root_pos shape {root.shape}")
    frame_count = len(root)
    start, anchor, end = args.transition_start, args.anchor_frame, args.transition_end
    if not (2 <= start <= anchor < end < frame_count):
        raise ValueError("require 2 <= transition-start <= anchor-frame < transition-end < frame_count")
    if args.tail_drift_m < 0.0:
        raise ValueError("tail-drift-m is measured toward the backrest and must be non-negative")

    payload = json.loads(args.chair_primitives.read_text(encoding="utf-8"))
    seat = np.asarray(primitive(payload, ("seat_support", "seat"))["center"], dtype=np.float64)
    back = np.asarray(primitive(payload, ("backrest", "back"))["center"], dtype=np.float64)
    back_axis = back[:2] - seat[:2]
    back_axis /= np.linalg.norm(back_axis)

    original_projection = root[:, :2] @ back_axis
    target_projection = float(original_projection[anchor] + args.tail_drift_m)
    raw_tail_drift = float(original_projection[-1] - original_projection[anchor])
    requested_correction = float(original_projection[-1] - target_projection)
    if requested_correction <= 0.0:
        raise RuntimeError("tail does not drift toward the backrest; no correction was written")
    if requested_correction > args.max_correction_m:
        raise RuntimeError(
            f"requested {requested_correction:.4f} m correction exceeds safety bound {args.max_correction_m:.4f} m"
        )

    # Monotone C2 settle curve: a no-overshoot path is more important here
    # than preserving the depth-drift velocity that we are rejecting.
    desired_projection = original_projection.copy()
    desired_projection[start : end + 1] = quintic_segment(
        float(original_projection[start]), 0.0, 0.0, target_projection, end - start
    )
    desired_projection[end + 1 :] = target_projection
    correction = desired_projection - original_projection
    corrected_root = root.copy()
    corrected_root[:, :2] += correction[:, None] * back_axis[None, :]

    updated = dict(motion)
    updated["root_pos"] = corrected_root.astype(np.asarray(motion["root_pos"]).dtype, copy=False)
    updated["postseat_drift_suppression"] = {
        "version": "v10",
        "policy": "C2 root-only correction along semantic seat-to-backrest axis after externally detected settle event",
        "joint_trajectory_modified": False,
        "root_orientation_modified": False,
        "transition_frames": [int(start), int(end)],
        "anchor_frame": int(anchor),
        "tail_drift_target_m": float(args.tail_drift_m),
    }
    args.output_motion.parent.mkdir(parents=True, exist_ok=True)
    with args.output_motion.open("wb") as handle:
        pickle.dump(updated, handle, protocol=pickle.HIGHEST_PROTOCOL)

    report = {
        "schema_version": 1,
        "status": "accepted",
        "input_motion": str(args.input_motion),
        "input_sha256": source_hash,
        "output_motion": str(args.output_motion),
        "output_sha256": sha256(args.output_motion),
        "input_motion_modified": sha256(args.input_motion) != source_hash,
        "semantic_back_axis_xy": back_axis.tolist(),
        "transition_frames": [int(start), int(end)],
        "anchor_frame": int(anchor),
        "raw_anchor_to_tail_back_drift_m": raw_tail_drift,
        "target_anchor_to_tail_back_drift_m": float(args.tail_drift_m),
        "maximum_root_depth_correction_m": float(np.max(np.abs(correction))),
        "joint_position_max_abs_difference": 0.0,
        "root_orientation_max_abs_difference": 0.0,
        "batch_safety_gate": {
            "max_correction_m": float(args.max_correction_m),
            "requires_external_seat_settle_event": True,
            "rejects_non_backrestward_or_oversized_drift": True,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

