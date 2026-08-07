#!/usr/bin/env python3
"""Infer a seated-support state and regulate only root depth before support.

The tool consumes semantic-contact measurements rather than clip-specific
frames.  For a chair it finds: (1) first persistent seat support, (2) the
first stable, non-penetrating near-backrest interval, and then reaches that
interval's robust root depth *before* support begins.  Root depth is constant
after support.  It never changes root orientation or any joint.

Stools and steps deliberately do not receive a horizontal root correction:
without a backrest there is no observable target depth.  They are routed for
static-support fitting only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _primitive(payload: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any] | None:
    for item in payload.get("primitives", []):
        if str(item.get("name")) in names:
            return item
    return None


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    edges = np.r_[0, np.flatnonzero(mask[1:] != mask[:-1]) + 1, len(mask)]
    return [(int(edges[i]), int(edges[i + 1] - 1)) for i in range(len(edges) - 1) if bool(mask[edges[i]])]


def _first_persistent(mask: np.ndarray, minimum: int) -> int | None:
    for start, end in _runs(mask):
        if end - start + 1 >= minimum:
            return start
    return None


def _monotone_hermite(start_value: float, end_value: float, start_slope: float, frames: int) -> np.ndarray:
    """Shape-preserving cubic from the original approach slope to rest."""
    if frames < 2:
        raise ValueError("pre-support regulation needs at least two frames")
    duration = float(frames)
    secant = (end_value - start_value) / duration
    if abs(secant) < 1e-10:
        return np.full(frames + 1, end_value, dtype=np.float64)
    sign = 1.0 if secant > 0.0 else -1.0
    slope = float(start_slope)
    if sign * slope < 0.0:
        slope = 0.0
    slope = sign * min(abs(slope), 3.0 * abs(secant))
    t = np.arange(frames + 1, dtype=np.float64) / duration
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    values = h00 * start_value + h10 * duration * slope + h01 * end_value
    # end slope is exactly zero.  h11 term is therefore absent.
    return values


def _load_motion(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict) or "root_pos" not in data:
        raise TypeError("robot motion must be a dictionary with root_pos")
    root = np.asarray(data["root_pos"], dtype=np.float64)
    if root.ndim != 2 or root.shape[1] != 3:
        raise ValueError(f"invalid root_pos shape {root.shape}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-motion", required=True, type=Path)
    parser.add_argument("--support-contacts", required=True, type=Path)
    parser.add_argument("--chair-primitives", required=True, type=Path)
    parser.add_argument("--output-motion", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seat-contact-tolerance-m", type=float, default=0.006)
    parser.add_argument("--backrest-min-gap-m", type=float, default=-0.005)
    parser.add_argument("--backrest-max-gap-m", type=float, default=0.10)
    parser.add_argument("--minimum-support-frames", type=int, default=6)
    parser.add_argument("--minimum-stable-frames", type=int, default=6)
    parser.add_argument("--approach-regulation-seconds", type=float, default=1.0)
    parser.add_argument("--max-depth-correction-m", type=float, default=0.35)
    args = parser.parse_args()

    source_hash = _sha256(args.robot_motion)
    motion = _load_motion(args.robot_motion)
    root = np.asarray(motion["root_pos"], dtype=np.float64)
    frame_count = len(root)
    contacts = np.load(args.support_contacts, allow_pickle=True)
    required = ("seat_signed_distance_m", "backrest_front_gap_m")
    missing = [key for key in required if key not in contacts]
    if missing:
        raise ValueError(f"support contacts missing {missing}")
    seat_distance = np.asarray(contacts["seat_signed_distance_m"], dtype=np.float64)
    back_gap = np.asarray(contacts["backrest_front_gap_m"], dtype=np.float64)
    if len(seat_distance) != frame_count or len(back_gap) != frame_count:
        raise ValueError("motion/contact frame counts differ")

    primitives = json.loads(args.chair_primitives.read_text(encoding="utf-8"))
    seat = _primitive(primitives, ("seat_support", "seat"))
    back = _primitive(primitives, ("backrest", "back"))
    object_kind = "chair" if seat is not None and back is not None else ("stool" if seat is not None else "unknown")
    if object_kind != "chair":
        report = {
            "schema_version": 1,
            "status": "rejected_no_observable_backrest_depth_target",
            "object_kind": object_kind,
            "policy": "no horizontal root correction for stool/step/unknown support",
            "input_motion_modified": False,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    seat_center = np.asarray(seat["center"], dtype=np.float64)
    back_center = np.asarray(back["center"], dtype=np.float64)
    back_axis = back_center[:2] - seat_center[:2]
    axis_norm = float(np.linalg.norm(back_axis))
    if axis_norm < 1e-8:
        raise ValueError("seat and backrest have no horizontal separation")
    back_axis /= axis_norm
    depth = root[:, :2] @ back_axis

    supported = np.isfinite(seat_distance) & (np.abs(seat_distance) <= args.seat_contact_tolerance_m)
    support_onset = _first_persistent(supported, args.minimum_support_frames)
    near_back = (
        supported
        & np.isfinite(back_gap)
        & (back_gap >= args.backrest_min_gap_m)
        & (back_gap <= args.backrest_max_gap_m)
    )
    stable_runs = [(start, end) for start, end in _runs(near_back) if end - start + 1 >= args.minimum_stable_frames]
    stable_runs = [(start, end) for start, end in stable_runs if support_onset is not None and start >= support_onset]
    if support_onset is None or not stable_runs:
        report = {
            "schema_version": 1,
            "status": "rejected_no_stable_supported_nonpenetrating_backrest_interval",
            "object_kind": object_kind,
            "support_onset_frame": support_onset,
            "stable_runs": stable_runs,
            "input_motion_modified": False,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    # Earliest valid interval is used: later intervals may be post-contact
    # drift, an intentional movement, or a monocular depth failure.
    stable_start, stable_end = stable_runs[0]
    target_depth = float(np.median(depth[stable_start : stable_end + 1]))
    onset_depth = float(depth[support_onset])
    correction_at_onset = target_depth - onset_depth
    if abs(correction_at_onset) > args.max_depth_correction_m:
        raise RuntimeError(
            f"required pre-support correction {correction_at_onset:.4f} m exceeds safety bound "
            f"{args.max_depth_correction_m:.4f} m"
        )

    lead = max(2, int(round(args.approach_regulation_seconds * args.fps)))
    regulation_start = max(2, support_onset - lead)
    desired_depth = depth.copy()
    raw_slope = float((depth[regulation_start + 1] - depth[regulation_start - 1]) / 2.0)
    desired_depth[regulation_start : support_onset + 1] = _monotone_hermite(
        float(depth[regulation_start]), target_depth, raw_slope, support_onset - regulation_start
    )
    desired_depth[support_onset + 1 :] = target_depth
    depth_correction = desired_depth - depth
    maximum_total_correction = float(np.max(np.abs(depth_correction)))
    # The approach spline can overshoot the correction present at support onset.
    # Gate the actual signal, not only its endpoint; otherwise a batch run may
    # silently exceed its declared safety bound before the sit event.
    if maximum_total_correction > args.max_depth_correction_m:
        report = {
            "schema_version": 1,
            "status": "rejected_total_root_depth_correction_exceeds_safety_bound",
            "object_kind": object_kind,
            "support_onset_frame": int(support_onset),
            "stable_interval": [int(stable_start), int(stable_end)],
            "depth_correction_at_support_onset_m": correction_at_onset,
            "maximum_root_depth_correction_m": maximum_total_correction,
            "max_depth_correction_m": args.max_depth_correction_m,
            "input_motion_modified": False,
            "policy": "do not emit a motion candidate when the realised spline exceeds its bound",
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    corrected_root = root.copy()
    corrected_root[:, :2] += depth_correction[:, None] * back_axis[None, :]

    corrected = dict(motion)
    corrected["root_pos"] = corrected_root.astype(np.asarray(motion["root_pos"]).dtype, copy=False)
    corrected["semantic_support_root_regulation"] = {
        "version": "v1",
        "object_kind": object_kind,
        "policy": "infer support and stable nonpenetrating backrest interval; finish depth regulation before support; hold depth thereafter",
        "support_onset_frame": int(support_onset),
        "stable_interval": [int(stable_start), int(stable_end)],
        "root_orientation_modified": False,
        "joint_trajectory_modified": False,
    }
    args.output_motion.parent.mkdir(parents=True, exist_ok=True)
    with args.output_motion.open("wb") as handle:
        pickle.dump(corrected, handle, protocol=pickle.HIGHEST_PROTOCOL)

    post_lock = desired_depth[support_onset:]
    report = {
        "schema_version": 1,
        "status": "accepted_kinematic_review_only",
        "object_kind": object_kind,
        "input_motion": str(args.robot_motion),
        "input_sha256": source_hash,
        "output_motion": str(args.output_motion),
        "output_sha256": _sha256(args.output_motion),
        "input_motion_modified": _sha256(args.robot_motion) != source_hash,
        "semantic_back_axis_xy": back_axis.tolist(),
        "support_onset_frame": int(support_onset),
        "stable_interval": [int(stable_start), int(stable_end)],
        "target_depth_rule": "median robot root depth over earliest stable supported nonpenetrating near-backrest interval",
        "raw_depth_onset_to_tail_m": float(depth[-1] - depth[support_onset]),
        "corrected_depth_onset_to_tail_m": float(post_lock[-1] - post_lock[0]),
        "maximum_post_support_depth_drift_m": float(np.max(np.abs(post_lock - post_lock[0]))),
        "pre_support_regulation_frames": [int(regulation_start), int(support_onset)],
        "depth_correction_at_support_onset_m": correction_at_onset,
        "maximum_root_depth_correction_m": float(np.max(np.abs(depth_correction))),
        "root_orientation_max_abs_difference": 0.0,
        "joint_position_max_abs_difference": 0.0,
        "batch_quality_gates": {
            "requires_semantic_seat_and_backrest": True,
            "requires_persistent_seat_support": True,
            "requires_stable_nonpenetrating_near_backrest_interval": True,
            "uses_earliest_valid_interval_not_tail": True,
            "max_depth_correction_m": args.max_depth_correction_m,
        },
        "physical_status": "not a dynamics result; require separate robot-scene collision and WBC validation",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

