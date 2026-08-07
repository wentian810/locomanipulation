#!/usr/bin/env python3
"""Reject unsafe root-XY contact locking before it can alter frozen GMR motion."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import h5py
import mujoco
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runs(mask: np.ndarray, minimum_frames: int) -> List[np.ndarray]:
    output: List[np.ndarray] = []
    start: int | None = None
    for index, active in enumerate(mask.tolist() + [False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            if index - start >= minimum_frames:
                output.append(np.arange(start, index, dtype=np.int64))
            start = None
    return output


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact-h5", required=True, type=Path)
    parser.add_argument("--robot-motion", required=True, type=Path)
    parser.add_argument("--robot-xml", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--left-foot-body", default="left_ankle_roll_link")
    parser.add_argument("--right-foot-body", default="right_ankle_roll_link")
    parser.add_argument("--minimum-contact-run", type=int, default=3)
    parser.add_argument("--max-root-xy-correction-m", type=float, default=0.08)
    parser.add_argument("--max-root-xy-correction-p95-m", type=float, default=0.05)
    parser.add_argument("--max-double-support-conflict-m", type=float, default=0.03)
    args = parser.parse_args()
    if args.minimum_contact_run < 2:
        parser.error("--minimum-contact-run must be at least two")
    for value in (
        args.max_root_xy_correction_m,
        args.max_root_xy_correction_p95_m,
        args.max_double_support_conflict_m,
    ):
        if not np.isfinite(value) or value <= 0.0:
            parser.error("all feasibility limits must be finite and positive")
    return args


def _body_id(model: mujoco.MjModel, name: str) -> int:
    result = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if result < 0:
        raise ValueError("robot body is missing: " + name)
    return result


def main() -> None:
    args = _parse_args()
    h5_path = args.contact_h5.resolve()
    motion_path = args.robot_motion.resolve()
    xml_path = args.robot_xml.resolve()
    for path in (h5_path, motion_path, xml_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    with motion_path.open("rb") as stream:
        motion = pickle.load(stream)
    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_rot = np.asarray(motion["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
    frames = len(root_pos)
    fps = float(motion["fps"])
    if root_pos.shape != (frames, 3) or root_rot.shape != (frames, 4) or dof_pos.shape[0] != frames:
        raise ValueError("robot motion pose arrays are inconsistent")
    with h5py.File(h5_path, "r") as archive:
        contacts = {
            "left": np.asarray(archive["contacts/left_foot"], dtype=bool),
            "right": np.asarray(archive["contacts/right_foot"], dtype=bool),
        }
        h5_fps = float(archive.attrs["fps"])
    if any(value.shape != (frames,) for value in contacts.values()) or not np.isclose(fps, h5_fps, atol=1e-6):
        raise ValueError("H5 contacts and robot motion do not share a frame contract")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if model.nq - 7 != dof_pos.shape[1]:
        raise ValueError("GMR DoF count does not match robot XML")
    data = mujoco.MjData(model)
    bodies = {
        "left": _body_id(model, args.left_foot_body),
        "right": _body_id(model, args.right_foot_body),
    }
    positions = np.zeros((frames, 2, 2), dtype=np.float64)
    for index in range(frames):
        data.qpos[:3] = root_pos[index]
        data.qpos[3:7] = root_rot[index][[3, 0, 1, 2]]
        data.qpos[7:] = dof_pos[index]
        mujoco.mj_forward(model, data)
        positions[index, 0] = data.xpos[bodies["left"], :2]
        positions[index, 1] = data.xpos[bodies["right"], :2]

    correction = np.full((frames, 2, 2), np.nan, dtype=np.float64)
    side_report: Dict[str, Any] = {}
    for side, index in (("left", 0), ("right", 1)):
        segments = []
        for segment in _runs(contacts[side], args.minimum_contact_run):
            required = positions[segment[0], index] - positions[segment, index]
            correction[segment, index] = required
            magnitude = np.linalg.norm(required, axis=1)
            segments.append(
                {
                    "frame_range": [int(segment[0]), int(segment[-1])],
                    "frame_count": int(len(segment)),
                    "required_root_xy_correction_m": _summary(magnitude),
                }
            )
        active = np.isfinite(correction[:, index, 0])
        magnitude = np.linalg.norm(correction[active, index], axis=1)
        side_report[side] = {
            "contact_frame_count": int(np.count_nonzero(contacts[side])),
            "lockable_contact_frame_count": int(np.count_nonzero(active)),
            "contact_segments": segments,
            "required_root_xy_correction_m": _summary(magnitude),
        }

    dual = np.isfinite(correction[:, 0, 0]) & np.isfinite(correction[:, 1, 0])
    conflict = np.linalg.norm(correction[dual, 0] - correction[dual, 1], axis=1)
    reasons: List[str] = []
    for side, details in side_report.items():
        requirement = details["required_root_xy_correction_m"]
        if details["lockable_contact_frame_count"]:
            if requirement["maximum"] > args.max_root_xy_correction_m:
                reasons.append(side + "_root_xy_correction_exceeds_maximum")
            if requirement["p95"] > args.max_root_xy_correction_p95_m:
                reasons.append(side + "_root_xy_correction_exceeds_p95_limit")
    conflict_summary = _summary(conflict)
    if len(conflict) and conflict_summary["p95"] > args.max_double_support_conflict_m:
        reasons.append("double_support_contact_lock_conflict")
    if not any(details["lockable_contact_frame_count"] for details in side_report.values()):
        reasons.append("no_lockable_contact_segment")

    report = {
        "schema_version": 1,
        "purpose": "bstro_contact_lock_feasibility_for_frozen_gmr",
        "robot_motion": str(motion_path),
        "robot_motion_sha256": _sha256(motion_path),
        "robot_motion_modified": False,
        "contact_h5": str(h5_path),
        "contact_h5_sha256": _sha256(h5_path),
        "robot_xml": str(xml_path),
        "frame_count": frames,
        "fps": fps,
        "quality_gate": {
            "max_root_xy_correction_m": args.max_root_xy_correction_m,
            "max_root_xy_correction_p95_m": args.max_root_xy_correction_p95_m,
            "max_double_support_conflict_m": args.max_double_support_conflict_m,
        },
        "feet": side_report,
        "double_support_required_root_xy_conflict_m": conflict_summary,
        "accepted_for_root_xy_contact_lock": not reasons,
        "rejection_reasons": sorted(set(reasons)),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"accepted": report["accepted_for_root_xy_contact_lock"], "reasons": report["rejection_reasons"]}))


if __name__ == "__main__":
    main()
