#!/usr/bin/env python3
"""Audit static root translation contact windows before expensive IK.

The scan writes no motion.  It only probes a reference qpos with uniform
free-base Y/Z offsets to establish whether shallow seat support and no other
chair collision are even geometrically available in a fixed scene.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


SUPPORT_BODIES = frozenset({
    "pelvis", "left_hip_pitch_link", "right_hip_pitch_link",
    "left_hip_roll_link", "right_hip_roll_link",
    "left_hip_yaw_link", "right_hip_yaw_link",
})
CHAIR_PREFIXES = ("seat_support_geom", "backrest_geom", "leg_front_", "leg_back_")


def qpos_from_motion(motion: dict[str, Any]) -> np.ndarray:
    root = np.asarray(motion["root_pos"], dtype=np.float64)
    rotation = np.asarray(motion["root_rot"], dtype=np.float64)
    dof = np.asarray(motion["dof_pos"], dtype=np.float64)
    qpos = np.empty((len(root), 7 + dof.shape[1]), dtype=np.float64)
    qpos[:, :3] = root
    qpos[:, 3:7] = rotation[:, [3, 0, 1, 2]]
    qpos[:, 7:] = dof
    return qpos


def name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""


def contact_summary(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, object]:
    support: list[float] = []
    forbidden: list[dict[str, object]] = []
    for contact in data.contact[: data.ncon]:
        first, second = name(model, int(contact.geom1)), name(model, int(contact.geom2))
        if first.startswith("gmr_official_collision_") and second.startswith(CHAIR_PREFIXES):
            robot_geom, chair_geom = int(contact.geom1), second
        elif second.startswith("gmr_official_collision_") and first.startswith(CHAIR_PREFIXES):
            robot_geom, chair_geom = int(contact.geom2), first
        else:
            continue
        robot_body = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[robot_geom])
        ) or ""
        distance = float(contact.dist)
        if robot_body in SUPPORT_BODIES and chair_geom == "seat_support_geom":
            support.append(distance)
        else:
            forbidden.append({"robot_body": robot_body, "chair_geom": chair_geom, "distance_m": distance})
    return {
        "support_count": len(support), "support_min_distance_m": min(support) if support else None,
        "forbidden_count": len(forbidden), "forbidden_min_distance_m": min((item["distance_m"] for item in forbidden), default=None),
        "forbidden": forbidden,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-motion", required=True, type=Path)
    parser.add_argument("--task-xml", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--frames", required=True, help="comma-separated source frame indices")
    parser.add_argument("--min-root-y-offset-m", type=float, default=0.0)
    parser.add_argument("--max-root-y-offset-m", type=float, default=0.0)
    parser.add_argument("--min-root-z-offset-m", type=float, default=-0.06)
    parser.add_argument("--max-root-z-offset-m", type=float, default=0.10)
    parser.add_argument("--step-m", type=float, default=0.001)
    parser.add_argument("--max-seat-penetration-m", type=float, default=0.003)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("refusing to overwrite a feasibility audit")
    if (
        args.step_m <= 0.0
        or args.min_root_y_offset_m > args.max_root_y_offset_m
        or args.min_root_z_offset_m > args.max_root_z_offset_m
        or args.max_seat_penetration_m <= 0.0
    ):
        raise ValueError("invalid root-translation scan range")
    frames = [int(value) for value in args.frames.split(",") if value.strip()]
    if not frames:
        raise ValueError("frames must be nonempty")
    with args.robot_motion.open("rb") as handle:
        motion = pickle.load(handle)
    qpos = qpos_from_motion(motion)
    model = mujoco.MjModel.from_xml_path(str(args.task_xml))
    if model.nq != qpos.shape[1] or model.nkey != len(qpos) or not np.allclose(model.key_qpos, qpos, atol=1e-6, rtol=0.0):
        raise ValueError("task XML keyframes do not exactly match robot motion")
    if any(frame < 0 or frame >= len(qpos) for frame in frames):
        raise ValueError("scan frame is out of range")
    data = mujoco.MjData(model)
    y_offsets = np.arange(
        args.min_root_y_offset_m,
        args.max_root_y_offset_m + 0.5 * args.step_m,
        args.step_m,
    )
    z_offsets = np.arange(
        args.min_root_z_offset_m,
        args.max_root_z_offset_m + 0.5 * args.step_m,
        args.step_m,
    )
    scans = []
    for frame in frames:
        entries = []
        first_strict = None
        for y_offset in y_offsets:
            for z_offset in z_offsets:
                data.qpos[:] = qpos[frame]
                data.qpos[1] += y_offset
                data.qpos[2] += z_offset
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                contacts = contact_summary(model, data)
                entry = {
                    "root_y_offset_m": float(y_offset),
                    "root_z_offset_m": float(z_offset),
                    **contacts,
                }
                entries.append(entry)
                if (
                    first_strict is None
                    and contacts["support_count"]
                    and contacts["forbidden_count"] == 0
                    and contacts["support_min_distance_m"] >= -args.max_seat_penetration_m
                ):
                    first_strict = entry
        scans.append({"frame": frame, "first_strict_support": first_strict, "offsets": entries})
    report = {
        "schema_version": 1, "purpose": "static_gmr_chair_vertical_feasibility_audit",
        "status": "audited_no_motion_modified", "robot_motion": str(args.robot_motion), "task_xml": str(args.task_xml),
        "settings": {
            "min_root_y_offset_m": args.min_root_y_offset_m,
            "max_root_y_offset_m": args.max_root_y_offset_m,
            "min_root_z_offset_m": args.min_root_z_offset_m,
            "max_root_z_offset_m": args.max_root_z_offset_m,
            "step_m": args.step_m,
            "max_seat_penetration_m": args.max_seat_penetration_m,
        },
        "scans": scans,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "strict_support": [{"frame": row["frame"], "entry": row["first_strict_support"]} for row in scans]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
