#!/usr/bin/env python3
"""Classify actual chair contacts along an already-produced MJPC trajectory.

This is an audit only: CSV qpos/qvel are replayed into MuJoCo with
``mj_forward`` to identify which robot bodies touch which fixed chair geometry.
It never steps, controls, or rewrites the original rollout.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np


SUPPORT_BODIES = frozenset({
    "pelvis", "left_hip_pitch_link", "right_hip_pitch_link",
    "left_hip_roll_link", "right_hip_roll_link",
    "left_hip_yaw_link", "right_hip_yaw_link",
})
CHAIR_PREFIXES = ("seat_support_geom", "backrest_geom", "leg_front_", "leg_back_")


def load_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.reader(line for line in handle if not line.startswith("#"))]
    if len(rows) < 2:
        raise ValueError("trajectory contains no data rows")
    return rows[0], np.asarray(rows[1:], dtype=np.float64)


def numbered_columns(header: list[str], prefix: str) -> list[int]:
    pairs = [(int(name.removeprefix(prefix)), index) for index, name in enumerate(header) if name.startswith(prefix)]
    return [index for _, index in sorted(pairs)]


def geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--task-xml", required=True, type=Path)
    parser.add_argument("--seat-first-frame", required=True, type=int)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--max-forbidden-penetration-m", type=float, default=0.005)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("refusing to overwrite a contact-type audit")
    header, values = load_csv(args.trajectory)
    qpos_columns = numbered_columns(header, "qpos_")
    qvel_columns = numbered_columns(header, "qvel_")
    model = mujoco.MjModel.from_xml_path(str(args.task_xml))
    if len(qpos_columns) != model.nq or len(qvel_columns) != model.nv:
        raise ValueError("trajectory qpos/qvel dimensions do not match task XML")
    if not 0 <= args.seat_first_frame < len(values):
        raise ValueError("seat-first-frame is outside trajectory")
    seat_force_index = header.index("semantic_seat_normal_force_n") if "semantic_seat_normal_force_n" in header else None
    data = mujoco.MjData(model)
    phases = {
        "pre_sit": {"frames": 0, "support_frames": 0, "forbidden_violation_frames": 0, "minimum_distance_m": None, "pairs": defaultdict(lambda: {"frames": 0, "min_distance_m": None})},
        "seated": {"frames": 0, "support_frames": 0, "forbidden_violation_frames": 0, "minimum_distance_m": None, "pairs": defaultdict(lambda: {"frames": 0, "min_distance_m": None})},
    }
    violations: list[dict[str, object]] = []
    for frame, row in enumerate(values):
        phase_name = "pre_sit" if frame < args.seat_first_frame else "seated"
        phase = phases[phase_name]
        phase["frames"] += 1
        data.qpos[:] = row[qpos_columns]
        data.qvel[:] = row[qvel_columns]
        mujoco.mj_forward(model, data)
        support_seen = False
        frame_forbidden: list[dict[str, object]] = []
        for contact in data.contact[: data.ncon]:
            first, second = geom_name(model, int(contact.geom1)), geom_name(model, int(contact.geom2))
            if first.startswith("gmr_official_collision_") and second.startswith(CHAIR_PREFIXES):
                robot_geom, chair_geom = int(contact.geom1), second
            elif second.startswith("gmr_official_collision_") and first.startswith(CHAIR_PREFIXES):
                robot_geom, chair_geom = int(contact.geom2), first
            else:
                continue
            robot_body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[robot_geom])) or ""
            distance = float(contact.dist)
            is_support = robot_body in SUPPORT_BODIES and chair_geom == "seat_support_geom"
            support_seen |= is_support
            pair_name = f"{robot_body}->{chair_geom}"
            pair = phase["pairs"][pair_name]
            pair["frames"] += 1
            pair["min_distance_m"] = distance if pair["min_distance_m"] is None else min(pair["min_distance_m"], distance)
            phase["minimum_distance_m"] = distance if phase["minimum_distance_m"] is None else min(phase["minimum_distance_m"], distance)
            if not is_support and distance < -args.max_forbidden_penetration_m:
                frame_forbidden.append({"robot_body": robot_body, "chair_geom": chair_geom, "distance_m": distance})
        if support_seen:
            phase["support_frames"] += 1
        if frame_forbidden:
            phase["forbidden_violation_frames"] += 1
            violations.append({"frame": frame, "phase": phase_name, "contacts": frame_forbidden})
    for phase in phases.values():
        phase["support_frame_ratio"] = phase["support_frames"] / phase["frames"] if phase["frames"] else 0.0
        phase["pairs"] = dict(sorted(phase["pairs"].items()))
    report = {
        "schema_version": 1,
        "purpose": "actual_mjpc_trajectory_chair_contact_type_audit",
        "status": "audited_no_state_modified",
        "inputs": {"trajectory": str(args.trajectory), "task_xml": str(args.task_xml)},
        "seat_first_frame": args.seat_first_frame,
        "max_forbidden_penetration_m": args.max_forbidden_penetration_m,
        "recorded_seat_force_available": seat_force_index is not None,
        "recorded_seat_force_summary_n": (
            {"pre_sit_max": float(np.max(values[:args.seat_first_frame, seat_force_index])), "seated_max": float(np.max(values[args.seat_first_frame:, seat_force_index])), "seated_mean": float(np.mean(values[args.seat_first_frame:, seat_force_index]))}
            if seat_force_index is not None else None
        ),
        "phases": phases,
        "forbidden_penetration_violations": violations,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "pre_sit": phases["pre_sit"], "seated": phases["seated"], "forbidden_violation_frame_count": len(violations)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
