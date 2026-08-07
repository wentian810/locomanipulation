#!/usr/bin/env python3
"""Identify the exact G1-chair contacts in an executed MJPC trajectory.

This is a read-only diagnostic: it re-evaluates each recorded ``qpos`` with
``mj_forward`` in a fresh data object.  It never advances physics, writes a
model file, moves the chair, or changes the recorded trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np


CHAIR_GEOM_NAMES = {
    "seat_support_geom",
    "leg_front_left_geom",
    "leg_front_right_geom",
    "leg_back_left_geom",
    "leg_back_right_geom",
    "backrest_geom",
}
ROBOT_COLLISION_PREFIXES = (
    "gmr_physics_proxy_",
    "gmr_collision_source_",
    "gmr_custom_collision_",
    "gmr_native_collision_",
    "gmr_official_collision_",
)


def _load_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.reader(line for line in handle if not line.startswith("#"))]
    if len(rows) < 2:
        raise ValueError(f"trajectory has no data rows: {path}")
    values = np.asarray(rows[1:], dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(rows[0]):
        raise ValueError("trajectory header and data width disagree")
    return rows[0], values


def _qpos_indices(header: list[str]) -> list[int]:
    indexed = [
        (int(name.removeprefix("qpos_")), index)
        for index, name in enumerate(header)
        if name.startswith("qpos_")
    ]
    return [index for _, index in sorted(indexed)]


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    return "" if name is None else name


def _chair_robot_pair(model: mujoco.MjModel, geom1: int, geom2: int) -> tuple[str, str] | None:
    name1 = _geom_name(model, geom1)
    name2 = _geom_name(model, geom2)
    if name1.startswith(ROBOT_COLLISION_PREFIXES) and name2 in CHAIR_GEOM_NAMES:
        return name1, name2
    if name2.startswith(ROBOT_COLLISION_PREFIXES) and name1 in CHAIR_GEOM_NAMES:
        return name2, name1
    return None


def _record(
    local_frame: int,
    time_s: float,
    robot_geom: str,
    chair_geom: str,
    contact: mujoco.MjContact,
) -> dict[str, object]:
    return {
        "local_frame": local_frame,
        "time_s": time_s,
        "robot_geom": robot_geom,
        "chair_geom": chair_geom,
        "distance_m": float(contact.dist),
        "penetration_m": float(max(0.0, -contact.dist)),
        "position_world_m": [float(value) for value in contact.pos],
        "normal_world": [float(value) for value in contact.frame[:3]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-model", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--seat-start-local-frame", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.task_model))
    data = mujoco.MjData(model)
    header, values = _load_csv(args.trajectory)
    qpos_indices = _qpos_indices(header)
    if len(qpos_indices) != model.nq:
        raise ValueError(
            f"trajectory qpos width {len(qpos_indices)} does not match model.nq {model.nq}"
        )
    if not 0 <= args.seat_start_local_frame < len(values):
        raise ValueError("seat-start-local-frame is outside trajectory")
    try:
        time_index = header.index("time")
    except ValueError as error:
        raise ValueError("trajectory is missing time column") from error

    minimum: dict[str, object] | None = None
    minimum_by_pair: dict[str, dict[str, object]] = {}
    minimum_pre_sit: dict[str, object] | None = None
    minimum_seated: dict[str, object] | None = None
    contact_frame_count = 0

    for local_frame, row in enumerate(values):
        data.qpos[:] = row[qpos_indices]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        observed_this_frame = False
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            pair = _chair_robot_pair(model, contact.geom1, contact.geom2)
            if pair is None:
                continue
            observed_this_frame = True
            record = _record(local_frame, float(row[time_index]), pair[0], pair[1], contact)
            pair_key = f"{pair[0]}::{pair[1]}"
            if minimum is None or record["distance_m"] < minimum["distance_m"]:
                minimum = record
            if pair_key not in minimum_by_pair or record["distance_m"] < minimum_by_pair[pair_key]["distance_m"]:
                minimum_by_pair[pair_key] = record
            if local_frame < args.seat_start_local_frame:
                if minimum_pre_sit is None or record["distance_m"] < minimum_pre_sit["distance_m"]:
                    minimum_pre_sit = record
            elif minimum_seated is None or record["distance_m"] < minimum_seated["distance_m"]:
                minimum_seated = record
        if observed_this_frame:
            contact_frame_count += 1

    report = {
        "schema_version": 1,
        "purpose": "read_only_executed_g1_static_chair_contact_diagnosis",
        "physical_contract": "recorded_qpos_only; mj_forward_diagnostic_only; no_mj_step_or_state_rewrite",
        "task_model": str(args.task_model),
        "trajectory": str(args.trajectory),
        "frame_count": int(len(values)),
        "seat_start_local_frame": args.seat_start_local_frame,
        "chair_robot_contact_frame_count": contact_frame_count,
        "minimum_contact": minimum,
        "minimum_pre_sit_contact": minimum_pre_sit,
        "minimum_seated_contact": minimum_seated,
        "minimum_contact_by_geom_pair": minimum_by_pair,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
