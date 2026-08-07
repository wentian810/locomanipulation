#!/usr/bin/env python3
"""Find the smallest planar seat-frame translation that restores G1 seat contact.

This is a deterministic scene-alignment step for transferring an already
verified seated G1 pose to a chair with a slightly different seat footprint.
Only the robot free-base XY translation *in the chair's own seat frame* is
searched.  Joint angles, root height, chair geometry and all runtime dynamics
are unchanged.  A later gravity-only landing run remains the acceptance test.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


CHAIR_GEOMS = frozenset((
    "seat_support_geom", "backrest_geom", "leg_front_left_geom",
    "leg_front_right_geom", "leg_back_left_geom", "leg_back_right_geom",
))
SUPPORT_BODIES = frozenset((
    "pelvis", "left_hip_pitch_link", "right_hip_pitch_link",
    "left_hip_roll_link", "right_hip_roll_link",
    "left_hip_yaw_link", "right_hip_yaw_link",
))


def _name(model: mujoco.MjModel, kind: mujoco.mjtObj, identifier: int) -> str:
    return mujoco.mj_id2name(model, kind, identifier) or ""


def _free_qpos_address(model: mujoco.MjModel) -> int:
    ids = [index for index in range(model.njnt) if model.jnt_type[index] == mujoco.mjtJoint.mjJNT_FREE]
    if len(ids) != 1:
        raise ValueError("task must contain exactly one free joint")
    return int(model.jnt_qposadr[ids[0]])


def _contact_metrics(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[int, float, float]:
    support_count, support_min, forbidden_min = 0, 0.0, 0.0
    for index in range(data.ncon):
        contact = data.contact[index]
        first = _name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
        second = _name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
        if first in CHAIR_GEOMS:
            chair_geom, robot_geom = first, int(contact.geom2)
        elif second in CHAIR_GEOMS:
            chair_geom, robot_geom = second, int(contact.geom1)
        else:
            continue
        robot_body = _name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[robot_geom]))
        if chair_geom == "seat_support_geom" and robot_body in SUPPORT_BODIES:
            support_count += 1
            support_min = min(support_min, float(contact.dist))
        else:
            forbidden_min = min(forbidden_min, float(contact.dist))
    return support_count, support_min, forbidden_min


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-task-xml", required=True, type=Path)
    parser.add_argument("--input-key-name", required=True)
    parser.add_argument("--output-task-xml", required=True, type=Path)
    parser.add_argument("--output-key-name", required=True)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--search-radius-m", type=float, default=0.05)
    parser.add_argument("--search-step-m", type=float, default=0.005)
    parser.add_argument("--max-forbidden-penetration-m", type=float, default=0.003)
    args = parser.parse_args()
    if args.output_task_xml.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite selected-seat XML/report")
    if args.search_radius_m <= 0.0 or args.search_step_m <= 0.0:
        raise ValueError("search radius and step must be positive")
    if args.max_forbidden_penetration_m < 0.0:
        raise ValueError("max forbidden penetration must be non-negative")

    model = mujoco.MjModel.from_xml_path(str(args.input_task_xml))
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.input_key_name)
    seat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "seat_support_geom")
    if key_id < 0 or seat_id < 0:
        raise ValueError("input key or seat_support_geom is absent")
    root_address = _free_qpos_address(model)
    base_qpos = model.key_qpos[key_id].copy()
    data = mujoco.MjData(model)
    data.qpos[:] = base_qpos
    mujoco.mj_forward(model, data)
    seat_rotation = data.geom_xmat[seat_id].reshape(3, 3).copy()
    values = np.arange(-args.search_radius_m, args.search_radius_m + 0.5 * args.search_step_m, args.search_step_m)
    candidates: list[tuple[tuple[float, float, float], np.ndarray, int, float, float]] = []
    for local_x in values:
        for local_y in values:
            qpos = base_qpos.copy()
            local_shift = np.array((local_x, local_y, 0.0), dtype=np.float64)
            qpos[root_address:root_address + 3] += seat_rotation @ local_shift
            data.qpos[:] = qpos
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            support_count, support_min, forbidden_min = _contact_metrics(model, data)
            if support_count == 0 or forbidden_min < -args.max_forbidden_penetration_m:
                continue
            # Prioritize no unwanted chair collision, a shallow support pair,
            # then the smallest seat-frame correction.
            rank = (
                max(0.0, -forbidden_min),
                abs(support_min),
                float(np.linalg.norm(local_shift[:2])),
            )
            candidates.append((rank, qpos, support_count, support_min, forbidden_min))
    if not candidates:
        raise RuntimeError("no shallow supported pose inside requested planar search region")
    candidates.sort(key=lambda entry: entry[0])
    rank, selected_qpos, support_count, support_min, forbidden_min = candidates[0]
    selected_shift_world = selected_qpos[root_address:root_address + 3] - base_qpos[root_address:root_address + 3]
    selected_shift_local = seat_rotation.T @ selected_shift_world

    tree = ET.parse(args.input_task_xml)
    root = tree.getroot()
    keyframe = root.find("keyframe")
    if keyframe is None:
        raise ValueError("input task has no keyframe element")
    if any(key.get("name") == args.output_key_name for key in keyframe.findall("key")):
        raise ValueError(f"output key {args.output_key_name!r} already exists")
    ET.SubElement(keyframe, "key", {
        "name": args.output_key_name,
        "qpos": " ".join(f"{value:.17g}" for value in selected_qpos),
    })
    args.output_task_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.output_task_xml, encoding="utf-8", xml_declaration=True)
    output_model = mujoco.MjModel.from_xml_path(str(args.output_task_xml))
    output_id = mujoco.mj_name2id(output_model, mujoco.mjtObj.mjOBJ_KEY, args.output_key_name)
    if output_id < 0 or not np.allclose(output_model.key_qpos[output_id], selected_qpos, atol=1e-12, rtol=0.0):
        raise RuntimeError("selected keyframe does not round-trip through MuJoCo")
    report = {
        "schema_version": 1,
        "status": "minimum_planar_supported_seat_key_ready_for_gravity_validation",
        "input_task_xml": str(args.input_task_xml.resolve()),
        "input_key_name": args.input_key_name,
        "output_task_xml": str(args.output_task_xml.resolve()),
        "output_key_name": args.output_key_name,
        "output_key_id": int(output_id),
        "search": {"radius_m": args.search_radius_m, "step_m": args.search_step_m},
        "selection": {
            "local_seat_shift_m": selected_shift_local.tolist(),
            "world_root_shift_m": selected_shift_world.tolist(),
            "support_contact_count": support_count,
            "support_contact_distance_m": support_min,
            "forbidden_contact_distance_m": forbidden_min,
            "ranking": list(rank),
            "candidate_count": len(candidates),
        },
        "contract": {
            "changes": "one keyframe with the minimum seat-frame planar root translation",
            "does_not_change": ["root_height", "root_orientation", "joint_positions", "chair_geometry", "actuators", "runtime_state", "external_forces"],
        },
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
