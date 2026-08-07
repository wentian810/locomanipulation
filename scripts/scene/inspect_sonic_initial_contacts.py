#!/usr/bin/env python3
"""Inspect initial floor/chair clearance for one GMR state in a SONIC scene."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-xml", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0)
    return parser.parse_args()


def set_gmr_state(model: mujoco.MjModel, data: mujoco.MjData, motion_path: Path, frame: int) -> None:
    with motion_path.open("rb") as handle:
        motion = pickle.load(handle)
    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_rot_xyzw = np.asarray(motion["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
    dof_names = list(motion["dof_names"])
    if not 0 <= frame < len(root_pos):
        raise IndexError(f"frame {frame} outside [0, {len(root_pos)})")
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[:3] = root_pos[frame]
    data.qpos[3:7] = root_rot_xyzw[frame, (3, 0, 1, 2)]
    for name, value in zip(dof_names, dof_pos[frame]):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"scene misses joint {name!r}")
        data.qpos[model.jnt_qposadr[joint_id]] = value
    mujoco.mj_forward(model, data)


def name(model: mujoco.MjModel, obj: mujoco.mjtObj, index: int) -> str:
    result = mujoco.mj_id2name(model, obj, index)
    return result if result is not None else f"{obj.name}[{index}]"


def main() -> None:
    args = parse_args()
    if not args.scene_xml.is_file() or not args.motion.is_file():
        raise FileNotFoundError("--scene-xml and --motion must be files")
    model = mujoco.MjModel.from_xml_path(str(args.scene_xml))
    data = mujoco.MjData(model)
    set_gmr_state(model, data, args.motion, args.frame)

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0:
        raise ValueError("scene has no floor geom")
    static_ids = [floor_id]
    for geom_name in ("seat_support_geom", "backrest_geom"):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id >= 0:
            static_ids.append(geom_id)

    distances: list[dict[str, object]] = []
    for geom_id in range(model.ngeom):
        if model.geom_contype[geom_id] != 2 or model.geom_conaffinity[geom_id] != 0:
            continue
        for static_id in static_ids:
            fromto = np.zeros(6, dtype=np.float64)
            distance = float(mujoco.mj_geomDistance(model, data, geom_id, static_id, 10.0, fromto))
            distances.append({
                "robot_geom": name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
                "robot_body": name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])),
                "static_geom": name(model, mujoco.mjtObj.mjOBJ_GEOM, static_id),
                "distance_m": distance,
                "nearest_points": fromto.tolist(),
            })
    distances.sort(key=lambda entry: float(entry["distance_m"]))

    contacts = []
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        contacts.append({
            "geom1": name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1),
            "geom2": name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2),
            "distance_m": float(contact.dist),
        })
    body_heights = {}
    for body_name in ("pelvis", "left_ankle_roll_link", "right_ankle_roll_link", "torso_link"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id >= 0:
            body_heights[body_name] = data.xpos[body_id].tolist()

    report = {
        "scene": str(args.scene_xml.resolve()),
        "motion": str(args.motion.resolve()),
        "frame": args.frame,
        "root_qpos": data.qpos[:7].tolist(),
        "body_positions": body_heights,
        "initial_contacts": contacts,
        "nearest_robot_static_pairs": distances[:12],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
