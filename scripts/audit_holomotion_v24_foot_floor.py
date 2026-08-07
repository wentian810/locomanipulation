#!/usr/bin/env python3
"""Audit the physically simulated G1 foot contact patches against the floor.

This is deliberately a replay *inspection*: it reconstructs qpos from the
logged HoloMotion ``mj_step`` states and calls ``mj_forward`` only to query
geometry.  It neither advances time nor changes qpos, so the report cannot
hide a discontinuity by re-simulating the trajectory.

The four small collision spheres mounted under each ankle-roll link are the
G1 model's active sole patches.  Mesh-only visual foot geoms have no collision
role and are therefore intentionally excluded from this check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


ANKLE_BODY_BY_SIDE = {
    "left": "left_ankle_roll_link",
    "right": "right_ankle_roll_link",
}


def _read_metadata(reference: np.lib.npyio.NpzFile) -> dict[str, object]:
    raw = reference["metadata"]
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    parsed = json.loads(str(raw))
    if not isinstance(parsed, dict):
        raise ValueError("reference metadata is not a JSON object")
    return parsed


def _qpos_from_rollout(
    model: mujoco.MjModel,
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos_reference_order: np.ndarray,
    dof_names: list[str],
) -> np.ndarray:
    """Map HoloMotion XYZW root rotations to MuJoCo WXYZ qpos exactly once."""
    frame_count = len(root_pos)
    if root_rot_xyzw.shape != (frame_count, 4):
        raise ValueError("roll-out root rotations must be [T,4]")
    if dof_pos_reference_order.shape != (frame_count, len(dof_names)):
        raise ValueError("roll-out dof positions do not match reference name order")
    free_joint_ids = [
        joint_id for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
    ]
    if len(free_joint_ids) != 1:
        raise ValueError("task must contain exactly one free base")
    root_address = int(model.jnt_qposadr[free_joint_ids[0]])
    qpos = np.zeros((frame_count, model.nq), dtype=np.float64)
    qpos[:, root_address:root_address + 3] = root_pos
    qpos[:, root_address + 3:root_address + 7] = root_rot_xyzw[:, [3, 0, 1, 2]]
    for column, name in enumerate(dof_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"task XML lacks reference joint {name!r}")
        qpos[:, int(model.jnt_qposadr[joint_id])] = dof_pos_reference_order[:, column]
    return qpos


def _active_sole_spheres(model: mujoco.MjModel) -> dict[str, list[int]]:
    """Find the collidable spherical sole patches in a topology-safe way."""
    result: dict[str, list[int]] = {}
    for side, body_name in ANKLE_BODY_BY_SIDE.items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"task XML lacks {body_name!r}")
        geom_ids = [
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) == body_id
            and int(model.geom_type[geom_id]) == mujoco.mjtGeom.mjGEOM_SPHERE
            and int(model.geom_contype[geom_id]) != 0
            and int(model.geom_conaffinity[geom_id]) != 0
        ]
        if not geom_ids:
            raise ValueError(f"no active collision-sphere sole patches on {body_name!r}")
        result[side] = geom_ids
    return result


def _frame_minimum_distances(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    floor_id: int,
    sole_spheres: dict[str, list[int]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return signed floor clearances and sole-surface heights per side/frame."""
    data = mujoco.MjData(model)
    distances = {side: np.empty(len(qpos), dtype=np.float64) for side in sole_spheres}
    surface_heights = {side: np.empty(len(qpos), dtype=np.float64) for side in sole_spheres}
    floor_z = float(model.geom_pos[floor_id, 2])
    for frame, state in enumerate(qpos):
        data.qpos[:] = state
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for side, geom_ids in sole_spheres.items():
            distances[side][frame] = min(
                float(mujoco.mj_geomDistance(model, data, geom_id, floor_id, 0.25, None))
                for geom_id in geom_ids
            )
            # The checked geoms are spheres and the floor is a horizontal MuJoCo plane.
            surface_heights[side][frame] = min(
                float(data.geom_xpos[geom_id, 2] - model.geom_size[geom_id, 0] - floor_z)
                for geom_id in geom_ids
            )
    return distances, surface_heights


def _side_report(distance: np.ndarray, surface_height: np.ndarray, max_penetration: float) -> dict[str, object]:
    height_step = np.diff(surface_height)
    return {
        "minimum_signed_floor_distance_m": float(np.min(distance)),
        "minimum_sole_surface_height_m": float(np.min(surface_height)),
        "p01_sole_surface_height_m": float(np.percentile(surface_height, 1.0)),
        "maximum_frame_to_frame_sole_height_change_m": (
            float(np.max(np.abs(height_step))) if len(height_step) else 0.0
        ),
        "frames_below_negative_penetration_tolerance": int(np.count_nonzero(distance < -max_penetration)),
        "pass": bool(np.min(distance) >= -max_penetration),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-npz", required=True, type=Path)
    parser.add_argument("--reference-npz", required=True, type=Path)
    parser.add_argument("--task-xml", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-floor-penetration-m", type=float, default=0.003)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite an existing foot-floor audit")
    if args.max_floor_penetration_m < 0.0:
        raise ValueError("max-floor-penetration-m must be non-negative")

    with np.load(args.reference_npz, allow_pickle=False) as reference, np.load(args.rollout_npz, allow_pickle=False) as rollout:
        metadata = _read_metadata(reference)
        dof_names = metadata.get("dof_names")
        if not isinstance(dof_names, list) or not all(isinstance(name, str) for name in dof_names):
            raise ValueError("reference metadata lacks ordered dof_names")
        root_pos = np.asarray(rollout["robot_global_translation"], dtype=np.float64)[:, 0]
        root_rot_xyzw = np.asarray(rollout["robot_global_rotation_quat"], dtype=np.float64)[:, 0]
        dof_pos = np.asarray(rollout["robot_dof_pos"], dtype=np.float64)

    model = mujoco.MjModel.from_xml_path(str(args.task_xml))
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0 or int(model.geom_type[floor_id]) != mujoco.mjtGeom.mjGEOM_PLANE:
        raise ValueError("task must expose a horizontal plane geom named 'floor'")
    qpos = _qpos_from_rollout(model, root_pos, root_rot_xyzw, dof_pos, dof_names)
    sole_spheres = _active_sole_spheres(model)
    distances, surface_heights = _frame_minimum_distances(model, qpos, floor_id, sole_spheres)
    sides = {
        side: _side_report(distances[side], surface_heights[side], args.max_floor_penetration_m)
        for side in ANKLE_BODY_BY_SIDE
    }
    report = {
        "schema_version": 1,
        "purpose": "frozen_holomotion_actual_state_foot_floor_v24_audit",
        "rollout": str(args.rollout_npz.resolve()),
        "reference": str(args.reference_npz.resolve()),
        "task_xml": str(args.task_xml.resolve()),
        "frame_count": int(len(qpos)),
        "inspection_contract": {
            "state_source": "logged actual MuJoCo mj_step rollout",
            "query": "mj_forward plus mj_geomDistance only; no stepping and no qpos reset",
            "sole_definition": "all collidable sphere geoms on each ankle_roll_link",
            "floor_definition": "geom named floor",
        },
        "thresholds": {"max_floor_penetration_m": args.max_floor_penetration_m},
        "sides": sides,
        "pass": bool(all(item["pass"] for item in sides.values())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
