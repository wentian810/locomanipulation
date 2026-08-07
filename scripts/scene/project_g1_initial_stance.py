#!/usr/bin/env python3
"""Project a G1 reference's initial stance onto flat, physical foot contact.

This is an offline kinematic initialisation repair.  It finds a bounded change
to the two ankle-pitch and two ankle-roll joints, plus one *global* root-Z
translation, so both official foot collision patches are level with the task
floor.  The ankle correction can be held briefly then smoothly released into
the observed GMR motion.  Runtime still initializes once and advances only by
bounded torques and ``mj_step``.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares


ANKLE_JOINTS = (
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
)


def _smoothstep5(progress: np.ndarray) -> np.ndarray:
    progress = np.clip(progress, 0.0, 1.0)
    return progress**3 * (10.0 - 15.0 * progress + 6.0 * progress**2)


def _foot_geom_groups(model: mujoco.MjModel) -> tuple[list[int], list[int]]:
    groups: list[list[int]] = []
    for side in ("left", "right"):
        ids = []
        for geom_id in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            body = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])
            ) or ""
            if name.startswith("gmr_official_collision_") and body == f"{side}_ankle_roll_link":
                if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_SPHERE:
                    raise ValueError(f"{name} is not a sphere; point-contact stance projection is invalid")
                ids.append(geom_id)
        if not ids:
            raise ValueError(f"task lacks official foot collision geoms for {side} foot")
        groups.append(ids)
    return groups[0], groups[1]


def _qpos(motion: dict[str, object], frame: int) -> np.ndarray:
    root = np.asarray(motion["root_pos"], dtype=np.float64)[frame]
    quat_xyzw = np.asarray(motion["root_rot"], dtype=np.float64)[frame]
    dof = np.asarray(motion["dof_pos"], dtype=np.float64)[frame]
    return np.concatenate((root, quat_xyzw[[3, 0, 1, 2]], dof))


def _rebuild_local_body_positions(
    robot_xml: Path, root_pos: np.ndarray, root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray, body_names: np.ndarray,
) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    data = mujoco.MjData(model)
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    ids = np.asarray([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(name)) for name in body_names
    ], dtype=np.int32)
    if pelvis < 0 or np.any(ids < 0):
        raise ValueError("robot XML cannot rebuild all reference body positions")
    result = np.empty((len(root_pos), len(ids), 3), dtype=np.float32)
    for frame in range(len(root_pos)):
        data.qpos[:3] = root_pos[frame]
        data.qpos[3:7] = root_rot_xyzw[frame, [3, 0, 1, 2]]
        data.qpos[7:] = dof_pos[frame]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        rotation = data.xmat[pelvis].reshape(3, 3)
        result[frame] = (rotation.T @ (data.xpos[ids] - data.xpos[pelvis]).T).T
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-motion", required=True, type=Path)
    parser.add_argument("--probe-task-xml", required=True, type=Path)
    parser.add_argument("--robot-xml", required=True, type=Path)
    parser.add_argument("--output-motion", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--hold-through-frame", type=int, required=True)
    parser.add_argument("--ramp-end-frame", type=int, required=True)
    parser.add_argument("--target-foot-penetration-m", type=float, default=0.001)
    args = parser.parse_args()
    if args.output_motion.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite an initial-stance projection")
    if args.target_foot_penetration_m < 0.0:
        raise ValueError("target-foot-penetration-m must be non-negative")
    with args.input_motion.open("rb") as handle:
        motion = pickle.load(handle)
    for key in ("root_pos", "root_rot", "dof_pos", "dof_names", "link_body_list"):
        if key not in motion:
            raise ValueError(f"input motion lacks {key!r}")
    frame_count = len(np.asarray(motion["root_pos"]))
    if not 0 <= args.hold_through_frame < args.ramp_end_frame < frame_count:
        raise ValueError("require 0 <= hold-through-frame < ramp-end-frame < frame count")

    model = mujoco.MjModel.from_xml_path(str(args.probe_task_xml))
    source_qpos = _qpos(motion, 0)
    if source_qpos.shape != (model.nq,):
        raise ValueError("probe task qpos layout does not match input motion")
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ANKLE_JOINTS]
    if any(joint_id < 0 for joint_id in joint_ids):
        raise ValueError("probe task lacks one or more ankle joints")
    qpos_indices = np.asarray([model.jnt_qposadr[joint_id] for joint_id in joint_ids], dtype=np.int32)
    left_geoms, right_geoms = _foot_geom_groups(model)
    foot_geoms = left_geoms + right_geoms
    data = mujoco.MjData(model)
    surface_radius = np.asarray([model.geom_size[geom_id, 0] for geom_id in foot_geoms])

    def residual(x: np.ndarray) -> np.ndarray:
        data.qpos[:] = source_qpos
        data.qpos[2] = x[0]
        data.qpos[qpos_indices] = x[1:]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        surface_height = data.geom_xpos[foot_geoms, 2] - surface_radius
        # All eight contact patches share the same floor level.  A small joint
        # regularizer selects the closest GMR-compatible solution.
        return np.concatenate((
            100.0 * (surface_height + args.target_foot_penetration_m),
            0.25 * (x[1:] - source_qpos[qpos_indices]),
        ))

    lower = np.empty(5, dtype=np.float64)
    upper = np.empty(5, dtype=np.float64)
    lower[0], upper[0] = source_qpos[2] - 0.06, source_qpos[2] + 0.03
    for offset, joint_id in enumerate(joint_ids, start=1):
        lower[offset], upper[offset] = model.jnt_range[joint_id]
    initial = np.concatenate(([source_qpos[2]], source_qpos[qpos_indices]))
    solve = least_squares(
        residual, initial, bounds=(lower, upper), loss="soft_l1", f_scale=1.0,
        max_nfev=150, xtol=1e-10, ftol=1e-10, gtol=1e-10,
    )
    if not solve.success:
        raise RuntimeError(f"initial stance solver failed: {solve.message}")
    solved = solve.x
    data.qpos[:] = source_qpos
    data.qpos[2] = solved[0]
    data.qpos[qpos_indices] = solved[1:]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    surface_height = data.geom_xpos[foot_geoms, 2] - surface_radius

    result = copy.deepcopy(motion)
    root_pos = np.asarray(motion["root_pos"], dtype=np.float64).copy()
    root_rot = np.asarray(motion["root_rot"], dtype=np.float64).copy()
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64).copy()
    dof_names = [str(name) for name in motion["dof_names"]]
    dof_columns = [dof_names.index(name) for name in ANKLE_JOINTS]
    root_z_delta = float(solved[0] - source_qpos[2])
    ankle_delta = solved[1:] - source_qpos[qpos_indices]
    root_pos[:, 2] += root_z_delta
    weights = np.ones(frame_count, dtype=np.float64)
    release = np.arange(args.hold_through_frame + 1, args.ramp_end_frame + 1)
    weights[release] = 1.0 - _smoothstep5(
        (release - args.hold_through_frame) /
        (args.ramp_end_frame - args.hold_through_frame)
    )
    weights[args.ramp_end_frame + 1:] = 0.0
    dof_pos[:, dof_columns] += weights[:, None] * ankle_delta[None, :]
    result["root_pos"] = root_pos.astype(np.float32)
    result["dof_pos"] = dof_pos.astype(np.float32)
    result["local_body_pos"] = _rebuild_local_body_positions(
        args.robot_xml, root_pos, root_rot, dof_pos, np.asarray(motion["link_body_list"])
    )
    result["initial_stance_projection"] = {
        "mode": "flat_official_foot_collision_projection_with_smooth_ankle_release",
        "global_root_z_delta_m": root_z_delta,
        "ankle_delta_rad": dict(zip(ANKLE_JOINTS, (float(value) for value in ankle_delta))),
        "hold_through_frame": args.hold_through_frame,
        "ramp_end_frame": args.ramp_end_frame,
    }
    report = {
        "schema_version": 1,
        "purpose": "offline_initial_g1_stance_contact_projection",
        "status": "ready_for_bounded_mj_step_validation",
        "physical_contract": {
            "offline_change": "one global root-Z translation plus bounded ankle reference correction",
            "runtime": "initialize_once_then_bounded_actuator_torque_and_mj_step",
            "forbidden": ["runtime_root_state_reimposition", "xfrc_applied", "mocap_weld"],
        },
        "inputs": {"motion": str(args.input_motion), "probe_task_xml": str(args.probe_task_xml)},
        "solution": {
            "root_z_delta_m": root_z_delta,
            "ankle_delta_rad": dict(zip(ANKLE_JOINTS, (float(value) for value in ankle_delta))),
            "surface_height_m": {
                "min": float(np.min(surface_height)), "max": float(np.max(surface_height)),
            },
            "hold_through_frame": args.hold_through_frame,
            "ramp_end_frame": args.ramp_end_frame,
        },
        "output_motion": str(args.output_motion),
    }
    args.output_motion.parent.mkdir(parents=True, exist_ok=True)
    with args.output_motion.open("wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
