#!/usr/bin/env python3
"""Apply a bounded, contact-state-aware local leg IK postprocess to GMR motion.

This utility is deliberately independent of a clip identity.  It never changes
root translation, root rotation, or non-leg joints.  A fast foot that remains
near the floor is treated as a swing candidate and lifted by local leg IK; a
slow labelled support foot is brought to a small positive sole clearance.
Every local solve is bounded and recorded so the downstream quality gate can
reject the result rather than hiding an infeasible recovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.optimize import least_squares


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        motion = pickle.load(stream)
    if not isinstance(motion, dict):
        raise ValueError("robot motion must be a dictionary")
    return motion


def _save(path: Path, motion: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(motion, stream)


def _foot_meshes(
    model: mujoco.MjModel, body_name: str
) -> tuple[int, list[tuple[int, np.ndarray]]]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"missing foot body: {body_name}")
    meshes: list[tuple[int, np.ndarray]] = []
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != body_id:
            continue
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        if int(model.geom_group[geom_id]) != 1:
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        meshes.append((geom_id, model.mesh_vert[start : start + count].copy()))
    if not meshes:
        raise ValueError(f"{body_name} has no visible mesh")
    return body_id, meshes


def _foot_state(
    data: mujoco.MjData,
    body_id: int,
    meshes: list[tuple[int, np.ndarray]],
) -> tuple[np.ndarray, float]:
    min_z = float("inf")
    for geom_id, vertices in meshes:
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        world = vertices @ rotation.T + data.geom_xpos[geom_id]
        min_z = min(min_z, float(np.min(world[:, 2])))
    return data.xpos[body_id, :2].copy(), min_z


def _set_pose(
    data: mujoco.MjData,
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof: np.ndarray,
) -> None:
    data.qpos[:3] = root_pos
    data.qpos[3:7] = root_rot_xyzw[[3, 0, 1, 2]]
    data.qpos[7:] = dof
    mujoco.mj_forward(data.model, data)


def _all_states(
    model: mujoco.MjModel,
    root: np.ndarray,
    root_rot: np.ndarray,
    dof: np.ndarray,
    feet: dict[str, tuple[int, list[tuple[int, np.ndarray]]]],
) -> tuple[np.ndarray, np.ndarray]:
    frames = len(root)
    xy = np.zeros((frames, 2, 2), dtype=np.float64)
    clearance = np.zeros((frames, 2), dtype=np.float64)
    data = mujoco.MjData(model)
    for frame in range(frames):
        _set_pose(data, root[frame], root_rot[frame], dof[frame])
        for side_index, side in enumerate(("left", "right")):
            xy[frame, side_index], clearance[frame, side_index] = _foot_state(
                data, *feet[side]
            )
    return xy, clearance


def _runs(mask: np.ndarray) -> list[np.ndarray]:
    runs: list[np.ndarray] = []
    start: int | None = None
    for index, enabled in enumerate(mask.tolist() + [False]):
        if enabled and start is None:
            start = index
        elif not enabled and start is not None:
            runs.append(np.arange(start, index, dtype=np.int64))
            start = None
    return runs


def _leg_indices(
    model: mujoco.MjModel, dof_names: list[str], side: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = [
        index
        for index, name in enumerate(dof_names)
        if name.startswith(f"{side}_")
        and any(token in name for token in ("hip_", "knee", "ankle_"))
    ]
    if len(ids) != 6:
        raise ValueError(f"expected six {side} leg DoFs, found {len(ids)}")
    lower = []
    upper = []
    for index in ids:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, dof_names[index]
        )
        if joint_id < 0:
            raise ValueError(f"robot joint is missing: {dof_names[index]}")
        lower.append(float(model.jnt_range[joint_id, 0]))
        upper.append(float(model.jnt_range[joint_id, 1]))
    return np.asarray(ids, dtype=np.int64), np.asarray(lower), np.asarray(upper)


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-motion", required=True, type=Path)
    parser.add_argument("--robot-xml", required=True, type=Path)
    parser.add_argument("--output-motion", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--left-foot-body", default="left_ankle_roll_link")
    parser.add_argument("--right-foot-body", default="right_ankle_roll_link")
    parser.add_argument("--ground-z-m", type=float, default=0.0)
    parser.add_argument("--max-stable-speed-m-s", type=float, default=0.08)
    parser.add_argument("--support-contact-height-m", type=float, default=0.015)
    parser.add_argument("--target-support-clearance-m", type=float, default=0.001)
    parser.add_argument("--target-swing-clearance-m", type=float, default=0.018)
    parser.add_argument("--max-local-joint-delta-rad", type=float, default=0.12)
    parser.add_argument("--max-local-joint-l2-rad", type=float, default=0.20)
    parser.add_argument("--max-swing-relabel-ratio", type=float, default=0.35)
    parser.add_argument("--min-support-frames-per-foot", type=int, default=3)
    args = parser.parse_args()
    if not 0.0 < args.target_support_clearance_m < args.support_contact_height_m:
        raise ValueError("support clearance target must be inside the contact band")
    if args.target_swing_clearance_m <= args.support_contact_height_m:
        raise ValueError("swing clearance target must be above the contact band")
    if args.max_local_joint_delta_rad <= 0.0 or args.max_local_joint_l2_rad <= 0.0:
        raise ValueError("local joint bounds must be positive")

    motion = _load(args.input_motion)
    root = np.asarray(motion["root_pos"], dtype=np.float64)
    root_rot = np.asarray(motion["root_rot"], dtype=np.float64)
    original_dof = np.asarray(motion["dof_pos"], dtype=np.float64)
    frames = len(root)
    if root.shape != (frames, 3) or root_rot.shape != (frames, 4):
        raise ValueError("root arrays have inconsistent shapes")
    if original_dof.ndim != 2 or original_dof.shape[0] != frames:
        raise ValueError("dof array has incompatible shape")
    left_labels = np.asarray(motion.get("support_left_contact", []), dtype=bool)
    right_labels = np.asarray(motion.get("support_right_contact", []), dtype=bool)
    if left_labels.shape != (frames,) or right_labels.shape != (frames,):
        raise ValueError("input motion lacks per-foot support labels")
    dof_names = [str(item) for item in np.asarray(motion["dof_names"]).tolist()]
    if len(dof_names) != original_dof.shape[1]:
        raise ValueError("dof_names do not match dof_pos")
    fps = float(motion.get("fps", 30.0))
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("motion fps must be positive")

    model = mujoco.MjModel.from_xml_path(str(args.robot_xml))
    if model.nq - 7 != original_dof.shape[1]:
        raise ValueError("robot XML DoF count does not match motion")
    feet = {
        "left": _foot_meshes(model, args.left_foot_body),
        "right": _foot_meshes(model, args.right_foot_body),
    }
    indices = {side: _leg_indices(model, dof_names, side) for side in feet}
    initial_xy, initial_z = _all_states(model, root, root_rot, original_dof, feet)
    initial_z -= args.ground_z_m
    initial_speed = np.zeros((frames, 2), dtype=np.float64)
    if frames > 1:
        initial_speed[1:] = np.linalg.norm(np.diff(initial_xy, axis=0), axis=2) * fps
        initial_speed[:-1] = np.maximum(initial_speed[:-1], initial_speed[1:])

    dof = original_dof.copy()
    labels = {"left": left_labels.copy(), "right": right_labels.copy()}
    tasks: list[dict[str, Any]] = []
    for side_index, side in enumerate(("left", "right")):
        near = (
            (initial_z[:, side_index] >= -0.005)
            & (initial_z[:, side_index] <= args.support_contact_height_m)
        )
        moving = near & (initial_speed[:, side_index] > args.max_stable_speed_m_s)
        support_repair = labels[side] & ~moving & (
            (initial_z[:, side_index] < -0.005)
            | (initial_z[:, side_index] > args.support_contact_height_m)
        )
        for frame in np.flatnonzero(moving):
            tasks.append({"frame": int(frame), "side": side, "kind": "swing_lift"})
        for frame in np.flatnonzero(support_repair):
            tasks.append({"frame": int(frame), "side": side, "kind": "support_level"})

    solve_data = mujoco.MjData(model)
    task_reports: list[dict[str, Any]] = []
    failures: list[str] = []
    for task in tasks:
        frame = task["frame"]
        side = task["side"]
        kind = task["kind"]
        side_index = 0 if side == "left" else 1
        ids, model_lower, model_upper = indices[side]
        base = dof[frame, ids].copy()
        lower = np.maximum(model_lower, base - args.max_local_joint_delta_rad)
        upper = np.minimum(model_upper, base + args.max_local_joint_delta_rad)
        target = (
            args.target_swing_clearance_m
            if kind == "swing_lift"
            else args.target_support_clearance_m
        )

        def state(candidate: np.ndarray) -> float:
            trial = dof[frame].copy()
            trial[ids] = candidate
            _set_pose(solve_data, root[frame], root_rot[frame], trial)
            return _foot_state(solve_data, *feet[side])[1] - args.ground_z_m

        def residual(candidate: np.ndarray) -> np.ndarray:
            height = state(candidate)
            if kind == "swing_lift":
                contact = max(0.0, target - height)
            else:
                contact = height - target
            return np.concatenate((
                np.asarray([500.0 * contact], dtype=np.float64),
                12.0 * (candidate - base),
            ))

        solution = least_squares(
            residual,
            base,
            bounds=(lower, upper),
            max_nfev=100,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        height = state(solution.x)
        delta = solution.x - base
        reached = (
            height >= target - 0.0015
            if kind == "swing_lift"
            else abs(height - target) <= 0.003
        )
        within_bound = (
            float(np.max(np.abs(delta))) <= args.max_local_joint_delta_rad + 1e-7
            and float(np.linalg.norm(delta)) <= args.max_local_joint_l2_rad + 1e-7
        )
        accepted = bool(solution.success and reached and within_bound)
        task_reports.append({
            **task,
            "initial_clearance_m": float(initial_z[frame, side_index]),
            "final_clearance_m": float(height),
            "target_clearance_m": float(target),
            "joint_delta_l2_rad": float(np.linalg.norm(delta)),
            "joint_delta_max_abs_rad": float(np.max(np.abs(delta))),
            "solver_success": bool(solution.success),
            "accepted": accepted,
        })
        if accepted:
            dof[frame, ids] = solution.x
            if kind == "swing_lift":
                labels[side][frame] = False
        else:
            failures.append(f"{side}:{kind}:frame_{frame}")

    final_xy, final_z = _all_states(model, root, root_rot, dof, feet)
    final_z -= args.ground_z_m
    final_speed = np.zeros((frames, 2), dtype=np.float64)
    if frames > 1:
        final_speed[1:] = np.linalg.norm(np.diff(final_xy, axis=0), axis=2) * fps
        final_speed[:-1] = np.maximum(final_speed[:-1], final_speed[1:])
    changed = np.abs(dof - original_dof) > 1e-9
    leg_mask = np.zeros(original_dof.shape[1], dtype=bool)
    for ids, _, _ in indices.values():
        leg_mask[ids] = True
    if np.any(changed[:, ~leg_mask]):
        failures.append("non_leg_joint_changed")
    relabelled: dict[str, int] = {}
    for side_index, side in enumerate(("left", "right")):
        original = left_labels if side == "left" else right_labels
        count = int(np.count_nonzero(original & ~labels[side]))
        relabelled[side] = count
        original_count = int(np.count_nonzero(original))
        if original_count and count / original_count > args.max_swing_relabel_ratio:
            failures.append(f"{side}:swing_relabel_ratio")
        stable = labels[side] & (final_z[:, side_index] >= -0.005) & (
            final_z[:, side_index] <= args.support_contact_height_m
        ) & (final_speed[:, side_index] <= args.max_stable_speed_m_s)
        if int(np.count_nonzero(stable)) < args.min_support_frames_per_foot:
            failures.append(f"{side}:insufficient_stable_support")

    result = dict(motion)
    result["dof_pos"] = dof.astype(np.asarray(motion["dof_pos"]).dtype, copy=False)
    result["support_left_contact"] = labels["left"]
    result["support_right_contact"] = labels["right"]
    result["contact_postprocess"] = {
        "kind": "bounded_local_leg_ik_swing_clearance",
        "input_sha256": _sha256(args.input_motion),
        "root_xy_unchanged": bool(np.array_equal(root, np.asarray(motion["root_pos"]))),
        "root_rotation_unchanged": bool(np.array_equal(root_rot, np.asarray(motion["root_rot"]))),
        "weights_modified": False,
    }
    _save(args.output_motion, result)
    report = {
        "schema_version": 1,
        "kind": "bounded_local_leg_ik_swing_clearance",
        "input_motion": str(args.input_motion),
        "input_sha256": _sha256(args.input_motion),
        "output_motion": str(args.output_motion),
        "output_sha256": _sha256(args.output_motion),
        "weights_modified": False,
        "root_pos_unchanged": bool(np.array_equal(root, np.asarray(result["root_pos"]))),
        "root_rot_unchanged": bool(np.array_equal(root_rot, np.asarray(result["root_rot"]))),
        "non_leg_joint_unchanged": not bool(np.any(changed[:, ~leg_mask])),
        "tasks": task_reports,
        "task_count": int(len(tasks)),
        "accepted_task_count": int(sum(item["accepted"] for item in task_reports)),
        "swing_relabelled_frame_count": relabelled,
        "initial_sole_clearance_m": {
            side: _summary(initial_z[:, index])
            for index, side in enumerate(("left", "right"))
        },
        "final_sole_clearance_m": {
            side: _summary(final_z[:, index])
            for index, side in enumerate(("left", "right"))
        },
        "failures": sorted(set(failures)),
        "accepted": not failures,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "accepted": report["accepted"],
        "task_count": report["task_count"],
        "accepted_task_count": report["accepted_task_count"],
        "failures": report["failures"],
    }))


if __name__ == "__main__":
    main()
