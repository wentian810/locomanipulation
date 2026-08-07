#!/usr/bin/env python3
"""Solve short, evidence-backed GMR foot-contact windows without training.

The solver fuses background-compensated 2-D phases with GMR geometry.  It only
locks a foot when the image says it is planted; image-moving near-ground feet
are constrained to lift instead.  Root XY/Z and the two six-DoF legs are solved
jointly over short windows with hard per-frame bounds and temporal penalties.
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


SIDES = ("left", "right")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runs(mask: np.ndarray, minimum: int) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    start: int | None = None
    for index, enabled in enumerate(mask.tolist() + [False]):
        if enabled and start is None:
            start = index
        elif not enabled and start is not None:
            if index - start >= minimum:
                result.append(np.arange(start, index, dtype=np.int64))
            start = None
    return result


def _merge_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not windows:
        return []
    result: list[tuple[int, int]] = []
    for first, last in sorted(windows):
        if not result or first > result[-1][1] + 1:
            result.append((first, last))
        else:
            result[-1] = (result[-1][0], max(result[-1][1], last))
    return result


def _foot_meshes(model: mujoco.MjModel, body_name: str) -> tuple[int, list[tuple[int, np.ndarray]]]:
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body < 0:
        raise ValueError(f"missing body {body_name}")
    meshes: list[tuple[int, np.ndarray]] = []
    for geom in range(model.ngeom):
        if int(model.geom_bodyid[geom]) != body or int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        if int(model.geom_group[geom]) != 1:
            continue
        mesh = int(model.geom_dataid[geom])
        start = int(model.mesh_vertadr[mesh])
        count = int(model.mesh_vertnum[mesh])
        meshes.append((geom, model.mesh_vert[start : start + count].copy()))
    if not meshes:
        raise ValueError(f"no visible mesh for {body_name}")
    return body, meshes


def _state(data: mujoco.MjData, foot: tuple[int, list[tuple[int, np.ndarray]]]) -> tuple[np.ndarray, float]:
    body, meshes = foot
    low = float("inf")
    for geom, vertices in meshes:
        matrix = data.geom_xmat[geom].reshape(3, 3)
        low = min(low, float(np.min((vertices @ matrix.T + data.geom_xpos[geom])[:, 2])))
    return data.xpos[body, :2].copy(), low


def _set_pose(data: mujoco.MjData, root: np.ndarray, rotation: np.ndarray, dof: np.ndarray) -> None:
    data.qpos[:3] = root
    data.qpos[3:7] = rotation[[3, 0, 1, 2]]
    data.qpos[7:] = dof
    mujoco.mj_forward(data.model, data)


def _states(
    model: mujoco.MjModel,
    root: np.ndarray,
    rotation: np.ndarray,
    dof: np.ndarray,
    feet: dict[str, tuple[int, list[tuple[int, np.ndarray]]]],
) -> tuple[np.ndarray, np.ndarray]:
    xy = np.zeros((len(root), 2, 2), dtype=np.float64)
    height = np.zeros((len(root), 2), dtype=np.float64)
    data = mujoco.MjData(model)
    for frame in range(len(root)):
        _set_pose(data, root[frame], rotation[frame], dof[frame])
        for side_index, side in enumerate(SIDES):
            xy[frame, side_index], height[frame, side_index] = _state(data, feet[side])
    return xy, height


def _leg_indices(model: mujoco.MjModel, names: list[str], side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = [
        index
        for index, name in enumerate(names)
        if name.startswith(f"{side}_") and any(token in name for token in ("hip_", "knee", "ankle_"))
    ]
    if len(indices) != 6:
        raise ValueError(f"expected six {side} leg joints, found {len(indices)}")
    lower: list[float] = []
    upper: list[float] = []
    for index in indices:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, names[index])
        if joint < 0:
            raise ValueError(f"joint not present in XML: {names[index]}")
        lower.append(float(model.jnt_range[joint, 0]))
        upper.append(float(model.jnt_range[joint, 1]))
    return np.asarray(indices), np.asarray(lower), np.asarray(upper)


def _summary(vectors: np.ndarray) -> dict[str, float]:
    values = np.linalg.norm(np.asarray(vectors, dtype=np.float64), axis=-1).reshape(-1)
    return {
        "maximum": float(np.max(values, initial=0.0)),
        "median": float(np.median(values)) if len(values) else 0.0,
        "p95": float(np.percentile(values, 95)) if len(values) else 0.0,
    }


def _speed(xy: np.ndarray, fps: float) -> np.ndarray:
    result = np.zeros(xy.shape[:2], dtype=np.float64)
    if len(xy) > 1:
        result[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=2) * fps
        result[:-1] = np.maximum(result[:-1], result[1:])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-motion", required=True, type=Path)
    parser.add_argument("--robot-xml", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output-motion", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--speed-tolerance-m-s", type=float, default=0.25)
    parser.add_argument("--contact-height-m", type=float, default=0.015)
    parser.add_argument("--support-target-clearance-m", type=float, default=0.002)
    parser.add_argument("--swing-target-clearance-m", type=float, default=0.020)
    parser.add_argument("--minimum-episode-frames", type=int, default=5)
    parser.add_argument("--window-padding-frames", type=int, default=3)
    parser.add_argument("--max-root-xy-m", type=float, default=0.08)
    parser.add_argument("--max-root-z-m", type=float, default=0.035)
    parser.add_argument("--max-leg-joint-delta-rad", type=float, default=0.22)
    parser.add_argument("--max-leg-joint-l2-rad", type=float, default=0.34)
    parser.add_argument("--max-anchor-error-m", type=float, default=0.075)
    args = parser.parse_args()
    if not 0 < args.support_target_clearance_m < args.contact_height_m < args.swing_target_clearance_m:
        raise ValueError("clearance targets must be ordered support < contact < swing")

    with args.input_motion.open("rb") as stream:
        motion = pickle.load(stream)
    if not isinstance(motion, dict):
        raise ValueError("input motion is not a dictionary")
    root_base = np.asarray(motion["root_pos"], dtype=np.float64)
    rotation = np.asarray(motion["root_rot"], dtype=np.float64)
    dof_base = np.asarray(motion["dof_pos"], dtype=np.float64)
    frames = len(root_base)
    if root_base.shape != (frames, 3) or rotation.shape != (frames, 4) or dof_base.shape[0] != frames:
        raise ValueError("motion arrays have incompatible shapes")
    labels = {
        "left": np.asarray(motion.get("support_left_contact"), dtype=bool),
        "right": np.asarray(motion.get("support_right_contact"), dtype=bool),
    }
    if any(value.shape != (frames,) for value in labels.values()):
        raise ValueError("input motion lacks per-foot support labels")
    with np.load(args.evidence, allow_pickle=False) as archive:
        phases = {side: np.asarray(archive[f"{side}_phase"], dtype=np.int8) for side in SIDES}
    if any(value.shape != (frames,) for value in phases.values()):
        raise ValueError("evidence does not match robot frame count")
    fps = float(motion.get("fps", 30.0))
    model = mujoco.MjModel.from_xml_path(str(args.robot_xml))
    if model.nq - 7 != dof_base.shape[1]:
        raise ValueError("motion DoF count does not match robot XML")
    feet = {"left": _foot_meshes(model, "left_ankle_roll_link"), "right": _foot_meshes(model, "right_ankle_roll_link")}
    names = [str(value) for value in np.asarray(motion["dof_names"]).tolist()]
    legs = {side: _leg_indices(model, names, side) for side in SIDES}
    base_xy, base_height = _states(model, root_base, rotation, dof_base, feet)
    base_speed = _speed(base_xy, fps)

    anchor_xy = np.full((frames, 2, 2), np.nan, dtype=np.float64)
    support_goal = np.zeros((frames, 2), dtype=bool)
    swing_goal = np.zeros((frames, 2), dtype=bool)
    episode_records: list[dict[str, Any]] = []
    windows: list[tuple[int, int]] = []
    for side_index, side in enumerate(SIDES):
        planted = phases[side] == 1
        moving = phases[side] == 2
        near = (base_height[:, side_index] >= -0.005) & (base_height[:, side_index] <= args.contact_height_m)
        planted_slide = planted & near & (base_speed[:, side_index] > args.speed_tolerance_m_s)
        for episode in _runs(planted_slide, args.minimum_episode_frames):
            first, last = int(episode[0]), int(episode[-1])
            reference = first
            for candidate in range(first - 1, -1, -1):
                if planted[candidate] and near[candidate] and base_speed[candidate, side_index] <= args.speed_tolerance_m_s:
                    reference = candidate
                    break
            anchor_xy[episode, side_index] = base_xy[reference, side_index]
            support_goal[episode, side_index] = True
            windows.append((max(0, first - args.window_padding_frames), min(frames - 1, last + args.window_padding_frames)))
            episode_records.append({
                "side": side,
                "kind": "visual_planted_root_and_leg_lock",
                "frame_range": [first, last],
                "anchor_frame": reference,
                "requested_xy": _summary(base_xy[episode, side_index] - base_xy[reference, side_index]),
            })
        support_bad_height = labels[side] & planted & ((base_height[:, side_index] < -0.005) | (base_height[:, side_index] > args.contact_height_m))
        for episode in _runs(support_bad_height, args.minimum_episode_frames):
            support_goal[episode, side_index] = True
            windows.append((max(0, int(episode[0]) - args.window_padding_frames), min(frames - 1, int(episode[-1]) + args.window_padding_frames)))
        moving_slide = moving & near & (base_speed[:, side_index] > args.speed_tolerance_m_s)
        for episode in _runs(moving_slide, args.minimum_episode_frames):
            swing_goal[episode, side_index] = True
            windows.append((max(0, int(episode[0]) - args.window_padding_frames), min(frames - 1, int(episode[-1]) + args.window_padding_frames)))
            episode_records.append({
                "side": side,
                "kind": "visual_moving_swing_clearance",
                "frame_range": [int(episode[0]), int(episode[-1])],
            })
    windows = _merge_windows(windows)

    root = root_base.copy()
    dof = dof_base.copy()
    window_records: list[dict[str, Any]] = []
    failures: list[str] = []
    for first, last in windows:
        frame_ids = np.arange(first, last + 1, dtype=np.int64)
        count = len(frame_ids)
        root_reference = root_base[frame_ids].copy()
        dof_reference = dof_base[frame_ids].copy()
        lower_parts = [np.tile(np.array([-args.max_root_xy_m, -args.max_root_xy_m, -args.max_root_z_m]), count)]
        upper_parts = [np.tile(np.array([args.max_root_xy_m, args.max_root_xy_m, args.max_root_z_m]), count)]
        inherited_limit_mismatch = 0
        for side in SIDES:
            _, model_lower, model_upper = legs[side]
            ids = legs[side][0]
            local = dof_reference[:, ids]
            relative_lower = np.maximum(
                -args.max_leg_joint_delta_rad, model_lower[None] - local
            )
            relative_upper = np.minimum(
                args.max_leg_joint_delta_rad, model_upper[None] - local
            )
            # The frozen GMR output can already sit a few milliradians outside
            # the XML range. Keep that inherited state fixed at zero delta;
            # this optimizer must never reinterpret it as a new correction.
            inconsistent = relative_lower > relative_upper
            inherited_limit_mismatch += int(np.count_nonzero(inconsistent))
            relative_lower[inconsistent] = -args.max_leg_joint_delta_rad
            relative_upper[inconsistent] = args.max_leg_joint_delta_rad
            lower_parts.append(relative_lower.reshape(-1))
            upper_parts.append(relative_upper.reshape(-1))
        lower = np.concatenate(lower_parts)
        upper = np.concatenate(upper_parts)
        initial = np.clip(np.zeros_like(lower), lower, upper)
        data = mujoco.MjData(model)

        def unpack(values: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
            cursor = 0
            root_delta = values[cursor : cursor + count * 3].reshape(count, 3)
            cursor += count * 3
            leg_delta: dict[str, np.ndarray] = {}
            for side in SIDES:
                leg_delta[side] = values[cursor : cursor + count * 6].reshape(count, 6)
                cursor += count * 6
            return root_delta, leg_delta

        def residual(values: np.ndarray) -> np.ndarray:
            root_delta, leg_delta = unpack(values)
            output: list[float] = []
            for local_index, frame in enumerate(frame_ids):
                trial_dof = dof_reference[local_index].copy()
                for side in SIDES:
                    trial_dof[legs[side][0]] += leg_delta[side][local_index]
                _set_pose(data, root_reference[local_index] + root_delta[local_index], rotation[frame], trial_dof)
                for side_index, side in enumerate(SIDES):
                    xy, height = _state(data, feet[side])
                    if support_goal[frame, side_index]:
                        output.extend((120.0 * (height - args.support_target_clearance_m),))
                        anchor = anchor_xy[frame, side_index]
                        if np.all(np.isfinite(anchor)):
                            output.extend((120.0 * (xy - anchor)).tolist())
                    if swing_goal[frame, side_index]:
                        output.append(120.0 * max(0.0, args.swing_target_clearance_m - height))
            output.extend((6.0 * root_delta[:, :2]).reshape(-1).tolist())
            output.extend((12.0 * root_delta[:, 2]).reshape(-1).tolist())
            for side in SIDES:
                output.extend((7.0 * leg_delta[side]).reshape(-1).tolist())
            if count > 1:
                output.extend((28.0 * np.diff(root_delta, axis=0)).reshape(-1).tolist())
                for side in SIDES:
                    output.extend((14.0 * np.diff(leg_delta[side], axis=0)).reshape(-1).tolist())
            return np.asarray(output, dtype=np.float64)

        solution = least_squares(residual, initial, bounds=(lower, upper), max_nfev=260, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        root_delta, leg_delta = unpack(solution.x)
        root[frame_ids] = root_reference + root_delta
        for side in SIDES:
            dof[frame_ids[:, None], legs[side][0]] = dof_reference[:, legs[side][0]] + leg_delta[side]
        root_l2 = _summary(root_delta[:, :2])
        leg_l2 = {side: _summary(leg_delta[side]) for side in SIDES}
        if any(value["maximum"] > args.max_leg_joint_l2_rad + 1e-6 for value in leg_l2.values()):
            failures.append(f"window_{first}_{last}:leg_l2_exceeds_limit")
        window_records.append({
            "frame_range": [first, last],
            "solver_success": bool(solution.success),
            "cost": float(solution.cost),
            "root_xy_delta_m": root_l2,
            "root_z_delta_m": _summary(root_delta[:, 2:3]),
            "leg_delta_rad": leg_l2,
            "inherited_joint_limit_mismatch_entries": inherited_limit_mismatch,
        })
        if not solution.success:
            failures.append(f"window_{first}_{last}:solver_failed")

    final_xy, final_height = _states(model, root, rotation, dof, feet)
    final_speed = _speed(final_xy, fps)
    anchor_errors: list[float] = []
    for side_index, side in enumerate(SIDES):
        anchored = np.all(np.isfinite(anchor_xy[:, side_index]), axis=1)
        if np.any(anchored):
            anchor_errors.extend(np.linalg.norm(final_xy[anchored, side_index] - anchor_xy[anchored, side_index], axis=1).tolist())
        fulfilled_swing = swing_goal[:, side_index] & (final_height[:, side_index] >= args.swing_target_clearance_m - 0.002)
        labels[side][fulfilled_swing] = False
        unresolved_swing = swing_goal[:, side_index] & ~fulfilled_swing
        if np.any(unresolved_swing):
            failures.append(f"{side}:unresolved_swing_clearance")
    anchor_errors_array = np.asarray(anchor_errors, dtype=np.float64)
    if len(anchor_errors_array) and float(np.percentile(anchor_errors_array, 95)) > args.max_anchor_error_m:
        failures.append("anchor_error_p95_exceeds_limit")
    root_delta_total = root - root_base
    if float(np.max(np.linalg.norm(root_delta_total[:, :2], axis=1))) > args.max_root_xy_m + 1e-6:
        failures.append("root_xy_bound_exceeded")
    if float(np.max(np.abs(root_delta_total[:, 2]))) > args.max_root_z_m + 1e-6:
        failures.append("root_z_bound_exceeded")
    non_leg = np.ones(dof.shape[1], dtype=bool)
    for side in SIDES:
        non_leg[legs[side][0]] = False
    if np.any(np.abs(dof[:, non_leg] - dof_base[:, non_leg]) > 1e-8):
        failures.append("non_leg_dof_changed")

    output = dict(motion)
    output["root_pos"] = root.astype(np.asarray(motion["root_pos"]).dtype, copy=False)
    output["dof_pos"] = dof.astype(np.asarray(motion["dof_pos"]).dtype, copy=False)
    output["support_left_contact"] = labels["left"]
    output["support_right_contact"] = labels["right"]
    output["two_d_contact_window_optimization"] = {
        "kind": "background_compensated_2d_contact_window_ik",
        "input_sha256": _sha256(args.input_motion),
        "evidence_sha256": _sha256(args.evidence),
        "weights_modified": False,
    }
    args.output_motion.parent.mkdir(parents=True, exist_ok=True)
    with args.output_motion.open("wb") as stream:
        pickle.dump(output, stream)
    report = {
        "schema_version": 1,
        "kind": "background_compensated_2d_contact_window_ik",
        "input_motion": str(args.input_motion),
        "input_sha256": _sha256(args.input_motion),
        "evidence": str(args.evidence),
        "evidence_sha256": _sha256(args.evidence),
        "output_motion": str(args.output_motion),
        "output_sha256": _sha256(args.output_motion),
        "weights_modified": False,
        "root_rotation_unchanged": bool(np.array_equal(rotation, np.asarray(motion["root_rot"]))),
        "non_leg_dof_unchanged": not bool(np.any(np.abs(dof[:, non_leg] - dof_base[:, non_leg]) > 1e-8)),
        "episodes": episode_records,
        "windows": window_records,
        "anchor_error_m": {
            "maximum": float(np.max(anchor_errors_array, initial=0.0)),
            "p95": float(np.percentile(anchor_errors_array, 95)) if len(anchor_errors_array) else 0.0,
        },
        "final_speed_m_s": {side: _summary(final_speed[:, index:index + 1]) for index, side in enumerate(SIDES)},
        "failures": sorted(set(failures)),
        "accepted": not failures,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "accepted": report["accepted"],
        "window_count": len(windows),
        "anchor_error_m": report["anchor_error_m"],
        "failures": report["failures"],
    }))


if __name__ == "__main__":
    main()
