#!/usr/bin/env python3
"""Calibrate visible robot-foot ground contact without clip-specific frame rules.

This stage preserves the observed root x/y, root rotation, and all non-leg joints.
It runs only when an existing support-evidence stream exposes a small, stable
visible-mesh ground offset.  Otherwise it copies the input motion unchanged.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares


def _load(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _save(path: Path, motion: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(motion, handle)


def _support_labels(motion: dict, frames: int) -> tuple[np.ndarray, str]:
    """Return 0/1/2 labels from the richest available upstream evidence."""
    for key, value in reversed(list(motion.items())):
        if isinstance(value, dict) and "source_support_labels" in value:
            labels = np.asarray(value["source_support_labels"], dtype=np.int8)
            if labels.shape == (frames,):
                return labels, f"{key}.source_support_labels"
    left = np.asarray(motion.get("support_left_contact", []), dtype=bool)
    right = np.asarray(motion.get("support_right_contact", []), dtype=bool)
    if left.shape == (frames,) and right.shape == (frames,):
        labels = np.zeros(frames, dtype=np.int8)
        labels[left & ~right] = 1
        labels[right & ~left] = 2
        labels[left & right] = 1
        return labels, "support_left_contact/support_right_contact"
    return np.zeros(frames, dtype=np.int8), "none"


def _runs(mask: np.ndarray) -> list[list[int]]:
    result: list[list[int]] = []
    for item in np.flatnonzero(mask):
        frame = int(item)
        if not result or frame > result[-1][-1] + 1:
            result.append([frame])
        else:
            result[-1].append(frame)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-motion", required=True, type=Path)
    parser.add_argument("--robot-xml", required=True, type=Path)
    parser.add_argument(
        "--contact-anchors", type=Path, default=None,
        help="Optional semantic sit mask; omitted for generic support-only calibration.",
    )
    parser.add_argument("--output-motion", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--left-foot-body", default="left_ankle_roll_link")
    parser.add_argument("--right-foot-body", default="right_ankle_roll_link")
    parser.add_argument("--target-clearance-m", type=float, default=0.0)
    parser.add_argument(
        "--contact-safety-margin-m", type=float, default=0.001,
        help=(
            "Positive clearance reserved inside the contact objective so numerical "
            "least-squares residuals cannot turn a nominal zero-clearance stance "
            "into visible penetration."
        ),
    )
    parser.add_argument("--max-global-offset-m", type=float, default=0.005)
    parser.add_argument("--max-support-spread-m", type=float, default=0.002)
    parser.add_argument("--min-support-coverage", type=float, default=0.8)
    parser.add_argument("--sit-fade-frames", type=int, default=10)
    parser.add_argument("--repair-trigger-m", type=float, default=-0.001)
    parser.add_argument("--max-stance-clearance-m", type=float, default=0.012)
    parser.add_argument("--min-stance-repair-frames", type=int, default=3)
    parser.add_argument("--temporal-padding-frames", type=int, default=2)
    parser.add_argument("--max-local-joint-delta-rad", type=float, default=0.08)
    parser.add_argument("--max-local-root-z-m", type=float, default=0.06)
    parser.add_argument("--root-z-regularization", type=float, default=16.0)
    parser.add_argument("--root-temporal-weight", type=float, default=100.0)
    parser.add_argument("--stance-contact-weight", type=float, default=6000.0)
    args = parser.parse_args()
    contact_objective_m = args.target_clearance_m + args.contact_safety_margin_m
    if args.max_stance_clearance_m <= contact_objective_m:
        raise ValueError(
            "--max-stance-clearance-m must exceed target clearance plus the contact safety margin"
        )
    if args.min_stance_repair_frames < 1:
        raise ValueError("--min-stance-repair-frames must be positive")
    if args.max_local_root_z_m <= 0.0:
        raise ValueError("--max-local-root-z-m must be positive")
    if args.stance_contact_weight <= 0.0:
        raise ValueError("--stance-contact-weight must be positive")

    motion = _load(args.input_motion)
    root = np.asarray(motion["root_pos"], dtype=np.float64)
    quat = np.asarray(motion["root_rot"], dtype=np.float64)
    q_original = np.asarray(motion["dof_pos"], dtype=np.float64)
    names = [str(item) for item in np.asarray(motion["dof_names"]).tolist()]
    frames = len(root)
    if quat.shape != (frames, 4) or q_original.shape[0] != frames:
        raise ValueError("motion arrays do not share a frame dimension")
    if args.contact_anchors is None:
        sit_mask = np.zeros(frames, dtype=bool)
    else:
        anchors = np.load(args.contact_anchors)
        if "sit_mask" not in anchors:
            raise ValueError("contact anchors are missing sit_mask")
        sit_mask = np.asarray(anchors["sit_mask"], dtype=bool)
        if sit_mask.shape != (frames,):
            raise ValueError("contact anchors do not match motion")
    sit_indices = np.flatnonzero(sit_mask)
    pre_sit = int(sit_indices[0]) if len(sit_indices) else frames

    model = mujoco.MjModel.from_xml_path(str(args.robot_xml))
    data = mujoco.MjData(model)
    foot_bodies = {"left": args.left_foot_body, "right": args.right_foot_body}
    foot_meshes: dict[str, list[int]] = {}
    for side, body_name in foot_bodies.items():
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body < 0:
            raise ValueError(f"foot body {body_name!r} not found")
        geoms = [
            geom for geom in range(model.ngeom)
            if model.geom_bodyid[geom] == body
            and model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
            and model.geom_group[geom] == 1
        ]
        if not geoms:
            raise ValueError(f"{body_name!r} has no visible mesh")
        foot_meshes[side] = geoms

    def foot_min(frame: int, side: str, joints: np.ndarray, roots: np.ndarray) -> float:
        data.qpos[:3] = roots[frame]
        data.qpos[3:7] = quat[frame][[3, 0, 1, 2]]
        data.qpos[7:] = joints
        mujoco.mj_forward(model, data)
        heights: list[np.ndarray] = []
        for geom in foot_meshes[side]:
            start = model.mesh_vertadr[model.geom_dataid[geom]]
            count = model.mesh_vertnum[model.geom_dataid[geom]]
            vertices = model.mesh_vert[start:start + count]
            rotation = data.geom_xmat[geom].reshape(3, 3)
            heights.append((data.geom_xpos[geom] + vertices @ rotation.T)[:, 2])
        return float(np.min(np.concatenate(heights)))

    labels, support_source = _support_labels(motion, frames)
    raw_left = np.asarray(motion.get("support_left_contact", []), dtype=bool)
    raw_right = np.asarray(motion.get("support_right_contact", []), dtype=bool)
    if raw_left.shape == (frames,) and raw_right.shape == (frames,):
        support_masks = {"left": raw_left, "right": raw_right}
        support_source = "support_left_contact/support_right_contact"
    else:
        support_masks = {
            "left": labels == 1,
            "right": labels == 2,
        }

    def long_stance(mask: np.ndarray) -> np.ndarray:
        eligible = np.zeros(frames, dtype=bool)
        for run in _runs(mask):
            if len(run) >= args.min_stance_repair_frames:
                eligible[np.asarray(run, dtype=np.int64)] = True
        return eligible

    eligible_masks = {side: long_stance(mask) for side, mask in support_masks.items()}
    q = q_original.copy()
    calibrated_root = root.copy()
    support_values = []
    for frame in range(pre_sit):
        if support_masks["left"][frame]:
            support_values.append(foot_min(frame, "left", q[frame], calibrated_root))
        if support_masks["right"][frame]:
            support_values.append(foot_min(frame, "right", q[frame], calibrated_root))
    support_values = np.asarray(support_values, dtype=np.float64)
    support_frames = np.logical_or(
        support_masks["left"][:pre_sit],
        support_masks["right"][:pre_sit],
    )
    coverage = float(np.count_nonzero(support_frames) / max(pre_sit, 1))
    median = float(np.median(support_values)) if len(support_values) else float("nan")
    spread = (
        float(np.quantile(support_values, 0.9) - np.quantile(support_values, 0.1))
        if len(support_values) else float("inf")
    )
    offset = median - args.target_clearance_m if len(support_values) else float("nan")
    global_root_calibration_accepted = bool(
        coverage >= args.min_support_coverage
        and 0.0 < offset <= args.max_global_offset_m
        and spread <= args.max_support_spread_m
    )
    episodes: list[dict] = []
    if global_root_calibration_accepted:
        weight = np.zeros(frames, dtype=np.float64)
        weight[:pre_sit] = 1.0
        for index in range(args.sit_fade_frames):
            frame = pre_sit + index
            if frame < frames:
                weight[frame] = 0.5 + 0.5 * np.cos(
                    np.pi * (index + 1) / (args.sit_fade_frames + 1)
                )
        calibrated_root[:, 2] -= offset * weight

    before = np.array([
        [foot_min(frame, side, q[frame], calibrated_root) for side in ("left", "right")]
        for frame in range(pre_sit)
    ])
    violation = np.zeros(pre_sit, dtype=bool)
    for side_index, side in enumerate(("left", "right")):
        eligible = eligible_masks[side][:pre_sit]
        violation |= eligible & (
            (before[:, side_index] < args.repair_trigger_m)
            | (before[:, side_index] > args.max_stance_clearance_m)
        )
    for run in _runs(violation):
        active_first, active_last = run[0], run[-1]
        first = max(0, active_first - args.temporal_padding_frames)
        last = min(pre_sit - 1, active_last + args.temporal_padding_frames)
        frame_ids = np.arange(first, last + 1)
        active = (frame_ids >= active_first) & (frame_ids <= active_last)
        sides = [
            side for side in ("left", "right")
            if np.any(eligible_masks[side][frame_ids[active]])
        ]
        if not sides:
            continue
        joint_ids = {
            side: np.asarray([
                index for index, name in enumerate(names)
                if name.startswith(f"{side}_")
                and any(token in name for token in ("hip_", "knee", "ankle_"))
            ], dtype=np.int32)
            for side in sides
        }
        if any(len(ids) != 6 for ids in joint_ids.values()):
            raise ValueError("expected six leg joints for every constrained support side")
        base = {side: q[frame_ids][:, ids].copy() for side, ids in joint_ids.items()}
        base_root_z = calibrated_root[frame_ids, 2].copy()
        lower = [np.full(len(frame_ids), -args.max_local_root_z_m)]
        upper = [np.full(len(frame_ids), args.max_local_root_z_m)]
        for side in sides:
            ids = joint_ids[side]
            lower.append(np.tile(model.jnt_range[ids + 1, 0], len(frame_ids)))
            upper.append(np.tile(model.jnt_range[ids + 1, 1], len(frame_ids)))

        def residual(values: np.ndarray) -> np.ndarray:
            root_delta = values[:len(frame_ids)]
            cursor = len(frame_ids)
            shaped: dict[str, np.ndarray] = {}
            trial = q[frame_ids].copy()
            for side in sides:
                count = len(frame_ids) * len(joint_ids[side])
                shaped[side] = values[cursor:cursor + count].reshape(base[side].shape)
                trial[:, joint_ids[side]] = shaped[side]
                cursor += count
            trial_roots = calibrated_root.copy()
            trial_roots[frame_ids, 2] = base_root_z + root_delta
            result: list[float] = []
            for local_index, frame in enumerate(frame_ids):
                for side in sides:
                    if eligible_masks[side][frame]:
                        result.append(
                            args.stance_contact_weight * (
                                foot_min(int(frame), side, trial[local_index], trial_roots)
                                - contact_objective_m
                            )
                        )
            result.extend(args.root_z_regularization * root_delta)
            for side in sides:
                result.extend((22.0 * (shaped[side] - base[side])).ravel())
            if len(frame_ids) > 1:
                result.extend(args.root_temporal_weight * np.diff(root_delta))
                for side in sides:
                    result.extend((70.0 * np.diff(shaped[side] - base[side], axis=0)).ravel())
            return np.asarray(result)

        initial = np.concatenate(
            [np.zeros(len(frame_ids)), *(value.ravel() for value in base.values())]
        )
        solution = least_squares(
            residual, initial, bounds=(np.concatenate(lower), np.concatenate(upper)),
            max_nfev=320, ftol=1e-8, xtol=1e-8, gtol=1e-8,
        )
        root_delta = solution.x[:len(frame_ids)]
        calibrated_root[frame_ids, 2] = base_root_z + root_delta
        cursor = len(frame_ids)
        joint_delta_max = 0.0
        for side in sides:
            count = len(frame_ids) * len(joint_ids[side])
            shaped = solution.x[cursor:cursor + count].reshape(base[side].shape)
            q[frame_ids[:, None], joint_ids[side]] = shaped
            joint_delta_max = max(
                joint_delta_max,
                float(np.max(np.linalg.norm(shaped - base[side], axis=1))),
            )
            cursor += count
        active_before = [
            before[frame_ids[active], side_index]
            for side_index, side in enumerate(("left", "right"))
            if side in sides
        ]
        episodes.append({
            "sides": sides,
            "reason": (
                "penetration"
                if min(float(np.min(value)) for value in active_before) < args.repair_trigger_m
                else "floating_stance"
            ),
            "active_range": [active_first, active_last],
            "window": [first, last],
            "joint_delta_l2_max": joint_delta_max,
            "root_z_delta_max_m": float(np.max(np.abs(root_delta))),
        })
    after = np.array([
        [foot_min(frame, side, q[frame], calibrated_root) for side in ("left", "right")]
        for frame in range(pre_sit)
    ])
    local_delta = float(np.max(np.linalg.norm(q - q_original, axis=1))
                        if len(q) else 0.0)
    root_z_delta = float(np.max(np.abs(calibrated_root[:, 2] - root[:, 2]))
                         if len(root) else 0.0)
    unresolved = []
    for side_index, side in enumerate(("left", "right")):
        eligible = eligible_masks[side][:pre_sit]
        invalid = eligible & (
            (after[:, side_index] < args.repair_trigger_m - 0.00005)
            | (after[:, side_index] > args.max_stance_clearance_m + 0.00005)
        )
        unresolved.extend(
            {"side": side, "range": [run[0], run[-1]]}
            for run in _runs(invalid)
        )
    accepted = bool(
        (np.any(eligible_masks["left"][:pre_sit]) or np.any(eligible_masks["right"][:pre_sit]))
        and local_delta <= args.max_local_joint_delta_rad
        and root_z_delta <= args.max_local_root_z_m
        and not unresolved
    )
    if not accepted:
        calibrated_root = root.copy()
        q = q_original.copy()
        episodes = []

    output = dict(motion)
    output["root_pos"] = calibrated_root
    output["dof_pos"] = q.astype(np.float32)
    report = {
        "schema_version": 1,
        "method": "visible_render_mesh_stance_contact_calibration_with_temporal_local_ik",
        "accepted": accepted,
        "support_source": support_source,
        "pre_sit_frames": pre_sit,
        "support_coverage": coverage,
        "eligible_stance_frames": {
            side: int(np.count_nonzero(mask[:pre_sit]))
            for side, mask in eligible_masks.items()
        },
        "global_root_calibration_accepted": global_root_calibration_accepted,
        "support_clearance_median_m": median,
        "support_clearance_p10_p90_spread_m": spread,
        "target_clearance_m": args.target_clearance_m,
        "contact_objective_clearance_m": contact_objective_m,
        "contact_safety_margin_m": args.contact_safety_margin_m,
        "applied_root_z_offset_m": float(-offset) if accepted else 0.0,
        "max_global_offset_m": args.max_global_offset_m,
        "root_xy_unchanged": True,
        "root_rotation_unchanged": True,
        "minimum_visible_foot_clearance_before_m": float(np.min(before)) if len(before) else None,
        "minimum_visible_foot_clearance_after_m": float(np.min(after)) if len(after) else None,
        "maximum_stance_clearance_before_m": {
            side: (
                float(np.max(before[eligible_masks[side][:pre_sit], index]))
                if np.any(eligible_masks[side][:pre_sit]) else None
            )
            for index, side in enumerate(("left", "right"))
        },
        "maximum_stance_clearance_after_m": {
            side: (
                float(np.max(after[eligible_masks[side][:pre_sit], index]))
                if np.any(eligible_masks[side][:pre_sit]) else None
            )
            for index, side in enumerate(("left", "right"))
        },
        "max_stance_clearance_m": args.max_stance_clearance_m,
        "max_local_joint_delta_rad": local_delta,
        "max_root_z_correction_m": root_z_delta,
        "max_allowed_root_z_correction_m": args.max_local_root_z_m,
        "stance_contact_weight": args.stance_contact_weight,
        "episodes": episodes,
        "unresolved_stance_episodes": unresolved,
    }
    output["visible_ground_contact_calibration"] = report
    _save(args.output_motion, output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
