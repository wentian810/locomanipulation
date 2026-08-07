#!/usr/bin/env python3
"""Constrained robot-side seat-contact projection for a static semantic chair."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize

from evaluate_gmr_chair_contacts import (
    _body_name,
    _combine_mjcf,
    _load_motion,
    _robot_chair_contact_geoms,
    _robot_collision_geoms,
)


def set_qpos(data, root, rot, dof):
    data.qpos[:3] = root
    data.qpos[3:7] = rot[[3, 0, 1, 2]]
    data.qpos[7:] = dof


def names(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Project a GMR sit episode above a semantic-chair seat."
    )
    parser.add_argument("--robot-motion", required=True, type=Path)
    parser.add_argument("--scene-mujoco-xml", required=True, type=Path)
    parser.add_argument("--robot-xml", required=True, type=Path)
    parser.add_argument("--contact-anchors", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--seat-geom", default="seat_support_geom")
    parser.add_argument(
        "--backrest-geom",
        default="",
        help="Optional semantic-chair backrest geom; enables a pelvis/backrest constraint.",
    )
    parser.add_argument("--support-body", default="right_hip_pitch_link")
    parser.add_argument("--back-support-body", default="pelvis")
    parser.add_argument(
        "--repair-joints",
        default=(
            "left_hip_pitch_joint,left_hip_roll_joint,left_hip_yaw_joint,"
            "left_knee_joint,right_hip_pitch_joint,right_hip_roll_joint,"
            "right_hip_yaw_joint,right_knee_joint"
        ),
    )
    parser.add_argument("--clearance-m", type=float, default=0.003)
    parser.add_argument("--support-target-m", type=float, default=0.008)
    parser.add_argument("--activation-distance-m", type=float, default=0.35)
    parser.add_argument(
        "--support-proximity-m",
        type=float,
        default=0.06,
        help=(
            "Only start a root-height support correction after the semantic "
            "support body is this close to the seat in the unmodified motion."
        ),
    )
    parser.add_argument(
        "--minimum-proximity-frames",
        type=int,
        default=8,
        help="Require this many consecutive near-seat frames before correcting.",
    )
    parser.add_argument(
        "--root-z-ramp-frames",
        type=int,
        default=12,
        help="C1 ramp length before a persistent semantic-seat support event.",
    )
    parser.add_argument("--back-contact-target-m", type=float, default=0.025)
    parser.add_argument("--max-back-shift-m", type=float, default=0.20)
    parser.add_argument("--back-objective-weight", type=float, default=40.0)
    parser.add_argument("--back-ramp-frames", type=int, default=6)
    parser.add_argument("--min-root-dz-m", type=float, default=-0.18)
    parser.add_argument("--max-root-dz-m", type=float, default=0.05)
    parser.add_argument("--sample-stride", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=350)
    args = parser.parse_args()
    if args.sample_stride < 1 or args.maxiter < 1:
        raise ValueError("sample-stride and maxiter must be positive")
    if args.clearance_m < 0 or args.support_target_m < args.clearance_m:
        raise ValueError("support target must exceed clearance")
    if args.back_contact_target_m < args.clearance_m:
        raise ValueError("back-contact target must exceed clearance")
    if args.max_back_shift_m < 0 or args.back_objective_weight < 0:
        raise ValueError("back-shift bound and objective weight must be non-negative")
    if args.back_ramp_frames < 0:
        raise ValueError("back-ramp-frames must be non-negative")
    if args.support_proximity_m <= args.clearance_m:
        raise ValueError("support-proximity-m must exceed clearance-m")
    if args.minimum_proximity_frames < 1 or args.root_z_ramp_frames < 0:
        raise ValueError("invalid semantic support persistence or ramp length")

    motion = _load_motion(args.robot_motion)
    root = np.asarray(motion["root_pos"], dtype=np.float64)
    rot = np.asarray(motion["root_rot"], dtype=np.float64)
    dof = np.asarray(motion["dof_pos"], dtype=np.float64)
    anchors = np.load(args.contact_anchors)
    sit_mask = np.asarray(anchors["sit_mask"], dtype=bool)
    if sit_mask.shape != (len(root),):
        raise ValueError("contact anchors do not match robot motion")
    back_mask = (
        np.asarray(anchors["back_contact"], dtype=bool)
        if args.backrest_geom
        else np.zeros(len(root), dtype=bool)
    )
    if back_mask.shape != (len(root),):
        raise ValueError("back-contact anchors do not match robot motion")
    sit = np.flatnonzero(sit_mask)
    if not len(sit):
        raise RuntimeError("no source-SMPL sit frames")
    samples = np.unique(np.r_[sit[::args.sample_stride], sit[-1]]).astype(int)

    combined = _combine_mjcf(args.robot_xml, args.scene_mujoco_xml)
    try:
        model = mujoco.MjModel.from_xml_path(str(combined))
        data = mujoco.MjData(model)
        if model.nq != 7 + dof.shape[1]:
            raise ValueError("robot motion and XML DOF counts differ")
        seat = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, args.seat_geom)
        if seat < 0:
            raise ValueError(f"seat geom {args.seat_geom!r} not found")
        dof_names = [str(item) for item in motion.get("dof_names", [])]
        if not dof_names:
            raise ValueError("robot motion has no dof_names")
        repair = []
        for name in names(args.repair_joints):
            if name not in dof_names:
                raise ValueError(f"missing motion joint {name}")
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"missing XML joint {name}")
            index = dof_names.index(name)
            qadr = int(model.jnt_qposadr[jid])
            if qadr != 7 + index:
                raise ValueError(f"{name}: qpos ordering mismatch")
            repair.append((name, index, int(jid), qadr))
        robot_geoms = _robot_chair_contact_geoms(model)
        support_geoms = [
            geom for geom in robot_geoms
            if _body_name(model, geom) == args.support_body
        ]
        if not support_geoms:
            raise ValueError(f"support body {args.support_body!r} has no collision geom")

        back_enabled = bool(args.backrest_geom and np.any(back_mask))
        backrest = -1
        back_support_geoms = []
        back_shift_direction_xy = np.zeros(2, dtype=np.float64)
        if back_enabled:
            backrest = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, args.backrest_geom
            )
            if backrest < 0:
                raise ValueError(f"backrest geom {args.backrest_geom!r} not found")
            # Backrest contact is a torso task, not a lower-body seat task.
            # Use the complete proxy set here while keeping the seat audit
            # deliberately restricted to pelvis/hips/knees.
            back_support_geoms = [
                geom for geom in _robot_collision_geoms(model)
                if _body_name(model, geom) == args.back_support_body
            ]
            if not back_support_geoms:
                raise ValueError(
                    f"back support body {args.back_support_body!r} has no collision geom"
                )
            reference_frame = int(np.flatnonzero(back_mask)[0])
            set_qpos(data, root[reference_frame], rot[reference_frame], dof[reference_frame])
            mujoco.mj_forward(model, data)
            back_axis = np.asarray(data.geom_xmat[backrest], dtype=np.float64).reshape(3, 3)[:, 1]
            robot_to_back = (
                np.asarray(data.geom_xpos[back_support_geoms[0]], dtype=np.float64)
                - np.asarray(data.geom_xpos[backrest], dtype=np.float64)
            )
            sign = 1.0 if float(np.dot(robot_to_back, back_axis)) >= 0.0 else -1.0
            front_direction_xy = sign * back_axis[:2]
            norm = float(np.linalg.norm(front_direction_xy))
            if norm <= 1e-8:
                raise RuntimeError("backrest has no horizontal front direction")
            # Positive scalar shifts the robot from the chair front towards its backrest.
            back_shift_direction_xy = -front_direction_xy / norm

        variable_count = 1 + int(back_enabled) + len(repair)
        joint_offset = 1 + int(back_enabled)

        def configure(frame, x):
            set_qpos(data, root[frame], rot[frame], dof[frame])
            data.qpos[2] += x[0]
            if back_enabled:
                data.qpos[:2] += x[1] * back_shift_direction_xy
            for delta, (_, _, _, qadr) in zip(x[joint_offset:], repair):
                data.qpos[qadr] += delta
            mujoco.mj_forward(model, data)

        def distance(geom):
            return float(mujoco.mj_geomDistance(model, data, geom, int(seat), 2.0, None))

        def back_distance():
            return min(
                float(mujoco.mj_geomDistance(model, data, geom, int(backrest), 2.0, None))
                for geom in back_support_geoms
            )

        solved = []
        details = []
        support_proximity = []
        previous = np.zeros(variable_count)
        for frame in samples:
            configure(frame, np.zeros(variable_count))
            zero_support_distance = min(distance(geom) for geom in support_geoms)
            back_active = bool(back_enabled and back_mask[frame])
            active = list(support_geoms)
            bounds = [(args.min_root_dz_m, args.max_root_dz_m)]
            if back_enabled:
                bounds.append((0.0, args.max_back_shift_m))
            for _, index, jid, _ in repair:
                bounds.append((
                    float(model.jnt_range[jid, 0] - dof[frame, index]),
                    float(model.jnt_range[jid, 1] - dof[frame, index]),
                ))

            def distances(x):
                configure(frame, x)
                values = [distance(geom) for geom in active]
                if back_active:
                    values.append(back_distance())
                return np.asarray(values)

            def support_distance(x):
                configure(frame, x)
                return min(distance(geom) for geom in support_geoms)

            def objective(x):
                back_penalty = 0.0
                if back_active:
                    configure(frame, x)
                    back_penalty = args.back_objective_weight * (
                        back_distance() - args.back_contact_target_m
                    ) ** 2
                return float(
                    # The source root is the image-coordinate boundary.  Never
                    # bias it downward: only move it as much as semantic support
                    # geometry actually requires.
                    0.08 * x[0] ** 2
                    + 0.03 * np.dot(x[joint_offset:], x[joint_offset:])
                    + 0.02 * (x[1] ** 2 if back_enabled else 0.0)
                    + 0.0 * np.dot(x - previous, x - previous)
                    + 20.0 * (support_distance(x) - args.support_target_m) ** 2
                    + back_penalty
                )

            lower, upper = np.asarray(bounds).T

            def seed(root_dz, hip_delta, back_shift=0.0):
                value = np.zeros(variable_count, dtype=np.float64)
                value[0] = root_dz
                if back_enabled:
                    value[1] = back_shift
                if repair:
                    value[joint_offset] = hip_delta
                return np.clip(value, lower, upper)

            seeds = [
                np.clip(previous, lower, upper),
                seed(0.0, 0.0),
                seed(-0.05, -0.6, 0.10),
                seed(0.04, -0.8, 0.14),
            ]
            candidates = []
            for seed in seeds:
                result = minimize(
                    objective,
                    seed,
                    method="SLSQP",
                    bounds=bounds,
                    constraints={
                        "type": "ineq",
                        "fun": lambda x: distances(x) - args.clearance_m,
                    },
                    options={"maxiter": args.maxiter, "ftol": 1e-5, "disp": False},
                )
                x = np.asarray(result.x, dtype=np.float64)
                minimum = float(np.min(distances(x)))
                if minimum >= args.clearance_m - 1e-4:
                    candidates.append((float(objective(x)), result, x, minimum))
            if not candidates:
                minimum = float(np.min(distances(seeds[0])))
                raise RuntimeError(
                    f"frame {frame}: no valid seat projection (minimum {minimum:.4f} m)"
                )
            _, result, x, minimum = min(candidates, key=lambda item: item[0])
            if minimum < args.clearance_m - 1e-4:
                raise RuntimeError(
                    f"frame {frame}: no valid seat projection (minimum {minimum:.4f} m)"
                )
            solved.append(x)
            details.append({
                "frame": int(frame),
                "unmodified_support_distance_m": float(zero_support_distance),
                "support_proximity_active": bool(
                    zero_support_distance <= args.support_proximity_m
                ),
                "optimizer_success": bool(result.success),
                "message": str(result.message),
                "minimum_distance_m": minimum,
                "support_distance_m": float(support_distance(x)),
                "root_dz_m": float(x[0]),
                "back_contact_active": back_active,
                "back_shift_m": float(x[1]) if back_enabled else 0.0,
                "back_distance_m": float(back_distance()) if back_active else None,
                "joint_delta_rad": {
                    name: float(delta)
                    for delta, (name, _, _, _) in zip(x[joint_offset:], repair)
                },
            })
            support_proximity.append(
                bool(zero_support_distance <= args.support_proximity_m)
            )
            previous = x

        solved = np.asarray(solved)
        frame_axis = np.arange(len(root), dtype=float)
        root_out, dof_out = root.copy(), dof.copy()
        root_z_signal = np.zeros(len(root), dtype=np.float64)
        raw_root_z_signal = np.interp(
            frame_axis, samples, solved[:, 0], left=0.0, right=0.0
        )
        raw_root_z_signal[~sit_mask] = 0.0
        # A per-frame constrained optimum is noisy at contact onset.  The
        # physical event is instead a persistent approach of a semantically
        # selected support body.  For each sit episode, use a robust constant
        # height correction and ease it in before contact.  This is independent
        # of clip frame numbers, chair pose, and camera coordinates.
        proximity_frames = np.asarray(samples, dtype=int)[np.asarray(support_proximity)]
        components = []
        if len(proximity_frames):
            split_at = np.where(np.diff(proximity_frames) > args.sample_stride)[0] + 1
            components = [part for part in np.split(proximity_frames, split_at) if len(part)]
        persistent = [
            component
            for component in components
            if component[-1] - component[0] + args.sample_stride
            >= args.minimum_proximity_frames
        ]
        support_events = []
        for component in persistent:
            onset = int(component[0])
            settled = float(np.median(raw_root_z_signal[component]))
            ramp_start = max(0, onset - args.root_z_ramp_frames)
            ramp_indices = np.arange(ramp_start, onset + 1)
            if len(ramp_indices) > 1:
                u = (ramp_indices - ramp_start) / float(onset - ramp_start)
                root_z_signal[ramp_indices] = settled * (3.0 * u**2 - 2.0 * u**3)
            else:
                root_z_signal[ramp_indices] = settled
            end = onset
            while end + 1 < len(sit_mask) and sit_mask[end + 1]:
                end += 1
            root_z_signal[onset:end + 1] = settled
            support_events.append({
                "onset_frame": onset,
                "end_frame": end,
                "settled_root_dz_m": settled,
            })
        root_out[:, 2] += root_z_signal
        back_signal = np.zeros(len(root), dtype=np.float64)
        if back_enabled:
            raw_back_signal = np.interp(
                frame_axis, samples, solved[:, 1], left=0.0, right=0.0
            )
            active_frames = np.flatnonzero(back_mask)
            split_at = np.where(np.diff(active_frames) > 1)[0] + 1
            for component in np.split(active_frames, split_at):
                first, last = int(component[0]), int(component[-1])
                back_signal[first:last + 1] = raw_back_signal[first:last + 1]
                for offset in range(1, args.back_ramp_frames + 1):
                    fraction = (args.back_ramp_frames + 1 - offset) / (args.back_ramp_frames + 1)
                    before, after = first - offset, last + offset
                    if before >= 0 and sit_mask[before]:
                        back_signal[before] = max(back_signal[before], raw_back_signal[first] * fraction)
                    if after < len(root) and sit_mask[after]:
                        back_signal[after] = max(back_signal[after], raw_back_signal[last] * fraction)
            back_signal[~sit_mask] = 0.0
        for variable in range(solved.shape[1]):
            signal = np.interp(frame_axis, samples, solved[:, variable], left=0.0, right=0.0)
            signal[~sit_mask] = 0.0
            if variable == 0:
                continue
            elif back_enabled and variable == 1:
                root_out[:, :2] += back_signal[:, None] * back_shift_direction_xy[None, :]
            else:
                _, index, jid, _ = repair[variable - joint_offset]
                dof_out[:, index] = np.clip(
                    dof_out[:, index] + signal,
                    model.jnt_range[jid, 0],
                    model.jnt_range[jid, 1],
                )

        dense = np.full(len(root), np.nan)
        dense_back = np.full(len(root), np.nan)
        for frame in sit:
            set_qpos(data, root_out[frame], rot[frame], dof_out[frame])
            mujoco.mj_forward(model, data)
            dense[frame] = min(distance(geom) for geom in support_geoms)
            if back_enabled:
                dense_back[frame] = back_distance()
        values = dense[sit]
        back_values = dense_back[back_mask] if back_enabled else np.asarray([], dtype=np.float64)
        all_back_values = dense_back[sit] if back_enabled else np.asarray([], dtype=np.float64)
        accepted = bool(
            np.all(values >= args.clearance_m - 1e-4)
            and (not len(all_back_values) or np.all(all_back_values >= args.clearance_m - 1e-4))
        )
        report = {
            "schema_version": 1,
            "purpose": "gmr_constrained_seat_contact_projection",
            "status": "accepted" if accepted else "rejected_after_dense_audit",
            "source_motion": str(args.robot_motion),
            "support_body": args.support_body,
            "support_event": {
                "proximity_m": args.support_proximity_m,
                "minimum_persistent_frames": args.minimum_proximity_frames,
                "root_z_ramp_frames": args.root_z_ramp_frames,
                "events": support_events,
                "root_xy_max_change_m": float(
                    np.max(np.abs(root_out[:, :2] - root[:, :2]))
                ),
            },
            "backrest_enabled": back_enabled,
            "backrest_geom": args.backrest_geom if back_enabled else None,
            "back_support_body": args.back_support_body if back_enabled else None,
            "back_contact_target_m": args.back_contact_target_m if back_enabled else None,
            "back_ramp_frames": args.back_ramp_frames if back_enabled else None,
            "back_shift_direction_xy": back_shift_direction_xy.tolist() if back_enabled else None,
            "back_contact_frames": int(back_mask.sum()),
            "sit_frame_range": [int(sit[0]), int(sit[-1])],
            "sit_frames": int(len(sit)),
            "sample_stride": args.sample_stride,
            "clearance_m": args.clearance_m,
            "support_target_m": args.support_target_m,
            "repair_joints": [item[0] for item in repair],
            "samples": details,
            "dense_audit": {
                "minimum_distance_m": float(np.min(values)),
                "median_distance_m": float(np.median(values)),
                "penetration_frame_ratio": float(np.mean(values < -1e-5)),
                "backrest_minimum_distance_m": float(np.min(back_values)) if len(back_values) else None,
                "backrest_median_distance_m": float(np.median(back_values)) if len(back_values) else None,
                "backrest_penetration_frame_ratio": float(np.mean(back_values < -1e-5)) if len(back_values) else None,
                "backrest_minimum_distance_all_sit_m": float(np.min(all_back_values)) if len(all_back_values) else None,
            },
            "note": (
                "Robot-side feasibility candidate; original GMR remains the "
                "GVHMR-motion-fidelity reference."
            ),
        }
        if not accepted:
            raise RuntimeError(json.dumps(report, ensure_ascii=False))
        output = dict(motion)
        output["root_pos"] = root_out.astype(np.asarray(motion["root_pos"]).dtype, copy=False)
        output["dof_pos"] = dof_out.astype(np.asarray(motion["dof_pos"]).dtype, copy=False)
        output["scene_seat_contact_projection"] = report
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("wb") as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        combined.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

