#!/usr/bin/env python3
"""Synthesize a robot-side seated target for a fixed semantic chair.

The source GMR pose is only a regularizer.  The hard geometric targets are
expressed in the chair frame: pelvis on the seat, with optional foot-floor
targets added only after chair support is feasible.  The result is an IK
candidate, *not* a final motion.
It is accepted only when its actual MuJoCo contacts have seat support through a
pelvis/hip proxy, floor support through a foot proxy, and no arm/wrist or knee
support substitution.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from evaluate_gmr_chair_contacts import (
    _body_name,
    _combine_mjcf,
    _load_motion,
    _robot_chair_contact_geoms,
    _robot_collision_geoms,
)


CHAIR_NAMES = (
    "seat_support_geom,backrest_geom,leg_front_left_geom,leg_front_right_geom,"
    "leg_back_left_geom,leg_back_right_geom"
)
SEAT_SUPPORT_BODIES = {
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
}
FOOT_BODIES = {"left_ankle_pitch_link", "right_ankle_pitch_link"}
FORBIDDEN_SEAT_BODIES = {
    "left_wrist_pitch_link", "left_wrist_roll_link", "left_wrist_yaw_link",
    "right_wrist_pitch_link", "right_wrist_roll_link", "right_wrist_yaw_link",
    "left_elbow_link", "right_elbow_link",
}
FORBIDDEN_FLOOR_BODIES = {"left_knee_link", "right_knee_link"}


def _names(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise ValueError("expected at least one comma-separated name")
    return names


def _configure_full_chair_and_floor(
    model: mujoco.MjModel,
    chair_ids: set[int],
    robot_ids: set[int],
    floor_id: int,
) -> None:
    chair_bit, robot_bit, floor_bit = 64, 8, 16
    for geom_id in chair_ids:
        model.geom_contype[geom_id] = chair_bit
        model.geom_conaffinity[geom_id] = robot_bit
    for geom_id in robot_ids:
        model.geom_contype[geom_id] = robot_bit
        model.geom_conaffinity[geom_id] = chair_bit | floor_bit
    model.geom_contype[floor_id] = floor_bit
    model.geom_conaffinity[floor_id] = robot_bit


def _contact_rows(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    chair_ids: set[int],
    robot_ids: set[int],
    floor_id: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        first, second = int(contact.geom1), int(contact.geom2)
        if first in chair_ids and second in robot_ids:
            robot_id, surface = second, "chair"
        elif second in chair_ids and first in robot_ids:
            robot_id, surface = first, "chair"
        elif first == floor_id and second in robot_ids:
            robot_id, surface = second, "floor"
        elif second == floor_id and first in robot_ids:
            robot_id, surface = first, "floor"
        else:
            continue
        rows.append({
            "surface": surface,
            "robot_geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, robot_id),
            "robot_body": _body_name(model, robot_id),
            "distance_m": float(contact.dist),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve a physical seated G1 target.")
    parser.add_argument("--robot-motion", required=True, type=Path)
    parser.add_argument("--scene-mujoco-xml", required=True, type=Path)
    parser.add_argument("--robot-xml", required=True, type=Path)
    parser.add_argument("--output-motion", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--source-frame", type=int, default=-1)
    parser.add_argument("--chair-geom-names", default=CHAIR_NAMES)
    parser.add_argument("--floor-geom", default="floor")
    parser.add_argument(
        "--pelvis-lift-from-source-m",
        type=float,
        default=0.015,
        help="lift the source G1 pelvis proxy along the certified chair normal",
    )
    parser.add_argument(
        "--foot-forward-offset-m",
        type=float,
        default=0.0,
        help="additional shift of source ankle targets toward the chair front (-seat-y)",
    )
    parser.add_argument(
        "--foot-lateral-offset-m",
        type=float,
        default=0.0,
        help="symmetric additional separation of source ankle targets along seat-x",
    )
    parser.add_argument(
        "--foot-contact-overlap-m",
        type=float,
        default=0.001,
        help="small compliant overlap of each foot collision proxy into the floor",
    )
    parser.add_argument(
        "--enforce-foot-ground-contact",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="require foot floor support only after a chair-supported target is feasible",
    )
    parser.add_argument("--max-nfev", type=int, default=500)
    parser.add_argument("--seed-count", type=int, default=10)
    args = parser.parse_args()
    if (
        args.max_nfev < 1 or args.seed_count < 2
        or not 0.0 <= args.pelvis_lift_from_source_m <= 0.10
        or not 0.0 <= args.foot_contact_overlap_m <= 0.004
    ):
        raise ValueError("invalid optimizer or contact target parameter")

    motion = _load_motion(args.robot_motion)
    root_source = np.asarray(motion["root_pos"], dtype=np.float64)
    rot_source = np.asarray(motion["root_rot"], dtype=np.float64)
    dof_source = np.asarray(motion["dof_pos"], dtype=np.float64)
    frame = args.source_frame if args.source_frame >= 0 else len(root_source) - 1
    if not 0 <= frame < len(root_source):
        raise ValueError("source-frame is outside the GMR motion")
    if dof_source.shape[0] != len(root_source):
        raise ValueError("GMR root and joint frame count mismatch")

    combined = _combine_mjcf(args.robot_xml, args.scene_mujoco_xml)
    try:
        model = mujoco.MjModel.from_xml_path(str(combined))
        data = mujoco.MjData(model)
        if model.nq != 7 + dof_source.shape[1]:
            raise ValueError("GMR joint count does not match the robot MJCF")
        chair_names = _names(args.chair_geom_names)
        chair_ids = {
            int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
            for name in chair_names
        }
        if -1 in chair_ids or "seat_support_geom" not in chair_names:
            raise ValueError("semantic chair collision geoms are incomplete")
        floor_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, args.floor_geom))
        if floor_id < 0:
            raise ValueError("floor collision geom is absent")
        robot_ids = set(_robot_chair_contact_geoms(model)) | set(_robot_collision_geoms(model))
        _configure_full_chair_and_floor(model, chair_ids, robot_ids, floor_id)

        seat_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "seat_support_geom"))
        pelvis_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"))
        knee_ids = [
            int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
            for name in ("left_knee_link", "right_knee_link")
        ]
        if min([pelvis_id, *knee_ids]) < 0:
            raise ValueError("required pelvis or knee body is absent")
        wrist_ids = [
            int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
            for name in ("left_wrist_yaw_link", "right_wrist_yaw_link")
        ]
        if min(wrist_ids) < 0:
            raise ValueError("required wrist body is absent")
        pelvis_geoms = [
            geom_id for geom_id in robot_ids if _body_name(model, geom_id) == "pelvis"
        ]
        if not pelvis_geoms:
            raise ValueError("pelvis has no physical collision proxy")
        pelvis_geom_id = pelvis_geoms[0]
        foot_geom_ids = []
        for body_name in ("left_ankle_pitch_link", "right_ankle_pitch_link"):
            geoms = [geom_id for geom_id in robot_ids if _body_name(model, geom_id) == body_name]
            if not geoms:
                raise ValueError(f"{body_name} has no physical collision proxy")
            foot_geom_ids.append(geoms[0])
        lower_body_collision_names = SEAT_SUPPORT_BODIES | {
            "left_hip_yaw_link", "right_hip_yaw_link",
            "left_knee_link", "right_knee_link",
            "left_ankle_pitch_link", "right_ankle_pitch_link",
        }
        lower_chair_pairs = [
            (robot_id, chair_id)
            for robot_id in robot_ids
            if _body_name(model, robot_id) in lower_body_collision_names
            for chair_id in chair_ids
        ]
        forbidden_chair_geom_ids = [
            geom_id for geom_id in robot_ids
            if _body_name(model, geom_id) in FORBIDDEN_SEAT_BODIES
        ]

        source_quat = np.asarray(rot_source[frame], dtype=np.float64)[[3, 0, 1, 2]]
        source_rotation = Rotation.from_quat(source_quat[[1, 2, 3, 0]])
        source_dof = dof_source[frame].copy()
        dof_names = list(motion["dof_names"])
        if len(dof_names) != len(source_dof):
            raise ValueError("GMR dof_names and dof_pos length differ")
        dof_lower = np.empty_like(source_dof)
        dof_upper = np.empty_like(source_dof)
        for index, name in enumerate(dof_names):
            joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
            if joint_id < 0 or int(model.jnt_qposadr[joint_id]) != 7 + index:
                raise ValueError(f"GMR joint {name!r} does not match the model qpos order")
            dof_lower[index], dof_upper[index] = model.jnt_range[joint_id]

        # Establish chair-local targets.  The backrest lies along +seat-y in
        # all certified chairs, so feet are deliberately placed toward -seat-y.
        data.qpos[:3] = root_source[frame]
        data.qpos[3:7] = source_quat
        data.qpos[7:] = source_dof
        mujoco.mj_forward(model, data)
        seat_pos = data.geom_xpos[seat_id].copy()
        seat_rotation = data.geom_xmat[seat_id].reshape(3, 3).copy()
        seat_size = model.geom_size[seat_id].copy()
        chair_x, chair_y, chair_z = seat_rotation[:, 0], seat_rotation[:, 1], seat_rotation[:, 2]
        floor_height = float(model.geom_pos[floor_id, 2])
        source_pelvis_local = seat_rotation.T @ (data.geom_xpos[pelvis_geom_id] - seat_pos)
        source_foot_local = np.asarray([
            seat_rotation.T @ (data.geom_xpos[geom_id] - seat_pos)
            for geom_id in foot_geom_ids
        ])
        pelvis_target = (
            seat_pos
            + source_pelvis_local[0] * chair_x
            + source_pelvis_local[1] * chair_y
            + (source_pelvis_local[2] + args.pelvis_lift_from_source_m) * chair_z
        )
        foot_targets = np.asarray([
            seat_pos
            + (source_foot_local[foot_index, 0]
               + (1.0 if foot_index == 0 else -1.0) * args.foot_lateral_offset_m) * chair_x
            + (source_foot_local[foot_index, 1] - args.foot_forward_offset_m) * chair_y
            for foot_index in range(2)
        ])
        foot_vertical_half_extent = np.asarray([
            np.sum(np.abs(data.geom_xmat[geom_id].reshape(3, 3)[2]) * model.geom_size[geom_id])
            for geom_id in foot_geom_ids
        ])
        foot_targets[:, 2] = (
            floor_height + foot_vertical_half_extent - args.foot_contact_overlap_m
        )
        wrist_targets = np.asarray([
            pelvis_target + (seat_size[0] + 0.18) * chair_x + 0.20 * chair_z,
            pelvis_target - (seat_size[0] + 0.18) * chair_x + 0.20 * chair_z,
        ])

        # x = root position (3), source-root rotation delta (3), direct GMR dofs.
        def configure(x: np.ndarray) -> None:
            data.qpos[:3] = x[:3]
            delta_rotation = Rotation.from_rotvec(x[3:6])
            rotation = delta_rotation * source_rotation
            quat_xyzw = rotation.as_quat()
            data.qpos[3:7] = quat_xyzw[[3, 0, 1, 2]]
            data.qpos[7:] = x[6:]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)

        source_x = np.r_[root_source[frame], np.zeros(3), source_dof]
        arm_indices = np.asarray([
            index for index, name in enumerate(dof_names)
            if any(token in name for token in ("shoulder", "elbow", "wrist"))
        ], dtype=int)
        arm_reference_dof = dof_source[0, arm_indices]
        lower = np.r_[seat_pos - np.array([1.0, 1.0, 0.1]), [-0.9, -0.9, -0.9], dof_lower]
        upper = np.r_[seat_pos + np.array([1.0, 1.0, 1.4]), [0.9, 0.9, 0.9], dof_upper]
        root_seed = pelvis_target - (data.geom_xpos[pelvis_geom_id] - root_source[frame])
        seeds = [source_x.copy()]
        seated_seed = source_x.copy()
        seated_seed[:3] = root_seed
        seeds.append(np.clip(seated_seed, lower, upper))
        # The final GVHMR-derived pose can contain crossed legs.  Seated IK is
        # non-convex, so seed from several earlier observed leg configurations
        # and a small deterministic lower-body spread rather than accepting a
        # local minimum that leaves a hip inside the seat.
        source_frame_seeds = np.linspace(
            max(0, frame - 120), frame, num=max(2, args.seed_count // 2), dtype=int
        )
        for seed_frame in np.unique(source_frame_seeds):
            data.qpos[:3] = root_source[seed_frame]
            data.qpos[3:7] = rot_source[seed_frame][[3, 0, 1, 2]]
            data.qpos[7:] = dof_source[seed_frame]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            frame_rotation = Rotation.from_quat(rot_source[seed_frame])
            delta_rotation = (frame_rotation * source_rotation.inv()).as_rotvec()
            seed = np.r_[
                root_source[seed_frame]
                + (pelvis_target - data.geom_xpos[pelvis_geom_id]),
                delta_rotation,
                dof_source[seed_frame],
            ]
            seeds.append(np.clip(seed, lower, upper))
        leg_indices = [
            index for index, name in enumerate(dof_names)
            if any(token in name for token in ("hip", "knee", "ankle"))
        ]
        generator = np.random.default_rng(20260803)
        for _ in range(max(0, args.seed_count - len(seeds))):
            seed = seated_seed.copy()
            seed[6 + np.asarray(leg_indices, dtype=int)] = np.clip(
                seed[6 + np.asarray(leg_indices, dtype=int)]
                + generator.normal(0.0, 0.55, size=len(leg_indices)),
                lower[6 + np.asarray(leg_indices, dtype=int)],
                upper[6 + np.asarray(leg_indices, dtype=int)],
            )
            seeds.append(seed)

        def residual(x: np.ndarray) -> np.ndarray:
            configure(x)
            pelvis_error = data.geom_xpos[pelvis_geom_id] - pelvis_target
            foot_error = data.geom_xpos[foot_geom_ids] - foot_targets
            wrist_error = data.xpos[wrist_ids] - wrist_targets
            # Keep knees above the floor; allowing knee-floor support creates a
            # kneeling solution rather than a seated one.
            knee_below = np.maximum(
                0.0, floor_height + 0.06 - data.xpos[knee_ids, 2]
            )
            # For the lower body, mj_geomDistance agrees with the actual
            # narrow-phase contact depth.  It lets IK see an impending hip or
            # knee/seat intersection before the post-solve contact audit.
            collision_penetration = np.asarray([
                max(
                    0.0,
                    -float(mujoco.mj_geomDistance(model, data, robot_id, chair_id, 1.0, None))
                    - 0.001,
                )
                for robot_id, chair_id in lower_chair_pairs
            ])
            arm_clearance_residual = []
            for geom_id in forbidden_chair_geom_ids:
                local = seat_rotation.T @ (data.geom_xpos[geom_id] - seat_pos)
                inside_x = max(0.0, 1.0 - abs(float(local[0])) / (seat_size[0] + 0.05))
                inside_y = max(0.0, 1.0 - abs(float(local[1])) / (seat_size[1] + 0.05))
                below_top = max(0.0, seat_size[2] + 0.06 - float(local[2]))
                arm_clearance_residual.append(inside_x * inside_y * below_top)
            regularization = x - source_x
            return np.r_[
                np.asarray([100.0, 100.0, 200.0]) * pelvis_error,
                (
                    (foot_error * np.asarray([3.0, 3.0, 45.0])).reshape(-1)
                    if args.enforce_foot_ground_contact else np.asarray([])
                ),
                30.0 * knee_below,
                400.0 * collision_penetration,
                80.0 * np.asarray(arm_clearance_residual),
                25.0 * wrist_error.reshape(-1),
                0.2 * regularization[:3],
                0.3 * regularization[3:6],
                0.08 * regularization[6:],
                0.7 * (x[6 + arm_indices] - arm_reference_dof),
            ]

        seating_iterations: list[dict[str, float]] = []
        solution = seated_seed
        optimizer = None
        contacts: list[dict[str, object]] = []
        for iteration in range(3):
            candidates: list[tuple[float, np.ndarray, object]] = []
            for seed in seeds:
                result = least_squares(
                    residual, np.clip(seed, lower, upper), bounds=(lower, upper),
                    max_nfev=args.max_nfev, ftol=1e-10, xtol=1e-10, gtol=1e-10,
                )
                candidates.append((float(np.dot(result.fun, result.fun)), result.x.copy(), result))
            _, solution, optimizer = min(candidates, key=lambda item: item[0])
            configure(solution)
            contacts = _contact_rows(model, data, chair_ids, robot_ids, floor_id)
            support_penetration = max(
                [
                    -float(row["distance_m"])
                    for row in contacts
                    if row["surface"] == "chair"
                    and row["robot_body"] in SEAT_SUPPORT_BODIES
                ],
                default=0.0,
            )
            seating_iterations.append({
                "iteration": float(iteration),
                "pelvis_target_z_m": float(pelvis_target[2]),
                "chair_support_penetration_m": support_penetration,
            })
            if support_penetration <= 0.003:
                break
            # Correct the target in the semantic chair normal.  This is a
            # robot-pose correction computed from the actual contact depth,
            # never a chair-height edit.
            pelvis_target = pelvis_target + (support_penetration - 0.002) * chair_z
            wrist_targets = np.asarray([
                pelvis_target + (seat_size[0] + 0.18) * chair_x + 0.20 * chair_z,
                pelvis_target - (seat_size[0] + 0.18) * chair_x + 0.20 * chair_z,
            ])
            seeds = [solution]
        contacts = _contact_rows(model, data, chair_ids, robot_ids, floor_id)
        chair_support = [
            row for row in contacts
            if row["surface"] == "chair" and row["robot_body"] in SEAT_SUPPORT_BODIES
        ]
        foot_support = [
            row for row in contacts
            if row["surface"] == "floor" and row["robot_body"] in FOOT_BODIES
        ]
        forbidden = [
            row for row in contacts
            if (
                row["surface"] == "chair" and row["robot_body"] in FORBIDDEN_SEAT_BODIES
            ) or (
                row["surface"] == "floor" and row["robot_body"] in FORBIDDEN_FLOOR_BODIES
            )
        ]
        max_contact_penetration = max(
            [-float(row["distance_m"]) for row in contacts], default=0.0
        )
        accepted = bool(
            optimizer.success
            and any(row["robot_body"] == "pelvis" for row in chair_support)
            and (not args.enforce_foot_ground_contact or len(foot_support) > 0)
            and not forbidden
            and max_contact_penetration <= 0.004
        )
        report = {
            "schema_version": 1,
            "purpose": "gmr_physical_seated_target_ik",
            "status": "accepted" if accepted else "rejected",
            "source_motion": str(args.robot_motion),
            "source_frame": int(frame),
            "scene_mujoco_xml": str(args.scene_mujoco_xml),
            "contact_contract": {
                "chair_geoms": chair_names,
                "required_chair_support_bodies": sorted(SEAT_SUPPORT_BODIES),
                "required_floor_support_bodies": (
                    sorted(FOOT_BODIES) if args.enforce_foot_ground_contact else []
                ),
                "forbidden_chair_support_bodies": sorted(FORBIDDEN_SEAT_BODIES),
                "forbidden_floor_support_bodies": sorted(FORBIDDEN_FLOOR_BODIES),
                "maximum_contact_penetration_m": 0.004,
            },
            "chair_frame_targets": {
                "pelvis_proxy_center_world_m": pelvis_target.tolist(),
                "foot_collision_proxy_world_m": foot_targets.tolist(),
                "wrist_yaw_world_m": wrist_targets.tolist(),
            },
            "optimizer": {
                "success": bool(optimizer.success),
                "message": str(optimizer.message),
                "nfev": int(optimizer.nfev),
                "cost": float(optimizer.cost),
                "seating_iterations": seating_iterations,
            },
            "residuals": {
                "pelvis_target_error_m": float(np.linalg.norm(data.geom_xpos[pelvis_geom_id] - pelvis_target)),
                "foot_proxy_target_error_m": np.linalg.norm(data.geom_xpos[foot_geom_ids] - foot_targets, axis=1).tolist(),
                "wrist_target_error_m": np.linalg.norm(data.xpos[wrist_ids] - wrist_targets, axis=1).tolist(),
                "knee_heights_m": data.xpos[knee_ids, 2].tolist(),
            },
            "contacts": contacts,
            "maximum_contact_penetration_m": max_contact_penetration,
            "next_required_stage": "free_base_mj_step_settle_then_contact_phase_tracking",
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if not accepted:
            raise RuntimeError(json.dumps(report, ensure_ascii=False))
        output = dict(motion)
        output["physical_seated_target_qpos"] = np.r_[solution[:3], data.qpos[3:7], solution[6:]].astype(np.float64)
        output["physical_seated_target_report"] = report
        args.output_motion.parent.mkdir(parents=True, exist_ok=True)
        with args.output_motion.open("wb") as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        combined.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
