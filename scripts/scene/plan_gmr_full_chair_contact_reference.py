#!/usr/bin/env python3
"""Make a robot-side GMR reference feasible for a fixed semantic chair.

This planner never moves the reconstructed chair.  It operates only on the
robot reference that will subsequently be tracked by ``simulate_gmr_scene_contacts``.
Every candidate is evaluated in the combined MuJoCo scene against the semantic
seat, backrest, and four legs.  The resulting motion is still only a *reference*;
it is not eligible for rendering until a separate continuous ``mj_step`` replay
accepts it.
"""

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


DEFAULT_CHAIR_GEOMS = (
    "seat_support_geom,backrest_geom,leg_front_left_geom,leg_front_right_geom,"
    "leg_back_left_geom,leg_back_right_geom"
)
DEFAULT_SUPPORT_BODIES = (
    "pelvis,left_hip_pitch_link,left_hip_roll_link,right_hip_pitch_link,"
    "right_hip_roll_link"
)
DEFAULT_REPAIR_JOINTS = (
    "left_hip_pitch_joint,left_hip_roll_joint,left_hip_yaw_joint,left_knee_joint,"
    "right_hip_pitch_joint,right_hip_roll_joint,right_hip_yaw_joint,right_knee_joint"
)


def _names(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("expected at least one comma-separated name")
    return result


def _set_qpos(
    data: mujoco.MjData,
    root: np.ndarray,
    rotation_xyzw: np.ndarray,
    dof: np.ndarray,
) -> None:
    data.qpos[:3] = root
    data.qpos[3:7] = rotation_xyzw[[3, 0, 1, 2]]
    data.qpos[7:] = dof


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    if name is None:
        raise ValueError(f"unnamed geom {geom_id}")
    return str(name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan a full-semantic-chair collision-feasible GMR reference."
    )
    parser.add_argument("--robot-motion", required=True, type=Path)
    parser.add_argument("--scene-mujoco-xml", required=True, type=Path)
    parser.add_argument("--robot-xml", required=True, type=Path)
    parser.add_argument("--contact-anchors", required=True, type=Path)
    parser.add_argument("--output-motion", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--chair-geom-names", default=DEFAULT_CHAIR_GEOMS)
    parser.add_argument("--support-bodies", default=DEFAULT_SUPPORT_BODIES)
    parser.add_argument("--repair-joints", default=DEFAULT_REPAIR_JOINTS)
    parser.add_argument("--clearance-m", type=float, default=0.003)
    parser.add_argument("--support-target-m", type=float, default=0.008)
    parser.add_argument(
        "--activation-distance-m",
        type=float,
        default=0.18,
        help="only near-chair robot/scene pairs become nonlinear constraints",
    )
    parser.add_argument("--max-root-xy-shift-m", type=float, default=0.16)
    parser.add_argument("--min-root-dz-m", type=float, default=-0.08)
    parser.add_argument("--max-root-dz-m", type=float, default=0.18)
    parser.add_argument("--max-joint-delta-rad", type=float, default=0.90)
    parser.add_argument("--root-objective-weight", type=float, default=40.0)
    parser.add_argument("--joint-objective-weight", type=float, default=0.20)
    parser.add_argument("--temporal-objective-weight", type=float, default=1.0)
    parser.add_argument("--support-objective-weight", type=float, default=80.0)
    parser.add_argument("--maxiter", type=int, default=90)
    args = parser.parse_args()

    if args.clearance_m < 0.0 or args.support_target_m < args.clearance_m:
        raise ValueError("support target must be at least the requested clearance")
    if args.activation_distance_m <= args.clearance_m:
        raise ValueError("activation distance must exceed clearance")
    if args.max_root_xy_shift_m <= 0.0 or args.max_root_dz_m < args.min_root_dz_m:
        raise ValueError("invalid root-displacement bounds")
    if args.max_joint_delta_rad <= 0.0 or args.maxiter < 1:
        raise ValueError("invalid joint bound or iteration count")

    motion = _load_motion(args.robot_motion)
    root = np.asarray(motion["root_pos"], dtype=np.float64)
    rotation = np.asarray(motion["root_rot"], dtype=np.float64)
    dof = np.asarray(motion["dof_pos"], dtype=np.float64)
    if root.ndim != 2 or root.shape[1] != 3 or rotation.shape != (len(root), 4):
        raise ValueError("unexpected root trajectory shape")
    if dof.shape[0] != len(root):
        raise ValueError("root and joint trajectory length mismatch")
    anchors = np.load(args.contact_anchors)
    sit_mask = np.asarray(anchors["sit_mask"], dtype=bool)
    if sit_mask.shape != (len(root),):
        raise ValueError("contact anchors do not match robot motion")
    sit_frames = np.flatnonzero(sit_mask)
    if not len(sit_frames):
        raise ValueError("source-SMPL contact anchors contain no sit frames")

    combined_xml = _combine_mjcf(args.robot_xml, args.scene_mujoco_xml)
    try:
        model = mujoco.MjModel.from_xml_path(str(combined_xml))
        data = mujoco.MjData(model)
        if model.nq != 7 + dof.shape[1]:
            raise ValueError(
                f"robot motion has {dof.shape[1]} joints but combined model has nq={model.nq}"
            )
        dof_names = list(motion["dof_names"])
        requested_joints = _names(args.repair_joints)
        repair: list[tuple[str, int, int]] = []
        for name in requested_joints:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0 or name not in dof_names:
                raise ValueError(f"repair joint {name!r} is not shared by the motion and MJCF")
            index = dof_names.index(name)
            qpos_address = int(model.jnt_qposadr[joint_id])
            if qpos_address != 7 + index:
                raise ValueError(f"{name}: qpos ordering differs from GMR motion")
            repair.append((name, index, joint_id))

        chair_names = _names(args.chair_geom_names)
        chair_ids: list[int] = []
        for name in chair_names:
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id < 0:
                raise ValueError(f"semantic chair geom {name!r} is absent")
            chair_ids.append(int(geom_id))
        seat_id = chair_ids[chair_names.index("seat_support_geom")]
        robot_geom_ids = sorted(
            set(_robot_chair_contact_geoms(model)) | set(_robot_collision_geoms(model))
        )
        if not robot_geom_ids:
            raise ValueError("combined model has no robot collision proxies")
        support_names = set(_names(args.support_bodies))
        support_geoms = [
            geom_id for geom_id in robot_geom_ids
            if _body_name(model, geom_id) in support_names
        ]
        if not support_geoms:
            raise ValueError("support-bodies selects no robot collision proxy")
        # Match the runtime full-chair collision contract before using MuJoCo's
        # narrow phase for planning.  This deliberately disables unrelated
        # self-collision categories in this temporary model; the subsequent
        # replay rebuilds its complete chair-and-floor contact configuration.
        chair_bit, robot_bit = 64, 8
        for geom_id in chair_ids:
            model.geom_contype[geom_id] = chair_bit
            model.geom_conaffinity[geom_id] = robot_bit
        for geom_id in robot_geom_ids:
            model.geom_contype[geom_id] = robot_bit
            model.geom_conaffinity[geom_id] = chair_bit
        all_pairs = [(robot_id, chair_id) for robot_id in robot_geom_ids for chair_id in chair_ids]
        robot_geom_set = set(robot_geom_ids)
        chair_geom_set = set(chair_ids)
        support_pairs = [(robot_id, seat_id) for robot_id in support_geoms]

        def configure(frame: int, x: np.ndarray) -> None:
            planned_root = root[frame].copy()
            planned_root += x[:3]
            planned_dof = dof[frame].copy()
            for delta, (_, index, _) in zip(x[3:], repair):
                planned_dof[index] += delta
            _set_qpos(data, planned_root, rotation[frame], planned_dof)
            mujoco.mj_forward(model, data)

        def distance(pair: tuple[int, int]) -> float:
            return float(mujoco.mj_geomDistance(model, data, pair[0], pair[1], 2.0, None))

        def distances(pairs: list[tuple[int, int]]) -> np.ndarray:
            return np.asarray([distance(pair) for pair in pairs], dtype=np.float64)

        def chair_contacts() -> dict[tuple[int, int], float]:
            """Return actual MuJoCo contact depths for robot--chair pairs.

            ``mj_geomDistance`` is used only as a support-attraction
            diagnostic below.  In this G1 model it can report a spurious
            negative distance for a distant wrist box, whereas MuJoCo's
            narrow-phase contact list correctly contains no such pair.  The
            collision-feasibility constraint must therefore use the same
            narrow phase that ``mj_step`` will use.
            """
            values: dict[tuple[int, int], float] = {}
            for contact_id in range(data.ncon):
                contact = data.contact[contact_id]
                first, second = int(contact.geom1), int(contact.geom2)
                if first in robot_geom_set and second in chair_geom_set:
                    pair = (first, second)
                elif second in robot_geom_set and first in chair_geom_set:
                    pair = (second, first)
                else:
                    continue
                values[pair] = min(values.get(pair, float("inf")), float(contact.dist))
            return values

        variable_count = 3 + len(repair)
        lower = np.asarray(
            [-args.max_root_xy_shift_m, -args.max_root_xy_shift_m, args.min_root_dz_m]
            + [-args.max_joint_delta_rad] * len(repair), dtype=np.float64,
        )
        upper = np.asarray(
            [args.max_root_xy_shift_m, args.max_root_xy_shift_m, args.max_root_dz_m]
            + [args.max_joint_delta_rad] * len(repair), dtype=np.float64,
        )
        root_out, dof_out = root.copy(), dof.copy()
        correction = np.zeros((len(root), variable_count), dtype=np.float64)
        details: list[dict[str, object]] = []
        previous = np.zeros(variable_count, dtype=np.float64)

        for frame in sit_frames:
            configure(int(frame), np.zeros(variable_count))
            source_contacts = chair_contacts()
            source_support_distance = float(np.min(distances(support_pairs)))
            active_pairs = sorted(source_contacts)
            requires_projection = bool(active_pairs)
            if not requires_projection:
                details.append({
                    "frame": int(frame),
                    "planned": False,
                    "source_min_full_chair_contact_distance_m": None,
                    "source_support_distance_m": source_support_distance,
                })
                previous.fill(0.0)
                continue

            def objective(x: np.ndarray) -> float:
                configure(int(frame), x)
                support_distance = float(np.min(distances(support_pairs)))
                return float(
                    args.root_objective_weight * np.dot(x[:3], x[:3])
                    + args.joint_objective_weight * np.dot(x[3:], x[3:])
                    + args.temporal_objective_weight * np.dot(x - previous, x - previous)
                    + args.support_objective_weight
                    * (support_distance - args.support_target_m) ** 2
                )

            def inequality(x: np.ndarray) -> np.ndarray:
                configure(int(frame), x)
                contacts = chair_contacts()
                # Once a colliding pair has separated, give SLSQP a positive
                # margin.  New collision pairs are caught by the dense audit
                # and, ultimately, by the following continuous mj_step pass.
                return np.asarray([
                    contacts.get(pair, args.clearance_m + 0.01) - args.clearance_m
                    for pair in active_pairs
                ], dtype=np.float64)

            seeds = [np.clip(previous, lower, upper), np.zeros(variable_count)]
            # A raised seed resolves the common source-GMR case where the pelvis
            # proxy begins inside the semantic seat.  It is a robot correction,
            # never a scene translation.
            raised = np.zeros(variable_count)
            source_penetration = min(source_contacts.values())
            raised[2] = min(args.max_root_dz_m, max(0.02, -source_penetration + args.clearance_m))
            seeds.append(np.clip(raised, lower, upper))
            candidates: list[tuple[float, np.ndarray, float, float, bool, str]] = []
            for seed in seeds:
                result = minimize(
                    objective,
                    seed,
                    method="SLSQP",
                    bounds=list(zip(lower, upper)),
                    constraints={"type": "ineq", "fun": inequality},
                    options={"maxiter": args.maxiter, "ftol": 1e-7, "disp": False},
                )
                minimum = float(np.min(inequality(result.x)))
                configure(int(frame), result.x)
                support_distance = float(np.min(distances(support_pairs)))
                if minimum >= -1e-5:
                    candidates.append((
                        float(objective(result.x)), result.x.copy(), minimum,
                        support_distance, bool(result.success), str(result.message),
                    ))
            if not candidates:
                details.append({
                    "frame": int(frame),
                    "planned": True,
                    "accepted": False,
                    "source_min_full_chair_contact_distance_m": float(source_penetration),
                    "source_support_distance_m": source_support_distance,
                    "active_pair_count": len(active_pairs),
                })
                report = {
                    "schema_version": 1,
                    "purpose": "full_semantic_chair_robot_reference_plan",
                    "status": "rejected",
                    "reason": f"frame {frame} has no collision-feasible robot correction within declared bounds",
                    "source_motion": str(args.robot_motion),
                    "scene_mujoco_xml": str(args.scene_mujoco_xml),
                    "chair_geoms": chair_names,
                    "clearance_m": args.clearance_m,
                    "details": details,
                }
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                raise RuntimeError(report["reason"])
            _, solution, minimum, support_distance, success, message = min(
                candidates, key=lambda candidate: candidate[0]
            )
            correction[frame] = solution
            root_out[frame] += solution[:3]
            for delta, (_, index, joint_id) in zip(solution[3:], repair):
                dof_out[frame, index] = np.clip(
                    dof_out[frame, index] + delta,
                    model.jnt_range[joint_id, 0], model.jnt_range[joint_id, 1],
                )
            previous = solution
            details.append({
                "frame": int(frame),
                "planned": True,
                "accepted": True,
                "optimizer_success": success,
                "optimizer_message": message,
                "source_min_full_chair_contact_distance_m": float(source_penetration),
                "source_support_distance_m": source_support_distance,
                "active_pair_count": len(active_pairs),
                "minimum_active_constraint_margin_m": minimum,
                "planned_support_distance_m": support_distance,
                "root_delta_m": solution[:3].tolist(),
                "joint_delta_rad": {
                    name: float(delta) for delta, (name, _, _) in zip(solution[3:], repair)
                },
            })

        dense_minimum = np.full(len(sit_frames), np.nan, dtype=np.float64)
        dense_support = np.full(len(sit_frames), np.nan, dtype=np.float64)
        for output_index, frame in enumerate(sit_frames):
            _set_qpos(data, root_out[frame], rotation[frame], dof_out[frame])
            mujoco.mj_forward(model, data)
            contacts = chair_contacts()
            if contacts:
                dense_minimum[output_index] = min(contacts.values())
            dense_support[output_index] = float(np.min(distances(support_pairs)))
        contact_values = dense_minimum[np.isfinite(dense_minimum)]
        accepted = bool(
            (not len(contact_values) or np.all(contact_values >= -1e-5))
        )
        report = {
            "schema_version": 1,
            "purpose": "full_semantic_chair_robot_reference_plan",
            "status": "accepted" if accepted else "rejected_after_dense_audit",
            "source_motion": str(args.robot_motion),
            "scene_mujoco_xml": str(args.scene_mujoco_xml),
            "chair_geoms": chair_names,
            "support_bodies": sorted(support_names),
            "repair_joints": [name for name, _, _ in repair],
            "clearance_m": args.clearance_m,
            "support_target_m": args.support_target_m,
            "sit_frame_range": [int(sit_frames[0]), int(sit_frames[-1])],
            "sit_frames": int(len(sit_frames)),
            "dense_audit": {
                "actual_contact_frame_count": int(len(contact_values)),
                "minimum_full_chair_contact_distance_m": (
                    float(np.min(contact_values)) if len(contact_values) else None
                ),
                "median_full_chair_contact_distance_m": (
                    float(np.median(contact_values)) if len(contact_values) else None
                ),
                "minimum_support_distance_m": float(np.min(dense_support)),
                "median_support_distance_m": float(np.median(dense_support)),
            },
            "max_robot_root_delta_m": float(np.max(np.linalg.norm(correction[:, :3], axis=1))),
            "max_robot_joint_delta_rad": float(np.max(np.abs(correction[:, 3:]))),
            "details": details,
            "next_required_stage": "continuous_mj_step_full_chair_replay",
        }
        args.output_motion.parent.mkdir(parents=True, exist_ok=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if not accepted:
            raise RuntimeError("dense full-chair audit rejected the planned reference")
        output = dict(motion)
        output["root_pos"] = root_out.astype(np.asarray(motion["root_pos"]).dtype, copy=False)
        output["dof_pos"] = dof_out.astype(np.asarray(motion["dof_pos"]).dtype, copy=False)
        output["full_chair_contact_reference_plan"] = report
        with args.output_motion.open("wb") as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        combined_xml.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
