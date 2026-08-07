#!/usr/bin/env python3
"""Apply the V24 physical gates to a frozen HoloMotion ``mj_step`` roll-out.

The policy's own metrics are useful diagnostics but not an acceptance label.
This script independently checks the actual logged state against the exported
GMR reference, mechanical limits, root-height continuity, and every collidable
robot--chair pair with MuJoCo's zero-cutoff signed-distance query.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


CHAIR_GEOMS = frozenset({
    "seat_support_geom",
    "backrest_geom",
    "leg_front_left_geom",
    "leg_front_right_geom",
    "leg_back_left_geom",
    "leg_back_right_geom",
})
SEAT_SUPPORT_BODY_NAMES = {
    "pelvis",
    "left_hip_pitch_link", "right_hip_pitch_link",
    "left_hip_roll_link", "right_hip_roll_link",
    "left_hip_yaw_link", "right_hip_yaw_link",
}


def _read_metadata(reference: np.lib.npyio.NpzFile) -> dict[str, object]:
    raw = reference["metadata"]
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    parsed = json.loads(str(raw))
    if not isinstance(parsed, dict):
        raise ValueError("reference metadata is not a JSON object")
    return parsed


def _read_optional_json_scalar(archive: np.lib.npyio.NpzFile, key: str) -> dict[str, object] | None:
    """Read a provenance scalar written by the evaluator, if this revision has it."""
    if key not in archive:
        return None
    raw = archive[key]
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    parsed = json.loads(str(raw))
    if not isinstance(parsed, dict):
        raise ValueError(f"{key} must be a JSON object")
    return parsed


def _robot_collision_geoms(model: mujoco.MjModel) -> list[int]:
    """Return collidable robot geoms, excluding world, floor and semantic chair."""
    result: list[int] = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name == "floor" or name in CHAIR_GEOMS:
            continue
        if int(model.geom_bodyid[geom_id]) == 0:
            continue
        if model.geom_contype[geom_id] == 0 or model.geom_conaffinity[geom_id] == 0:
            continue
        result.append(geom_id)
    if not result:
        raise ValueError("scene has no collidable robot geoms")
    return result


def _qpos_from_rollout(
    model: mujoco.MjModel,
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos_reference_order: np.ndarray,
    dof_names: list[str],
) -> np.ndarray:
    frame_count = len(root_pos)
    if root_rot_xyzw.shape != (frame_count, 4):
        raise ValueError("roll-out root rotations must be [T,4]")
    if dof_pos_reference_order.shape != (frame_count, len(dof_names)):
        raise ValueError("roll-out dof positions do not match reference name order")
    qpos = np.zeros((frame_count, model.nq), dtype=np.float64)
    free_joint_ids = [
        joint_id for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
    ]
    if len(free_joint_ids) != 1:
        raise ValueError("task must contain exactly one free base")
    root_address = int(model.jnt_qposadr[free_joint_ids[0]])
    qpos[:, root_address:root_address + 3] = root_pos
    qpos[:, root_address + 3:root_address + 7] = root_rot_xyzw[:, [3, 0, 1, 2]]
    for column, name in enumerate(dof_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"task XML lacks reference joint {name!r}")
        qpos[:, int(model.jnt_qposadr[joint_id])] = dof_pos_reference_order[:, column]
    return qpos


def _chair_clearance(model: mujoco.MjModel, qpos: np.ndarray) -> dict[str, np.ndarray]:
    robot_geoms = _robot_collision_geoms(model)
    chair_geom_ids = {
        name: int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
        for name in CHAIR_GEOMS
    }
    if any(identifier < 0 for identifier in chair_geom_ids.values()):
        raise ValueError("static semantic chair is incomplete")
    data = mujoco.MjData(model)
    all_min = np.zeros(len(qpos), dtype=np.float64)
    support_min = np.zeros(len(qpos), dtype=np.float64)
    forbidden_min = np.zeros(len(qpos), dtype=np.float64)
    seat_id = chair_geom_ids["seat_support_geom"]
    for frame, state in enumerate(qpos):
        data.qpos[:] = state
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        all_distance = 0.0
        support_distance = 0.0
        forbidden_distance = 0.0
        for robot_geom in robot_geoms:
            body_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[robot_geom])
            ) or ""
            for chair_name, chair_geom in chair_geom_ids.items():
                distance = float(mujoco.mj_geomDistance(model, data, robot_geom, chair_geom, 0.0, None))
                all_distance = min(all_distance, distance)
                if chair_geom == seat_id and body_name in SEAT_SUPPORT_BODY_NAMES:
                    support_distance = min(support_distance, distance)
                else:
                    forbidden_distance = min(forbidden_distance, distance)
        all_min[frame] = all_distance
        support_min[frame] = support_distance
        forbidden_min[frame] = forbidden_distance
    return {
        "all_min_distance_m": all_min,
        "support_seat_min_distance_m": support_min,
        "forbidden_min_distance_m": forbidden_min,
    }


def _joint_limit_report(model: mujoco.MjModel, qpos: np.ndarray, tolerance: float) -> tuple[float, list[dict[str, object]]]:
    maximum_excess = 0.0
    violations: list[dict[str, object]] = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        if not model.jnt_limited[joint_id]:
            continue
        qpos_address = int(model.jnt_qposadr[joint_id])
        lower, upper = (float(value) for value in model.jnt_range[joint_id])
        values = qpos[:, qpos_address]
        excess = np.maximum.reduce((np.zeros_like(values), lower - values, values - upper))
        maximum = float(np.max(excess))
        maximum_excess = max(maximum_excess, maximum)
        if maximum > tolerance:
            frame = int(np.argmax(excess))
            violations.append({
                "joint": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id),
                "frame": frame,
                "qpos_rad": float(values[frame]),
                "range_rad": [lower, upper],
                "excess_rad": maximum,
            })
    return maximum_excess, violations


def _torque_bounds(model: mujoco.MjModel, torque: np.ndarray, dof_names: list[str]) -> dict[str, object]:
    if torque.shape != (len(torque), len(dof_names)):
        raise ValueError("roll-out torque array does not match reference DoF order")
    limits = np.empty(len(dof_names), dtype=np.float64)
    for column, name in enumerate(dof_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0 or not model.jnt_actfrclimited[joint_id]:
            raise ValueError(f"task has no official actuator-force range for {name!r}")
        lower, upper = model.jnt_actfrcrange[joint_id]
        limits[column] = max(abs(float(lower)), abs(float(upper)))
    absolute = np.abs(torque)
    excess = absolute - limits[None, :]
    maximum_excess = float(np.max(excess))
    saturation_ratio = np.max(absolute / limits[None, :], axis=1)
    return {
        "maximum_excess_nm": maximum_excess,
        "maximum_saturation_ratio": float(np.max(saturation_ratio)),
        "p95_saturation_ratio": float(np.percentile(saturation_ratio, 95.0)),
        "pass": bool(maximum_excess <= 1e-5),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-npz", required=True, type=Path)
    parser.add_argument("--reference-npz", required=True, type=Path)
    parser.add_argument("--task-xml", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--floor-only", action="store_true")
    parser.add_argument(
        "--end-frame-exclusive", type=int, default=None,
        help=(
            "audit only the initial [0,end) physical phase.  This is used for "
            "the no-chair prerequisite before an independently evidenced seat "
            "contact interval; the remaining frames must be audited in the "
            "separate static-chair run."
        ),
    )
    parser.add_argument("--seat-contact-first-frame", type=int, default=None)
    parser.add_argument("--max-root-rmse-m", type=float, default=0.10)
    parser.add_argument("--max-root-p95-position-error-m", type=float, default=0.10)
    parser.add_argument("--max-root-z-frame-jump-m", type=float, default=0.04)
    parser.add_argument("--joint-limit-tolerance-rad", type=float, default=1e-4)
    parser.add_argument("--min-seat-support-normal-force-n", type=float, default=20.0)
    parser.add_argument("--min-seat-supported-frame-ratio", type=float, default=0.60)
    parser.add_argument(
        "--min-seat-median-weight-fraction", type=float, default=0.25,
        help=(
            "require the median seat normal force after the seat cue to carry at least this "
            "fraction of robot weight.  This distinguishes a true seated load path from a "
            "single-link graze that happens to exceed the contact threshold."
        ),
    )
    parser.add_argument("--max-geometry-chair-penetration-m", type=float, default=0.003)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite an existing V24 audit")
    if args.floor_only and args.seat_contact_first_frame is not None:
        raise ValueError("--floor-only cannot request a seat-contact interval")
    if not 0.0 <= args.min_seat_median_weight_fraction <= 1.0:
        raise ValueError("min-seat-median-weight-fraction must lie in [0,1]")

    with np.load(args.reference_npz, allow_pickle=False) as reference, np.load(args.rollout_npz, allow_pickle=False) as rollout:
        metadata = _read_metadata(reference)
        dof_names = metadata.get("dof_names")
        if not isinstance(dof_names, list) or not all(isinstance(name, str) for name in dof_names):
            raise ValueError("reference metadata lacks ordered dof_names")
        reference_root = np.asarray(reference["ref_global_translation"], dtype=np.float64)[:, 0]
        actual_root = np.asarray(rollout["robot_global_translation"], dtype=np.float64)[:, 0]
        actual_root_rot = np.asarray(rollout["robot_global_rotation_quat"], dtype=np.float64)[:, 0]
        actual_dof = np.asarray(rollout["robot_dof_pos"], dtype=np.float64)
        actual_torque = np.asarray(rollout["robot_dof_torque"], dtype=np.float64)
        if len(actual_root) != len(reference_root):
            raise ValueError(f"roll-out/reference frame mismatch {len(actual_root)} != {len(reference_root)}")
        if "semantic_seat_normal_force_n" in rollout:
            seat_force = np.asarray(rollout["semantic_seat_normal_force_n"], dtype=np.float64)
        else:
            seat_force = None
        runtime_control_contract = _read_optional_json_scalar(rollout, "runtime_control_contract")
        rollout_reference_metadata = _read_optional_json_scalar(rollout, "metadata")

    end_frame = args.end_frame_exclusive
    if end_frame is None:
        end_frame = len(actual_root)
    if not 1 <= end_frame <= len(actual_root):
        raise ValueError("end-frame-exclusive must be inside the roll-out")
    reference_root = reference_root[:end_frame]
    actual_root = actual_root[:end_frame]
    actual_root_rot = actual_root_rot[:end_frame]
    actual_dof = actual_dof[:end_frame]
    actual_torque = actual_torque[:end_frame]
    if seat_force is not None:
        seat_force = seat_force[:end_frame]

    model = mujoco.MjModel.from_xml_path(str(args.task_xml))
    qpos = _qpos_from_rollout(model, actual_root, actual_root_rot, actual_dof, dof_names)
    error = actual_root - reference_root
    root_rmse = float(np.sqrt(np.mean(np.square(error))))
    root_p95 = float(np.percentile(np.linalg.norm(error, axis=1), 95.0))
    root_z_step = np.diff(actual_root[:, 2])
    max_root_z_jump = float(np.max(np.abs(root_z_step))) if len(root_z_step) else 0.0
    finite = bool(
        np.isfinite(actual_root).all() and np.isfinite(actual_dof).all()
        and np.isfinite(actual_torque).all() and np.isfinite(actual_root_rot).all()
    )
    joint_excess, joint_violations = _joint_limit_report(model, qpos, args.joint_limit_tolerance_rad)
    torque_report = _torque_bounds(model, actual_torque, dof_names)

    chair_report: dict[str, object] | None = None
    geometry_pass = True
    seat_pass = True
    if not args.floor_only:
        clearance = _chair_clearance(model, qpos)
        geometry_minimum = float(np.min(clearance["all_min_distance_m"]))
        geometry_pass = geometry_minimum >= -args.max_geometry_chair_penetration_m
        chair_report = {
            "distance_query": "mj_geomDistance_distmax_zero",
            "minimum_all_distance_m": geometry_minimum,
            "minimum_support_seat_distance_m": float(np.min(clearance["support_seat_min_distance_m"])),
            "minimum_forbidden_distance_m": float(np.min(clearance["forbidden_min_distance_m"])),
            "max_allowed_penetration_m": args.max_geometry_chair_penetration_m,
            "geometry_pass": geometry_pass,
        }
        if args.seat_contact_first_frame is not None:
            first = args.seat_contact_first_frame
            if not 0 <= first < len(qpos):
                raise ValueError("seat contact start is outside roll-out")
            if seat_force is None or len(seat_force) != len(qpos):
                raise ValueError("roll-out lacks same-frame semantic_seat_normal_force_n instrumentation")
            supported_ratio = float(np.mean(seat_force[first:] >= args.min_seat_support_normal_force_n))
            robot_weight = float(np.sum(model.body_mass) * abs(float(model.opt.gravity[2])))
            median_force = float(np.median(seat_force[first:]))
            median_weight_fraction = median_force / robot_weight if robot_weight > 0.0 else 0.0
            pre_sit_minimum = float(np.min(clearance["all_min_distance_m"][:first])) if first else 0.0
            seat_pass = bool(
                supported_ratio >= args.min_seat_supported_frame_ratio
                and median_weight_fraction >= args.min_seat_median_weight_fraction
                and pre_sit_minimum >= -args.max_geometry_chair_penetration_m
            )
            chair_report["seating"] = {
                "first_frame": first,
                "supported_frame_ratio": supported_ratio,
                "min_support_force_n": args.min_seat_support_normal_force_n,
                "median_support_force_n": median_force,
                "robot_weight_n": robot_weight,
                "median_weight_fraction": median_weight_fraction,
                "min_median_weight_fraction": args.min_seat_median_weight_fraction,
                "pre_sit_minimum_geometry_distance_m": pre_sit_minimum,
                "pass": seat_pass,
            }

    accepted = bool(
        finite
        and root_rmse <= args.max_root_rmse_m
        and root_p95 <= args.max_root_p95_position_error_m
        and max_root_z_jump <= args.max_root_z_frame_jump_m
        and not joint_violations
        and torque_report["pass"]
        and geometry_pass
        and seat_pass
    )
    report = {
        "schema_version": 1,
        "purpose": "frozen_holomotion_bounded_mj_step_v24_audit",
        "rollout": str(args.rollout_npz.resolve()),
        "reference": str(args.reference_npz.resolve()),
        "task_xml": str(args.task_xml.resolve()),
        "frame_count": int(len(qpos)),
        "audited_frame_range": [0, int(end_frame - 1)],
        "physical_contract": {
            "controller": "frozen HoloMotion ONNX policy; optional post-seat scaling applies only to internal joint torque",
            "integration": "continuous MuJoCo mj_step after one initial state",
            "forbidden": ["per_frame_qpos_reset", "xfrc_applied", "mocap_weld", "moving_scene_geometry"],
            "runtime_control": runtime_control_contract,
            "controller_reference_calibration": (
                None if rollout_reference_metadata is None
                else rollout_reference_metadata.get("controller_reference_calibration")
            ),
        },
        "finite_state_and_control": finite,
        "root_tracking": {"rmse_m": root_rmse, "p95_position_error_m": root_p95},
        "root_height": {"max_frame_jump_m": max_root_z_jump, "threshold_m": args.max_root_z_frame_jump_m},
        "joint_limits": {
            "tolerance_rad": args.joint_limit_tolerance_rad,
            "maximum_excess_rad": joint_excess,
            "violations": joint_violations,
            "pass": not joint_violations,
        },
        "actuator_torque": torque_report,
        "chair": chair_report,
        "acceptance": {
            "max_root_rmse_m": args.max_root_rmse_m,
            "max_root_p95_position_error_m": args.max_root_p95_position_error_m,
            "max_root_z_frame_jump_m": args.max_root_z_frame_jump_m,
            "physical_smoke_pass": accepted,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
