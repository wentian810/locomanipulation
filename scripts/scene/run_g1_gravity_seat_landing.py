#!/usr/bin/env python3
"""Run one gravity-only G1-to-chair landing from a single initial keyframe.

The script is a physical-contact baseline, not a GMR trajectory renderer.  It
sets ``qpos``/``qvel`` exactly once at time zero, applies only bounded joint
torques thereafter, and advances with ``mj_step``.  The fixed semantic chair
is never transformed, kinematically driven, or externally loaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


SUPPORT_BODIES = {
    "pelvis",
    "left_hip_pitch_link", "right_hip_pitch_link",
    "left_hip_roll_link", "right_hip_roll_link",
    "left_hip_yaw_link", "right_hip_yaw_link",
}
CHAIR_GEOMS = {
    "seat_support_geom", "backrest_geom", "leg_front_left_geom",
    "leg_front_right_geom", "leg_back_left_geom", "leg_back_right_geom",
}


def _name(model: mujoco.MjModel, object_type: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, object_type, index) or ""


def _actuator_state_addresses(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    qpos_addresses, dof_addresses = [], []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError("gravity-seat baseline requires one hinge motor per actuator")
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        dof_addresses.append(int(model.jnt_dofadr[joint_id]))
    return np.asarray(qpos_addresses), np.asarray(dof_addresses)


def _robot_weight(model: mujoco.MjModel) -> float:
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id < 0:
        raise ValueError("task lacks the G1 pelvis body")
    descendants = {pelvis_id}
    changed = True
    while changed:
        changed = False
        for body_id in range(1, model.nbody):
            if body_id not in descendants and int(model.body_parentid[body_id]) in descendants:
                descendants.add(body_id)
                changed = True
    return float(sum(model.body_mass[body_id] for body_id in descendants) * abs(model.opt.gravity[2]))


def _robot_collision_geoms(model: mujoco.MjModel) -> list[int]:
    """Find robot collision geoms by model topology, not one XML's names."""
    result: list[int] = []
    for geom_id in range(model.ngeom):
        name = _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name == "floor" or name in CHAIR_GEOMS:
            continue
        if int(model.geom_bodyid[geom_id]) == 0:
            continue
        if model.geom_contype[geom_id] == 0 or model.geom_conaffinity[geom_id] == 0:
            continue
        result.append(geom_id)
    if not result:
        raise ValueError("task has no collidable robot geoms")
    return result


def _chair_contact_metrics(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    result = {
        "support_normal_force_n": 0.0,
        "other_chair_normal_force_n": 0.0,
        "support_contact_count": 0.0,
        "other_chair_contact_count": 0.0,
        "minimum_other_chair_contact_distance_m": float("inf"),
        # Contact pairs are insufficient for an audit: a mesh/capsule deeply
        # contained in another geom need not produce a useful contact pair.
        # These two values are populated below with zero-cutoff direct geom
        # distances for every official G1 collision geom against every
        # semantic-chair geom.  A large positive cutoff is not reliable for
        # MuJoCo's general convex collider, so this audit only asks the robust
        # question relevant here: is there negative-distance penetration?
        "minimum_support_seat_geometry_distance_m": float("inf"),
        "minimum_forbidden_chair_geometry_distance_m": float("inf"),
    }
    force = np.zeros(6, dtype=np.float64)
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        first = _name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
        second = _name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
        if first in CHAIR_GEOMS:
            chair_geom, robot_id = first, int(contact.geom2)
        elif second in CHAIR_GEOMS:
            chair_geom, robot_id = second, int(contact.geom1)
        else:
            continue
        body_name = _name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[robot_id]))
        mujoco.mj_contactForce(model, data, contact_id, force)
        is_seat_support = body_name in SUPPORT_BODIES and chair_geom == "seat_support_geom"
        if is_seat_support:
            result["support_normal_force_n"] += float(force[0])
            result["support_contact_count"] += 1.0
        else:
            result["other_chair_normal_force_n"] += float(force[0])
            result["other_chair_contact_count"] += 1.0
            result["minimum_other_chair_contact_distance_m"] = min(
                result["minimum_other_chair_contact_distance_m"], float(contact.dist)
            )
    if not np.isfinite(result["minimum_other_chair_contact_distance_m"]):
        result["minimum_other_chair_contact_distance_m"] = float("nan")
    robot_collision_ids = _robot_collision_geoms(model)
    chair_collision_ids = [
        geom_id for geom_id in range(model.ngeom)
        if _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) in CHAIR_GEOMS
    ]
    for robot_id in robot_collision_ids:
        body_name = _name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[robot_id]))
        for chair_id in chair_collision_ids:
            chair_name = _name(model, mujoco.mjtObj.mjOBJ_GEOM, chair_id)
            distance = float(mujoco.mj_geomDistance(model, data, robot_id, chair_id, 0.0, None))
            is_allowed_support_pair = (
                body_name in SUPPORT_BODIES and chair_name == "seat_support_geom"
            )
            key = (
                "minimum_support_seat_geometry_distance_m"
                if is_allowed_support_pair else "minimum_forbidden_chair_geometry_distance_m"
            )
            result[key] = min(result[key], distance)
    for key in (
        "minimum_support_seat_geometry_distance_m",
        "minimum_forbidden_chair_geometry_distance_m",
    ):
        if not np.isfinite(result[key]):
            result[key] = float("nan")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-xml", required=True, type=Path)
    parser.add_argument("--reference-key", type=int, default=220)
    parser.add_argument("--initial-root-shift", nargs=3, type=float, default=(0.0, 0.18, 0.12))
    parser.add_argument("--duration-s", type=float, default=2.0)
    parser.add_argument("--kp-nm-per-rad", type=float, default=80.0)
    parser.add_argument("--kd-nms-per-rad", type=float, default=8.0)
    parser.add_argument("--max-torso-orientation-error-deg", type=float, default=15.0)
    parser.add_argument("--sample-period-s", type=float, default=0.01)
    parser.add_argument("--arrays", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if min(args.duration_s, args.kp_nm_per_rad, args.kd_nms_per_rad, args.sample_period_s,
           args.max_torso_orientation_error_deg) <= 0.0:
        raise ValueError("duration, gains, and sample period must be positive")

    model = mujoco.MjModel.from_xml_path(str(args.task_xml))
    if not 0 <= args.reference_key < model.nkey:
        raise ValueError("reference key is outside task keyframes")
    if model.nu != model.nq - 7:
        raise ValueError("task must expose exactly one torque actuator per G1 hinge")
    qpos_addresses, dof_addresses = _actuator_state_addresses(model)
    target_q = model.key_qpos[args.reference_key].copy()
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    if torso_id < 0:
        raise ValueError("task lacks the G1 torso_link body")
    target_data = mujoco.MjData(model)
    target_data.qpos[:] = target_q
    mujoco.mj_forward(model, target_data)
    target_torso_rotation = target_data.xmat[torso_id].reshape(3, 3).copy()
    data = mujoco.MjData(model)
    data.qpos[:] = target_q
    data.qpos[:3] += np.asarray(args.initial_root_shift, dtype=np.float64)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)  # the only state initialisation before stepping

    steps = int(round(args.duration_s / model.opt.timestep))
    sample_stride = max(1, int(round(args.sample_period_s / model.opt.timestep)))
    samples: list[dict[str, float]] = []
    for step in range(steps):
        error = target_q[qpos_addresses] - data.qpos[qpos_addresses]
        velocity = data.qvel[dof_addresses]
        torque = args.kp_nm_per_rad * error - args.kd_nms_per_rad * velocity
        data.ctrl[:] = np.clip(torque, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        mujoco.mj_step(model, data)
        if step % sample_stride != 0 and step + 1 != steps:
            continue
        contact = _chair_contact_metrics(model, data)
        torso_rotation = data.xmat[torso_id].reshape(3, 3)
        torso_error = np.arccos(np.clip(
            (np.trace(target_torso_rotation.T @ torso_rotation) - 1.0) / 2.0, -1.0, 1.0
        ))
        samples.append({
            "time_s": float(data.time), "root_x_m": float(data.qpos[0]),
            "root_y_m": float(data.qpos[1]), "root_z_m": float(data.qpos[2]),
            "root_speed_mps": float(np.linalg.norm(data.qvel[:3])),
            "torso_orientation_error_deg": float(np.degrees(torso_error)),
            "actuator_abs_torque_sum_nm": float(np.sum(np.abs(data.ctrl))), **contact,
        })

    if not samples:
        raise RuntimeError("landing simulation produced no samples")
    arrays = {key: np.asarray([sample[key] for sample in samples], dtype=np.float64) for key in samples[0]}
    tail = max(1, len(samples) // 4)
    mass_gravity = _robot_weight(model)
    support_tail = arrays["support_normal_force_n"][-tail:]
    other_tail = arrays["other_chair_normal_force_n"][-tail:]
    other_distance = arrays["minimum_other_chair_contact_distance_m"][-tail:]
    support_geometry_distance = arrays["minimum_support_seat_geometry_distance_m"][-tail:]
    forbidden_geometry_distance = arrays["minimum_forbidden_chair_geometry_distance_m"][-tail:]
    torso_error_tail = arrays["torso_orientation_error_deg"][-tail:]
    support_ratio = float(np.median(support_tail) / mass_gravity) if mass_gravity > 0.0 else float("nan")
    root_z = arrays["root_z_m"]
    status = "accepted_gravity_seat_landing" if (
        np.mean(arrays["support_contact_count"][-tail:] > 0.0) >= 0.8
        and 0.5 <= support_ratio <= 1.5
        # Backrest contact is physically valid; penetration beyond the
        # contact tolerance is not.  Chair legs are included in this gate.
        and (not np.isfinite(other_distance).any() or float(np.nanmin(other_distance)) >= -0.003)
        # Audit every geometry pair as well as generated contact pairs.  This
        # rejects deep containment that MuJoCo's contact list can omit.
        and (not np.isfinite(support_geometry_distance).any() or float(np.nanmin(support_geometry_distance)) >= -0.003)
        and (not np.isfinite(forbidden_geometry_distance).any() or float(np.nanmin(forbidden_geometry_distance)) >= -0.003)
        and float(np.median(torso_error_tail)) <= args.max_torso_orientation_error_deg
        and float(np.max(np.abs(np.diff(root_z)))) <= 0.04
    ) else "rejected_gravity_seat_landing"
    args.arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.arrays, **arrays)
    report = {
        "schema_version": 1,
        "purpose": "single_initialization_gravity_seat_contact_baseline",
        "status": status,
        "physical_contract": {
            "initialization": "qpos_and_qvel_written_once_before_first_mj_step",
            "runtime": "bounded_joint_PD_torque_plus_mj_step",
            "scene": "fixed_semantic_chair",
            "forbidden": ["post_initialization_qpos_write", "post_initialization_qvel_write", "xfrc_applied", "mocap_weld"],
        },
        "settings": {
            "reference_key": args.reference_key,
            "initial_root_shift_m": list(args.initial_root_shift),
            "duration_s": args.duration_s,
            "kp_nm_per_rad": args.kp_nm_per_rad,
            "kd_nms_per_rad": args.kd_nms_per_rad,
        },
        "summary": {
            "robot_weight_n": mass_gravity,
            "tail_median_support_normal_force_n": float(np.median(support_tail)),
            "tail_median_support_to_weight_ratio": support_ratio,
            "tail_max_other_chair_normal_force_n": float(np.max(other_tail)),
            "tail_minimum_other_chair_contact_distance_m": (
                float(np.nanmin(other_distance)) if np.isfinite(other_distance).any() else None
            ),
            "tail_minimum_support_seat_geometry_distance_m": (
                float(np.nanmin(support_geometry_distance))
                if np.isfinite(support_geometry_distance).any() else None
            ),
            "tail_minimum_forbidden_chair_geometry_distance_m": (
                float(np.nanmin(forbidden_geometry_distance))
                if np.isfinite(forbidden_geometry_distance).any() else None
            ),
            "tail_median_torso_orientation_error_deg": float(np.median(torso_error_tail)),
            "root_z_range_m": [float(np.min(root_z)), float(np.max(root_z))],
            "max_sampled_root_z_step_m": float(np.max(np.abs(np.diff(root_z)))) if len(root_z) > 1 else 0.0,
            "initial_qpos_wxyz": (target_q + np.r_[np.asarray(args.initial_root_shift), np.zeros(model.nq - 3)]).tolist(),
            "settled_final_qpos_wxyz": data.qpos.tolist(),
            "arrays": str(args.arrays),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
