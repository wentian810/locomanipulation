#!/usr/bin/env python3
"""Synthesize and audit a collision-aware G1 seated *target* with Mink.

This is deliberately not a simulator and it does not modify a GMR motion.
It produces one terminal pose candidate from a compiled task that already
contains the immutable VideoMimic chair and revision-matched Unitree G1
collision geometry.  A candidate is useful only if a separate controller can
later reach it using bounded joint torque and ``mj_step``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mink
import mujoco
import numpy as np


CHAIR_PREFIXES = ("seat_", "leg_", "backrest_")
SUPPORT_BODIES = {
    "pelvis",
    "left_hip_pitch_link", "right_hip_pitch_link",
    "left_hip_roll_link", "right_hip_roll_link",
    "left_hip_yaw_link", "right_hip_yaw_link",
}
HIP_SUPPORT_GEOMS = ("gmr_official_collision_9", "gmr_official_collision_24")
FOOT_SITES = ("tracking[ltoe]", "tracking[lheel]", "tracking[rtoe]", "tracking[rheel]")


def _name(model: mujoco.MjModel, object_type: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, object_type, index) or ""


def _robot_and_chair_geoms(model: mujoco.MjModel) -> tuple[list[str], list[str], list[str], list[str]]:
    chair, support, other = [], [], []
    for geom_id in range(model.ngeom):
        geom_name = _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name.startswith(CHAIR_PREFIXES):
            chair.append(geom_name)
        if not geom_name.startswith("gmr_official_collision_"):
            continue
        body_name = _name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id]))
        (support if body_name in SUPPORT_BODIES else other).append(geom_name)
    if not chair or not support or not other:
        raise ValueError("task must contain official G1 collision geoms and semantic chair geoms")
    return chair, support, other, [name for name in chair if name != "seat_support_geom"]


def _contact_groups(
    model: mujoco.MjModel,
    configuration: mink.Configuration,
    support: set[str],
    other: set[str],
    chair: set[str],
) -> dict[str, list[dict[str, object]]]:
    mujoco.mj_collision(model, configuration.data)
    groups: dict[str, list[dict[str, object]]] = {
        "support_seat": [], "support_nonseat": [], "other_chair": [],
    }
    for contact_id in range(configuration.data.ncon):
        contact = configuration.data.contact[contact_id]
        first = _name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
        second = _name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
        robot = first if first.startswith("gmr_official_collision_") else second
        scene = second if robot == first else first
        if scene not in chair:
            continue
        item = {"distance_m": float(contact.dist), "robot_geom": robot, "chair_geom": scene}
        if robot in support and scene == "seat_support_geom":
            groups["support_seat"].append(item)
        elif robot in support:
            groups["support_nonseat"].append(item)
        elif robot in other:
            groups["other_chair"].append(item)
    return groups


def _minimum_distance(entries: list[dict[str, object]]) -> float | None:
    return min((float(entry["distance_m"]) for entry in entries), default=None)


def _translated_target(configuration: mink.Configuration, name: str, frame_type: str, shift: np.ndarray) -> mink.SE3:
    raw = configuration.get_transform_frame_to_world(name, frame_type)
    return mink.SE3.from_rotation_and_translation(raw.rotation(), raw.translation() + shift)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-xml", required=True, type=Path)
    parser.add_argument("--reference-key", type=int, default=220)
    parser.add_argument("--output-pose", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--solver", default="proxqp")
    parser.add_argument("--target-root-shift", nargs=3, type=float, default=(0.0, 0.18, 0.08))
    parser.add_argument("--seed-root-shift", nargs=3, type=float, default=(0.0, 0.18, 0.12))
    parser.add_argument("--clearance-m", type=float, default=0.003)
    parser.add_argument("--max-support-penetration-m", type=float, default=0.003)
    args = parser.parse_args()
    if args.iterations <= 0 or args.clearance_m <= 0 or args.max_support_penetration_m <= 0:
        raise ValueError("iterations and distance tolerances must be positive")

    model = mujoco.MjModel.from_xml_path(str(args.task_xml))
    if not 0 <= args.reference_key < model.nkey:
        raise ValueError("reference key is outside the task keyframes")
    reference = mink.Configuration(model)
    key_name = _name(model, mujoco.mjtObj.mjOBJ_KEY, args.reference_key)
    reference.update_from_keyframe(key_name)
    raw_q = reference.q.copy()
    configuration = mink.Configuration(model, raw_q.copy())
    target_shift = np.asarray(args.target_root_shift, dtype=np.float64)
    seed_shift = np.asarray(args.seed_root_shift, dtype=np.float64)
    configuration.q[:3] += seed_shift
    configuration.update(configuration.q)

    chair, support, other, chair_nonseat = _robot_and_chair_geoms(model)
    for geom_name in HIP_SUPPORT_GEOMS:
        if geom_name not in support:
            raise ValueError(f"task lacks expected official hip support geometry {geom_name!r}")

    pelvis_task = mink.FrameTask("pelvis", "body", position_cost=100.0, orientation_cost=100.0)
    pelvis_task.set_target(_translated_target(reference, "pelvis", "body", target_shift))
    torso_task = mink.FrameTask("torso_link", "body", position_cost=5.0, orientation_cost=20.0)
    torso_task.set_target(_translated_target(reference, "torso_link", "body", target_shift))
    tasks: list[mink.Task] = [pelvis_task, torso_task]
    # These two collision geometries are the only permitted seat-support
    # region.  Keeping their desired world poses tied to the reference avoids
    # a solver result that escapes the chair by folding the hips elsewhere.
    for geom_name in HIP_SUPPORT_GEOMS:
        task = mink.FrameTask(geom_name, "geom", position_cost=20.0, orientation_cost=0.2)
        task.set_target(_translated_target(reference, geom_name, "geom", target_shift))
        tasks.append(task)
    for site_name in FOOT_SITES:
        task = mink.FrameTask(site_name, "site", position_cost=8.0, orientation_cost=0.0)
        task.set_target_from_configuration(reference)
        tasks.append(task)
    posture_task = mink.PostureTask(model, cost=5e-3)
    posture_task.set_target(raw_q)
    tasks.append(posture_task)

    limits: list[mink.Limit] = [
        mink.ConfigurationLimit(model),
        # No body other than the designated support can meet any chair part.
        mink.CollisionAvoidanceLimit(
            model, [(other, chair), (support, chair_nonseat)], gain=0.85,
            minimum_distance_from_collisions=args.clearance_m,
            collision_detection_distance=0.5,
        ),
        # A tiny negative margin is necessary for the later gravity-driven
        # settle, but avoids accepting a pose embedded in the seat.
        mink.CollisionAvoidanceLimit(
            model, [(support, ["seat_support_geom"])], gain=0.85,
            minimum_distance_from_collisions=-args.max_support_penetration_m,
            collision_detection_distance=0.5,
        ),
    ]
    history = []
    for iteration in range(args.iterations):
        velocity = mink.solve_ik(configuration, tasks, 0.01, args.solver, limits=limits)
        configuration.integrate_inplace(velocity, 0.01)
        if (iteration + 1) % 25 == 0 or iteration + 1 == args.iterations:
            groups = _contact_groups(model, configuration, set(support), set(other), set(chair))
            history.append({
                "iteration": iteration + 1,
                "root_xyz_delta_m": (configuration.q[:3] - raw_q[:3]).tolist(),
                "joint_l2_delta_rad": float(np.linalg.norm(configuration.q[7:] - raw_q[7:])),
                "min_distances_m": {name: _minimum_distance(entries) for name, entries in groups.items()},
            })

    groups = _contact_groups(model, configuration, set(support), set(other), set(chair))
    support_min = _minimum_distance(groups["support_seat"])
    other_min = _minimum_distance(groups["other_chair"])
    nonseat_min = _minimum_distance(groups["support_nonseat"])
    accepted = bool(
        len(groups["support_seat"]) >= 2
        and support_min is not None
        and -args.max_support_penetration_m - 1e-6 <= support_min <= 0.0
        and (other_min is None or other_min >= -args.clearance_m - 1e-6)
        and (nonseat_min is None or nonseat_min >= -args.clearance_m - 1e-6)
    )
    args.output_pose.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_pose, qpos=configuration.q, reference_qpos=raw_q)
    report = {
        "schema_version": 1,
        "purpose": "offline_collision_aware_g1_seated_target",
        "status": "accepted_static_seated_target" if accepted else "rejected_static_seated_target",
        "physical_contract": {
            "offline_operation": "Mink_kinematic_target_synthesis_only",
            "scene": "immutable_VideoMimic_semantic_chair",
            "robot_collision_geometry": "revision_matched_official_Unitree_MJCF",
            "future_runtime_requirement": "initialize_once_then_bounded_joint_torque_and_mj_step",
            "forbidden_runtime_mechanisms": ["root_state_reimposition", "xfrc_applied", "mocap_weld"],
        },
        "inputs": {"task_xml": str(args.task_xml), "reference_key": args.reference_key, "reference_key_name": key_name},
        "settings": {
            "target_root_shift_m": target_shift.tolist(), "seed_root_shift_m": seed_shift.tolist(),
            "clearance_m": args.clearance_m, "max_support_penetration_m": args.max_support_penetration_m,
            "iterations": args.iterations, "solver": args.solver,
        },
        "result": {
            "root_xyz_delta_m": (configuration.q[:3] - raw_q[:3]).tolist(),
            "joint_l2_delta_rad": float(np.linalg.norm(configuration.q[7:] - raw_q[7:])),
            "contact_groups": groups, "history": history,
        },
        "output_pose": str(args.output_pose),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
