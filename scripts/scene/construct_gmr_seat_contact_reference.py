#!/usr/bin/env python3
"""Create a bounded, contact-ready GMR seat-reference candidate.

The input GMR motion is never overwritten.  For the seated interval only, the
script finds the smallest downward free-base displacement that creates a
shallow *actual MuJoCo* seat contact, then uses IK to keep both feet at their
source world poses.  It does not apply any force, weld, mocap attachment, or
runtime state reset; a separate torque-controlled ``mj_step`` rollout is still
required before the candidate can be called physically valid.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import mink
import mujoco
import numpy as np


SUPPORT_BODY_NAMES = frozenset({
    "pelvis",
    "left_hip_pitch_link", "right_hip_pitch_link",
    "left_hip_roll_link", "right_hip_roll_link",
    "left_hip_yaw_link", "right_hip_yaw_link",
})
FOOT_SITE_NAMES = ("tracking[ltoe]", "tracking[lheel]", "tracking[rtoe]", "tracking[rheel]")
CHAIR_PREFIXES = ("seat_support_geom", "backrest_geom", "leg_front_", "leg_back_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def qpos_from_motion(motion: dict[str, Any]) -> np.ndarray:
    root = np.asarray(motion["root_pos"], dtype=np.float64)
    quat_xyzw = np.asarray(motion["root_rot"], dtype=np.float64)
    dof = np.asarray(motion["dof_pos"], dtype=np.float64)
    if root.ndim != 2 or root.shape[1] != 3 or quat_xyzw.shape != (len(root), 4):
        raise ValueError("source robot_motion root arrays are invalid")
    if dof.ndim != 2 or len(dof) != len(root):
        raise ValueError("source robot_motion dof_pos is invalid")
    result = np.empty((len(root), 7 + dof.shape[1]), dtype=np.float64)
    result[:, :3] = root
    result[:, 3:7] = quat_xyzw[:, [3, 0, 1, 2]]
    result[:, 7:] = dof
    return result


def motion_fields(qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return qpos[:, :3].copy(), qpos[:, [4, 5, 6, 3]].copy(), qpos[:, 7:].copy()


def body_name(model: mujoco.MjModel, body_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""


def geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""


def contact_groups(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, list[dict[str, object]]]:
    """Partition generated robot-chair contacts by allowed seat support."""
    result: dict[str, list[dict[str, object]]] = {"support_seat": [], "forbidden": []}
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        first, second = int(contact.geom1), int(contact.geom2)
        first_name, second_name = geom_name(model, first), geom_name(model, second)
        if first_name.startswith("gmr_official_collision_") and second_name.startswith(CHAIR_PREFIXES):
            robot_geom, chair_geom = first, second
        elif second_name.startswith("gmr_official_collision_") and first_name.startswith(CHAIR_PREFIXES):
            robot_geom, chair_geom = second, first
        else:
            continue
        item = {
            "robot_body": body_name(model, int(model.geom_bodyid[robot_geom])),
            "robot_geom": geom_name(model, robot_geom),
            "chair_geom": geom_name(model, chair_geom),
            "distance_m": float(contact.dist),
        }
        if item["robot_body"] in SUPPORT_BODY_NAMES and item["chair_geom"] == "seat_support_geom":
            result["support_seat"].append(item)
        else:
            result["forbidden"].append(item)
    return result


def support_surface_clearance(
    model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[float, dict[str, object]]:
    """Return the closest zero-cutoff official-G1/chair signed distance.

    Generated contact pairs are not a sufficient containment test.  We query
    every official G1 collision geom against every semantic-chair geom, but
    use ``distmax=0``: MuJoCo documents that a large positive cutoff is
    approximate for general convex pairs.  The resulting value is therefore a
    robust penetration gate (negative means collision) rather than a claimed
    positive clearance.  Permitted support contact is still required
    separately through ``contact_groups``.
    """
    seat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "seat_support_geom")
    if seat_id < 0:
        raise ValueError("task is missing seat_support_geom")
    closest: dict[str, object] | None = None
    for geom_id in range(model.ngeom):
        name = geom_name(model, geom_id)
        if not name.startswith("gmr_official_collision_"):
            continue
        robot_body = body_name(model, int(model.geom_bodyid[geom_id]))
        for chair_id in range(model.ngeom):
            chair_name = geom_name(model, chair_id)
            if not chair_name.startswith(CHAIR_PREFIXES):
                continue
            distance = float(mujoco.mj_geomDistance(
                model, data, geom_id, chair_id, 0.0, None
            ))
            if closest is None or distance < float(closest["distance_m"]):
                closest = {
                    "robot_body": robot_body,
                    "robot_geom": name,
                    "chair_geom": chair_name,
                    "distance_m": distance,
                }
    if closest is None:
        raise ValueError("task has no official G1 and semantic-chair collision geometry")
    return float(closest["distance_m"]), closest


def shallow_seat_offset(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    max_lower_m: float,
    max_raise_m: float,
    scan_step_m: float,
    max_penetration_m: float,
) -> tuple[float | None, dict[str, list[dict[str, object]]], str]:
    """Find a shallow seat-contact seed, preferring one with no forbidden contact.

    A G1 is shorter than the source SMPL body.  A metric-correct scene can
    therefore require a small *upward* pelvis correction to remove raw seat
    penetration.  If the only shallow seed still touches a chair leg, retain it
    for collision-aware IK rather than rejecting before IK has a chance to move
    the limb; final acceptance remains strictly support-only.
    """
    last_groups: dict[str, list[dict[str, object]]] = {"support_seat": [], "forbidden": []}
    offsets = [0.0]
    for amount in np.arange(scan_step_m, max(max_lower_m, max_raise_m) + 0.5 * scan_step_m, scan_step_m):
        if amount <= max_raise_m + 1e-12:
            offsets.append(float(amount))
        if amount <= max_lower_m + 1e-12:
            offsets.append(float(-amount))
    permissive: tuple[float, dict[str, list[dict[str, object]]]] | None = None
    for offset in offsets:
        data.qpos[:] = qpos
        data.qpos[2] += offset
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        groups = contact_groups(model, data)
        last_groups = groups
        support = groups["support_seat"]
        forbidden = groups["forbidden"]
        minimum = min((float(item["distance_m"]) for item in support), default=None)
        support_surface_minimum, _ = support_surface_clearance(model, data)
        if (
            support and minimum is not None and minimum >= -max_penetration_m
            and support_surface_minimum >= -max_penetration_m
        ):
            if not forbidden:
                return float(offset), groups, "strict_support_only_seed"
            if permissive is None:
                permissive = (float(offset), groups)
    if permissive is not None:
        return permissive[0], permissive[1], "forbidden_contact_ik_seed"
    return None, last_groups, "no_shallow_support_seed"


def translated_target(configuration: mink.Configuration, name: str, frame_type: str, z_offset_m: float) -> mink.SE3:
    source = configuration.get_transform_frame_to_world(name, frame_type)
    translation = source.translation().copy()
    translation[2] += z_offset_m
    return mink.SE3.from_rotation_and_translation(source.rotation(), translation)


def rebuild_local_body_positions(
    model: mujoco.MjModel, qpos: np.ndarray, names: np.ndarray
) -> np.ndarray:
    ids = np.asarray([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(name)) for name in names
    ], dtype=np.int32)
    if np.any(ids < 0):
        missing = [str(names[index]) for index in np.flatnonzero(ids < 0)]
        raise ValueError(f"source link_body_list has absent bodies: {missing}")
    data = mujoco.MjData(model)
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    result = np.empty((len(qpos), len(ids), 3), dtype=np.float32)
    for frame, pose in enumerate(qpos):
        data.qpos[:] = pose
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        pelvis_rotation = data.xmat[pelvis_id].reshape(3, 3)
        result[frame] = (pelvis_rotation.T @ (data.xpos[ids] - data.xpos[pelvis_id]).T).T
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-motion", required=True, type=Path)
    parser.add_argument("--task-xml", required=True, type=Path)
    parser.add_argument("--task-build-report", required=True, type=Path)
    parser.add_argument("--output-motion", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--sit-start-frame", type=int, required=True)
    parser.add_argument("--sit-end-frame", type=int, default=None)
    parser.add_argument("--max-root-lower-m", type=float, default=0.06)
    parser.add_argument("--max-root-raise-m", type=float, default=0.10)
    parser.add_argument("--contact-scan-step-m", type=float, default=0.001)
    parser.add_argument(
        "--post-ik-extra-lower-m", type=float, default=0.02,
        help="when foot-preserving IK loses the raw seat contact, try this much additional pelvis lowering",
    )
    parser.add_argument(
        "--post-ik-extra-raise-m", type=float, default=0.008,
        help="when a metric scene makes the raw pose penetrate the seat, try this much additional pelvis raising",
    )
    parser.add_argument(
        "--post-ik-scan-step-m", type=float, default=0.005,
        help="increment for the post-IK extra-lowering search",
    )
    parser.add_argument("--max-seat-penetration-m", type=float, default=0.003)
    parser.add_argument("--ik-iterations", type=int, default=80)
    parser.add_argument("--ik-solver", default="proxqp")
    parser.add_argument("--max-foot-drift-m", type=float, default=0.015)
    parser.add_argument("--max-joint-delta-rad", type=float, default=0.25)
    parser.add_argument("--max-root-step-m", type=float, default=0.05)
    parser.add_argument(
        "--pre-sit-ramp-frames", type=int, default=6,
        help="foot-preserving IK frames used to ramp a nonzero first seated pelvis offset without a root step",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(
        args.max_root_lower_m, args.max_root_raise_m, args.contact_scan_step_m,
        args.max_seat_penetration_m, args.post_ik_scan_step_m,
    ) <= 0.0 or min(args.post_ik_extra_lower_m, args.post_ik_extra_raise_m) < 0.0:
        raise ValueError("contact distances must be positive")
    if args.ik_iterations < 1 or min(args.max_foot_drift_m, args.max_joint_delta_rad, args.max_root_step_m) <= 0.0:
        raise ValueError("IK and quality thresholds must be positive")
    if args.pre_sit_ramp_frames < 0:
        raise ValueError("pre-sit-ramp-frames must be nonnegative")
    if args.output_motion.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite an existing seat-reference candidate")

    with args.source_motion.open("rb") as handle:
        source = pickle.load(handle)
    for field in ("root_pos", "root_rot", "dof_pos", "link_body_list"):
        if field not in source:
            raise ValueError(f"source motion lacks {field!r}")
    source_qpos = qpos_from_motion(source)
    model = mujoco.MjModel.from_xml_path(str(args.task_xml))
    if model.nq != source_qpos.shape[1] or model.nkey != len(source_qpos):
        raise ValueError("task keyframes and source motion dimensions disagree")
    if not np.allclose(model.key_qpos, source_qpos, atol=1e-6, rtol=0.0):
        raise ValueError("task keyframes do not exactly represent --source-motion")
    build_report = json.loads(args.task_build_report.read_text(encoding="utf-8"))
    recorded_hash = build_report.get("inputs", {}).get("robot_motion_sha256")
    source_hash = sha256(args.source_motion)
    if recorded_hash != source_hash:
        raise ValueError("task build report robot_motion SHA does not match --source-motion")

    end = len(source_qpos) - 1 if args.sit_end_frame is None else args.sit_end_frame
    if not 0 <= args.sit_start_frame <= end < len(source_qpos):
        raise ValueError("seat reference range is outside the source motion")
    for site_name in FOOT_SITE_NAMES:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name) < 0:
            raise ValueError(f"task is missing foot tracking site {site_name!r}")

    data = mujoco.MjData(model)
    requested_offset = np.zeros(len(source_qpos), dtype=np.float64)
    contact_available = np.zeros(len(source_qpos), dtype=bool)
    scan_records: list[dict[str, object]] = []
    for frame in range(args.sit_start_frame, end + 1):
        offset, groups, seed_kind = shallow_seat_offset(
            model, data, source_qpos[frame], args.max_root_lower_m, args.max_root_raise_m,
            args.contact_scan_step_m, args.max_seat_penetration_m,
        )
        scan_records.append({"frame": frame, "offset_m": offset, "seed_kind": seed_kind, "groups": groups})
        if offset is not None:
            requested_offset[frame] = offset
            contact_available[frame] = True

    solved_qpos = source_qpos.copy()
    ramp_start = max(0, args.sit_start_frame - args.pre_sit_ramp_frames)
    previous = source_qpos[ramp_start - 1].copy() if ramp_start else source_qpos[0].copy()
    foot_drifts: list[float] = []
    solved_contacts: list[dict[str, object]] = []
    unsolved_frames: list[int] = []
    pre_sit_ramp: list[dict[str, object]] = []
    unsolved_pre_sit_frames: list[int] = []
    pelvis_task = mink.FrameTask("pelvis", "body", position_cost=100.0, orientation_cost=40.0)
    foot_tasks = [mink.FrameTask(site, "site", position_cost=30.0, orientation_cost=0.0) for site in FOOT_SITE_NAMES]
    posture_task = mink.PostureTask(model, cost=0.02)

    robot_geoms = [
        geom_name(model, geom_id) for geom_id in range(model.ngeom)
        if geom_name(model, geom_id).startswith("gmr_official_collision_")
    ]
    chair_geoms = [
        geom_name(model, geom_id) for geom_id in range(model.ngeom)
        if geom_name(model, geom_id).startswith(CHAIR_PREFIXES)
    ]
    support_geoms = [
        name for name in robot_geoms
        if body_name(model, int(model.geom_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)]))
        in SUPPORT_BODY_NAMES
    ]
    other_geoms = [name for name in robot_geoms if name not in support_geoms]
    nonseat_geoms = [name for name in chair_geoms if name != "seat_support_geom"]
    limits: list[mink.Limit] = [
        mink.ConfigurationLimit(model),
        mink.CollisionAvoidanceLimit(
            model, [(other_geoms, chair_geoms), (support_geoms, nonseat_geoms)],
            gain=0.85, minimum_distance_from_collisions=0.0,
            collision_detection_distance=0.10,
        ),
    ]

    # Smooth a nonzero first seated correction before the declared sit event.
    # These frames have no permitted seat contact; they merely retain the source
    # feet while the pelvis is guided into a continuous, collision-free approach.
    first_offset = requested_offset[args.sit_start_frame]
    for frame in range(ramp_start, args.sit_start_frame):
        fraction = (frame - ramp_start + 1) / (args.sit_start_frame - ramp_start + 1)
        target_offset = float(fraction * first_offset)
        reference = mink.Configuration(model, source_qpos[frame].copy())
        reference.update(source_qpos[frame])
        configuration = mink.Configuration(model, previous.copy())
        configuration.update(previous)
        pelvis_task.set_target(translated_target(reference, "pelvis", "body", target_offset))
        for task, site_name in zip(foot_tasks, FOOT_SITE_NAMES):
            task.set_target_from_configuration(reference)
        posture_task.set_target(source_qpos[frame])
        tasks: list[mink.Task] = [pelvis_task, *foot_tasks, posture_task]
        for _ in range(args.ik_iterations):
            velocity = mink.solve_ik(configuration, tasks, 0.01, args.ik_solver, limits=limits)
            configuration.integrate_inplace(velocity, 0.01)
        data.qpos[:] = configuration.q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        groups = contact_groups(model, data)
        support_surface_minimum, _ = support_surface_clearance(model, data)
        drift = 0.0
        for site_name in FOOT_SITE_NAMES:
            target = reference.get_transform_frame_to_world(site_name, "site").translation()
            actual = configuration.get_transform_frame_to_world(site_name, "site").translation()
            drift = max(drift, float(np.linalg.norm(actual - target)))
        valid = bool(
            not groups["support_seat"] and not groups["forbidden"]
            and support_surface_minimum >= -args.max_seat_penetration_m
            and drift <= args.max_foot_drift_m
        )
        pre_sit_ramp.append({
            "frame": frame, "target_root_z_offset_m": target_offset,
            "actual_root_z_offset_m": float(configuration.q[2] - source_qpos[frame, 2]),
            "support_surface_min_distance_m": support_surface_minimum,
            "foot_drift_m": drift, "valid": valid,
        })
        if not valid:
            unsolved_pre_sit_frames.append(frame)
        solved_qpos[frame] = configuration.q
        previous = configuration.q.copy()

    for frame in range(args.sit_start_frame, end + 1):
        offset = requested_offset[frame]
        if not contact_available[frame]:
            unsolved_frames.append(frame)
            continue
        reference = mink.Configuration(model, source_qpos[frame].copy())
        reference.update(source_qpos[frame])
        # The source-pose contact scan is only a seed.  Strictly preserving the
        # feet can bend the legs enough to lose that contact, so retry a few
        # progressively lower pelvis targets.  The chair remains fixed and the
        # chosen state must still pass the generated-MuJoCo-contact test.
        trial_offsets = [float(offset)]
        maximum_extra = max(args.post_ik_extra_lower_m, args.post_ik_extra_raise_m)
        for extra in np.arange(args.post_ik_scan_step_m, maximum_extra + 0.5 * args.post_ik_scan_step_m, args.post_ik_scan_step_m):
            if extra <= args.post_ik_extra_raise_m + 1e-12:
                candidate_offset = float(offset + extra)
                if candidate_offset <= args.max_root_raise_m + 1e-12:
                    trial_offsets.append(candidate_offset)
            if extra <= args.post_ik_extra_lower_m + 1e-12:
                candidate_offset = float(offset - extra)
                if candidate_offset >= -args.max_root_lower_m - 1e-12:
                    trial_offsets.append(candidate_offset)

        chosen: tuple[mink.Configuration, dict[str, list[dict[str, object]]], float, float, float] | None = None
        trial_summary: list[dict[str, object]] = []
        for target_offset in trial_offsets:
            configuration = mink.Configuration(model, previous.copy())
            configuration.update(previous)
            pelvis_task.set_target(translated_target(reference, "pelvis", "body", target_offset))
            for task, site_name in zip(foot_tasks, FOOT_SITE_NAMES):
                task.set_target_from_configuration(reference)
            posture_task.set_target(source_qpos[frame])
            tasks: list[mink.Task] = [pelvis_task, *foot_tasks, posture_task]
            for _ in range(args.ik_iterations):
                velocity = mink.solve_ik(configuration, tasks, 0.01, args.ik_solver, limits=limits)
                configuration.integrate_inplace(velocity, 0.01)
            data.qpos[:] = configuration.q
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            groups = contact_groups(model, data)
            support = groups["support_seat"]
            forbidden = groups["forbidden"]
            support_min = min((float(item["distance_m"]) for item in support), default=None)
            support_surface_minimum, support_surface_closest = support_surface_clearance(model, data)
            drift = 0.0
            for site_name in FOOT_SITE_NAMES:
                target = reference.get_transform_frame_to_world(site_name, "site").translation()
                actual = configuration.get_transform_frame_to_world(site_name, "site").translation()
                drift = max(drift, float(np.linalg.norm(actual - target)))
            valid = bool(
                support and not forbidden and support_min is not None
                and support_min >= -args.max_seat_penetration_m
                and support_surface_minimum >= -args.max_seat_penetration_m
                and drift <= args.max_foot_drift_m
            )
            trial_summary.append({
                "target_root_z_offset_m": target_offset,
                "support_min_distance_m": support_min,
                "support_surface_min_distance_m": support_surface_minimum,
                "support_surface_closest_pair": support_surface_closest,
                "forbidden_count": len(forbidden), "foot_drift_m": drift,
                "valid": valid,
            })
            if valid:
                chosen = (configuration, groups, target_offset, drift, support_surface_minimum)
                break

        if chosen is None:
            unsolved_frames.append(frame)
            solved_contacts.append({
                "frame": frame, "raw_contact_root_z_offset_m": float(offset),
                "trials": trial_summary, "status": "no_valid_foot_preserving_contact",
            })
            continue
        configuration, groups, target_offset, drift, support_surface_minimum = chosen
        solved_qpos[frame] = configuration.q
        previous = configuration.q.copy()
        foot_drifts.append(drift)
        support = groups["support_seat"]
        support_min = min(float(item["distance_m"]) for item in support)
        solved_contacts.append({
            "frame": frame, "raw_contact_root_z_offset_m": float(offset),
            "target_root_z_offset_m": target_offset,
            "actual_root_z_offset_m": float(configuration.q[2] - source_qpos[frame, 2]),
            "support_min_distance_m": support_min,
            "support_surface_min_distance_m": support_surface_minimum,
            "forbidden": groups["forbidden"],
            "foot_drift_m": drift, "trials": trial_summary, "status": "valid",
        })

    root, rotation, dof = motion_fields(solved_qpos)
    root_step = np.linalg.norm(np.diff(root, axis=0), axis=1)
    joint_delta = np.abs(dof - np.asarray(source["dof_pos"], dtype=np.float64))
    accepted = bool(
        not unsolved_frames
        and not unsolved_pre_sit_frames
        and (not foot_drifts or max(foot_drifts) <= args.max_foot_drift_m)
        and float(np.max(joint_delta)) <= args.max_joint_delta_rad
        and float(np.max(root_step)) <= args.max_root_step_m
    )
    result = copy.deepcopy(source)
    result["root_pos"] = root.astype(np.float32)
    result["root_rot"] = rotation.astype(np.float32)
    result["dof_pos"] = dof.astype(np.float32)
    result["local_body_pos"] = rebuild_local_body_positions(
        model, solved_qpos, np.asarray(source["link_body_list"])
    )
    result["seat_reference_mode"] = "bounded_static_chair_contact_ik_candidate_v2"
    result["seat_reference_source_motion_sha256"] = source_hash
    result["seat_reference_task_xml"] = str(args.task_xml)

    report = {
        "schema_version": 1,
        "purpose": "bounded_gmr_seat_contact_reference_candidate",
        "status": "accepted_kinematic_seat_reference_requires_mj_step" if accepted else "rejected_kinematic_seat_reference",
        "physical_contract": {
            "scene": "one_static_chair_from_the_task_XML",
            "operation": "offline_reference_kinematics_only",
            "runtime_required_for_promotion": "bounded_joint_torque_controller_plus_continuous_mj_step",
            "forbidden": ["per_frame_chair_motion", "external_pelvis_force", "mocap_weld", "runtime_root_state_reimposition"],
        },
        "inputs": {
            "source_motion": str(args.source_motion), "source_motion_sha256": source_hash,
            "task_xml": str(args.task_xml), "task_build_report": str(args.task_build_report),
        },
        "seat_interval": [args.sit_start_frame, end],
        "settings": {
            "max_root_lower_m": args.max_root_lower_m,
            "max_root_raise_m": args.max_root_raise_m,
            "contact_scan_step_m": args.contact_scan_step_m,
            "post_ik_extra_lower_m": args.post_ik_extra_lower_m,
            "post_ik_extra_raise_m": args.post_ik_extra_raise_m,
            "post_ik_scan_step_m": args.post_ik_scan_step_m,
            "pre_sit_ramp_frames": args.pre_sit_ramp_frames,
            "max_seat_penetration_m": args.max_seat_penetration_m,
            "ik_iterations": args.ik_iterations,
        },
        "contact_scan": scan_records,
        "solved_contact": solved_contacts,
        "unsolved_frames": unsolved_frames,
        "pre_sit_ramp": pre_sit_ramp,
        "unsolved_pre_sit_frames": unsolved_pre_sit_frames,
        "quality": {
            "max_root_step_m": float(np.max(root_step)),
            "max_allowed_root_step_m": args.max_root_step_m,
            "max_joint_delta_rad": float(np.max(joint_delta)),
            "max_allowed_joint_delta_rad": args.max_joint_delta_rad,
            "max_foot_drift_m": float(max(foot_drifts)) if foot_drifts else None,
            "max_allowed_foot_drift_m": args.max_foot_drift_m,
        },
        "output_motion": str(args.output_motion),
    }
    args.output_motion.parent.mkdir(parents=True, exist_ok=True)
    with args.output_motion.open("wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "quality": report["quality"], "unsolved_frames": unsolved_frames}, ensure_ascii=False))


if __name__ == "__main__":
    main()
