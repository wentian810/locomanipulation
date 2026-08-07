#!/usr/bin/env python3
"""Audit an MJPC roll-out against its GMR root reference without rendering.

The script intentionally reports only state produced by the bounded-actuator
``mj_step`` roll-out.  It never writes qpos, qvel, mocap, or external forces.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np


CHAIR_GEOM_PREFIXES = ("seat_", "leg_", "backrest_")
SEAT_SUPPORT_BODY_NAMES = {
    "pelvis",
    "left_hip_pitch_link", "right_hip_pitch_link",
    "left_hip_roll_link", "right_hip_roll_link",
    "left_hip_yaw_link", "right_hip_yaw_link",
}


def _load_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.reader(line for line in handle if not line.startswith("#"))]
    if len(rows) < 2:
        raise ValueError(f"trajectory has no data rows: {path}")
    header = rows[0]
    values = np.asarray(rows[1:], dtype=np.float64)
    if values.shape[1] != len(header):
        raise ValueError("trajectory header and data width disagree")
    return header, values


def _column_indices(header: list[str], prefix: str) -> list[int]:
    numbered: list[tuple[int, int]] = []
    for index, name in enumerate(header):
        if name.startswith(prefix):
            numbered.append((int(name.removeprefix(prefix)), index))
    return [index for _, index in sorted(numbered)]


def _named_column(header: list[str], values: np.ndarray, name: str) -> np.ndarray:
    try:
        return values[:, header.index(name)]
    except ValueError as error:
        raise ValueError(f"trajectory is missing required column {name!r}") from error


def _full_geometry_chair_clearance(
    model: mujoco.MjModel, qpos: np.ndarray,
) -> dict[str, np.ndarray]:
    """Audit penetration without relying solely on generated contact pairs.

    MuJoCo documents that a large positive ``distmax`` can be inaccurate for
    general convex collision pairs.  With a zero cutoff the query answers the
    only question needed for this audit: whether a pair has negative signed
    distance.  Zero therefore means clear or touching, not a fabricated
    positive clearance measurement.
    """
    robot_geoms = [
        geom_id for geom_id in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").startswith(
            "gmr_official_collision_"
        )
    ]
    chair_geoms = [
        geom_id for geom_id in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").startswith(
            CHAIR_GEOM_PREFIXES
        )
    ]
    seat_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "seat_support_geom")
    if not robot_geoms or not chair_geoms or seat_geom < 0:
        raise ValueError("task lacks official G1 collision geoms or semantic-chair collision geoms")
    data = mujoco.MjData(model)
    support_seat = np.zeros(len(qpos), dtype=np.float64)
    forbidden = np.zeros(len(qpos), dtype=np.float64)
    for frame, state in enumerate(qpos):
        data.qpos[:] = state
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        support_min = 0.0
        forbidden_min = 0.0
        for robot_geom in robot_geoms:
            body_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[robot_geom])
            ) or ""
            for chair_geom in chair_geoms:
                distance = float(mujoco.mj_geomDistance(
                    model, data, robot_geom, chair_geom, 0.0, None
                ))
                if body_name in SEAT_SUPPORT_BODY_NAMES and chair_geom == seat_geom:
                    support_min = min(support_min, distance)
                else:
                    forbidden_min = min(forbidden_min, distance)
        support_seat[frame] = support_min
        forbidden[frame] = forbidden_min
    return {
        "support_seat_min_distance_m": support_seat,
        "forbidden_min_distance_m": forbidden,
        "all_pair_min_distance_m": np.minimum(support_seat, forbidden),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--task-xml", required=True, type=Path)
    parser.add_argument("--robot-motion", required=True, type=Path)
    parser.add_argument("--start-frame", required=True, type=int)
    parser.add_argument("--max-root-z-frame-jump-m", type=float, default=0.04)
    parser.add_argument("--max-root-rmse-m", type=float, default=0.10)
    parser.add_argument(
        "--max-root-p95-position-error-m", type=float, default=0.10,
        help=(
            "95th-percentile Euclidean root tracking error gate.  This blocks "
            "a short fall that can otherwise be hidden by a low whole-clip RMSE."
        ),
    )
    parser.add_argument("--joint-limit-tolerance-rad", type=float, default=1e-4)
    parser.add_argument("--seat-contact-first-local-frame", type=int, default=None)
    parser.add_argument("--min-seat-support-normal-force-n", type=float, default=20.0)
    parser.add_argument("--min-seat-supported-frame-ratio", type=float, default=0.60)
    parser.add_argument("--max-chair-contact-penetration-m", type=float, default=0.01)
    parser.add_argument(
        "--max-geometry-chair-penetration-m", type=float, default=0.003,
        help="zero-cutoff official-collision penetration tolerance for the static chair",
    )
    parser.add_argument(
        "--pre-sit-contact-end-local-frame",
        type=int,
        default=None,
        help=(
            "exclusive local frame bound for a pre-sit physical-chair "
            "clearance gate"
        ),
    )
    parser.add_argument(
        "--max-pre-sit-chair-penetration-m",
        type=float,
        default=0.005,
        help="maximum permitted actual G1-chair penetration before the seated interval",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    header, values = _load_csv(args.trajectory)
    qpos_indices = _column_indices(header, "qpos_")
    model = mujoco.MjModel.from_xml_path(str(args.task_xml))
    if len(qpos_indices) != model.nq:
        raise ValueError("trajectory qpos dimensions do not match task XML")
    if not np.isfinite(args.joint_limit_tolerance_rad) or args.joint_limit_tolerance_rad < 0.0:
        raise ValueError("joint-limit-tolerance-rad must be finite and non-negative")
    if (not np.isfinite(args.max_root_p95_position_error_m) or
            args.max_root_p95_position_error_m < 0.0):
        raise ValueError("max-root-p95-position-error-m must be finite and non-negative")
    qpos = values[:, qpos_indices]
    root = qpos[:, :3]
    if (not np.isfinite(args.max_geometry_chair_penetration_m) or
            args.max_geometry_chair_penetration_m < 0.0):
        raise ValueError("max-geometry-chair-penetration-m must be finite and non-negative")
    geometry_clearance = _full_geometry_chair_clearance(model, qpos)
    geometry_support_minimum = float(np.min(geometry_clearance["support_seat_min_distance_m"]))
    geometry_forbidden_minimum = float(np.min(geometry_clearance["forbidden_min_distance_m"]))
    geometry_pass = bool(
        geometry_support_minimum >= -args.max_geometry_chair_penetration_m
        and geometry_forbidden_minimum >= -args.max_geometry_chair_penetration_m
    )
    with args.robot_motion.open("rb") as handle:
        motion = pickle.load(handle)
    reference = np.asarray(motion["root_pos"], dtype=np.float64)
    source_frames: np.ndarray | None = None
    if "source_frame" in header:
        source_frames = values[:, header.index("source_frame")].astype(np.int64)
        if (np.any(np.diff(source_frames) < 0) or source_frames[0] < args.start_frame or
                source_frames[-1] >= len(reference)):
            raise ValueError("trajectory source_frame column is invalid for robot_motion")
        reference_root = reference[source_frames]
    else:
        begin = args.start_frame
        end = begin + len(root)
        if not 0 <= begin < end <= len(reference):
            raise ValueError("reference range is outside robot_motion")
        reference_root = reference[begin:end]
    error = root - reference_root
    root_z_delta = np.diff(root[:, 2])
    finite = bool(np.isfinite(values).all())
    max_jump = float(np.max(np.abs(root_z_delta))) if len(root_z_delta) else 0.0
    rmse = float(np.sqrt(np.mean(np.square(error))))
    p95_error = float(np.percentile(np.linalg.norm(error, axis=1), 95))
    joint_limit_violations: list[dict[str, object]] = []
    maximum_joint_limit_excess = 0.0
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        if not model.jnt_limited[joint_id]:
            continue
        qpos_address = int(model.jnt_qposadr[joint_id])
        lower, upper = (float(value) for value in model.jnt_range[joint_id])
        positions = qpos[:, qpos_address]
        excess = np.maximum.reduce((np.zeros_like(positions), lower - positions, positions - upper))
        maximum = float(np.max(excess))
        maximum_joint_limit_excess = max(maximum_joint_limit_excess, maximum)
        if maximum > args.joint_limit_tolerance_rad:
            frame = int(np.argmax(excess))
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            joint_limit_violations.append({
                "joint": joint_name or f"<unnamed:{joint_id}>",
                "frame": frame,
                "qpos_rad": float(positions[frame]),
                "range_rad": [lower, upper],
                "excess_rad": maximum,
            })
    joint_limits_pass = not joint_limit_violations
    seating: dict[str, object] | None = None
    seat_pass = True
    if args.seat_contact_first_local_frame is not None:
        first = args.seat_contact_first_local_frame
        if not 0 <= first < len(root):
            raise ValueError("seat contact frame is outside trajectory")
        seat_force = _named_column(header, values, "semantic_seat_normal_force_n")[first:]
        chair_distance = _named_column(header, values, "semantic_chair_min_contact_distance_m")[first:]
        support = seat_force >= args.min_seat_support_normal_force_n
        supported_ratio = float(np.mean(support))
        minimum_distance = float(np.min(chair_distance))
        minimum_support_geometry_distance = float(
            np.min(geometry_clearance["support_seat_min_distance_m"][first:])
        )
        minimum_forbidden_geometry_distance = float(
            np.min(geometry_clearance["forbidden_min_distance_m"][first:])
        )
        seat_pass = bool(
            supported_ratio >= args.min_seat_supported_frame_ratio
            and minimum_distance >= -args.max_chair_contact_penetration_m
            and minimum_support_geometry_distance >= -args.max_geometry_chair_penetration_m
            and minimum_forbidden_geometry_distance >= -args.max_geometry_chair_penetration_m
        )
        seating = {
            "first_local_frame": first,
            "min_seat_support_normal_force_n": args.min_seat_support_normal_force_n,
            "supported_frame_ratio": supported_ratio,
            "minimum_chair_contact_distance_m": minimum_distance,
            "max_allowed_penetration_m": args.max_chair_contact_penetration_m,
            "minimum_support_seat_geometry_distance_m": minimum_support_geometry_distance,
            "minimum_forbidden_chair_geometry_distance_m": minimum_forbidden_geometry_distance,
            "max_allowed_geometry_penetration_m": args.max_geometry_chair_penetration_m,
            "pass": seat_pass,
        }
    pre_sit_chair_clearance: dict[str, object] | None = None
    pre_sit_clearance_pass = True
    if args.pre_sit_contact_end_local_frame is not None:
        end_local = args.pre_sit_contact_end_local_frame
        if not 1 <= end_local <= len(root):
            raise ValueError("pre-sit contact end frame is outside trajectory")
        chair_distance = _named_column(
            header, values, "semantic_chair_min_contact_distance_m"
        )[:end_local]
        minimum_distance = float(np.min(chair_distance))
        minimum_geometry_distance = float(
            np.min(geometry_clearance["all_pair_min_distance_m"][:end_local])
        )
        pre_sit_clearance_pass = bool(
            minimum_distance >= -args.max_pre_sit_chair_penetration_m
            and minimum_geometry_distance >= -args.max_geometry_chair_penetration_m
        )
        pre_sit_chair_clearance = {
            "end_local_frame_exclusive": end_local,
            "minimum_chair_contact_distance_m": minimum_distance,
            "minimum_geometry_chair_distance_m": minimum_geometry_distance,
            "max_allowed_penetration_m": args.max_pre_sit_chair_penetration_m,
            "max_allowed_geometry_penetration_m": args.max_geometry_chair_penetration_m,
            "pass": pre_sit_clearance_pass,
        }
    report = {
        "schema_version": 1,
        "purpose": "bounded_actuator_mj_step_trajectory_audit",
        "trajectory": str(args.trajectory),
        "task_xml": str(args.task_xml),
        "frame_count": int(len(root)),
        "source_frame_range": (
            [int(source_frames[0]), int(source_frames[-1])]
            if source_frames is not None else [args.start_frame, args.start_frame + len(root) - 1]
        ),
        "finite_state_and_control": finite,
        "root_tracking": {
            "rmse_m": rmse,
            "p95_position_error_m": p95_error,
            "max_p95_position_error_m": args.max_root_p95_position_error_m,
        },
        "root_height": {
            "min_m": float(np.min(root[:, 2])),
            "max_m": float(np.max(root[:, 2])),
            "max_frame_jump_m": max_jump,
            "threshold_m": args.max_root_z_frame_jump_m,
        },
        "joint_limits": {
            "tolerance_rad": args.joint_limit_tolerance_rad,
            "maximum_excess_rad": maximum_joint_limit_excess,
            "violations": joint_limit_violations,
            "pass": joint_limits_pass,
        },
        "semantic_seating": seating,
        "pre_sit_chair_clearance": pre_sit_chair_clearance,
        "full_geometry_chair_clearance": {
            "distance_query": "mj_geomDistance_distmax_zero",
            "minimum_support_seat_distance_m": geometry_support_minimum,
            "minimum_forbidden_distance_m": geometry_forbidden_minimum,
            "max_allowed_penetration_m": args.max_geometry_chair_penetration_m,
            "pass": geometry_pass,
        },
        "acceptance": {
            "max_root_rmse_m": args.max_root_rmse_m,
            "max_root_p95_position_error_m": args.max_root_p95_position_error_m,
            "physical_smoke_pass": bool(
                finite
                and rmse <= args.max_root_rmse_m
                and p95_error <= args.max_root_p95_position_error_m
                and max_jump <= args.max_root_z_frame_jump_m
                and joint_limits_pass
                and geometry_pass
                and seat_pass
                and pre_sit_clearance_pass
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
