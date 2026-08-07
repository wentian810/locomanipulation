#!/usr/bin/env python3
"""Diagnose whether a GMR sit segment can settle through MuJoCo contact.

This intentionally is *not* a kinematic playback renderer.  The G1 free-base
position and orientation are assigned once from ``--start-key`` before the
first simulation step, then remain completely unreferenced.  During the
simulation the controller receives only the 29 interpolated GMR joint-angle
targets, produces bounded torques, and advances the fixed chair scene with
``mj_step``.  Consequently, a passed result means that gravity and the
official collision geometry, not a per-frame root correction, put the robot on
the seat.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from run_g1_gravity_seat_landing import (
    _actuator_state_addresses,
    _chair_contact_metrics,
    _robot_weight,
)


def _joint_target(
    key_qpos: np.ndarray,
    frame: float,
    qpos_addresses: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate only hinge targets; the free root is omitted."""
    lower = int(np.floor(frame))
    upper = min(lower + 1, key_qpos.shape[0] - 1)
    alpha = frame - lower
    return (1.0 - alpha) * key_qpos[lower, qpos_addresses] + alpha * key_qpos[upper, qpos_addresses]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-xml", required=True, type=Path)
    parser.add_argument("--start-key", type=int, default=90)
    parser.add_argument("--end-key", type=int, default=220)
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--kp-nm-per-rad", type=float, default=80.0)
    parser.add_argument("--kd-nms-per-rad", type=float, default=8.0)
    parser.add_argument("--sample-period-s", type=float, default=0.01)
    parser.add_argument("--arrays", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if not (0 <= args.start_key < args.end_key):
        raise ValueError("require 0 <= start-key < end-key")
    if min(args.source_fps, args.settle_s, args.kp_nm_per_rad, args.kd_nms_per_rad, args.sample_period_s) <= 0.0:
        raise ValueError("all timing and gain arguments must be positive")

    model = mujoco.MjModel.from_xml_path(str(args.task_xml))
    if args.end_key >= model.nkey:
        raise ValueError("requested segment exceeds task keyframes")
    if model.nu != model.nq - 7:
        raise ValueError("expected one torque actuator for every G1 hinge")
    qpos_addresses, dof_addresses = _actuator_state_addresses(model)
    source_duration_s = (args.end_key - args.start_key) / args.source_fps
    duration_s = source_duration_s + args.settle_s
    key_qpos = model.key_qpos.copy()

    # The sole state assignment.  In particular, the root is never shifted,
    # re-oriented, or re-written after this point.
    data = mujoco.MjData(model)
    data.qpos[:] = key_qpos[args.start_key]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    sample_stride = max(1, int(round(args.sample_period_s / model.opt.timestep)))
    steps = int(round(duration_s / model.opt.timestep))
    samples: list[dict[str, float]] = []
    for step in range(steps):
        source_time_s = min(data.time, source_duration_s)
        source_frame = args.start_key + source_time_s * args.source_fps
        target = _joint_target(key_qpos, source_frame, qpos_addresses)
        position_error = target - data.qpos[qpos_addresses]
        torque = args.kp_nm_per_rad * position_error - args.kd_nms_per_rad * data.qvel[dof_addresses]
        data.ctrl[:] = np.clip(torque, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        mujoco.mj_step(model, data)

        if step % sample_stride != 0 and step + 1 != steps:
            continue
        reference = _joint_target(
            key_qpos,
            args.start_key + min(data.time, source_duration_s) * args.source_fps,
            qpos_addresses,
        )
        contact = _chair_contact_metrics(model, data)
        samples.append({
            "time_s": float(data.time),
            "source_frame": float(args.start_key + min(data.time, source_duration_s) * args.source_fps),
            "root_x_m": float(data.qpos[0]),
            "root_y_m": float(data.qpos[1]),
            "root_z_m": float(data.qpos[2]),
            "root_speed_mps": float(np.linalg.norm(data.qvel[:3])),
            "joint_rmse_rad": float(np.sqrt(np.mean((data.qpos[qpos_addresses] - reference) ** 2))),
            "actuator_abs_torque_sum_nm": float(np.sum(np.abs(data.ctrl))),
            **contact,
        })

    if not samples:
        raise RuntimeError("simulation produced no samples")
    arrays = {
        name: np.asarray([sample[name] for sample in samples], dtype=np.float64)
        for name in samples[0]
    }
    tail = max(1, int(round(args.settle_s / args.sample_period_s)))
    weight_n = _robot_weight(model)
    support_tail = arrays["support_normal_force_n"][-tail:]
    other_distance = arrays["minimum_other_chair_contact_distance_m"][-tail:]
    support_ratio = float(np.median(support_tail) / weight_n)
    root_z = arrays["root_z_m"]
    tail_support_fraction = float(np.mean(arrays["support_contact_count"][-tail:] > 0.0))
    finite_other = np.isfinite(other_distance)
    tail_min_other_distance = float(np.min(other_distance[finite_other])) if np.any(finite_other) else None
    max_root_step = float(np.max(np.abs(np.diff(root_z)))) if len(root_z) > 1 else 0.0

    # These gates intentionally separate a clean physical demonstration from
    # a merely finite rollout.  A failed run remains a diagnostic artifact.
    accepted = (
        tail_support_fraction >= 0.8
        and 0.5 <= support_ratio <= 1.5
        and (tail_min_other_distance is None or tail_min_other_distance >= -0.003)
        and max_root_step <= 0.04
        and float(np.percentile(arrays["joint_rmse_rad"], 95.0)) <= 0.45
    )
    args.arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.arrays, **arrays)
    report = {
        "schema_version": 1,
        "purpose": "free_base_gmr_joint_only_physical_sit_segment_diagnostic",
        "status": "accepted_physical_sit_segment" if accepted else "rejected_physical_sit_segment",
        "physical_contract": {
            "initialization": "source_start_key_qpos_and_zero_qvel_written_once_before_first_mj_step",
            "runtime": "GMR_joint_angle_PD_torque_plus_mj_step",
            "root_policy": "free_base_position_and_orientation_never_targeted_or_written_after_initialization",
            "scene": "fixed_semantic_chair_with_matching_visual_and_collision_geometry",
            "forbidden": ["post_initialization_qpos_write", "post_initialization_qvel_write", "xfrc_applied", "mocap_weld"],
        },
        "settings": {
            "start_key": args.start_key,
            "end_key": args.end_key,
            "source_fps": args.source_fps,
            "source_duration_s": source_duration_s,
            "settle_s": args.settle_s,
            "kp_nm_per_rad": args.kp_nm_per_rad,
            "kd_nms_per_rad": args.kd_nms_per_rad,
        },
        "summary": {
            "robot_weight_n": weight_n,
            "tail_support_contact_fraction": tail_support_fraction,
            "tail_median_support_normal_force_n": float(np.median(support_tail)),
            "tail_median_support_to_weight_ratio": support_ratio,
            "tail_minimum_other_chair_contact_distance_m": tail_min_other_distance,
            "p95_joint_rmse_rad": float(np.percentile(arrays["joint_rmse_rad"], 95.0)),
            "root_z_range_m": [float(np.min(root_z)), float(np.max(root_z))],
            "max_sampled_root_z_step_m": max_root_step,
            "initial_qpos_wxyz": key_qpos[args.start_key].tolist(),
            "final_qpos_wxyz": data.qpos.tolist(),
            "arrays": str(args.arrays),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
