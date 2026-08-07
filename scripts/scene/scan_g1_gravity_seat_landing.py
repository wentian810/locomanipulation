#!/usr/bin/env python3
"""Find a bounded, gravity-stable G1 seat landing for a fixed MuJoCo chair.

Each candidate writes qpos/qvel exactly once before its first ``mj_step``.
After that, it applies only saturated joint PD torques to the supplied GMR
keyframe.  The chair never moves and no external force, weld, or runtime state
write is used.  The scan therefore selects an initial condition, not a visual
or kinematic correction.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from run_g1_gravity_seat_landing import _actuator_state_addresses, _chair_contact_metrics, _robot_weight


@dataclass
class Landing:
    shift: np.ndarray
    accepted: bool
    summary: dict[str, float | list[float] | None]
    arrays: dict[str, np.ndarray]
    final_qpos: np.ndarray


def torso_orientation_error_deg(target: np.ndarray, actual: np.ndarray) -> float:
    cosine = np.clip((np.trace(target.T @ actual) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def simulate(
    model: mujoco.MjModel,
    target_qpos: np.ndarray,
    target_torso_rotation: np.ndarray,
    qpos_addresses: np.ndarray,
    dof_addresses: np.ndarray,
    robot_weight_n: float,
    shift: np.ndarray,
    duration_s: float,
    kp_nm_per_rad: float,
    kd_nms_per_rad: float,
    max_torso_error_deg: float,
    sample_period_s: float,
) -> Landing:
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    data = mujoco.MjData(model)
    data.qpos[:] = target_qpos
    data.qpos[:3] += shift
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)  # The single initialization before mj_step.

    step_count = int(round(duration_s / model.opt.timestep))
    sample_stride = max(1, int(round(sample_period_s / model.opt.timestep)))
    samples: list[dict[str, float]] = []
    for step in range(step_count):
        error = target_qpos[qpos_addresses] - data.qpos[qpos_addresses]
        torque = kp_nm_per_rad * error - kd_nms_per_rad * data.qvel[dof_addresses]
        data.ctrl[:] = np.clip(torque, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        mujoco.mj_step(model, data)
        if step % sample_stride != 0 and step + 1 != step_count:
            continue
        contact = _chair_contact_metrics(model, data)
        samples.append({
            "time_s": float(data.time),
            "root_x_m": float(data.qpos[0]),
            "root_y_m": float(data.qpos[1]),
            "root_z_m": float(data.qpos[2]),
            "root_speed_mps": float(np.linalg.norm(data.qvel[:3])),
            "torso_orientation_error_deg": torso_orientation_error_deg(
                target_torso_rotation, data.xmat[torso_id].reshape(3, 3)
            ),
            **contact,
        })
    arrays = {
        key: np.asarray([sample[key] for sample in samples], dtype=np.float64)
        for key in samples[0]
    }
    tail_size = max(1, len(samples) // 4)
    support = arrays["support_normal_force_n"][-tail_size:]
    support_ratio = float(np.median(support) / robot_weight_n)
    other_distance = arrays["minimum_other_chair_contact_distance_m"][-tail_size:]
    has_other_distance = bool(np.isfinite(other_distance).any())
    minimum_other_distance = float(np.nanmin(other_distance)) if has_other_distance else None
    support_geometry_distance = arrays["minimum_support_seat_geometry_distance_m"][-tail_size:]
    forbidden_geometry_distance = arrays["minimum_forbidden_chair_geometry_distance_m"][-tail_size:]
    minimum_support_geometry_distance = (
        float(np.nanmin(support_geometry_distance))
        if bool(np.isfinite(support_geometry_distance).any()) else None
    )
    minimum_forbidden_geometry_distance = (
        float(np.nanmin(forbidden_geometry_distance))
        if bool(np.isfinite(forbidden_geometry_distance).any()) else None
    )
    root_z = arrays["root_z_m"]
    max_root_step = float(np.max(np.abs(np.diff(root_z)))) if len(root_z) > 1 else 0.0
    support_coverage = float(np.mean(arrays["support_contact_count"][-tail_size:] > 0.0))
    torso_error = float(np.median(arrays["torso_orientation_error_deg"][-tail_size:]))
    accepted = bool(
        support_coverage >= 0.8
        and 0.5 <= support_ratio <= 1.5
        and (minimum_other_distance is None or minimum_other_distance >= -0.003)
        # ``data.contact`` alone can miss deep geom containment.  The shared
        # helper also evaluates every official G1 collision geom directly.
        and (minimum_support_geometry_distance is None or minimum_support_geometry_distance >= -0.003)
        and (minimum_forbidden_geometry_distance is None or minimum_forbidden_geometry_distance >= -0.003)
        and torso_error <= max_torso_error_deg
        and max_root_step <= 0.04
    )
    summary: dict[str, float | list[float] | None] = {
        "initial_root_shift_m": shift.tolist(),
        "tail_support_coverage": support_coverage,
        "tail_median_support_normal_force_n": float(np.median(support)),
        "tail_median_support_to_weight_ratio": support_ratio,
        "tail_minimum_other_chair_contact_distance_m": minimum_other_distance,
        "tail_minimum_support_seat_geometry_distance_m": minimum_support_geometry_distance,
        "tail_minimum_forbidden_chair_geometry_distance_m": minimum_forbidden_geometry_distance,
        "tail_median_torso_orientation_error_deg": torso_error,
        "root_z_range_m": [float(np.min(root_z)), float(np.max(root_z))],
        "max_sampled_root_z_step_m": max_root_step,
    }
    return Landing(shift=shift, accepted=accepted, summary=summary, arrays=arrays, final_qpos=data.qpos.copy())


def selection_key(landing: Landing) -> tuple[float, ...]:
    summary = landing.summary
    ratio = float(summary["tail_median_support_to_weight_ratio"])
    torso = float(summary["tail_median_torso_orientation_error_deg"])
    return (
        float(np.linalg.norm(landing.shift)),
        abs(ratio - 1.0),
        torso,
        float(summary["max_sampled_root_z_step_m"]),
    )


def rejection_diagnostics(landing: Landing, max_torso_error_deg: float) -> dict[str, float]:
    """Expose failed audit gates instead of returning an opaque empty scan."""
    summary = landing.summary
    ratio = float(summary["tail_median_support_to_weight_ratio"])
    violations = {
        "support_coverage_shortfall": max(0.0, 0.8 - float(summary["tail_support_coverage"])),
        "support_ratio_below_range": max(0.0, 0.5 - ratio),
        "support_ratio_above_range": max(0.0, ratio - 1.5),
        "torso_orientation_excess_deg": max(
            0.0, float(summary["tail_median_torso_orientation_error_deg"]) - max_torso_error_deg
        ),
        "root_z_step_excess_m": max(0.0, float(summary["max_sampled_root_z_step_m"]) - 0.04),
    }
    for label, key in (
        ("other_contact_penetration_m", "tail_minimum_other_chair_contact_distance_m"),
        ("support_geometry_penetration_m", "tail_minimum_support_seat_geometry_distance_m"),
        ("forbidden_geometry_penetration_m", "tail_minimum_forbidden_chair_geometry_distance_m"),
    ):
        distance = summary[key]
        violations[label] = max(0.0, -0.003 - float(distance)) if distance is not None else 0.0
    return violations


def rejection_key(landing: Landing, max_torso_error_deg: float) -> tuple[float, ...]:
    violations = rejection_diagnostics(landing, max_torso_error_deg)
    # Penetration is a hard physical failure, so rank it before softer tracking
    # deviations when preserving one failed candidate for diagnosis.
    return (
        violations["forbidden_geometry_penetration_m"],
        violations["support_geometry_penetration_m"],
        sum(violations.values()),
        *selection_key(landing),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-xml", required=True, type=Path)
    parser.add_argument(
        "--reference-key", type=int, default=None,
        help="task keyframe used as the PD target (exclusive with --reference-qpos-npz)",
    )
    parser.add_argument(
        "--reference-qpos-npz", type=Path, default=None,
        help="NPZ containing a qpos array used as the PD target and initial pose seed",
    )
    parser.add_argument("--arrays", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--duration-s", type=float, default=2.0)
    parser.add_argument("--kp-nm-per-rad", type=float, default=80.0)
    parser.add_argument("--kd-nms-per-rad", type=float, default=8.0)
    parser.add_argument("--max-torso-orientation-error-deg", type=float, default=15.0)
    parser.add_argument("--sample-period-s", type=float, default=0.01)
    parser.add_argument("--root-x-m", nargs="+", type=float, default=(-0.08, -0.04, 0.0, 0.04, 0.08))
    parser.add_argument("--root-y-m", nargs="+", type=float, default=(-0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12))
    parser.add_argument("--root-z-m", nargs="+", type=float, default=(0.0, 0.02, 0.04, 0.06, 0.08))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.arrays.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite existing gravity landing output")
    if min(args.duration_s, args.kp_nm_per_rad, args.sample_period_s, args.max_torso_orientation_error_deg) <= 0:
        raise ValueError("duration, gains, sampling, and torso threshold must be positive")
    if args.kd_nms_per_rad < 0:
        raise ValueError("kd-nms-per-rad must be non-negative")
    for axis, values, limit in (("x", args.root_x_m, 0.12), ("y", args.root_y_m, 0.12), ("z", args.root_z_m, 0.12)):
        if not values or any(not np.isfinite(value) or abs(value) > limit for value in values):
            raise ValueError(f"root-{axis}-m must be finite and within +/-{limit} m")

    model = mujoco.MjModel.from_xml_path(str(args.task_xml))
    if (args.reference_key is None) == (args.reference_qpos_npz is None):
        raise ValueError("provide exactly one of --reference-key or --reference-qpos-npz")
    if model.nu != model.nq - 7:
        raise ValueError("task must expose exactly one hinge motor per G1 joint")
    qpos_addresses, dof_addresses = _actuator_state_addresses(model)
    if args.reference_key is not None:
        if not 0 <= args.reference_key < model.nkey:
            raise ValueError("reference-key is outside task keyframes")
        target_qpos = model.key_qpos[args.reference_key].copy()
        target_source: dict[str, object] = {"task_keyframe": args.reference_key}
    else:
        assert args.reference_qpos_npz is not None
        with np.load(args.reference_qpos_npz) as payload:
            if "qpos" not in payload:
                raise ValueError("reference-qpos-npz must contain qpos")
            target_qpos = np.asarray(payload["qpos"], dtype=np.float64).copy()
        if target_qpos.shape != (model.nq,) or not np.isfinite(target_qpos).all():
            raise ValueError("reference qpos must be a finite vector matching the task model")
        target_source = {"qpos_npz": str(args.reference_qpos_npz)}
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    target_data = mujoco.MjData(model)
    target_data.qpos[:] = target_qpos
    mujoco.mj_forward(model, target_data)
    target_torso_rotation = target_data.xmat[torso_id].reshape(3, 3).copy()
    robot_weight_n = _robot_weight(model)

    landings: list[Landing] = []
    for x in args.root_x_m:
        for y in args.root_y_m:
            for z in args.root_z_m:
                landings.append(simulate(
                    model, target_qpos, target_torso_rotation, qpos_addresses, dof_addresses,
                    robot_weight_n, np.asarray((x, y, z), dtype=np.float64), args.duration_s,
                    args.kp_nm_per_rad, args.kd_nms_per_rad,
                    args.max_torso_orientation_error_deg, args.sample_period_s,
                ))
    accepted = sorted((landing for landing in landings if landing.accepted), key=selection_key)
    best = accepted[0] if accepted else None
    diagnostic = best if best is not None else min(
        landings, key=lambda landing: rejection_key(landing, args.max_torso_orientation_error_deg)
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.arrays.parent.mkdir(parents=True, exist_ok=True)
    if best is not None:
        np.savez_compressed(args.arrays, **best.arrays, final_qpos_wxyz=best.final_qpos)
        initial_qpos = target_qpos.copy()
        initial_qpos[:3] += best.shift
        gravity_summary: dict[str, object] = {
            **best.summary,
            "initial_qpos_wxyz": initial_qpos.tolist(),
            "settled_final_qpos_wxyz": best.final_qpos.tolist(),
            "arrays": str(args.arrays),
        }
    else:
        gravity_summary = None
        # Retain one best failed trajectory and the exact failed gates.  It is
        # diagnostic only and deliberately has no field shape that the
        # materializer can mistake for an accepted landing.
        np.savez_compressed(args.arrays, **diagnostic.arrays, final_qpos_wxyz=diagnostic.final_qpos)
    report = {
        "schema_version": 1,
        "purpose": "bounded_initialization_gravity_seat_landing_scan",
        "status": "accepted_gravity_seat_landing" if best is not None else "no_accepted_gravity_seat_landing",
        "physical_contract": {
            "initialization": "each candidate writes qpos_and_qvel_once_before_first_mj_step",
            "runtime": "bounded_joint_PD_torque_plus_mj_step",
            "scene": "fixed_semantic_chair",
            "forbidden": ["post_initialization_qpos_write", "post_initialization_qvel_write", "xfrc_applied", "mocap_weld"],
        },
        "settings": {
            "target": target_source,
            "duration_s": args.duration_s,
            "kp_nm_per_rad": args.kp_nm_per_rad,
            "kd_nms_per_rad": args.kd_nms_per_rad,
            "root_shift_grid_m": {"x": args.root_x_m, "y": args.root_y_m, "z": args.root_z_m},
        },
        "robot_weight_n": robot_weight_n,
        "candidate_count": len(landings),
        "accepted_candidate_count": len(accepted),
        # Keep this field compatible with materialize_gmr_physical_seat_reference.py:
        # it contains one verified initialization and its mj_step-settled state.
        "summary": gravity_summary,
        "selected": gravity_summary,
        "accepted_candidates": [landing.summary for landing in accepted[:20]],
        "closest_rejected_candidate": (
            None if best is not None else {
                **diagnostic.summary,
                "rejection_diagnostics": rejection_diagnostics(
                    diagnostic, args.max_torso_orientation_error_deg
                ),
                "arrays": str(args.arrays),
            }
        ),
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if best is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
