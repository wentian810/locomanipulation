#!/usr/bin/env python3
"""Run the frozen ProtoMotions G1 tracker with or without a static chair.

This is an inference-only bridge.  It deliberately uses the tracker's native
G1 collision model and pretrained ONNX policy.  In ``--floor-only`` mode no
chair may be present; this is the prerequisite stability gate before scene
interaction is attempted.  Otherwise it imports a static chair MJCF.  The
robot state is set exactly once at the reference's first frame; every later
state comes solely from bounded PD actuation and ``mj_step``.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import yaml


CHAIR_GEOMS = frozenset({
    "seat_support_geom",
    "backrest_geom",
    "leg_front_left_geom",
    "leg_front_right_geom",
    "leg_back_left_geom",
    "leg_back_right_geom",
})

DEFAULT_UNITREE_ACTUATOR_LIMITS = Path(__file__).with_name(
    "g1_unitree_actuator_limits.json"
)
DEFAULT_OFFICIAL_G1_MJCF = (
    Path(__file__).resolve().parents[2]
    / "third_party/unitree_ros_official/robots/g1_description/g1_29dof_rev_1_0.xml"
)


def apply_official_g1_physical_limits(
    model: mujoco.MjModel,
    actuator_limits_path: Path,
    official_g1_mjcf: Path,
) -> dict[str, object]:
    """Apply the revision-matched G1 bounds to the frozen policy's model.

    The upstream zero-shot demo intentionally leaves its implicit-PD actuators
    unbounded.  That is useful for a visual demo but is not an admissible V24
    physical rollout.  This function changes only MuJoCo's static actuator
    force caps and mechanical hinge ranges before the one permitted initial
    state is set.  It never changes a reference, robot state, or scene pose.
    """
    raw = json.loads(actuator_limits_path.read_text(encoding="utf-8"))
    torque_limits = raw.get("joint_torque_limits_nm")
    if raw.get("schema_version") != 1 or not isinstance(torque_limits, dict):
        raise ValueError(f"unsupported Unitree actuator-limit config: {actuator_limits_path}")
    official_root = ET.parse(official_g1_mjcf).getroot()
    official_ranges: dict[str, tuple[float, float]] = {}
    for joint in official_root.findall(".//joint"):
        name = joint.get("name")
        range_text = joint.get("range")
        if name and range_text:
            values = tuple(float(value) for value in range_text.split())
            if len(values) == 2 and values[0] < values[1]:
                official_ranges[name] = values

    applied: list[dict[str, object]] = []
    for actuator_id in range(model.nu):
        if model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
            raise ValueError("frozen tracker actuator must use a scalar joint transmission")
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not joint_name or joint_name not in torque_limits or joint_name not in official_ranges:
            raise ValueError(f"missing official G1 limit for policy joint {joint_name!r}")
        torque_limit = float(torque_limits[joint_name])
        if not np.isfinite(torque_limit) or torque_limit <= 0.0:
            raise ValueError(f"invalid torque limit for {joint_name!r}: {torque_limit}")
        lower, upper = official_ranges[joint_name]
        model.actuator_forcelimited[actuator_id] = 1
        model.actuator_forcerange[actuator_id] = (
            -torque_limit,
            torque_limit,
        )
        model.jnt_limited[joint_id] = 1
        model.jnt_range[joint_id] = (lower, upper)
        applied.append({
            "joint": joint_name,
            "torque_limit_nm": torque_limit,
            "range_rad": [lower, upper],
        })
    return {
        "actuator_limit_config": str(actuator_limits_path),
        "official_g1_mjcf": str(official_g1_mjcf),
        "actuated_joint_count": len(applied),
        "limits": applied,
    }


def chair_contact_metrics(
    model: mujoco.MjModel, data: mujoco.MjData, chair_geom_ids: set[int], floor_geom_id: int
) -> tuple[int, float, int, float, float]:
    """Return all-chair and seat-only contact counts, forces, and min distance."""
    seat_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "seat_support_geom")
    chair_count = 0
    chair_force = 0.0
    seat_count = 0
    seat_force = 0.0
    min_distance = float("inf")
    wrench = np.zeros(6, dtype=np.float64)
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom_pair = {int(contact.geom1), int(contact.geom2)}
        chair_members = geom_pair & chair_geom_ids
        if not chair_members:
            continue
        # A semantic chair must not be counted as colliding with its floor or
        # with another chair primitive; only robot--chair contacts matter.
        other = geom_pair - chair_members
        if not other or floor_geom_id in other or other & chair_geom_ids:
            continue
        mujoco.mj_contactForce(model, data, contact_index, wrench)
        normal_force = max(0.0, float(wrench[0]))
        chair_count += 1
        chair_force += normal_force
        min_distance = min(min_distance, float(contact.dist))
        if seat_geom_id in chair_members:
            seat_count += 1
            seat_force += normal_force
    return chair_count, chair_force, seat_count, seat_force, min_distance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker-repo", required=True, type=Path)
    parser.add_argument("--reference-qpos-csv", required=True, type=Path)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--scene-mjcf", required=True, type=Path)
    parser.add_argument(
        "--unitree-actuator-limits", type=Path,
        default=DEFAULT_UNITREE_ACTUATOR_LIMITS,
        help="official G1 force-cap mapping applied to every frozen PD actuator",
    )
    parser.add_argument(
        "--official-g1-mjcf", type=Path,
        default=DEFAULT_OFFICIAL_G1_MJCF,
        help="revision-matched G1 hinge ranges applied before physical rollout",
    )
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--expected-seat-start-source-frame", type=int, default=179)
    parser.add_argument(
        "--floor-only", action="store_true",
        help="require a floor-only model and evaluate the frozen policy before chair interaction",
    )
    parser.add_argument("--max-floor-root-rmse-m", type=float, default=0.10)
    parser.add_argument("--floor-root-height-min-m", type=float, default=0.25)
    parser.add_argument("--max-floor-root-step-m", type=float, default=0.08)
    args = parser.parse_args()
    if args.output_npz.exists() or args.summary_json.exists():
        raise FileExistsError("refusing to overwrite an existing physical-tracker artifact")

    tracker_repo = args.tracker_repo.resolve()
    sys.path.insert(0, str(tracker_repo))
    from pipeline.deploy.motion_utils import MotionPlayer
    from pipeline.deploy.mujoco_runner import (
        build_onnx_inputs,
        load_mujoco_model,
        read_robot_state,
        set_initial_pose,
    )
    from pipeline.deploy.state_utils import apply_heading_offset_np, compute_yaw_offset_np
    from pipeline.proto_bridge import qpos_to_motion_data

    qpos = np.loadtxt(args.reference_qpos_csv, delimiter=",")
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"expected reference qpos [T,36], got {qpos.shape}")
    with args.onnx.with_suffix(".yaml").open(encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle)
    robot_metadata = metadata["robot"]
    timing = metadata["timing"]
    control = metadata["control"]
    motion_metadata = metadata["motion"]
    runtime = metadata["_runtime"]
    if int(robot_metadata["num_dofs"]) != 29:
        raise ValueError("the frozen tracker is not a 29-DoF G1 policy")

    # Reference FK must use the policy's unmodified 33-body robot model.  The
    # chair is intentionally absent here: it is a static environment object,
    # not a policy body or a source of altered network observations.
    canonical_policy_mjcf = tracker_repo / "pipeline/assets/proto_g1/g1_holo_compat.xml"
    motion_data = qpos_to_motion_data(qpos, args.source_fps, canonical_policy_mjcf)
    player = MotionPlayer(motion_data)
    control_dt = float(timing["control_dt"])
    physics_dt = float(timing["physics_dt"])
    decimation = int(timing["decimation"])
    if not np.isclose(control_dt, physics_dt * decimation):
        raise ValueError("tracker timing does not match its MuJoCo decimation")

    model, data = load_mujoco_model(
        str(args.scene_mjcf), control["stiffness"], control["damping"], physics_dt, args.onnx.parent
    )
    if model.nq != 36 or model.nu != 29:
        raise ValueError(f"combined scene model mismatch: nq={model.nq}, nu={model.nu}")
    physical_limits = apply_official_g1_physical_limits(
        model, args.unitree_actuator_limits, args.official_g1_mjcf
    )
    chair_geom_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in CHAIR_GEOMS
    }
    if args.floor_only:
        if any(identifier >= 0 for identifier in chair_geom_ids):
            raise ValueError("--floor-only requires a model with no semantic chair geoms")
        chair_geom_ids = set()
    elif any(identifier < 0 for identifier in chair_geom_ids):
        raise ValueError("combined scene lacks one or more semantic-chair collision geoms")
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    output_names = [output.name for output in session.get_outputs()]
    if "joint_pos_targets" not in output_names:
        raise ValueError(f"ONNX outputs lack joint_pos_targets: {output_names}")

    # The only legal direct state write is this initial configuration.  The
    # subsequent loop writes controls then advances physical time by mj_step.
    set_initial_pose(model, data, player)
    actual_qpos: list[np.ndarray] = []
    actual_qvel: list[np.ndarray] = []
    reference_root: list[np.ndarray] = []
    reference_dof: list[np.ndarray] = []
    pd_target_log: list[np.ndarray] = []
    chair_counts: list[int] = []
    chair_forces: list[float] = []
    seat_counts: list[int] = []
    seat_forces: list[float] = []
    min_distances: list[float] = []

    previous_pd: np.ndarray | None = None
    before_previous_pd: np.ndarray | None = None
    previous_action: np.ndarray | None = None
    filtered_target: np.ndarray | None = None
    heading_offset: np.ndarray | None = None
    accel_limit = control.get("pd_target_max_accel")
    ema_alpha = float(control.get("action_ema_alpha", 1.0))

    for frame_index in range(player.total_frames):
        state = read_robot_state(
            data, int(robot_metadata["anchor_body_index"]), int(robot_metadata["root_body_index"])
        )
        reference = player.get_state_at_frame(frame_index)
        if heading_offset is None:
            heading_offset = compute_yaw_offset_np(
                state["body_rot"][int(robot_metadata["anchor_body_index"])],
                player.get_state_at_frame(0)["body_rot"][int(robot_metadata["anchor_body_index"])],
            )
        future = player.get_future_references(frame_index, motion_metadata["future_step_indices"])
        future["body_rot"] = apply_heading_offset_np(heading_offset, future["body_rot"])
        inputs = build_onnx_inputs(
            state, future, runtime["onnx_name_to_in_key"], int(robot_metadata["anchor_body_index"]),
            int(robot_metadata["num_dofs"]), previous_action,
        )
        output_by_name = dict(zip(output_names, session.run(output_names, inputs)))
        pd_target = np.asarray(output_by_name["joint_pos_targets"]).squeeze().copy()
        if accel_limit is not None and previous_pd is not None and before_previous_pd is not None:
            delta = pd_target - previous_pd
            previous_delta = previous_pd - before_previous_pd
            pd_target = previous_pd + previous_delta + np.clip(
                delta - previous_delta, -float(accel_limit), float(accel_limit)
            )
        before_previous_pd = previous_pd
        previous_pd = pd_target.copy()
        if ema_alpha < 1.0:
            if filtered_target is None:
                filtered_target = pd_target.copy()
            pd_target = ema_alpha * pd_target + (1.0 - ema_alpha) * filtered_target
            filtered_target = pd_target.copy()
        previous_action = pd_target.copy()

        data.ctrl[:] = pd_target
        for _ in range(decimation):
            mujoco.mj_step(model, data)
        metrics = (
            chair_contact_metrics(model, data, chair_geom_ids, floor_geom_id)
            if chair_geom_ids else (0, 0.0, 0, 0.0, float("inf"))
        )
        actual_qpos.append(data.qpos.copy())
        actual_qvel.append(data.qvel.copy())
        reference_root.append(np.asarray(reference["body_pos"][0]).copy())
        reference_dof.append(np.asarray(reference["dof_pos"]).copy())
        pd_target_log.append(pd_target)
        chair_counts.append(metrics[0])
        chair_forces.append(metrics[1])
        seat_counts.append(metrics[2])
        seat_forces.append(metrics[3])
        min_distances.append(metrics[4])

    actual = np.stack(actual_qpos)
    reference_root_array = np.stack(reference_root)
    source_frames = np.arange(player.total_frames, dtype=np.float64) * control_dt * args.source_fps
    expected_seat = (
        source_frames >= float(args.expected_seat_start_source_frame)
        if chair_geom_ids else np.zeros(len(source_frames), dtype=bool)
    )
    seat_force_array = np.asarray(seat_forces)
    finite = bool(np.isfinite(actual).all())
    root_error = np.linalg.norm(actual[:, :3] - reference_root_array, axis=1)
    root_step = np.linalg.norm(np.diff(actual[:, :3], axis=0), axis=1)
    finite_distances = np.asarray([value for value in min_distances if np.isfinite(value)])
    expected_seat_force = seat_force_array[expected_seat]
    seat_support_ratio = float(np.mean(expected_seat_force > 1.0)) if expected_seat_force.size else 0.0
    min_chair_distance = float(np.min(finite_distances)) if finite_distances.size else float("inf")
    stable_floor_gate = bool(
        finite
        and float(np.sqrt(np.mean(root_error**2))) <= args.max_floor_root_rmse_m
        and float(np.min(actual[:, 2])) >= args.floor_root_height_min_m
        and (float(np.max(root_step)) if root_step.size else 0.0) <= args.max_floor_root_step_m
    )
    summary = {
        "schema_version": 1,
        "purpose": (
            "frozen_pretrained_tracker_floor_only_physical_gate"
            if args.floor_only else "frozen_pretrained_tracker_static_chair_physics_audit"
        ),
        "status": (
            "accepted_frozen_pretrained_floor_gate" if args.floor_only and stable_floor_gate
            else "rejected_frozen_pretrained_floor_gate" if args.floor_only
            else "measured_not_promoted"
        ),
        "physics_contract": {
            "initial_qpos_qvel_write_count": 1,
            "qpos_or_qvel_writes_after_initialization": 0,
            "external_body_force_writes": 0,
            "control": "frozen_onnx_PD_targets",
            "actuator_force_limits": "official_Unitree_G1_per_joint",
            "joint_limits": "revision_matched_official_Unitree_G1",
            "physics_advance": f"mj_step x {decimation} at {physics_dt:.6f}s per control frame",
        },
        "inputs": {
            "reference_qpos_csv": str(args.reference_qpos_csv),
            "scene_mjcf": str(args.scene_mjcf),
            "floor_only": bool(args.floor_only),
            "onnx": str(args.onnx),
            "physical_limits": physical_limits,
            "expected_seat_start_source_frame": int(args.expected_seat_start_source_frame),
        },
        "frames": int(player.total_frames),
        "finite": finite,
        "root_tracking_rmse_m": float(np.sqrt(np.mean(root_error**2))),
        "root_tracking_p95_m": float(np.quantile(root_error, 0.95)),
        "root_height_min_m": float(np.min(actual[:, 2])),
        "root_height_max_m": float(np.max(actual[:, 2])),
        "root_step_max_m": float(np.max(root_step)) if root_step.size else 0.0,
        "stable_floor_gate": stable_floor_gate,
        "chair_contact_frames": int(np.sum(np.asarray(chair_counts) > 0)),
        "seat_contact_frames": int(np.sum(np.asarray(seat_counts) > 0)),
        "expected_seat_frames": int(np.sum(expected_seat)),
        "expected_seat_force_support_ratio": seat_support_ratio,
        "min_robot_chair_contact_distance_m": min_chair_distance if np.isfinite(min_chair_distance) else None,
        "max_robot_chair_penetration_m": max(0.0, -min_chair_distance) if np.isfinite(min_chair_distance) else 0.0,
        "trajectory_npz": str(args.output_npz),
    }
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        time=np.arange(player.total_frames, dtype=np.float64) * control_dt,
        source_frame=source_frames,
        actual_qpos=actual,
        actual_qvel=np.stack(actual_qvel),
        reference_root_pos=reference_root_array,
        reference_dof_pos=np.stack(reference_dof),
        pd_target=np.stack(pd_target_log),
        chair_contact_count=np.asarray(chair_counts, dtype=np.int32),
        chair_normal_force=np.asarray(chair_forces),
        seat_contact_count=np.asarray(seat_counts, dtype=np.int32),
        seat_normal_force=seat_force_array,
        min_robot_chair_contact_distance=np.asarray(min_distances),
    )
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
