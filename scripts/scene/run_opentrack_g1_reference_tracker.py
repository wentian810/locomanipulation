#!/usr/bin/env python3
"""Evaluate a frozen OpenTrack G1 policy on one GMR robot reference.

This is an inference-only, floor-only gate.  It converts the GMR 29-DoF
motion to OpenTrack's native MuJoCo trajectory format, sets the *physical*
robot state exactly once, and advances it only through the frozen policy's
torque commands and ``mj_step``.  The reference model is allowed to receive
kinematic writes because it exists solely to construct observations; the
physical model is not reset after initialization.

Do not use this script to render an interaction result.  It deliberately has
no chair input.  A static chair may be introduced only after this floor gate
passes, using one identical collision/visual scene definition.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys
import xml.etree.ElementTree as element_tree
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def _joint_names(model: mujoco.MjModel) -> list[str]:
    """Return the model joint names, rejecting unnamed joints early."""
    names: list[str] = []
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name is None:
            raise ValueError(f"model has an unnamed joint at id {joint_id}")
        names.append(name)
    return names


def _continuous_unit_quaternions_xyzw(quaternions_xyzw: np.ndarray) -> np.ndarray:
    """Normalize XYZW quaternions and remove only equivalent-sign discontinuities."""
    quaternions = np.asarray(quaternions_xyzw, dtype=np.float64).copy()
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms < 1e-8):
        raise ValueError("reference contains a non-finite or zero-norm root quaternion")
    quaternions /= norms[:, None]
    for frame_index in range(1, len(quaternions)):
        if np.dot(quaternions[frame_index - 1], quaternions[frame_index]) < 0.0:
            quaternions[frame_index] *= -1.0
    return quaternions


def _resample_gmr_qpos(
    root_pos: np.ndarray,
    root_quat_xyzw: np.ndarray,
    dof_pos: np.ndarray,
    source_fps: float,
    target_fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample translation/joints linearly and root orientation with SLERP."""
    if source_fps <= 0.0 or target_fps <= 0.0:
        raise ValueError("source_fps and target_fps must both be positive")
    frame_count = len(root_pos)
    if frame_count < 3:
        raise ValueError("at least three reference frames are required for velocity estimation")

    source_times = np.arange(frame_count, dtype=np.float64) / source_fps
    duration = float(source_times[-1])
    target_times = np.arange(0.0, duration + 1e-12, 1.0 / target_fps, dtype=np.float64)
    if len(target_times) < 3:
        raise ValueError("resampling produced fewer than three frames")

    root_pos_out = np.column_stack(
        [np.interp(target_times, source_times, root_pos[:, axis]) for axis in range(3)]
    )
    dof_out = np.column_stack(
        [np.interp(target_times, source_times, dof_pos[:, axis]) for axis in range(dof_pos.shape[1])]
    )
    # GMR serializes ``root_rot`` in SciPy's standard XYZW convention.
    # MuJoCo qpos uses WXYZ, so conversion occurs exactly once after SLERP.
    rotations_xyzw = _continuous_unit_quaternions_xyzw(root_quat_xyzw)
    root_quat_out_xyzw = Slerp(source_times, Rotation.from_quat(rotations_xyzw))(target_times).as_quat()
    root_quat_out_wxyz = root_quat_out_xyzw[:, [3, 0, 1, 2]]
    qpos = np.concatenate([root_pos_out, root_quat_out_wxyz, dof_out], axis=1)
    return qpos, target_times


def _estimate_qvel(qpos: np.ndarray, timestep: float) -> np.ndarray:
    """Estimate a MuJoCo free-joint velocity from a resampled qpos sequence."""
    if timestep <= 0.0:
        raise ValueError("timestep must be positive")
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"expected qpos [T, 36], got {qpos.shape}")
    qvel = np.empty((len(qpos), 35), dtype=np.float64)
    qvel[:, :3] = np.gradient(qpos[:, :3], timestep, axis=0, edge_order=2)
    qvel[:, 6:] = np.gradient(qpos[:, 7:], timestep, axis=0, edge_order=2)

    for frame_index in range(len(qpos)):
        before = max(0, frame_index - 1)
        after = min(len(qpos) - 1, frame_index + 1)
        delta_seconds = (after - before) * timestep
        angular_velocity = np.zeros(3, dtype=np.float64)
        if delta_seconds > 0.0:
            mujoco.mju_subQuat(
                angular_velocity,
                qpos[after, 3:7],
                qpos[before, 3:7],
            )
            angular_velocity /= delta_seconds
        qvel[frame_index, 3:6] = angular_velocity
    return qvel


def _load_gmr_reference(
    motion_pkl: Path,
    source_fps: float,
    target_fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load GMR's robot-motion schema without inventing a root correction."""
    with motion_pkl.open("rb") as handle:
        motion: dict[str, Any] = pickle.load(handle)
    required = ("root_pos", "root_rot", "dof_pos")
    missing = [key for key in required if key not in motion]
    if missing:
        raise KeyError(f"GMR motion is missing required keys: {missing}")

    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_rot_xyzw = np.asarray(motion["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"expected root_pos [T, 3], got {root_pos.shape}")
    if root_rot_xyzw.shape != (len(root_pos), 4):
        raise ValueError(f"expected root_rot [T, 4], got {root_rot_xyzw.shape}")
    if dof_pos.shape != (len(root_pos), 29):
        raise ValueError(f"expected dof_pos [T, 29], got {dof_pos.shape}")
    if not (np.isfinite(root_pos).all() and np.isfinite(root_rot_xyzw).all() and np.isfinite(dof_pos).all()):
        raise ValueError("GMR reference contains non-finite values")

    qpos, times = _resample_gmr_qpos(root_pos, root_rot_xyzw, dof_pos, source_fps, target_fps)
    qvel = _estimate_qvel(qpos, 1.0 / target_fps)
    return qpos, qvel, times


def _write_opentrack_reference(
    destination: Path,
    qpos: np.ndarray,
    qvel: np.ndarray,
    frequency: float,
    canonical_model: mujoco.MjModel,
    joint_names: list[str],
) -> None:
    """Write the minimal official ``Trajectory`` NPZ schema for one G1 clip."""
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite reference artifact: {destination}")
    if canonical_model.nq != 36 or canonical_model.nv != 35 or canonical_model.nu != 29:
        raise ValueError(
            "OpenTrack reference model must be the free-base 29-DoF G1 "
            f"(got nq={canonical_model.nq}, nv={canonical_model.nv}, nu={canonical_model.nu})"
        )
    if len(joint_names) != canonical_model.njnt:
        raise ValueError("reference joint-name count does not match the canonical G1 model")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        qpos=qpos,
        qvel=qvel,
        split_points=np.asarray([0, len(qpos)], dtype=np.int64),
        joint_names=np.asarray(joint_names, dtype=object),
        njnt=np.asarray(canonical_model.njnt, dtype=np.int64),
        jnt_type=np.asarray(canonical_model.jnt_type, dtype=np.int32),
        frequency=np.asarray(float(frequency), dtype=np.float64),
    )


def _assert_joint_contract(gmr_model: mujoco.MjModel, opentrack_model: mujoco.MjModel) -> None:
    """Reject a run if the 29 target joints cannot be mapped by name and order."""
    gmr_names = _joint_names(gmr_model)
    tracker_names = _joint_names(opentrack_model)
    if gmr_model.nq != 36 or gmr_model.nv != 35 or gmr_model.nu != 29:
        raise ValueError("GMR robot MJCF is not the expected free-base 29-DoF G1")
    if (
        gmr_model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE
        or opentrack_model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE
    ):
        raise ValueError("both G1 models must place their single free root joint first")
    # The free-joint *name* has no semantic bearing on the 36-value qpos
    # layout.  The 29 actuated hinge names/order, however, must match exactly.
    if gmr_names[1:] != tracker_names[1:]:
        mismatch = [
            (index, left, right)
            for index, (left, right) in enumerate(zip(gmr_names[1:], tracker_names[1:]), start=1)
            if left != right
        ]
        raise ValueError(
            "GMR/OpenTrack joint ordering differs; no implicit remapping is allowed. "
            f"First differences: {mismatch[:5]}; controlled counts: {len(gmr_names) - 1} vs {len(tracker_names) - 1}"
        )


def _read_free_joint_name(xml_path: Path) -> str:
    """Read the free-root name without compiling an environment-dependent MJCF."""
    xml_root = element_tree.parse(xml_path).getroot()
    names = [
        element.get("name")
        for element in xml_root.iter()
        if element.tag.rsplit("}", maxsplit=1)[-1] == "freejoint" and element.get("name")
    ]
    if len(names) != 1:
        raise ValueError(f"expected exactly one named freejoint in {xml_path}, found {names}")
    return names[0]


def _first_frame_below(values: np.ndarray, threshold: float) -> int | None:
    indices = np.flatnonzero(values < threshold)
    return int(indices[0]) if indices.size else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker-repo", required=True, type=Path)
    parser.add_argument("--motion-pkl", required=True, type=Path)
    parser.add_argument("--gmr-g1-mjcf", required=True, type=Path)
    parser.add_argument("--tracker-root-mjcf", required=True, type=Path)
    parser.add_argument("--checkpoint-config", required=True, type=Path)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--reference-npz", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--control-fps", type=float, default=50.0)
    parser.add_argument("--task", default="G1TrackingGeneral")
    parser.add_argument("--root-height-floor", type=float, default=0.25)
    parser.add_argument("--max-root-step", type=float, default=0.08)
    args = parser.parse_args()

    output_paths = (args.reference_npz, args.output_npz, args.summary_json)
    existing = [path for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing artifact(s): {existing}")
    tracker_repo = args.tracker_repo.resolve()
    for path in (
        args.motion_pkl,
        args.gmr_g1_mjcf,
        args.tracker_root_mjcf,
        args.checkpoint_config,
        args.onnx,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    gmr_model = mujoco.MjModel.from_xml_path(str(args.gmr_g1_mjcf))
    if gmr_model.nq != 36 or gmr_model.nv != 35 or gmr_model.nu != 29:
        raise ValueError("GMR robot MJCF is not the expected free-base 29-DoF G1")
    reference_qpos, reference_qvel, reference_times = _load_gmr_reference(
        args.motion_pkl,
        args.source_fps,
        args.control_fps,
    )
    reference_joint_names = _joint_names(gmr_model)
    # GMR names the free root ``pelvis`` while the OpenTrack environment names
    # the same 7-DoF free joint ``root``.  The name belongs to the trajectory
    # schema, so use the tracker name here; the 29 actuated names remain
    # strictly matched by ``_assert_joint_contract`` below.
    reference_joint_names[0] = _read_free_joint_name(args.tracker_root_mjcf)
    _write_opentrack_reference(
        args.reference_npz,
        reference_qpos,
        reference_qvel,
        args.control_fps,
        gmr_model,
        reference_joint_names,
    )

    # These settings must precede JAX/MuJoCo-GL imports, matching OpenTrack's evaluator.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sys.path.insert(0, str(tracker_repo))
    os.chdir(tracker_repo)
    import onnxruntime as ort  # pylint: disable=import-outside-toplevel
    import track_mj as tmj  # pylint: disable=import-outside-toplevel
    from track_mj.envs.g1_tracking_dagger.play.play_g1_env_tracking_general import (  # pylint: disable=import-outside-toplevel
        PlayG1TrackingGeneralEnv,
    )
    from track_mj.learning.models.dagger.policy import get_policy_onnx  # pylint: disable=import-outside-toplevel
    from track_mj.learning.models.dagger.policy_args import ONNXPolicyArgs, PolicyArgs  # pylint: disable=import-outside-toplevel

    checkpoint = json.loads(args.checkpoint_config.read_text(encoding="utf-8"))
    dataset_dir = args.reference_npz.parent.parent
    expected_dataset_root = tracker_repo / "storage" / "data" / "mocap"
    try:
        dataset_name = str(dataset_dir.relative_to(expected_dataset_root))
    except ValueError as error:
        raise ValueError(
            "reference_npz must be placed under OpenTrack/storage/data/mocap/<dataset>/UnitreeG1"
        ) from error
    if args.reference_npz.parent.name != "UnitreeG1":
        raise ValueError("reference_npz parent directory must be named UnitreeG1")
    motion_name = args.reference_npz.stem

    checkpoint_env = copy.deepcopy(checkpoint["env_config"])
    checkpoint_env["reference_traj_config"]["name"] = {dataset_name: [motion_name]}
    task_cfg = tmj.registry.get(args.task, "tracking_config")
    env_cfg = task_cfg.env_config
    env_cfg.reference_traj_config.name = checkpoint_env["reference_traj_config"]["name"]
    env_cfg.update(checkpoint_env)
    policy_args = PolicyArgs.from_config_dict(checkpoint["policy_config"]["policy_args"])
    if args.control_fps <= 0.0 or not np.isclose(env_cfg.sim_dt * int(1.0 / env_cfg.sim_dt / args.control_fps), 1.0 / args.control_fps):
        raise ValueError(
            f"requested control_fps={args.control_fps} is not an integer multiple of sim_dt={env_cfg.sim_dt}"
        )

    env_class = tmj.registry.get(args.task, "tracking_dagger_play_env_class")
    env: PlayG1TrackingGeneralEnv = env_class(
        terrain_type=env_cfg.terrain_type,
        config=env_cfg,
        play_ref_motion=False,
        use_viewer=False,
        use_renderer=False,
        exp_name="gmr_floor_only_physical_gate",
    )
    # OpenTrack assembles the final model from ``g1_mjx.xml`` and its terrain;
    # loading that robot XML in isolation is invalid because its collision list
    # intentionally references the environment floor.  Verify against this
    # assembled physical model, not against a hand-built stand-in.
    _assert_joint_contract(gmr_model, env.mj_model)
    loaded_reference_qpos = np.asarray(env.th.traj.data.qpos, dtype=np.float64)
    if loaded_reference_qpos.shape != reference_qpos.shape or not np.allclose(
        loaded_reference_qpos, reference_qpos, atol=1e-8, rtol=0.0
    ):
        max_difference = (
            float(np.max(np.abs(loaded_reference_qpos - reference_qpos)))
            if loaded_reference_qpos.shape == reference_qpos.shape
            else None
        )
        raise RuntimeError(
            "OpenTrack altered the generated reference qpos before physics; "
            f"shape={loaded_reference_qpos.shape} expected={reference_qpos.shape} max_abs={max_difference}"
        )
    policy = get_policy_onnx(
        ONNXPolicyArgs(
            onnx_dir=str(args.onnx),
            **vars(policy_args),
        )
    )
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].shape[-1] != 156 or session.get_outputs()[0].shape[-1] != 29:
        raise ValueError("unexpected OpenTrack v2 ONNX tensor dimensions")

    # This reset performs the single permitted physical qpos/qvel initialization.
    state = env.reset()
    actual_qpos: list[np.ndarray] = []
    actual_qvel: list[np.ndarray] = []
    reference_qpos_log: list[np.ndarray] = []
    reference_qvel_log: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    applied_torque: list[np.ndarray] = []
    control_frames = env.th.traj.data.qpos.shape[0] - env.th.n_trajectories - 1
    if control_frames <= 0:
        raise RuntimeError("reference trajectory contains no executable control frames")

    for _ in range(control_frames):
        ref_qpos, ref_qvel = env.th.get_current_traj_data_fast(state.info["traj_info"])
        observation = state.obs[policy_args.policy_obs_key].reshape(1, -1).astype(np.float32)
        action = np.asarray(policy.infer({policy_args.policy_obs_key: observation})[0], dtype=np.float64)
        if action.shape != (29,) or not np.isfinite(action).all():
            raise RuntimeError(f"invalid policy action shape/value: {action.shape}")
        state = env.step(state, action)
        actual_qpos.append(env.mj_data.qpos.copy())
        actual_qvel.append(env.mj_data.qvel.copy())
        reference_qpos_log.append(np.asarray(ref_qpos, dtype=np.float64).copy())
        reference_qvel_log.append(np.asarray(ref_qvel, dtype=np.float64).copy())
        actions.append(action)
        applied_torque.append(env.mj_data.ctrl.copy())
    env.close()

    actual_qpos_array = np.stack(actual_qpos)
    actual_qvel_array = np.stack(actual_qvel)
    reference_qpos_array = np.stack(reference_qpos_log)
    reference_qvel_array = np.stack(reference_qvel_log)
    root_error = np.linalg.norm(actual_qpos_array[:, :3] - reference_qpos_array[:, :3], axis=1)
    root_step = np.linalg.norm(np.diff(actual_qpos_array[:, :3], axis=0), axis=1)
    finite = bool(np.isfinite(actual_qpos_array).all() and np.isfinite(actual_qvel_array).all())
    min_height = float(np.min(actual_qpos_array[:, 2]))
    max_step = float(np.max(root_step)) if root_step.size else 0.0
    stable = bool(finite and min_height >= args.root_height_floor and max_step <= args.max_root_step)
    summary = {
        "schema_version": 1,
        "purpose": "frozen_opentrack_floor_only_physical_gate",
        "status": "measured_not_promoted",
        "physics_contract": {
            "physical_qpos_qvel_initialization_count": 1,
            "physical_qpos_qvel_writes_after_initialization": 0,
            "external_body_force_writes": 0,
            "controller": "frozen_opentrack_lafan1_v2_onnx_residual_action",
            "physics_advance": "MuJoCo mj_step only after initialization",
            "reference_model_is_kinematic_observation_ghost": True,
        },
        "inputs": {
            "motion_pkl": str(args.motion_pkl),
            "gmr_g1_mjcf": str(args.gmr_g1_mjcf),
            "tracker_root_mjcf": str(args.tracker_root_mjcf),
            "reference_free_joint_name": reference_joint_names[0],
            "tracker_model": "OpenTrack assembled G1 + floor environment",
            "checkpoint_config": str(args.checkpoint_config),
            "onnx": str(args.onnx),
            "reference_npz": str(args.reference_npz),
            "source_fps": args.source_fps,
            "control_fps": args.control_fps,
        },
        "frames": int(len(actual_qpos_array)),
        "finite": finite,
        "root_tracking_rmse_m": float(np.sqrt(np.mean(root_error**2))),
        "root_tracking_p95_m": float(np.quantile(root_error, 0.95)),
        "root_height_min_m": min_height,
        "root_height_max_m": float(np.max(actual_qpos_array[:, 2])),
        "root_step_max_m": max_step,
        "first_root_below_floor_frame": _first_frame_below(actual_qpos_array[:, 2], args.root_height_floor),
        "stable_floor_gate": stable,
        "trajectory_npz": str(args.output_npz),
        "source_duration_s": float(reference_times[-1]),
    }
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        time=np.arange(len(actual_qpos_array), dtype=np.float64) / args.control_fps,
        actual_qpos=actual_qpos_array,
        actual_qvel=actual_qvel_array,
        reference_qpos=reference_qpos_array,
        reference_qvel=reference_qvel_array,
        policy_action=np.stack(actions),
        applied_torque=np.stack(applied_torque),
    )
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
