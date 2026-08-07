#!/usr/bin/env python3
"""Run an official VideoMimic policy on one reconstructed G1 reference.

The bridge has strict ownership: it writes only an Isaac Gym rollout artifact.
It does not alter GVHMR, PHC, GMR, the reconstructed scene, or the reference H5.
The caller supplies the current clip's retargeted G1 reference and scene mesh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


class ContractError(RuntimeError):
    """Raised when a reusable VideoMimic policy contract is not satisfied."""


def _prepare_videomimic_runtime() -> None:
    """Make direct interpreter invocation behave like an activated VM environment."""
    prefix = Path(sys.executable).resolve().parent.parent
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["PATH"] = str(prefix / "bin") + os.pathsep + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = str(prefix / "lib") + os.pathsep + os.environ.get(
        "LD_LIBRARY_PATH", ""
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_contract(policy_state: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    """Compare all state entries before requiring PyTorch's strict load."""
    import torch

    payload = torch.load(checkpoint, map_location="cpu")
    saved = payload.get("model_state_dict") if isinstance(payload, dict) else None
    if not isinstance(saved, dict):
        raise ContractError(f"checkpoint has no model_state_dict: {checkpoint}")
    expected_keys = set(policy_state)
    saved_keys = set(saved)
    missing = sorted(expected_keys - saved_keys)
    unexpected = sorted(saved_keys - expected_keys)
    shape_mismatches = {
        key: {"checkpoint": list(saved[key].shape), "policy": list(policy_state[key].shape)}
        for key in sorted(expected_keys & saved_keys)
        if tuple(saved[key].shape) != tuple(policy_state[key].shape)
    }
    rejected = bool(missing or unexpected or shape_mismatches)
    return {
        "status": "rejected" if rejected else "pass",
        "strict_load_required": True,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_iteration": payload.get("iter") if isinstance(payload, dict) else None,
        "policy_key_count": len(expected_keys),
        "checkpoint_key_count": len(saved_keys),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-h5", required=True, type=Path)
    parser.add_argument("--terrain-obj", required=True, type=Path)
    parser.add_argument("--policy-run", required=True)
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=-1,
        help="Checkpoint iteration to load; -1 selects the latest checkpoint in --policy-run.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument(
        "--task",
        choices=("g1_deepmimic_root_heightfield_no_history_dagger",),
        default="g1_deepmimic_root_heightfield_no_history_dagger",
    )
    parser.add_argument("--motion-name", default="motion")
    parser.add_argument("--max-link-error-m", type=float, default=0.30)
    parser.add_argument("--max-root-error-m", type=float, default=0.35)
    parser.add_argument("--allow-resets", type=int, default=0)
    parser.add_argument("--termination-link-error-m", type=float, default=10.0)
    parser.add_argument("--respawn-z-offset-m", type=float, default=0.1)
    parser.add_argument("--action-scale", type=float)
    parser.add_argument("--arm-stiffness-scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.max_link_error_m <= 0.0 or args.max_root_error_m <= 0.0:
        parser.error("tracking-error thresholds must be positive")
    if args.termination_link_error_m <= 0.0 or not np.isfinite(args.respawn_z_offset_m):
        parser.error("termination-link-error must be positive and respawn-z-offset finite")
    if args.action_scale is not None and (
        args.action_scale <= 0.0 or not np.isfinite(args.action_scale)
    ):
        parser.error("--action-scale must be positive and finite when provided")
    if args.arm_stiffness_scale <= 0.0 or not np.isfinite(args.arm_stiffness_scale):
        parser.error("--arm-stiffness-scale must be positive and finite")
    if args.allow_resets < 0:
        parser.error("--allow-resets must be non-negative")
    if args.steps < 0:
        parser.error("--steps must be non-negative; zero selects the reference length")
    if args.checkpoint < -1:
        parser.error("--checkpoint must be -1 or a non-negative iteration")
    if not args.motion_name or "/" in args.motion_name or "\\" in args.motion_name:
        parser.error("--motion-name must be one safe directory name")
    return args


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ContractError(f"missing {label}: {resolved}")
    return resolved


def _link_exact(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source:
            raise ContractError(
                f"refusing to mix a different input at {destination}; "
                f"expected {source}, found {destination.resolve()}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source)

def _materialize_rl_reference(source: Path, destination: Path) -> None:
    """Copy a G1 H5 and add only the public RL loader's metadata aliases."""
    import h5py
    import shutil

    if destination.is_symlink():
        if destination.resolve() != source:
            raise ContractError(
                f"refusing to replace unrelated generated reference: {destination}"
            )
        destination.unlink()
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    with h5py.File(destination, "r+") as archive:
        for plain_name in ("fps", "joint_names", "link_names"):
            slash_name = f"/{plain_name}"
            if slash_name not in archive.attrs:
                if plain_name not in archive.attrs:
                    raise ContractError(
                        f"G1 reference lacks required metadata: {plain_name}"
                    )
                archive.attrs[slash_name] = archive.attrs[plain_name]


def _vm_args(task: str) -> Any:
    # Isaac Gym must be imported before torch.  Importing it here also makes the
    # requirement explicit for callers running this bridge outside conda.
    _prepare_videomimic_runtime()
    import isaacgym  # noqa: F401
    from legged_gym.utils import get_args

    original = sys.argv[:]
    sys.argv = [
        original[0],
        "--task",
        task,
        "--num_envs",
        "1",
        "--headless",
    ]
    try:
        args, unknown = get_args()
    finally:
        sys.argv = original
    if unknown:
        raise ContractError(f"unexpected VideoMimic arguments: {unknown}")
    return args


def _to_numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def _reference_step_count(env: Any) -> int:
    lengths = getattr(env.replay_data_loader, "sequence_lengths", None)
    if lengths is None or len(lengths) == 0:
        raise ContractError("VideoMimic replay loader did not expose sequence_lengths")
    value = lengths[0]
    return int(value.item() if hasattr(value, "item") else value)


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    reference_h5 = _require_file(args.reference_h5, "G1 reference H5")
    terrain_obj = _require_file(args.terrain_obj, "terrain OBJ")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ContractError(f"output already exists; choose a new rollout name: {output}")

    workspace = output.parent / f"policy_input_{output.stem}"
    motion_dir = workspace / args.motion_name
    _materialize_rl_reference(reference_h5, motion_dir / "retarget_poses_g1.h5")
    _link_exact(terrain_obj, motion_dir / "background_mesh.obj")

    vm_args = _vm_args(args.task)
    __import__("legged_gym.envs")
    from legged_gym.utils import task_registry

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.env.test = True
    env_cfg.viser.enabled = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.cast_mesh_to_heightfield = False
    env_cfg.deepmimic.use_amass = False
    env_cfg.deepmimic.use_human_videos = True
    env_cfg.deepmimic.data_root = str(workspace)
    env_cfg.deepmimic.alt_data_root = ""
    env_cfg.deepmimic.human_motion_source = args.motion_name
    env_cfg.deepmimic.randomize_start_offset = False
    env_cfg.deepmimic.respawn_z_offset = args.respawn_z_offset_m
    env_cfg.deepmimic.link_pos_error_threshold = args.termination_link_error_m
    env_cfg.deepmimic.viz_replay = False
    if args.action_scale is not None:
        env_cfg.control.action_scale = args.action_scale
    if args.arm_stiffness_scale != 1.0:
        damping_scale = float(np.sqrt(args.arm_stiffness_scale))
        for joint_group in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow"):
            env_cfg.control.stiffness[joint_group] *= args.arm_stiffness_scale
            env_cfg.control.damping[joint_group] *= damping_scale

    # Build the exact task policy first, then compare every checkpoint tensor
    # and use the runner's strict loader. The library default is strict=False.
    train_cfg.runner.resume = False
    train_cfg.runner.load_model_strict = True
    train_cfg.runner.use_wandb = False
    train_cfg.algorithm.bc_loss_coef = 0.0

    env, _ = task_registry.make_env(name=args.task, args=vm_args, env_cfg=env_cfg)
    runner, loaded_train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=vm_args, train_cfg=train_cfg
    )
    from legged_gym import LEGGED_GYM_ROOT_DIR
    from legged_gym.utils.helpers import get_load_path

    checkpoint_root = Path(LEGGED_GYM_ROOT_DIR) / "logs" / str(
        loaded_train_cfg.runner.experiment_name
    )
    checkpoint_path = Path(
        get_load_path(
            str(checkpoint_root), load_run=args.policy_run, checkpoint=args.checkpoint
        )
    ).resolve()
    checkpoint_contract = _checkpoint_contract(
        runner.alg.actor_critic.state_dict(), checkpoint_path
    )
    checkpoint_report = output.with_suffix(".checkpoint.json")
    _write_report(checkpoint_report, checkpoint_contract)
    if checkpoint_contract["status"] != "pass":
        raise ContractError(
            "checkpoint is incompatible with the selected official VideoMimic "
            f"policy; see {checkpoint_report}"
        )
    runner.load(str(checkpoint_path), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    reference_step_count = _reference_step_count(env)
    if args.steps:
        step_count = args.steps
        terminal_frame_omitted = False
    else:
        if reference_step_count <= 1:
            raise ContractError(
                f"reference is too short for a no-reset rollout: {reference_step_count}"
            )
        # IsaacGym resets exactly when the final reference frame is consumed.
        # That reset state is not a physically simulated reference sample and
        # must not be counted as a failed motion frame.
        step_count = reference_step_count - 1
        terminal_frame_omitted = True
    if step_count <= 0:
        raise ContractError(f"invalid resolved rollout length: {step_count}")

    observations = env.get_observations()
    root_states: list[np.ndarray] = []
    dof_positions: list[np.ndarray] = []
    target_root_positions: list[np.ndarray] = []
    target_dof_positions: list[np.ndarray] = []
    body_positions: list[np.ndarray] = []
    body_quaternions: list[np.ndarray] = []
    rewards: list[float] = []
    done_flags: list[bool] = []
    timeout_flags: list[bool] = []
    root_tracking_errors: list[float] = []
    max_link_tracking_errors: list[float] = []
    expected_terminal_timeout_count = 0
    unexpected_done_count = 0
    tracked_link_names = np.asarray(list(getattr(env, "tracked_body_names", getattr(env.cfg.asset, "tracked_body_names", ()))), dtype=str)
    if tracked_link_names.shape != (int(env.link_pos_error.shape[1]),):
        raise ContractError(f"unexpected tracked-link contract: names={tracked_link_names.shape}, link_error={tuple(env.link_pos_error.shape)}")
    link_tracking_errors: list[np.ndarray] = []
    for step_index in range(step_count):
        actions = policy(
            {key: value.detach() for key, value in observations.items()},
            monitor_activations=False,
        )
        observations, reward, done, _ = env.step(actions.detach())
        done_flag = bool(_to_numpy(done)[0])
        timeout_flag = bool(_to_numpy(env.time_out_buf)[0])
        is_expected_terminal_timeout = (
            terminal_frame_omitted
            and step_index == step_count - 1
            and done_flag
            and timeout_flag
        )
        if is_expected_terminal_timeout:
            # Isaac Gym has already reset this final state. It has no matching
            # physical reference sample, so exclude it from replay metrics.
            expected_terminal_timeout_count += 1
            break
        root_states.append(_to_numpy(env.root_states[0]))
        dof_positions.append(_to_numpy(env.dof_pos[0]))
        target_root_positions.append(_to_numpy(env.target_root_pos[0]))
        target_dof_positions.append(_to_numpy(env.target_dofs[0]))
        body_positions.append(_to_numpy(env.env_rigid_body_pos[0]))
        body_quaternions.append(_to_numpy(env.rigid_body_quat[0]))
        rewards.append(float(_to_numpy(reward)[0]))
        done_flags.append(done_flag)
        timeout_flags.append(timeout_flag)
        root_tracking_errors.append(float(np.linalg.norm(_to_numpy(env.env_root_pos[0] - env.target_root_pos[0]))))
        link_error = _to_numpy(env.link_pos_error[0])
        link_norm = np.linalg.norm(link_error, axis=-1).astype(np.float32)
        link_tracking_errors.append(link_norm)
        max_link_tracking_errors.append(float(link_norm.max()))
        if done_flag:
            # Any other reset is an early failure. Do not continue with the
            # automatically respawned state and make the shortened rollout explicit.
            unexpected_done_count += 1
            break

    if not root_states:
        raise ContractError("Isaac Gym rollout produced no physical reference samples")

    link_tracking = np.stack(link_tracking_errors).astype(np.float32)
    if not np.isfinite(link_tracking).all():
        raise ContractError("Isaac Gym rollout has non-finite per-link tracking metrics")
    roots = np.stack(root_states).astype(np.float32)
    dofs = np.stack(dof_positions).astype(np.float32)
    if not np.isfinite(roots).all() or not np.isfinite(dofs).all():
        raise ContractError("Isaac Gym rollout contains NaN or Inf")
    root_tracking = np.asarray(root_tracking_errors, dtype=np.float32)
    max_link_tracking = np.asarray(max_link_tracking_errors, dtype=np.float32)
    if not np.isfinite(root_tracking).all() or not np.isfinite(max_link_tracking).all():
        raise ContractError("Isaac Gym rollout has non-finite tracking metrics")
    dof_names = np.asarray(list(getattr(env, "dof_names", ())), dtype=str)
    body_names = np.asarray(list(getattr(env, "body_names", ())), dtype=str)
    if body_names.shape != (body_positions[0].shape[0],):
        raise ContractError(
            f"unexpected body-name contract: names={body_names.shape}, positions={body_positions[0].shape}"
        )
    if dof_names.shape != (dofs.shape[1],):
        raise ContractError(
            f"unexpected DOF-name contract: names={dof_names.shape}, dofs={dofs.shape}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        root_states=roots,
        dof_positions=dofs,
        dof_names=dof_names,
        target_root_positions=np.stack(target_root_positions).astype(np.float32),
        target_dof_positions=np.stack(target_dof_positions).astype(np.float32),
        actual_body_positions=np.stack(body_positions).astype(np.float32),
        actual_body_quaternions=np.stack(body_quaternions).astype(np.float32),
        body_names=body_names,
        rewards=np.asarray(rewards, dtype=np.float32),
        done=np.asarray(done_flags, dtype=np.bool_),
        time_out=np.asarray(timeout_flags, dtype=np.bool_),
        tracked_link_names=tracked_link_names,
        link_tracking_error_by_link_m=link_tracking,
        root_tracking_error_m=root_tracking,
        max_link_tracking_error_m=max_link_tracking,
    )
    done_count = int(np.count_nonzero(done_flags))
    timeout_count = int(np.count_nonzero(timeout_flags))
    termination_count = done_count - timeout_count
    root_error_p95 = float(np.quantile(root_tracking, 0.95))
    root_error_max = float(root_tracking.max())
    link_error_p95 = float(np.quantile(max_link_tracking, 0.95))
    link_error_max = float(max_link_tracking.max())
    rejection_reasons: list[str] = []
    if unexpected_done_count > args.allow_resets:
        rejection_reasons.append("episode_reset")
    if root_error_max > args.max_root_error_m:
        rejection_reasons.append("root_tracking_error")
    if link_error_max > args.max_link_error_m:
        rejection_reasons.append("link_tracking_error")
    report = {
        "status": "pass" if not rejection_reasons else "rejected",
        "execution_contract": "official_videomimic_rl_isaacgym",
        "policy_run": args.policy_run,
        "task": args.task,
        "checkpoint_contract": checkpoint_contract,
        "execution_parameters": {
            "requested_checkpoint": int(args.checkpoint),
            "termination_link_error_m": float(args.termination_link_error_m),
            "respawn_z_offset_m": float(args.respawn_z_offset_m),
            "action_scale": float(env_cfg.control.action_scale),
            "arm_stiffness_scale": float(args.arm_stiffness_scale),
        },
        "steps": int(len(root_states)),
        "requested_steps": int(step_count),
        "reference_steps": int(reference_step_count),
        "expected_terminal_timeout_count": int(expected_terminal_timeout_count),
        "unexpected_done_count": int(unexpected_done_count),
        "terminal_reference_frame_omitted": terminal_frame_omitted,
        "reference_h5": str(reference_h5),
        "terrain_obj": str(terrain_obj),
        "workspace": str(workspace),
        "output": str(output),
        "root_z_min_m": float(roots[:, 2].min()),
        "root_xy_displacement_m": float(np.linalg.norm(roots[-1, :2] - roots[0, :2])),
        "per_link": {
            str(name): {"p95_m": float(np.quantile(link_tracking[:, index], 0.95)), "max_m": float(link_tracking[:, index].max())}
            for index, name in enumerate(tracked_link_names)
        },
        "done_count": done_count,
        "timeout_count": timeout_count,
        "termination_count": termination_count,
        "quality_gate": {
            "accepted": not rejection_reasons,
            "rejection_reasons": rejection_reasons,
            "allow_resets": int(args.allow_resets),
            "termination_link_error_m": float(args.termination_link_error_m),
            "max_root_error_allowed_m": float(args.max_root_error_m),
            "max_link_error_allowed_m": float(args.max_link_error_m),
            "root_error_p95_m": root_error_p95,
            "root_error_max_m": root_error_max,
            "link_error_p95_m": link_error_p95,
            "link_error_max_m": link_error_max,
        },
        "reward_mean": float(np.mean(rewards)),
        "dof_count": int(dofs.shape[1]),
        "dof_names": dof_names.tolist(),
    }
    _write_report(output.with_suffix(".json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
