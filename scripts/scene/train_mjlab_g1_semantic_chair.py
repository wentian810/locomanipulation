#!/usr/bin/env python3
"""Train Mjlab's existing G1 tracker in a certified semantic-chair scene.

The only scene-specific code here is a small MJCF-spec adapter: it injects the
already certified six collision primitives into Mjlab's normal G1 tracking
task.  Dynamics, actuator limits, policy optimisation and ``mj_step`` remain
in Unitree RL Mjlab/MuJoCo; this script never applies a base wrench or rewrites
qpos during rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import torch


REQUIRED_CHAIR_COMPONENTS = (
    "seat_support",
    "backrest",
    "leg_front_left",
    "leg_front_right",
    "leg_back_left",
    "leg_back_right",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_certified_chair(path: Path) -> tuple[dict, ...]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("frame") != "mujoco_world_z_up":
        raise ValueError("chair primitives must be expressed in mujoco_world_z_up")
    by_name = {item.get("name"): item for item in document.get("primitives", [])}
    missing = [name for name in REQUIRED_CHAIR_COMPONENTS if name not in by_name]
    if missing:
        raise ValueError("chair contract missing components: %s" % missing)
    chair = []
    for name in REQUIRED_CHAIR_COMPONENTS:
        item = by_name[name]
        if item.get("type") != "box":
            raise ValueError("chair component %s is not a box" % name)
        center = np.asarray(item.get("center"), dtype=np.float64)
        extents = np.asarray(item.get("extents"), dtype=np.float64)
        quaternion = np.asarray(item.get("quat_wxyz"), dtype=np.float64)
        if center.shape != (3,) or extents.shape != (3,) or quaternion.shape != (4,):
            raise ValueError("invalid dimensions for chair component %s" % name)
        if np.any(extents <= 0.0) or not np.isfinite(center).all() or not np.isfinite(quaternion).all():
            raise ValueError("invalid geometry values for chair component %s" % name)
        quaternion /= np.linalg.norm(quaternion)
        chair.append(
            {
                "name": name,
                "center": center,
                "half_extents": 0.5 * extents,
                "quat_wxyz": quaternion,
            }
        )
    return tuple(chair)


def _chair_entity_spec(chair: tuple[dict, ...]):
    """Return an Mjlab Entity spec for the static, collidable chair boxes."""

    def build() -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        body = spec.worldbody.add_body(name="chair_root")
        for component in chair:
            geom = body.add_geom()
            geom.name = "%s_geom" % component["name"]
            geom.type = mujoco.mjtGeom.mjGEOM_BOX
            geom.pos = component["center"].tolist()
            geom.size = component["half_extents"].tolist()
            geom.quat = component["quat_wxyz"].tolist()
            geom.contype = 1
            geom.conaffinity = 1
            geom.condim = 3
            geom.friction = [0.8, 0.005, 0.0001]
            geom.rgba = [0.72, 0.52, 0.30, 1.0]
        return spec

    return build


def _load_true_sit_event(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    events = report.get("physical_transition_events", [])
    if report.get("decision") != "pass" or not events:
        raise ValueError(
            "true-sit gate did not pass; refusing to introduce chair-contact control"
        )
    return report


def _phase_mask(env, command_name: str, frame_ranges: tuple[tuple[int, int], ...]) -> torch.Tensor:
    """Return a per-world true mask from Mjlab's actual motion phase."""
    motion = env.command_manager.get_term(command_name)
    time_steps = motion.time_steps
    mask = torch.zeros_like(time_steps, dtype=torch.bool)
    for start, end in frame_ranges:
        mask |= (time_steps >= start) & (time_steps <= end)
    return mask


def _pre_contact_anchor_position_reward(
    env,
    command_name: str,
    std: float,
    contact_reference_ranges: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    """Track GMR root position only before the reference reaches the chair."""
    from src.tasks.tracking.mdp.rewards import motion_global_anchor_position_error_exp

    tracking = motion_global_anchor_position_error_exp(
        env, command_name=command_name, std=std
    )
    return tracking * (~_phase_mask(env, command_name, contact_reference_ranges))


def _seat_contact_observation(env, sensor_name: str) -> torch.Tensor:
    data = env.scene[sensor_name].data
    assert data.found is not None
    return (data.found > 0).any(dim=1, keepdim=True).float()


def _settled_seat_contact_reward(
    env,
    command_name: str,
    sensor_name: str,
    support_ranges: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    """Reward contact existence only; gravity, not the reward, sets the load."""
    data = env.scene[sensor_name].data
    assert data.found is not None
    hip_or_pelvis_is_supported = (data.found > 0).any(dim=1).float()
    return _phase_mask(env, command_name, support_ranges).float() * hip_or_pelvis_is_supported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjlab-repo", type=Path, required=True)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--chair-primitives", type=Path, required=True)
    parser.add_argument("--true-sit-event", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--steps-per-env", type=int, default=4)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()
    if args.num_envs < 1 or args.max_iterations < 1 or args.steps_per_env < 1:
        raise ValueError("num-envs, max-iterations and steps-per-env must be positive")
    for path in (
        args.mjlab_repo,
        args.motion_file,
        args.chair_primitives,
        args.true_sit_event,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    true_sit_report = _load_true_sit_event(args.true_sit_event)
    chair = _load_certified_chair(args.chair_primitives)
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(args.mjlab_repo))
    # Task registration normally happens in Unitree's train.py entrypoint.
    # This wrapper calls its reusable function directly, so import the package
    # explicitly before looking the official G1 tracking task up in the registry.
    import src.tasks  # noqa: F401
    from scripts.train import TrainConfig, run_train
    from mjlab.entity import EntityCfg
    from mjlab.managers.observation_manager import ObservationTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.sensor import ContactMatch, ContactSensorCfg

    cfg = TrainConfig.from_task("Unitree-G1-Tracking")
    cfg = replace(cfg, motion_file=str(args.motion_file.resolve()))
    # Each Mjlab world is an independent MuJoCo model.  Keep all world origins
    # at zero so the fixed chair and the GMR world-frame reference retain the
    # already certified relative transform in every parallel world.
    cfg.env.scene.num_envs = args.num_envs
    cfg.env.scene.env_spacing = 0.0
    if cfg.env.scene.terrain is not None:
        cfg.env.scene.terrain.num_envs = args.num_envs
        cfg.env.scene.terrain.env_spacing = 0.0
    # An Entity, rather than a late spec_fn mutation, makes the chair visible
    # to Mjlab contact sensors while preserving the same six collision boxes.
    cfg.env.scene.entities["semantic_chair"] = EntityCfg(
        spec_fn=_chair_entity_spec(chair)
    )
    contact_reference_ranges = tuple(
        tuple(event["near_surface_contact_reference_range"])
        for event in true_sit_report["physical_transition_events"]
    )
    support_ranges = tuple(
        tuple(event["physical_support_validation_range"])
        for event in true_sit_report["physical_transition_events"]
    )
    seat_sensor_name = "hip_pelvis_seat_contact"
    seat_sensor = ContactSensorCfg(
        name=seat_sensor_name,
        primary=ContactMatch(
            mode="body",
            pattern=("pelvis", "left_hip_yaw_link", "right_hip_yaw_link"),
            entity="robot",
        ),
        secondary=ContactMatch(
            mode="geom", pattern="semantic_chair/seat_support_geom"
        ),
        fields=("found", "dist"),
        reduce="mindist",
        num_slots=1,
    )
    cfg.env.scene.sensors = (cfg.env.scene.sensors or ()) + (seat_sensor,)
    # The GMR root is a trajectory reference, not a kinematic constraint.  As
    # soon as the reference reaches the chair's near-contact band, release its
    # world-position reward and let collision plus gravity determine height.
    cfg.env.rewards["motion_global_root_pos"] = RewardTermCfg(
        func=_pre_contact_anchor_position_reward,
        weight=0.5,
        params={
            "command_name": "motion",
            "std": 0.3,
            "contact_reference_ranges": contact_reference_ranges,
        },
    )
    cfg.env.rewards["settled_hip_pelvis_seat_contact"] = RewardTermCfg(
        func=_settled_seat_contact_reward,
        weight=0.25,
        params={
            "command_name": "motion",
            "sensor_name": seat_sensor_name,
            "support_ranges": support_ranges,
        },
    )
    cfg.env.observations["actor"].terms["seat_support_contact"] = ObservationTermCfg(
        func=_seat_contact_observation, params={"sensor_name": seat_sensor_name}
    )
    cfg.env.observations["critic"].terms["seat_support_contact"] = ObservationTermCfg(
        func=_seat_contact_observation, params={"sensor_name": seat_sensor_name}
    )
    cfg.agent.max_iterations = args.max_iterations
    cfg.agent.num_steps_per_env = args.steps_per_env
    cfg.agent.experiment_name = "g1_semantic_chair_tracking"

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = args.output_root / (stamp + "_gmr_semantic_chair")
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "controller": "unitree_rl_mjlab_Unitree-G1-Tracking",
        "motion_reference": str(args.motion_file),
        "motion_reference_sha256": _sha256(args.motion_file),
        "chair_primitives": str(args.chair_primitives),
        "chair_primitives_sha256": _sha256(args.chair_primitives),
        "chair_collision_geoms": [
            "semantic_chair/%s_geom" % name for name in REQUIRED_CHAIR_COMPONENTS
        ],
        "true_sit_event": str(args.true_sit_event),
        "true_sit_event_sha256": _sha256(args.true_sit_event),
        "physical_transition_events": true_sit_report["physical_transition_events"],
        "rollout_contract": {
            "free_base": True,
            "base_wrench": "forbidden",
            "qpos_reset_after_initialisation": "forbidden",
            "stepper": "Mjlab MuJoCo forward dynamics (mj_step equivalent)",
            "chair": "static six-component semantic collision asset",
            "seat_contact_reward": "binary hip/pelvis contact only; no contact-force target or added downward load",
            "root_reference_after_near_contact": "released; chair collision and gravity determine the seated height",
        },
        "training": {
            "num_envs": args.num_envs,
            "max_iterations": args.max_iterations,
            "steps_per_env": args.steps_per_env,
            "gpu_id": args.gpu_id,
        },
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    run_train("Unitree-G1-Tracking", cfg, run_dir)


if __name__ == "__main__":
    main()
