#!/usr/bin/env python3
"""Run SONIC's official MuJoCo bridge without hidden state resets.

The released SONIC ``DefaultEnv`` generates official Unitree PD torques and
advances MuJoCo with ``mj_step``.  Its interactive-demo fall handler calls
``mj_resetData`` when the pelvis falls below 0.2 m, which is unsuitable for
offline physics validation because it creates an artificial height jump.

This adapter changes only that failure policy: a fall is recorded and ends the
run.  It never writes qpos/qvel after the optional one-time initialisation and
never applies external forces.  All controls remain those received through the
official SONIC Unitree DDS bridge.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from types import MethodType
from typing import Any

import mujoco
import numpy as np
import yaml

from gear_sonic.utils.mujoco_sim.base_sim import BaseSimulator, GEAR_SONIC_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--initial-motion", type=Path)
    parser.add_argument("--initial-frame", type=int, default=0)
    parser.add_argument("--command-wait-s", type=float, default=30.0,
                        help="Wall-clock limit while waiting for SONIC's first LowCmd before mj_step.")
    parser.add_argument(
        "--controller-warmup-s",
        type=float,
        default=0.0,
        help="Additional wall-clock time to publish LowState before the first mj_step.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("SONIC config must contain a top-level YAML mapping")
    if config.get("ENABLE_ELASTIC_BAND", False):
        raise ValueError("ENABLE_ELASTIC_BAND must be false for physics validation")
    if float(config.get("SIMULATE_DT", 0.0)) <= 0.0:
        raise ValueError("SIMULATE_DT must be positive")
    scene = Path(str(config.get("ROBOT_SCENE", "")))
    if not scene.is_absolute():
        scene = GEAR_SONIC_ROOT / scene
    if not scene.is_file():
        raise FileNotFoundError(f"ROBOT_SCENE does not exist: {scene}")
    config["ROBOT_SCENE"] = str(scene.resolve())
    return config


def mark_fall_without_reset(env: Any) -> None:
    """Match SONIC's fall threshold but never invoke ``mj_resetData``."""
    env.fall = bool(env.mj_data.qpos[2] < 0.2)


def initialise_once(simulator: BaseSimulator, motion_path: Path, frame: int) -> dict[str, Any]:
    """Set one source frame before the first ``mj_step`` and clear velocity."""
    if not motion_path.is_file():
        raise FileNotFoundError(motion_path)
    with motion_path.open("rb") as handle:
        motion = pickle.load(handle)
    required = ("root_pos", "root_rot", "dof_pos", "dof_names")
    missing = [name for name in required if name not in motion]
    if missing:
        raise ValueError(f"initial motion misses fields: {missing}")

    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_rot_xyzw = np.asarray(motion["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
    dof_names = list(motion["dof_names"])
    if not 0 <= frame < len(root_pos):
        raise IndexError(f"initial frame {frame} is outside [0, {len(root_pos)})")
    if root_rot_xyzw.shape != (len(root_pos), 4) or dof_pos.shape[0] != len(root_pos):
        raise ValueError("inconsistent initial GMR motion arrays")

    env = simulator.sim_env
    model, data = env.mj_model, env.mj_data
    if data.qpos.size < 7:
        raise ValueError("scene has no floating-base qpos layout")
    data.qpos[:3] = root_pos[frame]
    # GMR serialises xyzw; MuJoCo requires wxyz.
    data.qpos[3:7] = root_rot_xyzw[frame, (3, 0, 1, 2)]
    for name, value in zip(dof_names, dof_pos[frame]):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"scene has no GMR joint {name!r}")
        data.qpos[model.jnt_qposadr[joint_id]] = value
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return {"motion": str(motion_path.resolve()), "frame": frame}


def record_step(simulator: BaseSimulator, records: dict[str, list[np.ndarray]]) -> None:
    env = simulator.sim_env
    records["time"].append(np.array(env.mj_data.time, dtype=np.float64))
    records["qpos"].append(env.mj_data.qpos.copy())
    records["qvel"].append(env.mj_data.qvel.copy())
    records["ctrl"].append(env.mj_data.ctrl.copy())
    records["xfrc_applied"].append(env.mj_data.xfrc_applied.copy())
    records["ncon"].append(np.array(env.mj_data.ncon, dtype=np.int32))


def wait_for_controller(simulator: BaseSimulator, timeout_s: float, warmup_s: float) -> float:
    """Wait for a command, then let SONIC finish its wall-clock startup phase.

    The released controller's INIT phase advances in wall-clock time while
    reading LowState.  Publishing the unchanged initial state during that
    phase is intentional: no MuJoCo state is advanced or overwritten before
    the learned controller has entered CONTROL.
    """
    if timeout_s <= 0.0:
        raise ValueError("--command-wait-s must be positive")
    env = simulator.sim_env
    deadline = time.monotonic() + timeout_s
    started = time.monotonic()
    # `low_cmd` itself is allocated when the bridge is constructed.  The
    # receipt flag, rather than object existence, is the only valid DDS
    # handshake: it proves a controller has consumed LowState and published
    # at least one command on rt/lowcmd.
    while not env.unitree_bridge.cmd_received():
        # This mirrors the state publication at the beginning of sim_step,
        # deliberately without computing torques or advancing MuJoCo.
        env.unitree_bridge.PublishLowState(env.prepare_obs())
        if time.monotonic() >= deadline:
            raise TimeoutError("SONIC did not publish a LowCmd before the command-wait deadline")
        time.sleep(min(float(simulator.config["SIMULATE_DT"]), 0.01))
    warmup_deadline = time.monotonic() + warmup_s
    while time.monotonic() < warmup_deadline:
        env.unitree_bridge.PublishLowState(env.prepare_obs())
        time.sleep(min(float(simulator.config["SIMULATE_DT"]), 0.01))
    return time.monotonic() - started


def write_result(
    output_dir: Path,
    records: dict[str, list[np.ndarray]],
    config: dict[str, Any],
    initialisation: dict[str, Any] | None,
    command_handshake_s: float,
    controller_warmup_s: float,
    termination: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.asarray(values) for name, values in records.items()}
    np.savez_compressed(output_dir / "sonic_physics_trace.npz", **arrays)
    root = arrays["qpos"][:, :3] if len(arrays["qpos"]) else np.empty((0, 3))
    root_steps = np.linalg.norm(np.diff(root, axis=0), axis=1) if len(root) > 1 else np.empty(0)
    max_external_wrench = (
        float(np.max(np.abs(arrays["xfrc_applied"]))) if len(arrays["xfrc_applied"]) else 0.0
    )
    report = {
        "status": termination,
        "steps": int(len(arrays["time"])),
        "scene": config["ROBOT_SCENE"],
        "simulate_dt_s": float(config["SIMULATE_DT"]),
        "physics_contract": {
            "integrator": "mujoco.mj_step",
            "controller": "official_sonic_unitree_dds_pd",
            "elastic_band": False,
            "external_wrenches_after_start": max_external_wrench,
            "qpos_qvel_writes_after_initialisation": 0,
            "fall_response": "terminate_without_mj_resetData",
        },
        "initialisation": initialisation,
        "command_handshake_s": command_handshake_s,
        "controller_warmup_s": controller_warmup_s,
        "root_height_min_m": float(np.min(root[:, 2])) if len(root) else None,
        "root_height_max_m": float(np.max(root[:, 2])) if len(root) else None,
        "max_root_step_m": float(np.max(root_steps)) if len(root_steps) else 0.0,
        "contact_count_max": int(np.max(arrays["ncon"])) if len(arrays["ncon"]) else 0,
    }
    (output_dir / "sonic_physics_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def main() -> None:
    args = parse_args()
    if args.duration_s <= 0.0:
        raise ValueError("--duration-s must be positive")
    if args.controller_warmup_s < 0.0:
        raise ValueError("--controller-warmup-s must be non-negative")
    config = load_config(args.config)
    simulator = BaseSimulator(config=config, env_name="default", onscreen=False, offscreen=False)
    # SONIC only creates this attribute when the interactive elastic band is
    # enabled, while ``sim_step`` reads it unconditionally.
    if not hasattr(simulator.sim_env, "elastic_band"):
        simulator.sim_env.elastic_band = None
    simulator.sim_env.check_fall = MethodType(mark_fall_without_reset, simulator.sim_env)

    initialisation = None
    if args.initial_motion is not None:
        initialisation = initialise_once(simulator, args.initial_motion, args.initial_frame)

    records: dict[str, list[np.ndarray]] = {
        "time": [], "qpos": [], "qvel": [], "ctrl": [], "xfrc_applied": [], "ncon": []
    }
    termination = "duration_reached"
    command_handshake_s = 0.0
    try:
        command_handshake_s = wait_for_controller(
            simulator, args.command_wait_s, args.controller_warmup_s
        )
    except TimeoutError:
        termination = "command_handshake_timeout"
    else:
        deadline = time.monotonic() + args.duration_s
        while time.monotonic() < deadline:
            step_start = time.monotonic()
            simulator.sim_env.sim_step()
            record_step(simulator, records)
            if simulator.sim_env.fall:
                termination = "fall_terminated_without_reset"
                break
            sleep_time = float(config["SIMULATE_DT"]) - (time.monotonic() - step_start)
            if sleep_time > 0.0:
                time.sleep(sleep_time)
    finally:
        write_result(
            args.output_dir,
            records,
            config,
            initialisation,
            command_handshake_s,
            args.controller_warmup_s,
            termination,
        )
        simulator.close()


if __name__ == "__main__":
    main()
