#!/usr/bin/env python3
"""Retarget GVHMR-hand sidecar data to the 22-DoF Sharpa Wave hand.

The body path remains independent.  Hand articulation is solved in target
space: the source MANO palm frame supplies bone directions, while the target
chain uses the Sharpa hand's own segment lengths.  Matching PIP, DIP and tip
sites removes the large null-space left by fingertip-only IK.  Conservative
biomechanical bend checks repair only impossible source frames, while a
reprojection-aware previous-pose task and velocity/acceleration limits suppress
temporal jumps without flattening reliable motion.
"""

from __future__ import annotations

import argparse
import pathlib

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


HERE = pathlib.Path(__file__).resolve().parent
PIPELINE_ROOT = HERE.parents[1]
DEFAULT_SHARPA_ROOT = (
    PIPELINE_ROOT
    / "do-as-i-do-main"
    / "retargeting"
    / "retargeting"
    / "assets"
    / "robots"
    / "sharpa"
)

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
FINGERTIP_IDX = np.asarray([4, 8, 12, 16, 20], dtype=np.int64)
FINGER_CHAINS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
CHAIN_LEVELS = ("pip", "dip", "tip")


def normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    out = quat / np.clip(norm, 1e-12, None)
    out[~np.isfinite(out).all(axis=1)] = np.asarray([1.0, 0.0, 0.0, 0.0])
    return out


def hold_invalid(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.asarray(values).copy()
    valid = np.asarray(valid, dtype=bool)
    if not valid.any():
        return np.zeros_like(out)
    first = int(np.flatnonzero(valid)[0])
    last = out[first].copy()
    out[: first + 1] = last
    for i in range(first + 1, len(out)):
        if valid[i]:
            last = out[i].copy()
        else:
            out[i] = last
    return out


def hold_invalid_quat_wxyz(quat: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = normalize_quat_wxyz(quat)
    valid = np.asarray(valid, dtype=bool)
    if not valid.any():
        out[:] = np.asarray([1.0, 0.0, 0.0, 0.0])
        return out
    first = int(np.flatnonzero(valid)[0])
    last = out[first].copy()
    out[: first + 1] = last
    for i in range(first + 1, len(out)):
        if valid[i]:
            if float(np.dot(out[i], last)) < 0.0:
                out[i] *= -1.0
            last = out[i].copy()
        else:
            out[i] = last
    return out


def smooth_reliability(
    reliability: np.ndarray,
    valid: np.ndarray,
    source_repaired: np.ndarray | None = None,
    window: int = 0,
) -> np.ndarray:
    raw = np.clip(np.asarray(reliability, dtype=np.float64).reshape(-1), 0.0, 1.0)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    if raw.shape[0] != valid.shape[0]:
        raise ValueError(f"reliability has {raw.shape[0]} frames; expected {valid.shape[0]}")
    out = raw.copy()
    window = int(window)
    if window > 1 and raw.size > 1:
        if window % 2 == 0:
            window += 1
        pad = window // 2
        padded = np.pad(raw, (pad, pad), mode="edge")
        kernel = np.ones(window, dtype=np.float64) / float(window)
        smoothed = np.convolve(padded, kernel, mode="valid")
        out = 0.5 * raw + 0.5 * smoothed
    if source_repaired is not None:
        repaired = np.asarray(source_repaired, dtype=bool).reshape(-1)
        if repaired.shape[0] == raw.shape[0]:
            # Repaired frames are useful continuity priors, not direct visual
            # observations. Do not let neighbouring confident frames promote
            # them above their source quality.
            out[repaired] = np.minimum(out[repaired], raw[repaired])
    out[~valid] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def load_targets(
    sidecar: pathlib.Path,
    side: str,
    scale: float,
    reproj_good_px: float,
    reproj_bad_px: float,
    reproj_good_ratio: float,
    reproj_bad_ratio: float,
    reliability_smooth_window: int,
) -> dict:
    with np.load(sidecar, allow_pickle=True) as data:
        joints = np.asarray(data[f"{side}_hand_joints_3d"], dtype=np.float64)
        quat = normalize_quat_wxyz(np.asarray(data[f"{side}_hand_wrist_quat"], dtype=np.float64))
        valid = np.asarray(data[f"{side}_hand_valid"], dtype=bool)
        if f"{side}_hand_wrist_frame_valid" in data.files:
            valid &= np.asarray(data[f"{side}_hand_wrist_frame_valid"], dtype=bool)
        for suffix in ("hand_bad_mask", "hand_spike_mask"):
            key = f"{side}_{suffix}"
            if key in data.files:
                valid &= ~np.asarray(data[key], dtype=bool)
        reproj_key = f"{side}_hand_reproj_error"
        reproj_error = (
            np.asarray(data[reproj_key], dtype=np.float64)
            if reproj_key in data.files
            else np.full(valid.shape, np.nan, dtype=np.float64)
        )
        reproj_relative_key = f"{side}_hand_reproj_error_relative"
        reproj_relative = (
            np.asarray(data[reproj_relative_key], dtype=np.float64)
            if reproj_relative_key in data.files
            else np.full(valid.shape, np.nan, dtype=np.float64)
        )
        quality_key = f"{side}_hand_quality"
        source_quality = (
            np.asarray(data[quality_key], dtype=np.float64)
            if quality_key in data.files
            else None
        )
        source_reliable_key = f"{side}_hand_source_reliable"
        source_reliable = (
            np.asarray(data[source_reliable_key], dtype=bool)
            if source_reliable_key in data.files
            else None
        )
        source_repaired_key = f"{side}_hand_source_repaired"
        source_repaired = (
            np.asarray(data[source_repaired_key], dtype=bool)
            if source_repaired_key in data.files
            else None
        )

    finite = np.isfinite(joints).all(axis=(1, 2)) & np.isfinite(quat).all(axis=1)
    nonzero = np.linalg.norm(joints.reshape(joints.shape[0], -1), axis=1) > 1e-8
    valid &= finite & nonzero
    reliability = valid.astype(np.float64)
    if source_quality is not None:
        reliability = np.clip(source_quality.reshape(-1), 0.0, 1.0)
        if reliability.shape[0] != valid.shape[0]:
            raise ValueError(
                f"{quality_key} has {reliability.shape[0]} frames; "
                f"expected {valid.shape[0]}"
            )
        reliability[~valid] = 0.0
        return_quality_directly = True
    else:
        return_quality_directly = False
    relative_finite = np.isfinite(reproj_relative) & (reproj_relative >= 0.0)
    pixel_finite = np.isfinite(reproj_error) & (reproj_error < 1e5)
    if not return_quality_directly and relative_finite.any():
        denom = max(float(reproj_bad_ratio) - float(reproj_good_ratio), 1e-6)
        reproj_reliability = np.clip(
            (float(reproj_bad_ratio) - reproj_relative) / denom,
            0.0,
            1.0,
        )
        reliability[relative_finite] *= reproj_reliability[relative_finite]
        fallback = ~relative_finite & pixel_finite
        pixel_denom = max(float(reproj_bad_px) - float(reproj_good_px), 1e-6)
        pixel_reliability = np.clip(
            (float(reproj_bad_px) - reproj_error) / pixel_denom,
            0.0,
            1.0,
        )
        reliability[fallback] *= pixel_reliability[fallback]
        reliability[~relative_finite & ~pixel_finite] = 0.0
    elif not return_quality_directly and pixel_finite.any():
        denom = max(float(reproj_bad_px) - float(reproj_good_px), 1e-6)
        reproj_reliability = np.clip(
            (float(reproj_bad_px) - reproj_error) / denom,
            0.0,
            1.0,
        )
        reliability[pixel_finite] *= reproj_reliability[pixel_finite]
        reliability[~pixel_finite] = 0.0
    reliability_raw = np.clip(reliability, 0.0, 1.0).astype(np.float32)
    reliability = smooth_reliability(
        reliability_raw,
        valid,
        source_repaired=source_repaired,
        window=reliability_smooth_window,
    )

    if valid.any():
        center = np.median(joints[valid, 0], axis=0)
    else:
        center = np.zeros(3)
    joints = (joints - center) * float(scale)

    palm_pos = hold_invalid(joints[:, 0], valid)
    tips = hold_invalid(joints[:, FINGERTIP_IDX], valid)
    joints = hold_invalid(joints, valid)
    palm_quat = hold_invalid_quat_wxyz(quat, valid)
    return {
        "palm_pos": palm_pos,
        "palm_quat": palm_quat,
        "tips": tips,
        "joints": joints,
        "valid": valid,
        "source_reliable": (
            source_reliable.astype(bool)
            if source_reliable is not None and source_reliable.shape[0] == valid.shape[0]
            else valid.copy()
        ),
        "source_repaired": (
            source_repaired.astype(bool)
            if source_repaired is not None and source_repaired.shape[0] == valid.shape[0]
            else np.zeros_like(valid, dtype=bool)
        ),
        "reliability_raw": reliability_raw,
        "reliability": reliability.astype(np.float32),
        "reproj_error": reproj_error.astype(np.float32),
        "reproj_error_relative": reproj_relative.astype(np.float32),
    }


def qpos_joint_names(model: mujoco.MjModel) -> list[str]:
    ordered = []
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name:
            ordered.append((int(model.jnt_qposadr[jid]), name))
    return [name for _, name in sorted(ordered)]


def smooth_and_limit(
    qpos: np.ndarray,
    model: mujoco.MjModel,
    window: int,
    max_delta: float,
    max_accel: float,
    reliability: np.ndarray | None = None,
    low_conf_accel_scale: float = 0.5,
    articulated_qpos_start: int = 6,
) -> np.ndarray:
    out = np.asarray(qpos, dtype=np.float32).copy()
    start = int(articulated_qpos_start)
    auxiliary_base = out[:, :start].copy()
    window = int(window)
    if window > 1:
        if window % 2 == 0:
            window += 1
        pad = window // 2
        padded = np.pad(out, ((pad, pad), (0, 0)), mode="edge")
        kernel = np.ones(window, dtype=np.float32) / float(window)
        out[:, start:] = np.stack(
            [
                np.convolve(padded[:, col], kernel, mode="valid")
                for col in range(start, out.shape[1])
            ],
            axis=1,
        ).astype(np.float32)
    max_delta = float(max_delta)
    max_accel = float(max_accel)
    if reliability is None:
        reliability = np.ones(out.shape[0], dtype=np.float32)
    reliability = np.clip(np.asarray(reliability, dtype=np.float32), 0.0, 1.0)
    low_conf_accel_scale = float(np.clip(low_conf_accel_scale, 0.0, 1.0))
    if max_delta > 0.0 or max_accel > 0.0:
        for frame in range(1, out.shape[0]):
            # The first six coordinates only place the standalone hand model
            # in target space during IK.  They are not exported to the robot;
            # smoothing/clamping them can create artificial palm-frame error.
            out[frame, :start] = auxiliary_base[frame]
            delta = out[frame] - out[frame - 1]
            if max_delta > 0.0:
                delta[start:] = np.clip(delta[start:], -max_delta, max_delta)
            if max_accel > 0.0 and frame >= 2:
                previous_delta = out[frame - 1] - out[frame - 2]
                accel_limit = max_accel * (
                    low_conf_accel_scale
                    + (1.0 - low_conf_accel_scale) * float(reliability[frame])
                )
                delta[start:] = np.clip(
                    delta[start:],
                    previous_delta[start:] - accel_limit,
                    previous_delta[start:] + accel_limit,
                )
                if max_delta > 0.0:
                    delta[start:] = np.clip(delta[start:], -max_delta, max_delta)
            out[frame] = out[frame - 1] + delta
    out[:, :start] = auxiliary_base
    for joint_id in range(model.njnt):
        if not bool(model.jnt_limited[joint_id]):
            continue
        qpos_adr = int(model.jnt_qposadr[joint_id])
        lo, hi = (float(v) for v in model.jnt_range[joint_id])
        out[:, qpos_adr] = np.clip(out[:, qpos_adr], lo, hi)
    return out


def direction_angle_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    dot = np.sum(first * second, axis=-1)
    return np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0)))


def repair_direction_runs(
    directions: list[np.ndarray],
    bad: np.ndarray,
    valid: np.ndarray,
    max_gap: int,
) -> list[np.ndarray]:
    """Replace short implausible runs without changing any bone length."""
    repaired = [np.asarray(item, dtype=np.float64).copy() for item in directions]
    bad = np.asarray(bad, dtype=bool)
    good = np.asarray(valid, dtype=bool) & ~bad
    if not bad.any() or not good.any():
        return repaired

    bad_indices = np.flatnonzero(bad)
    run_start = 0
    while run_start < len(bad_indices):
        run_end = run_start
        while (
            run_end + 1 < len(bad_indices)
            and bad_indices[run_end + 1] == bad_indices[run_end] + 1
        ):
            run_end += 1
        first = int(bad_indices[run_start])
        last = int(bad_indices[run_end])
        previous = np.flatnonzero(good[:first])
        following = np.flatnonzero(good[last + 1 :])
        previous_idx = int(previous[-1]) if previous.size else None
        following_idx = int(last + 1 + following[0]) if following.size else None
        run_length = last - first + 1

        for frame in range(first, last + 1):
            if (
                previous_idx is not None
                and following_idx is not None
                and run_length <= int(max_gap)
            ):
                alpha = (frame - previous_idx) / float(following_idx - previous_idx)
                for bone_idx, values in enumerate(repaired):
                    direction = (1.0 - alpha) * values[previous_idx] + alpha * values[following_idx]
                    repaired[bone_idx][frame] = direction / max(
                        float(np.linalg.norm(direction)),
                        1e-8,
                    )
            elif previous_idx is not None:
                for values in repaired:
                    values[frame] = values[previous_idx]
            elif following_idx is not None:
                for values in repaired:
                    values[frame] = values[following_idx]
        run_start = run_end + 1
    return repaired


def build_chain_targets(
    model: mujoco.MjModel,
    side: str,
    targets: dict,
    max_pip_bend_deg: float,
    max_dip_bend_deg: float,
    max_source_bone_delta_deg: float,
    max_source_bone_length_ratio: float,
    anatomic_repair_max_gap: int,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Build morphology-matched PIP/DIP/tip targets.

    Source bone directions are expressed in the MANO palm frame.  They are
    accumulated from each Sharpa MCP using Sharpa's own segment lengths, so
    the IK follows finger articulation rather than trying to stretch one hand
    model to the other hand's proportions.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    palm_site = model.site(f"{side}_palm").id
    rest_palm_pos = np.asarray(data.site_xpos[palm_site], dtype=np.float64).copy()
    rest_palm_rot = np.asarray(data.site_xmat[palm_site], dtype=np.float64).reshape(3, 3).copy()

    source_rot = Rotation.from_quat(targets["palm_quat"][:, [1, 2, 3, 0]])
    source_joints = np.asarray(targets["joints"], dtype=np.float64)
    palm_pos = np.asarray(targets["palm_pos"], dtype=np.float64)
    valid = np.asarray(targets["valid"], dtype=bool)
    result: dict[str, np.ndarray] = {}
    stats = {
        "anatomic_bad_frames": 0,
        "direction_spike_frames": 0,
        "bone_length_bad_frames": 0,
        "repaired_finger_frames": 0,
    }

    for finger in FINGERS:
        chain = FINGER_CHAINS[finger]
        rest_sites = [
            np.asarray(data.site_xpos[model.site(f"{side}_{finger}_{level}").id], dtype=np.float64).copy()
            for level in ("mcp", "pip", "dip", "tip")
        ]
        rest_mcp_local = rest_palm_rot.T @ (rest_sites[0] - rest_palm_pos)
        segment_lengths = [
            float(np.linalg.norm(rest_sites[idx + 1] - rest_sites[idx]))
            for idx in range(3)
        ]

        source_dirs_local = []
        source_bone_lengths = []
        for parent_idx, child_idx in zip(chain[:-1], chain[1:]):
            direction = source_joints[:, child_idx] - source_joints[:, parent_idx]
            norm = np.linalg.norm(direction, axis=-1, keepdims=True)
            source_bone_lengths.append(norm[:, 0])
            direction = direction / np.clip(norm, 1e-8, None)
            source_dirs_local.append(source_rot.inv().apply(direction))

        pip_bend = direction_angle_deg(source_dirs_local[0], source_dirs_local[1])
        dip_bend = direction_angle_deg(source_dirs_local[1], source_dirs_local[2])
        anatomic_bad = valid & (
            (pip_bend > float(max_pip_bend_deg))
            | (dip_bend > float(max_dip_bend_deg))
        )
        direction_spike = np.zeros(valid.shape, dtype=bool)
        threshold = float(max_source_bone_delta_deg)
        if threshold > 0.0 and len(valid) >= 3:
            for direction in source_dirs_local:
                incoming = direction_angle_deg(direction[1:-1], direction[:-2])
                outgoing = direction_angle_deg(direction[1:-1], direction[2:])
                bridge = direction_angle_deg(direction[:-2], direction[2:])
                isolated = (
                    (incoming > threshold)
                    & (outgoing > threshold)
                    & (bridge < threshold * 0.5)
                    & valid[:-2]
                    & valid[1:-1]
                    & valid[2:]
                )
                direction_spike[1:-1] |= isolated
        bone_length_bad = np.zeros(valid.shape, dtype=bool)
        length_ratio = float(max_source_bone_length_ratio)
        if length_ratio > 0.0:
            for bone_length in source_bone_lengths:
                length_ok = valid & np.isfinite(bone_length) & (bone_length > 1e-8)
                if length_ok.any():
                    median_length = float(np.median(bone_length[length_ok]))
                    relative_deviation = (
                        np.abs(bone_length - median_length) / max(median_length, 1e-8)
                    )
                    bone_length_bad |= length_ok & (relative_deviation > length_ratio)
        repair_bad = anatomic_bad | direction_spike | bone_length_bad
        source_dirs_local = repair_direction_runs(
            source_dirs_local,
            repair_bad,
            valid,
            anatomic_repair_max_gap,
        )
        stats["anatomic_bad_frames"] += int(anatomic_bad.sum())
        stats["direction_spike_frames"] += int(direction_spike.sum())
        stats["bone_length_bad_frames"] += int(bone_length_bad.sum())
        stats["repaired_finger_frames"] += int(repair_bad.sum())

        source_palm_matrix = source_rot.as_matrix()
        current = palm_pos + np.einsum(
            "tij,j->ti",
            source_palm_matrix,
            rest_mcp_local,
        )
        for level, direction_local, length in zip(CHAIN_LEVELS, source_dirs_local, segment_lengths):
            direction_world = np.einsum(
                "tij,tj->ti",
                source_palm_matrix,
                direction_local,
            )
            current = current + float(length) * direction_world
            result[f"{finger}_{level}"] = current.copy()

    return result, stats


def evaluate_chain_error(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    side: str,
    chain_targets: dict[str, np.ndarray],
) -> tuple[float, float]:
    data = mujoco.MjData(model)
    errors = []
    for frame in range(qpos.shape[0]):
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        frame_errors = []
        for finger in FINGERS:
            for level in CHAIN_LEVELS:
                site_id = model.site(f"{side}_{finger}_{level}").id
                target = chain_targets[f"{finger}_{level}"][frame]
                frame_errors.append(float(np.linalg.norm(data.site_xpos[site_id] - target)))
        errors.append(float(np.mean(frame_errors)))
    errors = np.asarray(errors, dtype=np.float64)
    return float(np.mean(errors)), float(np.quantile(errors, 0.95))


def solve_side(
    xml_path: pathlib.Path,
    side: str,
    targets: dict,
    solver: str,
    wrist_pos_cost: float,
    wrist_ori_cost: float,
    finger_pos_cost: float,
    joint_pos_cost: float,
    posture_cost: float,
    temporal_cost: float,
    low_conf_temporal_gain: float,
    min_target_confidence_scale: float,
    steps: int,
    init_steps: int,
    smooth_window: int,
    max_delta: float,
    max_accel: float,
    low_conf_accel_scale: float,
    max_pip_bend_deg: float,
    max_dip_bend_deg: float,
    max_source_bone_delta_deg: float,
    max_source_bone_length_ratio: float,
    anatomic_repair_max_gap: int,
) -> tuple[np.ndarray, list[str], float, float, dict[str, int]]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    configuration = mink.Configuration(model)
    chain_targets, source_stats = build_chain_targets(
        model,
        side,
        targets,
        max_pip_bend_deg,
        max_dip_bend_deg,
        max_source_bone_delta_deg,
        max_source_bone_length_ratio,
        anatomic_repair_max_gap,
    )

    palm_task = mink.FrameTask(
        f"{side}_palm",
        frame_type="site",
        position_cost=wrist_pos_cost,
        orientation_cost=wrist_ori_cost,
        lm_damping=1.0,
    )
    chain_tasks = {}
    for finger in FINGERS:
        for level in CHAIN_LEVELS:
            cost = finger_pos_cost if level == "tip" else joint_pos_cost
            chain_tasks[(finger, level)] = mink.FrameTask(
                f"{side}_{finger}_{level}",
                frame_type="site",
                position_cost=cost,
                orientation_cost=0.0,
                lm_damping=1.0,
            )
    neutral_cost = np.full(model.nv, float(posture_cost), dtype=np.float64)
    neutral_cost[:6] = 0.0
    posture_task = mink.PostureTask(model, cost=neutral_cost)
    posture_task.set_target(configuration.q.copy())
    temporal_cost_vector = np.full(model.nv, float(temporal_cost), dtype=np.float64)
    temporal_cost_vector[:6] = 0.0
    temporal_task = mink.PostureTask(model, cost=temporal_cost_vector)
    temporal_task.set_target(configuration.q.copy())
    target_tasks = [posture_task, palm_task, *chain_tasks.values()]
    frame_tasks = [*target_tasks, temporal_task]
    # Some target hands (for example BrainCo Revo2) expose coupled distal
    # joints in MJCF even though the hardware has fewer motors.  When the
    # model declares MuJoCo joint equalities, regulate them inside IK so the
    # visual solution remains executable on the real mechanism.
    if model.neq > 0:
        equality_task = mink.EqualityConstraintTask(
            model,
            cost=20.0,
            lm_damping=1.0,
        )
        target_tasks.append(equality_task)
        frame_tasks.append(equality_task)
    limits = [mink.ConfigurationLimit(model)]

    def set_targets(frame: int) -> None:
        palm = targets["palm_pos"][frame]
        quat = targets["palm_quat"][frame]
        palm_task.set_target(mink.SE3(wxyz_xyz=np.r_[quat, palm]))
        for (finger, level), task in chain_tasks.items():
            target = chain_targets[f"{finger}_{level}"][frame]
            task.set_target(mink.SE3(wxyz_xyz=np.r_[1.0, 0.0, 0.0, 0.0, target]))

    dt = float(model.opt.timestep)
    min_target_confidence_scale = float(
        np.clip(min_target_confidence_scale, 0.0, 1.0)
    )
    qpos = np.zeros((targets["palm_pos"].shape[0], model.nq), dtype=np.float32)
    set_targets(0)
    for _ in range(max(0, init_steps)):
        vel = mink.solve_ik(
            configuration,
            target_tasks,
            dt,
            solver,
            damping=1e-5,
            limits=limits,
        )
        configuration.integrate_inplace(vel, dt)
    for frame in range(qpos.shape[0]):
        set_targets(frame)
        temporal_task.set_target(configuration.q.copy())
        confidence = float(np.clip(targets["reliability"][frame], 0.0, 1.0))
        temporal_gain = 1.0 + float(low_conf_temporal_gain) * (1.0 - confidence)
        temporal_task.set_cost(temporal_cost_vector * temporal_gain)
        target_scale = float(min_target_confidence_scale) + (
            1.0 - float(min_target_confidence_scale)
        ) * confidence
        for (finger, level), task in chain_tasks.items():
            base_cost = finger_pos_cost if level == "tip" else joint_pos_cost
            task.set_position_cost(float(base_cost) * target_scale)
        for _ in range(max(1, steps)):
            vel = mink.solve_ik(
                configuration,
                frame_tasks,
                dt,
                solver,
                damping=1e-5,
                limits=limits,
            )
            configuration.integrate_inplace(vel, dt)
        qpos[frame] = configuration.q.copy()
    qpos = smooth_and_limit(
        qpos,
        model,
        smooth_window,
        max_delta,
        max_accel,
        targets["reliability"],
        low_conf_accel_scale,
    )
    error_mean, error_p95 = evaluate_chain_error(model, qpos, side, chain_targets)
    return qpos, qpos_joint_names(model), error_mean, error_p95, source_stats


def maybe_trim(targets: dict, max_frames: int) -> dict:
    if max_frames <= 0:
        return targets
    return {key: np.asarray(value)[:max_frames] for key, value in targets.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand_npz", required=True, type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--sharpa_root", default=DEFAULT_SHARPA_ROOT, type=pathlib.Path)
    parser.add_argument("--side", choices=["left", "right", "both"], default="both")
    parser.add_argument("--solver", default="daqp")
    parser.add_argument("--wrist_pos_cost", type=float, default=0.3)
    parser.add_argument("--wrist_ori_cost", type=float, default=0.2)
    parser.add_argument("--finger_pos_cost", type=float, default=5.0)
    parser.add_argument("--joint_pos_cost", type=float, default=3.0)
    parser.add_argument("--posture_cost", type=float, default=1e-2)
    parser.add_argument("--temporal_cost", type=float, default=3e-2)
    parser.add_argument("--low_conf_temporal_gain", type=float, default=3.0)
    parser.add_argument("--min_target_confidence_scale", type=float, default=0.10)
    parser.add_argument("--reliability_smooth_window", type=int, default=9)
    parser.add_argument("--low_conf_reliability_thr", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--init_steps", type=int, default=40)
    parser.add_argument("--smooth_window", type=int, default=9)
    parser.add_argument("--max_delta", type=float, default=0.08)
    parser.add_argument("--max_accel", type=float, default=0.10)
    parser.add_argument("--low_conf_accel_scale", type=float, default=0.5)
    parser.add_argument("--reproj_good_px", type=float, default=30.0)
    parser.add_argument("--reproj_bad_px", type=float, default=75.0)
    parser.add_argument("--reproj_good_ratio", type=float, default=0.15)
    parser.add_argument("--reproj_bad_ratio", type=float, default=0.45)
    parser.add_argument("--max_pip_bend_deg", type=float, default=125.0)
    parser.add_argument("--max_dip_bend_deg", type=float, default=105.0)
    parser.add_argument("--max_source_bone_delta_deg", type=float, default=45.0)
    parser.add_argument("--max_source_bone_length_ratio", type=float, default=0.25)
    parser.add_argument("--anatomic_repair_max_gap", type=int, default=15)
    parser.add_argument("--max_frames", type=int, default=0)
    args = parser.parse_args()

    sides = ("left", "right") if args.side == "both" else (args.side,)
    output = args.output or args.hand_npz.with_name("001_sharpa_hands.npz")

    out = {
        "source_hand_npz": str(args.hand_npz),
        "sharpa_root": str(args.sharpa_root),
        "scale": 1.0,
        "wrist_pos_cost": float(args.wrist_pos_cost),
        "wrist_ori_cost": float(args.wrist_ori_cost),
        "finger_pos_cost": float(args.finger_pos_cost),
        "joint_pos_cost": float(args.joint_pos_cost),
        "posture_cost": float(args.posture_cost),
        "temporal_cost": float(args.temporal_cost),
        "low_conf_temporal_gain": float(args.low_conf_temporal_gain),
        "min_target_confidence_scale": float(args.min_target_confidence_scale),
        "reliability_smooth_window": int(args.reliability_smooth_window),
        "low_conf_reliability_thr": float(args.low_conf_reliability_thr),
        "smooth_window": int(args.smooth_window),
        "max_delta": float(args.max_delta),
        "max_accel": float(args.max_accel),
        "low_conf_accel_scale": float(args.low_conf_accel_scale),
        "reproj_good_px": float(args.reproj_good_px),
        "reproj_bad_px": float(args.reproj_bad_px),
        "reproj_good_ratio": float(args.reproj_good_ratio),
        "reproj_bad_ratio": float(args.reproj_bad_ratio),
        "max_pip_bend_deg": float(args.max_pip_bend_deg),
        "max_dip_bend_deg": float(args.max_dip_bend_deg),
        "max_source_bone_delta_deg": float(args.max_source_bone_delta_deg),
        "max_source_bone_length_ratio": float(args.max_source_bone_length_ratio),
        "anatomic_repair_max_gap": int(args.anatomic_repair_max_gap),
    }
    for side in sides:
        targets = maybe_trim(
            load_targets(
                args.hand_npz,
                side,
                1.0,
                args.reproj_good_px,
                args.reproj_bad_px,
                args.reproj_good_ratio,
                args.reproj_bad_ratio,
                args.reliability_smooth_window,
            ),
            args.max_frames,
        )
        qpos, names, error_mean, error_p95, source_stats = solve_side(
            args.sharpa_root / f"{side}.xml",
            side,
            targets,
            args.solver,
            args.wrist_pos_cost,
            args.wrist_ori_cost,
            args.finger_pos_cost,
            args.joint_pos_cost,
            args.posture_cost,
            args.temporal_cost,
            args.low_conf_temporal_gain,
            args.min_target_confidence_scale,
            args.steps,
            args.init_steps,
            args.smooth_window,
            args.max_delta,
            args.max_accel,
            args.low_conf_accel_scale,
            args.max_pip_bend_deg,
            args.max_dip_bend_deg,
            args.max_source_bone_delta_deg,
            args.max_source_bone_length_ratio,
            args.anatomic_repair_max_gap,
        )
        out[f"{side}_qpos"] = qpos
        out[f"{side}_hand_qpos"] = qpos[:, 6:28]
        out[f"{side}_qpos_names"] = np.asarray(names)
        out[f"{side}_hand_qpos_names"] = np.asarray(names[6:28])
        out[f"{side}_valid"] = targets["valid"]
        out[f"{side}_source_reliable"] = targets["source_reliable"]
        out[f"{side}_source_repaired"] = targets["source_repaired"]
        out[f"{side}_reliability_raw"] = targets["reliability_raw"]
        out[f"{side}_reliability"] = targets["reliability"]
        low_conf = targets["reliability"] < float(args.low_conf_reliability_thr)
        out[f"{side}_low_confidence_mask"] = low_conf
        out[f"{side}_valid_frames"] = int(np.sum(targets["valid"]))
        out[f"{side}_source_reliable_frames"] = int(np.sum(targets["source_reliable"]))
        out[f"{side}_source_repaired_frames"] = int(np.sum(targets["source_repaired"]))
        out[f"{side}_low_confidence_frames"] = int(np.sum(low_conf))
        out[f"{side}_reliability_mean"] = float(np.mean(targets["reliability"]))
        out[f"{side}_reliability_p10"] = float(np.quantile(targets["reliability"], 0.10))
        out[f"{side}_chain_error_mean"] = float(error_mean)
        out[f"{side}_chain_error_p95"] = float(error_p95)
        for key, value in source_stats.items():
            out[f"{side}_{key}"] = int(value)
        print(
            f"{side}: valid={int(np.sum(targets['valid']))}/{len(targets['valid'])}, "
            f"reliability_mean={float(np.mean(targets['reliability'])):.3f}, "
            f"low_conf={int(np.sum(low_conf))}, "
            f"repaired_finger_frames={source_stats['repaired_finger_frames']}, "
            f"chain_error_mean={error_mean * 1000.0:.2f}mm, "
            f"p95={error_p95 * 1000.0:.2f}mm"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **out)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
