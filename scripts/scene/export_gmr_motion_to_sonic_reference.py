#!/usr/bin/env python3
"""Export a GMR robot motion as an offline GEAR-SONIC reference sequence.

This is deliberately a format adapter, not a motion repairer.  It preserves
the GMR root and 29-DoF joint reference, evaluates the fourteen body points
that SONIC expects with MuJoCo forward kinematics, and writes an independent
CSV directory for SONIC's offline reader.

Input contract
--------------
* ``robot_motion.pkl`` stores root quaternions in ``xyzw`` order.
* GMR joint values follow the 29-DoF MuJoCo motor order.

Output contract
---------------
* SONIC receives quaternions in ``wxyz`` order.
* Its control/reference cadence is 50 Hz.  Joint positions and translations
  are linearly resampled; orientations use shortest-arc SLERP.
* SONIC CSV joint columns use IsaacLab order, so the exported joint positions
  and velocities are explicitly reordered from GMR's MuJoCo order.
* The fourteen body entries and their IsaacLab indices are fixed by SONIC's
  ``motion.yaml``.  They are evaluated from named MuJoCo bodies, never from
  a guessed positional array index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path
from typing import Iterable, Sequence

import mujoco
import numpy as np


GMR_DOF_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

# Output index -> input MuJoCo index.  This is SONIC's published
# G1_ISAACLAB_TO_MUJOCO_DOF mapping, copied here so the adapter has one
# explicit, auditable convention boundary.
SONIC_ISAACLAB_TO_MUJOCO_DOF: tuple[int, ...] = (
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
    11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
)

# This exact order is SONIC's ``motion.yaml`` body_names order.  The integer
# is the corresponding IsaacLab body index recorded in metadata.txt.
SONIC_BODIES: tuple[tuple[int, str], ...] = (
    (0, "pelvis"),
    (4, "left_hip_roll_link"),
    (10, "left_knee_link"),
    (18, "left_ankle_roll_link"),
    (5, "right_hip_roll_link"),
    (11, "right_knee_link"),
    (19, "right_ankle_roll_link"),
    (9, "torso_link"),
    (16, "left_shoulder_roll_link"),
    (22, "left_elbow_link"),
    (28, "left_wrist_yaw_link"),
    (17, "right_shoulder_roll_link"),
    (23, "right_elbow_link"),
    (29, "right_wrist_yaw_link"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-motion", type=Path, required=True)
    parser.add_argument("--robot-xml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Parent directory consumed by SONIC --motion-data.")
    parser.add_argument("--motion-name", default="buying_gmr_50hz")
    parser.add_argument("--target-fps", type=float, default=50.0)
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacing this one generated motion directory.")
    return parser.parse_args()


def _normalise_quaternions_xyzw(quaternions: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternions, dtype=np.float64).copy()
    norms = np.linalg.norm(result, axis=1)
    if np.any(norms < 1e-10):
        raise ValueError("root_rot contains a zero-length quaternion")
    result /= norms[:, None]
    for frame in range(1, len(result)):
        if float(np.dot(result[frame - 1], result[frame])) < 0.0:
            result[frame] *= -1.0
    return result


def _slerp_xyzw(source_times: np.ndarray, source_quaternions: np.ndarray,
                target_times: np.ndarray) -> np.ndarray:
    """Shortest-arc SLERP for a monotonically increasing set of target times."""
    source_quaternions = _normalise_quaternions_xyzw(source_quaternions)
    result = np.empty((len(target_times), 4), dtype=np.float64)
    last = len(source_times) - 1
    for target_index, target_time in enumerate(target_times):
        left = int(np.searchsorted(source_times, target_time, side="right") - 1)
        left = max(0, min(left, last))
        right = min(left + 1, last)
        if left == right:
            result[target_index] = source_quaternions[left]
            continue
        alpha = (target_time - source_times[left]) / (source_times[right] - source_times[left])
        first = source_quaternions[left]
        second = source_quaternions[right]
        dot = float(np.clip(np.dot(first, second), -1.0, 1.0))
        if dot < 0.0:
            second = -second
            dot = -dot
        if dot > 0.9995:
            interpolated = first + alpha * (second - first)
            result[target_index] = interpolated / np.linalg.norm(interpolated)
            continue
        angle = math.acos(dot)
        sin_angle = math.sin(angle)
        result[target_index] = (
            math.sin((1.0 - alpha) * angle) / sin_angle * first
            + math.sin(alpha * angle) / sin_angle * second
        )
    return result


def _linear_resample(source_times: np.ndarray, values: np.ndarray,
                     target_times: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    flat = values.reshape(len(source_times), -1)
    result = np.empty((len(target_times), flat.shape[1]), dtype=np.float64)
    for column in range(flat.shape[1]):
        result[:, column] = np.interp(target_times, source_times, flat[:, column])
    return result.reshape((len(target_times),) + values.shape[1:])


def _quaternion_multiply_wxyz(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    fw, fx, fy, fz = first
    sw, sx, sy, sz = second
    return np.array((
        fw * sw - fx * sx - fy * sy - fz * sz,
        fw * sx + fx * sw + fy * sz - fz * sy,
        fw * sy - fx * sz + fy * sw + fz * sx,
        fw * sz + fx * sy - fy * sx + fz * sw,
    ), dtype=np.float64)


def _body_angular_velocity_wxyz(quaternions: np.ndarray, dt: float) -> np.ndarray:
    """Central finite-difference angular velocity in the world frame."""
    frames, bodies, _ = quaternions.shape
    result = np.zeros((frames, bodies, 3), dtype=np.float64)
    for frame in range(frames):
        before = max(0, frame - 1)
        after = min(frames - 1, frame + 1)
        elapsed = (after - before) * dt
        if elapsed == 0.0:
            continue
        for body in range(bodies):
            start = quaternions[before, body]
            end = quaternions[after, body]
            delta = _quaternion_multiply_wxyz(
                end, np.array((start[0], -start[1], -start[2], -start[3]))
            )
            if delta[0] < 0.0:
                delta *= -1.0
            vector_norm = float(np.linalg.norm(delta[1:]))
            if vector_norm < 1e-12:
                continue
            angle = 2.0 * math.atan2(vector_norm, float(delta[0]))
            result[frame, body] = delta[1:] / vector_norm * (angle / elapsed)
    return result


def _write_csv(path: Path, values: np.ndarray, headers: Sequence[str]) -> None:
    matrix = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    np.savetxt(path, matrix, delimiter=",", header=",".join(headers),
               comments="", fmt="%.9g")


def _flatten_headers(prefix: str, body_count: int, components: Iterable[str]) -> list[str]:
    return [f"{prefix}_{body}_{component}" for body in range(body_count) for component in components]


def _load_gmr_motion(path: Path) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    required = ("fps", "root_pos", "root_rot", "dof_pos", "dof_names")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"GMR motion is missing required fields: {missing}")
    fps = float(payload["fps"])
    root_pos = np.asarray(payload["root_pos"], dtype=np.float64)
    root_rot = np.asarray(payload["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(payload["dof_pos"], dtype=np.float64)
    dof_names = list(payload["dof_names"])
    if fps <= 0.0 or root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError("invalid GMR fps or root_pos shape")
    if root_rot.shape != (len(root_pos), 4) or dof_pos.shape != (len(root_pos), 29):
        raise ValueError("root_rot or dof_pos does not match the GMR frame count")
    if tuple(dof_names) != GMR_DOF_NAMES:
        raise ValueError("GMR dof_names do not match the required 29-DoF MuJoCo order")
    return fps, root_pos, root_rot, dof_pos, dof_names


def _evaluate_sonic_bodies(robot_xml: Path, root_pos: np.ndarray, root_rot_xyzw: np.ndarray,
                           dof_pos: np.ndarray, dof_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    data = mujoco.MjData(model)
    if model.nq < 7:
        raise ValueError("robot XML has no floating-base qpos layout")
    body_ids: list[int] = []
    for _, body_name in SONIC_BODIES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"robot XML has no required SONIC body '{body_name}'")
        body_ids.append(body_id)
    dof_addresses: list[int] = []
    for joint_name in dof_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"robot XML has no GMR joint '{joint_name}'")
        dof_addresses.append(int(model.jnt_qposadr[joint_id]))

    positions = np.empty((len(root_pos), len(SONIC_BODIES), 3), dtype=np.float64)
    quaternions = np.empty((len(root_pos), len(SONIC_BODIES), 4), dtype=np.float64)
    pelvis_id = body_ids[0]
    pelvis_position_error = 0.0
    pelvis_orientation_error_rad = 0.0
    for frame in range(len(root_pos)):
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.qpos[:3] = root_pos[frame]
        data.qpos[3:7] = root_rot_xyzw[frame, (3, 0, 1, 2)]
        data.qpos[dof_addresses] = dof_pos[frame]
        mujoco.mj_forward(model, data)
        positions[frame] = data.xpos[body_ids]
        quaternions[frame] = data.xquat[body_ids]
        pelvis_position_error = max(
            pelvis_position_error,
            float(np.linalg.norm(data.xpos[pelvis_id] - root_pos[frame])),
        )
        root_wxyz = root_rot_xyzw[frame, (3, 0, 1, 2)]
        cosine = abs(float(np.dot(data.xquat[pelvis_id], root_wxyz)))
        pelvis_orientation_error_rad = max(
            pelvis_orientation_error_rad,
            2.0 * math.acos(min(1.0, cosine)),
        )
    report = {
        "pelvis_root_position_max_error_m": pelvis_position_error,
        "pelvis_root_orientation_max_error_rad": pelvis_orientation_error_rad,
        "body_indices": [index for index, _ in SONIC_BODIES],
        "body_names": [name for _, name in SONIC_BODIES],
    }
    if pelvis_position_error > 1e-5 or pelvis_orientation_error_rad > 1e-5:
        raise RuntimeError(
            "robot XML root does not represent the GMR pelvis frame: "
            f"position error={pelvis_position_error:.3e}, "
            f"orientation error={pelvis_orientation_error_rad:.3e}"
        )
    return positions, quaternions, report


def main() -> None:
    args = _parse_args()
    if args.target_fps <= 0.0:
        raise ValueError("--target-fps must be positive")
    if not args.robot_motion.is_file() or not args.robot_xml.is_file():
        raise FileNotFoundError("--robot-motion and --robot-xml must both be files")
    motion_dir = args.output_dir / args.motion_name
    if motion_dir.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {motion_dir}; pass --overwrite explicitly")
    motion_dir.mkdir(parents=True, exist_ok=True)

    source_fps, source_root_pos, source_root_rot, source_dof, dof_names = _load_gmr_motion(args.robot_motion)
    source_times = np.arange(len(source_root_pos), dtype=np.float64) / source_fps
    # SONIC advances one reference frame every 20 ms.  The fractional tail
    # shorter than one tick is intentionally held by its normal end-of-motion
    # behavior instead of distorting all earlier timestamps.
    target_times = np.arange(0.0, source_times[-1] + 1e-12, 1.0 / args.target_fps)
    target_root_pos = _linear_resample(source_times, source_root_pos, target_times)
    target_root_rot_xyzw = _slerp_xyzw(source_times, source_root_rot, target_times)
    # Keep this representation in GMR/MuJoCo order for forward kinematics.
    target_dof_mujoco = _linear_resample(source_times, source_dof, target_times)

    body_pos, body_quat_wxyz, kinematics_report = _evaluate_sonic_bodies(
        args.robot_xml, target_root_pos, target_root_rot_xyzw, target_dof_mujoco, dof_names
    )
    target_dt = 1.0 / args.target_fps
    joint_vel_mujoco = np.gradient(target_dof_mujoco, target_dt, axis=0, edge_order=1)
    target_dof = target_dof_mujoco[:, SONIC_ISAACLAB_TO_MUJOCO_DOF]
    joint_vel = joint_vel_mujoco[:, SONIC_ISAACLAB_TO_MUJOCO_DOF]
    body_lin_vel = np.gradient(body_pos, target_dt, axis=0, edge_order=1)
    body_ang_vel = _body_angular_velocity_wxyz(body_quat_wxyz, target_dt)

    _write_csv(motion_dir / "joint_pos.csv", target_dof, [f"joint_{i}" for i in range(29)])
    _write_csv(motion_dir / "joint_vel.csv", joint_vel, [f"joint_vel_{i}" for i in range(29)])
    _write_csv(motion_dir / "body_pos.csv", body_pos, _flatten_headers("body", len(SONIC_BODIES), ("x", "y", "z")))
    _write_csv(motion_dir / "body_quat.csv", body_quat_wxyz, _flatten_headers("body", len(SONIC_BODIES), ("w", "x", "y", "z")))
    _write_csv(motion_dir / "body_lin_vel.csv", body_lin_vel, _flatten_headers("body", len(SONIC_BODIES), ("vel_x", "vel_y", "vel_z")))
    _write_csv(motion_dir / "body_ang_vel.csv", body_ang_vel, _flatten_headers("body", len(SONIC_BODIES), ("angvel_x", "angvel_y", "angvel_z")))

    source_sha256 = hashlib.sha256(args.robot_motion.read_bytes()).hexdigest()
    metadata = [
        f"Metadata for: {args.motion_name}", "=" * 30, "",
        "Body part indexes:", str([index for index, _ in SONIC_BODIES]), "",
        f"Total timesteps: {len(target_times)}",
        f"Reference fps: {args.target_fps:.9g}",
        "Root quaternion input: xyzw",
        "Body quaternion output: wxyz",
        f"Source robot_motion SHA256: {source_sha256}",
    ]
    (motion_dir / "metadata.txt").write_text("\n".join(metadata) + "\n", encoding="utf-8")
    report = {
        "status": "written",
        "source_motion": str(args.robot_motion),
        "source_sha256": source_sha256,
        "source_fps": source_fps,
        "source_frames": int(len(source_times)),
        "target_fps": args.target_fps,
        "target_frames": int(len(target_times)),
        "source_duration_s": float(source_times[-1]),
        "target_last_timestamp_s": float(target_times[-1]),
        "unrepresented_source_tail_s": float(source_times[-1] - target_times[-1]),
        "joint_position_shape": list(target_dof.shape),
        "input_joint_order": "GMR_MuJoCo_29dof",
        "output_joint_order": "SONIC_IsaacLab_29dof",
        "sonic_isaaclab_to_gmr_mujoco_dof": list(SONIC_ISAACLAB_TO_MUJOCO_DOF),
        "body_position_shape": list(body_pos.shape),
        "max_root_position_step_m": float(np.max(np.linalg.norm(np.diff(target_root_pos, axis=0), axis=1))),
        **kinematics_report,
    }
    report_path = args.output_dir / f"{args.motion_name}_conversion_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"motion_dir": str(motion_dir), "report": str(report_path), **report}, indent=2))


if __name__ == "__main__":
    main()
