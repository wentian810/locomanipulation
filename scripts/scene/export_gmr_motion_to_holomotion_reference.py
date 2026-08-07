#!/usr/bin/env python3
"""Export a GMR 29-DoF motion as HoloMotion's offline tracking reference.

The exporter is deliberately a format bridge, not a trajectory "repair": it
keeps the input GMR root pose and all 29 joint angles, only resampling them to
the frozen policy's control frequency.  Full robot-link global poses are
recomputed by MuJoCo FK from the revision-matched HoloMotion G1 XML.  This
avoids a dangerous implicit assumption that the body ordering in GMR and in a
pretrained tracker happen to agree.

The generated archive uses HoloMotion v1.4's ``ref_*`` schema.  It contains no
scene object: a static chair is later inserted as world geoms, which preserves
the network's fixed robot-body tensor dimension.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np


FORMAT_VERSION = "1.4.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_xyzw(quaternions: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternions, dtype=np.float64).copy()
    if result.ndim != 2 or result.shape[1] != 4:
        raise ValueError(f"expected quaternions [T,4], got {result.shape}")
    norms = np.linalg.norm(result, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms < 1e-8):
        raise ValueError("root_rot contains a non-finite or zero quaternion")
    result /= norms[:, None]
    # Quaternion signs are equivalent geometrically, but discontinuous signs
    # would create a false 2*pi angular velocity at a keyframe boundary.
    for index in range(1, len(result)):
        if np.dot(result[index - 1], result[index]) < 0.0:
            result[index] *= -1.0
    return result


def _slerp_xyzw(left: np.ndarray, right: np.ndarray, fraction: float) -> np.ndarray:
    cosine = float(np.clip(np.dot(left, right), -1.0, 1.0))
    if cosine < 0.0:
        right = -right
        cosine = -cosine
    if cosine > 0.9995:
        result = (1.0 - fraction) * left + fraction * right
        return result / np.linalg.norm(result)
    theta = np.arccos(cosine)
    sine = np.sin(theta)
    return (
        np.sin((1.0 - fraction) * theta) / sine * left
        + np.sin(fraction * theta) / sine * right
    )


def _resample_xyzw(quaternions: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    result = np.empty((len(target_times), 4), dtype=np.float64)
    source_count = len(source_times)
    for output_index, time_value in enumerate(target_times):
        right_index = int(np.searchsorted(source_times, time_value, side="right"))
        if right_index == 0:
            result[output_index] = quaternions[0]
        elif right_index >= source_count:
            result[output_index] = quaternions[-1]
        else:
            left_index = right_index - 1
            fraction = (time_value - source_times[left_index]) / (
                source_times[right_index] - source_times[left_index]
            )
            result[output_index] = _slerp_xyzw(
                quaternions[left_index], quaternions[right_index], float(fraction)
            )
    return result


def _resample_linear(values: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.stack(
        [np.interp(target_times, source_times, values[:, column]) for column in range(values.shape[1])],
        axis=1,
    )


def _time_derivative(values: np.ndarray, timestep: float) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values, dtype=np.float64)
    return np.gradient(values, timestep, axis=0, edge_order=2 if len(values) >= 3 else 1)


def _quat_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.array((
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ), dtype=np.float64)


def _world_angular_velocity_xyzw(quaternions: np.ndarray, timestep: float) -> np.ndarray:
    """Estimate world-frame angular velocity from body-to-world XYZW poses."""
    count = len(quaternions)
    output = np.zeros((count, 3), dtype=np.float64)
    if count < 2:
        return output
    for index in range(count):
        left = max(0, index - 1)
        right = min(count - 1, index + 1)
        if left == right:
            continue
        # q(right) * inverse(q(left)) is the rotation expressed in world axes.
        q_left_inv = quaternions[left].copy()
        q_left_inv[:3] *= -1.0
        delta = _quat_multiply_xyzw(quaternions[right], q_left_inv)
        if delta[3] < 0.0:
            delta *= -1.0
        vector_norm = float(np.linalg.norm(delta[:3]))
        if vector_norm < 1e-10:
            continue
        angle = 2.0 * np.arctan2(vector_norm, float(np.clip(delta[3], -1.0, 1.0)))
        output[index] = delta[:3] / vector_norm * (angle / ((right - left) * timestep))
    return output


def _read_active_joint_names(model: mujoco.MjModel) -> list[str]:
    result: list[str] = []
    for joint_id in range(model.njnt):
        joint_type = model.jnt_type[joint_id]
        if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name:
            result.append(name)
    return result


def _find_free_joint(model: mujoco.MjModel) -> int:
    free_joints = [
        joint_id for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
    ]
    if len(free_joints) != 1:
        raise ValueError(f"HoloMotion G1 XML must have exactly one free joint, found {len(free_joints)}")
    return free_joints[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-motion", required=True, type=Path)
    parser.add_argument("--robot-xml", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--target-fps", type=float, default=50.0)
    args = parser.parse_args()

    if args.output_npz.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite an existing HoloMotion reference")
    if not np.isfinite(args.target_fps) or args.target_fps <= 0.0:
        raise ValueError("--target-fps must be a finite positive value")

    input_motion = args.input_motion.resolve()
    robot_xml = args.robot_xml.resolve()
    with input_motion.open("rb") as stream:
        motion = pickle.load(stream)
    required = ("root_pos", "root_rot", "dof_pos", "dof_names", "fps")
    missing = [key for key in required if key not in motion]
    if missing:
        raise ValueError(f"GMR motion lacks required keys: {missing}")

    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_rot = _normalise_xyzw(np.asarray(motion["root_rot"], dtype=np.float64))
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
    dof_names = [str(name) for name in motion["dof_names"]]
    source_fps = float(motion["fps"])
    if (
        root_pos.ndim != 2 or root_pos.shape[1] != 3
        or dof_pos.ndim != 2
        or len(root_pos) != len(root_rot) or len(root_pos) != len(dof_pos)
    ):
        raise ValueError(
            "GMR arrays must be root_pos [T,3], root_rot [T,4], dof_pos [T,J] with a shared T"
        )
    if len(root_pos) < 2:
        raise ValueError("at least two GMR frames are required for physical tracking")
    if not np.isfinite(root_pos).all() or not np.isfinite(dof_pos).all():
        raise ValueError("GMR motion contains non-finite positions")
    if len(dof_names) != dof_pos.shape[1] or len(set(dof_names)) != len(dof_names):
        raise ValueError("dof_names must be unique and match dof_pos columns")
    if not np.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError("GMR motion fps must be finite and positive")

    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    if model.nq != 7 + len(dof_names):
        raise ValueError(
            f"robot XML qpos layout nq={model.nq} does not match one free base plus {len(dof_names)} joints"
        )
    active_names = _read_active_joint_names(model)
    if active_names != dof_names:
        raise ValueError(
            "GMR and HoloMotion active-joint orders differ; refusing index-based conversion. "
            f"GMR={dof_names}, HoloMotion={active_names}"
        )
    free_joint = _find_free_joint(model)
    free_qposadr = int(model.jnt_qposadr[free_joint])
    joint_qposadr = np.asarray([
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in dof_names
    ], dtype=np.int32)
    if np.any(joint_qposadr < 0):
        raise ValueError("HoloMotion XML lacks a required GMR joint")

    source_times = np.arange(len(root_pos), dtype=np.float64) / source_fps
    duration = float(source_times[-1])
    target_count = int(np.rint(duration * args.target_fps)) + 1
    target_times = np.arange(target_count, dtype=np.float64) / args.target_fps
    target_times[-1] = duration
    root_pos_out = _resample_linear(root_pos, source_times, target_times)
    root_rot_out = _resample_xyzw(root_rot, source_times, target_times)
    dof_pos_out = _resample_linear(dof_pos, source_times, target_times)
    dof_vel_out = _time_derivative(dof_pos_out, 1.0 / args.target_fps)

    body_count = model.nbody - 1
    global_translation = np.empty((target_count, body_count, 3), dtype=np.float64)
    global_rotation_xyzw = np.empty((target_count, body_count, 4), dtype=np.float64)
    data = mujoco.MjData(model)
    for frame in range(target_count):
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.qpos[free_qposadr:free_qposadr + 3] = root_pos_out[frame]
        data.qpos[free_qposadr + 3:free_qposadr + 7] = root_rot_out[frame, [3, 0, 1, 2]]
        data.qpos[joint_qposadr] = dof_pos_out[frame]
        mujoco.mj_forward(model, data)
        global_translation[frame] = data.xpos[1:]
        global_rotation_xyzw[frame] = data.xquat[1:, [1, 2, 3, 0]]

    global_velocity = _time_derivative(global_translation, 1.0 / args.target_fps)
    global_angular_velocity = np.empty((target_count, body_count, 3), dtype=np.float64)
    for body_index in range(body_count):
        global_angular_velocity[:, body_index] = _world_angular_velocity_xyzw(
            global_rotation_xyzw[:, body_index], 1.0 / args.target_fps
        )

    metadata = {
        "format_version": FORMAT_VERSION,
        "motion_fps": float(args.target_fps),
        "source_motion_path": str(input_motion),
        "source_motion_sha256": _sha256(input_motion),
        "robot_xml_path": str(robot_xml),
        "robot_xml_sha256": _sha256(robot_xml),
        "source_fps": source_fps,
        "source_frame_count": int(len(root_pos)),
        "resampled_frame_count": target_count,
        "dof_names": dof_names,
        "conversion": "GMR root/joints preserved; MuJoCo FK regenerated global robot body references",
        "quaternion_convention": "root input/output globals are XYZW; MuJoCo qpos assignment is WXYZ",
        "scene_contract": "no scene bodies included; static scene must be world geoms only",
    }
    arrays = {
        "metadata": np.asarray(json.dumps(metadata, ensure_ascii=False)),
        "ref_dof_pos": dof_pos_out.astype(np.float32),
        "ref_dof_vel": dof_vel_out.astype(np.float32),
        "ref_global_translation": global_translation.astype(np.float32),
        "ref_global_rotation_quat": global_rotation_xyzw.astype(np.float32),
        "ref_global_velocity": global_velocity.astype(np.float32),
        "ref_global_angular_velocity": global_angular_velocity.astype(np.float32),
    }
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **arrays)

    report = {
        "schema_version": 1,
        "status": "ready_for_frozen_holomotion_mj_step_tracking",
        "input": metadata,
        "output_npz": str(args.output_npz.resolve()),
        "shape": {key: list(value.shape) for key, value in arrays.items() if hasattr(value, "shape")},
        "physical_contract": {
            "reference_change": "temporal resampling only",
            "runtime": "one initial qpos/qvel write, bounded policy action, then continuous mj_step",
            "forbidden": ["root_state_reimposition", "xfrc_applied", "mocap_weld", "moving_scene_geometry"],
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
