#!/usr/bin/env python3
"""Export a GMR G1 reference as Unitree RL Mjlab motion-tracking data.

This is a format bridge only.  It never simulates or modifies a trajectory:
the output is the kinematic reference consumed by Mjlab's existing G1 motion
imitation environment.  Physical replay still has to use a free base and
``mj_step`` with the chair collision scene enabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_quaternions_xyzw(quaternions: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternions, dtype=np.float64).copy()
    norms = np.linalg.norm(result, axis=1)
    if np.any(norms < 1e-8):
        raise ValueError("root_rot contains a zero-norm quaternion")
    result /= norms[:, None]
    # q and -q describe the same orientation.  Keeping the temporal sign
    # continuous is essential before differentiating it into angular velocity.
    for frame in range(1, len(result)):
        if np.dot(result[frame - 1], result[frame]) < 0.0:
            result[frame] *= -1.0
    return result


def _quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _differentiate(values: np.ndarray, dt: float) -> np.ndarray:
    return np.gradient(np.asarray(values, dtype=np.float64), dt, axis=0, edge_order=1)


def _angular_velocity_world_wxyz(quaternions: np.ndarray, dt: float) -> np.ndarray:
    """Finite-difference world angular velocity for local-to-world quaternions."""
    q = np.asarray(quaternions, dtype=np.float64)
    q_dot = _differentiate(q, dt)
    q_inverse = q.copy()
    q_inverse[..., 1:] *= -1.0
    omega_quat = 2.0 * _quat_multiply_wxyz(q_dot, q_inverse)
    return omega_quat[..., 1:]


def _actuated_joint_names(model: mujoco.MjModel) -> list[str]:
    names = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name is None:
            raise ValueError("Mjlab XML has an unnamed articulated joint")
        names.append(name)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-motion", type=Path, required=True)
    parser.add_argument("--mjlab-xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    with args.robot_motion.open("rb") as handle:
        motion = pickle.load(handle)
    required = {"root_pos", "root_rot", "dof_pos", "dof_names", "fps"}
    missing = required.difference(motion)
    if missing:
        raise KeyError("robot motion missing keys: %s" % sorted(missing))

    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_rot_xyzw = _normalise_quaternions_xyzw(motion["root_rot"])
    joint_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
    fps = float(motion["fps"])
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError("root_pos must have shape [T, 3]")
    if root_rot_xyzw.shape != (len(root_pos), 4):
        raise ValueError("root_rot must have shape [T, 4]")
    if joint_pos.ndim != 2 or len(joint_pos) != len(root_pos):
        raise ValueError("dof_pos must have shape [T, joint_count]")
    if fps <= 0.0:
        raise ValueError("fps must be positive")

    model = mujoco.MjModel.from_xml_path(str(args.mjlab_xml))
    expected_joint_names = _actuated_joint_names(model)
    source_joint_names = list(motion["dof_names"])
    if source_joint_names != expected_joint_names:
        raise ValueError(
            "GMR-to-Mjlab joint order mismatch:\nsource=%s\nexpected=%s"
            % (source_joint_names, expected_joint_names)
        )
    if joint_pos.shape[1] != len(expected_joint_names):
        raise ValueError("dof_pos count does not match Mjlab articulated joint count")
    if model.nq != 7 + joint_pos.shape[1]:
        raise ValueError("Mjlab XML qpos layout is not a free base plus the GMR joints")

    frame_count = len(root_pos)
    qpos = np.empty((frame_count, model.nq), dtype=np.float64)
    qpos[:, :3] = root_pos
    qpos[:, 3:7] = root_rot_xyzw[:, [3, 0, 1, 2]]
    qpos[:, 7:] = joint_pos
    body_pos = np.empty((frame_count, model.nbody - 1, 3), dtype=np.float64)
    body_quat = np.empty((frame_count, model.nbody - 1, 4), dtype=np.float64)
    data = mujoco.MjData(model)
    for frame in range(frame_count):
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        # Mjlab's body_link arrays exclude the MuJoCo world body.  This is
        # verified by the official G1 tracking sample's [T, 30, ...] schema.
        body_pos[frame] = data.xpos[1:]
        body_quat[frame] = data.xquat[1:]

    dt = 1.0 / fps
    joint_vel = _differentiate(joint_pos, dt)
    body_lin_vel = _differentiate(body_pos, dt)
    body_ang_vel = _angular_velocity_world_wxyz(body_quat, dt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        fps=np.asarray([fps], dtype=np.float64),
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_pos_w=body_pos.astype(np.float32),
        body_quat_w=body_quat.astype(np.float32),
        body_lin_vel_w=body_lin_vel.astype(np.float32),
        body_ang_vel_w=body_ang_vel.astype(np.float32),
    )
    report = {
        "schema_version": 1,
        "mode": "kinematic_reference_export_only",
        "robot_motion": str(args.robot_motion),
        "robot_motion_sha256": _sha256(args.robot_motion),
        "mjlab_xml": str(args.mjlab_xml),
        "mjlab_xml_sha256": _sha256(args.mjlab_xml),
        "output": str(args.output),
        "frame_count": int(frame_count),
        "fps": fps,
        "joint_names": expected_joint_names,
        "body_names_excluding_world": [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            for body_id in range(1, model.nbody)
        ],
        "root_rotation_input": "GMR root_rot XYZW",
        "root_rotation_output": "MuJoCo/Mjlab body quaternions WXYZ",
        "physical_controller_contract": "This file is a reference only.  A valid replay must retain the free base, use torque-limited actions, have no root wrench or qpos reset, and advance with mj_step against the semantic chair collision model.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
