#!/usr/bin/env python3
"""Build a simple object proxy trajectory from a rendered robot motion.

This is a visualization bootstrap for hand-object debugging. It creates a
MuJoCo-friendly object_motion.npz by placing a primitive between the robot's
left and right hand links. The object is not inferred from pixels; it is a
stable proxy so GMR renders can show an interaction target before full object
reconstruction/contact optimization exists.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


LEFT_CANDIDATES = (
    "left_hand_link",
    "left_rubber_hand",
    "left_wrist_yaw_link",
    "left_wrist_pitch_link",
    "left_wrist_roll_link",
)
RIGHT_CANDIDATES = (
    "right_hand_link",
    "right_rubber_hand",
    "right_wrist_yaw_link",
    "right_wrist_pitch_link",
    "right_wrist_roll_link",
)


def parse_vec(text, count, default):
    if text is None or str(text).strip() == "":
        return np.asarray(default, dtype=np.float32)
    parts = [float(x) for x in str(text).replace(",", " ").split()]
    if len(parts) != count:
        raise ValueError(f"Expected {count} values, got {text!r}")
    return np.asarray(parts, dtype=np.float32)


def load_motion(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    fps = float(np.asarray(data.get("fps", 30.0)).reshape(-1)[0])
    root_pos = np.asarray(data["root_pos"], dtype=np.float32)
    root_rot_xyzw = np.asarray(data["root_rot"], dtype=np.float32)
    local_body_pos = np.asarray(data["local_body_pos"], dtype=np.float32)
    body_names = [str(x) for x in data["link_body_list"]]
    return data, fps, root_pos, root_rot_xyzw, local_body_pos, body_names


def quat_xyzw_to_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12, None)
    x, y, z, w = np.moveaxis(q, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.stack(
        [
            np.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], axis=-1),
            np.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], axis=-1),
            np.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], axis=-1),
        ],
        axis=-2,
    )


def matrix_to_quat_wxyz(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    quat = np.zeros((*matrix.shape[:-2], 4), dtype=np.float64)
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    trace = m00 + m11 + m22

    mask = trace > 0.0
    s = np.sqrt(np.clip(trace[mask] + 1.0, 1e-12, None)) * 2.0
    quat[mask, 0] = 0.25 * s
    quat[mask, 1] = (matrix[mask, 2, 1] - matrix[mask, 1, 2]) / s
    quat[mask, 2] = (matrix[mask, 0, 2] - matrix[mask, 2, 0]) / s
    quat[mask, 3] = (matrix[mask, 1, 0] - matrix[mask, 0, 1]) / s

    mask_x = (~mask) & (m00 > m11) & (m00 > m22)
    s = np.sqrt(np.clip(1.0 + m00[mask_x] - m11[mask_x] - m22[mask_x], 1e-12, None)) * 2.0
    quat[mask_x, 0] = (matrix[mask_x, 2, 1] - matrix[mask_x, 1, 2]) / s
    quat[mask_x, 1] = 0.25 * s
    quat[mask_x, 2] = (matrix[mask_x, 0, 1] + matrix[mask_x, 1, 0]) / s
    quat[mask_x, 3] = (matrix[mask_x, 0, 2] + matrix[mask_x, 2, 0]) / s

    mask_y = (~mask) & (~mask_x) & (m11 > m22)
    s = np.sqrt(np.clip(1.0 + m11[mask_y] - m00[mask_y] - m22[mask_y], 1e-12, None)) * 2.0
    quat[mask_y, 0] = (matrix[mask_y, 0, 2] - matrix[mask_y, 2, 0]) / s
    quat[mask_y, 1] = (matrix[mask_y, 0, 1] + matrix[mask_y, 1, 0]) / s
    quat[mask_y, 2] = 0.25 * s
    quat[mask_y, 3] = (matrix[mask_y, 1, 2] + matrix[mask_y, 2, 1]) / s

    mask_z = (~mask) & (~mask_x) & (~mask_y)
    s = np.sqrt(np.clip(1.0 + m22[mask_z] - m00[mask_z] - m11[mask_z], 1e-12, None)) * 2.0
    quat[mask_z, 0] = (matrix[mask_z, 1, 0] - matrix[mask_z, 0, 1]) / s
    quat[mask_z, 1] = (matrix[mask_z, 0, 2] + matrix[mask_z, 2, 0]) / s
    quat[mask_z, 2] = (matrix[mask_z, 1, 2] + matrix[mask_z, 2, 1]) / s
    quat[mask_z, 3] = 0.25 * s

    quat = quat / np.clip(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-12, None)
    return quat.astype(np.float32)


def normalize(vec, fallback):
    vec = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    out = np.broadcast_to(np.asarray(fallback, dtype=np.float64), vec.shape).copy()
    valid = norm[..., 0] > 1e-8
    out[valid] = vec[valid] / norm[valid]
    return out


def world_body_positions(root_pos, root_rot_xyzw, local_body_pos):
    rot = quat_xyzw_to_matrix(root_rot_xyzw)
    return root_pos[:, None, :] + np.einsum("tij,tbj->tbi", rot, local_body_pos)


def choose_body(body_names, explicit_name, candidates, side):
    if explicit_name:
        if explicit_name not in body_names:
            raise ValueError(f"{side} body {explicit_name!r} not found")
        return explicit_name
    for name in candidates:
        if name in body_names:
            return name
    raise ValueError(f"Could not find a {side} hand body. Available hand-like names: "
                     f"{[n for n in body_names if 'hand' in n.lower() or 'wrist' in n.lower()]}")


def smooth_positions(values, window):
    window = int(window)
    if window <= 1:
        return values
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.stack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(values.shape[1])], axis=1)


def object_orientation(left, right, up_axis):
    x_axis = normalize(right - left, [1.0, 0.0, 0.0])
    up = np.asarray(up_axis, dtype=np.float64)
    up = up / max(np.linalg.norm(up), 1e-8)
    y_axis = np.cross(up[None, :], x_axis)
    y_axis = normalize(y_axis, [0.0, 1.0, 0.0])
    z_axis = normalize(np.cross(x_axis, y_axis), up)
    matrix = np.stack([x_axis, y_axis, z_axis], axis=-1)
    return matrix_to_quat_wxyz(matrix)


def build(args):
    _, fps, root_pos, root_rot, local_body_pos, body_names = load_motion(args.robot_motion_path)
    body_pos = world_body_positions(root_pos, root_rot, local_body_pos)
    left_name = choose_body(body_names, args.left_body, LEFT_CANDIDATES, "left")
    right_name = choose_body(body_names, args.right_body, RIGHT_CANDIDATES, "right")
    left = body_pos[:, body_names.index(left_name)]
    right = body_pos[:, body_names.index(right_name)]

    if args.anchor == "left":
        position = left.copy()
    elif args.anchor == "right":
        position = right.copy()
    else:
        position = (left + right) * 0.5
    position = position + parse_vec(args.offset, 3, [0.0, 0.0, 0.0])[None, :]
    position = smooth_positions(position, args.smooth_window).astype(np.float32)

    quat = object_orientation(left, right, parse_vec(args.up_axis, 3, [0.0, 0.0, 1.0]))
    geom_size = parse_vec(args.geom_size, 3 if args.object_type == "box" else 1, [0.14, 0.09, 0.07] if args.object_type == "box" else [0.12])
    rgba = parse_vec(args.rgba, 4, [0.95, 0.62, 0.16, 0.9])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        schema_version=np.asarray(1, dtype=np.int32),
        source="robot_motion_hand_proxy",
        coordinate_system="mujoco_robot_world",
        object_type=np.asarray(args.object_type),
        geom_size=geom_size.astype(np.float32),
        rgba=rgba.astype(np.float32),
        fps=np.asarray(fps, dtype=np.float32),
        position=position.astype(np.float32),
        quat_wxyz=quat.astype(np.float32),
        left_anchor=left.astype(np.float32),
        right_anchor=right.astype(np.float32),
        left_anchor_name=np.asarray(left_name),
        right_anchor_name=np.asarray(right_name),
        anchor_mode=np.asarray(args.anchor),
    )

    summary = {
        "output": str(args.output),
        "frames": int(position.shape[0]),
        "fps": fps,
        "object_type": args.object_type,
        "geom_size": geom_size.tolist(),
        "left_anchor_name": left_name,
        "right_anchor_name": right_name,
        "position_min": position.min(axis=0).round(4).tolist(),
        "position_max": position.max(axis=0).round(4).tolist(),
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved object proxy: {args.output}")
    print(f"Saved summary: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_motion_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object_type", choices=["box", "sphere", "cylinder", "capsule"], default="box")
    parser.add_argument("--geom_size", default="", help="MuJoCo geom size. Box uses half extents, sphere uses radius.")
    parser.add_argument("--rgba", default="0.95,0.62,0.16,0.9")
    parser.add_argument("--anchor", choices=["both", "left", "right"], default="both")
    parser.add_argument("--left_body", default="")
    parser.add_argument("--right_body", default="")
    parser.add_argument("--offset", default="0,0,0")
    parser.add_argument("--up_axis", default="0,0,1")
    parser.add_argument("--smooth_window", type=int, default=9)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
