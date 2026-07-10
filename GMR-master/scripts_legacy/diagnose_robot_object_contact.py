#!/usr/bin/env python3
"""Measure distances from robot hand/finger bodies to an object proxy."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


HERE = Path(__file__).parent
ASSET_ROOT = HERE / ".." / "assets"
ROBOT_XML_DICT = {
    "unitree_h1_with_hand": ASSET_ROOT / "unitree_h1" / "h1_with_hand.xml",
    "unitree_h1_with_hand_wrist": ASSET_ROOT / "unitree_h1" / "h1_with_hand.xml",
    "unitree_g1_with_hands": ASSET_ROOT / "unitree_g1" / "g1_mocap_29dof_with_hands.xml",
}

DEFAULT_LEFT_BODIES = (
    "L_thumb_distal",
    "L_index_intermediate",
    "L_middle_intermediate",
    "L_ring_intermediate",
    "L_pinky_intermediate",
    "left_hand_link",
)
DEFAULT_RIGHT_BODIES = (
    "R_thumb_distal",
    "R_index_intermediate",
    "R_middle_intermediate",
    "R_ring_intermediate",
    "R_pinky_intermediate",
    "right_hand_link",
)


def load_motion(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    fps = float(np.asarray(data.get("fps", 30.0)).reshape(-1)[0])
    return fps, np.asarray(data["root_pos"], dtype=np.float32), np.asarray(data["root_rot"], dtype=np.float32), np.asarray(data["dof_pos"], dtype=np.float32)


def load_object(path):
    data = np.load(path, allow_pickle=True)
    pos_key = "position" if "position" in data else "positions"
    position = np.asarray(data[pos_key], dtype=np.float32)
    if "quat_wxyz" in data:
        quat = np.asarray(data["quat_wxyz"], dtype=np.float32)
    elif "quat_xyzw" in data:
        q = np.asarray(data["quat_xyzw"], dtype=np.float32)
        quat = q[:, [3, 0, 1, 2]]
    else:
        quat = np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (position.shape[0], 1))
    object_type = str(np.asarray(data["object_type"] if "object_type" in data else "box").reshape(-1)[0])
    size = np.asarray(data["geom_size"] if "geom_size" in data else data["size"], dtype=np.float32).reshape(-1)
    return object_type, size, position, quat


def quat_wxyz_to_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12, None)
    w, x, y, z = np.moveaxis(q, -1, 0)
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


def parse_body_list(text, fallback):
    if not text:
        return list(fallback)
    return [item.strip() for item in text.replace(",", " ").split() if item.strip()]


def box_distance(points, center, quat_wxyz, half_size):
    rot = quat_wxyz_to_matrix(quat_wxyz)
    local = (points - center[None, :]) @ rot
    q = np.abs(local) - half_size[None, :]
    outside = np.maximum(q, 0.0)
    outside_dist = np.linalg.norm(outside, axis=1)
    inside_dist = np.minimum(np.max(q, axis=1), 0.0)
    return outside_dist + inside_dist


def sphere_distance(points, center, radius):
    return np.linalg.norm(points - center[None, :], axis=1) - float(radius)


def diagnose(args):
    import mujoco as mj

    fps, root_pos, root_rot_xyzw, dof_pos = load_motion(args.robot_motion_path)
    object_type, object_size, object_pos, object_quat = load_object(args.object_motion_path)
    xml_path = Path(args.robot_xml) if args.robot_xml else ROBOT_XML_DICT[args.robot]
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)

    body_names = parse_body_list(args.left_bodies, DEFAULT_LEFT_BODIES) + parse_body_list(args.right_bodies, DEFAULT_RIGHT_BODIES)
    body_ids = []
    kept_names = []
    for name in body_names:
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            body_ids.append(body_id)
            kept_names.append(name)

    if not body_ids:
        raise ValueError("No requested hand/finger bodies were found in the robot XML")

    frame_indices = np.arange(0, root_pos.shape[0], max(1, args.skip), dtype=np.int64)
    if args.max_frames > 0:
        frame_indices = frame_indices[: args.max_frames]

    min_distances = []
    nearest_names = []
    hand_gap = []
    left_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "left_hand_link")
    right_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "right_hand_link")

    for frame_idx in frame_indices:
        data.qpos[:3] = root_pos[frame_idx]
        data.qpos[3:7] = root_rot_xyzw[frame_idx][[3, 0, 1, 2]]
        data.qpos[7:] = dof_pos[frame_idx]
        mj.mj_forward(model, data)

        obj_idx = int(round(frame_idx * (object_pos.shape[0] - 1) / max(root_pos.shape[0] - 1, 1)))
        points = data.xpos[np.asarray(body_ids, dtype=np.int32)]
        if object_type == "sphere":
            distances = sphere_distance(points, object_pos[obj_idx], object_size[0])
        else:
            half_size = object_size[:3] if object_size.size >= 3 else np.repeat(object_size[0], 3)
            distances = box_distance(points, object_pos[obj_idx], object_quat[obj_idx], half_size)
        nearest_idx = int(np.argmin(distances))
        min_distances.append(float(distances[nearest_idx]))
        nearest_names.append(kept_names[nearest_idx])
        if left_id >= 0 and right_id >= 0:
            hand_gap.append(float(np.linalg.norm(data.xpos[left_id] - data.xpos[right_id])))

    min_distances = np.asarray(min_distances, dtype=np.float32)
    hand_gap = np.asarray(hand_gap, dtype=np.float32)
    report = {
        "robot_motion_path": str(args.robot_motion_path),
        "object_motion_path": str(args.object_motion_path),
        "robot_xml": str(xml_path),
        "fps": fps,
        "sampled_frames": int(frame_indices.shape[0]),
        "bodies": kept_names,
        "object_type": object_type,
        "object_size": object_size.tolist(),
        "min_distance_p05_p25_p50_p75_p95_m": np.percentile(min_distances, [5, 25, 50, 75, 95]).round(5).tolist(),
        "min_distance_min_max_m": [float(min_distances.min()), float(min_distances.max())],
        "contact_like_frames_lt_3cm": int((min_distances < 0.03).sum()),
        "contact_like_frames_lt_8cm": int((min_distances < 0.08).sum()),
        "nearest_body_counts": {name: int(np.sum(np.asarray(nearest_names) == name)) for name in sorted(set(nearest_names))},
    }
    if hand_gap.size:
        report["hand_link_gap_p05_p25_p50_p75_p95_m"] = np.percentile(hand_gap, [5, 25, 50, 75, 95]).round(5).tolist()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved contact diagnostic: {args.output}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_motion_path", type=Path, required=True)
    parser.add_argument("--object_motion_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot", default="unitree_h1_with_hand")
    parser.add_argument("--robot_xml", default="")
    parser.add_argument("--left_bodies", default="")
    parser.add_argument("--right_bodies", default="")
    parser.add_argument("--skip", type=int, default=5)
    parser.add_argument("--max_frames", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    diagnose(parse_args())
