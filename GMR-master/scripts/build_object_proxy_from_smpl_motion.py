#!/usr/bin/env python3
"""Build a GMR-space object proxy from the source SMPL/SMPL-H motion.

Unlike ``build_object_proxy_from_robot_motion.py``, this does not place the
object between the already-retargeted robot hands. It estimates an interaction
target from the human hand joints, then maps that target into the robot world by
keeping its pelvis-relative position. This keeps the object independent from
robot arm IK failures, which is what we want for contact debugging.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from smpl_npz_to_robot_headless import (  # noqa: E402
    load_body_motion,
    resolve_smpl_joint_names,
    smpl_output_joints,
    world_rotation_for_inputs,
)


LEFT_CONTACT_NAMES = (
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "left_index3",
    "left_middle3",
    "left_ring3",
    "left_pinky3",
)
RIGHT_CONTACT_NAMES = (
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
    "right_index3",
    "right_middle3",
    "right_ring3",
    "right_pinky3",
)
LEFT_PALM_NAMES = ("left_wrist", "left_index1", "left_middle1", "left_ring1", "left_pinky1", "left_thumb1")
RIGHT_PALM_NAMES = ("right_wrist", "right_index1", "right_middle1", "right_ring1", "right_pinky1", "right_thumb1")
PELVIS_NAMES = ("pelvis", "Pelvis", "Hips")


def parse_vec(text, count, default):
    if text is None or str(text).strip() == "":
        return np.asarray(default, dtype=np.float32)
    parts = [float(x) for x in str(text).replace(",", " ").split()]
    if len(parts) != count:
        raise ValueError(f"Expected {count} values, got {text!r}")
    return np.asarray(parts, dtype=np.float32)


def load_robot_motion(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return {
        "fps": float(np.asarray(data.get("fps", 30.0)).reshape(-1)[0]),
        "root_pos": np.asarray(data["root_pos"], dtype=np.float32),
        "root_rot": np.asarray(data["root_rot"], dtype=np.float32),
    }


def first_index(names, candidates):
    for candidate in candidates:
        if candidate in names:
            return names.index(candidate)
    return None


def indices_for(names, candidates):
    return [names.index(name) for name in candidates if name in names]


def joint_group_center(joints, indices):
    if not indices:
        raise ValueError("No joint indices available for object proxy")
    return np.nanmean(joints[:, indices, :], axis=1)


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


def normalize(vec, fallback):
    vec = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    out = np.broadcast_to(np.asarray(fallback, dtype=np.float64), vec.shape).copy()
    valid = norm[..., 0] > 1e-8
    out[valid] = vec[valid] / norm[valid]
    return out


def matrix_to_quat_wxyz(matrix):
    rot = Rotation.from_matrix(matrix)
    quat_xyzw = rot.as_quat()
    return quat_xyzw[:, [3, 0, 1, 2]].astype(np.float32)


def object_orientation(left_center, right_center):
    x_axis = normalize(right_center - left_center, [1.0, 0.0, 0.0])
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    y_axis = normalize(np.cross(up[None, :], x_axis), [0.0, 1.0, 0.0])
    z_axis = normalize(np.cross(x_axis, y_axis), [0.0, 0.0, 1.0])
    matrix = np.stack([x_axis, y_axis, z_axis], axis=-1)
    return matrix_to_quat_wxyz(matrix)


def body_model_root_candidates(root, model_type):
    root = Path(root)
    if (root / model_type).exists():
        return [root]
    return [
        root,
        Path("gvhmr/assets"),
        Path("GVHMR-hand/GVHMR-main/inputs/checkpoints/body_models"),
        Path("GVHMR-main/inputs/checkpoints/body_models"),
        Path("locomotion_pipeline-main/assets"),
    ]


def load_smpl_with_fallback(source_npz, body_model_path, model_type, hand_npz):
    errors = []
    for candidate in body_model_root_candidates(body_model_path, model_type):
        try:
            return (*load_body_motion(source_npz, str(candidate), model_type, hand_npz), str(candidate))
        except Exception as exc:  # noqa: BLE001 - keep trying known local model roots
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Could not load body model:\n" + "\n".join(errors))


def build(args):
    smpl_data, body_model, smpl_output, human_height, used_body_root = load_smpl_with_fallback(
        args.source_npz, args.body_model_path, args.model_type, args.hand_npz or None
    )
    smpl_joints = smpl_output_joints(smpl_output, int(smpl_data["pose_body"].shape[0]))
    joint_names = resolve_smpl_joint_names(body_model, smpl_joints, args.model_type)
    if smpl_joints is None or joint_names is None:
        raise ValueError("Could not resolve SMPL joints for object proxy")

    frame_count = smpl_joints.shape[0]
    world_rot = world_rotation_for_inputs(args.coord_transform, args.human_yaw_offset_deg)
    joints_world = world_rot.apply(smpl_joints.reshape(-1, 3)).reshape(frame_count, smpl_joints.shape[1], 3)

    pelvis_idx = first_index(joint_names, PELVIS_NAMES)
    if pelvis_idx is None:
        pelvis_idx = 0
    pelvis = joints_world[:, pelvis_idx]

    left_contact = indices_for(joint_names, LEFT_CONTACT_NAMES)
    right_contact = indices_for(joint_names, RIGHT_CONTACT_NAMES)
    left_palm = indices_for(joint_names, LEFT_PALM_NAMES)
    right_palm = indices_for(joint_names, RIGHT_PALM_NAMES)
    if not left_contact:
        left_contact = left_palm
    if not right_contact:
        right_contact = right_palm

    left_center = joint_group_center(joints_world, left_contact)
    right_center = joint_group_center(joints_world, right_contact)
    left_palm_center = joint_group_center(joints_world, left_palm or left_contact)
    right_palm_center = joint_group_center(joints_world, right_palm or right_contact)

    if args.anchor == "left":
        human_object = left_center
    elif args.anchor == "right":
        human_object = right_center
    elif args.anchor == "palm":
        human_object = (left_palm_center + right_palm_center) * 0.5
    else:
        human_object = (left_center + right_center) * 0.5

    robot = load_robot_motion(args.robot_motion_path)
    target_frames = min(frame_count, robot["root_pos"].shape[0])
    robot_root = robot["root_pos"][:target_frames]
    rel = human_object[:target_frames] - pelvis[:target_frames]
    position = robot_root + rel * float(args.scale)
    position = position + parse_vec(args.offset, 3, [0.0, 0.0, 0.0])[None, :]

    geom_size = parse_vec(
        args.geom_size,
        3 if args.object_type == "box" else 1,
        [0.14, 0.09, 0.07] if args.object_type == "box" else [0.12],
    )
    if args.object_type == "box":
        min_center_z = float(geom_size[2] + args.ground_clearance)
    else:
        min_center_z = float(geom_size[0] + args.ground_clearance)
    position[:, 2] = np.maximum(position[:, 2], min_center_z)
    position = smooth_positions(position, args.smooth_window).astype(np.float32)

    quat = object_orientation(left_center[:target_frames], right_center[:target_frames])
    rgba = parse_vec(args.rgba, 4, [0.95, 0.62, 0.16, 0.9])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        schema_version=np.asarray(1, dtype=np.int32),
        source="smpl_human_hand_proxy",
        coordinate_system="mujoco_robot_world",
        object_type=np.asarray(args.object_type),
        geom_size=geom_size.astype(np.float32),
        rgba=rgba.astype(np.float32),
        fps=np.asarray(robot["fps"], dtype=np.float32),
        position=position.astype(np.float32),
        quat_wxyz=quat.astype(np.float32),
        human_object_zup=human_object[:target_frames].astype(np.float32),
        human_pelvis_zup=pelvis[:target_frames].astype(np.float32),
        left_contact_zup=left_center[:target_frames].astype(np.float32),
        right_contact_zup=right_center[:target_frames].astype(np.float32),
        body_model_root=np.asarray(used_body_root),
        human_height=np.asarray(human_height, dtype=np.float32),
        anchor_mode=np.asarray(args.anchor),
    )

    summary = {
        "output": str(args.output),
        "frames": int(target_frames),
        "fps": robot["fps"],
        "source": "smpl_human_hand_proxy",
        "body_model_root": used_body_root,
        "anchor": args.anchor,
        "object_type": args.object_type,
        "geom_size": geom_size.tolist(),
        "human_height": float(human_height),
        "position_min": position.min(axis=0).round(4).tolist(),
        "position_max": position.max(axis=0).round(4).tolist(),
        "mean_object_rel_pelvis": rel[:target_frames].mean(axis=0).round(4).tolist(),
        "left_contact_joint_count": len(left_contact),
        "right_contact_joint_count": len(right_contact),
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved SMPL object proxy: {args.output}")
    print(f"Saved summary: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_npz", type=Path, required=True)
    parser.add_argument("--robot_motion_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hand_npz", type=Path, default=None)
    parser.add_argument("--body_model_path", default=str(HERE / ".." / "assets" / "body_models"))
    parser.add_argument("--model_type", choices=["smpl", "smplh", "smplx"], default="smplh")
    parser.add_argument("--coord_transform", choices=["gvhmr", "none"], default="gvhmr")
    parser.add_argument("--human_yaw_offset_deg", type=float, default=180.0)
    parser.add_argument("--object_type", choices=["box", "sphere", "cylinder", "capsule"], default="box")
    parser.add_argument("--geom_size", default="")
    parser.add_argument("--rgba", default="0.95,0.62,0.16,0.9")
    parser.add_argument("--anchor", choices=["both", "left", "right", "palm"], default="both")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--offset", default="0,0,0")
    parser.add_argument("--ground_clearance", type=float, default=0.02)
    parser.add_argument("--smooth_window", type=int, default=15)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
