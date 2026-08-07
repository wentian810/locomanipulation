#!/usr/bin/env python3
"""Render logged, physically simulated HoloMotion states without re-simulating them.

The input ``*_robot.npz`` is the actual state produced by continuous MuJoCo
``mj_step`` in the accepted roll-out.  This renderer only assigns each logged
state for a visual frame and calls ``mj_forward``; it never calls ``mj_step``
or applies control.  It is therefore a visualisation of the audited trajectory,
not a second (and potentially different) physical replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np


def _metadata(reference: np.lib.npyio.NpzFile) -> dict[str, object]:
    raw = reference["metadata"]
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    value = json.loads(str(raw))
    if not isinstance(value, dict):
        raise ValueError("reference metadata must be a JSON object")
    return value


def _qpos_sequence(
    model: mujoco.MjModel,
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
    dof_names: list[str],
) -> np.ndarray:
    free_joints = [
        joint_id for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
    ]
    if len(free_joints) != 1:
        raise ValueError("task must have exactly one free joint")
    root_address = int(model.jnt_qposadr[free_joints[0]])
    qpos = np.zeros((len(root_pos), model.nq), dtype=np.float64)
    qpos[:, root_address:root_address + 3] = root_pos
    qpos[:, root_address + 3:root_address + 7] = root_rot_xyzw[:, [3, 0, 1, 2]]
    if dof_pos.shape != (len(qpos), len(dof_names)):
        raise ValueError("actual DoF matrix does not match reference DoF names")
    for column, name in enumerate(dof_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"task XML lacks joint {name!r}")
        qpos[:, int(model.jnt_qposadr[joint_id])] = dof_pos[:, column]
    return qpos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-npz", required=True, type=Path)
    parser.add_argument("--reference-npz", required=True, type=Path)
    parser.add_argument("--task-xml", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--camera-distance", type=float, default=4.2)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-12.0)
    args = parser.parse_args()
    if args.video.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite actual-rollout render/report")
    if args.width <= 0 or args.height <= 0 or args.stride <= 0:
        raise ValueError("width, height and stride must be positive")

    with np.load(args.reference_npz, allow_pickle=False) as reference, np.load(args.rollout_npz, allow_pickle=False) as rollout:
        metadata = _metadata(reference)
        dof_names = metadata.get("dof_names")
        if not isinstance(dof_names, list) or not all(isinstance(name, str) for name in dof_names):
            raise ValueError("reference lacks ordered dof_names")
        fps = float(metadata.get("motion_fps", 0.0))
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError("reference metadata motion_fps must be positive")
        root_pos = np.asarray(rollout["robot_global_translation"], dtype=np.float64)[:, 0]
        root_rot_xyzw = np.asarray(rollout["robot_global_rotation_quat"], dtype=np.float64)[:, 0]
        dof_pos = np.asarray(rollout["robot_dof_pos"], dtype=np.float64)

    model = mujoco.MjModel.from_xml_path(str(args.task_xml))
    # Some task XMLs retain MuJoCo's 640px offscreen default.  This is a
    # renderer allocation only; it does not alter any physics option or geom.
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), args.width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), args.height)
    qpos = _qpos_sequence(model, root_pos, root_rot_xyzw, dof_pos, dof_names)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = args.camera_distance
    camera.azimuth = args.camera_azimuth
    camera.elevation = args.camera_elevation
    camera.lookat[:] = np.array((
        float(np.mean(root_pos[:, 0])),
        float(np.mean(root_pos[:, 1])),
        0.65,
    ))

    args.video.parent.mkdir(parents=True, exist_ok=True)
    rendered_frames = 0
    with mujoco.Renderer(model, height=args.height, width=args.width) as renderer, imageio.get_writer(
        str(args.video), format="FFMPEG", fps=fps / args.stride, macro_block_size=1
    ) as writer:
        for frame in range(0, len(qpos), args.stride):
            data.qpos[:] = qpos[frame]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
            rendered_frames += 1

    report = {
        "schema_version": 1,
        "purpose": "accepted_holomotion_actual_mj_step_visualization",
        "rollout": str(args.rollout_npz.resolve()),
        "reference": str(args.reference_npz.resolve()),
        "task_xml": str(args.task_xml.resolve()),
        "video": str(args.video.resolve()),
        "frame_count": int(len(qpos)),
        "rendered_frame_count": rendered_frames,
        "render_fps": fps / args.stride,
        "contract": {
            "state_source": "actual states logged by accepted continuous MuJoCo mj_step roll-out",
            "render_operation": "qpos assignment plus mj_forward for rasterisation only",
            "does_not_do": ["mj_step", "controller_execution", "per-frame physical-state correction"],
        },
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
