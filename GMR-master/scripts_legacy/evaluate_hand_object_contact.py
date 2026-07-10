#!/usr/bin/env python3
"""Evaluate hand-object distance with actual MuJoCo robot FK.

Robot qpos is used only for a kinematic FK query.  Hand collision geometry
centres are compared with the tracked mesh in object-local coordinates; no
robot-root fixed offsets are used.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from render_robot_motion_headless import (
    DEFAULT_SHARPA_ROOT,
    ROBOT_XML_DICT,
    assign_named_qpos,
    build_h1_sharpa_visual_xml,
    camera_sample_index,
    joint_qpos_map,
    load_external_hand_motion,
    load_motion,
    load_object_motion,
    parse_vec,
    qpos_joint_names,
)


def _set_robot_frame(
    target,
    frame_idx,
    root_pos,
    root_rot_xyzw,
    dof_pos,
    qpos_map,
    source_dof_names,
    external_motion,
):
    target[:3] = root_pos[frame_idx]
    target[3:7] = root_rot_xyzw[frame_idx][[3, 0, 1, 2]]
    if external_motion is None:
        target[7 : 7 + dof_pos.shape[1]] = dof_pos[frame_idx]
        return

    class QposView:
        pass

    view = QposView()
    view.qpos = target
    assign_named_qpos(
        view,
        qpos_map,
        source_dof_names,
        dof_pos[frame_idx],
    )
    hand_idx = camera_sample_index(
        frame_idx,
        root_pos.shape[0],
        external_motion["left_qpos"].shape[0],
    )
    for side in ("left", "right"):
        assign_named_qpos(
            view,
            qpos_map,
            external_motion[f"{side}_names"],
            external_motion[f"{side}_qpos"][hand_idx],
        )


def _load_mesh_vertices(path: pathlib.Path) -> np.ndarray:
    import trimesh

    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            tuple(
                geometry
                for geometry in mesh.geometry.values()
                if isinstance(geometry, trimesh.Trimesh)
            )
        )
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError(f"mesh has no usable vertices: {path}")
    centres = np.asarray(mesh.triangles_center, dtype=np.float64)
    samples = np.concatenate([vertices, centres], axis=0)
    if len(samples) > 200_000:
        index = np.linspace(
            0,
            len(samples) - 1,
            200_000,
            dtype=np.int64,
        )
        samples = samples[index]
    return samples


def _longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    longest = 0
    best_start = -1
    current = 0
    current_start = 0
    for index, value in enumerate(np.asarray(mask, dtype=bool)):
        if value:
            if current == 0:
                current_start = index
            current += 1
            if current > longest:
                longest = current
                best_start = current_start
        else:
            current = 0
    return longest, best_start


def evaluate(args) -> dict:
    import mujoco as mj

    _, fps, root_pos, root_rot_xyzw, dof_pos, _, _ = load_motion(
        args.robot_motion
    )
    object_motion = load_object_motion(args.object_motion)
    base_xml = (
        pathlib.Path(args.robot_xml)
        if args.robot_xml
        else ROBOT_XML_DICT[args.robot]
    )
    external_motion = load_external_hand_motion(args.sharpa_hand_npz)
    source_dof_names = None
    xml_path = base_xml
    temporary_xml = None
    if external_motion is not None:
        source_model = mj.MjModel.from_xml_path(str(base_xml))
        source_dof_names = qpos_joint_names(source_model)
        legacy_quat = parse_vec(
            args.sharpa_mount_quat,
            4,
            [0.5, 0.5, 0.5, 0.5],
        )
        mount_pos = parse_vec(args.sharpa_mount_pos, 3, [0.055, 0.0, 0.0])
        temporary_xml = build_h1_sharpa_visual_xml(
            base_xml,
            args.sharpa_root,
            mount_pos,
            parse_vec(
                args.sharpa_left_mount_quat,
                4,
                [0.5, -0.5, 0.5, -0.5]
                if not args.sharpa_mount_quat
                else legacy_quat,
            ),
            parse_vec(
                args.sharpa_right_mount_quat,
                4,
                legacy_quat,
            ),
            left_mount_pos=parse_vec(args.sharpa_left_mount_pos, 3, mount_pos),
            right_mount_pos=parse_vec(args.sharpa_right_mount_pos, 3, mount_pos),
        )
        xml_path = temporary_xml

    try:
        model = mj.MjModel.from_xml_path(str(xml_path))
        data = mj.MjData(model)
        qpos_map = joint_qpos_map(model)
        hand_geoms: dict[str, list[int]] = {"left": [], "right": []}
        for geom_id in range(model.ngeom):
            name = mj.mj_id2name(
                model,
                mj.mjtObj.mjOBJ_GEOM,
                geom_id,
            ) or ""
            for side in ("left", "right"):
                if name.startswith(f"collision_hand_{side}_"):
                    hand_geoms[side].append(geom_id)
        if not any(hand_geoms.values()):
            raise RuntimeError("robot XML has no named hand collision geometry")

        vertices = _load_mesh_vertices(
            pathlib.Path(object_motion["visual_mesh_path"])
        )
        vertex_tree = cKDTree(vertices)
        frame_count = min(len(root_pos), len(object_motion["position"]))
        distances = {
            side: np.full(frame_count, np.inf, dtype=np.float64)
            for side in ("left", "right")
        }
        nearest_geoms = {
            side: np.full(frame_count, -1, dtype=np.int32)
            for side in ("left", "right")
        }
        target = data.qpos.copy()
        for frame in range(frame_count):
            _set_robot_frame(
                target,
                frame,
                root_pos,
                root_rot_xyzw,
                dof_pos,
                qpos_map,
                source_dof_names,
                external_motion,
            )
            data.qpos[:] = target
            mj.mj_forward(model, data)
            if not object_motion["valid"][frame]:
                continue
            object_rotation = Rotation.from_quat(
                object_motion["quat_wxyz"][frame][[1, 2, 3, 0]]
            ).as_matrix()
            object_position = object_motion["position"][frame]
            for side, geom_ids in hand_geoms.items():
                if not geom_ids:
                    continue
                world_points = data.geom_xpos[geom_ids]
                local_points = (
                    world_points - object_position[None, :]
                ) @ object_rotation
                vertex_distance, _ = vertex_tree.query(local_points, k=1)
                radii = model.geom_rbound[geom_ids]
                surface_distance = np.maximum(vertex_distance - radii, 0.0)
                best = int(np.argmin(surface_distance))
                distances[side][frame] = float(surface_distance[best])
                nearest_geoms[side][frame] = int(geom_ids[best])
    finally:
        if temporary_xml is not None:
            pathlib.Path(temporary_xml).unlink(missing_ok=True)

    combined = np.minimum(distances["left"], distances["right"])
    contact = combined <= args.contact_threshold_m
    duration, start = _longest_true_run(contact)
    suggested_release = (
        start + duration - 1
        if duration >= args.min_contact_frames
        else -1
    )
    finite = combined[np.isfinite(combined)]
    report = {
        "schema_version": 2,
        "method": "mujoco_fk_hand_collision_to_mesh_surface_samples",
        "fps": float(fps),
        "frame_count": int(frame_count),
        "contact_threshold_m": float(args.contact_threshold_m),
        "min_hand_object_distance_m": (
            float(np.min(finite)) if len(finite) else None
        ),
        "distance_p10_m": (
            float(np.percentile(finite, 10)) if len(finite) else None
        ),
        "contact_frame_ratio": float(np.mean(contact)),
        "contact_duration_frames": int(duration),
        "suggested_release_frame": int(suggested_release),
        "ok_for_dynamic_validation": bool(
            suggested_release >= 0
            and duration >= args.min_contact_frames
        ),
        "distance_trace_npz": str(args.output.with_suffix(".npz")),
        "warnings": [],
    }
    if suggested_release < 0:
        report["warnings"].append(
            "no sustained FK mesh contact window was found"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        left_distance_m=distances["left"].astype(np.float32),
        right_distance_m=distances["right"].astype(np.float32),
        contact=contact,
    )
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot_motion", type=pathlib.Path, required=True)
    parser.add_argument("--object_motion", type=pathlib.Path, required=True)
    parser.add_argument("--sharpa_hand_npz", default="")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--robot", default="unitree_h1_with_hand")
    parser.add_argument("--robot_xml", default="")
    parser.add_argument("--sharpa_root", default=str(DEFAULT_SHARPA_ROOT))
    parser.add_argument("--sharpa_mount_pos", default="0.055,0,0")
    parser.add_argument("--sharpa_left_mount_pos", default="")
    parser.add_argument("--sharpa_right_mount_pos", default="")
    parser.add_argument("--sharpa_mount_quat", default="")
    parser.add_argument("--sharpa_left_mount_quat", default="")
    parser.add_argument("--sharpa_right_mount_quat", default="")
    parser.add_argument("--contact_threshold_m", type=float, default=0.05)
    parser.add_argument("--min_contact_frames", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
