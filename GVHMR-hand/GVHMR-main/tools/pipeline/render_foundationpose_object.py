#!/usr/bin/env python3
"""Composite a reconstructed object mesh into the GVHMR in-camera video."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", os.environ.get("OBJECT_RENDER_GL", "egl"))

import cv2
import numpy as np


CV_CAMERA_TO_OPENGL = np.diag([1.0, -1.0, -1.0, 1.0])


def _load_motion(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        result = {key: data[key] for key in data.files}
    poses = np.asarray(result["pose_camera_object"], dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"pose_camera_object must be (T,4,4), got {poses.shape}")
    result["pose_camera_object"] = poses
    result["valid"] = np.asarray(
        result.get("valid", np.ones(len(poses), dtype=bool)),
        dtype=bool,
    )
    result["fps"] = float(np.asarray(result.get("fps", 30.0)).reshape(-1)[0])
    result["frame_offset"] = int(
        np.asarray(result.get("gvhmr_frame_offset", 0)).reshape(-1)[0]
    )
    result["mesh_path"] = Path(
        str(np.asarray(result["visual_mesh_path"]).reshape(-1)[0])
    )
    result["mesh_scale"] = np.asarray(
        result.get("mesh_scale", [1.0, 1.0, 1.0]),
        dtype=np.float64,
    ).reshape(3)
    result["K"] = np.asarray(result["K"], dtype=np.float64)
    return result


def _select_k(K: np.ndarray, index: int) -> np.ndarray:
    if K.shape == (3, 3):
        return K
    if K.ndim == 3 and K.shape[1:] == (3, 3):
        return K[min(index, len(K) - 1)]
    raise ValueError(f"K must be (3,3) or (T,3,3), got {K.shape}")


def _load_render_mesh(path: Path, scale: np.ndarray):
    import pyrender
    import trimesh

    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        mesh = (
            loaded.to_geometry()
            if hasattr(loaded, "to_geometry")
            else loaded.dump(concatenate=True)
        )
    else:
        mesh = loaded
    mesh = mesh.copy()
    mesh.vertices *= scale[None, :]
    if not hasattr(mesh.visual, "material"):
        mesh.visual.vertex_colors = np.tile(
            np.asarray([[242, 166, 51, 255]], dtype=np.uint8),
            (len(mesh.vertices), 1),
        )
    return pyrender.Mesh.from_trimesh(mesh, smooth=False)


def render(args) -> None:
    import pyrender

    motion = _load_motion(args.object_motion)
    if not motion["mesh_path"].exists():
        raise FileNotFoundError(motion["mesh_path"])

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or args.fps)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"invalid video dimensions: {width}x{height}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps or video_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create video: {args.output}")

    mesh = _load_render_mesh(motion["mesh_path"], motion["mesh_scale"])
    renderer = pyrender.OffscreenRenderer(width, height)
    scene = pyrender.Scene(
        bg_color=np.asarray([0.0, 0.0, 0.0, 0.0]),
        ambient_light=np.asarray([0.45, 0.45, 0.45]),
    )
    mesh_node = scene.add(mesh, pose=np.eye(4))

    first_k = _select_k(motion["K"], 0)
    camera = pyrender.IntrinsicsCamera(
        fx=float(first_k[0, 0]),
        fy=float(first_k[1, 1]),
        cx=float(first_k[0, 2]),
        cy=float(first_k[1, 2]),
        znear=args.znear,
        zfar=args.zfar,
    )
    camera_node = scene.add(camera, pose=np.eye(4))
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
    scene.add(light, pose=np.eye(4))

    frame_index = 0
    rendered_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            source_position = (
                (frame_index - motion["frame_offset"])
                * motion["fps"]
                / video_fps
            )
            source_index = int(round(source_position))
            in_range = 0 <= source_index < len(motion["pose_camera_object"])
            if in_range and motion["valid"][source_index]:
                K = _select_k(motion["K"], source_index)
                scene.remove_node(camera_node)
                camera = pyrender.IntrinsicsCamera(
                    fx=float(K[0, 0]),
                    fy=float(K[1, 1]),
                    cx=float(K[0, 2]),
                    cy=float(K[1, 2]),
                    znear=args.znear,
                    zfar=args.zfar,
                )
                camera_node = scene.add(camera, pose=np.eye(4))

                pose_gl = (
                    CV_CAMERA_TO_OPENGL
                    @ motion["pose_camera_object"][source_index]
                )
                scene.set_pose(mesh_node, pose=pose_gl)
                color_rgba, depth = renderer.render(
                    scene,
                    flags=pyrender.RenderFlags.RGBA
                    | pyrender.RenderFlags.SKIP_CULL_FACES,
                )
                mask = depth > 0
                if np.any(mask):
                    rendered_bgr = color_rgba[..., :3][..., ::-1]
                    alpha = np.zeros((height, width, 1), dtype=np.float32)
                    alpha[mask] = float(args.alpha)
                    frame = (
                        frame.astype(np.float32) * (1.0 - alpha)
                        + rendered_bgr.astype(np.float32) * alpha
                    ).clip(0, 255).astype(np.uint8)
                    rendered_count += 1
            writer.write(frame)
            frame_index += 1
    finally:
        cap.release()
        writer.release()
        renderer.delete()

    print(
        f"Saved GVHMR object composite: {args.output} "
        f"({rendered_count}/{frame_index} frames rendered)"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a FoundationPose object mesh into a GVHMR video."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--object_motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.82)
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--znear", type=float, default=0.01)
    parser.add_argument("--zfar", type=float, default=100.0)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be between 0 and 1")
    return args


if __name__ == "__main__":
    render(parse_args())
