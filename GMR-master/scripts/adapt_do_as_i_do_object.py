#!/usr/bin/env python3
"""Convert do-as-i-do RGB-only reconstruction into the common object contract.

Output intentionally matches the small subset consumed by
``run_object_reconstruction_bridge.py``:

* pose.npy: (T, 4, 4) T_camera_object in OpenCV camera coordinates
* pose_valid.npy: directly observed or interpolated frames inside the track
* pose_observed.npy: frames directly observed by the monocular tracker
* mesh/mesh.obj: object mesh with the hand-anchored metric scale applied
* cam_K.json: camera intrinsics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def _default_layout(raw_dir: Path, object_name: str) -> Path:
    base = (
        raw_dir
        / "obj_tracking_out"
        / object_name
        / "combined_visualization"
    )
    optimized = base / "layout_camera_frame_optimized.json"
    if optimized.exists():
        return optimized
    camera_frame = base / "layout_camera_frame.json"
    if camera_frame.exists():
        return camera_frame
    raise FileNotFoundError(
        f"no camera-frame layout found under {base}; finish do-as-i-do stage 4"
    )


def _layout_frames(layout: dict):
    frames_by_index = {}
    for obj in layout.get("objects", []):
        frame_idx = obj.get("frame_index", obj.get("frame_idx"))
        if frame_idx is None:
            continue
        pose = obj.get("local_to_scene", {})
        translation = pose.get("translation_camera_frame")
        quat_wxyz = pose.get("quat_wxyz_camera_frame")
        if translation is None or quat_wxyz is None:
            continue
        frames_by_index[int(frame_idx)] = (
            int(frame_idx),
            np.asarray(translation, dtype=np.float64),
            np.asarray(quat_wxyz, dtype=np.float64),
        )
    frames = sorted(frames_by_index.values(), key=lambda item: item[0])
    if not frames:
        raise ValueError("layout has no camera-frame object poses")
    return frames


def _frame_count(raw_dir: Path, frames) -> int:
    image_files = []
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        image_files.extend((raw_dir / "all_frames").glob(pattern))
    if image_files:
        return len(image_files)
    return frames[-1][0] + 1


def _interpolate_poses(frames, frame_count: int):
    indices = np.asarray([item[0] for item in frames], dtype=np.int64)
    translations = np.stack([item[1] for item in frames])
    quat_wxyz = np.stack([item[2] for item in frames])
    quat_wxyz /= np.clip(
        np.linalg.norm(quat_wxyz, axis=1, keepdims=True),
        1e-12,
        None,
    )
    target = np.arange(frame_count, dtype=np.float64)
    translation_full = np.stack(
        [
            np.interp(target, indices, translations[:, axis])
            for axis in range(3)
        ],
        axis=1,
    )

    rotations = Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]])
    if len(indices) == 1:
        rotation_full = Rotation.from_quat(
            np.repeat(rotations.as_quat(), frame_count, axis=0)
        )
    else:
        clipped_target = np.clip(target, indices[0], indices[-1])
        rotation_full = Slerp(indices, rotations)(clipped_target)

    poses = np.tile(np.eye(4, dtype=np.float64), (frame_count, 1, 1))
    poses[:, :3, :3] = rotation_full.as_matrix()
    poses[:, :3, 3] = translation_full
    observed = np.zeros(frame_count, dtype=bool)
    observed[indices[(indices >= 0) & (indices < frame_count)]] = True
    valid = (
        (np.arange(frame_count) >= indices[0])
        & (np.arange(frame_count) <= indices[-1])
    )
    return poses.astype(np.float32), valid, observed


def _mesh_scale(layout: dict, frames) -> float:
    optimization = layout.get("translation_scale_optimization", {})
    if optimization.get("mesh_scale") is not None:
        return float(optimization["mesh_scale"])
    first_index = frames[0][0]
    for obj in layout.get("objects", []):
        frame_idx = obj.get("frame_index", obj.get("frame_idx"))
        if frame_idx != first_index:
            continue
        scale = obj.get("local_to_scene", {}).get("scale")
        if scale:
            return float(np.asarray(scale).reshape(-1)[0])
    raise ValueError("layout does not contain a mesh scale")


def _default_mesh(
    raw_dir: Path,
    object_name: str,
    layout: dict,
) -> Path:
    optimization = layout.get("translation_scale_optimization", {})
    ref_frame = optimization.get("ref_frame")
    if ref_frame is None:
        raise ValueError(
            "layout has no ref_frame; pass --mesh explicitly"
        )
    path = (
        raw_dir
        / "video_segmentation"
        / "masks"
        / f"frame_{int(ref_frame):06d}_masks"
        / object_name
        / f"{object_name}.obj"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _load_intrinsics(raw_dir: Path, ref_frame: int) -> np.ndarray:
    npy_path = raw_dir / "all_frames" / f"{ref_frame:06d}_intrinsics.npy"
    if npy_path.exists():
        return np.asarray(np.load(npy_path), dtype=np.float64)
    txt_path = raw_dir / f"{ref_frame:04d}_intrinsics.txt"
    if txt_path.exists():
        values = np.loadtxt(txt_path)
        if np.asarray(values).shape == (3, 3):
            return np.asarray(values, dtype=np.float64)
        focal = float(np.asarray(values).reshape(-1)[0])
        image_path = raw_dir / "all_frames" / f"{ref_frame:06d}.png"
        import cv2

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        height, width = image.shape[:2]
        return np.asarray(
            [
                [focal, 0.0, width / 2.0],
                [0.0, focal, height / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )
    raise FileNotFoundError(
        f"intrinsics not found for frame {ref_frame} under {raw_dir}"
    )


def adapt(args) -> None:
    import trimesh

    raw_dir = args.raw_dir.resolve()
    layout_path = (
        args.layout.resolve()
        if args.layout
        else _default_layout(raw_dir, args.object_name)
    )
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    frames = _layout_frames(layout)
    frame_count = args.frame_count or _frame_count(raw_dir, frames)
    poses, valid, observed = _interpolate_poses(frames, frame_count)

    mesh_path = (
        args.mesh.resolve()
        if args.mesh
        else _default_mesh(raw_dir, args.object_name, layout)
    )
    scale = _mesh_scale(layout, frames)
    mesh = trimesh.load(mesh_path, force="scene", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = (
            mesh.to_geometry()
            if hasattr(mesh, "to_geometry")
            else mesh.dump(concatenate=True)
        )
    mesh = mesh.copy()
    mesh.vertices *= scale

    ref_frame = int(
        layout.get("translation_scale_optimization", {}).get(
            "ref_frame",
            frames[0][0],
        )
    )
    K = _load_intrinsics(raw_dir, ref_frame)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = args.output_dir / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    output_mesh = mesh_dir / "mesh.obj"
    mesh.export(output_mesh, file_type="obj")
    np.save(args.output_dir / "pose.npy", poses)
    np.save(args.output_dir / "pose_valid.npy", valid)
    np.save(args.output_dir / "pose_observed.npy", observed)

    image_path = raw_dir / "all_frames" / f"{ref_frame:06d}.png"
    width = height = None
    if image_path.exists():
        import cv2

        image = cv2.imread(str(image_path))
        if image is not None:
            height, width = image.shape[:2]
    camera = {
        "K": K.tolist(),
        "width": width,
        "height": height,
        "data_source": "do_as_i_do_rgb_monocular",
        "scale_source": "HaWoR hand anchored MoGe pointmap",
    }
    (args.output_dir / "cam_K.json").write_text(
        json.dumps(camera, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "source": "do_as_i_do_rgb_monocular",
        "raw_dir": str(raw_dir),
        "layout": str(layout_path),
        "source_mesh": str(mesh_path),
        "metric_mesh": str(output_mesh.resolve()),
        "mesh_scale_applied": scale,
        "frame_count": frame_count,
        "valid_frames": int(valid.sum()),
        "directly_observed_frames": int(observed.sum()),
        "interpolated_valid_frames": int(valid.sum() - observed.sum()),
        "pose_convention": "T_camera_object",
        "camera_convention": "opencv_x_right_y_down_z_forward",
        "scale_warning": (
            "Scale is inferred from monocular pointmaps anchored to HaWoR hand "
            "geometry; validate it before contact simulation."
        ),
    }
    (args.output_dir / "adapter_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved RGB-only object adapter: {args.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Adapt do-as-i-do RGB-only output to the common object contract."
    )
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--object_name", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--layout", type=Path, default=None)
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--frame_count", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    adapt(parse_args())
