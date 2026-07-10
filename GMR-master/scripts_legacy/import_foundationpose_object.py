#!/usr/bin/env python3
"""Import FoundationPose mesh/poses into the GVHMR + GMR object contract.

FoundationPose stores ``T_camera_object``. GVHMR stores ``T_world_to_camera``.
This script composes those transforms, applies the GVHMR alignment offset, and
then maps the object's pelvis-relative trajectory into the z-up GMR robot world.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _scalar(data, key, default):
    if key not in data:
        return default
    value = np.asarray(data[key]).reshape(-1)
    return value[0].item() if value.size else default


def _find_mesh(root: Path) -> Path:
    mesh_dir = root / "mesh"
    for name in ("mesh.obj", "mesh.ply", "mesh.stl", "mesh.glb"):
        candidate = mesh_dir / name
        if candidate.exists():
            return candidate.resolve()
    for suffix in ("*.obj", "*.ply", "*.stl", "*.glb"):
        candidates = sorted(mesh_dir.glob(suffix))
        if candidates:
            return candidates[0].resolve()
    raise FileNotFoundError(f"no mesh found under {mesh_dir}")


def _load_collision_manifest(path: Path | None) -> dict:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_intrinsics(path: Path | None, foundation_dir: Path, fallback) -> np.ndarray:
    selected = path
    if selected is None and (foundation_dir / "cam_K.json").exists():
        selected = foundation_dir / "cam_K.json"
    if selected is None:
        return np.asarray(fallback, dtype=np.float32)
    selected = selected.resolve()
    if selected.suffix.lower() == ".json":
        values = json.loads(selected.read_text(encoding="utf-8"))
        values = values.get("K", values)
        return np.asarray(values, dtype=np.float32)
    if selected.suffix.lower() == ".npy":
        return np.asarray(
            np.load(selected, allow_pickle=False),
            dtype=np.float32,
        )
    return np.asarray(np.loadtxt(selected), dtype=np.float32)


def _validate_poses(poses: np.ndarray, label: str) -> None:
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"{label} must have shape (T, 4, 4), got {poses.shape}")
    if not np.isfinite(poses).all():
        raise ValueError(f"{label} contains NaN or Inf")
    bottom = poses[:, 3]
    expected = np.asarray([0.0, 0.0, 0.0, 1.0])
    if not np.allclose(bottom, expected, atol=1e-4):
        raise ValueError(f"{label} contains invalid homogeneous transforms")


def _resolve_source_fps(
    explicit_fps: float | None,
    adapter_manifest: dict,
    robot_fps: float,
) -> float:
    source_fps = float(
        explicit_fps
        or adapter_manifest.get("fps")
        or robot_fps
    )
    if not np.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError(f"source FPS must be positive and finite, got {source_fps}")
    return source_fps


def export(args) -> None:
    foundation_dir = args.foundation_dir.resolve()
    clip_dir = args.clip_dir.resolve()
    default_pose = foundation_dir / "pose_optimized.npy"
    if not default_pose.exists():
        # Native FoundationPose exports this source filename.  The monocular
        # adapter itself always uses the explicit raw/optimized contract.
        default_pose = foundation_dir / "pose.npy"
    pose_path = (args.pose_path or default_pose).resolve()
    mesh_path = (args.mesh_path or _find_mesh(foundation_dir)).resolve()
    camera_path = (args.gvhmr_camera or clip_dir / "gvhmr_camera.npz").resolve()
    robot_motion_path = (args.robot_motion or clip_dir / "robot_motion.pkl").resolve()

    for required in (pose_path, mesh_path, camera_path, robot_motion_path):
        if not required.exists():
            raise FileNotFoundError(required)

    poses_camera = np.asarray(
        np.load(pose_path, allow_pickle=False),
        dtype=np.float64,
    )
    _validate_poses(poses_camera, "optimized object trajectory")
    if args.pose_convention == "camera_to_object":
        poses_camera = np.linalg.inv(poses_camera)
    pose_valid_path = foundation_dir / "pose_valid.npy"
    if pose_valid_path.exists():
        pose_valid = np.asarray(
            np.load(pose_valid_path, allow_pickle=False),
            dtype=bool,
        ).reshape(-1)
        if len(pose_valid) != len(poses_camera):
            raise ValueError(
                f"{pose_valid_path} has {len(pose_valid)} entries for "
                f"{len(poses_camera)} poses"
            )
    else:
        pose_valid = np.ones(len(poses_camera), dtype=bool)
    source_label = "foundationpose"
    adapter_manifest = {}
    adapter_manifest_path = foundation_dir / "adapter_manifest.json"
    if adapter_manifest_path.exists():
        adapter_manifest = json.loads(
            adapter_manifest_path.read_text(encoding="utf-8")
        )
        source_label = str(adapter_manifest.get("source", source_label))

    with np.load(camera_path, allow_pickle=False) as camera_npz:
        camera = {key: camera_npz[key] for key in camera_npz.files}
    world_to_camera = np.asarray(camera["T_w2c"], dtype=np.float64)
    _validate_poses(world_to_camera, "gvhmr_camera T_w2c")
    subject_world = np.asarray(camera["subject_world"], dtype=np.float64)
    world_to_zup = np.asarray(camera["world_to_isaac"], dtype=np.float64)
    alignment_offset = np.asarray(
        camera.get("alignment_offset_world", np.zeros(3)),
        dtype=np.float64,
    ).reshape(3)

    with robot_motion_path.open("rb") as file:
        robot_motion = pickle.load(file)
    robot_root = np.asarray(robot_motion["root_pos"], dtype=np.float64)
    robot_fps = float(np.asarray(robot_motion.get("fps", 30.0)).reshape(-1)[0])

    start = int(args.frame_offset)
    if start < 0 or start >= len(world_to_camera):
        raise ValueError(f"--frame_offset {start} is outside GVHMR camera sequence")
    world_to_camera = world_to_camera[start:]
    subject_world = subject_world[start:]

    source_count = min(
        len(poses_camera),
        len(world_to_camera),
        len(subject_world),
    )
    if source_count <= 0:
        raise ValueError("no overlapping FoundationPose/GVHMR frames")
    poses_camera = poses_camera[:source_count]
    pose_valid = pose_valid[:source_count]
    world_to_camera = world_to_camera[:source_count]
    subject_world = subject_world[:source_count]

    poses_world = np.linalg.inv(world_to_camera) @ poses_camera
    poses_world[:, :3, 3] += alignment_offset[None, :]

    yaw = Rotation.from_euler(
        "z",
        float(args.human_yaw_offset_deg),
        degrees=True,
    ).as_matrix()
    world_to_robot_axes = yaw @ world_to_zup

    source_fps = _resolve_source_fps(
        args.source_fps,
        adapter_manifest,
        robot_fps,
    )
    source_positions = (
        (np.arange(len(robot_root), dtype=np.float64) - start)
        * source_fps
        / robot_fps
    )
    robot_valid_time = (
        (source_positions >= 0.0)
        & (source_positions <= float(source_count - 1))
    )
    robot_indices = np.clip(
        np.rint(source_positions).astype(np.int64),
        0,
        source_count - 1,
    )
    sampled_world = poses_world[robot_indices]
    sampled_subject = subject_world[robot_indices]

    object_zup = sampled_world[:, :3, 3] @ world_to_zup.T
    subject_zup = sampled_subject @ world_to_zup.T
    relative_zup = object_zup - subject_zup
    position_robot = robot_root + relative_zup @ yaw.T
    position_robot += np.asarray(args.robot_offset, dtype=np.float64)[None, :]

    rotation_robot = (
        world_to_robot_axes[None, :, :] @ sampled_world[:, :3, :3]
    )
    quat_xyzw = Rotation.from_matrix(rotation_robot).as_quat()
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]

    valid_camera = (
        pose_valid
        & np.isfinite(poses_camera).all(axis=(1, 2))
        & (poses_camera[:, 2, 3] > 0.001)
    )
    valid_robot = valid_camera[robot_indices] & robot_valid_time

    # --- Sanity checks ---
    warnings_list: list[str] = []
    # Object-human distance check.
    obj_human_dist = np.linalg.norm(
        object_zup - subject_zup, axis=1
    )
    median_dist = float(np.median(obj_human_dist))
    if median_dist > 5.0:
        warnings_list.append(
            f"median object-human distance {median_dist:.2f}m > 5m; "
            "coordinate chain may be broken"
        )

    # Object height check (should not be far below ground).
    object_heights = object_zup[:, 2]
    below_ground = object_heights < -1.0
    if below_ground.mean() > 0.3:
        warnings_list.append(
            f"{below_ground.mean():.1%} of object positions are below z=-1m"
        )

    # Quaternion norm check.
    quat_norms = np.linalg.norm(quat_wxyz, axis=1)
    bad_quats = np.abs(quat_norms - 1.0) > 0.01
    if bad_quats.any():
        warnings_list.append(
            f"{bad_quats.sum()} quaternions have non-unit norm"
        )

    # FPS mapping sanity.
    if source_fps > 0 and robot_fps > 0:
        ratio = source_fps / robot_fps
        if ratio < 0.1 or ratio > 10.0:
            warnings_list.append(
                f"source_fps ({source_fps}) / robot_fps ({robot_fps}) = {ratio:.2f} "
                "is unusual"
            )

    collision = _load_collision_manifest(args.collision_manifest)
    collision_paths = collision.get("collision_mesh_paths", [])
    density = float(collision.get("density_kg_m3", args.density))
    mass_kg = collision.get("mass_kg")
    center_of_mass = collision.get("center_of_mass_m")
    diaginertia = collision.get("diaginertia_kg_m2")
    validation_mode = collision.get("validation_mode", "visual_only")
    if validation_mode == "physics":
        if mass_kg is None or center_of_mass is None or diaginertia is None:
            raise ValueError(
                "physics collision manifest lacks mass/COM/inertia"
            )
    friction = np.asarray(
        collision.get("friction", args.friction),
        dtype=np.float32,
    )
    solref = np.asarray(
        collision.get("solref", args.solref),
        dtype=np.float32,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gvhmr_output = args.output_dir / "object_motion_gvhmr.npz"
    gmr_output = args.output_dir / "object_motion_gmr.npz"

    common = {
        "schema_version": np.asarray(2, dtype=np.int32),
        "source": np.asarray(source_label),
        "units": np.asarray("meter"),
        "object_type": np.asarray("mesh"),
        "visual_mesh_path": np.asarray(str(mesh_path)),
        "collision_mesh_paths": np.asarray(collision_paths, dtype=str),
        "mesh_scale": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        "density": np.asarray(density, dtype=np.float32),
        "mass_kg": np.asarray(
            float(mass_kg) if mass_kg is not None else 0.0,
            dtype=np.float32,
        ),
        "center_of_mass_m": np.asarray(
            center_of_mass if center_of_mass is not None else [0.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        "diaginertia_kg_m2": np.asarray(
            diaginertia if diaginertia is not None else [0.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        "mass_source": np.asarray(
            collision.get("mass_source", "unavailable")
        ),
        "inertia_source": np.asarray(
            collision.get("inertia_source", "unavailable")
        ),
        "physics_asset_mode": np.asarray(validation_mode),
        "friction": friction,
        "solref": solref,
        "rgba": np.asarray(args.rgba, dtype=np.float32),
    }
    np.savez(
        gvhmr_output,
        **common,
        coordinate_system=np.asarray("opencv_camera"),
        fps=np.asarray(source_fps, dtype=np.float32),
        gvhmr_frame_offset=np.asarray(start, dtype=np.int32),
        pose_camera_object=poses_camera.astype(np.float32),
        pose_gvhmr_world_object=poses_world.astype(np.float32),
        valid=valid_camera,
        K=_load_intrinsics(
            args.foundation_intrinsics,
            foundation_dir,
            camera["K_fullimg"],
        ),
    )
    np.savez(
        gmr_output,
        **common,
        coordinate_system=np.asarray("mujoco_robot_world_zup"),
        fps=np.asarray(robot_fps, dtype=np.float32),
        position=position_robot.astype(np.float32),
        quat_wxyz=quat_wxyz.astype(np.float32),
        valid=valid_robot,
        source_frame_index=robot_indices.astype(np.int32),
        pose_gvhmr_world_object=sampled_world.astype(np.float32),
        subject_gvhmr_world=sampled_subject.astype(np.float32),
        robot_root=robot_root.astype(np.float32),
        human_yaw_offset_deg=np.asarray(
            args.human_yaw_offset_deg,
            dtype=np.float32,
        ),
    )

    # Save coordinate debug npz.
    debug_path = args.output_dir / "object_transform_debug.npz"
    np.savez(
        debug_path,
        poses_camera=poses_camera.astype(np.float32),
        world_to_camera=world_to_camera.astype(np.float32),
        poses_world_before_offset=(
            (np.linalg.inv(world_to_camera) @ poses_camera).astype(np.float32)
        ),
        poses_world_after_offset=poses_world.astype(np.float32),
        subject_world=subject_world.astype(np.float32),
        object_zup=object_zup.astype(np.float32),
        subject_zup=subject_zup.astype(np.float32),
        relative_zup=relative_zup.astype(np.float32),
        robot_root=robot_root.astype(np.float32),
        position_robot=position_robot.astype(np.float32),
        rotation_robot=rotation_robot.astype(np.float32),
        valid_camera=valid_camera,
        valid_robot=valid_robot,
    )
    print(f"Saved coordinate debug: {debug_path}")

    # Coordinate debug metadata for manifest.
    coordinate_debug = {
        "input_pose_convention": "T_camera_object",
        "camera_convention": "opencv_x_right_y_down_z_forward",
        "gvhmr_T_w2c_mean_z": round(
            float(np.mean(world_to_camera[:, 2, 3])), 3
        ),
        "alignment_offset_world": alignment_offset.tolist(),
        "world_to_isaac": world_to_zup.tolist(),
        "human_yaw_offset_deg": args.human_yaw_offset_deg,
        "robot_offset": list(args.robot_offset),
    }

    manifest = {
        "schema_version": 2,
        "foundation_dir": str(foundation_dir),
        "source": source_label,
        "foundation_pose": str(pose_path),
        "visual_mesh": str(mesh_path),
        "gvhmr_camera": str(camera_path),
        "robot_motion": str(robot_motion_path),
        "source_frames": source_count,
        "robot_frames": len(robot_root),
        "source_fps": source_fps,
        "robot_fps": robot_fps,
        "pose_convention": "T_camera_object",
        "units": "meter",
        "coordinate_debug": coordinate_debug,
        "collision_manifest": (
            str(args.collision_manifest.resolve())
            if args.collision_manifest
            else None
        ),
        "outputs": {
            "gvhmr": str(gvhmr_output),
            "gmr": str(gmr_output),
            "transform_debug": str(debug_path),
        },
        "warnings": [
            "GMR mapping preserves object position relative to the GVHMR subject root.",
            "Verify projected alignment before enabling dynamic contact simulation.",
        ] + warnings_list,
    }
    manifest_path = args.output_dir / "object_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved GVHMR object motion: {gvhmr_output}")
    print(f"Saved GMR object motion: {gmr_output}")
    print(f"Saved object manifest: {manifest_path}")


def _vec3(value: str) -> tuple[float, float, float]:
    parts = tuple(float(item) for item in value.replace(",", " ").split())
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three numbers")
    return parts


def _vec4(value: str) -> tuple[float, float, float, float]:
    parts = tuple(float(item) for item in value.replace(",", " ").split())
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("expected four numbers")
    return parts


def _vec2(value: str) -> tuple[float, float]:
    parts = tuple(float(item) for item in value.replace(",", " ").split())
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected two numbers")
    return parts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Map FoundationPose object reconstruction into GVHMR/GMR."
    )
    parser.add_argument("--foundation_dir", type=Path, required=True)
    parser.add_argument("--clip_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--pose_path", type=Path, default=None)
    parser.add_argument("--mesh_path", type=Path, default=None)
    parser.add_argument("--gvhmr_camera", type=Path, default=None)
    parser.add_argument("--robot_motion", type=Path, default=None)
    parser.add_argument("--collision_manifest", type=Path, default=None)
    parser.add_argument("--foundation_intrinsics", type=Path, default=None)
    parser.add_argument(
        "--pose_convention",
        choices=["object_to_camera", "camera_to_object"],
        default="object_to_camera",
    )
    parser.add_argument(
        "--frame_offset",
        type=int,
        default=0,
        help="GVHMR frame corresponding to FoundationPose pose frame 0.",
    )
    parser.add_argument("--source_fps", type=float, default=None)
    parser.add_argument("--human_yaw_offset_deg", type=float, default=0.0)
    parser.add_argument("--robot_offset", type=_vec3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--rgba", type=_vec4, default=(0.95, 0.65, 0.2, 0.9))
    parser.add_argument("--density", type=float, default=600.0)
    parser.add_argument("--friction", type=_vec3, default=(1.0, 0.05, 0.005))
    parser.add_argument("--solref", type=_vec2, default=(0.01, 1.0))
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())
