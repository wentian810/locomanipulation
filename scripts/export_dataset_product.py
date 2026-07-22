#!/usr/bin/env python3
"""Export compact, safe, reproducible per-clip digital-asset bundles.

The pipeline work directory may contain trusted-only pickle/PT caches and many
diagnostic files.  This exporter converts the final numerical results into
compressed NPZ files that load with ``allow_pickle=False`` and optionally
removes the work directory only after the exported bundle passes verification.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from source_provenance import build_source_provenance


SCHEMA_VERSION = 1
BODY_SOURCE_FILES = {
    "converted": "001_converted.npz",
    "smoothed": "001_smoothed.npz",
    "phc_smoothed": "001_phc_smoothed.npz",
    "phc_smoothed_grounded": "001_phc_smoothed_grounded.npz",
    "final": "001_final.npz",
}
ROBOT_MODELS = {
    "sharpa": (
        "unitree_g1",
        "GMR-master/assets/unitree_g1/g1_mocap_29dof.xml",
    ),
    "sharpa_g1": (
        "unitree_g1",
        "GMR-master/assets/unitree_g1/g1_mocap_29dof.xml",
    ),
    "sharpa_h1": (
        "unitree_h1_with_hand",
        "GMR-master/assets/unitree_h1/h1_with_hand.xml",
    ),
    "g1": (
        "unitree_g1_with_hands",
        "GMR-master/assets/unitree_g1/g1_mocap_29dof_with_hands.xml",
    ),
    "brainco": (
        "unitree_g1",
        "GMR-master/assets/unitree_g1/g1_mocap_29dof.xml",
    ),
}
MESH_SUFFIXES = {".obj", ".mtl", ".png", ".jpg", ".jpeg", ".webp"}


class ProductExportError(RuntimeError):
    """Raised when an asset bundle cannot be exported safely."""


def _json_read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProductExportError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProductExportError(f"JSON root is not an object: {path}")
    return value


def _json_write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _portable_value(value: Any) -> Any:
    """Remove producer-machine absolute paths from distributable metadata."""
    if isinstance(value, dict):
        return {str(key): _portable_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_value(item) for item in value]
    if isinstance(value, tuple):
        return [_portable_value(item) for item in value]
    if isinstance(value, str):
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            return candidate.name
    return value


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise ProductExportError(f"required NPZ is missing: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: np.asarray(archive[key]) for key in archive.files}
    except (OSError, ValueError) as exc:
        raise ProductExportError(
            f"NPZ must load with allow_pickle=False: {path}: {exc}"
        ) from exc


def _require_keys(data: dict[str, np.ndarray], keys: Iterable[str], label: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ProductExportError(f"{label} is missing fields: {', '.join(missing)}")


def _frame_count(array: np.ndarray, label: str) -> int:
    if array.ndim < 1 or array.shape[0] <= 0:
        raise ProductExportError(f"{label} has no frames: shape={array.shape}")
    return int(array.shape[0])


def _validate_numeric(name: str, value: np.ndarray) -> None:
    if value.dtype == object:
        raise ProductExportError(f"{name} uses forbidden object dtype")
    if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
        raise ProductExportError(f"{name} contains NaN or Inf")


def _validate_transforms(
    name: str,
    value: np.ndarray,
    valid: np.ndarray | None = None,
) -> None:
    transforms = np.asarray(value, dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ProductExportError(
            f"{name} must have shape (T,4,4), got {transforms.shape}"
        )
    selected = transforms
    if valid is not None:
        mask = np.asarray(valid, dtype=bool)
        if mask.shape != (len(transforms),):
            raise ProductExportError(f"{name} valid mask has shape {mask.shape}")
        selected = transforms[mask]
    if not len(selected):
        return
    _validate_numeric(name, selected)
    expected_bottom = np.asarray([0.0, 0.0, 0.0, 1.0])
    if not np.allclose(selected[:, 3, :], expected_bottom, atol=1e-4):
        raise ProductExportError(f"{name} has invalid homogeneous bottom rows")
    rotation = selected[:, :3, :3]
    identity = np.eye(3)[None]
    if not np.allclose(
        rotation @ np.swapaxes(rotation, 1, 2),
        identity,
        atol=2e-3,
    ):
        raise ProductExportError(f"{name} rotations are not orthonormal")
    if not np.allclose(np.linalg.det(rotation), 1.0, atol=2e-3):
        raise ProductExportError(f"{name} rotations do not have determinant +1")


def _save_npz(path: Path, values: dict[str, Any]) -> None:
    safe: dict[str, np.ndarray] = {}
    for key, value in values.items():
        array = np.asarray(value)
        _validate_numeric(key, array)
        safe[key] = array
    np.savez_compressed(path, **safe)
    # Re-open with the exact security setting expected from consumers.
    _load_npz(path)


def _scalar(data: dict[str, np.ndarray], key: str, default: Any = None) -> Any:
    if key not in data:
        return default
    value = np.asarray(data[key])
    if value.size != 1:
        raise ProductExportError(f"{key} must be scalar, got shape={value.shape}")
    return value.reshape(-1)[0].item()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolve_body_source(clip_dir: Path, source: str) -> tuple[Path, str]:
    source = str(source)
    if source == "auto":
        candidates = ("smoothed", "converted", "phc_smoothed")
    elif source in BODY_SOURCE_FILES:
        candidates = (source,)
    else:
        raise ProductExportError(f"unsupported gmr.source: {source!r}")
    for candidate in candidates:
        path = clip_dir / BODY_SOURCE_FILES[candidate]
        if path.is_file():
            stage = candidate
            if candidate == "final":
                resolved_name = path.resolve().name
                stage = next(
                    (
                        name
                        for name, filename in BODY_SOURCE_FILES.items()
                        if name != "final" and filename == resolved_name
                    ),
                    "final",
                )
            return path, stage
    expected = ", ".join(BODY_SOURCE_FILES[item] for item in candidates)
    raise ProductExportError(f"no selected body source in {clip_dir}; expected {expected}")


def _export_human(
    clip_dir: Path,
    bundle_dir: Path,
    gmr_source: str,
) -> tuple[dict[str, Any], int, float]:
    body_path, source_stage = _resolve_body_source(clip_dir, gmr_source)
    hands_path = clip_dir / "001_smplx_hands.npz"
    body = _load_npz(body_path)
    hands = _load_npz(hands_path)
    _require_keys(
        body,
        ("root_orient", "pose_body", "trans", "betas", "mocap_frame_rate"),
        body_path.name,
    )
    _require_keys(
        hands,
        (
            "left_hand_pose",
            "right_hand_pose",
            "left_hand_valid",
            "right_hand_valid",
        ),
        hands_path.name,
    )
    frames = _frame_count(body["trans"], "human translation")
    fps = float(_scalar(body, "mocap_frame_rate", 30.0))
    if fps <= 0:
        raise ProductExportError(f"invalid human fps: {fps}")
    if body["root_orient"].shape != (frames, 3):
        raise ProductExportError(
            f"root_orient must be ({frames}, 3), got {body['root_orient'].shape}"
        )
    if body["pose_body"].shape != (frames, 63):
        raise ProductExportError(
            f"pose_body must be ({frames}, 63), got {body['pose_body'].shape}"
        )
    for key in ("left_hand_pose", "right_hand_pose"):
        if hands[key].shape != (frames, 45):
            raise ProductExportError(
                f"{key} must be ({frames}, 45), got {hands[key].shape}"
            )
    for key in ("left_hand_valid", "right_hand_valid"):
        if hands[key].shape != (frames,):
            raise ProductExportError(
                f"{key} must be ({frames},), got {hands[key].shape}"
            )
    for key in (
        "left_hand_pose",
        "right_hand_pose",
        "left_hand_valid",
        "right_hand_valid",
    ):
        if _frame_count(hands[key], f"hands.{key}") != frames:
            raise ProductExportError(
                f"human/hand frame mismatch: human={frames}, {key}={len(hands[key])}"
            )

    output: dict[str, Any] = {
        "schema_version": np.int32(SCHEMA_VERSION),
        "fps": np.float32(fps),
        "frame_count": np.int32(frames),
        "units": np.asarray("meter_radian"),
        "coordinate_system": np.asarray("gvhmr_world_gravity_negative_y"),
        "rotation_representation": np.asarray("axis_angle_radians"),
        "model_type": np.asarray("smplh"),
        "source_body_stage": np.asarray(source_stage),
        "root_orient_axis_angle": body["root_orient"].astype(np.float32),
        "body_pose_axis_angle": body["pose_body"].astype(np.float32),
        "translation": body["trans"].astype(np.float32),
        "betas": body["betas"].astype(np.float32),
        "gender": np.asarray(str(_scalar(body, "gender", "neutral"))),
        "left_hand_pose_axis_angle": hands["left_hand_pose"].astype(np.float32),
        "right_hand_pose_axis_angle": hands["right_hand_pose"].astype(np.float32),
        "left_hand_valid": hands["left_hand_valid"].astype(bool),
        "right_hand_valid": hands["right_hand_valid"].astype(bool),
        "hand_backend": np.asarray(str(_scalar(hands, "hand_backend", "unknown"))),
    }
    optional = {
        "left_hand_quality": np.float32,
        "right_hand_quality": np.float32,
        "left_hand_source_reliable": bool,
        "right_hand_source_reliable": bool,
        "left_hand_source_repaired": bool,
        "right_hand_source_repaired": bool,
        "left_hand_bad_mask": bool,
        "right_hand_bad_mask": bool,
        "left_hand_spike_mask": bool,
        "right_hand_spike_mask": bool,
        "left_hand_bbox_xyxy": np.float32,
        "right_hand_bbox_xyxy": np.float32,
        "left_hand_reproj_error": np.float32,
        "right_hand_reproj_error": np.float32,
    }
    for key, dtype in optional.items():
        if key in hands:
            if _frame_count(hands[key], f"hands.{key}") != frames:
                raise ProductExportError(f"hands.{key} frame count is inconsistent")
            value = hands[key].astype(dtype)
            # Missing 2D observations are encoded as -1 in the product rather
            # than NaN so every numerical field has a strict finite contract.
            if np.issubdtype(value.dtype, np.floating):
                value = np.nan_to_num(
                    value,
                    nan=-1.0,
                    posinf=-1.0,
                    neginf=-1.0,
                )
            output[key] = value
    _save_npz(bundle_dir / "human_motion.npz", output)

    quality = {
        "source_body_stage": source_stage,
        "source_body_path": body_path.name,
        "frames": frames,
        "fps": fps,
        "left_hand_valid_ratio": float(np.mean(output["left_hand_valid"])),
        "right_hand_valid_ratio": float(np.mean(output["right_hand_valid"])),
    }
    for side in ("left", "right"):
        key = f"{side}_hand_quality"
        if key in output:
            quality[f"{side}_hand_quality_mean"] = float(np.mean(output[key]))
    return quality, frames, fps


def _export_phc_motion(
    clip_dir: Path,
    bundle_dir: Path,
    expected_frames: int,
    expected_fps: float,
) -> dict[str, Any]:
    path = clip_dir / "001_phc_smoothed.npz"
    data = _load_npz(path)
    _require_keys(
        data,
        ("root_orient", "pose_body", "trans", "betas", "mocap_frame_rate"),
        path.name,
    )
    if data["root_orient"].shape != (expected_frames, 3):
        raise ProductExportError(
            f"PHC root_orient has invalid shape: {data['root_orient'].shape}"
        )
    if data["pose_body"].shape != (expected_frames, 63):
        raise ProductExportError(
            f"PHC pose_body has invalid shape: {data['pose_body'].shape}"
        )
    if data["trans"].shape != (expected_frames, 3):
        raise ProductExportError(
            f"PHC trans has invalid shape: {data['trans'].shape}"
        )
    fps = float(_scalar(data, "mocap_frame_rate", expected_fps))
    if abs(fps - expected_fps) > 1e-3:
        raise ProductExportError(
            f"human/PHC fps mismatch: human={expected_fps}, PHC={fps}"
        )
    _save_npz(
        bundle_dir / "human_phc_motion.npz",
        {
            "schema_version": np.int32(SCHEMA_VERSION),
            "fps": np.float32(fps),
            "frame_count": np.int32(expected_frames),
            "units": np.asarray("meter_radian"),
            "coordinate_system": np.asarray(
                "gvhmr_world_gravity_negative_y"
            ),
            "rotation_representation": np.asarray("axis_angle_radians"),
            "model_type": np.asarray("smplh"),
            "source_body_stage": np.asarray("phc_smoothed"),
            "root_orient_axis_angle": data["root_orient"].astype(np.float32),
            "body_pose_axis_angle": data["pose_body"].astype(np.float32),
            "translation": data["trans"].astype(np.float32),
            "betas": data["betas"].astype(np.float32),
            "gender": np.asarray(str(_scalar(data, "gender", "neutral"))),
        },
    )
    return {"frames": expected_frames, "fps": fps, "source": path.name}


def _xml_dof_names(xml_path: Path, expected: int) -> np.ndarray:
    if not xml_path.is_file():
        raise ProductExportError(f"robot XML is missing: {xml_path}")
    try:
        worldbody = ET.parse(xml_path).getroot().find("worldbody")
    except ET.ParseError as exc:
        raise ProductExportError(f"invalid robot XML {xml_path}: {exc}") from exc
    if worldbody is None:
        raise ProductExportError(f"robot XML has no worldbody: {xml_path}")
    names = [
        str(node.attrib["name"])
        for node in worldbody.iter("joint")
        if node.attrib.get("name")
    ]
    if len(names) != expected:
        raise ProductExportError(
            f"robot XML DOF count mismatch: motion={expected}, XML={len(names)}"
        )
    return np.asarray(names)


def _export_robot(
    project_root: Path,
    clip_dir: Path,
    bundle_dir: Path,
    hand_model: str,
    expected_frames: int,
    expected_fps: float,
    human_path: Path,
    human_stage: str,
) -> dict[str, Any]:
    pkl_path = clip_dir / "robot_motion.pkl"
    if not pkl_path.is_file():
        raise ProductExportError(f"required trusted work file is missing: {pkl_path}")
    # This pickle is loaded only inside the trusted producer workspace.  It is
    # never copied to the distributable bundle.
    try:
        with pkl_path.open("rb") as stream:
            motion = pickle.load(stream)
    except Exception as exc:
        raise ProductExportError(f"cannot load trusted robot motion {pkl_path}: {exc}") from exc
    if not isinstance(motion, dict):
        raise ProductExportError(f"robot motion is not a dictionary: {pkl_path}")
    source_provenance = build_source_provenance(
        clip_dir,
        human_path,
        human_stage,
        motion,
    )
    for key in ("root_pos", "root_rot", "dof_pos"):
        if key not in motion:
            raise ProductExportError(f"robot_motion.pkl is missing {key}")
    root_pos = np.asarray(motion["root_pos"], dtype=np.float32)
    root_quat = np.asarray(motion["root_rot"], dtype=np.float32)
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float32)
    frames = _frame_count(root_pos, "robot root_pos")
    if frames != expected_frames:
        raise ProductExportError(
            f"human/robot frame mismatch: human={expected_frames}, robot={frames}"
        )
    if root_pos.shape != (frames, 3) or root_quat.shape != (frames, 4):
        raise ProductExportError(
            f"invalid robot root shapes: pos={root_pos.shape}, quat={root_quat.shape}"
        )
    if dof_pos.ndim != 2 or dof_pos.shape[0] != frames:
        raise ProductExportError(f"invalid robot dof_pos shape: {dof_pos.shape}")
    quat_norm = np.linalg.norm(root_quat, axis=1)
    if not np.allclose(quat_norm, 1.0, atol=1e-3):
        raise ProductExportError(
            f"robot root quaternion norm error: max={np.max(np.abs(quat_norm - 1))}"
        )
    fps = float(np.asarray(motion.get("fps", expected_fps)).reshape(-1)[0])
    if abs(fps - expected_fps) > 1e-3:
        raise ProductExportError(
            f"human/robot fps mismatch: human={expected_fps}, robot={fps}"
        )
    if hand_model not in ROBOT_MODELS:
        raise ProductExportError(f"unsupported gmr.hand_model: {hand_model}")
    embodiment, relative_xml = ROBOT_MODELS[hand_model]
    if "dof_names" in motion:
        dof_names = np.asarray(motion["dof_names"], dtype=str)
        if dof_names.shape != (dof_pos.shape[1],):
            raise ProductExportError("robot motion dof_names do not match dof_pos")
    else:
        dof_names = _xml_dof_names(
            project_root / relative_xml,
            dof_pos.shape[1],
        )

    _save_npz(
        bundle_dir / "robot_motion.npz",
        {
            "schema_version": np.int32(SCHEMA_VERSION),
            "fps": np.float32(fps),
            "frame_count": np.int32(frames),
            "units": np.asarray("meter_radian"),
            "coordinate_system": np.asarray("mujoco_world_z_up"),
            "root_quaternion_order": np.asarray("xyzw"),
            "embodiment": np.asarray(embodiment),
            "root_position": root_pos,
            "root_quat_xyzw": root_quat,
            "dof_position": dof_pos,
            "dof_names": dof_names,
            "human_yaw_offset_deg": np.float32(
                float(motion.get("human_yaw_offset_deg", 0.0))
            ),
        },
    )

    hands_path = clip_dir / "001_sharpa_chain_hands.npz"
    hand_exported = False
    if hand_model in {"sharpa", "sharpa_g1", "sharpa_h1"}:
        hands = _load_npz(hands_path)
        _require_keys(
            hands,
            (
                "left_hand_qpos",
                "right_hand_qpos",
                "left_hand_qpos_names",
                "right_hand_qpos_names",
                "left_valid",
                "right_valid",
            ),
            hands_path.name,
        )
        for key in ("left_hand_qpos", "right_hand_qpos", "left_valid", "right_valid"):
            if _frame_count(hands[key], f"robot hands.{key}") != frames:
                raise ProductExportError(f"robot hands.{key} frame count is inconsistent")
        hand_output: dict[str, Any] = {
            "schema_version": np.int32(SCHEMA_VERSION),
            "fps": np.float32(fps),
            "frame_count": np.int32(frames),
            "units": np.asarray("radian"),
            "embodiment": np.asarray("sharpa_dual_hand"),
            "left_qpos": hands["left_hand_qpos"].astype(np.float32),
            "right_qpos": hands["right_hand_qpos"].astype(np.float32),
            "left_qpos_names": hands["left_hand_qpos_names"].astype(str),
            "right_qpos_names": hands["right_hand_qpos_names"].astype(str),
            "left_valid": hands["left_valid"].astype(bool),
            "right_valid": hands["right_valid"].astype(bool),
        }
        for key in ("left_reliability", "right_reliability"):
            if key in hands:
                hand_output[key] = hands[key].astype(np.float32)
        _save_npz(bundle_dir / "robot_hand_motion.npz", hand_output)
        hand_exported = True
    return {
        "frames": frames,
        "fps": fps,
        "embodiment": embodiment,
        "dof_count": int(dof_pos.shape[1]),
        "separate_robot_hand_motion": hand_exported,
        "source_provenance": source_provenance,
    }


def _export_camera(
    clip_dir: Path,
    bundle_dir: Path,
    expected_frames: int,
    expected_fps: float,
) -> dict[str, Any]:
    camera = _load_npz(clip_dir / "gvhmr_camera.npz")
    required = (
        "camera_pos_world",
        "camera_target_world",
        "camera_pos_isaac",
        "camera_target_isaac",
        "subject_world",
        "subject_isaac",
        "T_w2c",
        "K_fullimg",
        "world_to_isaac",
        "alignment_offset_world",
        "gravity_axis",
    )
    _require_keys(camera, required, "gvhmr_camera.npz")
    for key in required[:8]:
        if _frame_count(camera[key], f"camera.{key}") != expected_frames:
            raise ProductExportError(f"camera.{key} frame count is inconsistent")
    if camera["T_w2c"].shape != (expected_frames, 4, 4):
        raise ProductExportError(f"invalid T_w2c shape: {camera['T_w2c'].shape}")
    if camera["K_fullimg"].shape != (expected_frames, 3, 3):
        raise ProductExportError(f"invalid K_fullimg shape: {camera['K_fullimg'].shape}")
    _validate_transforms("camera.T_w2c", camera["T_w2c"])
    intrinsics = np.asarray(camera["K_fullimg"], dtype=np.float64)
    if np.any(intrinsics[:, 0, 0] <= 0) or np.any(intrinsics[:, 1, 1] <= 0):
        raise ProductExportError("camera intrinsics have non-positive focal length")
    world_to_isaac = np.asarray(camera["world_to_isaac"], dtype=np.float64)
    if world_to_isaac.shape != (3, 3):
        raise ProductExportError(
            f"world_to_isaac must be (3,3), got {world_to_isaac.shape}"
        )
    if not np.allclose(
        world_to_isaac @ world_to_isaac.T,
        np.eye(3),
        atol=1e-5,
    ):
        raise ProductExportError("world_to_isaac is not orthonormal")
    output = {
        "schema_version": np.int32(SCHEMA_VERSION),
        "fps": np.float32(expected_fps),
        "frame_count": np.int32(expected_frames),
        "units": np.asarray("meter"),
        "world_coordinate_system": np.asarray("gvhmr_world_gravity_negative_y"),
        "render_coordinate_system": np.asarray("mujoco_world_z_up"),
        "camera_convention": np.asarray("opencv_x_right_y_down_z_forward"),
    }
    output.update({key: camera[key] for key in required})
    for key in ("horizontal_fov_deg", "person_idx"):
        if key in camera:
            output[key] = camera[key]
    _save_npz(bundle_dir / "camera.npz", output)
    return {
        "frames": expected_frames,
        "gravity_axis": str(_scalar(camera, "gravity_axis")),
        "camera_convention": "opencv_x_right_y_down_z_forward",
    }


def _object_status(clip_dir: Path) -> tuple[bool, str]:
    recon = clip_dir / "object_reconstruction"
    unavailable = recon / "object_unavailable.json"
    if unavailable.is_file():
        report = _json_read(unavailable)
        stage = str(report.get("stage", "object"))
        status = str(report.get("status", "failed"))
        return (
            False,
            f"{stage} status={status}; inspect "
            "object_reconstruction/object_unavailable.json",
        )
    validation_path = recon / "object_adapter_validation.json"
    if not validation_path.is_file():
        return False, "object_adapter_validation.json is missing"
    validation = _json_read(validation_path)
    if not bool(validation.get("ok", False)) or validation.get("errors"):
        return False, "object adapter validation did not pass"
    for name in ("object_motion_gvhmr.npz", "object_motion_gmr.npz"):
        if not (recon / name).is_file():
            return False, f"{name} is missing"
    return True, "ok"


def _copy_mesh_assets(source_mesh: Path, destination: Path) -> str:
    if not source_mesh.is_file():
        raise ProductExportError(f"object visual mesh is missing: {source_mesh}")
    destination.mkdir(parents=True, exist_ok=True)
    source_dir = source_mesh.parent.resolve()
    selected = {source_mesh.resolve()}
    if source_mesh.suffix.lower() == ".obj":
        for line in source_mesh.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines():
            if line.lower().startswith("mtllib "):
                selected.add(
                    (source_dir / line.split(maxsplit=1)[1].strip()).resolve()
                )
    for material in list(selected):
        if material.suffix.lower() != ".mtl" or not material.is_file():
            continue
        for line in material.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines():
            parts = line.strip().split()
            if (
                parts
                and parts[0].lower().startswith("map_")
                and len(parts) >= 2
            ):
                selected.add((source_dir / parts[-1]).resolve())
    for path in sorted(selected):
        if not _is_relative_to(path, source_dir) or not path.is_file():
            raise ProductExportError(
                f"unsafe or missing referenced mesh asset: {path}"
            )
        if path.suffix.lower() not in MESH_SUFFIXES:
            raise ProductExportError(
                f"unsupported referenced mesh asset: {path}"
            )
        shutil.copy2(path, destination / path.name)
    return f"object/mesh/{source_mesh.name}"


def _export_object(
    clip_dir: Path,
    bundle_dir: Path,
    expected_frames: int,
    expected_fps: float,
    expected_size_m: list[float] | tuple[float, float] | None = None,
    min_valid_ratio: float = 0.0,
    min_observed_ratio: float = 0.0,
) -> dict[str, Any]:
    recon = clip_dir / "object_reconstruction"
    validation = _json_read(recon / "object_adapter_validation.json")
    validation_valid_ratio = float(validation.get("valid_ratio", 0.0))
    validation_observed_ratio = float(validation.get("observed_ratio", 0.0))
    if validation_valid_ratio < min_valid_ratio:
        raise ProductExportError(
            "object valid ratio is below the product threshold: "
            f"{validation_valid_ratio:.3f} < {min_valid_ratio:.3f}"
        )
    if validation_observed_ratio < min_observed_ratio:
        raise ProductExportError(
            "object observed ratio is below the product threshold: "
            f"{validation_observed_ratio:.3f} < {min_observed_ratio:.3f}"
        )
    mesh_extents = validation.get("mesh_extents_m")
    if expected_size_m is not None:
        if (
            not isinstance(expected_size_m, (list, tuple))
            or len(expected_size_m) != 2
        ):
            raise ProductExportError(
                "object clip expected_size_m must be [min, max]"
            )
        if not isinstance(mesh_extents, list) or len(mesh_extents) != 3:
            raise ProductExportError(
                "object validation does not contain three mesh extents"
            )
        maximum_extent = float(max(mesh_extents))
        minimum, maximum = (float(item) for item in expected_size_m)
        if not minimum <= maximum_extent <= maximum:
            raise ProductExportError(
                "object mesh maximum extent is outside expected_size_m: "
                f"{maximum_extent:.3f}m not in [{minimum:.3f}, {maximum:.3f}]"
            )
    gvhmr = _load_npz(recon / "object_motion_gvhmr.npz")
    gmr = _load_npz(recon / "object_motion_gmr.npz")
    _require_keys(
        gvhmr,
        ("pose_camera_object", "pose_gvhmr_world_object", "valid", "K", "fps"),
        "object_motion_gvhmr.npz",
    )
    _require_keys(
        gmr,
        (
            "position",
            "quat_wxyz",
            "valid",
            "source_frame_index",
            "fps",
            "visual_mesh_path",
        ),
        "object_motion_gmr.npz",
    )
    robot_frames = _frame_count(gmr["position"], "object robot position")
    if robot_frames != expected_frames:
        raise ProductExportError(
            f"robot/object frame mismatch: robot={expected_frames}, object={robot_frames}"
        )
    if gmr["position"].shape != (expected_frames, 3):
        raise ProductExportError(f"invalid object position shape: {gmr['position'].shape}")
    if gmr["quat_wxyz"].shape != (expected_frames, 4):
        raise ProductExportError(f"invalid object quaternion shape: {gmr['quat_wxyz'].shape}")
    valid_robot = gmr["valid"].astype(bool)
    if valid_robot.shape != (expected_frames,):
        raise ProductExportError(f"invalid object valid shape: {valid_robot.shape}")
    camera_frames = _frame_count(
        gvhmr["pose_camera_object"],
        "object camera pose",
    )
    camera_valid = np.asarray(gvhmr["valid"], dtype=bool)
    if camera_valid.shape != (camera_frames,):
        raise ProductExportError(
            f"invalid object camera valid shape: {camera_valid.shape}"
        )
    if gvhmr["pose_gvhmr_world_object"].shape != (camera_frames, 4, 4):
        raise ProductExportError(
            "object camera/world pose frame counts are inconsistent"
        )
    _validate_transforms(
        "object.camera_pose_object",
        gvhmr["pose_camera_object"],
        camera_valid,
    )
    _validate_transforms(
        "object.gvhmr_world_pose_object",
        gvhmr["pose_gvhmr_world_object"],
        camera_valid,
    )
    intrinsics = np.asarray(gvhmr["K"])
    if intrinsics.shape not in {(3, 3), (camera_frames, 3, 3)}:
        raise ProductExportError(
            f"object camera intrinsics have invalid shape: {intrinsics.shape}"
        )
    _validate_numeric("object.camera_intrinsics", intrinsics)
    intrinsics_frames = (
        intrinsics[None] if intrinsics.shape == (3, 3) else intrinsics
    )
    if (
        np.any(intrinsics_frames[:, 0, 0] <= 0)
        or np.any(intrinsics_frames[:, 1, 1] <= 0)
    ):
        raise ProductExportError(
            "object camera intrinsics have non-positive focal length"
        )
    camera_fps = float(_scalar(gvhmr, "fps"))
    if camera_fps <= 0:
        raise ProductExportError(f"invalid object camera fps: {camera_fps}")
    source_indices = np.asarray(gmr["source_frame_index"], dtype=np.int64)
    if source_indices.shape != (expected_frames,):
        raise ProductExportError(
            "object source_frame_index has invalid shape: "
            f"{source_indices.shape}"
        )
    valid_indices = source_indices[valid_robot]
    if len(valid_indices):
        if valid_indices.min() < 0 or valid_indices.max() >= camera_frames:
            raise ProductExportError(
                "object source_frame_index is out of camera bounds"
            )
        if np.any(np.diff(valid_indices) < 0):
            raise ProductExportError(
                "object source_frame_index is not monotonic"
            )
    if valid_robot.any():
        norms = np.linalg.norm(gmr["quat_wxyz"][valid_robot], axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ProductExportError("object quat_wxyz contains non-unit quaternions")
    robot_fps = float(_scalar(gmr, "fps"))
    if abs(robot_fps - expected_fps) > 1e-3:
        raise ProductExportError(
            f"robot/object fps mismatch: robot={expected_fps}, object={robot_fps}"
        )
    if str(_scalar(gmr, "coordinate_system", "")) != "mujoco_robot_world_zup":
        raise ProductExportError("object GMR coordinate_system is not z-up robot world")
    if str(_scalar(gvhmr, "coordinate_system", "")) != "opencv_camera":
        raise ProductExportError("object GVHMR coordinate_system is not OpenCV camera")

    source_mesh = Path(str(_scalar(gmr, "visual_mesh_path"))).resolve()
    mesh_relative = _copy_mesh_assets(source_mesh, bundle_dir / "object" / "mesh")
    collision_relative: list[str] = []
    collision_dir = bundle_dir / "object" / "collision"
    collision_paths = gmr.get("collision_mesh_paths", np.asarray([], dtype=str))
    if len(collision_paths):
        collision_dir.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(collision_paths.astype(str)):
        source_path = Path(source).resolve()
        if not source_path.is_file():
            raise ProductExportError(f"collision mesh is missing: {source_path}")
        name = f"collision_{index:03d}{source_path.suffix.lower()}"
        shutil.copy2(source_path, collision_dir / name)
        collision_relative.append(f"object/collision/{name}")

    values: dict[str, Any] = {
        "schema_version": np.int32(SCHEMA_VERSION),
        "units": np.asarray("meter_radian"),
        "visual_mesh_path": np.asarray(mesh_relative),
        "collision_mesh_paths": np.asarray(collision_relative, dtype=str),
        "camera_coordinate_system": np.asarray("opencv_x_right_y_down_z_forward"),
        "camera_fps": np.float32(camera_fps),
        "camera_pose_object": gvhmr["pose_camera_object"].astype(np.float32),
        "gvhmr_world_pose_object": gvhmr["pose_gvhmr_world_object"].astype(np.float32),
        "camera_valid": camera_valid,
        "camera_intrinsics": gvhmr["K"].astype(np.float32),
        "robot_coordinate_system": np.asarray("mujoco_robot_world_z_up"),
        "robot_fps": np.float32(robot_fps),
        "robot_position": gmr["position"].astype(np.float32),
        "robot_quat_wxyz": gmr["quat_wxyz"].astype(np.float32),
        "robot_valid": valid_robot,
        "source_frame_index": source_indices.astype(np.int32),
    }
    for key in ("mesh_scale", "density", "friction", "solref", "rgba"):
        if key in gmr:
            values[key] = gmr[key]
    _save_npz(bundle_dir / "object_motion.npz", values)

    qa_dir = bundle_dir / "quality" / "object"
    qa_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "object_adapter_validation.json",
        "object_manifest.json",
        "hand_object_contact_report.json",
        "object_dynamic_sim.json",
    ):
        source = recon / name
        if source.is_file():
            _json_write(
                qa_dir / name,
                _portable_value(_json_read(source)),
            )
    return {
        "status": "valid",
        "camera_frames": camera_frames,
        "robot_frames": robot_frames,
        "valid_ratio": float(np.mean(valid_robot)),
        "observed_ratio": validation_observed_ratio,
        "mesh_extents_m": mesh_extents,
        "visual_mesh": mesh_relative,
        "collision_mesh_count": len(collision_relative),
    }


def _config_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "schema_version",
        "human",
        "locomotion",
        "phc",
        "gmr",
        "object",
        "product",
    )
    snapshot = {key: copy.deepcopy(config[key]) for key in keep if key in config}
    # Runtime paths, credentials and work/output locations are deliberately not
    # part of a distributable reproducibility snapshot.
    object_cfg = snapshot.get("object")
    if isinstance(object_cfg, dict):
        monocular = object_cfg.get("monocular")
        if isinstance(monocular, dict):
            for key in (
                "work_root",
                "reconstruction_script",
                "samhq_checkpoint",
                "cutie_checkpoint",
            ):
                monocular.pop(key, None)
    product_cfg = snapshot.get("product")
    if isinstance(product_cfg, dict):
        product_cfg.pop("root", None)
        product_cfg.pop("retention", None)
    return _portable_value(snapshot)


def _copy_preview(clip_dir: Path, bundle_dir: Path) -> str | None:
    candidates = (
        clip_dir / "composite_2x2.mp4",
        clip_dir / "unitree_g1_sharpa_gvhmr.mp4",
        clip_dir / "unitree_h1_with_hand_sharpa_gvhmr.mp4",
    )
    for source in candidates:
        if source.is_file():
            destination = bundle_dir / "preview_2x2.mp4"
            shutil.copy2(source, destination)
            return destination.name
    return None


def _payload_files(bundle_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in bundle_dir.rglob("*")
        if path.is_file()
        and path.name not in {"manifest.json", "checksums.sha256"}
    )


def _compact_motion_npz(bundle_dir: Path) -> list[str]:
    """Merge final motion components into one portable NPZ asset.

    The execution workspace can contain stage caches while a clip is running,
    but an exported dataset item has one numerical motion payload. Prefixing
    every field by its component makes the contract explicit and avoids
    duplicate generic names such as ``fps`` or ``translation``.
    """
    components = (
        ("human", "human_motion.npz"),
        ("human_phc", "human_phc_motion.npz"),
        ("robot", "robot_motion.npz"),
        ("sharpa", "robot_hand_motion.npz"),
        ("camera", "camera.npz"),
        ("object", "object_motion.npz"),
    )
    packed: dict[str, Any] = {
        "schema_version": np.int32(SCHEMA_VERSION),
        "format": np.asarray("locomanipulation_motion_v1"),
    }
    present: list[str] = []
    consumed: list[Path] = []
    for namespace, filename in components:
        path = bundle_dir / filename
        if not path.is_file():
            continue
        values = _load_npz(path)
        for key, value in values.items():
            packed[f"{namespace}__{key}"] = value
        present.append(namespace)
        consumed.append(path)
    if not present:
        raise ProductExportError("no motion NPZ components were produced")
    packed["components"] = np.asarray(present)
    destination = bundle_dir / "motion.npz"
    _save_npz(destination, packed)
    for path in consumed:
        path.unlink()
    return present


def verify_bundle(bundle_dir: Path) -> dict[str, Any]:
    manifest_path = bundle_dir / "manifest.json"
    checksums_path = bundle_dir / "checksums.sha256"
    if not manifest_path.is_file() or not checksums_path.is_file():
        raise ProductExportError(f"incomplete asset bundle: {bundle_dir}")
    manifest = _json_read(manifest_path)
    if manifest.get("validation", {}).get("status") != "passed":
        raise ProductExportError(f"asset manifest is not marked passed: {bundle_dir}")
    checked_paths: set[str] = set()
    for raw_line in checksums_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        digest, relative = raw_line.split("  ", 1)
        if relative in checked_paths:
            raise ProductExportError(
                f"duplicate checksum entry: {relative}"
            )
        path = bundle_dir / relative
        if not _is_relative_to(path.resolve(), bundle_dir.resolve()):
            raise ProductExportError(
                f"unsafe checksum path outside bundle: {relative}"
            )
        if not path.is_file() or _sha256(path) != digest:
            raise ProductExportError(f"checksum mismatch: {path}")
        checked_paths.add(relative)
    if not checked_paths:
        raise ProductExportError(f"empty checksum file: {checksums_path}")
    actual_paths = {
        str(path.relative_to(bundle_dir))
        for path in bundle_dir.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    }
    if checked_paths != actual_paths:
        missing = sorted(actual_paths - checked_paths)
        stale = sorted(checked_paths - actual_paths)
        raise ProductExportError(
            "checksum file does not cover the bundle exactly: "
            f"missing={missing}, stale={stale}"
        )
    # Ensure every numerical asset still has no pickle/object arrays.
    for path in bundle_dir.rglob("*.npz"):
        _load_npz(path)
    return {"checked_files": len(checked_paths), "status": "passed"}


def _resolve_export_roots(
    project_root: Path,
    config: dict[str, Any],
) -> tuple[Path, Path, Path]:
    project_root = project_root.resolve()
    output_value = Path(config["output"]["root"])
    output_root = (
        output_value.resolve()
        if output_value.is_absolute()
        else (project_root / output_value).resolve()
    )
    product_value = Path(
        config.get("product", {}).get("root", "assets/pipeline_products")
    )
    product_root = (
        product_value.resolve()
        if product_value.is_absolute()
        else (project_root / product_value).resolve()
    )
    return project_root, output_root, product_root


def _resolve_export_clip_paths(
    output_root: Path,
    product_root: Path,
    clip: str,
) -> tuple[Path, Path]:
    clip_dir = (output_root / clip).resolve()
    if clip_dir.parent != output_root or not clip_dir.is_dir():
        raise ProductExportError(
            f"clip={clip!r}: work directory is missing or unsafe: {clip_dir}"
        )
    final_dir = (product_root / clip).resolve()
    if final_dir.parent != product_root:
        raise ProductExportError(f"unsafe product clip name: {clip!r}")
    if _is_relative_to(final_dir, clip_dir):
        raise ProductExportError(
            "product.root must not be inside the clip work directory"
        )
    return clip_dir, final_dir


def _load_automated_quality_for_export(
    clip_dir: Path,
    config: dict[str, Any],
    clip: str,
) -> dict[str, Any] | None:
    product_cfg = config.get("product", {})
    quality_config = config.get("quality_evaluation", {})
    minimum = str(product_cfg.get("minimum_quality_status", "warn"))
    if minimum not in {"warn", "pass"}:
        raise ProductExportError(
            "product.minimum_quality_status must be warn or pass"
        )
    try:
        minimum_schema = int(
            product_cfg.get("minimum_quality_schema_version", 1)
        )
    except (TypeError, ValueError) as exc:
        raise ProductExportError(
            "product.minimum_quality_schema_version must be an integer"
        ) from exc
    if minimum_schema < 1:
        raise ProductExportError(
            "product.minimum_quality_schema_version must be at least 1"
        )
    quality_path = clip_dir / "quality_report.json"
    if not quality_path.is_file():
        if bool(quality_config.get("require_for_product", False)):
            raise ProductExportError(
                f"clip={clip!r}: quality_report.json is required for product export"
            )
        return None

    automated_quality = _json_read(quality_path)
    try:
        report_schema = int(automated_quality.get("schema_version", 0))
    except (TypeError, ValueError):
        report_schema = 0
    if report_schema < minimum_schema:
        raise ProductExportError(
            f"clip={clip!r}: quality report schema_version={report_schema} is "
            f"below product.minimum_quality_schema_version={minimum_schema}; "
            "rerun the quality stage before product export"
        )
    report_clip = automated_quality.get("clip")
    if report_clip is not None and str(report_clip) != clip:
        raise ProductExportError(
            f"clip={clip!r}: quality report belongs to {report_clip!r}"
        )
    quality_status = str(automated_quality.get("status", "fail"))
    ranks = {"fail": 0, "warn": 1, "pass": 2}
    if quality_status not in ranks:
        raise ProductExportError(
            f"clip={clip!r}: invalid quality status {quality_status!r}"
        )
    if ranks[quality_status] < ranks[minimum]:
        raise ProductExportError(
            f"clip={clip!r}: quality status {quality_status!r} is below "
            f"product.minimum_quality_status={minimum!r}"
        )
    return automated_quality


def _preflight_export_quality(
    project_root: Path,
    config: dict[str, Any],
    clips: Iterable[str],
) -> tuple[list[str], list[dict[str, str]]]:
    _, output_root, product_root = _resolve_export_roots(project_root, config)
    eligible = []
    skipped = []
    for clip in clips:
        try:
            clip_dir, _ = _resolve_export_clip_paths(
                output_root, product_root, clip
            )
            _load_automated_quality_for_export(clip_dir, config, clip)
            eligible.append(clip)
        except Exception as exc:
            skipped.append(
                {"clip": str(clip), "reason": str(exc)}
            )
    return eligible, skipped


def export_clip(
    project_root: Path,
    config: dict[str, Any],
    clip: str,
) -> dict[str, Any]:
    project_root, output_root, product_root = _resolve_export_roots(
        project_root, config
    )
    product_cfg = config.get("product", {})
    clip_dir, final_dir = _resolve_export_clip_paths(
        output_root, product_root, clip
    )
    automated_quality = _load_automated_quality_for_export(
        clip_dir, config, clip
    )
    product_root.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(
        tempfile.mkdtemp(prefix=f".{clip}.export-", dir=product_root)
    ).resolve()
    try:
        if automated_quality is not None:
            _json_write(
                stage_dir / "quality_report.json",
                _portable_value(automated_quality),
            )
        final_selection = None
        selection_path = clip_dir / "final_motion_selection.json"
        if selection_path.is_file():
            final_selection = _json_read(selection_path)
            _json_write(
                stage_dir / "final_motion_selection.json",
                _portable_value(final_selection),
            )

        human_quality, frames, fps = _export_human(
            clip_dir,
            stage_dir,
            str(config.get("gmr", {}).get("source", "smoothed")),
        )
        phc_quality = None
        if (
            bool(product_cfg.get("include_phc_motion", True))
            and bool(config.get("phc", {}).get("enabled", False))
        ):
            # When the canonical human component already is the PHC output,
            # writing it again under a second filename wastes space at scale.
            # Keep the provenance record but export a second body component
            # only when the selected human and PHC stages are genuinely
            # different.
            if str(human_quality["source_body_stage"]).startswith("phc"):
                phc_quality = {
                    "frames": frames,
                    "fps": fps,
                    "source": str(human_quality["source_body_path"]),
                    "stored_in_canonical_human_component": True,
                }
            else:
                phc_quality = _export_phc_motion(
                    clip_dir,
                    stage_dir,
                    frames,
                    fps,
                )
        robot_quality = _export_robot(
            project_root,
            clip_dir,
            stage_dir,
            str(config.get("gmr", {}).get("hand_model", "sharpa")),
            frames,
            fps,
            clip_dir / str(human_quality["source_body_path"]),
            str(human_quality["source_body_stage"]),
        )
        source_provenance = robot_quality["source_provenance"]
        provenance_comparison = source_provenance["comparison"]
        if provenance_comparison["status"] != "match":
            print(
                "[PRODUCT][PROVENANCE] "
                f"{clip}: {provenance_comparison['status']} "
                f"({provenance_comparison['reason']})",
                flush=True,
            )
        camera_quality = None
        if bool(product_cfg.get("include_camera", True)):
            camera_quality = _export_camera(
                clip_dir,
                stage_dir,
                frames,
                fps,
            )

        object_policy = str(product_cfg.get("object_policy", "if_valid"))
        if object_policy not in {"exclude", "if_valid", "require_valid"}:
            raise ProductExportError(
                "product.object_policy must be exclude, if_valid or require_valid"
            )
        object_enabled = bool(config.get("object", {}).get("enabled", False))
        if object_policy == "exclude":
            object_ok, object_reason = False, "excluded by product.object_policy"
        else:
            object_ok, object_reason = _object_status(clip_dir)
        object_quality: dict[str, Any] = {
            "status": "excluded" if object_policy == "exclude" else "unavailable",
            "reason": object_reason,
        }
        if object_policy != "exclude" and object_ok:
            object_config = config.get("object", {})
            clip_object_config = object_config.get("clips", {}).get(clip, {})
            quality_config = object_config.get("quality", {})
            object_quality = _export_object(
                clip_dir,
                stage_dir,
                frames,
                fps,
                expected_size_m=clip_object_config.get("expected_size_m"),
                min_valid_ratio=float(
                    quality_config.get("min_valid_ratio", 0.0)
                ),
                min_observed_ratio=float(
                    quality_config.get("min_observed_ratio", 0.0)
                ),
            )
        elif object_policy == "require_valid" and object_enabled:
            raise ProductExportError(
                f"valid object asset is required for {clip}: {object_reason}"
            )

        preview = None
        if bool(product_cfg.get("include_preview", True)):
            preview = _copy_preview(clip_dir, stage_dir)
            if preview is None and bool(product_cfg.get("require_preview", False)):
                raise ProductExportError(f"2x2 preview is required but missing for {clip}")

        snapshot = _config_snapshot(config)
        (stage_dir / "pipeline_config.yaml").write_text(
            yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

        has_object = (stage_dir / "object_motion.npz").is_file()
        motion_components = _compact_motion_npz(stage_dir)
        payload = _payload_files(stage_dir)
        file_records = {
            str(path.relative_to(stage_dir)): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in payload
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "clip": clip,
            "product_grade": "object_visual" if has_object else "human_motion",
            "frame_count": frames,
            "fps": fps,
            "files": file_records,
            "data_contract": {
                "motion": {
                    "file": "motion.npz",
                    "components": motion_components,
                    "field_namespace_separator": "__",
                    "final_selection": final_selection,
                },
                "object": (
                    {
                        "motion_namespace": "object",
                        "camera_pose": "T_camera_object; OpenCV x-right y-down z-forward",
                        "robot_quaternion": "wxyz; MuJoCo z-up world",
                    }
                    if has_object
                    else None
                ),
            },
            "quality": {
                "automated": (
                    {
                        "status": automated_quality.get("status"),
                        "score": automated_quality.get(
                            "macro_quality", {}
                        ).get("score", automated_quality.get("overall_score")),
                        "grade": automated_quality.get(
                            "macro_quality", {}
                        ).get("grade"),
                        "verdict": automated_quality.get(
                            "macro_quality", {}
                        ).get("verdict"),
                        "report": "quality_report.json",
                    }
                    if automated_quality is not None
                    else None
                ),
                "human": human_quality,
                "phc": phc_quality,
                "robot": robot_quality,
                "camera": camera_quality,
                "object": object_quality,
            },
            "provenance": source_provenance,
            "preview": preview,
            "validation": {"status": "passed"},
        }
        _json_write(stage_dir / "manifest.json", manifest)
        checksum_paths = sorted(
            path
            for path in stage_dir.rglob("*")
            if path.is_file() and path.name != "checksums.sha256"
        )
        (stage_dir / "checksums.sha256").write_text(
            "".join(
                f"{_sha256(path)}  {path.relative_to(stage_dir)}\n"
                for path in checksum_paths
            ),
            encoding="utf-8",
        )
        verify_bundle(stage_dir)
        if final_dir.exists():
            # Do not remove an arbitrary user directory just because it shares
            # the configured clip name. Only replace a valid generated bundle.
            verify_bundle(final_dir)
            shutil.rmtree(final_dir)
        os.replace(stage_dir, final_dir)
        result = {
            "clip": clip,
            "bundle": str(final_dir),
            "product_grade": manifest["product_grade"],
            "bytes": sum(path.stat().st_size for path in final_dir.rglob("*") if path.is_file()),
        }
        return result
    except Exception as exc:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise ProductExportError(
            f"clip={clip!r}; output_root={output_root}; "
            f"product_root={product_root}: {exc}"
        ) from exc


def _safe_remove_direct_child(path: Path, parent: Path, label: str) -> None:
    path = path.resolve()
    parent = parent.resolve()
    if path.parent != parent or not path.is_dir():
        raise ProductExportError(f"refusing to remove unsafe {label}: {path}")
    shutil.rmtree(path)


def prune_exported_workspace(
    project_root: Path,
    config: dict[str, Any],
    clip: str,
    bundle_dir: Path,
) -> dict[str, Any]:
    verify_bundle(bundle_dir)
    project_root = project_root.resolve()
    output_value = Path(config["output"]["root"])
    output_root = (
        output_value.resolve()
        if output_value.is_absolute()
        else (project_root / output_value).resolve()
    )
    clip_dir = (output_root / clip).resolve()
    if _is_relative_to(bundle_dir.resolve(), clip_dir):
        raise ProductExportError("refusing to prune a workspace containing its product")
    removed: list[str] = []
    if clip_dir.is_dir():
        _safe_remove_direct_child(clip_dir, output_root, "clip workspace")
        removed.append(str(clip_dir))

    retention = config.get("product", {}).get("retention", {})
    if bool(retention.get("prune_object_work_after_export", False)):
        work_value = Path(
            config.get("object", {})
            .get("monocular", {})
            .get("work_root", "object_work/do_as_i_do")
        )
        work_root = (
            work_value.resolve()
            if work_value.is_absolute()
            else (project_root / work_value).resolve()
        )
        object_work = (work_root / clip).resolve()
        if object_work.is_dir():
            _safe_remove_direct_child(object_work, work_root, "object workspace")
            removed.append(str(object_work))

    marker_dir = output_root / "_exported"
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / f"{clip}.json"
    _json_write(
        marker,
        {
            "schema_version": SCHEMA_VERSION,
            "clip": clip,
            "bundle": str(bundle_dir.resolve()),
            "removed": removed,
            "checksums_verified": True,
        },
    )
    return {"clip": clip, "removed": removed, "marker": str(marker)}


def _write_product_catalog(
    project_root: Path,
    config: dict[str, Any],
) -> Path:
    product_value = Path(
        config.get("product", {}).get("root", "assets/pipeline_products")
    )
    product_root = (
        product_value.resolve()
        if product_value.is_absolute()
        else (project_root.resolve() / product_value).resolve()
    )
    product_root.mkdir(parents=True, exist_ok=True)
    records = []
    for manifest_path in sorted(product_root.glob("*/manifest.json")):
        manifest = _json_read(manifest_path)
        bundle_dir = manifest_path.parent
        automated_quality = (
            manifest.get("quality", {}).get("automated") or {}
        )
        records.append(
            {
                "clip": str(manifest.get("clip", bundle_dir.name)),
                "product_grade": str(
                    manifest.get("product_grade", "unknown")
                ),
                "frame_count": manifest.get("frame_count"),
                "fps": manifest.get("fps"),
                "has_object": manifest.get("data_contract", {}).get(
                    "object"
                )
                is not None,
                "quality_status": automated_quality.get("status"),
                "pipeline_score": automated_quality.get("score"),
                "pipeline_grade": automated_quality.get("grade"),
                "pipeline_verdict": automated_quality.get("verdict"),
                "bytes": sum(
                    path.stat().st_size
                    for path in bundle_dir.rglob("*")
                    if path.is_file()
                ),
                "bundle": bundle_dir.name,
            }
        )
    destination = product_root / "catalog.jsonl"
    handle, temporary_name = tempfile.mkstemp(
        prefix=".catalog-",
        suffix=".jsonl",
        dir=product_root,
        text=True,
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            for record in records:
                stream.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def export_from_config(
    project_root: Path,
    config: dict[str, Any],
    clips: Iterable[str],
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    product_cfg = config.get("product", {})
    if not bool(product_cfg.get("enabled", False)):
        raise ProductExportError("product.enabled is false")
    clip_list = list(clips)
    if not clip_list:
        raise ProductExportError("no clips were requested for product export")
    if len(set(clip_list)) != len(clip_list):
        raise ProductExportError("duplicate clips were requested for product export")
    results = []
    completed = []
    skipped = []
    retention = product_cfg.get("retention", {})
    prune = bool(retention.get("prune_workspace_after_export", False))
    export_clips = clip_list
    if not dry_run:
        export_clips, skipped = _preflight_export_quality(
            project_root, config, clip_list
        )
        for item in skipped:
            print(
                f"[PRODUCT][SKIP] {item['clip']}: {item['reason']}",
                flush=True,
            )
    for clip in export_clips:
        if dry_run:
            print(f"[DRY-RUN] export product: {clip}", flush=True)
            if prune:
                print(
                    f"[DRY-RUN] prune verified workspace: {clip}",
                    flush=True,
                )
            continue
        try:
            result = export_clip(project_root, config, clip)
            print(
                f"[PRODUCT] {clip}: {result['product_grade']} -> {result['bundle']}",
                flush=True,
            )
            results.append(result)
            completed.append(clip)
        except Exception as exc:
            skipped.append(
                {"clip": str(clip), "reason": f"export failed: {exc}"}
            )
            print(
                f"[PRODUCT][SKIP] {clip}: export failed: {exc}",
                flush=True,
            )

    if dry_run:
        return results

    if results:
        try:
            catalog = _write_product_catalog(project_root, config)
        except Exception as exc:
            _, output_root, product_root = _resolve_export_roots(project_root, config)
            raise ProductExportError(
                "exported product bundles could not publish the catalog; "
                f"output_root={output_root}; product_root={product_root}; "
                f"completed_clips={completed!r}: {exc}"
            ) from exc
        print(f"[PRODUCT] catalog -> {catalog}", flush=True)
    else:
        print("[PRODUCT] no eligible clips were exported.", flush=True)
    if skipped:
        print(
            f"[PRODUCT] export summary: exported={len(results)} skipped={len(skipped)}",
            flush=True,
        )

    if prune:
        for result in results:
            clip = str(result["clip"])
            try:
                prune_result = prune_exported_workspace(
                    project_root,
                    config,
                    clip,
                    Path(result["bundle"]),
                )
                print(
                    f"[PRODUCT] pruned {len(prune_result['removed'])} "
                    "work directorie(s) after checksum verification",
                    flush=True,
                )
            except Exception as exc:
                raise ProductExportError(
                    "product bundles and catalog were published, but workspace "
                    f"pruning failed; phase=prune; clip={clip!r}: {exc}"
                ) from exc
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export product bundles through the same layered YAML configuration."
    )
    parser.add_argument(
        "--config_dir",
        "--config-dir",
        "--config",
        dest="config_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--default_config",
        "--default-config",
        dest="default_config",
        type=Path,
        default=Path("configs/default.yaml"),
    )
    parser.add_argument("--clip-filter", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from run_pipeline_from_config import PipelineRunner

    overrides = []
    if args.clip_filter is not None:
        overrides.append(("input.clip_filter", args.clip_filter))
    runner = PipelineRunner(
        default_config_path=args.default_config,
        config_path=args.config_dir,
        cli_overrides=overrides,
        dry_run=args.dry_run,
    )
    runner.check("product")
    videos = runner._completed_product_videos(report_missing=True)
    if not videos:
        print("[PRODUCT] no completed human/GMR clips; skipping export.")
        return
    clips = [path.stem for path in videos]
    export_from_config(runner.root, runner.config, clips, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
