#!/usr/bin/env python3
"""Evaluate pipeline clips without ground truth and assign pass/warn/fail tiers.

The report is intentionally operational rather than benchmark-oriented.  It
detects broken files, inconsistent frame counts, unstable body/hand/object
tracks, unplayable robot output, weak hand-object proximity, and failed
dynamic validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = 1
BODY_SOURCE_FILES = {
    "converted": "001_converted.npz",
    "smoothed": "001_smoothed.npz",
    "phc_smoothed": "001_phc_smoothed.npz",
}
ROBOT_XML = {
    "sharpa": "GMR-master/assets/unitree_h1/h1_with_hand.xml",
    "g1": "GMR-master/assets/unitree_g1/g1_mocap_29dof_with_hands.xml",
    "brainco": "GMR-master/assets/unitree_g1/g1_mocap_29dof.xml",
}


class QualityEvaluationError(RuntimeError):
    """Raised when a quality report cannot be produced."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualityEvaluationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise QualityEvaluationError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: np.asarray(archive[key]) for key in archive.files}
    except (OSError, ValueError) as exc:
        raise QualityEvaluationError(f"cannot safely load NPZ {path}: {exc}") from exc


def _load_robot(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            value = pickle.load(stream)
    except Exception as exc:
        raise QualityEvaluationError(
            f"cannot load trusted robot motion {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise QualityEvaluationError(f"robot motion is not a dictionary: {path}")
    return value


def _scalar(data: dict[str, np.ndarray], key: str, default: Any = None) -> Any:
    if key not in data:
        return default
    value = np.asarray(data[key])
    if value.size != 1:
        return default
    return value.reshape(-1)[0].item()


def _finite_percentile(value: np.ndarray, percentile: float) -> float | None:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return None
    return float(np.percentile(finite, percentile))


def _ratio(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=bool).reshape(-1)
    return float(np.mean(array)) if len(array) else 0.0


def _lower_is_better_score(value: float | None, good: float, bad: float) -> float:
    if value is None or not math.isfinite(value):
        return 0.0
    if value <= good:
        return 100.0
    if value >= bad:
        return 0.0
    return 100.0 * (bad - value) / (bad - good)


def _higher_is_better_score(value: float | None, bad: float, good: float) -> float:
    if value is None or not math.isfinite(value):
        return 0.0
    if value >= good:
        return 100.0
    if value <= bad:
        return 0.0
    return 100.0 * (value - bad) / (good - bad)


def _metric(
    value: Any,
    status: str,
    *,
    unit: str | None = None,
    source: str | None = None,
    details: Any = None,
) -> dict[str, Any]:
    result = {"value": value, "status": status}
    if unit:
        result["unit"] = unit
    if source:
        result["source"] = source
    if details is not None:
        result["details"] = details
    return result


def _resolve_body_source(clip_dir: Path, source: str) -> tuple[Path, str]:
    candidates = (
        ("smoothed", "converted", "phc_smoothed")
        if source == "auto"
        else (source,)
    )
    for name in candidates:
        filename = BODY_SOURCE_FILES.get(name)
        if filename and (clip_dir / filename).is_file():
            return clip_dir / filename, name
    expected = [BODY_SOURCE_FILES.get(name, name) for name in candidates]
    raise QualityEvaluationError(f"body source missing; expected one of {expected}")


def _video_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"readable": False, "frames": 0, "fps": 0.0}
    try:
        import cv2

        capture = cv2.VideoCapture(str(path))
        readable = bool(capture.isOpened())
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if readable else 0
        fps = float(capture.get(cv2.CAP_PROP_FPS)) if readable else 0.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if readable else 0
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if readable else 0
        capture.release()
        return {
            "readable": readable and frames > 0 and fps > 0 and width > 0 and height > 0,
            "frames": frames,
            "fps": fps,
            "width": width,
            "height": height,
        }
    except Exception as exc:
        return {"readable": False, "frames": 0, "fps": 0.0, "error": str(exc)}


def _quat_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = quat / np.clip(norm, 1e-12, None)
    x, y, z, w = np.moveaxis(quat, -1, 0)
    return np.stack(
        [
            np.stack(
                [
                    1 - 2 * (y * y + z * z),
                    2 * (x * y - z * w),
                    2 * (x * z + y * w),
                ],
                axis=-1,
            ),
            np.stack(
                [
                    2 * (x * y + z * w),
                    1 - 2 * (x * x + z * z),
                    2 * (y * z - x * w),
                ],
                axis=-1,
            ),
            np.stack(
                [
                    2 * (x * z - y * w),
                    2 * (y * z + x * w),
                    1 - 2 * (x * x + y * y),
                ],
                axis=-1,
            ),
        ],
        axis=-2,
    )


def _quat_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion)
    return _quat_xyzw_to_matrix(quat[..., [1, 2, 3, 0]])


def _robot_joint_ranges(
    project_root: Path,
    hand_model: str,
    expected_dofs: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    relative = ROBOT_XML.get(hand_model)
    if relative is None:
        return [], np.asarray([]), np.asarray([])
    xml_path = project_root / relative
    if not xml_path.is_file():
        return [], np.asarray([]), np.asarray([])
    root = ET.parse(xml_path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        return [], np.asarray([]), np.asarray([])
    compiler = root.find("compiler")
    degrees = (
        compiler is not None
        and compiler.attrib.get("angle", "degree").lower() == "degree"
    )
    names: list[str] = []
    lower: list[float] = []
    upper: list[float] = []
    for joint in worldbody.iter("joint"):
        name = joint.attrib.get("name")
        range_text = joint.attrib.get("range")
        if not name or not range_text:
            continue
        limits = np.fromstring(range_text, dtype=np.float64, sep=" ")
        if len(limits) != 2:
            continue
        names.append(name)
        lower.append(float(limits[0]))
        upper.append(float(limits[1]))
    if len(names) != expected_dofs:
        return [], np.asarray([]), np.asarray([])
    low = np.asarray(lower)
    high = np.asarray(upper)
    if degrees:
        low = np.deg2rad(low)
        high = np.deg2rad(high)
    return names, low, high


def _joint_limit_metrics(
    project_root: Path,
    hand_model: str,
    dof_position: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    dof = np.asarray(dof_position, dtype=np.float64)
    names, low, high = _robot_joint_ranges(
        project_root,
        hand_model,
        dof.shape[1] if dof.ndim == 2 else 0,
    )
    if not names:
        return {
            "ratio": None,
            "count": 0,
            "checked_entries": 0,
            "max_excess_rad": None,
            "joint_names_available": False,
        }
    below = low[None, :] - dof
    above = dof - high[None, :]
    excess = np.maximum(np.maximum(below, above), 0.0)
    violation = excess > tolerance
    return {
        "ratio": float(np.mean(violation)),
        "count": int(np.count_nonzero(violation)),
        "checked_entries": int(violation.size),
        "max_excess_rad": float(excess.max(initial=0.0)),
        "joint_names_available": True,
    }


def _load_obj_vertices(path: Path, max_vertices: int = 50_000) -> np.ndarray:
    vertices = []
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise QualityEvaluationError(f"mesh has no vertices: {path}")
    array = np.asarray(vertices, dtype=np.float64)
    if len(array) > max_vertices:
        index = np.linspace(0, len(array) - 1, max_vertices, dtype=np.int64)
        array = array[index]
    return array


def _mask_bbox(mask_path: Path) -> np.ndarray | None:
    try:
        import cv2

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None
    if mask is None:
        return None
    ys, xs = np.where(mask > 127)
    if len(xs) < 3:
        return None
    return np.asarray(
        [xs.min(), ys.min(), xs.max(), ys.max()],
        dtype=np.float64,
    )


def _bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, first[2] - first[0]) * max(
        0.0,
        first[3] - first[1],
    )
    area_second = max(0.0, second[2] - second[0]) * max(
        0.0,
        second[3] - second[1],
    )
    union = area_first + area_second - intersection
    return float(intersection / union) if union > 0 else 0.0


def _discover_mask(masks_dir: Path, frame_index: int, object_name: str) -> Path | None:
    frame_dir = masks_dir / f"frame_{frame_index:06d}_masks"
    candidates = [
        frame_dir / f"{object_name}.png",
        masks_dir / f"{frame_index:06d}.png",
        masks_dir / f"{frame_index:04d}.png",
    ]
    candidates.extend(sorted(frame_dir.glob("*.png")) if frame_dir.is_dir() else [])
    return next((path for path in candidates if path.is_file()), None)


def _compute_projection_iou(
    adapter_dir: Path,
    masks_dir: Path,
    object_name: str,
    max_samples: int,
) -> dict[str, Any]:
    pose_path = adapter_dir / "pose.npy"
    valid_path = adapter_dir / "pose_valid.npy"
    camera_path = adapter_dir / "cam_K.json"
    mesh_path = adapter_dir / "mesh" / "mesh.obj"
    if not all(path.is_file() for path in (pose_path, camera_path, mesh_path)):
        return {"mean": None, "samples": 0, "source": "missing_adapter_inputs"}
    poses = np.asarray(np.load(pose_path), dtype=np.float64)
    valid = (
        np.asarray(np.load(valid_path), dtype=bool).reshape(-1)
        if valid_path.is_file()
        else np.ones(len(poses), dtype=bool)
    )
    camera = _read_json(camera_path)
    intrinsics = np.asarray(camera.get("K"), dtype=np.float64)
    if intrinsics.shape != (3, 3):
        return {"mean": None, "samples": 0, "source": "invalid_intrinsics"}
    vertices = _load_obj_vertices(mesh_path)
    valid_indices = np.where(valid)[0]
    if not len(valid_indices):
        return {"mean": None, "samples": 0, "source": "no_valid_pose"}
    sample_count = min(max(1, int(max_samples)), len(valid_indices))
    sampled = valid_indices[
        np.linspace(0, len(valid_indices) - 1, sample_count, dtype=np.int64)
    ]
    homogeneous = np.concatenate(
        [vertices, np.ones((len(vertices), 1), dtype=np.float64)],
        axis=1,
    )
    values = []
    for frame_index in sampled:
        mask_path = _discover_mask(masks_dir, int(frame_index), object_name)
        if mask_path is None:
            continue
        mask_box = _mask_bbox(mask_path)
        if mask_box is None:
            continue
        camera_vertices = homogeneous @ poses[frame_index].T
        in_front = camera_vertices[:, 2] > 1e-3
        if np.count_nonzero(in_front) < 3:
            continue
        projected = camera_vertices[in_front, :3] @ intrinsics.T
        uv = projected[:, :2] / np.clip(projected[:, 2:3], 1e-12, None)
        projection_box = np.asarray(
            [
                uv[:, 0].min(),
                uv[:, 1].min(),
                uv[:, 0].max(),
                uv[:, 1].max(),
            ]
        )
        values.append(_bbox_iou(projection_box, mask_box))
    return {
        "mean": float(np.mean(values)) if values else None,
        "samples": len(values),
        "source": "mesh_projection_vs_segmentation_bbox",
    }


def _object_pose_jump_metrics(
    position: np.ndarray,
    quat_wxyz: np.ndarray,
    valid: np.ndarray,
    position_threshold_m: float,
    rotation_threshold_deg: float,
) -> dict[str, Any]:
    position = np.asarray(position, dtype=np.float64)
    quaternion = np.asarray(quat_wxyz, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    consecutive = valid[1:] & valid[:-1]
    if not np.any(consecutive):
        return {
            "count": 0,
            "ratio": 0.0,
            "position_jump_count": 0,
            "rotation_jump_count": 0,
            "checked_steps": 0,
        }
    position_step = np.linalg.norm(np.diff(position, axis=0), axis=1)
    normalized = quaternion / np.clip(
        np.linalg.norm(quaternion, axis=1, keepdims=True),
        1e-12,
        None,
    )
    dot = np.abs(np.sum(normalized[1:] * normalized[:-1], axis=1))
    rotation_step = np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))
    position_jump = (position_step > position_threshold_m) & consecutive
    rotation_jump = (rotation_step > rotation_threshold_deg) & consecutive
    combined = position_jump | rotation_jump
    checked = int(np.count_nonzero(consecutive))
    return {
        "count": int(np.count_nonzero(combined)),
        "ratio": float(np.count_nonzero(combined) / checked),
        "position_jump_count": int(np.count_nonzero(position_jump)),
        "rotation_jump_count": int(np.count_nonzero(rotation_jump)),
        "checked_steps": checked,
        "position_step_p95_m": float(
            np.percentile(position_step[consecutive], 95)
        ),
        "rotation_step_p95_deg": float(
            np.percentile(rotation_step[consecutive], 95)
        ),
    }


def _contact_proxy_metrics(
    robot: dict[str, Any],
    object_motion: dict[str, np.ndarray],
    mesh_path: Path,
    threshold_m: float,
) -> dict[str, Any]:
    required = ("root_pos", "root_rot", "local_body_pos", "link_body_list")
    if any(key not in robot for key in required) or not mesh_path.is_file():
        return {
            "ratio": None,
            "frames": 0,
            "minimum_distance_m": None,
            "method": "unavailable",
        }
    root_position = np.asarray(robot["root_pos"], dtype=np.float64)
    root_rotation = np.asarray(robot["root_rot"], dtype=np.float64)
    local_body = np.asarray(robot["local_body_pos"], dtype=np.float64)
    body_names = [str(name) for name in robot["link_body_list"]]
    hand_indices = [
        body_names.index(name)
        for name in ("left_hand_link", "right_hand_link")
        if name in body_names
    ]
    if not hand_indices:
        return {
            "ratio": None,
            "frames": 0,
            "minimum_distance_m": None,
            "method": "missing_hand_links",
        }
    object_position = np.asarray(object_motion["position"], dtype=np.float64)
    object_quaternion = np.asarray(
        object_motion["quat_wxyz"],
        dtype=np.float64,
    )
    object_valid = np.asarray(object_motion["valid"], dtype=bool).reshape(-1)
    frame_count = min(
        len(root_position),
        len(local_body),
        len(object_position),
        len(object_quaternion),
        len(object_valid),
    )
    root_matrix = _quat_xyzw_to_matrix(root_rotation[:frame_count])
    hand_local = local_body[:frame_count, hand_indices, :]
    hand_world = (
        root_position[:frame_count, None, :]
        + np.einsum("tij,thj->thi", root_matrix, hand_local)
    )
    object_matrix = _quat_wxyz_to_matrix(object_quaternion[:frame_count])
    relative_world = hand_world - object_position[:frame_count, None, :]
    hand_object_local = np.einsum(
        "thi,tij->thj",
        relative_world,
        object_matrix,
    )
    vertices = _load_obj_vertices(mesh_path, max_vertices=200_000)
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    below = np.maximum(lower[None, None, :] - hand_object_local, 0.0)
    above = np.maximum(hand_object_local - upper[None, None, :], 0.0)
    distance = np.linalg.norm(below + above, axis=2)
    nearest = np.min(distance, axis=1)
    valid_distance = nearest[object_valid[:frame_count]]
    if not len(valid_distance):
        return {
            "ratio": 0.0,
            "frames": 0,
            "minimum_distance_m": None,
            "method": "wrist_to_object_obb_proxy",
        }
    contact = valid_distance <= threshold_m
    return {
        "ratio": float(np.mean(contact)),
        "frames": int(np.count_nonzero(contact)),
        "valid_frames": int(len(valid_distance)),
        "minimum_distance_m": float(np.min(valid_distance)),
        "distance_p10_m": float(np.percentile(valid_distance, 10)),
        "threshold_m": float(threshold_m),
        "method": "wrist_to_object_obb_proxy",
    }


class ClipEvaluator:
    def __init__(
        self,
        project_root: Path,
        config: dict[str, Any],
        clip: str,
    ):
        self.project_root = project_root.resolve()
        self.config = config
        self.clip = clip
        output_value = Path(config["output"]["root"])
        self.output_root = (
            output_value.resolve()
            if output_value.is_absolute()
            else (self.project_root / output_value).resolve()
        )
        self.clip_dir = self.output_root / clip
        self.quality = config.get("quality_evaluation", {})
        self.threshold = self.quality.get("thresholds", {})
        self.metrics: dict[str, dict[str, Any]] = {}
        self.stage_status: dict[str, str] = {}
        self.stage_scores: dict[str, float] = {}
        self.critical: list[str] = []
        self.warnings: list[str] = []
        self.frame_counts: dict[str, int] = {}
        self.body: dict[str, np.ndarray] | None = None
        self.hands: dict[str, np.ndarray] | None = None
        self.robot: dict[str, Any] | None = None
        self.object_gmr: dict[str, np.ndarray] | None = None
        self.body_path: Path | None = None
        self.object_expected = self._object_expected()

    def _object_expected(self) -> bool:
        object_config = self.config.get("object", {})
        if not object_config.get("enabled", False):
            return False
        configured = object_config.get("clips", {})
        return (
            not object_config.get("only_configured_clips", True)
            or self.clip in configured
        )

    def _warn(self, reason: str) -> None:
        if reason not in self.warnings:
            self.warnings.append(reason)

    def _critical(self, reason: str) -> None:
        if reason not in self.critical:
            self.critical.append(reason)

    def _set_stage(
        self,
        name: str,
        status: str,
        score: float,
        reasons: Iterable[str] = (),
        *,
        critical: bool = False,
    ) -> None:
        self.stage_status[name] = status
        self.stage_scores[name] = round(float(np.clip(score, 0.0, 100.0)), 2)
        if status == "pass":
            return
        for reason in reasons:
            if critical and status == "fail":
                self._critical(reason)
            else:
                self._warn(reason)

    def _required_files(self) -> list[Path]:
        gmr_source = str(self.config.get("gmr", {}).get("source", "smoothed"))
        source_path, _ = _resolve_body_source(self.clip_dir, gmr_source)
        required = [
            source_path,
            self.clip_dir / "001_smplx_hands.npz",
            self.clip_dir / "robot_motion.pkl",
            self.clip_dir / "gvhmr_camera.npz",
        ]
        if self.config.get("phc", {}).get("enabled", False):
            required.append(self.clip_dir / "001_phc_smoothed.npz")
        if self.config.get("gmr", {}).get("hand_model", "sharpa") == "sharpa":
            required.append(self.clip_dir / "001_sharpa_chain_hands.npz")
        if self.config.get("gmr", {}).get("composite_2x2", True):
            required.append(self.clip_dir / "composite_2x2.mp4")
        if self.object_expected:
            recon = self.clip_dir / "object_reconstruction"
            required.extend(
                [
                    recon / "object_motion_gvhmr.npz",
                    recon / "object_motion_gmr.npz",
                    recon / "object_adapter_validation.json",
                    recon / "monocular_adapter" / "mesh" / "mesh.obj",
                ]
            )
        return required

    def evaluate_files(self) -> None:
        try:
            required = self._required_files()
        except QualityEvaluationError as exc:
            required = []
            self._critical(str(exc))
        missing = [
            str(path.relative_to(self.clip_dir))
            for path in required
            if not path.is_file()
        ]
        exists = bool(required) and not missing
        ratio = (
            float((len(required) - len(missing)) / len(required))
            if required
            else 0.0
        )
        self.metrics["required_files_exist"] = _metric(
            exists,
            "pass" if exists else "fail",
            details={
                "required_count": len(required),
                "existing_ratio": ratio,
                "missing": missing,
            },
        )
        if not exists:
            self._set_stage(
                "files",
                "fail",
                60.0 * ratio,
                [f"missing_required_file={name}" for name in missing]
                or ["required_file_set_unresolved"],
                critical=True,
            )
            return

        try:
            self.body_path, body_stage = _resolve_body_source(
                self.clip_dir,
                str(self.config.get("gmr", {}).get("source", "smoothed")),
            )
            self.body = _load_npz(self.body_path)
            self.hands = _load_npz(self.clip_dir / "001_smplx_hands.npz")
            self.robot = _load_robot(self.clip_dir / "robot_motion.pkl")
            self.frame_counts["human"] = int(len(self.body["trans"]))
            self.frame_counts["hands"] = int(len(self.hands["left_hand_valid"]))
            self.frame_counts["gmr"] = int(len(self.robot["root_pos"]))
            camera = _load_npz(self.clip_dir / "gvhmr_camera.npz")
            self.frame_counts["camera"] = int(len(camera["T_w2c"]))
            if (self.clip_dir / "001_sharpa_chain_hands.npz").is_file():
                robot_hands = _load_npz(
                    self.clip_dir / "001_sharpa_chain_hands.npz"
                )
                self.frame_counts["robot_hands"] = int(
                    len(robot_hands["left_hand_qpos"])
                )
            composite = _video_metadata(self.clip_dir / "composite_2x2.mp4")
            if composite["readable"]:
                self.frame_counts["preview_2x2"] = int(composite["frames"])
            if self.object_expected:
                self.object_gmr = _load_npz(
                    self.clip_dir
                    / "object_reconstruction"
                    / "object_motion_gmr.npz"
                )
                self.frame_counts["object_gmr"] = int(
                    len(self.object_gmr["position"])
                )
            counts = list(self.frame_counts.values())
            match_ratio = float(min(counts) / max(counts)) if counts else 0.0
            warn_ratio = float(self.threshold.get("frame_count_warn_ratio", 0.995))
            fail_ratio = float(self.threshold.get("frame_count_fail_ratio", 0.98))
            status = (
                "fail"
                if match_ratio < fail_ratio
                else "warn"
                if match_ratio < warn_ratio
                else "pass"
            )
            self.metrics["frame_count_match_ratio"] = _metric(
                match_ratio,
                status,
                details={
                    "counts": self.frame_counts,
                    "body_source": body_stage,
                },
            )
            reasons = (
                [f"frame_count_match_ratio={match_ratio:.4f}"]
                if status != "pass"
                else []
            )
            self._set_stage(
                "files",
                status,
                60.0 + 40.0 * match_ratio,
                reasons,
                critical=status == "fail",
            )
        except Exception as exc:
            self._set_stage(
                "files",
                "fail",
                0.0,
                [f"file_parse_error={exc}"],
                critical=True,
            )

    def evaluate_human(self) -> None:
        if self.body is None:
            self._set_stage(
                "human",
                "fail",
                0.0,
                ["human_motion_unavailable"],
                critical=True,
            )
            return
        try:
            fps = float(_scalar(self.body, "mocap_frame_rate", 30.0))
            translation = np.asarray(self.body["trans"], dtype=np.float64)
            pose = np.asarray(self.body["pose_body"], dtype=np.float64)
            root = np.asarray(self.body["root_orient"], dtype=np.float64)
            finite = bool(
                np.isfinite(translation).all()
                and np.isfinite(pose).all()
                and np.isfinite(root).all()
            )
            speed = (
                np.linalg.norm(np.diff(translation, axis=0), axis=1) * fps
                if len(translation) > 1
                else np.asarray([0.0])
            )
            speed_p95 = float(np.percentile(speed, 95))
            warn = float(self.threshold.get("root_speed_p95_warn_mps", 3.0))
            fail = float(self.threshold.get("root_speed_p95_fail_mps", 6.0))
            status = (
                "fail"
                if not finite or speed_p95 > fail
                else "warn"
                if speed_p95 > warn
                else "pass"
            )
            self.metrics["root_speed_p95"] = _metric(
                speed_p95,
                status,
                unit="m/s",
                source=self.body_path.name if self.body_path else None,
                details={
                    "finite": finite,
                    "maximum_mps": float(speed.max(initial=0.0)),
                },
            )
            reasons = []
            if not finite:
                reasons.append("human_motion_contains_nan_or_inf")
            if speed_p95 > warn:
                reasons.append(f"root_speed_p95={speed_p95:.3f}m/s")
            self._set_stage(
                "human",
                status,
                _lower_is_better_score(speed_p95, warn * 0.5, fail),
                reasons,
                critical=status == "fail",
            )
        except Exception as exc:
            self._set_stage(
                "human",
                "fail",
                0.0,
                [f"human_metric_error={exc}"],
                critical=True,
            )

    def evaluate_hands(self) -> None:
        if self.hands is None:
            self._set_stage(
                "hands",
                "fail",
                0.0,
                ["hand_motion_unavailable"],
                critical=True,
            )
            return
        valid_warn = float(self.threshold.get("hand_valid_warn_ratio", 0.85))
        valid_fail = float(self.threshold.get("hand_valid_fail_ratio", 0.50))
        reproj_warn = float(
            self.threshold.get("hand_reproj_p90_warn_relative", 0.35)
        )
        reproj_fail = float(
            self.threshold.get("hand_reproj_p90_fail_relative", 0.60)
        )
        spike_warn = float(self.threshold.get("hand_spike_warn_ratio", 0.03))
        spike_fail = float(self.threshold.get("hand_spike_fail_ratio", 0.10))
        repair_warn = float(self.threshold.get("hand_repaired_warn_ratio", 0.35))
        repair_fail = float(self.threshold.get("hand_repaired_fail_ratio", 0.60))
        scores = []
        stage = "pass"
        reasons = []
        try:
            for side in ("left", "right"):
                valid = _ratio(self.hands[f"{side}_hand_valid"])
                reproj = _finite_percentile(
                    self.hands[f"{side}_hand_reproj_error_relative"],
                    90,
                )
                spike = _ratio(self.hands[f"{side}_hand_spike_mask"])
                repaired_key = f"{side}_hand_source_repaired"
                repaired = (
                    _ratio(self.hands[repaired_key])
                    if repaired_key in self.hands
                    else 0.0
                )
                values = {
                    f"{side}_hand_valid_ratio": (
                        valid,
                        "fail"
                        if valid < valid_fail
                        else "warn"
                        if valid < valid_warn
                        else "pass",
                    ),
                    f"{side}_hand_reproj_error_relative_p90": (
                        reproj,
                        "fail"
                        if reproj is None or reproj > reproj_fail
                        else "warn"
                        if reproj > reproj_warn
                        else "pass",
                    ),
                    f"{side}_hand_spike_ratio": (
                        spike,
                        "fail"
                        if spike > spike_fail
                        else "warn"
                        if spike > spike_warn
                        else "pass",
                    ),
                    f"{side}_hand_repaired_ratio": (
                        repaired,
                        "fail"
                        if repaired > repair_fail
                        else "warn"
                        if repaired > repair_warn
                        else "pass",
                    ),
                }
                for key, (value, status) in values.items():
                    self.metrics[key] = _metric(value, status)
                    if status == "fail":
                        stage = "fail"
                    elif status == "warn" and stage == "pass":
                        stage = "warn"
                    if status != "pass":
                        text = "missing" if value is None else f"{value:.4f}"
                        reasons.append(f"{key}={text}")
                scores.extend(
                    [
                        _higher_is_better_score(valid, valid_fail, 0.95),
                        _lower_is_better_score(
                            reproj,
                            reproj_warn * 0.6,
                            reproj_fail,
                        ),
                        _lower_is_better_score(
                            spike,
                            0.0,
                            spike_fail,
                        ),
                        _lower_is_better_score(
                            repaired,
                            repair_warn * 0.4,
                            repair_fail,
                        ),
                    ]
                )
            self._set_stage(
                "hands",
                stage,
                float(np.mean(scores)) if scores else 0.0,
                reasons,
                critical=stage == "fail",
            )
        except Exception as exc:
            self._set_stage(
                "hands",
                "fail",
                0.0,
                [f"hand_metric_error={exc}"],
                critical=True,
            )

    def evaluate_gmr(self) -> None:
        if self.robot is None:
            self._set_stage(
                "gmr",
                "fail",
                0.0,
                ["robot_motion_unavailable"],
                critical=True,
            )
            return
        try:
            root_position = np.asarray(self.robot["root_pos"], dtype=np.float64)
            root_rotation = np.asarray(self.robot["root_rot"], dtype=np.float64)
            dof = np.asarray(self.robot["dof_pos"], dtype=np.float64)
            finite = bool(
                np.isfinite(root_position).all()
                and np.isfinite(root_rotation).all()
                and np.isfinite(dof).all()
            )
            norm_error = np.abs(np.linalg.norm(root_rotation, axis=1) - 1.0)
            bad_quaternion_ratio = float(np.mean(norm_error > 1e-3))
            joint = _joint_limit_metrics(
                self.project_root,
                str(self.config.get("gmr", {}).get("hand_model", "sharpa")),
                dof,
                float(self.threshold.get("joint_limit_tolerance_rad", 1e-3)),
            )
            joint_ratio = joint["ratio"]
            warn = float(
                self.threshold.get("joint_limit_violation_warn_ratio", 0.001)
            )
            fail = float(
                self.threshold.get("joint_limit_violation_fail_ratio", 0.01)
            )
            status = "pass"
            reasons = []
            if not finite or bad_quaternion_ratio > 0:
                status = "fail"
            if joint_ratio is None:
                if status == "pass":
                    status = "warn"
                reasons.append("joint_limit_violation_ratio=unavailable")
            elif joint_ratio > fail:
                status = "fail"
                reasons.append(
                    f"joint_limit_violation_ratio={joint_ratio:.6f}"
                )
            elif joint_ratio > warn:
                if status == "pass":
                    status = "warn"
                reasons.append(
                    f"joint_limit_violation_ratio={joint_ratio:.6f}"
                )
            if not finite:
                reasons.append("robot_motion_contains_nan_or_inf")
            if bad_quaternion_ratio > 0:
                reasons.append(
                    f"root_quaternion_bad_ratio={bad_quaternion_ratio:.6f}"
                )
            self.metrics["joint_limit_violation_ratio"] = _metric(
                joint_ratio,
                "fail"
                if joint_ratio is not None and joint_ratio > fail
                else "warn"
                if joint_ratio is None or joint_ratio > warn
                else "pass",
                details=joint,
            )
            self.metrics["gmr_root_quaternion_bad_ratio"] = _metric(
                bad_quaternion_ratio,
                "pass" if bad_quaternion_ratio == 0 else "fail",
                details={
                    "maximum_norm_error": float(norm_error.max(initial=0.0)),
                    "finite": finite,
                },
            )
            joint_score = (
                _lower_is_better_score(joint_ratio, 0.0, fail)
                if joint_ratio is not None
                else 40.0
            )
            score = (
                40.0 * float(finite)
                + 20.0 * float(bad_quaternion_ratio == 0)
                + 40.0 * joint_score / 100.0
            )
            self._set_stage(
                "gmr",
                status,
                score,
                reasons,
                critical=status == "fail",
            )
        except Exception as exc:
            self._set_stage(
                "gmr",
                "fail",
                0.0,
                [f"gmr_metric_error={exc}"],
                critical=True,
            )

    def evaluate_visualization(self) -> None:
        preview_path = self.clip_dir / "composite_2x2.mp4"
        metadata = _video_metadata(preview_path)
        human_frames = self.frame_counts.get("human", 0)
        ratio = (
            min(metadata["frames"], human_frames)
            / max(metadata["frames"], human_frames)
            if metadata["frames"] and human_frames
            else 0.0
        )
        readable = bool(metadata["readable"])
        warn_ratio = float(self.threshold.get("frame_count_warn_ratio", 0.995))
        fail_ratio = float(self.threshold.get("frame_count_fail_ratio", 0.98))
        status = (
            "fail"
            if not readable or ratio < fail_ratio
            else "warn"
            if ratio < warn_ratio
            else "pass"
        )
        self.metrics["visualization_frame_match_ratio"] = _metric(
            ratio,
            status,
            details={"path": preview_path.name, **metadata},
        )
        reasons = (
            []
            if status == "pass"
            else [
                "preview_2x2_unreadable"
                if not readable
                else f"visualization_frame_match_ratio={ratio:.4f}"
            ]
        )
        self._set_stage(
            "visualization",
            status,
            60.0 * float(readable) + 40.0 * ratio,
            reasons,
            critical=status == "fail",
        )

    def evaluate_object(self) -> None:
        if not self.object_expected:
            return
        recon = self.clip_dir / "object_reconstruction"
        unavailable = recon / "object_unavailable.json"
        if unavailable.is_file():
            report = _read_json(unavailable)
            self._set_stage(
                "object",
                "fail",
                0.0,
                [
                    "object_unavailable="
                    f"{report.get('stage', 'object')}:{report.get('status', 'failed')}"
                ],
                critical=True,
            )
            return
        if self.object_gmr is None:
            self._set_stage(
                "object",
                "fail",
                0.0,
                ["object_motion_unavailable"],
                critical=True,
            )
            return
        try:
            valid = np.asarray(self.object_gmr["valid"], dtype=bool).reshape(-1)
            valid_ratio = float(np.mean(valid)) if len(valid) else 0.0
            valid_warn = float(
                self.threshold.get("object_valid_warn_ratio", 0.80)
            )
            valid_fail = float(
                self.threshold.get("object_valid_fail_ratio", 0.35)
            )
            valid_status = (
                "fail"
                if valid_ratio < valid_fail
                else "warn"
                if valid_ratio < valid_warn
                else "pass"
            )
            self.metrics["object_valid_ratio"] = _metric(
                valid_ratio,
                valid_status,
            )
            jumps = _object_pose_jump_metrics(
                self.object_gmr["position"],
                self.object_gmr["quat_wxyz"],
                valid,
                float(
                    self.threshold.get("object_position_jump_m", 0.20)
                ),
                float(
                    self.threshold.get("object_rotation_jump_deg", 45.0)
                ),
            )
            jump_warn = float(
                self.threshold.get("object_jump_warn_ratio", 0.002)
            )
            jump_fail = float(
                self.threshold.get("object_jump_fail_ratio", 0.02)
            )
            jump_status = (
                "fail"
                if jumps["ratio"] > jump_fail
                else "warn"
                if jumps["ratio"] > jump_warn
                else "pass"
            )
            self.metrics["object_pose_jump_count"] = _metric(
                jumps["count"],
                jump_status,
                details=jumps,
            )

            projection_report = (
                recon / "projection_validation" / "projection_report.json"
            )
            if projection_report.is_file():
                projection = _read_json(projection_report)
                iou = projection.get("mask_iou_mean")
                projection_details = {
                    "samples": projection.get("frames_sampled"),
                    "source": "projection_report.json",
                }
            else:
                work_value = Path(
                    self.config.get("object", {})
                    .get("monocular", {})
                    .get("work_root", "object_work/do_as_i_do")
                )
                work_root = (
                    work_value.resolve()
                    if work_value.is_absolute()
                    else (self.project_root / work_value).resolve()
                )
                object_name = str(
                    self.config.get("object", {})
                    .get("clips", {})
                    .get(self.clip, {})
                    .get("object_name", "object")
                )
                adapter_manifest_path = (
                    recon / "monocular_adapter" / "adapter_manifest.json"
                )
                if adapter_manifest_path.is_file():
                    object_name = str(
                        _read_json(adapter_manifest_path).get(
                            "object_name",
                            object_name,
                        )
                    )
                projection_details = _compute_projection_iou(
                    recon / "monocular_adapter",
                    work_root / self.clip / "video_segmentation" / "masks",
                    object_name,
                    int(self.quality.get("projection_samples", 40)),
                )
                iou = projection_details["mean"]
            iou = float(iou) if iou is not None else None
            iou_warn = float(
                self.threshold.get("projection_iou_warn", 0.30)
            )
            iou_fail = float(
                self.threshold.get("projection_iou_fail", 0.15)
            )
            iou_status = (
                "fail"
                if iou is not None and iou < iou_fail
                else "warn"
                if iou is None or iou < iou_warn
                else "pass"
            )
            self.metrics["projection_bbox_iou_mean"] = _metric(
                iou,
                iou_status,
                details=projection_details,
            )
            statuses = (valid_status, jump_status, iou_status)
            stage = (
                "fail"
                if "fail" in statuses
                else "warn"
                if "warn" in statuses
                else "pass"
            )
            reasons = []
            if valid_status != "pass":
                reasons.append(f"object_valid_ratio={valid_ratio:.4f}")
            if jump_status != "pass":
                reasons.append(
                    f"object_pose_jump_count={jumps['count']}"
                )
            if iou_status != "pass":
                reasons.append(
                    "projection_bbox_iou_mean="
                    + ("missing" if iou is None else f"{iou:.4f}")
                )
            score = float(
                np.mean(
                    [
                        _higher_is_better_score(valid_ratio, valid_fail, 0.95),
                        _lower_is_better_score(
                            jumps["ratio"],
                            0.0,
                            jump_fail,
                        ),
                        _higher_is_better_score(
                            iou,
                            iou_fail,
                            0.50,
                        ),
                    ]
                )
            )
            self._set_stage(
                "object",
                stage,
                score,
                reasons,
                critical=stage == "fail",
            )
        except Exception as exc:
            self._set_stage(
                "object",
                "fail",
                0.0,
                [f"object_metric_error={exc}"],
                critical=True,
            )

    def evaluate_contact_and_dynamics(self) -> None:
        if not self.object_expected:
            return
        recon = self.clip_dir / "object_reconstruction"
        if self.robot is None or self.object_gmr is None:
            self.metrics["contact_frame_ratio"] = _metric(
                None,
                "fail",
                details={"method": "unavailable"},
            )
            self._set_stage(
                "contact",
                "fail",
                0.0,
                ["contact_frame_ratio=unavailable"],
            )
        else:
            mesh_path = Path(str(_scalar(self.object_gmr, "visual_mesh_path", "")))
            contact = _contact_proxy_metrics(
                self.robot,
                self.object_gmr,
                mesh_path,
                float(self.threshold.get("contact_distance_m", 0.15)),
            )
            ratio = contact["ratio"]
            pass_ratio = float(
                self.threshold.get("contact_frame_pass_ratio", 0.10)
            )
            fail_ratio = float(
                self.threshold.get("contact_frame_fail_ratio", 0.01)
            )
            status = (
                "fail"
                if ratio is None or ratio < fail_ratio
                else "warn"
                if ratio < pass_ratio
                else "pass"
            )
            self.metrics["contact_frame_ratio"] = _metric(
                ratio,
                status,
                details=contact,
            )
            reason = (
                []
                if status == "pass"
                else [
                    "contact_frame_ratio="
                    + ("unavailable" if ratio is None else f"{ratio:.4f}")
                ]
            )
            self._set_stage(
                "contact",
                status,
                _higher_is_better_score(ratio, fail_ratio, pass_ratio),
                reason,
            )

        dynamic_config = self.config.get("object", {}).get(
            "dynamic_validation",
            {},
        )
        dynamic_path = recon / "object_dynamic_sim.json"
        if dynamic_path.is_file():
            dynamic = _read_json(dynamic_path)
            success = bool(dynamic.get("dynamic_lift_success", False))
            exploded = bool(dynamic.get("object_exploded", False))
            force_spike = bool(dynamic.get("invalid_force_spike", False))
            frames = int(dynamic.get("frames", 0) or 0)
            release = int(dynamic.get("release_frame", 0) or 0)
            contact_frames = int(
                dynamic.get("contact_frames_after_release", 0) or 0
            )
            denominator = max(frames - release, 1)
            dynamic_contact_ratio = float(contact_frames / denominator)
            status = (
                "pass"
                if success and not exploded and not force_spike
                else "fail"
            )
            self.metrics["dynamic_lift_success"] = _metric(
                success,
                status,
                details={
                    "contact_frame_ratio": dynamic_contact_ratio,
                    "object_exploded": exploded,
                    "invalid_force_spike": force_spike,
                    "max_lift_m": dynamic.get("max_lift_m"),
                },
            )
            reasons = []
            if not success:
                reasons.append("dynamic_lift_success=false")
            if exploded:
                reasons.append("dynamic_object_exploded=true")
            if force_spike:
                reasons.append("dynamic_invalid_force_spike=true")
            self._set_stage(
                "dynamics",
                status,
                100.0 if status == "pass" else 0.0,
                reasons,
                critical=bool(dynamic_config.get("required", False)),
            )
        else:
            required = bool(
                dynamic_config.get("enabled", False)
                or dynamic_config.get("required", False)
            )
            status = "fail" if required else "warn"
            self.metrics["dynamic_lift_success"] = _metric(
                None,
                status,
                details={"reason": "dynamic_validation_not_run"},
            )
            self._set_stage(
                "dynamics",
                status,
                0.0,
                ["dynamic_validation=not_run"],
                critical=required,
            )

    def finalize(self) -> dict[str, Any]:
        mode = "object" if self.object_expected else "human_only"
        weights = self.quality.get("weights", {}).get(mode, {})
        if not weights:
            weights = (
                {
                    "files": 15,
                    "human": 25,
                    "hands": 35,
                    "gmr": 15,
                    "visualization": 10,
                }
                if mode == "human_only"
                else {
                    "files": 10,
                    "human": 15,
                    "hands": 20,
                    "object": 20,
                    "contact": 20,
                    "dynamics": 15,
                }
            )
        weighted = []
        total_weight = 0.0
        components = {}
        for stage, raw_weight in weights.items():
            weight = float(raw_weight)
            score = float(self.stage_scores.get(stage, 0.0))
            contribution = weight * score / 100.0
            components[stage] = {
                "weight": weight,
                "score": round(score, 2),
                "contribution": round(contribution, 2),
            }
            weighted.append(contribution)
            total_weight += weight
        overall = (
            100.0 * sum(weighted) / total_weight
            if total_weight > 0
            else 0.0
        )
        pass_score = float(self.quality.get("pass_score", 80.0))
        if not self.critical and overall < pass_score:
            self._warn(f"overall_score={overall:.2f}<{pass_score:.2f}")
        has_noncritical_failure = any(
            value == "fail"
            for key, value in self.stage_status.items()
            if key not in {"files", "human", "hands", "gmr", "object"}
        )
        status = (
            "fail"
            if self.critical
            else "warn"
            if self.warnings or has_noncritical_failure
            else "pass"
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "clip": self.clip,
            "mode": mode,
            "status": status,
            "stage_status": self.stage_status,
            "stage_scores": self.stage_scores,
            "overall_score": round(overall, 2),
            "score_components": components,
            "metrics": self.metrics,
            "critical_fail_reasons": self.critical,
            "warn_reasons": self.warnings,
        }

    def run(self) -> dict[str, Any]:
        if not self.clip_dir.is_dir():
            raise QualityEvaluationError(f"clip directory is missing: {self.clip_dir}")
        self.evaluate_files()
        self.evaluate_human()
        self.evaluate_hands()
        self.evaluate_gmr()
        if not self.object_expected:
            self.evaluate_visualization()
        else:
            self.evaluate_object()
            self.evaluate_contact_and_dynamics()
        return self.finalize()


def _summary_value(report: dict[str, Any], metric: str) -> Any:
    return report.get("metrics", {}).get(metric, {}).get("value")


def _write_batch_summary(output_root: Path) -> None:
    reports = []
    for path in sorted(output_root.glob("*/quality_report.json")):
        try:
            reports.append(_read_json(path))
        except QualityEvaluationError:
            continue
    jsonl_path = output_root / "quality_summary.jsonl"
    csv_path = output_root / "quality_summary.csv"
    handle, temporary_name = tempfile.mkstemp(
        prefix=".quality-summary-",
        suffix=".jsonl",
        dir=output_root,
        text=True,
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            for report in reports:
                stream.write(json.dumps(report, ensure_ascii=False) + "\n")
        os.replace(temporary_name, jsonl_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    fields = [
        "clip",
        "mode",
        "status",
        "overall_score",
        "files",
        "human",
        "hands",
        "gmr",
        "visualization",
        "object",
        "contact",
        "dynamics",
        "required_files_exist",
        "frame_count_match_ratio",
        "left_hand_valid_ratio",
        "right_hand_valid_ratio",
        "left_hand_reproj_error_relative_p90",
        "right_hand_reproj_error_relative_p90",
        "left_hand_spike_ratio",
        "right_hand_spike_ratio",
        "left_hand_repaired_ratio",
        "right_hand_repaired_ratio",
        "root_speed_p95",
        "joint_limit_violation_ratio",
        "object_valid_ratio",
        "object_pose_jump_count",
        "projection_bbox_iou_mean",
        "contact_frame_ratio",
        "dynamic_lift_success",
        "critical_fail_reasons",
        "warn_reasons",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            stage = report.get("stage_status", {})
            row = {
                "clip": report.get("clip"),
                "mode": report.get("mode"),
                "status": report.get("status"),
                "overall_score": report.get("overall_score"),
                **{name: stage.get(name, "") for name in fields[4:12]},
                **{
                    name: _summary_value(report, name)
                    for name in fields[12:29]
                },
                "critical_fail_reasons": " | ".join(
                    report.get("critical_fail_reasons", [])
                ),
                "warn_reasons": " | ".join(report.get("warn_reasons", [])),
            }
            writer.writerow(row)


def evaluate_clips(
    project_root: Path,
    config: dict[str, Any],
    clips: Iterable[str],
    *,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    project_root = project_root.resolve()
    output_value = Path(config["output"]["root"])
    output_root = (
        output_value.resolve()
        if output_value.is_absolute()
        else (project_root / output_value).resolve()
    )
    reports = []
    for clip in clips:
        if dry_run:
            print(f"[DRY-RUN] evaluate quality: {clip}", flush=True)
            continue
        report = ClipEvaluator(project_root, config, clip).run()
        destination = output_root / clip / "quality_report.json"
        _write_json(destination, report)
        print(
            f"[QUALITY] {clip}: {report['status']} "
            f"score={report['overall_score']:.2f} -> {destination}",
            flush=True,
        )
        reports.append(report)
    if not dry_run:
        _write_batch_summary(output_root)
        print(
            f"[QUALITY] summaries -> {output_root / 'quality_summary.csv'}",
            flush=True,
        )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate completed clips through layered pipeline YAML."
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
    runner.check("quality")
    clips = [path.stem for path in runner._quality_videos()]
    evaluate_clips(runner.root, runner.config, clips, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
