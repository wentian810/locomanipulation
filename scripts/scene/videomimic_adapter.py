"""Export existing locomanipulation outputs to VideoMimic's Stage-2 contract.

This adapter intentionally reuses the project's whole-body ViTPose tensor and
SMPL motion.  It never invokes VideoMimic's ViTPose or VIMO front ends.
"""

from __future__ import annotations

import json
import math
import pickle
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
MOTION_CANDIDATES = (
    "001_final.npz",
    "001_phc_smoothed_grounded.npz",
    "001_phc_smoothed.npz",
    "001_smoothed.npz",
)


class AdapterError(RuntimeError):
    """Raised for a contract violation that must stop reconstruction."""


@dataclass(frozen=True)
class ExportLayout:
    root: Path
    clip_id: str

    @property
    def inputs(self) -> Path:
        return self.root / "inputs"

    @property
    def videomimic(self) -> Path:
        return self.root / "videomimic"

    @property
    def image_dir(self) -> Path:
        return self.videomimic / "input_images" / self.clip_id / "cam01"

    @property
    def mask_root(self) -> Path:
        return self.videomimic / "input_masks" / self.clip_id / "cam01"

    @property
    def mask_data(self) -> Path:
        return self.mask_root / "mask_data"

    @property
    def bbox_data(self) -> Path:
        return self.mask_root / "json_data"

    @property
    def pose_dir(self) -> Path:
        return self.videomimic / "input_2d_poses" / self.clip_id / "cam01"

    @property
    def smpl_dir(self) -> Path:
        return self.videomimic / "input_3d_meshes" / self.clip_id / "cam01"

    @property
    def frame_map(self) -> Path:
        return self.inputs / "frame_map.npz"

    @property
    def report(self) -> Path:
        return self.root / "adapter_report.json"

    @property
    def status(self) -> Path:
        return self.root / "status.json"


def _json_dump(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise AdapterError(f"missing {label}: {path}")
    return path


def _require_empty_or_new(root: Path, overwrite: bool) -> None:
    entries = list(root.iterdir()) if root.exists() else []
    # The orchestration runner writes its command log before invoking this
    # subprocess.  It is not an adapter artifact and is safe to preserve.
    ignorable = {"command_log.txt"}
    if root.exists() and any(entry.name not in ignorable for entry in entries):
        if not overwrite:
            raise AdapterError(
                f"output already exists and is non-empty: {root}; use --overwrite only "
                "for this generated scene_work directory"
            )
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def resolve_motion_path(clip_dir: Path, requested: Path | None) -> tuple[Path, str]:
    if requested is not None:
        candidate = requested if requested.is_absolute() else clip_dir / requested
        return _require_file(candidate, "requested motion NPZ"), "explicit"
    for name in MOTION_CANDIDATES:
        candidate = clip_dir / name
        if candidate.is_file():
            return candidate, "automatic"
    raise AdapterError("no compatible motion NPZ found; tried " + ", ".join(MOTION_CANDIDATES))


def resolve_human_output_dir(clip_dir: Path, requested: Path | None) -> Path:
    if requested is not None:
        return _require_file(requested / "vitpose_wholebody.pt", "vitpose_wholebody.pt").parent
    candidate = clip_dir / "gvhmr_out" / clip_dir.name
    _require_file(candidate / "vitpose_wholebody.pt", "vitpose_wholebody.pt")
    _require_file(candidate / "preprocess" / "bbx.pt", "preprocess/bbx.pt")
    return candidate


def _load_torch(path: Path) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment error
        raise AdapterError("the exporter must run in vm1recon (PyTorch is required)") from exc
    return torch.load(path, map_location="cpu")


def _as_numpy(value: Any, label: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise AdapterError(f"{label} contains NaN or Inf")
    return array


def _axis_angle_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Vectorised Rodrigues conversion without depending on SciPy."""
    value = np.asarray(rotvec, dtype=np.float64)
    if value.shape[-1] != 3:
        raise AdapterError(f"axis-angle array must end in 3, got {value.shape}")
    flat = value.reshape(-1, 3)
    theta = np.linalg.norm(flat, axis=1)
    k = np.zeros_like(flat)
    nonzero = theta > 1.0e-12
    k[nonzero] = flat[nonzero] / theta[nonzero, None]
    kx = np.zeros((len(flat), 3, 3), dtype=np.float64)
    kx[:, 0, 1] = -k[:, 2]
    kx[:, 0, 2] = k[:, 1]
    kx[:, 1, 0] = k[:, 2]
    kx[:, 1, 2] = -k[:, 0]
    kx[:, 2, 0] = -k[:, 1]
    kx[:, 2, 1] = k[:, 0]
    eye = np.eye(3, dtype=np.float64)[None]
    sin = np.sin(theta)[:, None, None]
    cos = np.cos(theta)[:, None, None]
    result = eye + sin * kx + (1.0 - cos) * (kx @ kx)
    result[~nonzero] = eye
    return result.reshape(value.shape[:-1] + (3, 3)).astype(np.float32)


def _rotation_quality(rotations: np.ndarray) -> dict[str, float]:
    identity = np.eye(3, dtype=np.float32)
    orthogonality = np.max(np.abs(np.swapaxes(rotations, -1, -2) @ rotations - identity))
    determinant = np.linalg.det(rotations)
    return {
        "orthogonality_max_abs": float(orthogonality),
        "determinant_min": float(determinant.min()),
        "determinant_max": float(determinant.max()),
    }


def _video_info(video: Path) -> tuple[int, int, int]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,nb_read_frames", "-of", "json", str(video),
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    streams = json.loads(completed.stdout).get("streams", [])
    if len(streams) != 1:
        raise AdapterError(f"ffprobe could not read one video stream from {video}")
    stream = streams[0]
    raw_count = stream.get("nb_read_frames")
    if raw_count in {None, "N/A"}:
        raise AdapterError(f"ffprobe did not return a reliable frame count for {video}")
    return int(stream["width"]), int(stream["height"]), int(raw_count)


def _extract_frames(video: Path, destination: Path, count: int) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
        "-frames:v", str(count), "-start_number", "0", "-q:v", "2",
        str(destination / "frame_%05d.jpg"),
    ]
    subprocess.run(command, check=True)
    produced = sorted(destination.glob("frame_*.jpg"))
    if len(produced) != count:
        raise AdapterError(f"ffmpeg produced {len(produced)} frames, expected {count}")
    expected_names = [f"frame_{index:05d}.jpg" for index in range(count)]
    if [path.name for path in produced] != expected_names:
        raise AdapterError("extracted frame names are not contiguous zero-based indices")


def _extract_sampled_frames(
    video: Path, destination: Path, source_indices: np.ndarray, total: int
) -> None:
    indices = np.asarray(source_indices, dtype=np.int64)
    if indices.ndim != 1 or not len(indices) or indices[0] < 0 or indices[-1] >= total:
        raise AdapterError("sampled frame indices are outside the source video")
    scratch = destination.parent / ("." + destination.name + ".all_frames")
    if scratch.exists():
        raise AdapterError(f"temporary frame extraction directory already exists: {scratch}")
    try:
        _extract_frames(video, scratch, total)
        destination.mkdir(parents=True, exist_ok=True)
        for output_index, source_index in enumerate(indices):
            shutil.copy2(
                scratch / f"frame_{int(source_index):05d}.jpg",
                destination / f"frame_{output_index:05d}.jpg",
            )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _clamped_bbox(raw_box: np.ndarray, width: int, height: int, pose: np.ndarray) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(item) for item in raw_box[:4])
    x1, x2 = sorted((max(0.0, x1), min(float(width), x2)))
    y1, y2 = sorted((max(0.0, y1), min(float(height), y2)))
    if x2 - x1 >= 2.0 and y2 - y1 >= 2.0:
        return math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)
    confident = pose[:, 2] > 0.05
    if not np.any(confident):
        raise AdapterError("encountered an invalid bbox and no confident 2D pose to recover it")
    points = pose[confident, :2]
    x1, y1 = np.maximum(points.min(axis=0) - 4.0, 0.0)
    x2, y2 = np.minimum(points.max(axis=0) + 4.0, [width, height])
    if x2 - x1 < 2.0 or y2 - y1 < 2.0:
        raise AdapterError("recovered bbox is still invalid")
    return math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)


def _load_sources(
    motion_path: Path, camera_path: Path, human_output: Path
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    with np.load(motion_path, allow_pickle=False) as archive:
        required = {"poses", "root_orient", "pose_body", "trans", "betas", "gender", "mocap_frame_rate"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise AdapterError(f"motion NPZ is missing fields: {', '.join(missing)}")
        motion = {key: archive[key] for key in archive.files}
    with np.load(camera_path, allow_pickle=False) as archive:
        required = {"T_w2c", "K_fullimg", "gravity_axis"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise AdapterError(f"camera NPZ is missing fields: {', '.join(missing)}")
        camera = {key: archive[key] for key in archive.files}
    poses = _as_numpy(motion["poses"], "motion.poses")
    root = _as_numpy(motion["root_orient"], "motion.root_orient")
    if poses.ndim != 3 or poses.shape[1:] != (24, 3):
        raise AdapterError(f"motion.poses must have shape (T,24,3), got {poses.shape}")
    if root.shape != (poses.shape[0], 3):
        raise AdapterError(f"motion.root_orient must have shape (T,3), got {root.shape}")
    if not np.allclose(poses[:, 0], root, atol=1.0e-5):
        raise AdapterError("motion.poses[:,0] is inconsistent with root_orient")
    vitpose = _as_numpy(_load_torch(human_output / "vitpose_wholebody.pt"), "vitpose_wholebody")
    bbox_data = _load_torch(human_output / "preprocess" / "bbx.pt")
    if not isinstance(bbox_data, dict) or "bbx_xyxy" not in bbox_data or "bbx_conf" not in bbox_data:
        raise AdapterError("preprocess/bbx.pt must contain bbx_xyxy and bbx_conf")
    bboxes = _as_numpy(bbox_data["bbx_xyxy"], "bbx_xyxy")
    bbox_conf = _as_numpy(bbox_data["bbx_conf"], "bbx_conf")
    if vitpose.ndim == 4 and vitpose.shape[0] == 1:
        vitpose = vitpose[0]
    if bboxes.ndim == 3 and bboxes.shape[0] == 1:
        bboxes = bboxes[0]
    if bbox_conf.ndim == 2 and bbox_conf.shape[0] == 1:
        bbox_conf = bbox_conf[0]
    expected = poses.shape[0]
    if vitpose.shape != (expected, 133, 3):
        raise AdapterError(f"vitpose_wholebody must have shape ({expected},133,3), got {vitpose.shape}")
    if bboxes.shape != (expected, 4) or bbox_conf.shape != (expected,):
        raise AdapterError(f"bbox data must have shapes ({expected},4) and ({expected},), got {bboxes.shape} / {bbox_conf.shape}")
    return motion, camera, vitpose.astype(np.float32), bboxes.astype(np.float32), bbox_conf.astype(np.float32)


def export_videomimic_inputs(
    *,
    clip_dir: Path,
    work_video: Path,
    output_root: Path,
    motion_path: Path | None = None,
    camera_path: Path | None = None,
    human_output_dir: Path | None = None,
    clip_id: str | None = None,
    person_id: int = 1,
    max_frames: int = 0,
    frame_stride: int = 1,
    mask_mode: str = "bbox",
    overwrite: bool = False,
    static_camera_warn_threshold: float = 0.05,
) -> dict[str, Any]:
    """Create VideoMimic-compatible frames, masks, 2-D pose and SMPL pickles."""
    clip_dir = clip_dir.resolve()
    work_video = _require_file(work_video.resolve(), "work video")
    motion_file, selection_mode = resolve_motion_path(clip_dir, motion_path)
    camera_file = _require_file(
        (camera_path if camera_path is not None else clip_dir / "gvhmr_camera.npz").resolve(),
        "camera NPZ",
    )
    human_output = resolve_human_output_dir(clip_dir, human_output_dir)
    selected_clip_id = clip_id or clip_dir.name
    if not selected_clip_id or "/" in selected_clip_id or "\\" in selected_clip_id:
        raise AdapterError("clip_id must be a non-empty directory name")
    if person_id <= 0:
        raise AdapterError("person_id must be positive")
    if mask_mode not in {"bbox", "sam2"}:
        raise AdapterError(f"mask_mode must be 'bbox' or 'sam2', got {mask_mode!r}")
    _require_empty_or_new(output_root, overwrite)
    layout = ExportLayout(output_root, selected_clip_id)
    motion, camera, vitpose, bboxes, bbox_conf = _load_sources(motion_file, camera_file, human_output)
    total = motion["poses"].shape[0]
    width, height, video_frames = _video_info(work_video)
    if video_frames != total:
        raise AdapterError(
            f"frame-count mismatch: work video has {video_frames} frames but motion has {total}; "
            "refuse to truncate or cycle either sequence"
        )
    if frame_stride <= 0:
        raise AdapterError(f"frame_stride must be positive, got {frame_stride}")
    indices = np.arange(0, total, frame_stride, dtype=np.int64)
    if indices[-1] != total - 1:
        indices = np.append(indices, total - 1)
    if max_frames:
        if max_frames <= 0 or max_frames > len(indices):
            raise AdapterError(f"max_frames must be 0 or within 1..{len(indices)}, got {max_frames}")
        indices = indices[:max_frames]
    frame_count = int(len(indices))
    T_w2c = _as_numpy(camera["T_w2c"], "T_w2c").astype(np.float32)
    K = _as_numpy(camera["K_fullimg"], "K_fullimg").astype(np.float32)
    if T_w2c.shape != (total, 4, 4) or K.shape != (total, 3, 3):
        raise AdapterError(f"camera shapes must be ({total},4,4)/({total},3,3), got {T_w2c.shape}/{K.shape}")
    if not np.allclose(K, K[0], atol=1.0e-4):
        raise AdapterError("K_fullimg varies across frames; the current adapter requires one fixed intrinsic matrix")
    root_translation_world = _as_numpy(motion["trans"], "motion.trans").astype(np.float32)
    if root_translation_world.shape != (total, 3):
        raise AdapterError(f"expected motion.trans shape ({total},3), got {root_translation_world.shape}")
    root_world = _axis_angle_to_matrix(motion["root_orient"][indices])
    body = _axis_angle_to_matrix(motion["poses"][indices, 1:24])
    root_camera = np.einsum("tij,tjk->tik", T_w2c[indices, :3, :3], root_world).astype(np.float32)
    root_translation_camera = (
        np.einsum("tij,tj->ti", T_w2c[indices, :3, :3], root_translation_world[indices])
        + T_w2c[indices, :3, 3]
    ).astype(np.float32)
    root_quality = _rotation_quality(root_camera)
    body_quality = _rotation_quality(body)
    if root_quality["orthogonality_max_abs"] > 1.0e-4 or root_quality["determinant_min"] < 0.999:
        raise AdapterError(f"invalid converted root rotations: {root_quality}")
    if body_quality["orthogonality_max_abs"] > 1.0e-4 or body_quality["determinant_min"] < 0.999:
        raise AdapterError(f"invalid body rotations: {body_quality}")
    _extract_sampled_frames(work_video, layout.image_dir, indices, total)
    layout.inputs.mkdir(parents=True, exist_ok=True)
    layout.mask_data.mkdir(parents=True, exist_ok=True)
    layout.bbox_data.mkdir(parents=True, exist_ok=True)
    layout.pose_dir.mkdir(parents=True, exist_ok=True)
    layout.smpl_dir.mkdir(parents=True, exist_ok=True)
    beta = _as_numpy(motion["betas"], "motion.betas").reshape(-1)[:10].astype(np.float32)
    if beta.shape != (10,):
        raise AdapterError(f"expected at least 10 body-shape betas, got {motion['betas'].shape}")
    for index, source_index in enumerate(indices):
        source_index = int(source_index)
        pose = vitpose[source_index]
        x1, y1, x2, y2 = _clamped_bbox(bboxes[source_index], width, height, pose)
        score = float(np.clip(bbox_conf[source_index], 0.0, 1.0))
        mask = np.zeros((height, width), dtype=np.uint16)
        mask[y1:y2, x1:x2] = person_id
        np.savez_compressed(layout.mask_data / f"mask_{index:05d}.npz", mask=mask)
        label = {
            "instance_id": person_id,
            "class_name": "person",
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "score": score,
        }
        _json_dump(
            layout.bbox_data / f"mask_{index:05d}.json",
            {
                "mask_name": f"mask_{index:05d}.npz",
                "mask_height": height,
                "mask_width": width,
                "labels": {str(person_id): label},
                "area_ranking": [person_id],
                "areas": {str(person_id): (x2 - x1) * (y2 - y1)},
            },
        )
        _json_dump(
            layout.pose_dir / f"pose_{index:05d}.json",
            {str(person_id): {"bbox": [x1, y1, x2, y2, score], "keypoints": pose.tolist()}},
        )
        with (layout.smpl_dir / f"smpl_params_{index:05d}.pkl").open("wb") as handle:
            pickle.dump(
                {
                    person_id: {
                        "smpl_params": {
                            "global_orient": root_camera[index][None],
                            "body_pose": body[index],
                            "betas": beta,
                            "source_root_transl_camera": root_translation_camera[index][None],
                        }
                    }
                },
                handle,
                protocol=4,
            )
    meta = {
        "all_instance_ids": [person_id],
        "most_persistent_id": person_id,
        "largest_area_id": person_id,
        "highest_confidence_id": person_id,
        "sorted_by_avg_area": [person_id],
        "frame_counts": {str(person_id): frame_count},
        "total_areas": {str(person_id): int(sum((max(0, _clamped_bbox(bboxes[source_index], width, height, vitpose[source_index])[2] - _clamped_bbox(bboxes[source_index], width, height, vitpose[source_index])[0]) * max(0, _clamped_bbox(bboxes[source_index], width, height, vitpose[source_index])[3] - _clamped_bbox(bboxes[source_index], width, height, vitpose[source_index])[1])) for source_index in indices))},
    }
    meta["avg_areas"] = {str(person_id): meta["total_areas"][str(person_id)] / frame_count}
    _json_dump(layout.mask_root / "meta_data.json", meta)
    indices = indices.astype(np.int64, copy=False)
    np.savez_compressed(
        layout.frame_map,
        source_video_frame_index=indices,
        work_video_frame_index=indices,
        human_frame_index=indices,
        videomimic_frame_index=np.arange(frame_count, dtype=np.int64),
        timestamp_seconds=indices.astype(np.float64) / float(motion["mocap_frame_rate"]),
    )
    camera_delta = np.abs(T_w2c[indices] - T_w2c[indices[0]]).max(axis=(1, 2))
    warnings: list[str] = []
    if float(camera_delta.max()) > static_camera_warn_threshold:
        warnings.append(
            "gvhmr T_w2c varies across frames; this is recorded for Phase-2 coordinate "
            "validation and must not be treated as a fixed physical-camera proof"
        )
    gender = str(np.asarray(motion["gender"]).item()).lower()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "clip_id": selected_clip_id,
        "frame_count": frame_count,
        "source_frame_count": total,
        "frame_stride": frame_stride,
        "source_frame_indices": indices.astype(int).tolist(),
        "work_video": str(work_video),
        "human_source": str(motion_file),
        "human_source_selection": selection_mode,
        "camera_source": str(camera_file),
        "vitpose_source": str(human_output / "vitpose_wholebody.pt"),
        "bbox_source": str(human_output / "preprocess" / "bbx.pt"),
        "person_id": person_id,
        "gender": gender,
        "smpl_mapping": "poses[:, 1:24] -> 23 SMPL body joints",
        "root_rotation_mapping": "R_cam_body = T_w2c[:3,:3] @ R_world_body",
        "root_translation_mapping": "t_cam_root = T_w2c[:3,:3] @ t_world_root + T_w2c[:3,3]",
        "mask_source": ("videomimic_official_sam2_person_mask" if mask_mode == "sam2" else "bbox_rectangle_legacy"),
        "rotation_quality": {"root": root_quality, "body": body_quality},
        "image_size": {"width": width, "height": height},
        "camera": {
            "gravity_axis": str(np.asarray(camera["gravity_axis"]).item()),
            "T_w2c_max_abs_delta_from_frame_0": float(camera_delta.max()),
            "K_fullimg_static": True,
        },
        "outputs": {
            "frames": str(layout.image_dir),
            "masks": str(layout.mask_root),
            "pose2d": str(layout.pose_dir),
            "smpl": str(layout.smpl_dir),
            "frame_map": str(layout.frame_map),
        },
        "warnings": warnings,
    }
    _json_dump(layout.report, report)
    _json_dump(layout.status, {"schema_version": SCHEMA_VERSION, "stage": "videomimic_adapter", "status": "pass", "frame_count": frame_count, "warnings": warnings})
    return report
