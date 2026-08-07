#!/usr/bin/env python3
"""Cheap temporal YOLO-Pose admission gate for full-body motion videos.

This is deliberately a pre-GVHMR check.  It does not estimate 3D motion or
judge pose quality; it only rejects clips that do not provide enough 2D
evidence for a full-body + hands pipeline (for example upper-body-only shots).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


KEYPOINT_INDEX = {
    "nose": 0,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_ankle": 15,
    "right_ankle": 16,
}

# COCO person-pose joints used by the cheap external-support posture gate.
# This gate deliberately operates before GVHMR; it is a high-confidence 2D
# admission signal, not a claim of 3D contact reconstruction.
POSTURE_KEYPOINT_INDEX = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


def _jsonl_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        source = value.get("source") if isinstance(value, dict) else None
        if isinstance(source, str):
            records[source] = value
    return records


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(value)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, float(first[2] - first[0])) * max(
        0.0, float(first[3] - first[1])
    )
    area_second = max(0.0, float(second[2] - second[0])) * max(
        0.0, float(second[3] - second[1])
    )
    union = area_first + area_second - intersection
    return intersection / union if union > 0.0 else 0.0


def _video_signature(video: Path) -> dict[str, int]:
    stat = video.stat()
    return {"bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _configuration_signature(args: argparse.Namespace) -> str:
    values = {
        name: getattr(args, name)
        for name in (
            "model",
            "device",
            "samples",
            "imgsz",
            "person_confidence",
            "keypoint_confidence",
            "min_person_ratio",
            "min_top_ratio",
            "min_left_wrist_ratio",
            "min_right_wrist_ratio",
            "min_left_ankle_ratio",
            "min_right_ankle_ratio",
            "min_full_body_ratio",
            "min_person_height_ratio",
            "min_usable_scale_ratio",
            "max_competing_person_ratio",
            "min_competing_person_scale_ratio",
            "support_posture_mode",
            "support_posture_ratio",
            "support_posture_min_run_seconds",
            "support_posture_thigh_horizontal_cos",
            "support_posture_shin_vertical_cos",
            "support_posture_torso_vertical_cos",
            "support_posture_max_hip_knee_height_ratio",
            "support_posture_min_ankle_below_hip_ratio",
            "support_posture_max_ankle_height_difference_ratio",
            "support_posture_object_model",
            "support_posture_object_confidence",
            "support_posture_object_ratio",
            "support_posture_weak_pose_ratio",
        )
    }
    values["model"] = str(values["model"])
    if values["support_posture_object_model"] is not None:
        values["support_posture_object_model"] = str(
            values["support_posture_object_model"]
        )
    for label, raw_path in (
        ("model", args.model),
        ("support_posture_object_model", args.support_posture_object_model),
    ):
        if raw_path is None:
            continue
        model = Path(raw_path)
        if model.is_file():
            values[f"{label}_bytes"] = model.stat().st_size
            values[f"{label}_mtime_ns"] = model.stat().st_mtime_ns
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _uniform_indices(frame_count: int, samples: int) -> list[int]:
    if frame_count <= 0:
        return []
    count = min(max(1, samples), frame_count)
    return np.linspace(0, frame_count - 1, count, dtype=np.int64).tolist()


def _choose_primary(
    boxes: np.ndarray,
    scores: np.ndarray,
    previous: np.ndarray | None,
) -> int | None:
    if not len(boxes):
        return None
    areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(
        boxes[:, 3] - boxes[:, 1], 0.0
    )
    normalized_area = areas / max(float(areas.max(initial=1.0)), 1.0)
    score = 0.45 * np.asarray(scores, dtype=np.float64) + 0.55 * normalized_area
    if previous is not None:
        continuity = np.asarray([_iou(previous, box) for box in boxes])
        score = 0.45 * score + 0.55 * continuity
    return int(np.argmax(score))


def _keypoint_visible(
    coordinates: np.ndarray,
    confidence: np.ndarray,
    name: str,
    width: int,
    height: int,
    threshold: float,
) -> bool:
    index = KEYPOINT_INDEX[name]
    if index >= len(coordinates) or index >= len(confidence):
        return False
    x, y = coordinates[index]
    return bool(
        confidence[index] >= threshold
        and np.isfinite(x)
        and np.isfinite(y)
        and 0.0 <= x < width
        and 0.0 <= y < height
    )


def _posture_joint_visible(
    coordinates: np.ndarray,
    confidence: np.ndarray,
    name: str,
    width: int,
    height: int,
    threshold: float,
) -> bool:
    index = POSTURE_KEYPOINT_INDEX[name]
    if index >= len(coordinates) or index >= len(confidence):
        return False
    x, y = coordinates[index]
    return bool(
        confidence[index] >= threshold
        and np.isfinite(x)
        and np.isfinite(y)
        and 0.0 <= x < width
        and 0.0 <= y < height
    )


def _midpoint(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(first, dtype=np.float64) + np.asarray(second, dtype=np.float64))


def _axis_cosine(vector: np.ndarray, axis: int) -> float:
    length = float(np.linalg.norm(vector))
    if length <= 1e-6:
        return 0.0
    return float(abs(vector[axis]) / length)


def _seated_like_signature(
    coordinates: np.ndarray,
    confidence: np.ndarray,
    width: int,
    height: int,
    bbox: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Recognize a conservative, sustained chair-like lower-body posture.

    A high-confidence candidate has near-horizontal thighs, near-vertical
    shins, an upright torso, and hips close to the knee height.  A transient
    squat should not pass the later temporal-duration test; the gate is kept
    deliberately conservative because it has no object reconstruction yet.
    """
    visible = {
        name: _posture_joint_visible(
            coordinates,
            confidence,
            name,
            width,
            height,
            float(args.keypoint_confidence),
        )
        for name in POSTURE_KEYPOINT_INDEX
    }
    if not all(visible.values()):
        return None

    points = {
        name: np.asarray(coordinates[index], dtype=np.float64)
        for name, index in POSTURE_KEYPOINT_INDEX.items()
    }
    shoulders = _midpoint(points["left_shoulder"], points["right_shoulder"])
    hips = _midpoint(points["left_hip"], points["right_hip"])
    knees = _midpoint(points["left_knee"], points["right_knee"])
    ankles = _midpoint(points["left_ankle"], points["right_ankle"])
    body_scale = max(
        float(np.linalg.norm(shoulders - ankles)),
        float(max(0.0, bbox[3] - bbox[1])),
        1.0,
    )
    thigh_horizontal_cos = min(
        _axis_cosine(points["left_knee"] - points["left_hip"], axis=0),
        _axis_cosine(points["right_knee"] - points["right_hip"], axis=0),
    )
    shin_vertical_cos = min(
        _axis_cosine(points["left_ankle"] - points["left_knee"], axis=1),
        _axis_cosine(points["right_ankle"] - points["right_knee"], axis=1),
    )
    torso_vertical_cos = _axis_cosine(hips - shoulders, axis=1)
    hip_knee_height_ratio = float(abs(hips[1] - knees[1]) / body_scale)
    # A balanced one-leg pose can look like a chair pose in 2D: its lifted
    # thigh and shin may both align with our horizontal/vertical tests.  A
    # genuine seated or floor-seated posture has both ankle observations below
    # the pelvis in image coordinates.  This is deliberately a weak lower
    # bound, not an assumption that the feet touch the ground.
    left_ankle_below_hip_ratio = float(
        (points["left_ankle"][1] - hips[1]) / body_scale
    )
    right_ankle_below_hip_ratio = float(
        (points["right_ankle"][1] - hips[1]) / body_scale
    )
    ankle_height_difference_ratio = float(
        abs(points["left_ankle"][1] - points["right_ankle"][1]) / body_scale
    )
    matched = bool(
        thigh_horizontal_cos >= float(args.support_posture_thigh_horizontal_cos)
        and shin_vertical_cos >= float(args.support_posture_shin_vertical_cos)
        and torso_vertical_cos >= float(args.support_posture_torso_vertical_cos)
        and hip_knee_height_ratio
        <= float(args.support_posture_max_hip_knee_height_ratio)
        and left_ankle_below_hip_ratio
        >= float(args.support_posture_min_ankle_below_hip_ratio)
        and right_ankle_below_hip_ratio
        >= float(args.support_posture_min_ankle_below_hip_ratio)
        and ankle_height_difference_ratio
        <= float(args.support_posture_max_ankle_height_difference_ratio)
    )
    return {
        "matched": matched,
        "thigh_horizontal_cos": thigh_horizontal_cos,
        "shin_vertical_cos": shin_vertical_cos,
        "torso_vertical_cos": torso_vertical_cos,
        "hip_knee_height_ratio": hip_knee_height_ratio,
        "left_ankle_below_hip_ratio": left_ankle_below_hip_ratio,
        "right_ankle_below_hip_ratio": right_ankle_below_hip_ratio,
        "ankle_height_difference_ratio": ankle_height_difference_ratio,
    }


def _weak_nonfoot_support_signature(signature: dict[str, Any]) -> bool:
    """A permissive 2-D cue used only when furniture is also visible.

    The primary support-posture test is deliberately stronger.  This weaker
    form recovers oblique camera views, where a seated thigh is not horizontal
    in image space, but must never gate a clip on its own.
    """
    return bool(
        signature["thigh_horizontal_cos"] >= 0.30
        and signature["shin_vertical_cos"] >= 0.65
        and signature["torso_vertical_cos"] >= 0.80
        and signature["hip_knee_height_ratio"] <= 0.35
        and signature["left_ankle_below_hip_ratio"] >= 0.10
        and signature["right_ankle_below_hip_ratio"] >= 0.10
    )


def _support_object_near_primary(
    object_box: np.ndarray, primary_box: np.ndarray, hip_y: float
) -> bool:
    """Whether a chair/couch/bed/toilet box is plausibly under the subject.

    Requiring the object top to be at the pelvis height or lower eliminates a
    background chair beside a person who is merely bending to tie a shoe.
    """
    object_box = np.asarray(object_box, dtype=np.float64)
    primary_box = np.asarray(primary_box, dtype=np.float64)
    body_width = max(1.0, float(primary_box[2] - primary_box[0]))
    body_height = max(1.0, float(primary_box[3] - primary_box[1]))
    object_width = max(0.0, float(object_box[2] - object_box[0]))
    object_height = max(0.0, float(object_box[3] - object_box[1]))
    object_area_ratio = object_width * object_height / (body_width * body_height)
    object_center_x = 0.5 * float(object_box[0] + object_box[2])
    return bool(
        0.01 <= object_area_ratio <= 1.5
        and primary_box[0] - 0.25 * body_width
        <= object_center_x
        <= primary_box[2] + 0.25 * body_width
        and object_box[2] >= primary_box[0] - 0.15 * body_width
        and object_box[0] <= primary_box[2] + 0.15 * body_width
        and object_box[1] >= float(hip_y) - 0.02 * body_height
        and object_box[3] >= primary_box[1] + 0.40 * body_height
        and object_box[1] <= primary_box[3]
    )


def _longest_true_run_seconds(
    flags: Iterable[bool], indices: list[int], fps: float
) -> float:
    if fps <= 0.0 or not indices:
        return 0.0
    longest = 0.0
    start: int | None = None
    last: int | None = None
    for flag, index in zip(flags, indices):
        if flag:
            if start is None:
                start = int(index)
            last = int(index)
            continue
        if start is not None and last is not None:
            longest = max(longest, float(last - start) / fps)
        start = None
        last = None
    if start is not None and last is not None:
        longest = max(longest, float(last - start) / fps)
    return longest


def _ratio(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _evaluate_video(
    video: Path,
    detector: Any,
    support_object_detector: Any | None,
    args: argparse.Namespace,
    config_signature: str,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return {
            "schema_version": 1,
            "source": str(video.resolve()),
            "input_signature": _video_signature(video),
            "config_signature": config_signature,
            "status": "error",
            "eligible": False,
            "reasons": ["video_unreadable"],
        }
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    indices = _uniform_indices(frame_count, args.samples)
    previous: np.ndarray | None = None
    counts = {
        "primary_person": 0,
        "usable_scale": 0,
        "top": 0,
        "left_wrist": 0,
        "right_wrist": 0,
        "left_ankle": 0,
        "right_ankle": 0,
        "full_body": 0,
        "competing_person": 0,
        "seated_like": 0,
        "weak_nonfoot_support": 0,
        "near_support_object": 0,
    }
    seated_like_by_index: dict[int, bool] = {}
    decoded = 0
    failed_indices: list[int] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok or frame is None:
            failed_indices.append(int(index))
            continue
        decoded += 1
        height, width = frame.shape[:2]
        result = detector.predict(
            frame,
            conf=float(args.person_confidence),
            imgsz=int(args.imgsz),
            device=None if args.device == "auto" else args.device,
            verbose=False,
        )[0]
        if result.boxes is None or result.keypoints is None or not len(result.boxes):
            continue
        classes = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
        person_indices = np.where(classes == 0)[0]
        if not len(person_indices):
            continue
        all_boxes = result.boxes.xyxy.detach().cpu().numpy()
        all_scores = result.boxes.conf.detach().cpu().numpy()
        keypoint_xy = result.keypoints.xy.detach().cpu().numpy()
        keypoint_conf = result.keypoints.conf.detach().cpu().numpy()
        boxes = all_boxes[person_indices]
        choice = _choose_primary(boxes, all_scores[person_indices], previous)
        if choice is None:
            continue
        selected = int(person_indices[choice])
        bbox = all_boxes[selected]
        previous = bbox.copy()
        counts["primary_person"] += 1
        height_ratio = float(max(0.0, bbox[3] - bbox[1]) / max(height, 1))
        if height_ratio >= float(args.min_person_height_ratio):
            counts["usable_scale"] += 1
        primary_area = max(
            0.0,
            float(bbox[2] - bbox[0]) * float(bbox[3] - bbox[1]),
        )
        for other in person_indices:
            if int(other) == selected:
                continue
            other_bbox = all_boxes[int(other)]
            other_area = max(
                0.0,
                float(other_bbox[2] - other_bbox[0])
                * float(other_bbox[3] - other_bbox[1]),
            )
            # The GVHMR batch path is single-subject.  Ignore tiny background
            # detections, but reject a second person with a meaningful scale.
            if other_area / max(primary_area, 1.0) >= float(
                args.min_competing_person_scale_ratio
            ):
                counts["competing_person"] += 1
                break
        coordinates = keypoint_xy[selected]
        confidence = keypoint_conf[selected]
        posture = _seated_like_signature(
            coordinates,
            confidence,
            width,
            height,
            bbox,
            args,
        )
        if posture is not None and bool(posture["matched"]):
            counts["seated_like"] += 1
            seated_like_by_index[int(index)] = True
        if posture is not None and _weak_nonfoot_support_signature(posture):
            counts["weak_nonfoot_support"] += 1
        if support_object_detector is not None and posture is not None:
            hip_y = 0.5 * float(coordinates[11, 1] + coordinates[12, 1])
            object_result = support_object_detector.predict(
                frame,
                conf=float(args.support_posture_object_confidence),
                imgsz=int(args.imgsz),
                device=None if args.device == "auto" else args.device,
                verbose=False,
            )[0]
            if object_result.boxes is not None and len(object_result.boxes):
                object_classes = (
                    object_result.boxes.cls.detach().cpu().numpy().astype(np.int64)
                )
                object_boxes = object_result.boxes.xyxy.detach().cpu().numpy()
                names = object_result.names
                for object_class, object_box in zip(object_classes, object_boxes):
                    object_name = str(names[int(object_class)]).lower()
                    if object_name not in {"chair", "couch", "bed", "toilet"}:
                        continue
                    if _support_object_near_primary(object_box, bbox, hip_y):
                        counts["near_support_object"] += 1
                        break
        visible = {
            name: _keypoint_visible(
                coordinates,
                confidence,
                name,
                width,
                height,
                float(args.keypoint_confidence),
            )
            for name in KEYPOINT_INDEX
        }
        for name in visible:
            if visible[name]:
                counts["top" if name == "nose" else name] += 1
        if (
            visible["nose"]
            and visible["left_ankle"]
            and visible["right_ankle"]
            and (visible["left_wrist"] or visible["right_wrist"])
            and height_ratio >= float(args.min_person_height_ratio)
        ):
            counts["full_body"] += 1
    capture.release()

    denominator = len(indices)
    ratios = {name: _ratio(value, denominator) for name, value in counts.items()}
    seated_like_ratio = ratios["seated_like"]
    weak_nonfoot_support_ratio = ratios["weak_nonfoot_support"]
    near_support_object_ratio = ratios["near_support_object"]
    seated_like_run_seconds = _longest_true_run_seconds(
        [bool(seated_like_by_index.get(int(index), False)) for index in indices],
        indices,
        fps,
    )
    sustained_pose_nonfoot_support = bool(
        seated_like_ratio >= float(args.support_posture_ratio)
        and seated_like_run_seconds >= float(args.support_posture_min_run_seconds)
    )
    furniture_assisted_nonfoot_support = bool(
        weak_nonfoot_support_ratio >= float(args.support_posture_weak_pose_ratio)
        and near_support_object_ratio >= float(args.support_posture_object_ratio)
    )
    sustained_nonfoot_support = bool(
        sustained_pose_nonfoot_support or furniture_assisted_nonfoot_support
    )
    checks = {
        "primary_person": ratios["primary_person"] >= args.min_person_ratio,
        "usable_scale": ratios["usable_scale"] >= args.min_usable_scale_ratio,
        "top": ratios["top"] >= args.min_top_ratio,
        "left_wrist": ratios["left_wrist"] >= args.min_left_wrist_ratio,
        "right_wrist": ratios["right_wrist"] >= args.min_right_wrist_ratio,
        "left_ankle": ratios["left_ankle"] >= args.min_left_ankle_ratio,
        "right_ankle": ratios["right_ankle"] >= args.min_right_ankle_ratio,
        "full_body": ratios["full_body"] >= args.min_full_body_ratio,
        "competing_person": ratios["competing_person"]
        <= args.max_competing_person_ratio,
    }
    labels = {
        "primary_person": "primary_person_ratio",
        "usable_scale": "usable_person_scale_ratio",
        "top": "top_visible_ratio",
        "left_wrist": "left_wrist_visible_ratio",
        "right_wrist": "right_wrist_visible_ratio",
        "left_ankle": "left_ankle_visible_ratio",
        "right_ankle": "right_ankle_visible_ratio",
        "full_body": "full_body_evidence_ratio",
        "competing_person": "competing_person_ratio",
    }
    thresholds = {
        "primary_person": args.min_person_ratio,
        "usable_scale": args.min_usable_scale_ratio,
        "top": args.min_top_ratio,
        "left_wrist": args.min_left_wrist_ratio,
        "right_wrist": args.min_right_wrist_ratio,
        "left_ankle": args.min_left_ankle_ratio,
        "right_ankle": args.min_right_ankle_ratio,
        "full_body": args.min_full_body_ratio,
        "competing_person": args.max_competing_person_ratio,
    }
    reasons = [
        (
            f"{labels[name]}={ratios[name]:.3f}>{thresholds[name]:.3f}"
            if name == "competing_person"
            else f"{labels[name]}={ratios[name]:.3f}<{thresholds[name]:.3f}"
        )
        for name, passed in checks.items() if not passed
    ]
    if args.support_posture_mode == "gate" and sustained_nonfoot_support:
        if sustained_pose_nonfoot_support:
            reasons.append(
                "sustained_nonfoot_support_posture="
                f"ratio={seated_like_ratio:.3f}>={args.support_posture_ratio:.3f},"
                f"run={seated_like_run_seconds:.2f}s"
                f">={args.support_posture_min_run_seconds:.2f}s"
            )
        else:
            reasons.append(
                "furniture_assisted_nonfoot_support_posture="
                f"weak_pose_ratio={weak_nonfoot_support_ratio:.3f}"
                f">={args.support_posture_weak_pose_ratio:.3f},"
                f"near_support_object_ratio={near_support_object_ratio:.3f}"
                f">={args.support_posture_object_ratio:.3f}"
            )
    if decoded < max(1, int(0.8 * denominator)):
        reasons.insert(0, f"decoded_sample_ratio={_ratio(decoded, denominator):.3f}<0.800")
    eligible = not reasons
    return {
        "schema_version": 1,
        "source": str(video.resolve()),
        "input_signature": _video_signature(video),
        "config_signature": config_signature,
        "status": "eligible" if eligible else "excluded",
        "eligible": eligible,
        "video": {"frames": frame_count, "fps": fps},
        "sampled_frames": denominator,
        "decoded_frames": decoded,
        "failed_frame_indices": failed_indices,
        "ratios": ratios,
        "thresholds": {
            **{labels[name]: thresholds[name] for name in thresholds},
            "min_person_height_ratio": args.min_person_height_ratio,
            "min_competing_person_scale_ratio": (
                args.min_competing_person_scale_ratio
            ),
        },
        "diagnostics": {
            "support_posture": {
                "mode": args.support_posture_mode,
                "seated_like_ratio": seated_like_ratio,
                "longest_seated_like_run_seconds": seated_like_run_seconds,
                "sustained_nonfoot_support_posture": sustained_nonfoot_support,
                "sustained_pose_nonfoot_support": sustained_pose_nonfoot_support,
                "furniture_assisted_nonfoot_support": (
                    furniture_assisted_nonfoot_support
                ),
                "weak_nonfoot_support_ratio": weak_nonfoot_support_ratio,
                "near_support_object_ratio": near_support_object_ratio,
                "ratio_threshold": float(args.support_posture_ratio),
                "run_seconds_threshold": float(
                    args.support_posture_min_run_seconds
                ),
            }
        },
        "reasons": reasons,
    }


def _read_videos(args: argparse.Namespace) -> list[Path]:
    paths = [Path(value) for value in args.video]
    if args.video_list is not None:
        paths.extend(
            Path(line.strip())
            for line in args.video_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    unique: dict[str, Path] = {}
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        unique[str(resolved)] = resolved
    if not unique:
        raise ValueError("at least one --video or --video-list entry is required")
    return [unique[key] for key in sorted(unique)]


def _write_csv(path: Path, records: Iterable[dict[str, Any]]) -> None:
    fields = [
        "clip",
        "status",
        "eligible",
        "sampled_frames",
        "primary_person_ratio",
        "usable_person_scale_ratio",
        "top_visible_ratio",
        "left_wrist_visible_ratio",
        "right_wrist_visible_ratio",
        "left_ankle_visible_ratio",
        "right_ankle_visible_ratio",
        "full_body_evidence_ratio",
        "competing_person_ratio",
        "seated_like_posture_ratio",
        "longest_seated_like_run_seconds",
        "sustained_nonfoot_support_posture",
        "reasons",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for record in records:
                ratios = record.get("ratios", {})
                support = record.get("diagnostics", {}).get(
                    "support_posture", {}
                )
                writer.writerow(
                    {
                        "clip": Path(str(record.get("source", ""))).stem,
                        "status": record.get("status", "error"),
                        "eligible": record.get("eligible", False),
                        "sampled_frames": record.get("sampled_frames", 0),
                        "primary_person_ratio": ratios.get("primary_person", ""),
                        "usable_person_scale_ratio": ratios.get("usable_scale", ""),
                        "top_visible_ratio": ratios.get("top", ""),
                        "left_wrist_visible_ratio": ratios.get("left_wrist", ""),
                        "right_wrist_visible_ratio": ratios.get("right_wrist", ""),
                        "left_ankle_visible_ratio": ratios.get("left_ankle", ""),
                        "right_ankle_visible_ratio": ratios.get("right_ankle", ""),
                        "full_body_evidence_ratio": ratios.get("full_body", ""),
                        "competing_person_ratio": ratios.get("competing_person", ""),
                        "seated_like_posture_ratio": ratios.get("seated_like", ""),
                        "longest_seated_like_run_seconds": support.get(
                            "longest_seated_like_run_seconds", ""
                        ),
                        "sustained_nonfoot_support_posture": support.get(
                            "sustained_nonfoot_support_posture", ""
                        ),
                        "reasons": " | ".join(record.get("reasons", [])),
                    }
                )
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Admit only videos with temporal YOLO-Pose full-body evidence."
    )
    parser.add_argument("--video", action="append", default=[])
    parser.add_argument("--video-list", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--person-confidence", type=float, default=0.25)
    parser.add_argument("--keypoint-confidence", type=float, default=0.35)
    parser.add_argument("--min-person-ratio", type=float, default=0.80)
    parser.add_argument("--min-top-ratio", type=float, default=0.60)
    parser.add_argument("--min-left-wrist-ratio", type=float, default=0.30)
    parser.add_argument("--min-right-wrist-ratio", type=float, default=0.30)
    parser.add_argument("--min-left-ankle-ratio", type=float, default=0.55)
    parser.add_argument("--min-right-ankle-ratio", type=float, default=0.55)
    parser.add_argument("--min-full-body-ratio", type=float, default=0.35)
    parser.add_argument("--min-person-height-ratio", type=float, default=0.40)
    parser.add_argument("--min-usable-scale-ratio", type=float, default=0.60)
    parser.add_argument(
        "--max-competing-person-ratio",
        type=float,
        default=0.0,
        help="Maximum sampled-frame ratio with a meaningful second person.",
    )
    parser.add_argument(
        "--min-competing-person-scale-ratio",
        type=float,
        default=0.15,
        help="Ignore a second person smaller than this fraction of the primary box.",
    )
    parser.add_argument(
        "--support-posture-mode",
        choices=["off", "report", "gate"],
        default="off",
        help=(
            "Detect sustained seated-like external-support posture before "
            "GVHMR; report keeps the clip while gate excludes it."
        ),
    )
    parser.add_argument(
        "--support-posture-ratio",
        type=float,
        default=0.30,
        help="Minimum sampled-frame ratio for a sustained support-posture flag.",
    )
    parser.add_argument(
        "--support-posture-min-run-seconds",
        type=float,
        default=1.0,
        help="Minimum temporal duration of a seated-like posture run.",
    )
    parser.add_argument(
        "--support-posture-thigh-horizontal-cos",
        type=float,
        default=0.75,
        help="Minimum absolute horizontal cosine for both thighs.",
    )
    parser.add_argument(
        "--support-posture-shin-vertical-cos",
        type=float,
        default=0.80,
        help="Minimum absolute vertical cosine for both shins.",
    )
    parser.add_argument(
        "--support-posture-torso-vertical-cos",
        type=float,
        default=0.80,
        help="Minimum absolute vertical cosine for the shoulder-to-hip axis.",
    )
    parser.add_argument(
        "--support-posture-max-hip-knee-height-ratio",
        type=float,
        default=0.18,
        help="Maximum normalized hip-to-knee vertical separation.",
    )
    parser.add_argument(
        "--support-posture-min-ankle-below-hip-ratio",
        type=float,
        default=0.15,
        help=(
            "Minimum normalized vertical distance of each ankle below the hip; "
            "rejects one-leg balance poses that resemble seated geometry in 2D."
        ),
    )
    parser.add_argument(
        "--support-posture-max-ankle-height-difference-ratio",
        type=float,
        default=0.18,
        help=(
            "Maximum normalized difference in left/right ankle height; "
            "rejects a lifted-leg balance pose."
        ),
    )
    parser.add_argument(
        "--support-posture-object-model",
        type=Path,
        help=(
            "COCO object detector used only with support-posture report/gate; "
            "its chair/couch/bed/toilet evidence is combined with pose."
        ),
    )
    parser.add_argument(
        "--support-posture-object-confidence",
        type=float,
        default=0.05,
        help="Permissive object confidence; geometry checks suppress distant boxes.",
    )
    parser.add_argument(
        "--support-posture-object-ratio",
        type=float,
        default=0.06,
        help="Minimum sampled-frame ratio of near-person furniture evidence.",
    )
    parser.add_argument(
        "--support-posture-weak-pose-ratio",
        type=float,
        default=0.05,
        help="Minimum weak seated-pose ratio when furniture evidence is present.",
    )
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(
            f"YOLO-Pose model is missing: {args.model}. Download the configured "
            "local checkpoint before running the gate."
        )
    for name, value in vars(args).items():
        if (
            name.startswith("min_")
            or name.startswith("max_")
            or name.endswith("confidence")
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
    if args.samples <= 0 or args.imgsz <= 0:
        raise ValueError("samples and imgsz must be positive")
    posture_values = {
        "support_posture_ratio": args.support_posture_ratio,
        "support_posture_thigh_horizontal_cos": (
            args.support_posture_thigh_horizontal_cos
        ),
        "support_posture_shin_vertical_cos": args.support_posture_shin_vertical_cos,
        "support_posture_torso_vertical_cos": args.support_posture_torso_vertical_cos,
        "support_posture_max_hip_knee_height_ratio": (
            args.support_posture_max_hip_knee_height_ratio
        ),
        "support_posture_min_ankle_below_hip_ratio": (
            args.support_posture_min_ankle_below_hip_ratio
        ),
        "support_posture_max_ankle_height_difference_ratio": (
            args.support_posture_max_ankle_height_difference_ratio
        ),
        "support_posture_object_confidence": args.support_posture_object_confidence,
        "support_posture_object_ratio": args.support_posture_object_ratio,
        "support_posture_weak_pose_ratio": args.support_posture_weak_pose_ratio,
    }
    for name, value in posture_values.items():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if args.support_posture_min_run_seconds <= 0.0:
        raise ValueError("support_posture_min_run_seconds must be positive")
    if args.support_posture_mode != "off":
        if args.support_posture_object_model is None:
            raise ValueError(
                "support_posture_object_model is required when support_posture_mode "
                "is report or gate"
            )
        if not args.support_posture_object_model.is_file():
            raise FileNotFoundError(
                "support-posture object model is missing: "
                f"{args.support_posture_object_model}"
            )
    videos = _read_videos(args)
    signature = _configuration_signature(args)
    records = _jsonl_records(args.report)
    pending = []
    for video in videos:
        key = str(video.resolve())
        existing = records.get(key)
        if (
            not args.no_cache
            and isinstance(existing, dict)
            and existing.get("input_signature") == _video_signature(video)
            and existing.get("config_signature") == signature
        ):
            print(f"[FULLBODY] cached {video.name}: {existing.get('status')}", flush=True)
        else:
            pending.append(video)
    if pending:
        from ultralytics import YOLO

        detector = YOLO(str(args.model))
        support_object_detector = (
            YOLO(str(args.support_posture_object_model))
            if args.support_posture_mode != "off"
            else None
        )
        for video in pending:
            try:
                record = _evaluate_video(
                    video, detector, support_object_detector, args, signature
                )
            except Exception as exc:
                record = {
                    "schema_version": 1,
                    "source": str(video.resolve()),
                    "input_signature": _video_signature(video),
                    "config_signature": signature,
                    "status": "error",
                    "eligible": False,
                    "reasons": [f"preflight_error={type(exc).__name__}: {exc}"],
                }
            records[str(video.resolve())] = record
            support = record.get("diagnostics", {}).get("support_posture", {})
            posture_note = ""
            if support.get("sustained_nonfoot_support_posture", False):
                posture_note = (
                    " support_posture="
                    f"ratio={float(support.get('seated_like_ratio', 0.0)):.3f},"
                    f"run={float(support.get('longest_seated_like_run_seconds', 0.0)):.2f}s"
                )
            print(
                f"[FULLBODY] {video.name}: {record['status']} "
                f"reasons={' | '.join(record.get('reasons', [])) or '<none>'}"
                f"{posture_note}",
                flush=True,
            )
    ordered = [records[key] for key in sorted(records)]
    _atomic_text(
        args.report,
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in ordered),
    )
    _write_csv(args.csv, ordered)
    selected = [records[str(video.resolve())] for video in videos]
    eligible_count = sum(bool(record.get("eligible")) for record in selected)
    print(
        f"[FULLBODY] complete: {eligible_count}/{len(selected)} eligible; "
        f"report={args.csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()
