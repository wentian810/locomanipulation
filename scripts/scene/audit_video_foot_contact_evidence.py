#!/usr/bin/env python3
"""Audit 2-D foot support evidence against a GMR motion without changing it.

Foot keypoints are measured relative to a robust background affine flow rather
than raw image coordinates.  This makes a planted foot distinguishable from a
foot merely following camera motion.  The report deliberately separates
visual-planted GMR skating (not safe to relabel) from visual-moving GMR skating
(a possible swing-clearance candidate).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np
import torch


FOOT_KEYPOINTS = {"left": (17, 18, 19), "right": (20, 21, 22)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runs(mask: np.ndarray, minimum: int) -> list[list[int]]:
    result: list[list[int]] = []
    for index, enabled in enumerate(mask.tolist() + [False]):
        if enabled and (not result or result[-1][-1] != index - 1):
            result.append([index])
        elif enabled:
            result[-1].append(index)
    return [run for run in result if len(run) >= minimum]


def _segments(mask: np.ndarray, minimum: int) -> list[dict[str, int]]:
    return [
        {"frame_range": [run[0], run[-1]], "frame_count": len(run)}
        for run in _runs(mask, minimum)
    ]


def _load_keypoints(path: Path) -> np.ndarray:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise ValueError("vitpose_wholebody.pt is not a tensor")
    array = value.detach().cpu().numpy()
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[1:] != (133, 3):
        raise ValueError(f"unexpected VitPose shape: {array.shape}")
    return np.asarray(array, dtype=np.float64)


def _load_boxes(path: Path, frames: int) -> np.ndarray:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict) or "bbx_xyxy" not in value:
        raise ValueError("bbx.pt is missing bbx_xyxy")
    array = np.asarray(value["bbx_xyxy"], dtype=np.float64)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.shape != (frames, 4):
        raise ValueError(f"bbox shape {array.shape} does not match {frames} frames")
    return array


def _read_frames(path: Path, expected: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    capture.release()
    if len(frames) != expected:
        raise ValueError(f"video has {len(frames)} frames but VitPose has {expected}")
    return frames


def _background_mask(shape: tuple[int, int], bbox: np.ndarray, margin: int) -> np.ndarray:
    height, width = shape
    mask = np.full((height, width), 255, dtype=np.uint8)
    x1, y1, x2, y2 = np.round(bbox).astype(int)
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(width, x2 + margin)
    y2 = min(height, y2 + margin)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 0
    return mask


def _estimate_background_affines(
    frames: list[np.ndarray], boxes: np.ndarray, max_corners: int, margin: int
) -> tuple[np.ndarray, np.ndarray]:
    transforms = np.repeat(np.eye(3, dtype=np.float64)[None], len(frames), axis=0)
    inliers = np.zeros(len(frames), dtype=np.int32)
    for index in range(1, len(frames)):
        previous = frames[index - 1]
        current = frames[index]
        mask = _background_mask(previous.shape, boxes[index - 1], margin)
        points = cv2.goodFeaturesToTrack(
            previous,
            maxCorners=max_corners,
            qualityLevel=0.01,
            minDistance=7,
            mask=mask,
            blockSize=7,
        )
        if points is None or len(points) < 12:
            continue
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            previous,
            current,
            points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if next_points is None or status is None:
            continue
        valid = status.reshape(-1).astype(bool)
        source = points.reshape(-1, 2)[valid]
        target = next_points.reshape(-1, 2)[valid]
        if len(source) < 12:
            continue
        affine, keep = cv2.estimateAffinePartial2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.5,
            maxIters=2000,
            confidence=0.995,
            refineIters=10,
        )
        if affine is None or keep is None or int(keep.sum()) < 8:
            continue
        transforms[index, :2] = affine
        inliers[index] = int(keep.sum())
    return transforms, inliers


def _median_nan_filter(values: np.ndarray, size: int) -> np.ndarray:
    result = values.copy()
    radius = size // 2
    for index in range(len(values)):
        part = values[max(0, index - radius) : min(len(values), index + radius + 1)]
        finite = part[np.isfinite(part)]
        if len(finite):
            result[index] = float(np.median(finite))
    return result


def _visual_foot_motion(
    keypoints: np.ndarray,
    transforms: np.ndarray,
    min_confidence: float,
    plant_residual_px: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    positions: dict[str, np.ndarray] = {}
    residuals: dict[str, np.ndarray] = {}
    planted: dict[str, np.ndarray] = {}
    for side, ids in FOOT_KEYPOINTS.items():
        subset = keypoints[:, ids]
        confidence = subset[:, :, 2]
        valid_points = confidence >= min_confidence
        position = np.full((len(keypoints), 2), np.nan, dtype=np.float64)
        for frame in range(len(keypoints)):
            active = valid_points[frame]
            if np.any(active):
                weights = confidence[frame, active]
                position[frame] = np.average(subset[frame, active, :2], axis=0, weights=weights)
        residual = np.full(len(keypoints), np.nan, dtype=np.float64)
        for frame in range(1, len(keypoints)):
            if not np.all(np.isfinite(position[frame - 1])) or not np.all(np.isfinite(position[frame])):
                continue
            point = np.array([position[frame - 1, 0], position[frame - 1, 1], 1.0])
            expected = transforms[frame] @ point
            residual[frame] = float(np.linalg.norm(position[frame] - expected[:2]))
        filtered = _median_nan_filter(residual, 3)
        valid = np.isfinite(filtered) & (np.max(confidence, axis=1) >= min_confidence)
        positions[side] = position
        residuals[side] = filtered
        planted[side] = valid & (filtered <= plant_residual_px)
    return positions, residuals, planted


def _foot_meshes(model: mujoco.MjModel, body_name: str) -> tuple[int, list[tuple[int, np.ndarray]]]:
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body < 0:
        raise ValueError(f"missing body {body_name}")
    meshes: list[tuple[int, np.ndarray]] = []
    for geom in range(model.ngeom):
        if int(model.geom_bodyid[geom]) != body or int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        if int(model.geom_group[geom]) != 1:
            continue
        mesh = int(model.geom_dataid[geom])
        start = int(model.mesh_vertadr[mesh])
        count = int(model.mesh_vertnum[mesh])
        meshes.append((geom, model.mesh_vert[start : start + count].copy()))
    if not meshes:
        raise ValueError(f"no visible foot mesh for {body_name}")
    return body, meshes


def _gmr_states(
    motion_path: Path, robot_xml: Path
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    with motion_path.open("rb") as stream:
        motion = pickle.load(stream)
    root = np.asarray(motion["root_pos"], dtype=np.float64)
    rotation = np.asarray(motion["root_rot"], dtype=np.float64)
    dof = np.asarray(motion["dof_pos"], dtype=np.float64)
    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    if model.nq - 7 != dof.shape[1]:
        raise ValueError("robot motion and XML DoF counts differ")
    feet = {
        "left": _foot_meshes(model, "left_ankle_roll_link"),
        "right": _foot_meshes(model, "right_ankle_roll_link"),
    }
    xy = np.zeros((len(root), 2, 2), dtype=np.float64)
    height = np.zeros((len(root), 2), dtype=np.float64)
    data = mujoco.MjData(model)
    for frame in range(len(root)):
        data.qpos[:3] = root[frame]
        data.qpos[3:7] = rotation[frame][[3, 0, 1, 2]]
        data.qpos[7:] = dof[frame]
        mujoco.mj_forward(model, data)
        for side_index, side in enumerate(("left", "right")):
            body, meshes = feet[side]
            xy[frame, side_index] = data.xpos[body, :2]
            low = float("inf")
            for geom, vertices in meshes:
                matrix = data.geom_xmat[geom].reshape(3, 3)
                low = min(low, float(np.min(vertices @ matrix.T + data.geom_xpos[geom], axis=0)[2]))
            height[frame, side_index] = low
    fps = float(motion.get("fps", 30.0))
    speed = np.zeros((len(root), 2), dtype=np.float64)
    speed[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=2) * fps
    speed[:-1] = np.maximum(speed[:-1], speed[1:])
    return motion, height, speed, xy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--vitpose", required=True, type=Path)
    parser.add_argument("--bbox", required=True, type=Path)
    parser.add_argument("--robot-motion", required=True, type=Path)
    parser.add_argument("--robot-xml", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--min-confidence", type=float, default=0.45)
    parser.add_argument("--plant-residual-px", type=float, default=3.0)
    parser.add_argument("--minimum-run-frames", type=int, default=3)
    parser.add_argument("--background-margin-px", type=int, default=40)
    parser.add_argument("--max-background-corners", type=int, default=500)
    parser.add_argument("--gmr-speed-tolerance-m-s", type=float, default=0.25)
    parser.add_argument("--ground-contact-height-m", type=float, default=0.015)
    args = parser.parse_args()
    if args.plant_residual_px <= 0.0 or args.gmr_speed_tolerance_m_s <= 0.0:
        raise ValueError("motion tolerances must be positive")
    keypoints = _load_keypoints(args.vitpose)
    boxes = _load_boxes(args.bbox, len(keypoints))
    frames = _read_frames(args.video, len(keypoints))
    transforms, inliers = _estimate_background_affines(
        frames, boxes, args.max_background_corners, args.background_margin_px
    )
    positions, residuals, planted = _visual_foot_motion(
        keypoints, transforms, args.min_confidence, args.plant_residual_px
    )
    motion, gmr_height, gmr_speed, _ = _gmr_states(args.robot_motion, args.robot_xml)
    if len(gmr_height) != len(keypoints):
        raise ValueError("GMR and visual evidence frame counts differ")
    labels = {
        "left": np.asarray(motion.get("support_left_contact"), dtype=bool),
        "right": np.asarray(motion.get("support_right_contact"), dtype=bool),
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "background_compensated_2d_foot_contact_evidence",
        "weights_modified": False,
        "inputs": {
            "video": str(args.video),
            "vitpose": str(args.vitpose),
            "bbox": str(args.bbox),
            "robot_motion": str(args.robot_motion),
            "robot_motion_sha256": _sha256(args.robot_motion),
        },
        "parameters": {
            "plant_residual_px": args.plant_residual_px,
            "min_confidence": args.min_confidence,
            "gmr_speed_tolerance_m_s": args.gmr_speed_tolerance_m_s,
            "ground_contact_height_m": args.ground_contact_height_m,
        },
        "frame_count": int(len(keypoints)),
        "background_affine": {
            "median_inliers": float(np.median(inliers[1:])),
            "p05_inliers": float(np.percentile(inliers[1:], 5)),
            "failed_pair_count": int(np.count_nonzero(inliers[1:] < 8)),
        },
        "feet": {},
    }
    phase = {}
    for side_index, side in enumerate(("left", "right")):
        residual = residuals[side]
        visual_valid = np.isfinite(residual)
        near = (gmr_height[:, side_index] >= -0.005) & (gmr_height[:, side_index] <= args.ground_contact_height_m)
        sliding = near & (gmr_speed[:, side_index] > args.gmr_speed_tolerance_m_s)
        stable = labels[side] & near & ~sliding
        visual_planted = planted[side]
        visual_moving = visual_valid & ~visual_planted
        phase[side] = np.where(visual_planted, 1, np.where(visual_moving, 2, 0)).astype(np.int8)
        report["feet"][side] = {
            "visual_valid_frame_count": int(np.count_nonzero(visual_valid)),
            "visual_planted_frame_count": int(np.count_nonzero(visual_planted)),
            "visual_planted_segments": _segments(visual_planted, args.minimum_run_frames),
            "background_relative_residual_px": {
                "median": float(np.nanmedian(residual)),
                "p95": float(np.nanpercentile(residual, 95)),
            },
            "gmr_near_ground_sliding_frame_count": int(np.count_nonzero(sliding)),
            "gmr_stable_support_frame_count": int(np.count_nonzero(stable)),
            "gmr_sliding_visual_planted_frame_count": int(np.count_nonzero(sliding & visual_planted)),
            "gmr_sliding_visual_moving_frame_count": int(np.count_nonzero(sliding & visual_moving)),
            "gmr_sliding_visual_unknown_frame_count": int(np.count_nonzero(sliding & ~visual_valid)),
            "gmr_stable_visual_planted_frame_count": int(np.count_nonzero(stable & visual_planted)),
            "safe_swing_clearance_candidate_frames": int(np.count_nonzero(sliding & visual_moving)),
            "root_or_leg_contact_lock_required_frames": int(np.count_nonzero(sliding & visual_planted)),
        }
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.evidence,
        background_affine=transforms.astype(np.float32),
        background_inliers=inliers,
        left_foot_xy=positions["left"].astype(np.float32),
        right_foot_xy=positions["right"].astype(np.float32),
        left_residual_px=residuals["left"].astype(np.float32),
        right_residual_px=residuals["right"].astype(np.float32),
        left_phase=phase["left"],
        right_phase=phase["right"],
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "frame_count": report["frame_count"],
        "background_affine": report["background_affine"],
        "feet": report["feet"],
    }))


if __name__ == "__main__":
    main()
