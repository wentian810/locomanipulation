"""Conservative offline hand-crop tracking for full-body videos.

Detector boxes live in original-video coordinates. Optical flow supplies a
short-horizon motion proposal and an 8D Kalman filter reduces detector jitter
without inventing a hand after a long occlusion. The module has no model
dependency so it can be unit-tested independently of Hand4Whole++.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


TRACK_SOURCE_NONE = np.int8(0)
TRACK_SOURCE_OBSERVATION = np.int8(1)
TRACK_SOURCE_FLOW = np.int8(2)
TRACK_SOURCE_PREDICTION = np.int8(3)


@dataclass(frozen=True)
class TrackerConfig:
    """Conservative tracker parameters."""

    max_gap: int = 8
    max_prediction_gap: int = 2
    min_box_size: float = 8.0
    min_flow_points: int = 5
    observation_iou_gate: float = 0.01
    observation_center_gate: float = 1.75
    direct_observation_quality: float = 0.75


def _valid_box(box: np.ndarray, min_box_size: float) -> bool:
    box = np.asarray(box, dtype=np.float32).reshape(4)
    return bool(
        np.isfinite(box).all()
        and (box[2] - box[0]) >= float(min_box_size)
        and (box[3] - box[1]) >= float(min_box_size)
    )


def _clip_box(
    box: np.ndarray, width: int, height: int, min_box_size: float
) -> np.ndarray | None:
    box = np.asarray(box, dtype=np.float32).reshape(4).copy()
    if not np.isfinite(box).all():
        return None
    box[0::2] = np.clip(box[0::2], 0.0, max(float(width - 1), 0.0))
    box[1::2] = np.clip(box[1::2], 0.0, max(float(height - 1), 0.0))
    return box if _valid_box(box, min_box_size) else None


def _measurement_from_box(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32).reshape(4)
    width = max(float(x2 - x1), 1e-3)
    height = max(float(y2 - y1), 1e-3)
    return np.asarray(
        [(x1 + x2) * 0.5, (y1 + y2) * 0.5, np.log(width), np.log(height)],
        dtype=np.float64,
    )


def _box_from_measurement(measurement: np.ndarray) -> np.ndarray:
    cx, cy, log_width, log_height = np.asarray(
        measurement, dtype=np.float64
    ).reshape(4)
    width = float(np.exp(np.clip(log_width, -4.0, 12.0)))
    height = float(np.exp(np.clip(log_height, -4.0, 12.0)))
    return np.asarray(
        [cx - width * 0.5, cy - height * 0.5, cx + width * 0.5, cy + height * 0.5],
        dtype=np.float32,
    )


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(4)
    b = np.asarray(b, dtype=np.float32).reshape(4)
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = max(
        1e-6,
        (a[2] - a[0]) * (a[3] - a[1])
        + (b[2] - b[0]) * (b[3] - b[1])
        - intersection,
    )
    return float(intersection / union)


def _center_distance_ratio(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(4)
    b = np.asarray(b, dtype=np.float32).reshape(4)
    center_a = (a[:2] + a[2:]) * 0.5
    center_b = (b[:2] + b[2:]) * 0.5
    diagonal = max(1e-3, float(np.linalg.norm(b[2:] - b[:2])))
    return float(np.linalg.norm(center_a - center_b) / diagonal)


def _kalman_predict(
    state: np.ndarray, covariance: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    transition = np.eye(8, dtype=np.float64)
    transition[:4, 4:] = np.eye(4, dtype=np.float64)
    process_noise = np.diag(
        np.asarray(
            [4.0**2, 4.0**2, 0.04**2, 0.04**2, 2.0**2, 2.0**2, 0.02**2, 0.02**2]
        )
    )
    return transition @ state, transition @ covariance @ transition.T + process_noise


def _kalman_update(
    state: np.ndarray,
    covariance: np.ndarray,
    measurement: np.ndarray,
    *,
    is_flow: bool,
) -> tuple[np.ndarray, np.ndarray]:
    observation = np.zeros((4, 8), dtype=np.float64)
    observation[:, :4] = np.eye(4, dtype=np.float64)
    noise = np.asarray(
        [16.0**2, 16.0**2, 0.18**2, 0.18**2]
        if is_flow
        else [8.0**2, 8.0**2, 0.10**2, 0.10**2],
        dtype=np.float64,
    )
    innovation = measurement - observation @ state
    innovation_covariance = observation @ covariance @ observation.T + np.diag(noise)
    gain = covariance @ observation.T @ np.linalg.pinv(innovation_covariance)
    state = state + gain @ innovation
    covariance = (np.eye(8, dtype=np.float64) - gain @ observation) @ covariance
    return state, covariance


def _initial_state(measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    state = np.zeros(8, dtype=np.float64)
    state[:4] = measurement
    covariance = np.diag(
        np.asarray(
            [20.0**2, 20.0**2, 0.35**2, 0.35**2, 8.0**2, 8.0**2, 0.08**2, 0.08**2]
        )
    )
    return state, covariance


def _flow_points(gray: np.ndarray, box: np.ndarray) -> np.ndarray:
    height, width = gray.shape[:2]
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32).reshape(4)
    x1_i = int(np.clip(np.floor(x1), 0, max(width - 1, 0)))
    y1_i = int(np.clip(np.floor(y1), 0, max(height - 1, 0)))
    x2_i = int(np.clip(np.ceil(x2), 0, max(width - 1, 0)))
    y2_i = int(np.clip(np.ceil(y2), 0, max(height - 1, 0)))
    mask = np.zeros_like(gray, dtype=np.uint8)
    if x2_i > x1_i and y2_i > y1_i:
        mask[y1_i : y2_i + 1, x1_i : x2_i + 1] = 255
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=48,
        qualityLevel=0.01,
        minDistance=3,
        mask=mask,
        blockSize=5,
    )
    xs = np.linspace(x1 + 0.15 * (x2 - x1), x2 - 0.15 * (x2 - x1), num=4)
    ys = np.linspace(y1 + 0.15 * (y2 - y1), y2 - 0.15 * (y2 - y1), num=4)
    grid = np.asarray([[x, y] for y in ys for x in xs], dtype=np.float32).reshape(
        -1, 1, 2
    )
    return grid if corners is None else np.concatenate((corners.astype(np.float32), grid), axis=0)


def _flow_propagate(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    previous_box: np.ndarray,
    *,
    min_points: int,
) -> tuple[np.ndarray | None, int]:
    points = _flow_points(previous_gray, previous_box)
    if len(points) == 0:
        return None, 0
    next_points, forward_status, forward_error = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if next_points is None or forward_status is None:
        return None, 0
    back_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current_gray,
        previous_gray,
        next_points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if back_points is None or backward_status is None:
        return None, 0
    forward_ok = forward_status.reshape(-1).astype(bool)
    backward_ok = backward_status.reshape(-1).astype(bool)
    roundtrip = np.linalg.norm(
        points.reshape(-1, 2) - back_points.reshape(-1, 2), axis=1
    )
    flow_error = (
        forward_error.reshape(-1)
        if forward_error is not None
        else np.zeros(len(points), dtype=np.float32)
    )
    keep = (
        forward_ok
        & backward_ok
        & np.isfinite(roundtrip)
        & (roundtrip <= 1.5)
        & np.isfinite(flow_error)
        & (flow_error <= 35.0)
    )
    if int(keep.sum()) < int(min_points):
        return None, int(keep.sum())
    source = points.reshape(-1, 2)[keep]
    target = next_points.reshape(-1, 2)[keep]
    displacement = np.median(target - source, axis=0)
    center = (np.asarray(previous_box[:2]) + np.asarray(previous_box[2:])) * 0.5
    source_radius = np.linalg.norm(source - center[None], axis=1)
    target_center = center + displacement
    target_radius = np.linalg.norm(target - target_center[None], axis=1)
    usable_radius = source_radius > 4.0
    scale = 1.0
    if int(usable_radius.sum()) >= 3:
        scale = float(np.median(target_radius[usable_radius] / source_radius[usable_radius]))
        scale = float(np.clip(scale, 0.80, 1.25))
    width_height = (np.asarray(previous_box[2:]) - np.asarray(previous_box[:2])) * scale
    propagated_center = center + displacement
    propagated = np.concatenate(
        (propagated_center - width_height * 0.5, propagated_center + width_height * 0.5)
    )
    return propagated.astype(np.float32), int(keep.sum())


def track_bbox_sequence(
    frames: Any,
    observed_xyxy: np.ndarray,
    observed_valid: np.ndarray,
    config: TrackerConfig = TrackerConfig(),
    observed_quality: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Track one hand in original-video coordinates.

    observed_valid must reflect the detector existence decision, rather than
    only whether a fallback bbox is finite.
    """

    observed_xyxy = np.asarray(observed_xyxy, dtype=np.float32)
    observed_valid = np.asarray(observed_valid, dtype=bool).reshape(-1)
    if observed_xyxy.ndim != 2 or observed_xyxy.shape[1] != 4:
        raise ValueError(f"observed_xyxy must have shape (F,4), got {observed_xyxy.shape}")
    if observed_xyxy.shape[0] != observed_valid.shape[0]:
        raise ValueError("observed_xyxy and observed_valid disagree on frame count")
    if len(frames) < observed_xyxy.shape[0]:
        raise ValueError("frame sequence is shorter than observed bbox track")
    if observed_quality is None:
        observed_quality = np.zeros(observed_xyxy.shape[0], dtype=np.float32)
    else:
        observed_quality = np.asarray(observed_quality, dtype=np.float32).reshape(-1)
        if observed_quality.shape[0] != observed_xyxy.shape[0]:
            raise ValueError("observed_quality and observed_xyxy disagree on frame count")

    frame_count = observed_xyxy.shape[0]
    tracked_xyxy = np.zeros((frame_count, 4), dtype=np.float32)
    usable = np.zeros(frame_count, dtype=bool)
    source = np.full(frame_count, TRACK_SOURCE_NONE, dtype=np.int8)
    flow_points = np.zeros(frame_count, dtype=np.int16)
    rejected_observation = np.zeros(frame_count, dtype=bool)
    direct_observation = np.zeros(frame_count, dtype=bool)

    state: np.ndarray | None = None
    covariance: np.ndarray | None = None
    previous_gray: np.ndarray | None = None
    previous_box: np.ndarray | None = None
    missed = 0
    prediction_only = 0

    for frame_idx in range(frame_count):
        frame = np.asarray(frames[frame_idx])
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError(f"expected RGB frame at {frame_idx}, got {frame.shape}")
        height, width = frame.shape[:2]
        current_gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2GRAY)
        observation = _clip_box(
            observed_xyxy[frame_idx], width, height, config.min_box_size
        )
        observation_ok = bool(observed_valid[frame_idx] and observation is not None)
        high_quality_observation = bool(
            observation_ok
            and np.isfinite(observed_quality[frame_idx])
            and observed_quality[frame_idx] >= config.direct_observation_quality
        )

        flow_box = None
        if previous_gray is not None and previous_box is not None:
            flow_box, flow_points[frame_idx] = _flow_propagate(
                previous_gray,
                current_gray,
                previous_box,
                min_points=config.min_flow_points,
            )
            if flow_box is not None:
                flow_box = _clip_box(flow_box, width, height, config.min_box_size)

        if state is not None and covariance is not None:
            state, covariance = _kalman_predict(state, covariance)
        if flow_box is not None:
            flow_measurement = _measurement_from_box(flow_box)
            if state is None or covariance is None:
                state, covariance = _initial_state(flow_measurement)
            else:
                state, covariance = _kalman_update(
                    state, covariance, flow_measurement, is_flow=True
                )

        accepted_observation = False
        if observation_ok:
            if state is None or covariance is None:
                state, covariance = _initial_state(_measurement_from_box(observation))
                accepted_observation = True
            else:
                motion_box = _clip_box(
                    _box_from_measurement(state[:4]), width, height, config.min_box_size
                )
                gate_reject = (
                    flow_box is not None
                    and motion_box is not None
                    and _bbox_iou(observation, motion_box) < config.observation_iou_gate
                    and _center_distance_ratio(observation, motion_box)
                    > config.observation_center_gate
                    and not high_quality_observation
                )
                if gate_reject:
                    rejected_observation[frame_idx] = True
                else:
                    state, covariance = _kalman_update(
                        state, covariance, _measurement_from_box(observation), is_flow=False
                    )
                    accepted_observation = True

        if accepted_observation and high_quality_observation:
            # High-confidence observations define the crop exactly. The
            # Kalman state is still updated for an upcoming short occlusion,
            # but it must not blur a clear, fast gesture in the actual model.
            candidate = observation
            direct_observation[frame_idx] = True
        else:
            candidate = (
                _clip_box(
                    _box_from_measurement(state[:4]), width, height, config.min_box_size
                )
                if state is not None
                else None
            )
        if accepted_observation:
            missed = 0
            prediction_only = 0
            current_source = TRACK_SOURCE_OBSERVATION
        elif flow_box is not None:
            missed += 1
            prediction_only = 0
            current_source = TRACK_SOURCE_FLOW
        else:
            missed += 1
            prediction_only += 1
            current_source = TRACK_SOURCE_PREDICTION

        current_usable = bool(
            candidate is not None
            and missed <= int(config.max_gap)
            and prediction_only <= int(config.max_prediction_gap)
        )
        if current_usable:
            tracked_xyxy[frame_idx] = candidate
            usable[frame_idx] = True
            source[frame_idx] = current_source
            previous_box = candidate
        else:
            # Do not let a stale Kalman state pull a later correct hand back.
            state = None
            covariance = None
            previous_box = None
        previous_gray = current_gray

    return {
        "xyxy": tracked_xyxy,
        "usable": usable,
        "source": source,
        "flow_points": flow_points,
        "rejected_observation": rejected_observation,
        "direct_observation": direct_observation,
    }


def _trajectory_steps(boxes: np.ndarray, valid: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if len(boxes) < 2:
        return np.zeros(0, dtype=np.float32)
    centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    pair_valid = valid[:-1] & valid[1:]
    return np.linalg.norm(centers[1:] - centers[:-1], axis=1)[pair_valid].astype(
        np.float32
    )


def summarize_bbox_track(
    observed_xyxy: np.ndarray,
    observed_valid: np.ndarray,
    tracked: dict[str, np.ndarray],
) -> dict[str, float | int]:
    """Return audit metrics; it does not assert an accuracy improvement."""

    observed_steps = _trajectory_steps(observed_xyxy, observed_valid)
    tracked_steps = _trajectory_steps(tracked["xyxy"], tracked["usable"])

    def percentile(values: np.ndarray, p: float) -> float:
        return float(np.percentile(values, p)) if values.size else float("nan")

    source = np.asarray(tracked["source"], dtype=np.int8)
    return {
        "observed_frames": int(np.asarray(observed_valid, dtype=bool).sum()),
        "tracked_usable_frames": int(np.asarray(tracked["usable"], dtype=bool).sum()),
        "flow_propagated_frames": int((source == TRACK_SOURCE_FLOW).sum()),
        "prediction_only_frames": int((source == TRACK_SOURCE_PREDICTION).sum()),
        "rejected_detector_frames": int(
            np.asarray(tracked["rejected_observation"], dtype=bool).sum()
        ),
        "direct_observation_frames": int(
            np.asarray(tracked.get("direct_observation", np.zeros(0)), dtype=bool).sum()
        ),
        "observed_center_step_median_px": percentile(observed_steps, 50.0),
        "observed_center_step_p95_px": percentile(observed_steps, 95.0),
        "tracked_center_step_median_px": percentile(tracked_steps, 50.0),
        "tracked_center_step_p95_px": percentile(tracked_steps, 95.0),
    }
