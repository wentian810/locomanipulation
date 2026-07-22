"""Strict, auditable visibility gates for final hand refinement.

The gate deliberately separates *visual evidence is strong* from *the current
3D fit is correct*.  RGB/VitPose confidence cannot prove palm-facing or depth
ordering, so callers must not turn this mask into a semantic palm/back label.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class VisibleEvidenceConfig:
    hand_confidence: float = 0.45
    hand_min_keypoints: int = 8
    hand_mean_confidence: float = 0.50
    wrist_confidence: float = 0.45
    min_mcp_count: int = 3
    min_tip_count: int = 2
    body_confidence: float = 0.35
    min_forearm_px: float = 20.0
    min_bbox_to_forearm_ratio: float = 0.55
    min_bbox_diagonal_px: float = 96.0
    min_contiguous_frames: int = 3

    def to_dict(self) -> dict:
        return asdict(self)


# Reason codes are bit flags so a report can distinguish evidence absence from
# an optimizer rejection without overwriting the original hand track.
REASON_NOT_SOURCE_RELIABLE = 1 << 0
REASON_REPAIRED_OR_UNSAFE = 1 << 1
REASON_HAND_KEYPOINTS = 1 << 2
REASON_WRIST_KEYPOINT = 1 << 3
REASON_HAND_ANATOMY_COVERAGE = 1 << 4
REASON_BODY_FOREARM = 1 << 5
REASON_BBOX_GEOMETRY = 1 << 6
REASON_SHORT_RUN = 1 << 7
REASON_SCHEMA_UNAVAILABLE = 1 << 8


_UNSAFE_MASK_SUFFIXES = (
    "hand_temporal_fixed_mask",
    "hand_finger_fixed_mask",
    "hand_wrist_fixed_mask",
    "hand_low_evidence_mask",
    "hand_finger_low_evidence_mask",
    "hand_bbox_shrink_mask",
    "hand_bbox_jump_mask",
    "hand_bbox_overlap_mask",
    "hand_bbox_merge_fixed_mask",
    "hand_bbox_size_floor_mask",
    "hand_size_floor_mask",
)


def _as_bool_array(value, shape: tuple[int, int]) -> np.ndarray:
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    array = np.asarray(array, dtype=bool)
    if array.shape == shape:
        return array
    if array.shape == shape[1:] and shape[0] == 1:
        return array[None]
    raise ValueError(f"Expected mask shape {shape}, got {array.shape}")


def _as_float_array(value, shape_prefix: tuple[int, int], trailing: tuple[int, ...]) -> np.ndarray:
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    array = np.asarray(array, dtype=np.float32)
    expected = (*shape_prefix, *trailing)
    if array.shape == expected:
        return array
    if array.shape == expected[1:] and shape_prefix[0] == 1:
        return array[None]
    raise ValueError(f"Expected array shape {expected}, got {array.shape}")


def _minimum_run_mask(mask: np.ndarray, min_run: int) -> np.ndarray:
    """Keep only true runs that have at least ``min_run`` consecutive frames."""

    mask = np.asarray(mask, dtype=bool)
    if min_run <= 1:
        return mask.copy()
    output = np.zeros_like(mask)
    for person_idx in range(mask.shape[0]):
        start = None
        for frame_idx, value in enumerate(mask[person_idx]):
            if value and start is None:
                start = frame_idx
            elif not value and start is not None:
                if frame_idx - start >= min_run:
                    output[person_idx, start:frame_idx] = True
                start = None
        if start is not None and mask.shape[1] - start >= min_run:
            output[person_idx, start:] = True
    return output


def _unsafe_source_mask(
    mano: Mapping[str, object], side: str, shape: tuple[int, int]
) -> tuple[np.ndarray, list[str]]:
    missing = []
    unsafe = np.zeros(shape, dtype=bool)
    for suffix in _UNSAFE_MASK_SUFFIXES:
        key = f"{side}_{suffix}"
        if key not in mano:
            missing.append(key)
            continue
        unsafe |= _as_bool_array(mano[key], shape)
    return unsafe, missing


def build_visible_evidence(
    mano: Mapping[str, object],
    side: str,
    hand_keypoints: object,
    body_keypoints: object,
    config: VisibleEvidenceConfig = VisibleEvidenceConfig(),
) -> dict[str, object]:
    """Build strict evidence masks for one hand side.

    ``hand_keypoints`` has MANO/COCO-WholeBody order ``(P,F,21,3)`` and
    ``body_keypoints`` is the first 17 COCO body points ``(P,F,17,3)``.
    The function fails closed if the repair provenance needed to protect the
    optimizer is not available.
    """

    hand = hand_keypoints.detach().cpu().numpy() if hasattr(hand_keypoints, "detach") else hand_keypoints
    hand = np.asarray(hand, dtype=np.float32)
    if hand.ndim == 3:
        hand = hand[None]
    if hand.ndim != 4 or hand.shape[-2:] != (21, 3):
        raise ValueError(f"Expected hand keypoints (P,F,21,3), got {hand.shape}")
    shape = hand.shape[:2]
    body = _as_float_array(body_keypoints, shape, (17, 3))
    bbox_key = f"{side}_hand_bbox_xyxy"
    reliable_key = f"{side}_hand_reliable_mask"
    missing = [key for key in (bbox_key, reliable_key) if key not in mano]
    unsafe, unsafe_missing = _unsafe_source_mask(mano, side, shape)
    missing.extend(unsafe_missing)

    reason = np.zeros(shape, dtype=np.int32)
    if missing:
        reason[:] = REASON_SCHEMA_UNAVAILABLE
        return {
            "available": False,
            "missing_fields": missing,
            "visible_evidence_mask": np.zeros(shape, dtype=bool),
            "refine_eligible_mask": np.zeros(shape, dtype=bool),
            "reason_code": reason,
            "evidence_score": np.zeros(shape, dtype=np.float32),
            "high_confidence_count": np.zeros(shape, dtype=np.int16),
            "hand_bbox_diagonal_px": np.full(shape, np.nan, dtype=np.float32),
            "forearm_px": np.full(shape, np.nan, dtype=np.float32),
        }

    reliable = _as_bool_array(mano[reliable_key], shape)
    bbox = _as_float_array(mano[bbox_key], shape, (4,))
    conf = np.nan_to_num(hand[..., 2], nan=0.0, posinf=0.0, neginf=0.0)
    high = conf >= float(config.hand_confidence)
    high_count = high.sum(axis=-1).astype(np.int16)
    weighted_mean = np.sum(conf * high, axis=-1) / np.maximum(high_count, 1)
    wrist_ok = conf[..., 0] >= float(config.wrist_confidence)
    mcp_ok = high[..., (5, 9, 13, 17)].sum(axis=-1) >= int(config.min_mcp_count)
    tip_ok = high[..., (4, 8, 12, 16, 20)].sum(axis=-1) >= int(config.min_tip_count)

    # COCO-17 indices: L elbow/wrist 7/9, R elbow/wrist 8/10.
    elbow_idx, wrist_idx = (7, 9) if side == "left" else (8, 10)
    elbow = body[..., elbow_idx, :]
    body_wrist = body[..., wrist_idx, :]
    body_conf_ok = (
        (elbow[..., 2] >= float(config.body_confidence))
        & (body_wrist[..., 2] >= float(config.body_confidence))
    )
    forearm = np.linalg.norm(body_wrist[..., :2] - elbow[..., :2], axis=-1)
    forearm_ok = body_conf_ok & np.isfinite(forearm) & (forearm >= float(config.min_forearm_px))

    width = bbox[..., 2] - bbox[..., 0]
    height = bbox[..., 3] - bbox[..., 1]
    bbox_side = np.maximum(width, height)
    bbox_diag = np.sqrt(np.maximum(width, 0.0) ** 2 + np.maximum(height, 0.0) ** 2)
    bbox_ok = (
        np.isfinite(bbox).all(axis=-1)
        & (width > 1.0)
        & (height > 1.0)
        & (bbox_diag >= float(config.min_bbox_diagonal_px))
        & (bbox_side / np.maximum(forearm, 1e-6) >= float(config.min_bbox_to_forearm_ratio))
    )

    reason[~reliable] |= REASON_NOT_SOURCE_RELIABLE
    reason[unsafe] |= REASON_REPAIRED_OR_UNSAFE
    hand_points_ok = (high_count >= int(config.hand_min_keypoints)) & (
        weighted_mean >= float(config.hand_mean_confidence)
    )
    reason[~hand_points_ok] |= REASON_HAND_KEYPOINTS
    reason[~wrist_ok] |= REASON_WRIST_KEYPOINT
    reason[~(mcp_ok & tip_ok)] |= REASON_HAND_ANATOMY_COVERAGE
    reason[~forearm_ok] |= REASON_BODY_FOREARM
    reason[~bbox_ok] |= REASON_BBOX_GEOMETRY

    visible = reliable & ~unsafe & hand_points_ok & wrist_ok & mcp_ok & tip_ok & forearm_ok & bbox_ok
    eligible = _minimum_run_mask(visible, int(config.min_contiguous_frames))
    reason[visible & ~eligible] |= REASON_SHORT_RUN
    score = np.clip(weighted_mean, 0.0, 1.0) * np.clip(high_count / 21.0, 0.0, 1.0)
    return {
        "available": True,
        "missing_fields": [],
        "visible_evidence_mask": visible,
        "refine_eligible_mask": eligible,
        "reason_code": reason,
        "evidence_score": score.astype(np.float32),
        "high_confidence_count": high_count,
        "hand_bbox_diagonal_px": bbox_diag.astype(np.float32),
        "forearm_px": forearm.astype(np.float32),
    }
