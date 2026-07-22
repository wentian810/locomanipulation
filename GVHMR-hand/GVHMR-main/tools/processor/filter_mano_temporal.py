#!/usr/bin/env python
"""Offline temporal filter for HaMeR MANO tracks.

This pass is intentionally separate from HaMeR inference.  It treats future
frames only as evidence for whether the current observation is trustworthy, then
fills weak/outlier frames from nearby reliable MANO states.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


SIDE_BODY = {
    "left": {"wrist": 9, "elbow": 7},
    "right": {"wrist": 10, "elbow": 8},
}
TORSO_IDXS = (5, 6, 11, 12)


def as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def tensor_like(array, like):
    tensor = torch.from_numpy(np.asarray(array))
    if torch.is_tensor(like):
        return tensor.to(dtype=like.dtype)
    return tensor


def bool_tensor(array):
    return torch.from_numpy(np.asarray(array, dtype=bool))


def load_vitpose(path: Path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    arr = as_numpy(data).astype(np.float32)
    if arr.ndim != 4 or arr.shape[-2] < 133 or arr.shape[-1] < 3:
        raise ValueError(f"Expected vitpose_wholebody shape (P,F,133,3), got {arr.shape}")
    return arr


def hand_keypoints(vitpose_person, side):
    return vitpose_person[:, -42:-21] if side == "left" else vitpose_person[:, -21:]


def finite_xy_conf(points, conf_thr):
    points = np.asarray(points, dtype=np.float32)
    return np.isfinite(points[..., :2]).all(axis=-1) & np.isfinite(points[..., 2]) & (points[..., 2] > float(conf_thr))


def body_anchor(vitpose_person, conf_thr):
    body = np.asarray(vitpose_person[:, :17], dtype=np.float32)
    frame_count = body.shape[0]
    anchors = np.full((frame_count, 2), np.nan, dtype=np.float32)
    valid = np.zeros(frame_count, dtype=bool)
    for frame_idx in range(frame_count):
        torso = body[frame_idx, list(TORSO_IDXS)]
        ok = finite_xy_conf(torso, conf_thr)
        pts = torso[ok, :2]
        if pts.shape[0] < 2:
            ok_body = finite_xy_conf(body[frame_idx], conf_thr)
            pts = body[frame_idx, ok_body, :2]
        if pts.shape[0] >= 2:
            anchors[frame_idx] = np.mean(pts, axis=0)
            valid[frame_idx] = True
    return fill_missing_linear(anchors, valid), valid


def forearm_scale(vitpose_person, side, conf_thr):
    body = np.asarray(vitpose_person[:, :17], dtype=np.float32)
    wrist_idx = SIDE_BODY[side]["wrist"]
    elbow_idx = SIDE_BODY[side]["elbow"]
    wrist = body[:, wrist_idx]
    elbow = body[:, elbow_idx]
    valid = finite_xy_conf(wrist, conf_thr) & finite_xy_conf(elbow, conf_thr)
    dist = np.linalg.norm(wrist[:, :2] - elbow[:, :2], axis=1).astype(np.float32)
    valid = valid & np.isfinite(dist) & (dist > 1.0)
    filled = fill_missing_1d(dist, valid)
    finite = np.isfinite(filled) & (filled > 1.0)
    if np.any(finite):
        fallback = float(np.nanmedian(filled[finite]))
    else:
        fallback = 100.0
    filled = np.where(finite, filled, fallback).astype(np.float32)
    return np.clip(filled, 1.0, None), valid


def fill_missing_1d(values, valid):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    frame_count = values.shape[0]
    good = np.flatnonzero(valid & np.isfinite(values))
    if good.size == 0:
        return np.full(frame_count, np.nan, dtype=np.float32)
    if good.size == 1:
        return np.full(frame_count, values[good[0]], dtype=np.float32)
    x = np.arange(frame_count, dtype=np.float32)
    return np.interp(x, good.astype(np.float32), values[good]).astype(np.float32)


def fill_missing_linear(values, valid):
    values = np.asarray(values, dtype=np.float32)
    flat = values.reshape(values.shape[0], -1)
    out = np.empty_like(flat)
    for col in range(flat.shape[1]):
        out[:, col] = fill_missing_1d(flat[:, col], valid & np.isfinite(flat[:, col]))
    return out.reshape(values.shape)


def bbox_side(bbox):
    bbox = np.asarray(bbox, dtype=np.float32)
    return np.maximum(bbox[..., 2] - bbox[..., 0], bbox[..., 3] - bbox[..., 1])


def bbox_center(bbox):
    bbox = np.asarray(bbox, dtype=np.float32)
    return (bbox[..., :2] + bbox[..., 2:]) * 0.5


def bbox_iou(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    denom = area_a + area_b - inter
    return inter / denom if denom > 1e-6 else 0.0


def rotmat_to_quat(rot):
    rot = np.asarray(rot, dtype=np.float64)
    m00, m01, m02 = rot[0]
    m10, m11, m12 = rot[1]
    m20, m21, m22 = rot[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.asarray([(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s], dtype=np.float64)
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(max(1.0 + m00 - m11 - m22, 1e-12)) * 2.0
        quat = np.asarray([0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s], dtype=np.float64)
    elif m11 > m22:
        s = np.sqrt(max(1.0 + m11 - m00 - m22, 1e-12)) * 2.0
        quat = np.asarray([(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s], dtype=np.float64)
    else:
        s = np.sqrt(max(1.0 + m22 - m00 - m11, 1e-12)) * 2.0
        quat = np.asarray([(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s], dtype=np.float64)
    quat = quat / max(float(np.linalg.norm(quat)), 1e-12)
    if quat[3] < 0.0:
        quat *= -1.0
    return quat


def quat_to_rotmat(quat):
    x, y, z, w = np.asarray(quat, dtype=np.float64)
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.asarray(
        [
            [1.0 - yy - zz, xy - wz, xz + wy],
            [xy + wz, 1.0 - xx - zz, yz - wx],
            [xz - wy, yz + wx, 1.0 - xx - yy],
        ],
        dtype=np.float32,
    )


def slerp_quat(q0, q1, alpha):
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 = q0 / max(float(np.linalg.norm(q0)), 1e-12)
    q1 = q1 / max(float(np.linalg.norm(q1)), 1e-12)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        out = q0 + float(alpha) * (q1 - q0)
        return out / max(float(np.linalg.norm(out)), 1e-12)
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * float(alpha)
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return s0 * q0 + s1 * q1


def rolling_median(values, reliable, radius, min_count):
    values = np.asarray(values, dtype=np.float32)
    reliable = np.asarray(reliable, dtype=bool).reshape(-1)
    frame_count = values.shape[0]
    flat = values.reshape(frame_count, -1)
    out = np.full_like(flat, np.nan, dtype=np.float32)
    radius = int(radius)
    min_count = int(min_count)
    for frame_idx in range(frame_count):
        lo = max(0, frame_idx - radius)
        hi = min(frame_count, frame_idx + radius + 1)
        idx = np.arange(lo, hi)
        idx = idx[idx != frame_idx]
        good = idx[reliable[idx]]
        if good.size >= min_count:
            out[frame_idx] = np.nanmedian(flat[good], axis=0)
    return out.reshape(values.shape)


def hand_evidence_score(keyp, conf_thr, low_conf_thr, hi_min_keypoints):
    keyp = np.asarray(keyp, dtype=np.float32)
    finite = np.isfinite(keyp[:, :2]).all(axis=1) & np.isfinite(keyp[:, 2])
    conf = np.where(finite, keyp[:, 2], 0.0)
    high = conf > float(conf_thr)
    low = conf > float(low_conf_thr)
    selected = high if int(high.sum()) >= int(hi_min_keypoints) else low
    count = int(selected.sum())
    if count <= 0:
        return 0.0
    return float(np.mean(conf[selected]) * np.sqrt(count))


def wrist_distance_ratio(vitpose_frame, side, bbox):
    wrist_idx = SIDE_BODY[side]["wrist"]
    wrist = np.asarray(vitpose_frame[wrist_idx], dtype=np.float32)
    if not np.isfinite(wrist[:2]).all() or not np.isfinite(bbox).all():
        return np.inf
    side_len = max(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]), 1.0)
    return float(np.linalg.norm(bbox_center(bbox) - wrist[:2]) / side_len)


def support_score(vitpose_frame, keyp, side, bbox, args):
    evidence = hand_evidence_score(
        keyp,
        conf_thr=args.hand_conf_thr,
        low_conf_thr=args.hand_low_conf_thr,
        hi_min_keypoints=args.hand_hi_min_keypoints,
    )
    wrist_ratio = wrist_distance_ratio(vitpose_frame, side, bbox)
    if np.isfinite(wrist_ratio):
        return evidence * max(float(np.exp(-1.5 * wrist_ratio)), 0.05)
    return evidence * 0.5


def overlap_loser(vitpose_frame, left_keyp, right_keyp, left_bbox, right_bbox, left_reliable, right_reliable, args):
    if bbox_iou(left_bbox, right_bbox) < float(args.bbox_overlap_iou):
        return None

    if left_reliable and not right_reliable:
        return "right"
    if right_reliable and not left_reliable:
        return "left"

    left_own = wrist_distance_ratio(vitpose_frame, "left", left_bbox)
    left_other = wrist_distance_ratio(vitpose_frame, "right", left_bbox)
    right_own = wrist_distance_ratio(vitpose_frame, "right", right_bbox)
    right_other = wrist_distance_ratio(vitpose_frame, "left", right_bbox)
    left_crossed = np.isfinite(left_own) and np.isfinite(left_other) and left_other + 0.25 < left_own
    right_crossed = np.isfinite(right_own) and np.isfinite(right_other) and right_other + 0.25 < right_own
    if left_crossed and not right_crossed:
        return "left"
    if right_crossed and not left_crossed:
        return "right"

    left_score = support_score(vitpose_frame, left_keyp, "left", left_bbox, args)
    right_score = support_score(vitpose_frame, right_keyp, "right", right_bbox, args)
    ratio = float(args.bbox_overlap_score_ratio)
    if left_score * ratio < right_score:
        return "left"
    if right_score * ratio < left_score:
        return "right"

    left_center = bbox_center(left_bbox)
    right_center = bbox_center(right_bbox)
    center_dist = float(np.linalg.norm(left_center - right_center))
    same_crop = center_dist < 0.25 * min(float(bbox_side(left_bbox)), float(bbox_side(right_bbox)))
    if same_crop and bbox_iou(left_bbox, right_bbox) >= max(float(args.bbox_overlap_iou), 0.75):
        if np.isfinite(left_own) and np.isfinite(right_own):
            if left_own > right_own + 0.15:
                return "left"
            if right_own > left_own + 0.15:
                return "right"
        return "left" if left_score < right_score else "right"
    return None


def runs(mask):
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    frame_count = mask.shape[0]
    idx = 0
    while idx < frame_count:
        if not mask[idx]:
            idx += 1
            continue
        start = idx
        while idx + 1 < frame_count and mask[idx + 1]:
            idx += 1
        yield start, idx
        idx += 1


def fillable_mask(bad, reliable, max_gap, max_edge_hold):
    bad = np.asarray(bad, dtype=bool).reshape(-1)
    reliable = np.asarray(reliable, dtype=bool).reshape(-1)
    out = np.zeros_like(bad)
    for start, end in runs(bad):
        prev_idx = start - 1
        while prev_idx >= 0 and not reliable[prev_idx]:
            prev_idx -= 1
        next_idx = end + 1
        while next_idx < bad.shape[0] and not reliable[next_idx]:
            next_idx += 1

        has_prev = prev_idx >= 0
        has_next = next_idx < bad.shape[0]
        gap_len = end - start + 1
        if has_prev and has_next and gap_len <= int(max_gap):
            out[start : end + 1] = True
        elif has_prev and not has_next and (start - prev_idx) <= int(max_edge_hold):
            out[start : end + 1] = True
        elif has_next and not has_prev and (next_idx - end) <= int(max_edge_hold):
            out[start : end + 1] = True
    return out


def merge_short_good_islands(bad, max_len):
    bad = np.asarray(bad, dtype=bool).copy()
    if max_len <= 0:
        return bad
    good = ~bad
    for start, end in runs(good):
        touches_left_bad = start > 0 and bad[start - 1]
        touches_right_bad = end + 1 < bad.shape[0] and bad[end + 1]
        if touches_left_bad and touches_right_bad and (end - start + 1) <= int(max_len):
            bad[start : end + 1] = True
    return bad


def interp_linear(values, fill, reliable):
    values = np.asarray(values, dtype=np.float32)
    out = values.copy()
    flat = out.reshape(out.shape[0], -1)
    src = values.reshape(values.shape[0], -1)
    for start, end in runs(fill):
        prev_idx, next_idx = nearest_reliable(start, end, reliable)
        if prev_idx is not None and next_idx is not None:
            denom = float(next_idx - prev_idx)
            for t in range(start, end + 1):
                alpha = float(t - prev_idx) / denom
                flat[t] = (1.0 - alpha) * src[prev_idx].reshape(-1) + alpha * src[next_idx].reshape(-1)
        elif prev_idx is not None:
            flat[start : end + 1] = src[prev_idx].reshape(1, -1)
        elif next_idx is not None:
            flat[start : end + 1] = src[next_idx].reshape(1, -1)
    return flat.reshape(values.shape).astype(np.float32)


def interp_discrete(values, fill, reliable):
    values = np.asarray(values)
    out = values.copy()
    for start, end in runs(fill):
        prev_idx, next_idx = nearest_reliable(start, end, reliable)
        if prev_idx is None and next_idx is None:
            continue
        if prev_idx is None:
            out[start : end + 1] = values[next_idx]
            continue
        if next_idx is None:
            out[start : end + 1] = values[prev_idx]
            continue
        for t in range(start, end + 1):
            out[t] = values[prev_idx] if (t - prev_idx) <= (next_idx - t) else values[next_idx]
    return out


def interp_rotmats(values, fill, reliable):
    values = np.asarray(values, dtype=np.float32)
    out = values.copy()
    frame_count = out.shape[0]
    flat = out.reshape(frame_count, -1, 3, 3)
    src = values.reshape(frame_count, -1, 3, 3)
    for start, end in runs(fill):
        prev_idx, next_idx = nearest_reliable(start, end, reliable)
        if prev_idx is None and next_idx is None:
            continue
        if prev_idx is None:
            flat[start : end + 1] = src[next_idx]
            continue
        if next_idx is None:
            flat[start : end + 1] = src[prev_idx]
            continue
        for rot_idx in range(flat.shape[1]):
            q0 = rotmat_to_quat(src[prev_idx, rot_idx])
            q1 = rotmat_to_quat(src[next_idx, rot_idx])
            for offset, frame_idx in enumerate(range(start, end + 1), start=1):
                alpha = float(frame_idx - prev_idx) / float(next_idx - prev_idx)
                flat[frame_idx, rot_idx] = quat_to_rotmat(slerp_quat(q0, q1, alpha))
    return flat.reshape(values.shape).astype(np.float32)


def bbox_to_local(bbox, anchor, scale):
    center = bbox_center(bbox)
    size = np.asarray(bbox[..., 2:] - bbox[..., :2], dtype=np.float32)
    scale = np.clip(np.asarray(scale, dtype=np.float32), 1.0, None)
    return np.concatenate([(center - anchor) / scale[:, None], size / scale[:, None]], axis=1).astype(np.float32)


def bbox_from_local(local, anchor, scale):
    scale = np.clip(np.asarray(scale, dtype=np.float32), 1.0, None)
    center = anchor + local[:, :2] * scale[:, None]
    size = np.clip(local[:, 2:] * scale[:, None], 1.0, None)
    return np.concatenate([center - size * 0.5, center + size * 0.5], axis=1).astype(np.float32)


def interp_bbox(bbox, fill, reliable, anchor, scale):
    local = bbox_to_local(bbox, anchor, scale)
    fixed_local = interp_linear(local, fill, reliable)
    out = np.asarray(bbox, dtype=np.float32).copy()
    out[fill] = bbox_from_local(fixed_local, anchor, scale)[fill]
    return out


def apply_bbox_size_floor(bbox, reliable, apply_mask, anchor, scale, args):
    bbox = np.asarray(bbox, dtype=np.float32)
    apply_mask = np.asarray(apply_mask, dtype=bool).reshape(-1)
    if float(args.bbox_size_floor_ratio) <= 0.0:
        return bbox.copy(), np.zeros(bbox.shape[0], dtype=bool)

    local = bbox_to_local(bbox, anchor, scale)
    local_size = local[:, 2:]
    med_size = rolling_median(local_size, reliable, args.bbox_window, args.min_window_reliable)
    floor_size = med_size * float(args.bbox_size_floor_ratio)
    floor_size = np.maximum(floor_size, float(args.bbox_size_floor_min_forearm_ratio))
    finite_floor = np.isfinite(floor_size).all(axis=1)
    too_small = apply_mask & finite_floor & np.any(local_size < floor_size, axis=1)
    fixed_local = local.copy()
    fixed_local[too_small, 2:] = np.maximum(local_size[too_small], floor_size[too_small])
    out = bbox.copy()
    if np.any(too_small):
        out[too_small] = bbox_from_local(fixed_local, anchor, scale)[too_small]
    return out.astype(np.float32), too_small


def nearest_reliable(start, end, reliable):
    prev_idx = start - 1
    while prev_idx >= 0 and not reliable[prev_idx]:
        prev_idx -= 1
    next_idx = end + 1
    while next_idx < reliable.shape[0] and not reliable[next_idx]:
        next_idx += 1
    return (prev_idx if prev_idx >= 0 else None), (next_idx if next_idx < reliable.shape[0] else None)


def detect_side_outliers(mano, vitpose_person, side, person_idx, args):
    valid_key = f"{side}_hand_valid"
    bbox_key = f"{side}_hand_bbox_xyxy"
    error_key = f"{side}_hand_reproj_error"
    if valid_key not in mano or bbox_key not in mano:
        return None

    valid = as_numpy(mano[valid_key])[person_idx].astype(bool)
    bbox = as_numpy(mano[bbox_key])[person_idx].astype(np.float32)
    frame_count = valid.shape[0]
    error = np.full(frame_count, np.nan, dtype=np.float32)
    if error_key in mano:
        error = as_numpy(mano[error_key])[person_idx].astype(np.float32).reshape(-1)

    bbox_finite = np.isfinite(bbox).all(axis=1) & (bbox_side(bbox) > 1.0)
    error_bad = np.zeros(frame_count, dtype=bool)
    if args.reproj_error_thr > 0:
        finite_error = np.isfinite(error)
        error_bad = finite_error & (error > float(args.reproj_error_thr))

    keyp = hand_keypoints(vitpose_person, side)
    low_counts = np.sum(finite_xy_conf(keyp, args.hand_low_conf_thr), axis=1)
    low_evidence = low_counts < int(args.hand_min_keypoints)

    base_reliable = valid & bbox_finite & ~error_bad
    anchor, _ = body_anchor(vitpose_person, args.body_conf_thr)
    scale, scale_valid = forearm_scale(vitpose_person, side, args.body_conf_thr)
    side_len = bbox_side(bbox)
    side_ratio = side_len / np.clip(scale, 1.0, None)

    med_side = rolling_median(side_len, base_reliable, args.bbox_window, args.min_window_reliable)
    med_ratio = rolling_median(side_ratio, base_reliable & scale_valid, args.bbox_window, args.min_window_reliable)
    shrink_side = np.isfinite(med_side) & (side_len < float(args.bbox_shrink_ratio) * med_side)
    shrink_ratio = np.isfinite(med_ratio) & scale_valid & (side_ratio < float(args.bbox_shrink_ratio) * med_ratio)
    too_small_for_forearm = scale_valid & (side_ratio < float(args.min_bbox_forearm_ratio))
    shrink = base_reliable & (shrink_side | shrink_ratio | too_small_for_forearm)

    local_center = (bbox_center(bbox) - anchor) / np.clip(scale[:, None], 1.0, None)
    med_center = rolling_median(local_center, base_reliable, args.bbox_window, args.min_window_reliable)
    med_local_side = rolling_median(side_ratio, base_reliable, args.bbox_window, args.min_window_reliable)
    center_delta = np.linalg.norm(local_center - med_center, axis=1)
    jump_threshold = float(args.bbox_jump_ratio) * np.maximum(med_local_side, 0.35)
    jump = base_reliable & np.isfinite(center_delta) & np.isfinite(jump_threshold) & (center_delta > jump_threshold)

    evidence = base_reliable & low_evidence
    invalid = ~base_reliable
    return {
        "valid": valid,
        "bbox": bbox,
        "keypoints": keyp,
        "base_reliable": base_reliable,
        "invalid": invalid,
        "error_bad": error_bad,
        "low_evidence": evidence,
        "shrink": shrink,
        "jump": jump,
        "anchor": anchor,
        "scale": scale,
    }


def filter_person(mano, vitpose_person, person_idx, args):
    side_info = {
        side: detect_side_outliers(mano, vitpose_person, side, person_idx, args)
        for side in ("left", "right")
    }
    overlap = {side: np.zeros(vitpose_person.shape[0], dtype=bool) for side in ("left", "right")}
    if side_info["left"] is not None and side_info["right"] is not None:
        left = side_info["left"]
        right = side_info["right"]
        for frame_idx in range(vitpose_person.shape[0]):
            if not (np.isfinite(left["bbox"][frame_idx]).all() and np.isfinite(right["bbox"][frame_idx]).all()):
                continue
            loser = overlap_loser(
                vitpose_person[frame_idx],
                left["keypoints"][frame_idx],
                right["keypoints"][frame_idx],
                left["bbox"][frame_idx],
                right["bbox"][frame_idx],
                bool(left["base_reliable"][frame_idx]),
                bool(right["base_reliable"][frame_idx]),
                args,
            )
            if loser is not None:
                overlap[loser][frame_idx] = True

    stats = {}
    masks = {}
    for side, info in side_info.items():
        if info is None:
            continue
        bad = info["invalid"] | info["low_evidence"] | info["shrink"] | info["jump"] | overlap[side]
        bad = merge_short_good_islands(bad, args.merge_short_good)
        reliable = ~bad
        fill = fillable_mask(bad, reliable, args.max_interp_gap, args.max_edge_hold)
        new_valid = reliable | fill
        masks[side] = {
            "bad": bad,
            "reliable": reliable,
            "fill": fill,
            "new_valid": new_valid,
            "invalid": info["invalid"],
            "error_bad": info["error_bad"],
            "low_evidence": info["low_evidence"],
            "shrink": info["shrink"],
            "jump": info["jump"],
            "overlap": overlap[side],
            "anchor": info["anchor"],
            "scale": info["scale"],
        }
        bad_runs = [end - start + 1 for start, end in runs(bad)]
        stats[side] = {
            "frames": int(vitpose_person.shape[0]),
            "observed_valid": int(np.sum(info["valid"])),
            "reliable_before": int(np.sum(info["base_reliable"])),
            "invalid_or_reproj_bad": int(np.sum(info["invalid"])),
            "reproj_bad": int(np.sum(info["error_bad"])),
            "low_evidence": int(np.sum(info["low_evidence"])),
            "bbox_shrink": int(np.sum(info["shrink"])),
            "bbox_jump": int(np.sum(info["jump"])),
            "bbox_overlap_loser": int(np.sum(overlap[side])),
            "outlier_or_invalid": int(np.sum(bad)),
            "filled_frames": int(np.sum(fill)),
            "valid_after": int(np.sum(new_valid)),
            "unfilled_bad": int(np.sum(bad & ~fill)),
            "bad_run_count": int(len(bad_runs)),
            "bad_run_max": int(max(bad_runs) if bad_runs else 0),
        }
    return masks, stats


def apply_global_orient_fill_policy(values, fill, reliable, mode):
    """Apply the explicitly selected temporal policy to MANO wrist rotation.

    ``preserve`` is intentionally a diagnostic/A-B policy, not an automatic
    correction: it lets us determine whether the temporal interpolation itself
    is responsible for an orientation artifact.  It leaves the backend's
    per-frame estimate untouched, including in low-evidence frames.
    """
    if mode == "interpolate":
        return interp_rotmats(values, fill, reliable)
    if mode == "preserve":
        return np.asarray(values, dtype=np.float32).copy()
    raise ValueError(f"Unsupported global orientation fill mode: {mode}")


def apply_side_updates(out, mano, side, person_idx, mask, args):
    reliable = mask["reliable"]
    fill = mask["fill"]
    new_valid = mask["new_valid"]
    merge_fixed = np.zeros_like(fill, dtype=bool)
    global_orient_fixed = np.zeros_like(fill, dtype=bool)

    global_key = f"{side}_hand_global_orient"
    if global_key in out and args.global_orient_fill_mode == "interpolate":
        arr = as_numpy(out[global_key]).copy()
        arr[person_idx] = apply_global_orient_fill_policy(
            arr[person_idx],
            fill,
            reliable,
            args.global_orient_fill_mode,
        )
        out[global_key] = tensor_like(arr, mano[global_key])
        global_orient_fixed = fill.copy()

    pose_key = f"{side}_hand_pose"
    if pose_key in out:
        arr = as_numpy(out[pose_key]).copy()
        arr[person_idx] = interp_rotmats(arr[person_idx], fill, reliable)
        out[pose_key] = tensor_like(arr, mano[pose_key])

    for suffix in ("hand_joints_3d",):
        key = f"{side}_{suffix}"
        if key not in out:
            continue
        arr = as_numpy(out[key]).copy()
        arr[person_idx] = interp_linear(arr[person_idx], fill, reliable)
        out[key] = tensor_like(arr, mano[key])

    bbox_key = f"{side}_hand_bbox_xyxy"
    if bbox_key in out:
        arr = as_numpy(out[bbox_key]).copy()
        arr[person_idx] = interp_bbox(arr[person_idx], fill, reliable, mask["anchor"], mask["scale"])
        size_apply = fill | mask["low_evidence"] | mask["shrink"] | mask["jump"] | mask["overlap"]
        arr[person_idx], size_floor = apply_bbox_size_floor(
            arr[person_idx],
            reliable,
            size_apply,
            mask["anchor"],
            mask["scale"],
            args,
        )
        other_side = "right" if side == "left" else "left"
        other_key = f"{other_side}_hand_bbox_xyxy"
        if args.separation_fix_iou > 0.0 and other_key in out and bbox_key in mano:
            raw_bbox = as_numpy(mano[bbox_key])[person_idx].astype(np.float32)
            other_bbox = as_numpy(out[other_key])[person_idx].astype(np.float32)
            fixed_bbox = arr[person_idx]
            for frame_idx in np.flatnonzero(fill):
                if not (
                    np.isfinite(raw_bbox[frame_idx]).all()
                    and np.isfinite(other_bbox[frame_idx]).all()
                    and np.isfinite(fixed_bbox[frame_idx]).all()
                ):
                    continue
                fixed_iou = bbox_iou(fixed_bbox[frame_idx], other_bbox[frame_idx])
                raw_iou = bbox_iou(raw_bbox[frame_idx], other_bbox[frame_idx])
                fixed_dist = float(np.linalg.norm(bbox_center(fixed_bbox[frame_idx]) - bbox_center(other_bbox[frame_idx])))
                raw_dist = float(np.linalg.norm(bbox_center(raw_bbox[frame_idx]) - bbox_center(other_bbox[frame_idx])))
                close = fixed_dist < float(args.separation_fix_center_ratio) * min(
                    float(bbox_side(fixed_bbox[frame_idx])),
                    float(bbox_side(other_bbox[frame_idx])),
                )
                if (fixed_iou >= float(args.separation_fix_iou) or close) and (
                    raw_iou + float(args.separation_fix_iou_margin) < fixed_iou
                    or raw_dist > fixed_dist + float(args.separation_fix_center_margin)
                ):
                    fixed_bbox[frame_idx] = raw_bbox[frame_idx]
                    merge_fixed[frame_idx] = True
            arr[person_idx] = fixed_bbox
        out[bbox_key] = tensor_like(arr, mano[bbox_key])
    else:
        size_floor = np.zeros_like(fill, dtype=bool)

    error_key = f"{side}_hand_reproj_error"
    if error_key in out:
        arr = as_numpy(out[error_key]).copy()
        arr[person_idx] = interp_linear(arr[person_idx].reshape(-1, 1), fill, reliable).reshape(-1)
        arr[person_idx, fill] = np.minimum(arr[person_idx, fill], float(args.reproj_error_thr) * 0.5)
        out[error_key] = tensor_like(arr, mano[error_key])

    cand_key = f"{side}_hand_candidate_idx"
    if cand_key in out:
        arr = as_numpy(out[cand_key]).copy()
        arr[person_idx] = interp_discrete(arr[person_idx], fill, reliable)
        out[cand_key] = torch.from_numpy(arr).to(dtype=mano[cand_key].dtype)

    valid_key = f"{side}_hand_valid"
    if valid_key in out:
        arr = as_numpy(out[valid_key]).copy()
        arr[person_idx] = new_valid.astype(arr.dtype)
        out[valid_key] = tensor_like(arr, mano[valid_key])
    return merge_fixed, size_floor, global_orient_fixed


def append_mask(out, side, name, values_by_person):
    out[f"{side}_hand_{name}"] = bool_tensor(np.stack(values_by_person, axis=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mano_params", required=True)
    parser.add_argument("--vitpose_wholebody", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--person_idx", type=int, default=-1, help="Filter one person only; default filters all people present in mano_params")
    parser.add_argument("--reproj_error_thr", type=float, default=75.0)
    parser.add_argument("--body_conf_thr", type=float, default=0.2)
    parser.add_argument("--hand_conf_thr", type=float, default=0.35)
    parser.add_argument("--hand_low_conf_thr", type=float, default=0.2)
    parser.add_argument("--hand_hi_min_keypoints", type=int, default=6)
    parser.add_argument("--hand_min_keypoints", type=int, default=3)
    parser.add_argument("--bbox_window", type=int, default=30, help="Frames on each side used for local bbox medians")
    parser.add_argument("--min_window_reliable", type=int, default=5)
    parser.add_argument("--bbox_shrink_ratio", type=float, default=0.65)
    parser.add_argument("--min_bbox_forearm_ratio", type=float, default=0.45)
    parser.add_argument("--bbox_jump_ratio", type=float, default=1.75)
    parser.add_argument("--bbox_overlap_iou", type=float, default=0.55)
    parser.add_argument("--bbox_overlap_score_ratio", type=float, default=1.35)
    parser.add_argument("--separation_fix_iou", type=float, default=0.5)
    parser.add_argument("--separation_fix_iou_margin", type=float, default=0.08)
    parser.add_argument("--separation_fix_center_ratio", type=float, default=0.35)
    parser.add_argument("--separation_fix_center_margin", type=float, default=8.0)
    parser.add_argument("--bbox_size_floor_ratio", type=float, default=0.75)
    parser.add_argument("--bbox_size_floor_min_forearm_ratio", type=float, default=0.45)
    parser.add_argument("--max_interp_gap", type=int, default=60)
    parser.add_argument("--max_edge_hold", type=int, default=15)
    parser.add_argument("--merge_short_good", type=int, default=2)
    parser.add_argument(
        "--global_orient_fill_mode",
        choices=("interpolate", "preserve"),
        default="interpolate",
        help=(
            "Temporal policy for MANO global wrist rotation. 'interpolate' "
            "preserves legacy behavior; 'preserve' is an A/B diagnostic that "
            "does not overwrite low-evidence wrist orientations."
        ),
    )
    args = parser.parse_args()

    mano_path = Path(args.mano_params)
    vitpose_path = Path(args.vitpose_wholebody)
    output = Path(args.output)
    mano = torch.load(mano_path, map_location="cpu", weights_only=False)
    vitpose = load_vitpose(vitpose_path)

    if "left_hand_valid" in mano:
        mano_valid_key = "left_hand_valid"
    elif "right_hand_valid" in mano:
        mano_valid_key = "right_hand_valid"
    else:
        raise KeyError("mano_params must contain left_hand_valid or right_hand_valid")
    mano_valid = as_numpy(mano[mano_valid_key])
    people = int(mano_valid.shape[0])
    frame_count = int(mano_valid.shape[1])
    vitpose_frames = int(vitpose.shape[1])
    if vitpose_frames != frame_count:
        raise ValueError(
            "MANO/ViTPose frame-count mismatch: "
            f"{mano_valid_key} {tuple(mano_valid.shape)} from {mano_path} has "
            f"{frame_count} frames, while vitpose_wholebody {tuple(vitpose.shape)} "
            f"from {vitpose_path} has {vitpose_frames}. This indicates a stale "
            "GVHMR cache after an input/FPS change; rerun the pipeline so this "
            "clip is regenerated."
        )
    person_indices = [args.person_idx] if args.person_idx >= 0 else list(range(people))

    out = dict(mano)
    all_stats = {}
    collected = {
        side: {
            "temporal_fixed_mask": [],
            "temporal_bad_mask": [],
            "bbox_shrink_mask": [],
            "bbox_jump_mask": [],
            "bbox_overlap_mask": [],
            "bbox_merge_fixed_mask": [],
            "bbox_size_floor_mask": [],
            "low_evidence_mask": [],
            "reproj_bad_mask": [],
            "reliable_mask": [],
            "temporal_global_orient_interpolated_mask": [],
        }
        for side in ("left", "right")
    }

    empty = np.zeros(frame_count, dtype=bool)
    for person_idx in range(people):
        if person_idx not in person_indices or person_idx >= vitpose.shape[0]:
            for side in ("left", "right"):
                for name in collected[side]:
                    collected[side][name].append(empty.copy())
            continue
        masks, stats = filter_person(out, vitpose[person_idx], person_idx, args)
        all_stats[f"person_{person_idx}"] = stats
        for side in ("left", "right"):
            if side not in masks:
                for name in collected[side]:
                    collected[side][name].append(empty.copy())
                continue
            merge_fixed, size_floor, global_orient_fixed = apply_side_updates(
                out,
                mano,
                side,
                person_idx,
                masks[side],
                args,
            )
            collected[side]["temporal_fixed_mask"].append(masks[side]["fill"])
            collected[side]["temporal_bad_mask"].append(masks[side]["bad"] & ~masks[side]["fill"])
            collected[side]["bbox_shrink_mask"].append(masks[side]["shrink"])
            collected[side]["bbox_jump_mask"].append(masks[side]["jump"])
            collected[side]["bbox_overlap_mask"].append(masks[side]["overlap"])
            collected[side]["bbox_merge_fixed_mask"].append(merge_fixed)
            collected[side]["bbox_size_floor_mask"].append(size_floor)
            collected[side]["low_evidence_mask"].append(masks[side]["low_evidence"])
            collected[side]["reproj_bad_mask"].append(masks[side]["error_bad"])
            collected[side]["reliable_mask"].append(masks[side]["reliable"])
            collected[side]["temporal_global_orient_interpolated_mask"].append(
                global_orient_fixed
            )
            stats[side]["global_orient_filled"] = int(
                np.sum(global_orient_fixed)
            )

    for side in ("left", "right"):
        for name, values in collected[side].items():
            append_mask(out, side, name, values)
        # Make the existing diagnostic script pick up only unfilled bad frames.
        out[f"{side}_hand_bad_mask"] = out[f"{side}_hand_temporal_bad_mask"]

    out["temporal_filter_stats"] = all_stats
    out["temporal_filter_config"] = {
        "reproj_error_thr": float(args.reproj_error_thr),
        "bbox_window": int(args.bbox_window),
        "bbox_shrink_ratio": float(args.bbox_shrink_ratio),
        "min_bbox_forearm_ratio": float(args.min_bbox_forearm_ratio),
        "bbox_jump_ratio": float(args.bbox_jump_ratio),
        "bbox_overlap_iou": float(args.bbox_overlap_iou),
        "bbox_overlap_score_ratio": float(args.bbox_overlap_score_ratio),
        "separation_fix_iou": float(args.separation_fix_iou),
        "separation_fix_iou_margin": float(args.separation_fix_iou_margin),
        "separation_fix_center_ratio": float(args.separation_fix_center_ratio),
        "separation_fix_center_margin": float(args.separation_fix_center_margin),
        "bbox_size_floor_ratio": float(args.bbox_size_floor_ratio),
        "bbox_size_floor_min_forearm_ratio": float(args.bbox_size_floor_min_forearm_ratio),
        "max_interp_gap": int(args.max_interp_gap),
        "max_edge_hold": int(args.max_edge_hold),
        "merge_short_good": int(args.merge_short_good),
        "global_orient_fill_mode": args.global_orient_fill_mode,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output)
    summary_path = Path(args.summary) if args.summary else output.with_suffix(".json")
    summary = {
        "mano_params": str(mano_path),
        "vitpose_wholebody": str(vitpose_path),
        "output": str(output),
        "stats": all_stats,
        "config": out["temporal_filter_config"],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved temporal-fixed MANO params to {output}")
    print(f"Saved temporal filter summary to {summary_path}")
    for person, stats in all_stats.items():
        for side, side_stats in stats.items():
            print(
                f"  {person} {side}: observed_valid={side_stats['observed_valid']}, "
                f"outlier_or_invalid={side_stats['outlier_or_invalid']}, "
                f"filled={side_stats['filled_frames']}, unfilled={side_stats['unfilled_bad']}"
            )


if __name__ == "__main__":
    main()
