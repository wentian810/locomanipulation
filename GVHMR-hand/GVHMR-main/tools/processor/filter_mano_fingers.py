#!/usr/bin/env python
"""Offline temporal filter for MANO finger articulation.

This pass does not change wrist/global orientation.  It reduces fast finger
curl jumps, preserves recent grasp state when 2D evidence is weak, and can
rescue frames where ViTPose shows an open hand but MANO stays curled by borrowing
nearby open hand-pose examples from the same clip.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


FINGERTIPS = np.asarray([4, 8, 12, 16, 20], dtype=np.int64)
MCP_JOINTS = np.asarray([5, 9, 13, 17], dtype=np.int64)
PALM_SIZE_JOINTS = np.asarray([0, 1, 5, 9, 13, 17], dtype=np.int64)


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


def load_vitpose(path):
    arr = as_numpy(torch.load(path, map_location="cpu", weights_only=True)).astype(np.float32)
    if arr.ndim != 4 or arr.shape[-2] < 133 or arr.shape[-1] < 3:
        raise ValueError(f"Expected vitpose_wholebody shape (P,F,133,3), got {arr.shape}")
    return arr


def hand_keypoints(vitpose_person, side):
    return vitpose_person[:, -42:-21] if side == "left" else vitpose_person[:, -21:]


def curl_proxy_from_21(joints):
    joints = np.asarray(joints, dtype=np.float64)
    dist = np.linalg.norm(joints[:, FINGERTIPS] - joints[:, 0:1], axis=-1)
    open_dist = np.percentile(dist, 95, axis=0)
    open_dist = np.clip(open_dist, 1e-6, None)
    return np.clip(1.0 - dist / open_dist[None], -0.5, 1.0).mean(axis=1).astype(np.float32)


def finite_xy_conf(points, conf_thr):
    points = np.asarray(points, dtype=np.float32)
    return np.isfinite(points[..., :2]).all(axis=-1) & np.isfinite(points[..., 2]) & (points[..., 2] > float(conf_thr))


def hand_2d_open_raw(keypoints, low_conf_thr=0.2, min_keypoints=4):
    keypoints = np.asarray(keypoints, dtype=np.float32)
    frame_count = keypoints.shape[0]
    raw = np.full(frame_count, np.nan, dtype=np.float32)
    evidence = np.zeros(frame_count, dtype=np.float32)
    visible = np.zeros(frame_count, dtype=bool)
    for frame_idx in range(frame_count):
        kp = keypoints[frame_idx]
        finite = np.isfinite(kp[:, :2]).all(axis=1) & np.isfinite(kp[:, 2])
        valid = finite & (kp[:, 2] > float(low_conf_thr))
        if int(valid.sum()) < int(min_keypoints) or not valid[0]:
            continue
        mcp_valid = MCP_JOINTS[valid[MCP_JOINTS]]
        tip_valid = FINGERTIPS[valid[FINGERTIPS]]
        if tip_valid.size < 2:
            continue
        wrist = kp[0, :2]
        if mcp_valid.size >= 2:
            palm = np.median(np.linalg.norm(kp[mcp_valid, :2] - wrist[None], axis=1))
        else:
            pts = kp[valid, :2]
            palm = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 1.0) * 0.35
        if not np.isfinite(palm) or palm <= 1.0:
            continue
        tips = kp[tip_valid, :2]
        tip_extent = float(np.mean(np.linalg.norm(tips - wrist[None], axis=1)) / palm)
        diffs = tips[:, None, :] - tips[None, :, :]
        tip_spread = float(np.max(np.linalg.norm(diffs, axis=-1)) / palm)
        raw[frame_idx] = 0.7 * tip_extent + 0.3 * tip_spread
        evidence[frame_idx] = float(np.mean(kp[valid, 2]) * np.sqrt(int(valid.sum())))
        visible[frame_idx] = True
    return raw, evidence, visible


def normalize_percentile(values, lo_pct=20, hi_pct=85):
    values = np.asarray(values, dtype=np.float32)
    out = np.full_like(values, np.nan, dtype=np.float32)
    finite = np.isfinite(values)
    if int(finite.sum()) < 10:
        return out
    lo, hi = np.percentile(values[finite], [lo_pct, hi_pct])
    out[finite] = np.clip((values[finite] - float(lo)) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    return out


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
    theta = theta_0 * float(alpha)
    return (np.sin(theta_0 - theta) * q0 + np.sin(theta) * q1) / np.sin(theta_0)


def rotmats_to_quats(rotmats):
    flat = np.asarray(rotmats, dtype=np.float32).reshape(-1, 3, 3)
    quats = np.stack([rotmat_to_quat(rot) for rot in flat], axis=0)
    return quats.reshape(*rotmats.shape[:-2], 4).astype(np.float32)


def quats_to_rotmats(quats):
    flat = np.asarray(quats, dtype=np.float32).reshape(-1, 4)
    rotmats = np.stack([quat_to_rotmat(q) for q in flat], axis=0)
    return rotmats.reshape(*quats.shape[:-1], 3, 3).astype(np.float32)


def align_quat_sequence(quats):
    quats = np.asarray(quats, dtype=np.float32).copy()
    for frame_idx in range(1, quats.shape[0]):
        dot = np.sum(quats[frame_idx] * quats[frame_idx - 1], axis=-1)
        quats[frame_idx, dot < 0.0] *= -1.0
    return quats


def smooth_rotmats(rotmats, window):
    window = int(window)
    if window < 3:
        return np.asarray(rotmats, dtype=np.float32).copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    quats = align_quat_sequence(rotmats_to_quats(rotmats.reshape(rotmats.shape[0], -1, 3, 3)))
    padded = np.pad(quats, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    smooth = np.empty_like(quats)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    for joint_idx in range(quats.shape[1]):
        for comp in range(4):
            smooth[:, joint_idx, comp] = np.convolve(padded[:, joint_idx, comp], kernel, mode="valid")
    smooth /= np.clip(np.linalg.norm(smooth, axis=-1, keepdims=True), 1e-12, None)
    return quats_to_rotmats(smooth).reshape(rotmats.shape).astype(np.float32)


def slerp_rotmats_array(a, b, weights):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    out = a.copy().reshape(a.shape[0], -1, 3, 3)
    aa = a.reshape(a.shape[0], -1, 3, 3)
    bb = b.reshape(b.shape[0], -1, 3, 3)
    for frame_idx, weight in enumerate(weights):
        if weight <= 0.0:
            continue
        if weight >= 1.0:
            out[frame_idx] = bb[frame_idx]
            continue
        for joint_idx in range(out.shape[1]):
            q0 = rotmat_to_quat(aa[frame_idx, joint_idx])
            q1 = rotmat_to_quat(bb[frame_idx, joint_idx])
            out[frame_idx, joint_idx] = quat_to_rotmat(slerp_quat(q0, q1, weight))
    return out.reshape(a.shape).astype(np.float32)


def rotation_angle(rot):
    trace = np.trace(rot, axis1=-2, axis2=-1)
    return np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))


def max_pairwise_extent(joints, indices):
    joints = np.asarray(joints, dtype=np.float32)
    pts = joints[:, indices]
    diff = pts[:, :, None, :] - pts[:, None, :, :]
    return np.max(np.linalg.norm(diff, axis=-1), axis=(1, 2)).astype(np.float32)


def apply_joint_size_floor(joints, reference_joints, reference_mask, floor_ratio, reference_percentile):
    joints = np.asarray(joints, dtype=np.float32)
    reference_joints = np.asarray(reference_joints, dtype=np.float32)
    reference_mask = np.asarray(reference_mask, dtype=bool).reshape(-1)
    floor_ratio = float(floor_ratio)
    if floor_ratio <= 0.0:
        return joints.copy(), np.zeros(joints.shape[0], dtype=bool)

    ref_extent = max_pairwise_extent(reference_joints, PALM_SIZE_JOINTS)
    ref_good = reference_mask & np.isfinite(ref_extent) & (ref_extent > 1e-6)
    if int(ref_good.sum()) < 5:
        ref_good = np.isfinite(ref_extent) & (ref_extent > 1e-6)
    if int(ref_good.sum()) < 5:
        return joints.copy(), np.zeros(joints.shape[0], dtype=bool)

    target = float(np.percentile(ref_extent[ref_good], float(reference_percentile))) * floor_ratio
    cur_extent = max_pairwise_extent(joints, PALM_SIZE_JOINTS)
    fixed = np.isfinite(cur_extent) & (cur_extent > 1e-8) & (cur_extent < target)
    out = joints.copy()
    if np.any(fixed):
        root = out[fixed, 0:1]
        scale = (target / np.clip(cur_extent[fixed], 1e-8, None)).reshape(-1, 1, 1)
        out[fixed] = root + (out[fixed] - root) * scale
    return out.astype(np.float32), fixed


def rate_limit_joints(joints, max_delta):
    joints = np.asarray(joints, dtype=np.float32)
    max_delta = float(max_delta)
    if max_delta <= 0.0 or joints.shape[0] < 2:
        return joints.copy(), np.zeros(joints.shape[0], dtype=bool)
    out = joints.copy()
    limited = np.zeros(joints.shape[0], dtype=bool)
    for frame_idx in range(1, out.shape[0]):
        delta = out[frame_idx] - out[frame_idx - 1]
        dist = np.linalg.norm(delta, axis=-1)
        too_fast = np.isfinite(dist) & (dist > max_delta)
        if not np.any(too_fast):
            continue
        scale = max_delta / np.clip(dist[too_fast], 1e-8, None)
        out[frame_idx, too_fast] = out[frame_idx - 1, too_fast] + delta[too_fast] * scale[:, None]
        limited[frame_idx] = True
    return out.astype(np.float32), limited


def rate_limit_rotmats(rotmats, max_angle):
    rotmats = np.asarray(rotmats, dtype=np.float32)
    max_angle = float(max_angle)
    if max_angle <= 0.0 or rotmats.shape[0] < 2:
        return rotmats.copy(), np.zeros(rotmats.shape[0], dtype=bool)
    out = rotmats.copy().reshape(rotmats.shape[0], -1, 3, 3)
    limited = np.zeros(rotmats.shape[0], dtype=bool)
    for frame_idx in range(1, out.shape[0]):
        rel = out[frame_idx] @ np.swapaxes(out[frame_idx - 1], -1, -2)
        angles = rotation_angle(rel)
        for joint_idx, angle_value in enumerate(angles):
            if float(angle_value) <= max_angle:
                continue
            q0 = rotmat_to_quat(out[frame_idx - 1, joint_idx])
            q1 = rotmat_to_quat(out[frame_idx, joint_idx])
            out[frame_idx, joint_idx] = quat_to_rotmat(slerp_quat(q0, q1, max_angle / max(float(angle_value), 1e-8)))
            limited[frame_idx] = True
    return out.reshape(rotmats.shape).astype(np.float32), limited


def runs(mask):
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    idx = 0
    while idx < mask.shape[0]:
        if not mask[idx]:
            idx += 1
            continue
        start = idx
        while idx + 1 < mask.shape[0] and mask[idx + 1]:
            idx += 1
        yield start, idx
        idx += 1


def nearest_reliable(start, end, reliable):
    prev_idx = start - 1
    while prev_idx >= 0 and not reliable[prev_idx]:
        prev_idx -= 1
    next_idx = end + 1
    while next_idx < reliable.shape[0] and not reliable[next_idx]:
        next_idx += 1
    return (prev_idx if prev_idx >= 0 else None), (next_idx if next_idx < reliable.shape[0] else None)


def fillable_mask(candidates, reliable, max_gap, max_edge_hold):
    out = np.zeros_like(candidates, dtype=bool)
    for start, end in runs(candidates):
        prev_idx, next_idx = nearest_reliable(start, end, reliable)
        gap_len = end - start + 1
        if prev_idx is not None and next_idx is not None and gap_len <= int(max_gap):
            out[start : end + 1] = True
        elif prev_idx is not None and next_idx is None and gap_len <= int(max_edge_hold):
            out[start : end + 1] = True
        elif next_idx is not None and prev_idx is None and gap_len <= int(max_edge_hold):
            out[start : end + 1] = True
    return out


def interp_rotmats(values, fill, reliable):
    values = np.asarray(values, dtype=np.float32)
    out = values.copy().reshape(values.shape[0], -1, 3, 3)
    src = values.reshape(values.shape[0], -1, 3, 3)
    for start, end in runs(fill):
        prev_idx, next_idx = nearest_reliable(start, end, reliable)
        if prev_idx is None and next_idx is None:
            continue
        if prev_idx is None:
            out[start : end + 1] = src[next_idx]
            continue
        if next_idx is None:
            out[start : end + 1] = src[prev_idx]
            continue
        for frame_idx in range(start, end + 1):
            alpha = float(frame_idx - prev_idx) / float(next_idx - prev_idx)
            for joint_idx in range(out.shape[1]):
                q0 = rotmat_to_quat(src[prev_idx, joint_idx])
                q1 = rotmat_to_quat(src[next_idx, joint_idx])
                out[frame_idx, joint_idx] = quat_to_rotmat(slerp_quat(q0, q1, alpha))
    return out.reshape(values.shape).astype(np.float32)


def smooth_linear(values, window):
    values = np.asarray(values, dtype=np.float32)
    if int(window) < 3:
        return values.copy()
    window = int(window) + (int(window) % 2 == 0)
    pad = window // 2
    flat = values.reshape(values.shape[0], -1)
    padded = np.pad(flat, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    out = np.empty_like(flat)
    for col in range(flat.shape[1]):
        out[:, col] = np.convolve(padded[:, col], kernel, mode="valid")
    return out.reshape(values.shape).astype(np.float32)


def interp_linear(values, fill, reliable):
    values = np.asarray(values, dtype=np.float32)
    out = values.copy().reshape(values.shape[0], -1)
    src = values.reshape(values.shape[0], -1)
    for start, end in runs(fill):
        prev_idx, next_idx = nearest_reliable(start, end, reliable)
        if prev_idx is None and next_idx is None:
            continue
        if prev_idx is None:
            out[start : end + 1] = src[next_idx]
            continue
        if next_idx is None:
            out[start : end + 1] = src[prev_idx]
            continue
        for frame_idx in range(start, end + 1):
            alpha = float(frame_idx - prev_idx) / float(next_idx - prev_idx)
            out[frame_idx] = (1.0 - alpha) * src[prev_idx] + alpha * src[next_idx]
    return out.reshape(values.shape).astype(np.float32)


def robust_spike_mask(speed, mad_multiplier, abs_threshold):
    speed = np.asarray(speed, dtype=np.float32)
    finite = np.isfinite(speed)
    if int(finite.sum()) < 10:
        return np.zeros_like(speed, dtype=bool), float(abs_threshold)
    med = float(np.nanmedian(speed[finite]))
    mad = float(np.nanmedian(np.abs(speed[finite] - med)))
    threshold = max(float(abs_threshold), med + float(mad_multiplier) * max(1.4826 * mad, 1e-6))
    return finite & (speed > threshold), threshold


def nearest_true_indices(mask):
    idx = np.flatnonzero(mask)
    return idx


def open_rescue_pose(pose, joints, mismatch, open_anchor, weight):
    if not np.any(mismatch) or not np.any(open_anchor) or weight <= 0.0:
        return pose, joints, np.zeros_like(mismatch, dtype=bool)
    out_pose = pose.copy()
    out_joints = joints.copy()
    fixed = np.zeros_like(mismatch, dtype=bool)
    anchors = nearest_true_indices(open_anchor)
    for frame_idx in np.flatnonzero(mismatch):
        ref = int(anchors[np.argmin(np.abs(anchors - frame_idx))])
        target_pose = np.repeat(pose[ref : ref + 1], 1, axis=0)
        out_pose[frame_idx : frame_idx + 1] = slerp_rotmats_array(
            pose[frame_idx : frame_idx + 1],
            target_pose,
            np.asarray([weight], dtype=np.float32),
        )
        out_joints[frame_idx] = (1.0 - weight) * joints[frame_idx] + weight * joints[ref]
        fixed[frame_idx] = True
    return out_pose, out_joints, fixed


def process_side(mano, vitpose_person, side, person_idx, args):
    pose_key = f"{side}_hand_pose"
    joints_key = f"{side}_hand_joints_3d"
    global_key = f"{side}_hand_global_orient"
    valid_key = f"{side}_hand_valid"
    if pose_key not in mano or joints_key not in mano:
        return None

    pose = as_numpy(mano[pose_key])[person_idx].astype(np.float32)
    joints = as_numpy(mano[joints_key])[person_idx].astype(np.float32)
    global_orient = (
        as_numpy(mano[global_key])[person_idx].astype(np.float32)
        if global_key in mano
        else None
    )
    valid = as_numpy(mano.get(valid_key, np.ones((pose.shape[0],), dtype=bool)))[person_idx].astype(bool)
    frame_count = pose.shape[0]

    keypoints = hand_keypoints(vitpose_person[:frame_count], side)
    open_raw, evidence, visible = hand_2d_open_raw(
        keypoints,
        low_conf_thr=args.hand_low_conf_thr,
        min_keypoints=args.hand_min_keypoints,
    )
    open_score = normalize_percentile(open_raw)
    curl = curl_proxy_from_21(joints)
    curl_speed = np.zeros(frame_count, dtype=np.float32)
    curl_speed[1:] = np.abs(np.diff(curl))
    open_delta = np.zeros(frame_count, dtype=np.float32)
    finite_open = np.isfinite(open_score)
    open_delta[1:] = np.abs(np.nan_to_num(open_score[1:], nan=0.0) - np.nan_to_num(open_score[:-1], nan=0.0))
    open_support = visible & finite_open & (open_delta > float(args.open_change_support_thr))

    spike, spike_thr = robust_spike_mask(curl_speed, args.curl_spike_mad, args.curl_spike_abs)
    spike = spike & ~open_support
    low_evidence = (~visible) | (evidence < float(args.finger_evidence_thr))

    finite_curl = curl[np.isfinite(curl)]
    curl_high_thr = float(np.percentile(finite_curl, args.mismatch_curl_percentile)) if finite_curl.size else np.inf
    mismatch = (
        visible
        & finite_open
        & (open_score >= float(args.open_score_thr))
        & np.isfinite(curl)
        & (curl >= curl_high_thr)
        & (evidence >= float(args.finger_evidence_thr))
    )
    curl_low_thr = float(np.percentile(finite_curl, args.open_anchor_curl_percentile)) if finite_curl.size else -np.inf
    open_anchor = (
        valid
        & visible
        & finite_open
        & (open_score >= float(args.open_score_thr))
        & np.isfinite(curl)
        & (curl <= curl_low_thr)
        & (evidence >= float(args.finger_evidence_thr))
    )
    if int(open_anchor.sum()) == 0:
        visible_good = valid & visible & np.isfinite(curl)
        if int(visible_good.sum()) > 0:
            cutoff = np.percentile(curl[visible_good], 10)
            open_anchor = visible_good & (curl <= cutoff)

    fill_candidates = spike | low_evidence
    temporal_fixed_key = f"{side}_hand_temporal_fixed_mask"
    if temporal_fixed_key in mano:
        fill_candidates = fill_candidates | as_numpy(mano[temporal_fixed_key])[person_idx].astype(bool)
    reliable = valid & ~spike & ~low_evidence & ~mismatch
    fill = fillable_mask(fill_candidates, reliable, args.max_interp_gap, args.max_edge_hold)

    fixed_pose = interp_rotmats(pose, fill, reliable)
    fixed_joints = interp_linear(joints, fill, reliable)
    fixed_pose, fixed_joints, open_rescue = open_rescue_pose(
        fixed_pose,
        fixed_joints,
        mismatch,
        open_anchor,
        args.open_rescue_weight,
    )

    smoothed_pose = smooth_rotmats(fixed_pose, args.smooth_window)
    smoothed_joints = smooth_linear(fixed_joints, args.smooth_window)
    blend = np.full(frame_count, float(args.reliable_smooth_weight), dtype=np.float32)
    blend[low_evidence] = float(args.weak_smooth_weight)
    blend[spike | fill] = float(args.bad_smooth_weight)
    blend[open_rescue] = np.maximum(blend[open_rescue], float(args.open_rescue_smooth_weight))
    out_pose = slerp_rotmats_array(fixed_pose, smoothed_pose, np.clip(blend, 0.0, 1.0))
    out_joints = ((1.0 - blend[:, None, None]) * fixed_joints + blend[:, None, None] * smoothed_joints).astype(np.float32)
    out_pose, rate_limited = rate_limit_rotmats(out_pose, args.max_joint_angle_delta)
    out_joints, joint_rate_limited = rate_limit_joints(out_joints, args.max_joint_xyz_delta)
    out_joints, size_floor = apply_joint_size_floor(
        out_joints,
        joints,
        valid & ~low_evidence & ~spike,
        args.hand_size_floor_ratio,
        args.hand_size_reference_percentile,
    )

    out_global = global_orient
    wrist_rate_limited = np.zeros(frame_count, dtype=bool)
    wrist_fill = np.zeros(frame_count, dtype=bool)
    if global_orient is not None:
        # Smooth the backend's own wrist frame. Mixing it with a separate body
        # model or an explicit 180-degree candidate can introduce branch flips,
        # especially for Hand4Whole++, whose body model is not GVHMR's model.
        wrist_bad = (~valid) | low_evidence
        temporal_bad_key = f"{side}_hand_temporal_bad_mask"
        if temporal_bad_key in mano:
            wrist_bad |= as_numpy(mano[temporal_bad_key])[person_idx].astype(bool)
        wrist_reliable = ~wrist_bad
        wrist_fill = fillable_mask(
            wrist_bad,
            wrist_reliable,
            args.max_interp_gap,
            args.max_edge_hold,
        )
        fixed_global = interp_rotmats(global_orient, wrist_fill, wrist_reliable)
        smoothed_global = smooth_rotmats(fixed_global, args.wrist_smooth_window)
        wrist_blend = np.full(
            frame_count,
            float(args.wrist_reliable_smooth_weight),
            dtype=np.float32,
        )
        wrist_blend[low_evidence] = float(args.wrist_weak_smooth_weight)
        wrist_blend[wrist_bad | wrist_fill] = float(args.wrist_bad_smooth_weight)
        out_global = slerp_rotmats_array(
            fixed_global,
            smoothed_global,
            np.clip(wrist_blend, 0.0, 1.0),
        )
        out_global, wrist_rate_limited = rate_limit_rotmats(
            out_global,
            args.max_wrist_angle_delta,
        )

    raw_curl = curl
    fixed_curl = curl_proxy_from_21(out_joints)
    raw_palm_extent = max_pairwise_extent(joints, PALM_SIZE_JOINTS)
    fixed_palm_extent = max_pairwise_extent(out_joints, PALM_SIZE_JOINTS)
    return {
        "pose": out_pose,
        "joints": out_joints,
        "global_orient": out_global,
        "finger_fixed": fill | spike | low_evidence | open_rescue | size_floor | joint_rate_limited,
        "curl_spike": spike,
        "low_evidence": low_evidence,
        "open_mismatch": mismatch,
        "open_rescue": open_rescue,
        "open_anchor": open_anchor,
        "rate_limited": rate_limited,
        "joint_rate_limited": joint_rate_limited,
        "size_floor": size_floor,
        "wrist_fill": wrist_fill,
        "wrist_rate_limited": wrist_rate_limited,
        "open_score": open_score,
        "evidence": evidence,
        "raw_curl": raw_curl,
        "fixed_curl": fixed_curl,
        "stats": {
            "frames": int(frame_count),
            "low_evidence": int(np.sum(low_evidence)),
            "curl_spike": int(np.sum(spike)),
            "open2d_mano_curl_mismatch": int(np.sum(mismatch)),
            "open_anchor": int(np.sum(open_anchor)),
            "open_rescue": int(np.sum(open_rescue)),
            "rate_limited": int(np.sum(rate_limited)),
            "joint_rate_limited": int(np.sum(joint_rate_limited)),
            "size_floor": int(np.sum(size_floor)),
            "wrist_fill": int(np.sum(wrist_fill)),
            "wrist_rate_limited": int(np.sum(wrist_rate_limited)),
            "interp_fill": int(np.sum(fill)),
            "finger_fixed": int(np.sum(fill | spike | low_evidence | open_rescue | size_floor | joint_rate_limited)),
            "raw_curl_speed_p50_p90_p99": np.percentile(curl_speed, [50, 90, 99]).round(5).tolist(),
            "curl_spike_threshold": float(spike_thr),
            "raw_curl_std": float(np.nanstd(raw_curl)),
            "fixed_curl_std": float(np.nanstd(fixed_curl)),
            "raw_palm_extent_p05_p50_p95": np.percentile(raw_palm_extent[np.isfinite(raw_palm_extent)], [5, 50, 95]).round(5).tolist(),
            "fixed_palm_extent_p05_p50_p95": np.percentile(fixed_palm_extent[np.isfinite(fixed_palm_extent)], [5, 50, 95]).round(5).tolist(),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mano_params", required=True)
    parser.add_argument("--vitpose_wholebody", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--person_idx", type=int, default=-1)
    parser.add_argument("--hand_low_conf_thr", type=float, default=0.2)
    parser.add_argument("--hand_min_keypoints", type=int, default=4)
    parser.add_argument("--finger_evidence_thr", type=float, default=1.0)
    parser.add_argument("--open_score_thr", type=float, default=0.7)
    parser.add_argument("--mismatch_curl_percentile", type=float, default=55.0)
    parser.add_argument("--open_anchor_curl_percentile", type=float, default=35.0)
    parser.add_argument("--open_rescue_weight", type=float, default=0.45)
    parser.add_argument("--open_rescue_smooth_weight", type=float, default=0.65)
    parser.add_argument("--curl_spike_mad", type=float, default=8.0)
    parser.add_argument("--curl_spike_abs", type=float, default=0.12)
    parser.add_argument("--open_change_support_thr", type=float, default=0.35)
    parser.add_argument("--max_interp_gap", type=int, default=60)
    parser.add_argument("--max_edge_hold", type=int, default=20)
    parser.add_argument("--smooth_window", type=int, default=9)
    parser.add_argument("--reliable_smooth_weight", type=float, default=0.25)
    parser.add_argument("--weak_smooth_weight", type=float, default=0.75)
    parser.add_argument("--bad_smooth_weight", type=float, default=1.0)
    parser.add_argument("--max_joint_angle_delta", type=float, default=0.35, help="Maximum per-frame local MANO finger joint rotation in radians; <=0 disables")
    parser.add_argument("--max_joint_xyz_delta", type=float, default=0.06, help="Maximum per-frame MANO joint displacement in local hand units; <=0 disables")
    parser.add_argument("--wrist_smooth_window", type=int, default=11)
    parser.add_argument("--wrist_reliable_smooth_weight", type=float, default=0.20)
    parser.add_argument("--wrist_weak_smooth_weight", type=float, default=0.75)
    parser.add_argument("--wrist_bad_smooth_weight", type=float, default=1.0)
    parser.add_argument("--max_wrist_angle_delta", type=float, default=0.30, help="Maximum per-frame global wrist rotation in radians; <=0 disables")
    parser.add_argument("--hand_size_floor_ratio", type=float, default=0.94, help="Minimum palm extent relative to reliable raw hand size; <=0 disables")
    parser.add_argument("--hand_size_reference_percentile", type=float, default=50.0)
    args = parser.parse_args()

    mano_path = Path(args.mano_params)
    vitpose_path = Path(args.vitpose_wholebody)
    output = Path(args.output)
    mano = torch.load(mano_path, map_location="cpu", weights_only=False)
    vitpose = load_vitpose(vitpose_path)

    if "left_hand_pose" in mano:
        people = int(as_numpy(mano["left_hand_pose"]).shape[0])
        frame_count = int(as_numpy(mano["left_hand_pose"]).shape[1])
    elif "right_hand_pose" in mano:
        people = int(as_numpy(mano["right_hand_pose"]).shape[0])
        frame_count = int(as_numpy(mano["right_hand_pose"]).shape[1])
    else:
        raise KeyError("mano_params must contain hand_pose")
    person_indices = [args.person_idx] if args.person_idx >= 0 else list(range(people))

    out = dict(mano)
    stats = {}
    empty = np.zeros(frame_count, dtype=bool)
    collected = {
        side: {
            "finger_fixed_mask": [],
            "curl_spike_mask": [],
            "finger_low_evidence_mask": [],
            "open2d_mano_curl_mismatch_mask": [],
            "open_rescue_mask": [],
            "open_anchor_mask": [],
            "finger_rate_limited_mask": [],
            "joint_rate_limited_mask": [],
            "size_floor_mask": [],
            "wrist_fixed_mask": [],
        }
        for side in ("left", "right")
    }

    for person_idx in range(people):
        if person_idx not in person_indices or person_idx >= vitpose.shape[0]:
            for side in ("left", "right"):
                for values in collected[side].values():
                    values.append(empty.copy())
            continue
        stats[f"person_{person_idx}"] = {}
        for side in ("left", "right"):
            result = process_side(out, vitpose[person_idx], side, person_idx, args)
            if result is None:
                for values in collected[side].values():
                    values.append(empty.copy())
                continue
            pose_key = f"{side}_hand_pose"
            joints_key = f"{side}_hand_joints_3d"
            global_key = f"{side}_hand_global_orient"
            pose_arr = as_numpy(out[pose_key]).copy()
            joints_arr = as_numpy(out[joints_key]).copy()
            pose_arr[person_idx] = result["pose"]
            joints_arr[person_idx] = result["joints"]
            out[pose_key] = tensor_like(pose_arr, mano[pose_key])
            out[joints_key] = tensor_like(joints_arr, mano[joints_key])
            if global_key in out and result["global_orient"] is not None:
                global_arr = as_numpy(out[global_key]).copy()
                global_arr[person_idx] = result["global_orient"]
                out[global_key] = tensor_like(global_arr, mano[global_key])
            collected[side]["finger_fixed_mask"].append(result["finger_fixed"])
            collected[side]["curl_spike_mask"].append(result["curl_spike"])
            collected[side]["finger_low_evidence_mask"].append(result["low_evidence"])
            collected[side]["open2d_mano_curl_mismatch_mask"].append(result["open_mismatch"])
            collected[side]["open_rescue_mask"].append(result["open_rescue"])
            collected[side]["open_anchor_mask"].append(result["open_anchor"])
            collected[side]["finger_rate_limited_mask"].append(result["rate_limited"])
            collected[side]["joint_rate_limited_mask"].append(result["joint_rate_limited"])
            collected[side]["size_floor_mask"].append(result["size_floor"])
            collected[side]["wrist_fixed_mask"].append(
                result["wrist_fill"] | result["wrist_rate_limited"]
            )
            stats[f"person_{person_idx}"][side] = result["stats"]

    for side in ("left", "right"):
        for name, values in collected[side].items():
            out[f"{side}_hand_{name}"] = bool_tensor(np.stack(values, axis=0))

    out["finger_filter_stats"] = stats
    out["finger_filter_config"] = vars(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output)
    summary_path = Path(args.summary) if args.summary else output.with_suffix(".json")
    summary = {
        "mano_params": str(mano_path),
        "vitpose_wholebody": str(vitpose_path),
        "output": str(output),
        "stats": stats,
        "config": out["finger_filter_config"],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved finger-filtered MANO params to {output}")
    print(f"Saved finger filter summary to {summary_path}")
    for person, person_stats in stats.items():
        for side, side_stats in person_stats.items():
            print(
                f"  {person} {side}: fixed={side_stats['finger_fixed']}, "
                f"spikes={side_stats['curl_spike']}, low2d={side_stats['low_evidence']}, "
                f"open_rescue={side_stats['open_rescue']}, rate_limited={side_stats['rate_limited']}, "
                f"joint_limited={side_stats['joint_rate_limited']}, size_floor={side_stats['size_floor']}, "
                f"curl_std={side_stats['raw_curl_std']:.4f}->{side_stats['fixed_curl_std']:.4f}"
            )


if __name__ == "__main__":
    main()
