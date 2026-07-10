#!/usr/bin/env python
"""Confidence-weighted temporal MANO fusion for low-evidence hand frames.

This is deliberately *HySUP-inspired*, rather than a claim that the current
Hand4Whole++ model has two independent finger estimators.  In this project the
45-D finger articulation is produced by WiLoR; the body branch independently
constrains the wrist only (via ``filter_mano_wrist.py``).  For low-confidence
finger frames this module therefore blends WiLoR with an offline temporal
reference made from neighbouring reliable MANO frames.  It leaves the wrist
orientation untouched so the preceding body-aware wrist filter remains the
single authority for that seam.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def to_like(array, like):
    array = np.ascontiguousarray(array)
    if torch.is_tensor(like):
        return torch.from_numpy(array).to(dtype=like.dtype)
    return array


def preserve_dtype(array, like):
    """Store diagnostic arrays without coercing an integer code to bool."""

    array = np.ascontiguousarray(array)
    if torch.is_tensor(like):
        return torch.from_numpy(array)
    return array


def pose_to_matrices(value, joints=15):
    """Return MANO pose as (people, frames, joints, 3, 3) matrices."""

    array = as_numpy(value)
    had_people_axis = True
    if array.shape[-3:] == (joints, 3, 3):
        matrices = np.asarray(array, dtype=np.float64)
        if matrices.ndim == 4:
            matrices = matrices[None]
            had_people_axis = False
    elif array.shape[-1] == joints * 3:
        rotvec = np.asarray(array, dtype=np.float64).reshape(*array.shape[:-1], joints, 3)
        matrices = Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_matrix().reshape(
            *rotvec.shape[:-1], 3, 3
        )
        if matrices.ndim == 4:
            matrices = matrices[None]
            had_people_axis = False
    elif array.shape[-2:] == (joints, 3):
        rotvec = np.asarray(array, dtype=np.float64)
        matrices = Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_matrix().reshape(
            *rotvec.shape[:-1], 3, 3
        )
        if matrices.ndim == 4:
            matrices = matrices[None]
            had_people_axis = False
    else:
        raise ValueError(
            f"Expected MANO pose (...,{joints},3,3), (...,{joints * 3}) or (...,{joints},3); got {array.shape}"
        )
    if matrices.ndim != 5 or matrices.shape[2] != joints:
        raise ValueError(f"Expected (P,F,{joints},3,3) after conversion, got {matrices.shape}")
    return matrices, had_people_axis


def matrices_to_pose(matrices, like, had_people_axis):
    array = as_numpy(like)
    out = matrices[0] if not had_people_axis else matrices
    if array.shape[-3:] == (15, 3, 3):
        return to_like(out.astype(np.float32), like)
    rotvec = Rotation.from_matrix(out.reshape(-1, 3, 3)).as_rotvec()
    if array.shape[-1] == 45:
        return to_like(rotvec.reshape(*out.shape[:-3], 45).astype(np.float32), like)
    if array.shape[-2:] == (15, 3):
        return to_like(rotvec.reshape(*out.shape[:-2], 15, 3).astype(np.float32), like)
    raise ValueError(f"Cannot restore MANO pose layout {array.shape}")


def as_people_frames(value, people, frames, dtype, default):
    if value is None:
        return np.full((people, frames), default, dtype=dtype)
    array = np.asarray(as_numpy(value), dtype=dtype)
    if array.ndim == 1 and people == 1:
        array = array[None]
    if array.shape != (people, frames):
        raise ValueError(f"Expected ({people},{frames}), got {array.shape}")
    return array


def as_people_frame_vectors(value, people, frames, width):
    array = np.asarray(as_numpy(value), dtype=np.float64)
    if array.ndim == 2 and people == 1:
        array = array[None]
    if array.shape != (people, frames, width):
        raise ValueError(f"Expected ({people},{frames},{width}), got {array.shape}")
    return array


def as_people_frame_joints(value, people, frames):
    array = np.asarray(as_numpy(value), dtype=np.float64)
    if array.ndim == 3 and people == 1:
        array = array[None]
    if array.shape != (people, frames, 21, 3):
        raise ValueError(f"Expected ({people},{frames},21,3), got {array.shape}")
    return array


def rotation_angle(rotations):
    trace = np.trace(rotations, axis1=-2, axis2=-1)
    return np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))


def slerp_pair(first, second, weight):
    """SLERP a joint stack shaped (J,3,3) with one scalar weight."""

    quat_a = Rotation.from_matrix(first.reshape(-1, 3, 3)).as_quat()
    quat_b = Rotation.from_matrix(second.reshape(-1, 3, 3)).as_quat()
    dot = np.sum(quat_a * quat_b, axis=-1, keepdims=True)
    quat_b = np.where(dot < 0.0, -quat_b, quat_b)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    theta = np.arccos(dot)
    sine = np.sin(theta)
    weight = float(np.clip(weight, 0.0, 1.0))
    near = sine[:, 0] < 1e-7
    out = np.empty_like(quat_a)
    out[near] = (1.0 - weight) * quat_a[near] + weight * quat_b[near]
    if np.any(~near):
        out[~near] = (
            np.sin((1.0 - weight) * theta[~near]) / sine[~near] * quat_a[~near]
            + np.sin(weight * theta[~near]) / sine[~near] * quat_b[~near]
        )
    out /= np.clip(np.linalg.norm(out, axis=-1, keepdims=True), 1e-8, None)
    return Rotation.from_quat(out).as_matrix().reshape(first.shape)


def slerp_sequence(fallback, raw, alpha):
    """Vectorized SLERP for (P,F,J,3,3), with raw weight alpha=(P,F)."""

    shape = raw.shape
    weights = np.repeat(np.asarray(alpha, dtype=np.float64)[..., None], shape[2], axis=2).reshape(-1, 1)
    quat_a = Rotation.from_matrix(fallback.reshape(-1, 3, 3)).as_quat()
    quat_b = Rotation.from_matrix(raw.reshape(-1, 3, 3)).as_quat()
    dot = np.sum(quat_a * quat_b, axis=-1, keepdims=True)
    quat_b = np.where(dot < 0.0, -quat_b, quat_b)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    theta = np.arccos(dot)
    sine = np.sin(theta)
    near = sine[:, 0] < 1e-7
    out = np.empty_like(quat_a)
    out[near] = (1.0 - weights[near]) * quat_a[near] + weights[near] * quat_b[near]
    if np.any(~near):
        out[~near] = (
            np.sin((1.0 - weights[~near]) * theta[~near]) / sine[~near] * quat_a[~near]
            + np.sin(weights[~near] * theta[~near]) / sine[~near] * quat_b[~near]
        )
    out /= np.clip(np.linalg.norm(out, axis=-1, keepdims=True), 1e-8, None)
    return Rotation.from_quat(out).as_matrix().reshape(shape)


def smooth_scores(scores, window):
    window = int(window)
    if window <= 1 or scores.shape[1] <= 1:
        return scores
    if window % 2 == 0:
        window += 1
    radius = window // 2
    kernel = np.ones(window, dtype=np.float64) / float(window)
    out = np.empty_like(scores)
    for person in range(scores.shape[0]):
        padded = np.pad(scores[person], (radius, radius), mode="edge")
        out[person] = np.convolve(padded, kernel, mode="valid")
    return out


def evidence_quality(valid, reproj_error, bbox, pose, args):
    people, frames = valid.shape
    reproj = np.ones((people, frames), dtype=np.float64)
    finite_error = np.isfinite(reproj_error) & (reproj_error < 1e5)
    denom = max(float(args.reproj_bad_px) - float(args.reproj_good_px), 1e-6)
    reproj[finite_error] = np.clip(
        (float(args.reproj_bad_px) - reproj_error[finite_error]) / denom, 0.0, 1.0
    )
    reproj[~finite_error] = 0.0

    diag = np.linalg.norm(bbox[..., 2:] - bbox[..., :2], axis=-1)
    bbox_quality = np.ones((people, frames), dtype=np.float64)
    finite_bbox = np.isfinite(diag) & (diag > 0.0)
    bbox_denom = max(float(args.bbox_saturate) - float(args.bbox_min_diag), 1e-6)
    bbox_quality[finite_bbox] = np.clip(
        (diag[finite_bbox] - float(args.bbox_min_diag)) / bbox_denom, 0.0, 1.0
    )
    bbox_quality[~finite_bbox] = 0.0

    temporal = np.ones((people, frames), dtype=np.float64)
    if frames > 1:
        delta = pose[:, 1:] @ np.swapaxes(pose[:, :-1], -1, -2)
        mean_angle = rotation_angle(delta).mean(axis=2)
        temporal[:, 1:] = np.clip(
            (float(args.temporal_bad_rad) - mean_angle)
            / max(float(args.temporal_bad_rad) - float(args.temporal_good_rad), 1e-6),
            0.0,
            1.0,
        )
        temporal[:, 0] = temporal[:, 1]

    quality = np.sqrt(reproj * bbox_quality) * temporal
    quality *= valid.astype(np.float64)
    return np.clip(smooth_scores(quality, args.quality_smooth_window), 0.0, 1.0), diag


def temporal_reference(pose, joints, quality, anchor_quality, max_gap, edge_hold):
    """Build interpolation/hold fallback from frames with strong evidence."""

    people, frames = quality.shape
    pose_ref = pose.copy()
    joints_ref = joints.copy()
    source = np.zeros((people, frames), dtype=np.int8)  # 0 raw, 1 interp, 2 prev hold, 3 next hold
    for person in range(people):
        anchors = np.flatnonzero(quality[person] >= float(anchor_quality))
        if anchors.size == 0:
            continue
        for frame in range(frames):
            if quality[person, frame] >= float(anchor_quality):
                continue
            before = anchors[anchors < frame]
            after = anchors[anchors > frame]
            left = int(before[-1]) if before.size else None
            right = int(after[0]) if after.size else None
            if left is not None and right is not None and (right - left - 1) <= int(max_gap):
                fraction = (frame - left) / float(right - left)
                pose_ref[person, frame] = slerp_pair(
                    pose[person, left], pose[person, right], fraction
                )
                joints_ref[person, frame] = (
                    (1.0 - fraction) * joints[person, left] + fraction * joints[person, right]
                )
                source[person, frame] = 1
            elif left is not None and (frame - left) <= int(edge_hold):
                pose_ref[person, frame] = pose[person, left]
                joints_ref[person, frame] = joints[person, left]
                source[person, frame] = 2
            elif right is not None and (right - frame) <= int(edge_hold):
                pose_ref[person, frame] = pose[person, right]
                joints_ref[person, frame] = joints[person, right]
                source[person, frame] = 3
    return pose_ref, joints_ref, source


def stats(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    q = np.percentile(values, [0, 25, 50, 75, 95, 100])
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "min": float(q[0]),
        "p25": float(q[1]),
        "p50": float(q[2]),
        "p75": float(q[3]),
        "p95": float(q[4]),
        "max": float(q[5]),
    }


def fuse_side(mano, side, args):
    pose_key = f"{side}_hand_pose"
    joints_key = f"{side}_hand_joints_3d"
    if pose_key not in mano or joints_key not in mano:
        return {}, {"skipped": True, "reason": "missing_pose_or_joints"}

    pose, had_people_axis = pose_to_matrices(mano[pose_key])
    people, frames = pose.shape[:2]
    joints = as_people_frame_joints(mano[joints_key], people, frames)
    valid = as_people_frames(mano.get(f"{side}_hand_valid"), people, frames, bool, True)
    reproj = as_people_frames(
        mano.get(f"{side}_hand_reproj_error"), people, frames, np.float64, np.nan
    )
    bbox = (
        as_people_frame_vectors(mano[f"{side}_hand_bbox_xyxy"], people, frames, 4)
        if f"{side}_hand_bbox_xyxy" in mano
        else np.full((people, frames, 4), np.nan, dtype=np.float64)
    )

    quality, bbox_diag = evidence_quality(valid, reproj, bbox, pose, args)
    pose_ref, joints_ref, reference_source = temporal_reference(
        pose,
        joints,
        quality,
        args.anchor_quality,
        args.max_interp_gap,
        args.max_edge_hold,
    )
    alpha = np.where(
        valid,
        float(args.alpha_floor) + (1.0 - float(args.alpha_floor)) * quality,
        0.0,
    )
    # No temporal reference is evidence that the current pose is the only
    # available hypothesis; preserve it exactly instead of manufacturing a hold.
    alpha[reference_source == 0] = 1.0
    fused_pose = slerp_sequence(pose_ref, pose, alpha)
    fused_joints = alpha[..., None, None] * joints + (1.0 - alpha[..., None, None]) * joints_ref
    correction = rotation_angle(fused_pose @ np.swapaxes(pose, -1, -2)).mean(axis=2)
    fallback = (reference_source != 0) & (alpha < 1.0 - 1e-6)

    updates = {
        f"{side}_hand_pose_hysup_unfused": mano[pose_key].detach().cpu().clone()
        if torch.is_tensor(mano[pose_key])
        else np.asarray(mano[pose_key]).copy(),
        f"{side}_hand_joints_3d_hysup_unfused": mano[joints_key].detach().cpu().clone()
        if torch.is_tensor(mano[joints_key])
        else np.asarray(mano[joints_key]).copy(),
        pose_key: matrices_to_pose(fused_pose, mano[pose_key], had_people_axis),
        joints_key: to_like(
            (fused_joints[0] if as_numpy(mano[joints_key]).ndim == 3 else fused_joints).astype(np.float32),
            mano[joints_key],
        ),
        f"{side}_hand_hysup_quality": to_like(
            (quality[0] if as_numpy(mano[pose_key]).ndim in {2, 4} else quality).astype(np.float32),
            mano.get(f"{side}_hand_reproj_error", mano[pose_key]),
        ),
        f"{side}_hand_hysup_alpha": to_like(
            (alpha[0] if as_numpy(mano[pose_key]).ndim in {2, 4} else alpha).astype(np.float32),
            mano.get(f"{side}_hand_reproj_error", mano[pose_key]),
        ),
        f"{side}_hand_hysup_reference_source": preserve_dtype(
            (reference_source[0] if as_numpy(mano[pose_key]).ndim in {2, 4} else reference_source).astype(np.int8),
            mano.get(f"{side}_hand_valid", mano[pose_key]),
        ),
        f"{side}_hand_hysup_fallback_mask": to_like(
            (fallback[0] if as_numpy(mano[pose_key]).ndim in {2, 4} else fallback).astype(bool),
            mano.get(f"{side}_hand_valid", mano[pose_key]),
        ),
    }
    summary = {
        "skipped": False,
        "valid_frames": int(valid.sum()),
        "fallback_frames": int(fallback.sum()),
        "interpolated_frames": int(np.count_nonzero(reference_source == 1)),
        "previous_hold_frames": int(np.count_nonzero(reference_source == 2)),
        "next_hold_frames": int(np.count_nonzero(reference_source == 3)),
        "quality": stats(quality),
        "alpha": stats(alpha),
        "reproj_input_px": stats(reproj),
        "bbox_diag_px": stats(bbox_diag),
        "pose_correction_rad": stats(correction),
    }
    return updates, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mano_params", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--reproj_good_px", type=float, default=20.0)
    parser.add_argument("--reproj_bad_px", type=float, default=40.0)
    parser.add_argument("--bbox_min_diag", type=float, default=80.0)
    parser.add_argument("--bbox_saturate", type=float, default=140.0)
    parser.add_argument("--temporal_good_rad", type=float, default=0.12)
    parser.add_argument("--temporal_bad_rad", type=float, default=0.45)
    parser.add_argument("--quality_smooth_window", type=int, default=5)
    parser.add_argument("--anchor_quality", type=float, default=0.85)
    parser.add_argument("--alpha_floor", type=float, default=0.10)
    parser.add_argument("--max_interp_gap", type=int, default=12)
    parser.add_argument("--max_edge_hold", type=int, default=6)
    args = parser.parse_args()

    if not (args.reproj_bad_px > args.reproj_good_px):
        raise ValueError("--reproj_bad_px must be greater than --reproj_good_px")
    if not (args.bbox_saturate > args.bbox_min_diag):
        raise ValueError("--bbox_saturate must be greater than --bbox_min_diag")
    if not (args.temporal_bad_rad > args.temporal_good_rad):
        raise ValueError("--temporal_bad_rad must be greater than --temporal_good_rad")
    if not (0.0 <= args.anchor_quality <= 1.0 and 0.0 <= args.alpha_floor <= 1.0):
        raise ValueError("--anchor_quality and --alpha_floor must be in [0, 1]")

    source = Path(args.mano_params)
    output = Path(args.output)
    mano = torch.load(source, map_location="cpu", weights_only=False)
    fused = dict(mano)
    report = {
        "input": str(source),
        "output": str(output),
        "method": "confidence_weighted_temporal_articulation_fusion",
        "note": (
            "Finger fallback is a temporal MANO reference. Body-aware wrist fusion is handled by "
            "filter_mano_wrist.py before this stage because the body branch has no independent finger pose."
        ),
        "config": vars(args).copy(),
        "sides": {},
    }
    report["config"].pop("mano_params", None)
    report["config"].pop("output", None)
    report["config"].pop("summary", None)
    for side in ("left", "right"):
        updates, side_report = fuse_side(mano, side, args)
        fused.update(updates)
        report["sides"][side] = side_report
    fused["hysup_fusion_config"] = report["config"]
    fused["hysup_fusion_note"] = report["note"]
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fused, output)
    summary = Path(args.summary) if args.summary else output.with_suffix(".json")
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved HySUP-inspired fused MANO params to {output}")
    for side, details in report["sides"].items():
        print(f"  {side}: fallback_frames={details.get('fallback_frames', 0)}, alpha_p50={details.get('alpha', {}).get('p50', float('nan')):.3f}")


if __name__ == "__main__":
    main()
