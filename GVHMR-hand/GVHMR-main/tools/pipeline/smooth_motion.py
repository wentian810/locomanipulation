#!/usr/bin/env python
"""Temporally smooth SMPL NPZ motion before PHC/rendering."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rotation


def scalar(value, default=30.0):
    if value is None:
        return default
    arr = np.asarray(value).reshape(-1)
    return float(arr[0]) if arr.size else default


def valid_window(length, requested, polyorder):
    requested = int(requested)
    if requested <= polyorder or length < 3:
        return None
    if requested % 2 == 0:
        requested += 1
    window = min(requested, length if length % 2 == 1 else length - 1)
    if window <= polyorder or window < 3:
        return None
    return window


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def acceleration(trans, fps):
    trans = np.asarray(trans, dtype=np.float64)
    if trans.shape[0] < 3:
        return np.zeros(0, dtype=np.float64)
    return np.linalg.norm(trans[2:] - 2.0 * trans[1:-1] + trans[:-2], axis=-1) * float(fps) ** 2


def step_root_rotation_degrees(root_orient):
    root_orient = np.asarray(root_orient, dtype=np.float64)
    if root_orient.shape[0] < 2:
        return np.zeros(0, dtype=np.float64)
    rel = Rotation.from_rotvec(root_orient[1:]) * Rotation.from_rotvec(root_orient[:-1]).inv()
    return np.degrees(rel.magnitude())


def step_joint_rotation_degrees(poses):
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape[0] < 2:
        return np.zeros((0,) + poses.shape[1:-1], dtype=np.float64)
    flat = poses.reshape(poses.shape[0], -1, 3)
    rel = Rotation.from_rotvec(flat[1:].reshape(-1, 3)) * Rotation.from_rotvec(flat[:-1].reshape(-1, 3)).inv()
    return np.degrees(rel.magnitude()).reshape(flat.shape[0] - 1, flat.shape[1])


def angular_acceleration(poses, fps):
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape[0] < 3:
        return np.zeros((0,) + poses.shape[1:-1], dtype=np.float64)
    flat = poses.reshape(poses.shape[0], -1, 3)
    return np.linalg.norm(flat[2:] - 2.0 * flat[1:-1] + flat[:-2], axis=-1) * float(fps) ** 2


def interpolate_bad_frames(values, bad_mask):
    values = np.asarray(values, dtype=np.float64)
    bad_mask = np.asarray(bad_mask, dtype=bool)
    if values.shape[0] == 0 or not bad_mask.any() or bad_mask.all():
        return values

    frame_idx = np.arange(values.shape[0])
    good_idx = frame_idx[~bad_mask]
    repaired = values.copy()
    flat = repaired.reshape(values.shape[0], -1)
    good_flat = flat[~bad_mask]
    for dim in range(flat.shape[1]):
        flat[:, dim] = np.interp(frame_idx, good_idx, good_flat[:, dim])
    return flat.reshape(values.shape)


def acceleration_bad_mask(trans, fps, threshold, mad_multiplier, dilate):
    acc = acceleration(trans, fps)
    bad = np.zeros(trans.shape[0], dtype=bool)
    if acc.size == 0:
        return bad

    median = float(np.median(acc))
    mad = float(np.median(np.abs(acc - median)))
    robust_threshold = median + float(mad_multiplier) * 1.4826 * max(mad, 1e-6)
    cutoff = max(float(threshold), robust_threshold)
    bad_indices = np.flatnonzero(acc > cutoff) + 1
    for idx in bad_indices:
        start = max(0, idx - int(dilate))
        end = min(bad.shape[0], idx + int(dilate) + 1)
        bad[start:end] = True
    return bad


def smooth_trans(trans, fps, window, polyorder, acc_threshold, mad_multiplier, dilate):
    trans = np.asarray(trans, dtype=np.float64)
    bad = acceleration_bad_mask(trans, fps, acc_threshold, mad_multiplier, dilate)
    repaired = interpolate_bad_frames(trans, bad)
    smooth_window = valid_window(repaired.shape[0], window, polyorder)
    if smooth_window is not None:
        repaired = savgol_filter(repaired, smooth_window, int(polyorder), axis=0, mode="interp")
    return repaired.astype(np.float32), bad


def ensure_quat_continuity(quat):
    quat = quat.copy()
    for idx in range(1, quat.shape[0]):
        dots = np.sum(quat[idx] * quat[idx - 1], axis=-1, keepdims=True)
        quat[idx] = np.where(dots < 0.0, -quat[idx], quat[idx])
    return quat


def interpolate_quat_bad_frames(quat, bad_mask):
    bad_mask = np.asarray(bad_mask, dtype=bool)
    if quat.shape[0] == 0 or not bad_mask.any():
        return quat

    frame_idx = np.arange(quat.shape[0])
    repaired = quat.copy()
    if bad_mask.ndim == 1:
        if bad_mask.all():
            return quat
        good_idx = frame_idx[~bad_mask]
        flat = repaired.reshape(quat.shape[0], -1)
        good_flat = flat[~bad_mask]
        for dim in range(flat.shape[1]):
            flat[:, dim] = np.interp(frame_idx, good_idx, good_flat[:, dim])
        repaired = flat.reshape(quat.shape)
    else:
        if bad_mask.shape != quat.shape[:2]:
            raise ValueError(f"bad_mask shape {bad_mask.shape} does not match rotations {quat.shape[:2]}")
        for joint_idx in range(quat.shape[1]):
            joint_bad = bad_mask[:, joint_idx]
            if not joint_bad.any() or joint_bad.all():
                continue
            good_idx = frame_idx[~joint_bad]
            for dim in range(4):
                repaired[:, joint_idx, dim] = np.interp(
                    frame_idx,
                    good_idx,
                    repaired[~joint_bad, joint_idx, dim],
                )
    repaired /= np.clip(np.linalg.norm(repaired, axis=-1, keepdims=True), 1e-12, None)
    return repaired


def smooth_rotvec(rotvec, window, polyorder, bad_mask=None):
    rotvec = np.asarray(rotvec, dtype=np.float64)
    original_shape = rotvec.shape
    if rotvec.shape[0] < 3:
        return rotvec.astype(np.float32)

    flat = rotvec.reshape(rotvec.shape[0], -1, 3)
    quat = Rotation.from_rotvec(flat.reshape(-1, 3)).as_quat().reshape(flat.shape[0], flat.shape[1], 4)
    quat = ensure_quat_continuity(quat)
    if bad_mask is not None:
        quat = interpolate_quat_bad_frames(quat, bad_mask)
    smooth_window = valid_window(quat.shape[0], window, polyorder)
    if smooth_window is not None:
        quat = savgol_filter(quat, smooth_window, int(polyorder), axis=0, mode="interp")
        quat /= np.clip(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-12, None)
    smooth = Rotation.from_quat(quat.reshape(-1, 4)).as_rotvec().reshape(flat.shape)
    return smooth.reshape(original_shape).astype(np.float32)


def root_rotation_bad_mask(root_orient, threshold, mad_multiplier, dilate):
    steps = step_root_rotation_degrees(root_orient)
    bad = np.zeros(root_orient.shape[0], dtype=bool)
    if steps.size == 0:
        return bad

    median = float(np.median(steps))
    mad = float(np.median(np.abs(steps - median)))
    robust_threshold = median + float(mad_multiplier) * 1.4826 * max(mad, 1e-6)
    cutoff = max(float(threshold), robust_threshold)
    bad_step_indices = np.flatnonzero(steps > cutoff)
    for step_idx in bad_step_indices:
        frame_idx = int(step_idx) + 1
        start = max(0, frame_idx - int(dilate))
        end = min(bad.shape[0], frame_idx + int(dilate) + 1)
        bad[start:end] = True
    return bad


def robust_cutoffs(values, fixed_threshold, mad_multiplier):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[0] == 0:
        return np.full(values.shape[1], float(fixed_threshold), dtype=np.float64)
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median[None, :]), axis=0)
    robust_threshold = median + float(mad_multiplier) * 1.4826 * np.maximum(mad, 1e-6)
    return np.maximum(float(fixed_threshold), robust_threshold)


def dilate_joint_mask(mask, dilate):
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or int(dilate) <= 0:
        return mask
    expanded = mask.copy()
    for offset in range(1, int(dilate) + 1):
        expanded[offset:] |= mask[:-offset]
        expanded[:-offset] |= mask[offset:]
    return expanded


def joint_rotation_bad_mask(poses, fps, step_threshold, acc_threshold, mad_multiplier, dilate):
    poses = np.asarray(poses, dtype=np.float64)
    bad = np.zeros(poses.shape[:2], dtype=bool)
    if poses.shape[0] < 2:
        return bad

    steps = step_joint_rotation_degrees(poses)
    if steps.size and float(step_threshold) > 0.0:
        cutoffs = robust_cutoffs(steps, step_threshold, mad_multiplier)
        bad[1:] |= steps > cutoffs[None, :]

    acc = angular_acceleration(poses, fps)
    if acc.size and float(acc_threshold) > 0.0:
        cutoffs = robust_cutoffs(acc, acc_threshold, mad_multiplier)
        bad[1:-1] |= acc > cutoffs[None, :]

    return dilate_joint_mask(bad, dilate)


def mask_ratio(mask):
    mask = np.asarray(mask, dtype=bool)
    return float(np.mean(mask)) if mask.size else 0.0


def root_step_score(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, 95) + 0.10 * np.max(values))


def load_payload(path):
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def motion_poses(payload):
    if "poses" in payload:
        poses = np.asarray(payload["poses"], dtype=np.float32)
        if poses.ndim == 2:
            if poses.shape[1] % 3 != 0:
                raise ValueError(f"Cannot reshape poses with shape {poses.shape}")
            return poses.reshape(poses.shape[0], poses.shape[1] // 3, 3)
        return poses.copy()

    if "root_orient" not in payload or "pose_body" not in payload:
        raise ValueError("NPZ must contain poses or root_orient + pose_body")
    root = np.asarray(payload["root_orient"], dtype=np.float32)
    body = np.asarray(payload["pose_body"], dtype=np.float32).reshape(root.shape[0], -1, 3)
    poses = np.zeros((root.shape[0], max(24, body.shape[1] + 1), 3), dtype=np.float32)
    poses[:, 0, :] = root
    poses[:, 1 : body.shape[1] + 1, :] = body
    return poses


def update_payload(payload, poses, trans):
    payload = dict(payload)
    old_poses = payload.get("poses")
    if old_poses is not None and np.asarray(old_poses).ndim == 2:
        payload["poses"] = poses.reshape(poses.shape[0], -1).astype(np.float32)
    else:
        payload["poses"] = poses.astype(np.float32)
    payload["root_orient"] = poses[:, 0, :].astype(np.float32)
    if poses.shape[1] >= 22:
        payload["pose_body"] = poses[:, 1:22, :].reshape(poses.shape[0], 63).astype(np.float32)
    payload["trans"] = trans.astype(np.float32)
    return payload


def smooth_file(args):
    payload = load_payload(args.input)
    fps = scalar(payload.get("mocap_frame_rate"), 30.0)
    poses = motion_poses(payload)
    trans = np.asarray(payload["trans"], dtype=np.float32)

    before_acc = acceleration(trans, fps)
    before_root_steps = step_root_rotation_degrees(poses[:, 0, :])

    smooth_t, bad_trans = smooth_trans(
        trans,
        fps,
        args.trans_window,
        args.polyorder,
        args.acc_threshold,
        args.mad_multiplier,
        args.dilate,
    )
    root_bad = root_rotation_bad_mask(poses[:, 0, :], args.root_step_threshold, args.mad_multiplier, args.dilate)
    joint_bad = joint_rotation_bad_mask(
        poses,
        fps,
        args.joint_step_threshold,
        args.joint_acc_threshold,
        args.mad_multiplier,
        args.dilate,
    )
    joint_bad[:, 0] = False
    smooth_p = smooth_rotvec(poses, args.pose_window, args.polyorder, bad_mask=joint_bad)

    root_smoothed = smooth_rotvec(
        poses[:, 0:1, :],
        args.pose_window,
        args.polyorder,
        bad_mask=root_bad,
    )
    root_repaired = smooth_rotvec(
        poses[:, 0:1, :],
        0,
        args.polyorder,
        bad_mask=root_bad,
    )
    root_mode, root_choice = min(
        [
            ("original", poses[:, 0:1, :]),
            ("repaired", root_repaired),
            ("smoothed", root_smoothed),
        ],
        key=lambda item: root_step_score(step_root_rotation_degrees(item[1][:, 0, :])),
    )
    smooth_p[:, 0:1, :] = root_choice

    after_acc = acceleration(smooth_t, fps)
    after_root_steps = step_root_rotation_degrees(smooth_p[:, 0, :])
    before_pose_acc = angular_acceleration(poses, fps)
    after_pose_acc = angular_acceleration(smooth_p, fps)
    pose_delta = np.linalg.norm((smooth_p - poses).reshape(poses.shape[0], -1), axis=1)
    trans_delta = np.linalg.norm(smooth_t - trans, axis=1)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **update_payload(payload, smooth_p, smooth_t))

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "frames": int(poses.shape[0]),
        "fps": float(fps),
        "pose_window": int(args.pose_window),
        "trans_window": int(args.trans_window),
        "polyorder": int(args.polyorder),
        "bad_trans_frame_count": int(np.sum(bad_trans)),
        "bad_trans_frame_ratio": float(np.mean(bad_trans)) if bad_trans.size else 0.0,
        "bad_root_frame_count": int(np.sum(root_bad)),
        "bad_root_frame_ratio": float(np.mean(root_bad)) if root_bad.size else 0.0,
        "bad_joint_entry_count": int(np.sum(joint_bad)),
        "bad_joint_frame_count": int(np.sum(np.any(joint_bad, axis=1))) if joint_bad.size else 0,
        "bad_joint_frame_ratio": mask_ratio(np.any(joint_bad, axis=1)) if joint_bad.size else 0.0,
        "root_smoothing_mode": root_mode,
        "root_acceleration_before": summarize(before_acc),
        "root_acceleration_after": summarize(after_acc),
        "root_step_degrees_before": summarize(before_root_steps),
        "root_step_degrees_after": summarize(after_root_steps),
        "pose_angular_acceleration_before": summarize(before_pose_acc),
        "pose_angular_acceleration_after": summarize(after_pose_acc),
        "pose_delta_norm": summarize(pose_delta),
        "trans_delta_m": summarize(trans_delta),
    }
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    print(
        "smoothed: "
        f"root_acc_p95 {report['root_acceleration_before']['p95']:.2f}"
        f" -> {report['root_acceleration_after']['p95']:.2f}, "
        f"bad_trans={report['bad_trans_frame_count']}/{report['frames']}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--pose_window", type=int, default=5)
    parser.add_argument("--trans_window", type=int, default=7)
    parser.add_argument("--polyorder", type=int, default=2)
    parser.add_argument("--acc_threshold", type=float, default=14.7)
    parser.add_argument("--root_step_threshold", type=float, default=45.0)
    parser.add_argument("--joint_step_threshold", type=float, default=45.0)
    parser.add_argument("--joint_acc_threshold", type=float, default=450.0)
    parser.add_argument("--mad_multiplier", type=float, default=6.0)
    parser.add_argument("--dilate", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    smooth_file(parse_args())
