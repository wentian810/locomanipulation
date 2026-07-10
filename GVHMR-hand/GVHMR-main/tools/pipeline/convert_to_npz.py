#!/usr/bin/env python
"""Convert GVHMR-hand results to downstream-compatible motion files.

The locomotion/PHC stages in this workspace still consume a compact SMPL-style
24-joint axis-angle NPZ. GVHMR-hand stores body and MANO hand rotations as
rotation matrices with a leading person axis, so this converter writes:

1. A body-only NPZ compatible with the existing locomotion/PHC pipeline.
2. Optionally, an SMPL-X sidecar NPZ that preserves MANO hand poses.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from pytorch3d.transforms import matrix_to_axis_angle

try:
    from scipy.signal import savgol_filter
    from scipy.spatial.transform import Rotation, Slerp
except Exception:  # pragma: no cover - scipy is present in the normal env
    savgol_filter = None
    Rotation = None
    Slerp = None


def to_tensor(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    return torch.as_tensor(value)


def is_rotmat(tensor):
    return tensor.ndim >= 2 and tuple(tensor.shape[-2:]) == (3, 3)


def select_person(value, person_idx, name):
    tensor = to_tensor(value)
    if is_rotmat(tensor):
        if tensor.ndim == 5:
            return tensor[person_idx]
        # hand_global_orient from hand4wholepp: (P, F, 3, 3) — 4D rotmat
        if tensor.ndim == 4 and tensor.shape[0] <= 16 and tensor.shape[1] > 1:
            return tensor[person_idx]
        return tensor

    if tensor.ndim == 2 and name.endswith("_valid") and tensor.shape[0] <= 16:
        return tensor[person_idx]

    if tensor.ndim >= 4:
        looks_like_people = tensor.shape[0] <= 16 and tensor.shape[1] > 1
        if looks_like_people:
            return tensor[person_idx]

    if tensor.ndim == 3:
        looks_like_people = tensor.shape[0] <= 16 and tensor.shape[1] > 1
        if name == "body_pose":
            if tensor.shape[-1] != 3 and looks_like_people:
                return tensor[person_idx]
        elif name == "global_orient":
            if tensor.shape[0] == 1 and tensor.shape[1] > 1:
                return tensor[person_idx]
            if looks_like_people and tensor.shape[1] != 1:
                return tensor[person_idx]
        elif looks_like_people:
            return tensor[person_idx]
    return tensor


def as_axis_angle(value, person_idx, name):
    tensor = select_person(value, person_idx, name).float()
    if is_rotmat(tensor):
        aa = matrix_to_axis_angle(tensor).numpy()
    else:
        aa = tensor.numpy().astype(np.float32)

    if aa.ndim == 1:
        aa = aa[None, :]
    if aa.ndim >= 3:
        aa = aa.reshape(aa.shape[0], -1)
    return aa.astype(np.float32)


def as_trans(value, person_idx):
    tensor = select_person(value, person_idx, "transl").float().numpy()
    if tensor.ndim == 1:
        tensor = tensor[None, :]
    return tensor.reshape(tensor.shape[0], -1)[:, :3].astype(np.float32)


def as_betas(value, person_idx):
    tensor = select_person(value, person_idx, "betas").float().numpy()
    if tensor.ndim >= 2:
        tensor = tensor.reshape(tensor.shape[0], -1).mean(axis=0)
    return tensor.reshape(-1).astype(np.float32)


def fit_frames(array, frame_count, fill_value=0.0):
    array = np.asarray(array)
    if array.shape[0] == frame_count:
        return array
    if array.shape[0] > frame_count:
        return array[:frame_count]

    pad_shape = (frame_count - array.shape[0],) + array.shape[1:]
    pad = np.full(pad_shape, fill_value, dtype=array.dtype)
    return np.concatenate([array, pad], axis=0)


def _fill_frames_1d(array, frame_count, fill_value=np.nan):
    array = np.asarray(array).reshape(-1)
    if array.shape[0] == frame_count:
        return array
    if array.shape[0] > frame_count:
        return array[:frame_count]
    pad = np.full(frame_count - array.shape[0], fill_value, dtype=array.dtype)
    return np.concatenate([array, pad], axis=0)


def _merge_short_gaps(mask, max_gap):
    mask = np.asarray(mask, dtype=bool).copy()
    if max_gap <= 0 or mask.size == 0:
        return mask
    idx = np.flatnonzero(mask)
    if idx.size < 2:
        return mask
    for left, right in zip(idx[:-1], idx[1:]):
        gap = int(right - left - 1)
        if 0 < gap <= max_gap:
            mask[left + 1 : right] = True
    return mask


def _drop_long_runs(mask, max_burst):
    mask = np.asarray(mask, dtype=bool).copy()
    if max_burst <= 0 or mask.size == 0:
        return mask
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return mask
    splits = np.where(np.diff(idx) > 1)[0] + 1
    for run in np.split(idx, splits):
        if run.size > max_burst:
            mask[run] = False
    return mask


def _post_process_spike_mask(mask, gap_merge=2, max_burst=10):
    return _drop_long_runs(_merge_short_gaps(mask, gap_merge), max_burst)


def _pose_spike_mask(pose, valid, mad_multiplier=8.0, abs_threshold=1.2):
    pose = np.asarray(pose, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    if pose.shape[0] < 3 or int(valid.sum()) < 3:
        return np.zeros(pose.shape[0], dtype=bool)

    flat = pose.reshape(pose.shape[0], -1)
    speed = np.full(pose.shape[0], np.nan, dtype=np.float32)
    ok_pairs = valid[1:] & valid[:-1]
    delta = flat[1:] - flat[:-1]
    speed[1:][ok_pairs] = np.linalg.norm(delta[ok_pairs], axis=1)
    finite = np.isfinite(speed)
    if int(finite.sum()) < 3:
        return np.zeros(pose.shape[0], dtype=bool)

    med = float(np.nanmedian(speed[finite]))
    mad = float(np.nanmedian(np.abs(speed[finite] - med)))
    robust = med + float(mad_multiplier) * max(1.4826 * mad, 1e-6)
    threshold = max(float(abs_threshold), robust)
    mask = np.zeros(pose.shape[0], dtype=bool)
    mask[finite] = speed[finite] > threshold
    mask[0] = False
    return mask


def _interp_linear(values, bad_mask):
    values = np.asarray(values, dtype=np.float32)
    bad_mask = np.asarray(bad_mask, dtype=bool).reshape(-1)
    frame_count = values.shape[0]
    good_idx = np.flatnonzero(~bad_mask & np.all(np.isfinite(values), axis=1))
    if good_idx.size == 0:
        return np.zeros_like(values)
    if good_idx.size == 1:
        return np.repeat(values[good_idx[0] : good_idx[0] + 1], frame_count, axis=0)

    all_idx = np.arange(frame_count, dtype=np.float32)
    out = np.empty_like(values)
    for col in range(values.shape[1]):
        out[:, col] = np.interp(all_idx, good_idx.astype(np.float32), values[good_idx, col])
    return out


def _interp_components(values, bad_mask):
    values = np.asarray(values, dtype=np.float32)
    flat = values.reshape(values.shape[0], -1)
    return _interp_linear(flat, bad_mask).reshape(values.shape)


def _interp_rotvec(values, bad_mask):
    values = np.asarray(values, dtype=np.float32)
    bad_mask = np.asarray(bad_mask, dtype=bool).reshape(-1)
    frame_count = values.shape[0]
    good_idx = np.flatnonzero(~bad_mask & np.all(np.isfinite(values), axis=1))
    if good_idx.size == 0:
        return np.zeros_like(values)
    if good_idx.size == 1 or Rotation is None or Slerp is None:
        return _interp_linear(values, bad_mask)

    out = np.empty_like(values)
    first, last = int(good_idx[0]), int(good_idx[-1])
    out[:first] = values[first]
    out[last + 1 :] = values[last]
    rots = Rotation.from_rotvec(values[good_idx])
    slerp = Slerp(good_idx.astype(np.float64), rots)
    mid_idx = np.arange(first, last + 1, dtype=np.float64)
    out[first : last + 1] = slerp(mid_idx).as_rotvec().astype(np.float32)
    return out


def _interp_hand_pose(pose, bad_mask):
    pose3 = np.asarray(pose, dtype=np.float32).reshape(-1, 15, 3)
    out = np.empty_like(pose3)
    for joint_idx in range(15):
        out[:, joint_idx, :] = _interp_rotvec(pose3[:, joint_idx, :], bad_mask)
    return out.reshape(pose3.shape[0], 45)


def _smooth_components(values, window):
    values = np.asarray(values, dtype=np.float32)
    window = int(window)
    if window < 3 or values.shape[0] < 3:
        return values
    if window % 2 == 0:
        window += 1
    window = min(window, values.shape[0] if values.shape[0] % 2 else values.shape[0] - 1)
    if window < 3:
        return values
    if savgol_filter is not None and window >= 5:
        return savgol_filter(values, window_length=window, polyorder=2, axis=0, mode="interp").astype(np.float32)

    pad = window // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.stack([np.convolve(padded[:, col], kernel, mode="valid") for col in range(values.shape[1])], axis=1).astype(np.float32)


def load_smpl_params(results, space):
    key = f"smpl_params_{space}"
    if key in results:
        print(f"Using {key}")
        return results[key]
    if "smpl_params_global" in results:
        print("Using smpl_params_global")
        return results["smpl_params_global"]
    if "smpl_params_incam" in results:
        print("WARNING: using smpl_params_incam")
        return results["smpl_params_incam"]
    print("Using top-level params")
    return results


def body_npz_from_params(params, fps, person_idx):
    body_pose = as_axis_angle(params["body_pose"], person_idx, "body_pose")
    betas = as_betas(params["betas"], person_idx)
    transl = as_trans(params["transl"], person_idx)

    if "global_orient" in params:
        global_orient = as_axis_angle(params["global_orient"], person_idx, "global_orient")
    elif "global_orient_gv" in params:
        global_orient = as_axis_angle(params["global_orient_gv"], person_idx, "global_orient")
    else:
        raise KeyError("No global_orient / global_orient_gv in GVHMR params")

    frame_count = min(body_pose.shape[0], global_orient.shape[0], transl.shape[0])
    body_pose = body_pose[:frame_count, :63]
    global_orient = global_orient[:frame_count, :3]
    transl = transl[:frame_count, :3]

    poses_full = np.zeros((frame_count, 24, 3), dtype=np.float32)
    poses_full[:, 0, :] = global_orient
    poses_full[:, 1:22, :] = body_pose.reshape(frame_count, 21, 3)

    betas_full = np.zeros(16, dtype=np.float32)
    betas_full[: min(10, betas.shape[0])] = betas[:10]

    return {
        "poses": poses_full,
        "pose_body": body_pose.astype(np.float32),
        "root_orient": global_orient.astype(np.float32),
        "trans": transl.astype(np.float32),
        "trans_original": transl.copy().astype(np.float32),
        "betas": betas_full,
        "gender": np.asarray("neutral"),
        "mocap_frame_rate": np.asarray(fps, dtype=np.float32),
    }


def load_mano_params(path):
    if not path:
        return None
    mano_path = Path(path)
    if not mano_path.is_file():
        print(f"WARNING: mano_params not found: {mano_path}")
        return None
    return torch.load(mano_path, map_location="cpu", weights_only=False)


def hand_pose(mano, key, frame_count, person_idx):
    if mano is None or key not in mano:
        return np.zeros((frame_count, 45), dtype=np.float32)
    pose = as_axis_angle(mano[key], person_idx, key)
    pose = pose[:, :45]
    return fit_frames(pose, frame_count, 0.0).astype(np.float32)


def hand_valid(mano, key, frame_count, person_idx):
    if mano is None or key not in mano:
        return np.zeros(frame_count, dtype=bool)
    valid = select_person(mano[key], person_idx, key).numpy().astype(bool).reshape(-1)
    return fit_frames(valid, frame_count, False).astype(bool)


def hand_error(mano, key, frame_count, person_idx):
    if mano is None or key not in mano:
        return np.full(frame_count, np.nan, dtype=np.float32)
    error = select_person(mano[key], person_idx, key).float().numpy().reshape(-1)
    return _fill_frames_1d(error, frame_count, np.nan).astype(np.float32)


def hand_array(mano, key, frame_count, person_idx, trailing_shape, fill_value=0.0):
    if mano is None or key not in mano:
        return np.full((frame_count, *trailing_shape), fill_value, dtype=np.float32)
    array = select_person(mano[key], person_idx, key).float().numpy()
    array = array.reshape(array.shape[0], *trailing_shape)
    return fit_frames(array, frame_count, fill_value).astype(np.float32)


def optional_hand_mask(mano, key, frame_count, person_idx):
    if mano is None or key not in mano:
        return None
    value = select_person(mano[key], person_idx, key).numpy().astype(bool).reshape(-1)
    return fit_frames(value, frame_count, False).astype(bool)


def optional_hand_scalar(mano, key, frame_count, person_idx, dtype=np.float32):
    if mano is None or key not in mano:
        return None
    value = select_person(mano[key], person_idx, key).numpy().reshape(-1)
    return _fill_frames_1d(value, frame_count, 0).astype(dtype)


def optional_hysup_export(mano, side, frame_count, person_idx):
    """Carry P2 fusion diagnostics into the GMR sidecar without affecting pose."""

    out = {}
    for suffix, dtype in (
        ("hand_hysup_quality", np.float32),
        ("hand_hysup_alpha", np.float32),
        ("hand_hysup_reference_source", np.int8),
    ):
        value = optional_hand_scalar(mano, f"{side}_{suffix}", frame_count, person_idx, dtype)
        if value is not None:
            out[f"{side}_{suffix}"] = value
    mask = optional_hand_mask(mano, f"{side}_hand_hysup_fallback_mask", frame_count, person_idx)
    if mask is not None:
        out[f"{side}_hand_hysup_fallback_mask"] = mask
    return out


def build_hand_quality(
    mano,
    side,
    frame_count,
    person_idx,
    valid,
    reproj_error,
    hand_bbox_xyxy,
):
    """Build one continuous confidence signal shared by every robot-hand IK.

    Upstream temporal/finger filters already know which frames are direct
    observations and which were repaired.  Preserve that information instead
    of independently rediscovering bad frames downstream.
    """
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    reproj_error = np.asarray(reproj_error, dtype=np.float32).reshape(-1)
    bbox = np.asarray(hand_bbox_xyxy, dtype=np.float32).reshape(-1, 4)
    bbox_diag = np.linalg.norm(bbox[:, 2:] - bbox[:, :2], axis=1)
    relative_error = reproj_error / np.clip(bbox_diag, 1.0, None)

    # The same normalized reprojection interval used by target-space IK:
    # <=15% is strong evidence, >=45% contributes no image confidence.
    quality = np.full(frame_count, 0.5, dtype=np.float32)
    relative_finite = (
        np.isfinite(relative_error)
        & np.isfinite(bbox_diag)
        & (bbox_diag > 1.0)
    )
    quality[relative_finite] = np.clip(
        (0.45 - relative_error[relative_finite]) / (0.45 - 0.15),
        0.0,
        1.0,
    )
    pixel_finite = ~relative_finite & np.isfinite(reproj_error) & (reproj_error < 1e5)
    quality[pixel_finite] = np.clip(
        (75.0 - reproj_error[pixel_finite]) / (75.0 - 30.0),
        0.0,
        1.0,
    )

    source_reliable = optional_hand_mask(
        mano,
        f"{side}_hand_reliable_mask",
        frame_count,
        person_idx,
    )
    if source_reliable is None:
        source_reliable = valid.copy()

    repaired_masks = []
    for suffix in (
        "hand_temporal_fixed_mask",
        "hand_temporal_hold_mask",
        "hand_finger_low_evidence_mask",
    ):
        mask = optional_hand_mask(
            mano,
            f"{side}_{suffix}",
            frame_count,
            person_idx,
        )
        if mask is not None:
            repaired_masks.append(mask)
    source_repaired = (
        np.logical_or.reduce(repaired_masks)
        if repaired_masks
        else np.zeros(frame_count, dtype=bool)
    )

    # Weak direct observations remain usable but should not overpower the
    # temporal/posture prior. Repaired frames are governed mainly by that
    # prior. These are soft weights, not another invalid-frame filter.
    evidence_scale = np.where(source_reliable, 1.0, 0.35).astype(np.float32)
    evidence_scale[source_repaired] = np.minimum(
        evidence_scale[source_repaired],
        0.20,
    )
    quality *= evidence_scale
    quality[~valid] = 0.0
    quality = np.clip(quality, 0.0, 1.0).astype(np.float32)
    return {
        f"{side}_hand_quality": quality,
        f"{side}_hand_source_reliable": source_reliable,
        f"{side}_hand_source_repaired": source_repaired,
    }


def refine_hand_track(
    side,
    hand_pose_raw,
    hand_global_raw,
    valid_raw,
    reproj_error,
    hand_bbox_xyxy=None,
    joints_3d_raw=None,
    mode="clean",
    reproj_error_thr=75.0,
    reproj_error_ratio_thr=0.45,
    spike_mad_multiplier=8.0,
    spike_abs_threshold=1.2,
    gap_merge=2,
    max_burst=10,
    smooth_window=0,
):
    hand_pose_raw = np.asarray(hand_pose_raw, dtype=np.float32)
    hand_global_raw = np.asarray(hand_global_raw, dtype=np.float32)
    valid_raw = np.asarray(valid_raw, dtype=bool).reshape(-1)
    reproj_error = np.asarray(reproj_error, dtype=np.float32).reshape(-1)
    hand_bbox_xyxy = (
        None
        if hand_bbox_xyxy is None
        else np.asarray(hand_bbox_xyxy, dtype=np.float32).reshape(-1, 4)
    )
    joints_3d_raw = None if joints_3d_raw is None else np.asarray(joints_3d_raw, dtype=np.float32)
    frame_count = hand_pose_raw.shape[0]
    bbox_diag = np.full(frame_count, np.nan, dtype=np.float32)
    if hand_bbox_xyxy is not None:
        bbox_diag = np.linalg.norm(
            hand_bbox_xyxy[:, 2:] - hand_bbox_xyxy[:, :2],
            axis=1,
        ).astype(np.float32)
    bbox_valid = np.isfinite(bbox_diag) & (bbox_diag > 1.0)
    relative_error = reproj_error / np.clip(bbox_diag, 1.0, None)

    if mode == "raw":
        stats = {
            f"{side}_hand_valid_raw": valid_raw,
            f"{side}_hand_reproj_error": reproj_error,
            f"{side}_hand_reproj_error_relative": relative_error,
            f"{side}_hand_bbox_diag": bbox_diag,
            f"{side}_hand_bad_mask": ~valid_raw,
            f"{side}_hand_spike_mask": np.zeros(frame_count, dtype=bool),
        }
        return hand_pose_raw, hand_global_raw, joints_3d_raw, valid_raw, stats
    if mode != "clean":
        raise ValueError(f"Unsupported hand_refine_mode={mode!r}")

    reliable = valid_raw.copy()
    finite_error = np.isfinite(reproj_error)
    if reproj_error_ratio_thr > 0:
        ratio_bad = finite_error & bbox_valid & (
            relative_error > float(reproj_error_ratio_thr)
        )
        # Some old/corrupt sidecars do not contain a usable bbox.  Keep the
        # absolute-pixel rule only as a fallback for those frames.
        pixel_bad = (
            finite_error
            & ~bbox_valid
            & (float(reproj_error_thr) > 0.0)
            & (reproj_error > float(reproj_error_thr))
        )
        reliable &= ~(ratio_bad | pixel_bad)
    elif reproj_error_thr > 0:
        reliable &= ~(finite_error & (reproj_error > float(reproj_error_thr)))

    spike_pose = _pose_spike_mask(
        hand_pose_raw,
        reliable,
        mad_multiplier=spike_mad_multiplier,
        abs_threshold=spike_abs_threshold,
    )
    spike_global = _pose_spike_mask(
        hand_global_raw,
        reliable,
        mad_multiplier=spike_mad_multiplier,
        abs_threshold=spike_abs_threshold,
    )
    spike_mask = _post_process_spike_mask(spike_pose | spike_global, gap_merge=gap_merge, max_burst=max_burst)

    good = reliable & ~spike_mask
    bad_mask = ~good
    hand_pose = _interp_hand_pose(hand_pose_raw, bad_mask)
    hand_global = _interp_rotvec(hand_global_raw, bad_mask)
    joints_3d = _interp_components(joints_3d_raw, bad_mask) if joints_3d_raw is not None else None

    if smooth_window and smooth_window > 0:
        hand_pose = _smooth_components(hand_pose, smooth_window)
        hand_global = _smooth_components(hand_global, smooth_window)
        if joints_3d is not None:
            joints_shape = joints_3d.shape
            joints_3d = _smooth_components(joints_3d.reshape(joints_shape[0], -1), smooth_window).reshape(joints_shape)

    stats = {
        f"{side}_hand_valid_raw": valid_raw,
        f"{side}_hand_reproj_error": reproj_error,
        f"{side}_hand_reproj_error_relative": relative_error,
        f"{side}_hand_bbox_diag": bbox_diag,
        f"{side}_hand_bad_mask": bad_mask,
        f"{side}_hand_spike_mask": spike_mask,
    }
    print(
        f"   {side}_hand refine: raw_valid={int(valid_raw.sum())}, "
        f"good={int(good.sum())}, reproj_bad={int((valid_raw & ~reliable).sum())}, "
        f"spikes={int(spike_mask.sum())}"
    )
    if joints_3d is not None:
        joints_3d = joints_3d.astype(np.float32)
    return hand_pose.astype(np.float32), hand_global.astype(np.float32), joints_3d, good.astype(bool), stats


def _safe_normalize(vec, eps=1e-8):
    vec = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / np.clip(norm, eps, None)


def compute_wrist_rotation_from_joints(joints_3d, is_right):
    """Build the do-as-i-do MANO wrist frame from 21 hand joints."""
    joints_3d = np.asarray(joints_3d, dtype=np.float64)
    frame_count = joints_3d.shape[0]
    if joints_3d.shape[1] < 14:
        return np.repeat(np.eye(3, dtype=np.float64)[None], frame_count, axis=0)

    # Match the Do As I Do implementation in process_dataset.py: the z axis is
    # built from wrist joint 0 toward middle MCP joint 9.
    z_axis = _safe_normalize(joints_3d[:, 9] - joints_3d[:, 0])
    if is_right:
        y_aux = _safe_normalize(joints_3d[:, 5] - joints_3d[:, 13])
    else:
        y_aux = _safe_normalize(joints_3d[:, 13] - joints_3d[:, 5])
    x_axis = _safe_normalize(np.cross(y_aux, z_axis))
    y_axis = _safe_normalize(np.cross(z_axis, x_axis))

    rot = np.stack([x_axis, y_axis, z_axis], axis=-1)
    finite = np.isfinite(rot).all(axis=(1, 2))
    det_ok = np.abs(np.linalg.det(rot) - 1.0) < 0.15
    good = finite & det_ok
    if not np.all(good):
        rot[~good] = np.eye(3, dtype=np.float64)
    return rot


def _mean_quat_xyzw(quat_xyzw):
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    quat_xyzw = quat_xyzw[np.isfinite(quat_xyzw).all(axis=1)]
    if quat_xyzw.shape[0] == 0:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    ref = quat_xyzw[0]
    aligned = quat_xyzw.copy()
    for idx in range(aligned.shape[0]):
        if float(np.dot(aligned[idx], ref)) < 0.0:
            aligned[idx] *= -1.0
    mat = aligned.T @ aligned
    _, vecs = np.linalg.eigh(mat)
    mean = vecs[:, -1]
    if mean[3] < 0.0:
        mean *= -1.0
    return mean / np.clip(np.linalg.norm(mean), 1e-12, None)


def _wrist_frame_offsets(joints_3d, global_orient, is_right, valid):
    if Rotation is None:
        return None, None
    joints_3d = np.asarray(joints_3d, dtype=np.float32)
    global_orient = np.asarray(global_orient, dtype=np.float32).reshape(joints_3d.shape[0], 3)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    finite = np.isfinite(joints_3d).all(axis=(1, 2)) & np.isfinite(global_orient).all(axis=1)
    nonzero = np.linalg.norm(joints_3d.reshape(joints_3d.shape[0], -1), axis=1) > 1e-8
    good = valid & finite & nonzero
    if int(good.sum()) < 2:
        return None, None

    geom = Rotation.from_matrix(compute_wrist_rotation_from_joints(joints_3d, is_right))
    mano = Rotation.from_rotvec(global_orient)
    return mano.inv() * geom, good


def _smooth_quat_xyzw(quat_xyzw, valid, window):
    quat = np.asarray(quat_xyzw, dtype=np.float64).copy()
    valid = np.asarray(valid, dtype=bool)
    for idx in range(1, len(quat)):
        if float(np.dot(quat[idx], quat[idx - 1])) < 0.0:
            quat[idx] *= -1.0
    quat = _interp_components(quat, ~valid)
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if window > 1:
        pad = window // 2
        padded = np.pad(quat, ((pad, pad), (0, 0)), mode="edge")
        kernel = np.ones(window, dtype=np.float64) / float(window)
        quat = np.stack(
            [np.convolve(padded[:, col], kernel, mode="valid") for col in range(4)],
            axis=1,
        )
    return quat / np.clip(np.linalg.norm(quat, axis=1, keepdims=True), 1e-12, None)


def build_palm_orientation(
    side,
    joints_3d,
    global_orient,
    valid,
    wrist_offset_mode="constant",
    wrist_offset_smooth_window=15,
):
    frame_count = global_orient.shape[0]
    identity_quat = np.zeros((frame_count, 4), dtype=np.float32)
    identity_quat[:, 0] = 1.0
    empty = {
        f"{side}_hand_wrist_quat": identity_quat,
        f"{side}_hand_palm_normal": np.zeros((frame_count, 3), dtype=np.float32),
        f"{side}_hand_wrist_frame_valid": np.zeros(frame_count, dtype=bool),
        f"{side}_hand_wrist_frame_source": np.asarray("missing"),
    }
    if joints_3d is None or Rotation is None:
        return empty

    joints_3d = np.asarray(joints_3d, dtype=np.float32)
    global_orient = np.asarray(global_orient, dtype=np.float32).reshape(frame_count, 3)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    is_right = side == "right"
    finite = np.isfinite(joints_3d).all(axis=(1, 2)) & np.isfinite(global_orient).all(axis=1)
    nonzero = np.linalg.norm(joints_3d.reshape(frame_count, -1), axis=1) > 1e-8
    frame_valid = valid & finite & nonzero

    wrist_frame_source = "geom_joints"
    geom_rot = Rotation.from_matrix(compute_wrist_rotation_from_joints(joints_3d, is_right))
    offsets, offset_valid = _wrist_frame_offsets(
        joints_3d,
        global_orient,
        is_right,
        frame_valid,
    )
    if offsets is not None and wrist_offset_mode == "temporal":
        offset_quat = _smooth_quat_xyzw(
            offsets.as_quat(),
            offset_valid,
            wrist_offset_smooth_window,
        )
        wrist_rot = Rotation.from_rotvec(global_orient) * Rotation.from_quat(offset_quat)
        wrist_frame_source = "mano_global_with_temporal_do_as_i_do_offset"
    elif offsets is not None and wrist_offset_mode == "constant":
        offset = Rotation.from_quat(_mean_quat_xyzw(offsets.as_quat()[offset_valid]))
        wrist_rot = Rotation.from_rotvec(global_orient) * offset
        wrist_frame_source = "mano_global_with_constant_do_as_i_do_offset"
    elif wrist_offset_mode == "geom":
        wrist_rot = geom_rot
    else:
        wrist_rot = geom_rot

    quat_xyzw = wrist_rot.as_quat()
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]].astype(np.float32)
    palm_normal = wrist_rot.apply(np.repeat([[1.0, 0.0, 0.0]], frame_count, axis=0)).astype(np.float32)
    return {
        f"{side}_hand_wrist_quat": quat_wxyz,
        f"{side}_hand_palm_normal": palm_normal,
        f"{side}_hand_wrist_frame_valid": frame_valid,
        f"{side}_hand_wrist_frame_source": np.asarray(wrist_frame_source),
    }


def save_body_npz(body_npz, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **body_npz)
    frame_count = body_npz["trans"].shape[0]
    fps = float(np.asarray(body_npz["mocap_frame_rate"]).reshape(-1)[0])
    print(f"-> {output} ({frame_count} frames, {fps:g} fps, body-only)")


def save_smplx_sidecar(
    body_npz,
    mano,
    output,
    person_idx,
    hand_refine_mode="raw",
    hand_reproj_error_thr=75.0,
    hand_reproj_error_ratio_thr=0.45,
    hand_spike_mad_multiplier=8.0,
    hand_spike_abs_threshold=1.2,
    hand_refine_gap_merge=2,
    hand_refine_max_burst=10,
    hand_refine_smooth_window=0,
    hand_backend="unknown",
    hand_wrist_offset_mode="auto",
    hand_wrist_offset_smooth_window=15,
):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    frame_count = body_npz["trans"].shape[0]
    left_hand_pose_raw = hand_pose(mano, "left_hand_pose", frame_count, person_idx)
    right_hand_pose_raw = hand_pose(mano, "right_hand_pose", frame_count, person_idx)
    left_hand_global_raw = hand_pose(mano, "left_hand_global_orient", frame_count, person_idx)[:, :3]
    right_hand_global_raw = hand_pose(mano, "right_hand_global_orient", frame_count, person_idx)[:, :3]
    left_joints_3d_raw = hand_array(mano, "left_hand_joints_3d", frame_count, person_idx, (21, 3))
    right_joints_3d_raw = hand_array(mano, "right_hand_joints_3d", frame_count, person_idx, (21, 3))
    left_valid_raw = hand_valid(mano, "left_hand_valid", frame_count, person_idx)
    right_valid_raw = hand_valid(mano, "right_hand_valid", frame_count, person_idx)
    left_error = hand_error(mano, "left_hand_reproj_error", frame_count, person_idx)
    right_error = hand_error(mano, "right_hand_reproj_error", frame_count, person_idx)
    left_bbox = hand_array(mano, "left_hand_bbox_xyxy", frame_count, person_idx, (4,), np.nan)
    right_bbox = hand_array(mano, "right_hand_bbox_xyxy", frame_count, person_idx, (4,), np.nan)

    left_hand_pose, left_hand_global, left_joints_3d, left_valid, left_stats = refine_hand_track(
        "left",
        left_hand_pose_raw,
        left_hand_global_raw,
        left_valid_raw,
        left_error,
        hand_bbox_xyxy=left_bbox,
        joints_3d_raw=left_joints_3d_raw,
        mode=hand_refine_mode,
        reproj_error_thr=hand_reproj_error_thr,
        reproj_error_ratio_thr=hand_reproj_error_ratio_thr,
        spike_mad_multiplier=hand_spike_mad_multiplier,
        spike_abs_threshold=hand_spike_abs_threshold,
        gap_merge=hand_refine_gap_merge,
        max_burst=hand_refine_max_burst,
        smooth_window=hand_refine_smooth_window,
    )
    right_hand_pose, right_hand_global, right_joints_3d, right_valid, right_stats = refine_hand_track(
        "right",
        right_hand_pose_raw,
        right_hand_global_raw,
        right_valid_raw,
        right_error,
        hand_bbox_xyxy=right_bbox,
        joints_3d_raw=right_joints_3d_raw,
        mode=hand_refine_mode,
        reproj_error_thr=hand_reproj_error_thr,
        reproj_error_ratio_thr=hand_reproj_error_ratio_thr,
        spike_mad_multiplier=hand_spike_mad_multiplier,
        spike_abs_threshold=hand_spike_abs_threshold,
        gap_merge=hand_refine_gap_merge,
        max_burst=hand_refine_max_burst,
        smooth_window=hand_refine_smooth_window,
    )

    resolved_wrist_offset_mode = hand_wrist_offset_mode
    if resolved_wrist_offset_mode == "auto":
        resolved_wrist_offset_mode = "temporal" if hand_backend == "hand4wholepp" else "constant"
    left_wrist_stats = build_palm_orientation(
        "left",
        left_joints_3d,
        left_hand_global,
        left_valid,
        resolved_wrist_offset_mode,
        hand_wrist_offset_smooth_window,
    )
    right_wrist_stats = build_palm_orientation(
        "right",
        right_joints_3d,
        right_hand_global,
        right_valid,
        resolved_wrist_offset_mode,
        hand_wrist_offset_smooth_window,
    )
    left_quality = build_hand_quality(
        mano,
        "left",
        frame_count,
        person_idx,
        left_valid,
        left_error,
        left_bbox,
    )
    right_quality = build_hand_quality(
        mano,
        "right",
        frame_count,
        person_idx,
        right_valid,
        right_error,
        right_bbox,
    )
    left_hysup = optional_hysup_export(mano, "left", frame_count, person_idx)
    right_hysup = optional_hysup_export(mano, "right", frame_count, person_idx)

    jaw_eye = np.zeros((frame_count, 9), dtype=np.float32)
    full_pose_smplx = np.concatenate(
        [
            body_npz["root_orient"],
            body_npz["pose_body"],
            jaw_eye,
            left_hand_pose,
            right_hand_pose,
        ],
        axis=1,
    ).astype(np.float32)

    np.savez(
        output,
        poses=body_npz["poses"],
        full_pose_smplx=full_pose_smplx,
        pose_body=body_npz["pose_body"],
        root_orient=body_npz["root_orient"],
        global_orient=body_npz["root_orient"],
        trans=body_npz["trans"],
        transl=body_npz["trans"],
        trans_original=body_npz["trans_original"],
        betas=body_npz["betas"],
        gender=body_npz["gender"],
        mocap_frame_rate=body_npz["mocap_frame_rate"],
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        left_hand_global_orient=left_hand_global,
        right_hand_global_orient=right_hand_global,
        left_hand_joints_3d=left_joints_3d,
        right_hand_joints_3d=right_joints_3d,
        left_hand_valid=left_valid,
        right_hand_valid=right_valid,
        left_hand_pose_raw=left_hand_pose_raw,
        right_hand_pose_raw=right_hand_pose_raw,
        left_hand_global_orient_raw=left_hand_global_raw,
        right_hand_global_orient_raw=right_hand_global_raw,
        left_hand_joints_3d_raw=left_joints_3d_raw,
        right_hand_joints_3d_raw=right_joints_3d_raw,
        left_hand_bbox_xyxy=left_bbox,
        right_hand_bbox_xyxy=right_bbox,
        hand_backend=np.asarray(hand_backend),
        hand_refine_mode=np.asarray(hand_refine_mode),
        hand_reproj_error_thr=np.asarray(hand_reproj_error_thr, dtype=np.float32),
        hand_reproj_error_ratio_thr=np.asarray(hand_reproj_error_ratio_thr, dtype=np.float32),
        hand_wrist_offset_mode=np.asarray(resolved_wrist_offset_mode),
        hand_wrist_offset_smooth_window=np.asarray(
            hand_wrist_offset_smooth_window,
            dtype=np.int32,
        ),
        **left_stats,
        **right_stats,
        **left_wrist_stats,
        **right_wrist_stats,
        **left_quality,
        **right_quality,
        **left_hysup,
        **right_hysup,
    )
    print(
        f"-> {output} ({frame_count} frames, SMPL-X sidecar, "
        f"left_valid={int(left_valid.sum())}, right_valid={int(right_valid.sum())})"
    )


def convert(args):
    results = torch.load(args.gvhmr_results, map_location="cpu", weights_only=False)
    params = load_smpl_params(results, args.space)
    body_npz = body_npz_from_params(params, args.fps, args.person_idx)
    save_body_npz(body_npz, args.output)

    if args.smplx_output:
        mano = load_mano_params(args.mano_params)
        save_smplx_sidecar(
            body_npz,
            mano,
            args.smplx_output,
            args.person_idx,
            hand_refine_mode=args.hand_refine_mode,
            hand_reproj_error_thr=args.hand_reproj_error_thr,
            hand_reproj_error_ratio_thr=args.hand_reproj_error_ratio_thr,
            hand_spike_mad_multiplier=args.hand_spike_mad_multiplier,
            hand_spike_abs_threshold=args.hand_spike_abs_threshold,
            hand_refine_gap_merge=args.hand_refine_gap_merge,
            hand_refine_max_burst=args.hand_refine_max_burst,
            hand_refine_smooth_window=args.hand_refine_smooth_window,
            hand_backend=args.hand_backend,
            hand_wrist_offset_mode=args.hand_wrist_offset_mode,
            hand_wrist_offset_smooth_window=args.hand_wrist_offset_smooth_window,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gvhmr_results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--person_idx", type=int, default=0)
    parser.add_argument("--space", choices=["global", "incam"], default="global")
    parser.add_argument("--mano_params", default=None)
    parser.add_argument("--smplx_output", default=None)
    parser.add_argument("--hand_refine_mode", choices=["raw", "clean"], default="raw")
    parser.add_argument("--hand_reproj_error_thr", type=float, default=75.0, help="Pixels; <=0 disables reprojection-error rejection")
    parser.add_argument(
        "--hand_reproj_error_ratio_thr",
        type=float,
        default=0.45,
        help="Reprojection error / hand bbox diagonal; <=0 uses only the pixel threshold",
    )
    parser.add_argument("--hand_backend", default="unknown")
    parser.add_argument(
        "--hand_wrist_offset_mode",
        choices=["auto", "constant", "temporal", "geom"],
        default="auto",
    )
    parser.add_argument("--hand_wrist_offset_smooth_window", type=int, default=15)
    parser.add_argument("--hand_spike_mad_multiplier", type=float, default=8.0)
    parser.add_argument("--hand_spike_abs_threshold", type=float, default=1.2)
    parser.add_argument("--hand_refine_gap_merge", type=int, default=2)
    parser.add_argument("--hand_refine_max_burst", type=int, default=10)
    parser.add_argument("--hand_refine_smooth_window", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
