import argparse
import os
import os.path as osp
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot, Slerp
from scipy.signal import savgol_filter

sys.path.append(os.getcwd())

from poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree
from scripts.data_process.convert_zitai_to_phc import apply_world_alignment_to_smpl
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES


def _normalize_gender(gender_value):
    if isinstance(gender_value, np.ndarray):
        gender_value = gender_value.item()
    if isinstance(gender_value, bytes):
        gender_value = gender_value.decode("utf-8")
    if gender_value in [0, "0", "neutral", "n"]:
        return "neutral"
    if gender_value in [1, "1", "male", "m"]:
        return "male"
    if gender_value in [2, "2", "female", "f"]:
        return "female"
    return "neutral"


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _target_times(curr_len, target_len):
    return np.linspace(0.0, 1.0, target_len), np.linspace(0.0, 1.0, curr_len)


def _resample_linear(arr, target_len):
    curr_len = arr.shape[0]
    if curr_len == target_len:
        return arr
    if curr_len <= 1 or target_len <= 1:
        return np.repeat(arr[:1], target_len, axis=0)

    target_t, source_t = _target_times(curr_len, target_len)
    flat = np.asarray(arr, dtype=np.float64).reshape(curr_len, -1)
    out = np.empty((target_len, flat.shape[1]), dtype=np.float64)
    for dim in range(flat.shape[1]):
        out[:, dim] = np.interp(target_t, source_t, flat[:, dim])
    return out.reshape((target_len,) + arr.shape[1:]).astype(arr.dtype, copy=False)


def _resample_rotvec(arr, target_len):
    arr = np.asarray(arr)
    curr_len = arr.shape[0]
    if curr_len == target_len:
        return arr
    if curr_len <= 1 or target_len <= 1:
        return np.repeat(arr[:1], target_len, axis=0)

    original_shape = arr.shape
    flat = arr.reshape(curr_len, -1, 3)
    target_t, source_t = _target_times(curr_len, target_len)
    out = np.empty((target_len, flat.shape[1], 3), dtype=np.float64)
    for joint_idx in range(flat.shape[1]):
        rot = sRot.from_rotvec(flat[:, joint_idx, :])
        out[:, joint_idx, :] = Slerp(source_t, rot)(target_t).as_rotvec()
    return out.reshape((target_len,) + original_shape[1:]).astype(arr.dtype, copy=False)


def _resample_motion_value(key, value, target_len):
    arr = np.asarray(value)
    if arr.shape[0] == target_len:
        return arr
    if key == "poses":
        if arr.ndim == 2 and arr.shape[1] % 3 == 0:
            return _resample_rotvec(arr.reshape(arr.shape[0], -1, 3), target_len).reshape(target_len, arr.shape[1])
        return _resample_rotvec(arr, target_len)
    if key == "root_orient":
        return _resample_rotvec(arr.reshape(arr.shape[0], 1, 3), target_len).reshape(target_len, 3)
    if key == "pose_body":
        return _resample_rotvec(arr.reshape(arr.shape[0], -1, 3), target_len).reshape(target_len, arr.shape[1])
    if np.issubdtype(arr.dtype, np.number):
        return _resample_linear(arr, target_len)
    return np.repeat(arr[:1], target_len, axis=0)


def match_reference_timing(repaired, reference_npz):
    ref = np.load(reference_npz, allow_pickle=True)
    ref_poses = ref["poses"]
    target_len = ref_poses.shape[0]
    target_fps = float(ref["mocap_frame_rate"].item() if np.asarray(ref["mocap_frame_rate"]).shape == () else ref["mocap_frame_rate"])

    matched = {}
    for key, value in repaired.items():
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == repaired["poses"].shape[0]:
            matched[key] = _resample_motion_value(key, value, target_len)
        else:
            matched[key] = value

    matched["mocap_frame_rate"] = np.array(target_fps)
    return matched


def compute_linear_acceleration(
    trans, 
    fps: float, 
    smooth: bool = False,
    smooth_window_size: int = 5,
):
    '''
    计算平移的线速度 & 加速度
    Args:
        - translations: (T, 3)的平移数据
        - fps: float,
        - smooth: 是否对速度和加速度进行时间窗口平滑（Savitzky-Golay滤波）
        - smooth_window_size: 平滑窗口大小，必须为奇数，默认5
    Returns:
        - acceleration: np.ndarray, (T, 3)的线加速度数据，只有在calc_acceleration=True时才计算和返回
    '''
    trans = np.asarray(trans, dtype=np.float64)
    num_frames = int(trans.shape[0])
    if num_frames <= 1:
        return np.zeros(num_frames, dtype=np.float64)

    diff = np.diff(trans, axis=0)
    dist = np.linalg.norm(diff, axis=1)

    vel = np.zeros(num_frames, dtype=np.float64)
    vel[1:] = dist * float(fps)
    vel[0] = vel[1]

    if smooth and len(vel) >= smooth_window_size:
        if smooth_window_size % 2 == 0:
            smooth_window_size += 1  # 保证窗口大小为奇数
        smooth_window_size = min(smooth_window_size, len(vel))
        if smooth_window_size >= 3:
            polyorder = min(2, smooth_window_size - 1)
            vel = savgol_filter(vel, smooth_window_size, polyorder)

    acc = np.zeros(num_frames, dtype=np.float64)
    acc[1:] = np.abs(np.diff(vel) * float(fps))
    acc[0] = acc[1]

    if smooth and len(acc) >= smooth_window_size:
        if smooth_window_size % 2 == 0:
            smooth_window_size += 1
        smooth_window_size = min(smooth_window_size, len(acc))
        if smooth_window_size >= 3:
            polyorder = min(2, smooth_window_size - 1)
            acc = savgol_filter(acc, smooth_window_size, polyorder)
            
    return acc
    # TODO 补充平滑逻辑 smplpipeline/code/core/filter.py

def evaluate_speed_check(repaired, acceleration_threshold=14.7, smooth=True):
    trans = np.asarray(repaired["trans"], dtype=np.float64)
    fps_value = repaired.get("mocap_frame_rate", np.array(30.0))
    fps = float(np.asarray(fps_value).reshape(-1)[0]) if np.asarray(fps_value).ndim > 0 else float(fps_value)

    acceleration = compute_linear_acceleration(trans, fps, smooth=smooth)
    outlier_mask = acceleration > float(acceleration_threshold)
    outlier_indices = np.flatnonzero(outlier_mask).tolist()

    return {
        "ok": len(outlier_indices) == 0,
        "threshold": float(acceleration_threshold),
        "checked_frames": int(trans.shape[0]),
        "fps": fps,
        "outlier_count": len(outlier_indices),
        "outlier_indices": outlier_indices,
        "max_acceleration": float(acceleration.max()) if acceleration.size else 0.0,
    }


def convert_sequence(data_seq, gravity_axis="neg_z"):
    mujoco_2_smpl = [SMPL_MUJOCO_NAMES.index(q) for q in SMPL_BONE_ORDER_NAMES if q in SMPL_MUJOCO_NAMES]

    body_quat = _to_numpy(data_seq["body_quat"]).astype(np.float32)
    trans = _to_numpy(data_seq["trans"]).astype(np.float32)
    skeleton_tree = SkeletonTree.from_dict(data_seq["skeleton_tree"])
    offset = skeleton_tree.local_translation[0].cpu().numpy().astype(np.float32)

    sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(body_quat),
        torch.from_numpy(trans),
        is_local=True,
    )

    global_rot = sk_state.global_rotation.numpy()
    B, J, _ = global_rot.shape
    pose_quat_global = (
        sRot.from_quat(global_rot.reshape(-1, 4)) * sRot.from_quat([0.5, 0.5, 0.5, 0.5])
    ).as_quat().reshape(B, J, 4)

    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(pose_quat_global.astype(np.float32)),
        torch.from_numpy(trans),
        is_local=False,
    )

    local_rot = new_sk_state.local_rotation.numpy()
    pose_aa_mujoco = sRot.from_quat(local_rot.reshape(-1, 4)).as_rotvec().reshape(B, J, 3).astype(np.float32)
    pose_aa_smpl = pose_aa_mujoco[:, mujoco_2_smpl, :].astype(np.float32)

    betas_full = _to_numpy(data_seq["betas"]).reshape(-1)
    if betas_full.shape[0] >= 2:
        gender = _normalize_gender(betas_full[0].item())
        betas = betas_full[1:].astype(np.float32)
    else:
        gender = "neutral"
        betas = np.zeros(16, dtype=np.float32)

    root_trans_offset = trans - offset[None, :]
    pose_aa_smpl_flat, root_trans_offset = apply_world_alignment_to_smpl(
        pose_aa_smpl.reshape(B, 72),
        root_trans_offset,
        gravity_axis=gravity_axis,
        inverse=True,
    )
    pose_aa_smpl = pose_aa_smpl_flat.reshape(B, 24, 3)

    return {
        "poses": pose_aa_smpl.astype(np.float32),
        "root_orient": pose_aa_smpl[:, 0, :].astype(np.float32),
        "pose_body": pose_aa_smpl[:, 1:22, :].reshape(B, 63).astype(np.float32),
        "trans": root_trans_offset.astype(np.float32),
        "trans_original": root_trans_offset.astype(np.float32),
        "betas": betas.astype(np.float32),
        "gender": np.array(gender),
        "mocap_frame_rate": np.array(float(data_seq.get("fps", 60.0))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="PHC output states pkl, e.g. output/states/phc_comp_3-xxxx.pkl")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save repaired SMPL npz files")
    parser.add_argument("--only_longest", action="store_true", help="Only export the longest repaired segment.")
    parser.add_argument("--force_fps", type=float, default=None, help="Override mocap_frame_rate in exported npz.")
    parser.add_argument("--reference_npz", type=str, default=None, help="Original SMPL npz whose frame count and fps should be matched.")
    parser.add_argument("--gravity_axis", type=str, default="neg_z", choices=["neg_z", "neg_y"], help="Exported SMPL gravity axis.")
    args = parser.parse_args()

    data = joblib.load(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = list(data.items())
    if args.only_longest and items:
        items = [max(items, key=lambda item: _to_numpy(item[1]["trans"]).shape[0])]

    for key, data_seq in items:
        repaired = convert_sequence(data_seq, gravity_axis=args.gravity_axis)
        if args.reference_npz is not None:
            repaired = match_reference_timing(repaired, args.reference_npz)
        if args.force_fps is not None:
            repaired["mocap_frame_rate"] = np.array(float(args.force_fps))
        out_file = output_dir / f"{key}.npz"
        np.savez_compressed(out_file, **repaired)
        print(f"Saved {out_file}")


if __name__ == "__main__":
    main()
