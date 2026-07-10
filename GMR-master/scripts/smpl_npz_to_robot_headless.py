import argparse
import pathlib
import pickle
from typing import Dict

import mujoco as mj
import numpy as np
import smplx
import torch
from rich import print
from scipy.spatial.transform import Rotation, Slerp
from scipy.signal import savgol_filter
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.hand_retarget import (
    apply_finger_angles_to_dof_pos,
    clamp_angles,
    compute_finger_angles_from_raw,
    get_hand_spec,
    has_hand_joints,
    prepare_hand_poses,
)
from general_motion_retargeting.hand_wrist_utils import (
    compute_wrist_frame_offset,
    compute_wrist_rotation_from_joints,
)
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting.utils.smpl import (
    get_gvhmr_data_offline_fast,
    get_smplx_data_offline_fast,
    load_hand_metadata_from_sidecar,
    load_hand_poses_from_sidecar,
)


HERE = pathlib.Path(__file__).parent


def scalar(value, default=30.0):
    if value is None:
        return default
    arr = np.asarray(value).reshape(-1)
    return float(arr[0]) if arr.size else default


def normalize_gender(value):
    gender = str(np.asarray(value).item()).lower() if np.asarray(value).shape == () else str(value).lower()
    return gender if gender in {"male", "female", "neutral"} else "neutral"


def normalize_betas(betas, model_type):
    betas = np.asarray(betas, dtype=np.float32)
    if betas.ndim > 1:
        betas = betas.mean(axis=0)
    betas = betas.reshape(-1)
    target = 10 if model_type in {"smpl", "smplh"} else 16
    if betas.shape[0] < target:
        betas = np.pad(betas, (0, target - betas.shape[0]))
    return betas[:target].astype(np.float32)


def fallback_human_height_from_betas(betas):
    return float(1.66 + 0.1 * float(np.asarray(betas).reshape(-1)[0]))


def estimate_human_height_from_shape(body_model, betas, model_type):
    """Estimate SMPL body height from a rest-pose mesh instead of betas[0]."""
    fallback = fallback_human_height_from_betas(betas)
    try:
        betas_tensor = torch.tensor(betas, dtype=torch.float32).view(1, -1)
        kwargs = {
            "betas": betas_tensor,
            "global_orient": torch.zeros(1, 3).float(),
            "body_pose": torch.zeros(1, 63).float(),
            "transl": torch.zeros(1, 3).float(),
            "return_full_pose": False,
        }
        if model_type in {"smplh", "smplx"}:
            kwargs["left_hand_pose"] = torch.zeros(1, 45).float()
            kwargs["right_hand_pose"] = torch.zeros(1, 45).float()
        if model_type == "smplx":
            kwargs["jaw_pose"] = torch.zeros(1, 3).float()
            kwargs["leye_pose"] = torch.zeros(1, 3).float()
            kwargs["reye_pose"] = torch.zeros(1, 3).float()
            kwargs["expression"] = torch.zeros(1, 10).float()

        with torch.no_grad():
            output = body_model(**kwargs)
        points = output.vertices if hasattr(output, "vertices") else output.joints
        points_np = points.detach().cpu().numpy().reshape(-1, 3)
        points_np = points_np[np.isfinite(points_np).all(axis=1)]
        if points_np.size == 0:
            return fallback, "betas_fallback_empty_shape"
        height = float(np.max(points_np[:, 1]) - np.min(points_np[:, 1]))
        if 1.0 <= height <= 2.4:
            return height, "rest_mesh_y_extent"
        return fallback, f"betas_fallback_unreasonable_shape_{height:.3f}"
    except Exception as exc:
        return fallback, f"betas_fallback_error_{type(exc).__name__}"


def fit_hand_pose_to_frames(hand_pose, num_frames, target_dim):
    if hand_pose is None:
        return torch.zeros(num_frames, target_dim).float()
    pose = hand_pose.float()
    if pose.shape[0] == num_frames:
        return pose
    if pose.shape[0] <= 0:
        return torch.zeros(num_frames, target_dim).float()
    if pose.shape[0] == 1:
        return pose[:1].repeat(num_frames, 1)

    pose_np = pose.detach().cpu().numpy().astype(np.float32)
    src_t = np.linspace(0.0, 1.0, pose_np.shape[0], dtype=np.float32)
    dst_t = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    out = np.empty((num_frames, pose_np.shape[1]), dtype=np.float32)
    for col in range(pose_np.shape[1]):
        out[:, col] = np.interp(dst_t, src_t, pose_np[:, col])
    return torch.from_numpy(out).float()


def geom_lower_z(model, data, geom_id):
    geom_type = int(model.geom_type[geom_id])
    pos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    mat = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)

    if geom_type == int(mj.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[geom_id])
        vert_start = int(model.mesh_vertadr[mesh_id])
        vert_count = int(model.mesh_vertnum[mesh_id])
        verts = np.asarray(model.mesh_vert[vert_start : vert_start + vert_count], dtype=np.float64)
        world_verts = pos + verts @ mat.T
        return float(np.min(world_verts[:, 2]))

    if geom_type == int(mj.mjtGeom.mjGEOM_SPHERE):
        return float(pos[2] - size[0])

    if geom_type == int(mj.mjtGeom.mjGEOM_BOX):
        corners = np.asarray(
            [
                [sx, sy, sz]
                for sx in (-size[0], size[0])
                for sy in (-size[1], size[1])
                for sz in (-size[2], size[2])
            ],
            dtype=np.float64,
        )
        world_corners = pos + corners @ mat.T
        return float(np.min(world_corners[:, 2]))

    if geom_type in (int(mj.mjtGeom.mjGEOM_CAPSULE), int(mj.mjtGeom.mjGEOM_CYLINDER)):
        radius = float(size[0])
        half_length = float(size[1])
        axis = mat[:, 2]
        radial_drop = radius * np.sqrt(max(0.0, 1.0 - float(axis[2]) ** 2))
        return float(pos[2] - abs(float(axis[2])) * half_length - radial_drop)

    if geom_type == int(mj.mjtGeom.mjGEOM_ELLIPSOID):
        return float(pos[2] - np.linalg.norm(mat[2, :] * size))

    return float(pos[2] - float(model.geom_rbound[geom_id]))


def _geom_selected_for_floor(model, geom_id, selector="all"):
    selector = str(selector or "all").lower()
    if selector == "all":
        return True
    geom_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom_id) or ""
    body_name = mj.mj_id2name(
        model,
        mj.mjtObj.mjOBJ_BODY,
        int(model.geom_bodyid[geom_id]),
    ) or ""
    text = f"{geom_name} {body_name}".lower()
    if selector == "foot":
        return any(token in text for token in ("foot", "toe", "ankle"))
    raise ValueError(f"Unsupported floor geometry selector={selector!r}")


def robot_geom_lowest_heights(xml_file, root_pos, root_rot_xyzw, dof_pos, selector="all"):
    model = mj.MjModel.from_xml_path(str(xml_file))
    data = mj.MjData(model)
    lowest_heights = []
    selected_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) != 0
        and _geom_selected_for_floor(model, geom_id, selector=selector)
    ]
    if not selected_geom_ids and str(selector or "all").lower() != "all":
        selected_geom_ids = [
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) != 0
        ]

    for frame_idx in range(root_pos.shape[0]):
        data.qpos[:3] = root_pos[frame_idx]
        data.qpos[3:7] = root_rot_xyzw[frame_idx][[3, 0, 1, 2]]
        data.qpos[7:] = dof_pos[frame_idx]
        mj.mj_forward(model, data)

        frame_lowest = np.inf
        for geom_id in selected_geom_ids:
            frame_lowest = min(frame_lowest, geom_lower_z(model, data, geom_id))
        lowest_heights.append(frame_lowest)

    return np.asarray(lowest_heights, dtype=np.float32)


def resolve_body_model_root(body_model_path, model_type):
    path = pathlib.Path(body_model_path).expanduser()
    if (path / model_type).exists():
        return str(path)
    if path.name.lower() == model_type and path.exists():
        return str(path.parent)
    return str(path)


def load_body_motion(npz_path, body_model_path, model_type, hand_npz=None):
    if pathlib.Path(npz_path).suffix.lower() == ".pkl":
        return load_hmr2_pkl_motion(npz_path, body_model_path, model_type, hand_npz)
    return load_npz_body_motion(npz_path, body_model_path, model_type, hand_npz)


def load_npz_body_motion(npz_path, body_model_path, model_type, hand_npz=None):
    with np.load(npz_path, allow_pickle=True) as data:
        pose_body = np.asarray(data["pose_body"], dtype=np.float32)
        root_orient = np.asarray(data["root_orient"], dtype=np.float32)
        trans = np.asarray(data["trans"], dtype=np.float32)
        gender = normalize_gender(data["gender"] if "gender" in data.files else "neutral")
        betas = normalize_betas(data["betas"], model_type)
        fps = scalar(data["mocap_frame_rate"] if "mocap_frame_rate" in data.files else None, 30.0)

    body_model = smplx.create(
        resolve_body_model_root(body_model_path, model_type),
        model_type,
        gender=gender,
        use_pca=False,
        num_betas=int(betas.shape[0]),
    )

    num_frames = pose_body.shape[0]
    betas_tensor = torch.tensor(betas).float().view(1, -1).repeat(num_frames, 1)

    # Load hand poses from sidecar if provided, otherwise use zeros
    left_hand, right_hand = None, None
    if hand_npz is not None:
        left_hand, right_hand, _, _ = load_hand_poses_from_sidecar(hand_npz)

    left_hand_dim = 6 if getattr(body_model, "use_pca", False) else 45
    right_hand_dim = 6 if getattr(body_model, "use_pca", False) else 45

    left_hand_pose = fit_hand_pose_to_frames(left_hand, num_frames, left_hand_dim)
    right_hand_pose = fit_hand_pose_to_frames(right_hand, num_frames, right_hand_dim)

    kwargs = {
        "betas": betas_tensor,
        "global_orient": torch.tensor(root_orient).float(),
        "body_pose": torch.tensor(pose_body).float(),
        "transl": torch.tensor(trans).float(),
        "return_full_pose": True,
    }
    if model_type in {"smplh", "smplx"}:
        kwargs["left_hand_pose"] = left_hand_pose
        kwargs["right_hand_pose"] = right_hand_pose
    if model_type == "smplx":
        kwargs["jaw_pose"] = torch.zeros(num_frames, 3).float()
        kwargs["leye_pose"] = torch.zeros(num_frames, 3).float()
        kwargs["reye_pose"] = torch.zeros(num_frames, 3).float()
        kwargs["expression"] = torch.zeros(num_frames, 10).float()

    smpl_output = body_model(**kwargs)
    motion_data = {
        "pose_body": pose_body,
        "root_orient": root_orient,
        "trans": trans,
        "betas": betas,
        "gender": np.array(gender),
        "mocap_frame_rate": np.array(fps),
    }
    human_height, human_height_source = estimate_human_height_from_shape(
        body_model,
        betas,
        model_type,
    )
    motion_data["human_height_source"] = np.array(human_height_source)
    return motion_data, body_model, smpl_output, human_height


def load_hmr2_pkl_motion(pkl_path, body_model_path, model_type, hand_npz=None):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    pose_body = np.asarray(data["body_pose"], dtype=np.float32)
    root_orient = np.asarray(data["global_orient"], dtype=np.float32)
    trans = np.asarray(data["transl"], dtype=np.float32)
    gender = normalize_gender(data.get("gender", "neutral"))
    betas = normalize_betas(data["betas"], model_type)
    fps = scalar(data.get("fps", data.get("mocap_frame_rate", 30.0)), 30.0)

    body_model = smplx.create(
        resolve_body_model_root(body_model_path, model_type),
        model_type,
        gender=gender,
        use_pca=False,
        num_betas=int(betas.shape[0]),
    )

    num_frames = pose_body.shape[0]
    betas_tensor = torch.tensor(betas).float().view(1, -1).repeat(num_frames, 1)

    # Load hand poses from sidecar if provided, otherwise use zeros
    left_hand, right_hand = None, None
    if hand_npz is not None:
        left_hand, right_hand, _, _ = load_hand_poses_from_sidecar(hand_npz)

    left_hand_dim = 6 if getattr(body_model, "use_pca", False) else 45
    right_hand_dim = 6 if getattr(body_model, "use_pca", False) else 45

    left_hand_pose = fit_hand_pose_to_frames(left_hand, num_frames, left_hand_dim)
    right_hand_pose = fit_hand_pose_to_frames(right_hand, num_frames, right_hand_dim)

    kwargs = {
        "betas": betas_tensor,
        "global_orient": torch.tensor(root_orient).float(),
        "body_pose": torch.tensor(pose_body).float(),
        "transl": torch.tensor(trans).float(),
        "return_full_pose": True,
    }
    if model_type in {"smplh", "smplx"}:
        kwargs["left_hand_pose"] = left_hand_pose
        kwargs["right_hand_pose"] = right_hand_pose
    if model_type == "smplx":
        kwargs["jaw_pose"] = torch.zeros(num_frames, 3).float()
        kwargs["leye_pose"] = torch.zeros(num_frames, 3).float()
        kwargs["reye_pose"] = torch.zeros(num_frames, 3).float()
        kwargs["expression"] = torch.zeros(num_frames, 10).float()

    smpl_output = body_model(**kwargs)
    motion_data = {
        "pose_body": pose_body,
        "root_orient": root_orient,
        "trans": trans,
        "betas": betas,
        "gender": np.array(gender),
        "mocap_frame_rate": np.array(fps),
    }
    human_height, human_height_source = estimate_human_height_from_shape(
        body_model,
        betas,
        model_type,
    )
    motion_data["human_height_source"] = np.array(human_height_source)
    return motion_data, body_model, smpl_output, human_height


def output_path_for(src_file, src_root, tgt_root, output_name=None):
    rel = src_file.relative_to(src_root)
    if output_name:
        return tgt_root / rel.parent / output_name
    return (tgt_root / rel).with_suffix(".pkl")


def valid_smooth_window(num_frames, requested_window, polyorder):
    window = int(requested_window)
    if window <= 0 or num_frames < 3:
        return None
    if window % 2 == 0:
        window += 1
    window = min(window, num_frames if num_frames % 2 == 1 else num_frames - 1)
    if window <= int(polyorder) or window < 3:
        return None
    return window


def smooth_array(values, window, polyorder):
    smooth_window = valid_smooth_window(values.shape[0], window, polyorder)
    if smooth_window is None:
        return values
    return savgol_filter(values, smooth_window, int(polyorder), axis=0, mode="interp")


def ensure_quat_continuity(quat):
    quat = quat.copy()
    for idx in range(1, quat.shape[0]):
        if float(np.dot(quat[idx], quat[idx - 1])) < 0.0:
            quat[idx] *= -1.0
    return quat


def smooth_quat(quat, window, polyorder):
    smooth_window = valid_smooth_window(quat.shape[0], window, polyorder)
    if smooth_window is None:
        return quat
    quat = ensure_quat_continuity(quat)
    quat = savgol_filter(quat, smooth_window, int(polyorder), axis=0, mode="interp")
    quat /= np.clip(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-12, None)
    return quat


def smooth_qpos(qpos_list, window, polyorder):
    if int(window) <= 0:
        return qpos_list
    qpos = qpos_list.copy()
    qpos[:, :3] = smooth_array(qpos[:, :3], window, polyorder)
    qpos[:, 3:7] = smooth_quat(qpos[:, 3:7], window, polyorder)
    qpos[:, 7:] = smooth_array(qpos[:, 7:], window, polyorder)
    return qpos


def median_filter_1d(values, window):
    window = int(window)
    if window <= 1:
        return values
    if window % 2 == 0:
        window += 1
    if values.shape[0] < window:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.asarray(
        [np.median(padded[i : i + window]) for i in range(values.shape[0])],
        dtype=np.float32,
    )


def limit_frame_delta(values, max_delta):
    max_delta = float(max_delta)
    if max_delta <= 0.0 or values.shape[0] < 2:
        return values
    out = values.astype(np.float32, copy=True)
    for idx in range(1, out.shape[0]):
        delta = float(np.clip(out[idx] - out[idx - 1], -max_delta, max_delta))
        out[idx] = out[idx - 1] + delta
    return out


def apply_deadband(values, deadband):
    deadband = float(deadband)
    if deadband <= 0.0 or values.shape[0] < 2:
        return values
    out = values.astype(np.float32, copy=True)
    for idx in range(1, out.shape[0]):
        if abs(float(out[idx] - out[idx - 1])) < deadband:
            out[idx] = out[idx - 1]
    return out


def smooth_finger_angles(
    finger_angles,
    window,
    polyorder,
    median_window=5,
    max_delta=0.08,
    deadband=0.005,
):
    return {
        name: apply_deadband(
            limit_frame_delta(
                smooth_array(
                    median_filter_1d(np.asarray(values, dtype=np.float32).reshape(-1), median_window).reshape(-1, 1),
                    window,
                    polyorder,
                ).reshape(-1),
                max_delta,
            ),
            deadband,
        ).astype(np.float32)
        for name, values in finger_angles.items()
    }


def resolve_hand_npz_for_source(src_file, args):
    if args.hand_npz:
        hand_path = pathlib.Path(args.hand_npz).expanduser()
        if not hand_path.is_absolute():
            hand_path = pathlib.Path.cwd() / hand_path
        hand_path = hand_path.resolve()
        if not hand_path.is_file():
            raise FileNotFoundError(f"--hand_npz not found: {hand_path}")
        return hand_path

    if not args.auto_hand_npz:
        return None

    # First look beside the source file (follow symlinks to the real dir).
    # Compacted pipeline outputs keep body NPZ files under
    # _intermediate/npz while the hand sidecar stays at the clip root, so
    # include that parent as an explicit fallback.
    real_dir = src_file.resolve().parent
    candidate_dirs = [real_dir]
    if real_dir.name == "npz" and real_dir.parent.name == "_intermediate":
        candidate_dirs.append(real_dir.parent.parent)
    candidate_dirs.append(src_file.parent)

    seen = set()
    for directory in candidate_dirs:
        directory = directory.resolve()
        if directory in seen:
            continue
        seen.add(directory)
        candidate = directory / args.hand_npz_name
        if candidate.is_file():
            return candidate
    return None


def rotate_human_frames(frames, yaw_offset_deg):
    yaw = float(yaw_offset_deg)
    if abs(yaw) < 1e-8:
        return frames
    yaw_rot = Rotation.from_euler("z", yaw, degrees=True)
    rotated = []
    for frame in frames:
        new_frame = {}
        for body_name, (pos, quat_wxyz) in frame.items():
            pos = np.asarray(pos, dtype=np.float64)
            rot = Rotation.from_quat(np.asarray(quat_wxyz, dtype=np.float64), scalar_first=True)
            new_frame[body_name] = (
                yaw_rot.apply(pos),
                (yaw_rot * rot).as_quat(scalar_first=True),
            )
        rotated.append(new_frame)
    return rotated


def normalize_quat_wxyz(quat):
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.clip(norm, 1e-12, None)


def rotation_from_wxyz(quat_wxyz):
    quat_wxyz = normalize_quat_wxyz(quat_wxyz)
    return Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]])


def as_wxyz(rotation):
    quat_xyzw = rotation.as_quat()
    return quat_xyzw[[3, 0, 1, 2]]


def resample_numeric(values, target_count):
    values = np.asarray(values)
    if values.shape[0] == target_count:
        return values
    if values.shape[0] == 0:
        return values
    if values.shape[0] == 1:
        return np.repeat(values, target_count, axis=0)

    flat = values.reshape(values.shape[0], -1).astype(np.float32)
    src_t = np.linspace(0.0, 1.0, flat.shape[0], dtype=np.float32)
    dst_t = np.linspace(0.0, 1.0, target_count, dtype=np.float32)
    out = np.empty((target_count, flat.shape[1]), dtype=np.float32)
    for col in range(flat.shape[1]):
        out[:, col] = np.interp(dst_t, src_t, flat[:, col])
    return out.reshape((target_count, *values.shape[1:]))


def resample_bool(values, target_count):
    values = np.asarray(values, dtype=bool).reshape(-1)
    if values.shape[0] == target_count:
        return values
    if values.shape[0] == 0:
        return np.zeros(target_count, dtype=bool)
    idx = np.rint(np.linspace(0, values.shape[0] - 1, target_count)).astype(np.int64)
    return values[idx]


def resample_quat_wxyz(quat, target_count):
    quat = normalize_quat_wxyz(np.asarray(quat, dtype=np.float64))
    if quat.shape[0] == target_count:
        return quat.astype(np.float32)
    if quat.shape[0] == 0:
        out = np.zeros((target_count, 4), dtype=np.float32)
        out[:, 0] = 1.0
        return out
    if quat.shape[0] == 1:
        return np.repeat(quat.astype(np.float32), target_count, axis=0)

    quat = ensure_quat_continuity(quat.astype(np.float32)).astype(np.float64)
    src_t = np.linspace(0.0, 1.0, quat.shape[0], dtype=np.float64)
    dst_t = np.linspace(0.0, 1.0, target_count, dtype=np.float64)
    rots = Rotation.from_quat(quat[:, [1, 2, 3, 0]])
    return np.asarray([as_wxyz(rot) for rot in Slerp(src_t, rots)(dst_t)], dtype=np.float32)


def fit_hand_metadata_to_frames(metadata, frame_count):
    fitted = {}
    for key, value in metadata.items():
        array = np.asarray(value)
        if array.ndim == 0 or array.dtype.kind in {"U", "S", "O"}:
            fitted[key] = value
        elif key.endswith("_wrist_quat"):
            fitted[key] = resample_quat_wxyz(array, frame_count)
        elif key.endswith("_valid"):
            fitted[key] = resample_bool(array, frame_count)
        elif array.shape[0] == frame_count:
            fitted[key] = array
        else:
            fitted[key] = resample_numeric(array, frame_count)
    return fitted


def apply_hand_wrist_orientation_override(frames, hand_metadata):
    debug = {}
    if not hand_metadata:
        return debug

    for side in ("left", "right"):
        joint_name = f"{side}_wrist"
        quat_key = f"{side}_hand_wrist_quat"
        valid_key = f"{side}_hand_wrist_frame_valid"
        if quat_key not in hand_metadata or not frames or joint_name not in frames[0]:
            debug[f"{side}_hand_wrist_orientation_frames"] = 0
            continue

        hand_quat = normalize_quat_wxyz(np.asarray(hand_metadata[quat_key], dtype=np.float64))
        valid = (
            np.asarray(hand_metadata[valid_key], dtype=bool).reshape(-1)
            if valid_key in hand_metadata
            else np.ones(hand_quat.shape[0], dtype=bool)
        )
        frame_count = min(len(frames), hand_quat.shape[0], valid.shape[0])
        anchor_candidates = [idx for idx in range(frame_count) if valid[idx] and joint_name in frames[idx]]
        if not anchor_candidates:
            debug[f"{side}_hand_wrist_orientation_frames"] = 0
            continue

        anchor = anchor_candidates[0]
        smpl_anchor = Rotation.from_quat(np.asarray(frames[anchor][joint_name][1], dtype=np.float64), scalar_first=True)
        hand_anchor = rotation_from_wxyz(hand_quat[anchor])
        align = smpl_anchor * hand_anchor.inv()

        applied = 0
        for idx in range(frame_count):
            if not valid[idx] or joint_name not in frames[idx]:
                continue
            pos, _ = frames[idx][joint_name]
            target = align * rotation_from_wxyz(hand_quat[idx])
            frames[idx][joint_name] = (pos, target.as_quat(scalar_first=True))
            applied += 1
        debug[f"{side}_hand_wrist_orientation_frames"] = applied
    return debug


# GVHMR → Z-up coordinate transform (same as in utils/smpl.py get_gvhmr_data_offline_fast)
_R_GVHMR_TO_ZUP = Rotation.from_matrix([
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0],
])


def apply_geometric_wrist_orientation(
    frames,
    smplx_joints,
    joint_names,
    hand_metadata,
    model_type="smplx",
    coord_transform="gvhmr",
    human_yaw_offset_deg=0.0,
):
    """Override wrist orientations with the Do As I Do MCP-frame method.

    The previous experimental path wrote the raw MCP geometric frame into the
    SMPL wrist target and then applied a robot-specific fixed roll.  That makes
    G1 decompose a palm-frame mismatch into wrist pitch/yaw, which is exactly
    where the visual artifacts show up.  The Do As I Do pipeline instead
    estimates a constant per-hand offset between MANO global wrist orientation
    and the MCP frame, then applies ``R_mano * R_offset`` every frame.

    Parameters
    ----------
    frames : list of dict
        IK-target frames (already in Z-up).  Modified in-place.
    smplx_joints : (T, J, 3) ndarray or None
        SMPL joint positions in the original camera (Y-up) frame.
    joint_names : list of str or None
    hand_metadata : dict
        May contain ``{side}_hand_global_orient`` — used ONLY as fallback
        when *smplx_joints* / *joint_names* are unavailable.
    model_type : str
    coord_transform : str
        ``"gvhmr"`` applies GVHMR Y-up → Z-up to the geometric frame.
    human_yaw_offset_deg : float
        Same yaw offset already applied to ``frames``.

    Returns debug dict with per-side frame counts.
    """
    debug = {}
    if model_type not in {"smplh", "smplx"}:
        return debug

    need_coord_xform = (coord_transform == "gvhmr")
    world_rot = Rotation.identity()
    if need_coord_xform:
        world_rot = _R_GVHMR_TO_ZUP * world_rot
    yaw = float(human_yaw_offset_deg)
    if abs(yaw) > 1e-8:
        world_rot = Rotation.from_euler("z", yaw, degrees=True) * world_rot

    can_do_geometric = (
        smplx_joints is not None
        and joint_names is not None
        and smplx_joints.shape[0] > 0
    )

    for side in ("left", "right"):
        joint_name = f"{side}_wrist"
        if frames is None or not frames:
            debug[f"{side}_geometric_wrist_frames"] = 0
            continue
        if joint_name not in frames[0]:
            debug[f"{side}_geometric_wrist_frames"] = 0
            continue

        is_right = (side == "right")
        frame_count = min(len(frames), smplx_joints.shape[0] if can_do_geometric else 0)
        orient_key = f"{side}_hand_global_orient"
        valid_key = f"{side}_hand_valid"
        wrist_valid_key = f"{side}_hand_wrist_frame_valid"

        if can_do_geometric and frame_count >= 2 and hand_metadata is not None and orient_key in hand_metadata:
            mano_rotvec = np.asarray(hand_metadata[orient_key], dtype=np.float64)
            frame_count = min(frame_count, mano_rotvec.shape[0])
            if frame_count < 2:
                debug[f"{side}_geometric_wrist_frames"] = 0
                continue

            valid = np.ones(frame_count, dtype=bool)
            if valid_key in hand_metadata:
                valid &= np.asarray(hand_metadata[valid_key], dtype=bool).reshape(-1)[:frame_count]
            if wrist_valid_key in hand_metadata:
                valid &= np.asarray(hand_metadata[wrist_valid_key], dtype=bool).reshape(-1)[:frame_count]
            valid_indices = np.flatnonzero(valid)
            if valid_indices.size < 2:
                debug[f"{side}_geometric_wrist_frames"] = 0
                continue

            R_offset = compute_wrist_frame_offset(
                smplx_joints[:frame_count][valid_indices],
                mano_rotvec[:frame_count][valid_indices],
                is_right=is_right,
                joint_names=joint_names,
                joint_format="smpl",
            )
            R_mano = Rotation.from_rotvec(mano_rotvec[:frame_count])
            R_target = world_rot * (R_mano * R_offset)

            applied = 0
            for idx in range(frame_count):
                if not valid[idx] or joint_name not in frames[idx]:
                    continue
                pos, _ = frames[idx][joint_name]
                frames[idx][joint_name] = (pos, R_target[idx].as_quat(scalar_first=True))
                applied += 1

            debug[f"{side}_geometric_wrist_frames"] = applied

        elif hand_metadata is not None:
            # ---- Fallback: MANO global_orient → quaternion override -------
            if orient_key not in hand_metadata:
                debug[f"{side}_geometric_wrist_frames"] = 0
                continue

            mano_rotvec = np.asarray(hand_metadata[orient_key], dtype=np.float64)
            fc = min(len(frames), mano_rotvec.shape[0])
            if fc < 2:
                debug[f"{side}_geometric_wrist_frames"] = 0
                continue

            R_mano = Rotation.from_rotvec(mano_rotvec[:fc])
            R_mano = world_rot * R_mano

            # Fallback alignment at frame 0
            smpl_anchor = Rotation.from_quat(
                np.asarray(frames[0][joint_name][1], dtype=np.float64),
                scalar_first=True,
            )
            align = smpl_anchor * R_mano[0].inv()

            applied = 0
            for idx in range(fc):
                if joint_name not in frames[idx]:
                    continue
                pos, _ = frames[idx][joint_name]
                target = align * R_mano[idx]
                frames[idx][joint_name] = (pos, target.as_quat(scalar_first=True))
                applied += 1

            debug[f"{side}_geometric_wrist_frames"] = applied
        else:
            debug[f"{side}_geometric_wrist_frames"] = 0

    return debug


def relax_orientation_tasks(retargeter, body_names):
    names = [name.strip() for name in str(body_names).split(",") if name.strip()]
    if not names:
        return []
    relaxed = []
    for task_map in (retargeter.human_body_to_task1, retargeter.human_body_to_task2):
        for body_name, task in task_map.items():
            if body_name in names:
                task.set_orientation_cost(0.0)
                relaxed.append(body_name)
    return sorted(set(relaxed))


def robot_joint_qpos_indices(model) -> Dict[str, int]:
    indices = {}
    for joint_id in range(model.njnt):
        joint_type = model.jnt_type[joint_id]
        if joint_type not in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            continue
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id)
        if not name:
            continue
        indices[name] = int(model.jnt_qposadr[joint_id])
    return indices


def clamp_dof_pos_to_joint_ranges(dof_pos, model):
    out = np.asarray(dof_pos, dtype=np.float32).copy()
    for joint_id in range(model.njnt):
        if not bool(model.jnt_limited[joint_id]):
            continue
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in (int(mj.mjtJoint.mjJNT_HINGE), int(mj.mjtJoint.mjJNT_SLIDE)):
            continue
        qpos_adr = int(model.jnt_qposadr[joint_id])
        dof_idx = qpos_adr - 7
        if dof_idx < 0 or dof_idx >= out.shape[1]:
            continue
        low, high = np.asarray(model.jnt_range[joint_id], dtype=np.float32)
        out[:, dof_idx] = np.clip(out[:, dof_idx], low, high)
    return out


def lower_body_joint_names(model):
    names = []
    for name in robot_joint_qpos_indices(model):
        lower = name.lower()
        if any(token in lower for token in ("hip", "knee", "ankle", "toe", "foot", "leg")):
            names.append(name)
    return sorted(names)


def apply_retarget_mode(qpos_list, retargeter, args):
    mode = str(args.retarget_mode).lower()
    if mode == "full":
        return qpos_list, {}

    qpos = qpos_list.copy()
    debug = {"retarget_mode": mode}
    if mode != "upper_body":
        raise ValueError(f"Unsupported retarget_mode={args.retarget_mode!r}")

    rest_qpos = qpos[0].copy()
    joint_indices = robot_joint_qpos_indices(retargeter.model)
    locked_names = lower_body_joint_names(retargeter.model)
    locked_qpos_indices = [joint_indices[name] for name in locked_names if joint_indices[name] < qpos.shape[1]]
    if locked_qpos_indices:
        qpos[:, locked_qpos_indices] = rest_qpos[locked_qpos_indices]

    if args.upper_body_root_mode == "fixed":
        qpos[:, :7] = rest_qpos[:7]
    elif args.upper_body_root_mode == "fixed_xy":
        qpos[:, :2] = rest_qpos[:2]
    elif args.upper_body_root_mode != "free":
        raise ValueError(f"Unsupported upper_body_root_mode={args.upper_body_root_mode!r}")

    debug["locked_lower_body_joints"] = locked_names
    debug["upper_body_root_mode"] = args.upper_body_root_mode
    return qpos, debug


WRIST_ORIENTATION_OVERRIDE_ROBOTS = {"unitree_h1_with_hand_wrist"}
PALM_ROLL_ROBOTS = {
    "unitree_g1",
    "unitree_g1_with_hands",
    "unitree_h1_with_hand",
    "unitree_h1_with_hand_wrist",
}
WRIST_PITCH_YAW_STABILIZE_ROBOTS = {"unitree_g1", "unitree_g1_with_hands"}
WRIST_PITCH_YAW_SOFT_LIMIT_ROBOTS = {"unitree_g1", "unitree_g1_with_hands"}


def wrist_orientation_override_enabled(args):
    return (
        args.hand_wrist_orientation_mode == "override_frames"
        and (
            args.robot in WRIST_ORIENTATION_OVERRIDE_ROBOTS
            or bool(getattr(args, "force_hand_wrist_orientation_override", False))
        )
    )


def palm_roll_retarget_enabled(args, hand_npz):
    mode = str(args.palm_roll_mode).lower()
    if mode == "off":
        return False
    if mode == "on":
        return hand_npz is not None
    if mode == "auto":
        return hand_npz is not None and args.robot in PALM_ROLL_ROBOTS and not wrist_orientation_override_enabled(args)
    raise ValueError(f"Unsupported palm_roll_mode={args.palm_roll_mode!r}")


def wrist_pitch_yaw_stabilization_enabled(args):
    mode = str(args.wrist_pitch_yaw_stabilize).lower()
    if mode == "off":
        return False
    if mode == "on":
        return True
    if mode == "soft_limit":
        return False
    if mode == "auto":
        return (
            args.robot in WRIST_PITCH_YAW_STABILIZE_ROBOTS
            and args.robot not in WRIST_PITCH_YAW_SOFT_LIMIT_ROBOTS
            and not wrist_orientation_override_enabled(args)
        )
    raise ValueError(f"Unsupported wrist_pitch_yaw_stabilize={args.wrist_pitch_yaw_stabilize!r}")


def wrist_pitch_yaw_soft_limit_enabled(args):
    mode = str(args.wrist_pitch_yaw_stabilize).lower()
    if mode == "soft_limit":
        return True
    if mode == "auto":
        return args.robot in WRIST_PITCH_YAW_SOFT_LIMIT_ROBOTS
    if mode in {"off", "on"}:
        return False
    raise ValueError(f"Unsupported wrist_pitch_yaw_stabilize={args.wrist_pitch_yaw_stabilize!r}")


def resolve_smpl_joint_names(body_model, smplx_joints, model_type):
    if smplx_joints is None:
        return None
    try:
        from smplx.joint_names import JOINT_NAMES as _SMPLX_NAMES
        from smplx.joint_names import SMPLH_JOINT_NAMES as _SMPLH_NAMES
    except ImportError:
        _SMPLX_NAMES = None
        _SMPLH_NAMES = None

    n_joints = smplx_joints.shape[1]
    if model_type == "smplh" and _SMPLH_NAMES is not None:
        return list(_SMPLH_NAMES[:n_joints])
    if model_type == "smplx" and _SMPLX_NAMES is not None:
        return list(_SMPLX_NAMES[:n_joints])
    if hasattr(body_model, "joint_names"):
        return list(body_model.joint_names)
    return None


def smpl_output_joints(smpl_output, frame_count):
    if not hasattr(smpl_output, "joints"):
        return None
    smplx_joints = smpl_output.joints.detach().cpu().numpy()
    if smplx_joints.ndim == 2:
        smplx_joints = smplx_joints.reshape(frame_count, -1, 3)
    return smplx_joints


def world_rotation_for_inputs(coord_transform, human_yaw_offset_deg):
    world_rot = Rotation.identity()
    if coord_transform == "gvhmr":
        world_rot = _R_GVHMR_TO_ZUP * world_rot
    yaw = float(human_yaw_offset_deg)
    if abs(yaw) > 1e-8:
        world_rot = Rotation.from_euler("z", yaw, degrees=True) * world_rot
    return world_rot


def joint_index(joint_names, name):
    try:
        return joint_names.index(name)
    except (AttributeError, ValueError):
        return None


def normalized_vectors(values, eps=1e-8):
    values = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    valid = norm[..., 0] > eps
    out = np.zeros_like(values)
    out[valid] = values[valid] / norm[valid]
    return out, valid


def fill_invalid_numeric(values, valid):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    if not np.any(valid):
        return None
    if np.all(valid):
        return values
    x = np.arange(values.shape[0], dtype=np.float64)
    return np.interp(x, x[valid], values[valid])


def reject_direction_jumps(vectors, valid, min_dot=-0.2):
    """Reject sudden palm-normal jumps while preserving gradual turns."""
    vectors = np.asarray(vectors, dtype=np.float64).copy()
    valid = np.asarray(valid, dtype=bool).reshape(-1).copy()
    rejected = 0
    last = None
    threshold = float(min_dot)
    for idx in range(vectors.shape[0]):
        if not valid[idx]:
            continue
        if last is None:
            last = vectors[idx].copy()
            continue
        dot = float(np.dot(vectors[idx], last))
        if dot < threshold:
            valid[idx] = False
            rejected += 1
            continue
        last = vectors[idx].copy()
    return vectors, valid, rejected


def sidecar_palm_normal(hand_metadata, side, frame_count, world_rot, normal_dot_min=-0.2):
    key = f"{side}_hand_palm_normal"
    if hand_metadata is None or key not in hand_metadata:
        return None, None, 0

    normal = np.asarray(hand_metadata[key], dtype=np.float64)
    if normal.ndim != 2 or normal.shape[1] != 3:
        return None, None, 0
    normal = normal[:frame_count]
    normal = world_rot.apply(normal)
    normal, normal_valid = normalized_vectors(normal)

    count = normal.shape[0]
    valid = normal_valid.copy()
    valid_key = f"{side}_hand_valid"
    wrist_valid_key = f"{side}_hand_wrist_frame_valid"
    if valid_key in hand_metadata:
        valid &= np.asarray(hand_metadata[valid_key], dtype=bool).reshape(-1)[:count]
    if wrist_valid_key in hand_metadata:
        valid &= np.asarray(hand_metadata[wrist_valid_key], dtype=bool).reshape(-1)[:count]

    normal, valid, rejected = reject_direction_jumps(
        normal,
        valid,
        min_dot=normal_dot_min,
    )
    return normal, valid, rejected


def sidecar_joint_palm_normal(hand_metadata, side, frame_count, world_rot, normal_dot_min=-0.2):
    key = f"{side}_hand_joints_3d"
    if hand_metadata is None or key not in hand_metadata:
        return None, None, 0

    joints = np.asarray(hand_metadata[key], dtype=np.float64)
    if joints.ndim != 3 or joints.shape[1] < 14 or joints.shape[2] != 3:
        return None, None, 0
    joints = joints[:frame_count]
    count = joints.shape[0]

    finite = np.isfinite(joints).all(axis=(1, 2))
    nonzero = np.linalg.norm(joints.reshape(count, -1), axis=1) > 1e-8
    valid = finite & nonzero
    valid_key = f"{side}_hand_valid"
    wrist_valid_key = f"{side}_hand_wrist_frame_valid"
    if valid_key in hand_metadata:
        valid &= np.asarray(hand_metadata[valid_key], dtype=bool).reshape(-1)[:count]
    if wrist_valid_key in hand_metadata:
        valid &= np.asarray(hand_metadata[wrist_valid_key], dtype=bool).reshape(-1)[:count]

    wrist = joints[:, 0]
    middle_mcp = joints[:, 9]
    index_mcp = joints[:, 5]
    ring_mcp = joints[:, 13]
    z_axis, z_valid = normalized_vectors(middle_mcp - wrist)
    if side == "right":
        y_aux, y_valid = normalized_vectors(index_mcp - ring_mcp)
    else:
        y_aux, y_valid = normalized_vectors(ring_mcp - index_mcp)
    palm_normal, normal_valid = normalized_vectors(np.cross(y_aux, z_axis))

    valid &= z_valid & y_valid & normal_valid
    palm_normal = world_rot.apply(palm_normal)
    palm_normal, valid, rejected = reject_direction_jumps(
        palm_normal,
        valid,
        min_dot=normal_dot_min,
    )
    return palm_normal, valid, rejected


def smpl_mcp_palm_normal(smplx_joints, joint_names, side, frame_count, world_rot):
    if smplx_joints is None or joint_names is None:
        return None, None
    is_right = side == "right"
    try:
        R_hand = compute_wrist_rotation_from_joints(
            smplx_joints[:frame_count],
            joint_names,
            is_right=is_right,
        )
    except ValueError:
        return None, None
    normal = R_hand.apply(np.repeat([[1.0, 0.0, 0.0]], frame_count, axis=0))
    normal = world_rot.apply(normal)
    return normalized_vectors(normal)


def align_palm_normal_to_body_reference(
    palm_normal,
    source_valid,
    body_normal,
    body_valid,
    dot_min=0.0,
):
    if body_normal is None or body_valid is None:
        return palm_normal, 0

    aligned = np.asarray(palm_normal, dtype=np.float64).copy()
    source_valid = np.asarray(source_valid, dtype=bool).reshape(-1)
    body_valid = np.asarray(body_valid, dtype=bool).reshape(-1)
    count = min(aligned.shape[0], body_normal.shape[0], source_valid.shape[0], body_valid.shape[0])
    if count <= 0:
        return aligned, 0

    dots = np.sum(aligned[:count] * body_normal[:count], axis=-1)
    flip = source_valid[:count] & body_valid[:count] & np.isfinite(dots) & (dots < float(dot_min))
    if np.any(flip):
        aligned_head = aligned[:count]
        aligned_head[flip] *= -1.0
    return aligned, int(np.count_nonzero(flip))


def compute_palm_roll_delta(
    smplx_joints,
    joint_names,
    side,
    hand_metadata,
    coord_transform,
    human_yaw_offset_deg,
    source="auto",
    normal_dot_min=-0.2,
    body_align="auto",
    body_align_dot_min=0.0,
):
    source = str(source or "auto").lower()
    if source not in {"auto", "sidecar", "sidecar_joints", "smpl"}:
        raise ValueError(f"Unsupported palm_roll_source={source!r}")
    body_align = str(body_align or "auto").lower()
    if body_align not in {"auto", "on", "off"}:
        raise ValueError(f"Unsupported palm_roll_body_align={body_align!r}")

    arm_required = [f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"]
    arm_indices = [joint_index(joint_names, name) for name in arm_required]
    if any(idx is None for idx in arm_indices):
        return None, 0, "missing_arm", 0, 0

    frame_count = smplx_joints.shape[0]
    is_right = side == "right"
    world_rot = world_rotation_for_inputs(coord_transform, human_yaw_offset_deg)
    shoulder, elbow, wrist = [
        world_rot.apply(smplx_joints[:, idx, :])
        for idx in arm_indices
    ]

    palm_normal = None
    source_used = "smpl_mcp"
    rejected_jumps = 0
    body_alignment_flips = 0
    source_valid = np.ones(frame_count, dtype=bool)
    if source in {"auto", "sidecar_joints"}:
        sidecar_normal, sidecar_valid, rejected_jumps = sidecar_joint_palm_normal(
            hand_metadata,
            side,
            frame_count,
            world_rot,
            normal_dot_min=normal_dot_min,
        )
        if sidecar_normal is not None and sidecar_valid is not None and int(sidecar_valid.sum()) >= 2:
            palm_normal = sidecar_normal
            source_valid = sidecar_valid
            source_used = "sidecar_joint_palm_normal"
        elif source == "sidecar_joints":
            return None, int(sidecar_valid.sum()) if sidecar_valid is not None else 0, "sidecar_joints_missing", rejected_jumps, 0

    if palm_normal is None and source in {"auto", "sidecar"}:
        sidecar_normal, sidecar_valid, rejected_jumps = sidecar_palm_normal(
            hand_metadata,
            side,
            frame_count,
            world_rot,
            normal_dot_min=normal_dot_min,
        )
        if sidecar_normal is not None and sidecar_valid is not None and int(sidecar_valid.sum()) >= 2:
            palm_normal = sidecar_normal
            source_valid = sidecar_valid
            source_used = "sidecar_palm_normal"
        elif source == "sidecar":
            return None, int(sidecar_valid.sum()) if sidecar_valid is not None else 0, "sidecar_missing", rejected_jumps, 0

    if palm_normal is None:
        smpl_required = [f"{side}_middle1", f"{side}_index1", f"{side}_ring1"]
        smpl_indices = [joint_index(joint_names, name) for name in smpl_required]
        if any(idx is None for idx in smpl_indices):
            return None, 0, "missing_smpl_hand", rejected_jumps, 0
        R_hand = compute_wrist_rotation_from_joints(smplx_joints, joint_names, is_right=is_right)
        R_hand = world_rot * R_hand
        palm_normal = R_hand.apply(np.repeat([[1.0, 0.0, 0.0]], frame_count, axis=0))
    elif body_align != "off":
        body_normal, body_valid = smpl_mcp_palm_normal(
            smplx_joints,
            joint_names,
            side,
            frame_count,
            world_rot,
        )
        palm_normal, body_alignment_flips = align_palm_normal_to_body_reference(
            palm_normal,
            source_valid,
            body_normal,
            body_valid,
            dot_min=body_align_dot_min,
        )
        if body_alignment_flips:
            palm_normal, source_valid, extra_rejected = reject_direction_jumps(
                palm_normal,
                source_valid,
                min_dot=normal_dot_min,
            )
            rejected_jumps += int(extra_rejected)

    forearm_axis, forearm_valid = normalized_vectors(wrist - elbow)
    upper_from_elbow, upper_valid = normalized_vectors(shoulder - elbow)
    arm_ref, ref_valid = normalized_vectors(np.cross(forearm_axis, upper_from_elbow))
    palm_proj = palm_normal - np.sum(palm_normal * forearm_axis, axis=-1, keepdims=True) * forearm_axis
    palm_proj, palm_valid = normalized_vectors(palm_proj)

    valid = forearm_valid & upper_valid & ref_valid & palm_valid & source_valid[:frame_count]
    if source_used == "smpl_mcp":
        valid_key = f"{side}_hand_valid"
        wrist_valid_key = f"{side}_hand_wrist_frame_valid"
        if hand_metadata is not None and valid_key in hand_metadata:
            valid &= np.asarray(hand_metadata[valid_key], dtype=bool).reshape(-1)[:frame_count]
        if hand_metadata is not None and wrist_valid_key in hand_metadata:
            valid &= np.asarray(hand_metadata[wrist_valid_key], dtype=bool).reshape(-1)[:frame_count]
    if np.sum(valid) < 2:
        return None, int(np.sum(valid)), source_used, rejected_jumps, body_alignment_flips

    cross_ref_palm = np.cross(arm_ref, palm_proj)
    roll = np.arctan2(
        np.sum(cross_ref_palm * forearm_axis, axis=-1),
        np.sum(arm_ref * palm_proj, axis=-1),
    )
    roll = np.unwrap(roll)
    roll = fill_invalid_numeric(roll, valid)
    if roll is None:
        return None, int(np.sum(valid)), source_used, rejected_jumps, body_alignment_flips

    valid_indices = np.flatnonzero(valid)
    anchor = int(valid_indices[0])
    delta = roll - roll[anchor]
    return delta.astype(np.float32), int(np.sum(valid)), source_used, rejected_jumps, body_alignment_flips


def stabilize_palm_roll_branch(
    delta,
    branch_step=2.0 * np.pi,
    branch_candidates=2,
    anchor_frames=45,
    transition_weight=1.0,
    branch_penalty=0.001,
    anchor_penalty=0.08,
    max_abs=0.0,
    range_penalty=3.0,
):
    raw = np.asarray(delta, dtype=np.float64).reshape(-1)
    if raw.size == 0 or not np.isfinite(raw).any():
        return raw.astype(np.float32), {
            "branch_shift_frames": 0,
            "branch_switches": 0,
            "branch_offsets": [],
        }

    if not np.isfinite(branch_step) or abs(branch_step) < 1e-6:
        branch_step = np.pi
    branch_step = abs(float(branch_step))
    branch_candidates = max(0, int(branch_candidates))
    offsets = np.arange(-branch_candidates, branch_candidates + 1, dtype=np.float64)
    candidates = raw[:, None] + offsets[None, :] * branch_step
    frame_count, candidate_count = candidates.shape

    finite = np.isfinite(raw)
    first_valid = int(np.flatnonzero(finite)[0])
    anchor_end = min(frame_count, first_valid + max(1, int(anchor_frames)))
    anchor_mask = finite[first_valid:anchor_end]
    if np.any(anchor_mask):
        anchor = float(np.median(raw[first_valid:anchor_end][anchor_mask]))
    else:
        anchor = float(raw[first_valid])

    unary = np.zeros_like(candidates)
    unary += float(branch_penalty) * np.abs(offsets)[None, :]

    if max_abs and max_abs > 0.0:
        range_over = np.maximum(0.0, np.abs(candidates) - float(max_abs))
        unary += float(range_penalty) * np.square(range_over)

    if anchor_frames > 0 and anchor_penalty > 0.0:
        early_end = min(frame_count, first_valid + int(anchor_frames))
        unary[first_valid:early_end] += float(anchor_penalty) * np.abs(candidates[first_valid:early_end] - anchor)
        # The first usable frame defines the visible wrist pose, so keep it on
        # the unshifted branch unless the input itself is unusable.
        unary[first_valid] += 1e6 * np.abs(offsets)

    cost = np.full((frame_count, candidate_count), np.inf, dtype=np.float64)
    back = np.zeros((frame_count, candidate_count), dtype=np.int16)
    cost[first_valid] = unary[first_valid]

    for frame in range(first_valid + 1, frame_count):
        transition = float(transition_weight) * np.square(
            candidates[frame][None, :] - candidates[frame - 1][:, None]
        )
        total = cost[frame - 1][:, None] + transition
        best_prev = np.argmin(total, axis=0)
        cost[frame] = total[best_prev, np.arange(candidate_count)] + unary[frame]
        back[frame] = best_prev.astype(np.int16)

    path = np.zeros(frame_count, dtype=np.int16)
    path[-1] = int(np.argmin(cost[-1]))
    for frame in range(frame_count - 1, first_valid, -1):
        path[frame - 1] = back[frame, path[frame]]
    path[:first_valid] = path[first_valid]

    stabilized = candidates[np.arange(frame_count), path]
    # Preserve the original first-frame wrist pose exactly. This keeps the
    # branch fix from becoming a static wrist offset.
    stabilized -= stabilized[first_valid] - raw[first_valid]

    offset_path = offsets[path].astype(np.int16)
    branch_switches = int(np.count_nonzero(np.diff(offset_path[first_valid:])))
    branch_shift_frames = int(np.count_nonzero(offset_path))
    branch_offsets = sorted(int(v) for v in np.unique(offset_path) if int(v) != 0)
    return stabilized.astype(np.float32), {
        "branch_shift_frames": branch_shift_frames,
        "branch_switches": branch_switches,
        "branch_offsets": branch_offsets,
    }


def apply_palm_roll_to_wrist_joints(dof_pos, model, smplx_joints, joint_names, hand_metadata, args):
    debug = {"palm_roll_mode": args.palm_roll_mode}
    if smplx_joints is None or joint_names is None:
        debug["palm_roll_applied"] = False
        return dof_pos, debug

    out = np.asarray(dof_pos, dtype=np.float32).copy()
    joint_indices = robot_joint_qpos_indices(model)
    applied_sides = []
    side_joint_candidates = {
        "left": ("left_wrist_roll_joint", "left_hand_joint"),
        "right": ("right_wrist_roll_joint", "right_hand_joint"),
    }
    for side in ("left", "right"):
        joint_name = None
        qpos_idx = None
        for candidate in side_joint_candidates[side]:
            if candidate in joint_indices:
                joint_name = candidate
                qpos_idx = joint_indices[candidate]
                break
        if qpos_idx is None:
            debug[f"{side}_palm_roll_frames"] = 0
            continue
        debug[f"{side}_palm_roll_joint"] = joint_name
        dof_idx = qpos_idx - 7
        if dof_idx < 0 or dof_idx >= out.shape[1]:
            debug[f"{side}_palm_roll_frames"] = 0
            continue

        delta, valid_frames, source_used, rejected_jumps, body_alignment_flips = compute_palm_roll_delta(
            smplx_joints[: out.shape[0]],
            joint_names,
            side,
            hand_metadata,
            args.coord_transform,
            args.human_yaw_offset_deg,
            source=args.palm_roll_source,
            normal_dot_min=args.palm_roll_normal_dot_min,
            body_align=args.palm_roll_body_align,
            body_align_dot_min=args.palm_roll_body_align_dot_min,
        )
        debug[f"{side}_palm_roll_frames"] = int(valid_frames)
        debug[f"{side}_palm_roll_source"] = source_used
        debug[f"{side}_palm_roll_normal_jumps_rejected"] = int(rejected_jumps)
        debug[f"{side}_palm_roll_body_alignment_flips"] = int(body_alignment_flips)
        if delta is None:
            continue

        sign = float(args.left_palm_roll_sign if side == "left" else args.right_palm_roll_sign)
        delta = sign * float(args.palm_roll_gain) * delta
        max_abs = float(getattr(args, "palm_roll_max_abs", 0.0))
        branch_mode = str(getattr(args, "palm_roll_branch_mode", "off")).lower()
        if branch_mode != "off":
            delta, branch_debug = stabilize_palm_roll_branch(
                delta,
                # A directed palm normal is 2*pi-periodic.  Using pi here
                # considers the opposite side of the hand equivalent and can
                # introduce a false half-turn after otherwise-correct frames.
                branch_step=2.0 * np.pi * max(abs(float(args.palm_roll_gain)), 1e-6),
                branch_candidates=args.palm_roll_branch_candidates,
                anchor_frames=args.palm_roll_branch_anchor_frames,
                transition_weight=args.palm_roll_branch_transition_weight,
                branch_penalty=args.palm_roll_branch_penalty,
                anchor_penalty=args.palm_roll_branch_anchor_penalty,
                max_abs=max_abs,
                range_penalty=args.palm_roll_branch_range_penalty,
            )
            for key, value in branch_debug.items():
                debug[f"{side}_palm_roll_{key}"] = value
        smooth_window = valid_smooth_window(delta.shape[0], args.palm_roll_smooth_window, args.smooth_polyorder)
        if smooth_window is not None:
            delta = savgol_filter(delta.reshape(-1, 1), smooth_window, int(args.smooth_polyorder), axis=0, mode="interp").reshape(-1)
        if max_abs > 0.0:
            delta = np.clip(delta, -max_abs, max_abs)
        delta = limit_frame_delta(delta.astype(np.float32), args.palm_roll_max_delta)
        if max_abs > 0.0:
            delta = np.clip(delta, -max_abs, max_abs)

        base = float(out[0, dof_idx])
        out[:, dof_idx] = base + delta.astype(np.float32)
        debug[f"{side}_palm_roll_delta_min"] = float(np.min(delta))
        debug[f"{side}_palm_roll_delta_max"] = float(np.max(delta))
        if max_abs > 0.0:
            saturation_margin = max(1e-4, 0.01 * max_abs)
            debug[f"{side}_palm_roll_saturated_frames"] = int(
                np.count_nonzero(np.abs(delta) >= max_abs - saturation_margin)
            )
        else:
            debug[f"{side}_palm_roll_saturated_frames"] = 0
        debug[f"{side}_palm_roll_max_abs"] = max_abs
        debug[f"{side}_palm_roll_branch_mode"] = branch_mode
        applied_sides.append(side)

    debug["palm_roll_applied"] = bool(applied_sides)
    debug["palm_roll_sides"] = applied_sides
    return out, debug


def apply_wrist_pitch_yaw_stabilization(dof_pos, model, args):
    debug = {
        "wrist_pitch_yaw_stabilize": args.wrist_pitch_yaw_stabilize,
        "wrist_pitch_yaw_stabilized": False,
    }
    if not wrist_pitch_yaw_stabilization_enabled(args):
        return dof_pos, debug

    out = np.asarray(dof_pos, dtype=np.float32).copy()
    joint_indices = robot_joint_qpos_indices(model)
    stabilized = []
    for side in ("left", "right"):
        for axis, value in (("pitch", args.wrist_pitch_neutral), ("yaw", args.wrist_yaw_neutral)):
            joint_name = f"{side}_wrist_{axis}_joint"
            qpos_idx = joint_indices.get(joint_name)
            if qpos_idx is None:
                continue
            dof_idx = qpos_idx - 7
            if 0 <= dof_idx < out.shape[1]:
                out[:, dof_idx] = float(value)
                stabilized.append(joint_name)

    debug["wrist_pitch_yaw_stabilized"] = bool(stabilized)
    debug["wrist_pitch_yaw_stabilized_joints"] = stabilized
    return out, debug


def soft_limit_joint_angle(angle, limit_low, limit_high, margin_ratio=0.15):
    values = np.asarray(angle, dtype=np.float64)
    out = values.copy()
    low = float(limit_low)
    high = float(limit_high)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return out.astype(np.float32)

    margin_ratio = max(0.0, min(float(margin_ratio), 0.49))
    margin = margin_ratio * (high - low)
    if margin <= 1e-8:
        return np.clip(out, low, high).astype(np.float32)

    inner_low = low + margin
    inner_high = high - margin

    low_mask = out < inner_low
    if np.any(low_mask):
        t_low = np.clip((inner_low - out[low_mask]) / margin, 0.0, 1.0)
        out[low_mask] = inner_low - margin * (1.0 - np.cos(t_low * np.pi / 2.0))

    high_mask = out > inner_high
    if np.any(high_mask):
        t_high = np.clip((out[high_mask] - inner_high) / margin, 0.0, 1.0)
        out[high_mask] = inner_high + margin * (1.0 - np.cos(t_high * np.pi / 2.0))

    return np.clip(out, low, high).astype(np.float32)


def apply_wrist_pitch_yaw_soft_limit(dof_pos, model, args):
    debug = {
        "wrist_pitch_yaw_soft_limit": args.wrist_pitch_yaw_stabilize,
        "wrist_pitch_yaw_soft_limited": False,
    }
    if not wrist_pitch_yaw_soft_limit_enabled(args):
        return dof_pos, debug

    out = np.asarray(dof_pos, dtype=np.float32).copy()
    target_names = {
        f"{side}_wrist_{axis}_joint"
        for side in ("left", "right")
        for axis in ("pitch", "yaw")
    }
    processed = []
    changed_frames = 0
    margin_ratio = float(args.wrist_pitch_yaw_soft_margin_ratio)

    for joint_id in range(model.njnt):
        if not bool(model.jnt_limited[joint_id]):
            continue
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in (int(mj.mjtJoint.mjJNT_HINGE), int(mj.mjtJoint.mjJNT_SLIDE)):
            continue
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id)
        if name not in target_names:
            continue
        qpos_adr = int(model.jnt_qposadr[joint_id])
        dof_idx = qpos_adr - 7
        if dof_idx < 0 or dof_idx >= out.shape[1]:
            continue

        low, high = np.asarray(model.jnt_range[joint_id], dtype=np.float32)
        before = out[:, dof_idx].copy()
        after = soft_limit_joint_angle(before, low, high, margin_ratio=margin_ratio)
        out[:, dof_idx] = after
        changed = np.abs(after - before) > 1e-6
        changed_frames += int(np.count_nonzero(changed))
        processed.append(name)
        debug[f"{name}_soft_limit_low"] = float(low)
        debug[f"{name}_soft_limit_high"] = float(high)
        debug[f"{name}_soft_limit_changed_frames"] = int(np.count_nonzero(changed))

    debug["wrist_pitch_yaw_soft_limited"] = bool(processed)
    debug["wrist_pitch_yaw_soft_limited_joints"] = processed
    debug["wrist_pitch_yaw_soft_limit_changed_frames"] = int(changed_frames)
    debug["wrist_pitch_yaw_soft_margin_ratio"] = margin_ratio
    return out, debug


def convert_file(src_file, tgt_file, args, device, hand_npz=None):
    smpl_data, body_model, smpl_output, human_height = load_body_motion(
        src_file, args.body_model_path, args.model_type, str(hand_npz) if hand_npz else None
    )
    if args.coord_transform == "gvhmr":
        frames, aligned_fps = get_gvhmr_data_offline_fast(
            smpl_data, body_model, smpl_output, tgt_fps=args.target_fps
        )
    else:
        frames, aligned_fps = get_smplx_data_offline_fast(
            smpl_data, body_model, smpl_output, tgt_fps=args.target_fps
        )
    frames = rotate_human_frames(frames, args.human_yaw_offset_deg)
    smplx_joints = None
    joint_names = None
    if args.model_type in {"smplh", "smplx"}:
        smplx_joints = smpl_output_joints(smpl_output, len(frames))
        joint_names = resolve_smpl_joint_names(body_model, smplx_joints, args.model_type)

    hand_metadata = {}
    hand_wrist_mode = args.hand_wrist_orientation_mode
    wrist_override_enabled = wrist_orientation_override_enabled(args)
    hand_orientation_debug = {
        "hand_wrist_orientation_mode": hand_wrist_mode,
        "hand_wrist_orientation_override_enabled": bool(wrist_override_enabled),
    }
    if hand_npz is not None:
        hand_metadata = fit_hand_metadata_to_frames(
            load_hand_metadata_from_sidecar(hand_npz, missing_ok=True),
            len(frames),
        )
        if hand_wrist_mode == "override_frames" and not wrist_override_enabled:
            hand_orientation_debug["hand_wrist_orientation_override_skipped_robot"] = args.robot
        elif wrist_override_enabled:
            # Preferred path: geometric MCP-frame correction (ported from Do As I Do).
            # Uses SMPL FK joints + sidecar MANO global_orient to compute a
            # denoised wrist orientation. Falls back to the original quaternion
            # override if the sidecar lacks MANO global_orient.
            geo_debug = apply_geometric_wrist_orientation(
                frames, smplx_joints, joint_names, hand_metadata,
                model_type=args.model_type,
                coord_transform=args.coord_transform,
                human_yaw_offset_deg=args.human_yaw_offset_deg,
            )
            hand_orientation_debug.update(geo_debug)

            # Fall back to original sidecar quaternion override for any side
            # where geometric correction produced 0 frames and a quaternion is available
            quat_debug = apply_hand_wrist_orientation_override(frames, hand_metadata)
            for k, v in quat_debug.items():
                if geo_debug.get(k, 0) == 0:
                    hand_orientation_debug[k] = v

    retargeter = GMR(
        src_human="smplx",
        tgt_robot=args.robot,
        actual_human_height=human_height,
        solver=args.solver,
        verbose=args.verbose,
    )
    relaxed_orientation_bodies = relax_orientation_tasks(retargeter, args.relax_orientation_bodies)

    qpos_list = []
    for frame in frames:
        qpos_list.append(retargeter.retarget(frame).copy())
    qpos_list = np.asarray(qpos_list)
    qpos_list, retarget_mode_debug = apply_retarget_mode(qpos_list, retargeter, args)
    qpos_list = smooth_qpos(qpos_list, args.smooth_window, args.smooth_polyorder)

    root_pos = qpos_list[:, :3]
    root_rot = qpos_list[:, 3:7][:, [1, 2, 3, 0]]
    dof_pos = qpos_list[:, 7:]

    # ------------------------------------------------------------------
    # Hand finger retargeting (additive – does not affect body-only flow)
    # ------------------------------------------------------------------
    hand_applied = False
    hand_valid_summary = {"hand_retarget_mode": args.hand_retarget_mode}
    hand_spec = get_hand_spec(args.robot)
    if args.hand_retarget_mode == "axis_component" and hand_npz is not None and hand_spec is not None:
        if has_hand_joints(retargeter.model, hand_spec.keys()):
            # Load raw hand poses directly from the sidecar NPZ
            left_raw, right_raw, left_valid, right_valid = load_hand_poses_from_sidecar(
                hand_npz, missing_ok=False
            )
            if left_raw is not None and right_raw is not None:
                sidecar_is_prefiltered = (
                    "left_hand_quality" in hand_metadata
                    and "right_hand_quality" in hand_metadata
                )
                left_pose, right_pose, left_valid_np, right_valid_np = prepare_hand_poses(
                    left_raw.numpy(),
                    right_raw.numpy(),
                    left_valid.numpy() if left_valid is not None else None,
                    right_valid.numpy() if right_valid is not None else None,
                    frame_count=dof_pos.shape[0],
                    invalid_mode=args.hand_invalid_mode,
                    interp_max_gap=args.hand_interp_max_gap,
                    # A quality-bearing sidecar already contains the same
                    # filtered MANO track shown by GVHMR. Avoid creating a
                    # third gesture version with the legacy spike cleaner.
                    clean_spikes=not sidecar_is_prefiltered,
                )
                finger_angles = compute_finger_angles_from_raw(
                    left_pose, right_pose, hand_spec,
                    curl_power=float(getattr(args, "hand_curl_power", 1.0)),
                )
                if float(args.hand_angle_scale) != 1.0:
                    finger_angles = {
                        name: np.asarray(values, dtype=np.float32) * float(args.hand_angle_scale)
                        for name, values in finger_angles.items()
                    }
                finger_angles = smooth_finger_angles(
                    finger_angles,
                    args.hand_smooth_window,
                    args.hand_smooth_polyorder,
                    median_window=args.hand_median_window,
                    max_delta=args.hand_max_delta,
                    deadband=args.hand_deadband,
                )
                finger_angles = clamp_angles(finger_angles, retargeter.model)
                dof_pos = apply_finger_angles_to_dof_pos(dof_pos, finger_angles, retargeter.model)
                hand_applied = True
                hand_valid_summary = {
                    "left_hand_valid_frames": int(np.sum(left_valid_np)),
                    "right_hand_valid_frames": int(np.sum(right_valid_np)),
                    "hand_frames": int(dof_pos.shape[0]),
                    "hand_spike_cleaning": (
                        "skipped_prefiltered_sidecar"
                        if sidecar_is_prefiltered
                        else "legacy_gmr_cleaning"
                    ),
                }
                if args.verbose:
                    nz = sum(1 for a in finger_angles.values() if float(np.max(np.abs(a))) > 1e-6)
                    print(
                        f"[hand] Applied finger angles to {nz}/{len(finger_angles)} joints "
                        f"from {hand_npz}"
                    )
        elif args.verbose:
            print(f"[hand] Robot '{args.robot}' has a hand spec but no matching joints in the MuJoCo model")
    elif args.hand_retarget_mode == "axis_component" and hand_npz is not None and args.verbose:
        print(f"[hand] No hand spec for robot '{args.robot}'; skipping finger retargeting")
    elif args.hand_retarget_mode == "off" and args.verbose:
        print("[hand] Built-in robot finger retargeting disabled (external hand trajectory expected)")

    palm_roll_debug = {}
    if palm_roll_retarget_enabled(args, hand_npz):
        dof_pos, palm_roll_debug = apply_palm_roll_to_wrist_joints(
            dof_pos, retargeter.model, smplx_joints, joint_names, hand_metadata, args
        )
        if args.verbose:
            for side in ("left", "right"):
                src = palm_roll_debug.get(f"{side}_palm_roll_source", "?")
                n_frames = palm_roll_debug.get(f"{side}_palm_roll_frames", 0)
                rejected = palm_roll_debug.get(f"{side}_palm_roll_normal_jumps_rejected", 0)
                body_flips = palm_roll_debug.get(f"{side}_palm_roll_body_alignment_flips", 0)
                switches = palm_roll_debug.get(f"{side}_palm_roll_branch_switches", 0)
                offsets = palm_roll_debug.get(f"{side}_palm_roll_branch_offsets", [])
                print(
                    f"[palm_roll] {side}: source={src}, valid_frames={n_frames}, "
                    f"jumps_rejected={rejected}, body_flips={body_flips}, branch_switches={switches}, "
                    f"branch_offsets={offsets}"
                )
    else:
        palm_roll_debug = {
            "palm_roll_mode": args.palm_roll_mode,
            "palm_roll_applied": False,
        }

    dof_pos, wrist_stabilization_debug = apply_wrist_pitch_yaw_stabilization(
        dof_pos, retargeter.model, args
    )
    dof_pos, wrist_soft_limit_debug = apply_wrist_pitch_yaw_soft_limit(
        dof_pos, retargeter.model, args
    )

    dof_pos = clamp_dof_pos_to_joint_ranges(dof_pos, retargeter.model)

    kinematics_model = KinematicsModel(retargeter.xml_file, device=device)
    num_frames = root_pos.shape[0]
    fk_root_pos = torch.zeros((num_frames, 3), device=device)
    fk_root_rot = torch.zeros((num_frames, 4), device=device)
    fk_root_rot[:, -1] = 1.0
    local_body_pos, _ = kinematics_model.forward_kinematics(
        fk_root_pos,
        fk_root_rot,
        torch.from_numpy(dof_pos).to(device=device, dtype=torch.float),
    )
    if args.height_adjust_mode in {
        "per_frame_geom",
        "global_geom",
        "per_frame_foot_geom",
        "global_foot_geom",
    }:
        selector = "foot" if args.height_adjust_mode.endswith("_foot_geom") else "all"
        lowest_height = robot_geom_lowest_heights(
            retargeter.xml_file,
            root_pos,
            root_rot,
            dof_pos,
            selector=selector,
        )
        if args.height_adjust_mode in {"per_frame_geom", "per_frame_foot_geom"}:
            root_pos[:, 2] = root_pos[:, 2] - lowest_height + args.ground_offset
        else:
            root_pos[:, 2] = root_pos[:, 2] - float(np.min(lowest_height)) + args.ground_offset
    elif args.height_adjust_mode != "none":
        body_pos, _ = kinematics_model.forward_kinematics(
            torch.from_numpy(root_pos).to(device=device, dtype=torch.float),
            torch.from_numpy(root_rot).to(device=device, dtype=torch.float),
            torch.from_numpy(dof_pos).to(device=device, dtype=torch.float),
        )
        if args.height_adjust_mode == "per_frame":
            lowest_height = torch.min(body_pos[..., 2], dim=1).values.detach().cpu().numpy()
            root_pos[:, 2] = root_pos[:, 2] - lowest_height + args.ground_offset
        else:
            lowest_height = torch.min(body_pos[..., 2]).item()
            root_pos[:, 2] = root_pos[:, 2] - lowest_height + args.ground_offset

    motion_data = {
        "fps": aligned_fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        # Store an explicit column contract for safe NPZ product export.
        # MuJoCo's first six velocity DoFs belong to the floating base; the
        # exported dof_pos starts after that base and contains hinge joints.
        "dof_names": [
            mj.mj_id2name(
                retargeter.model,
                mj.mjtObj.mjOBJ_JOINT,
                retargeter.model.dof_jntid[index],
            )
            for index in range(6, retargeter.model.nv)
        ],
        "embodiment": args.robot,
        "local_body_pos": local_body_pos.detach().cpu().numpy(),
        "link_body_list": kinematics_model.body_names,
        "source_motion": str(src_file),
        "model_type": args.model_type,
        "actual_human_height": float(human_height),
        "human_height_source": str(np.asarray(smpl_data.get("human_height_source", "unknown")).item()),
        "smooth_window": int(args.smooth_window),
        "smooth_polyorder": int(args.smooth_polyorder),
        "height_adjust_mode": args.height_adjust_mode,
        "height_adjust_geom_selector": (
            "foot" if args.height_adjust_mode.endswith("_foot_geom") else "all"
        ),
        "retarget_mode": args.retarget_mode,
        "human_yaw_offset_deg": float(args.human_yaw_offset_deg),
        "relaxed_orientation_bodies": relaxed_orientation_bodies,
        "hand_npz": str(hand_npz) if hand_npz else "",
        **hand_orientation_debug,
        **palm_roll_debug,
        **wrist_stabilization_debug,
        **wrist_soft_limit_debug,
        **retarget_mode_debug,
        **hand_valid_summary,
    }
    if hand_applied:
        motion_data["hand_retarget_applied"] = True
    for key in (
        "left_hand_wrist_quat",
        "right_hand_wrist_quat",
        "left_hand_palm_normal",
        "right_hand_palm_normal",
        "left_hand_wrist_frame_valid",
        "right_hand_wrist_frame_valid",
        "left_hand_wrist_frame_source",
        "right_hand_wrist_frame_source",
    ):
        if key in hand_metadata:
            motion_data[key] = hand_metadata[key]

    tgt_file.parent.mkdir(parents=True, exist_ok=True)
    with open(tgt_file, "wb") as f:
        pickle.dump(motion_data, f)


def main():
    parser = argparse.ArgumentParser(
        description="Headless retargeting from GVHMR/Locomotion SMPL-style NPZ files to GMR robot motion PKL."
    )
    parser.add_argument("--src_root", required=True, type=pathlib.Path)
    parser.add_argument("--tgt_root", required=True, type=pathlib.Path)
    parser.add_argument("--pattern", default="*/selected/001/001_selected.npz")
    parser.add_argument(
        "--include_clips",
        default="",
        help="Optional comma-separated clip directory names to retarget from the matched source set.",
    )
    parser.add_argument(
        "--output_name",
        default="",
        help="Optional fixed output filename under each matched source's relative parent, e.g. robot_motion.pkl.",
    )
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--body_model_path", default=str(HERE / ".." / "assets" / "body_models"))
    parser.add_argument("--model_type", choices=["smpl", "smplh", "smplx"], default="smplh")
    parser.add_argument(
        "--hand_npz",
        default=None,
        type=str,
        help="Optional path to an SMPL-X hand sidecar NPZ (e.g. 001_smplx_hands.npz). "
        "When provided, this exact file is used for every source motion.",
    )
    parser.add_argument(
        "--hand_npz_name",
        default="001_smplx_hands.npz",
        help="Filename to auto-detect beside each source NPZ when --hand_npz is not provided.",
    )
    parser.add_argument(
        "--auto_hand_npz",
        action="store_true",
        help="Auto-detect a hand sidecar beside each source NPZ when --hand_npz is not provided.",
    )
    parser.add_argument(
        "--hand_invalid_mode",
        choices=["hold", "interp", "zero", "raw"],
        default="hold",
        help="How to handle invalid hand frames from the sidecar valid masks.",
    )
    parser.add_argument(
        "--hand_retarget_mode",
        choices=["axis_component", "off"],
        default="axis_component",
        help=(
            "Built-in robot finger mapping. Use off when an external target-space "
            "hand trajectory such as Sharpa is rendered with the body motion."
        ),
    )
    parser.add_argument(
        "--hand_interp_max_gap",
        default=15,
        type=int,
        help="For --hand_invalid_mode interp, interpolate invalid gaps up to this many frames; 0 interpolates all gaps.",
    )
    parser.add_argument("--target_fps", default=30, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--solver", default="daqp")
    parser.add_argument(
        "--coord_transform",
        choices=["gvhmr", "none"],
        default="gvhmr",
        help="Use gvhmr for GVHMR/Locomotion Y-up outputs; use none for AMASS-style SMPL-X.",
    )
    parser.add_argument(
        "--human_yaw_offset_deg",
        default=0.0,
        type=float,
        help="Rotate all human IK targets around the robot Z-up axis; use 180 for monocular front/back ambiguity.",
    )
    parser.add_argument(
        "--relax_orientation_bodies",
        default=None,
        help="Comma-separated SMPL body names whose GMR orientation target should be disabled.",
    )
    parser.add_argument(
        "--height_adjust_mode",
        choices=[
            "global",
            "per_frame",
            "global_geom",
            "per_frame_geom",
            "global_foot_geom",
            "per_frame_foot_geom",
            "none",
        ],
        default="global",
        help=(
            "global/per_frame use body origins; global_geom/per_frame_geom use MuJoCo geometry; "
            "global_foot_geom/per_frame_foot_geom use only foot/toe/ankle geometry; "
            "none keeps raw root height."
        ),
    )
    parser.add_argument("--no_height_adjust", dest="height_adjust_mode", action="store_const", const="none")
    parser.add_argument("--ground_offset", default=0.0, type=float)
    parser.add_argument("--smooth_window", default=9, type=int, help="Savgol smoothing window in frames; 0 disables smoothing.")
    parser.add_argument("--smooth_polyorder", default=2, type=int)
    parser.add_argument("--no_smooth", dest="smooth_window", action="store_const", const=0)
    parser.add_argument(
        "--hand_smooth_window",
        default=25,
        type=int,
        help="Savgol smoothing window for finger joints only; larger values reduce HaMeR jitter.",
    )
    parser.add_argument("--hand_smooth_polyorder", default=2, type=int)
    parser.add_argument(
        "--hand_median_window",
        default=5,
        type=int,
        help="Median filter window for finger joints before Savgol smoothing.",
    )
    parser.add_argument(
        "--hand_max_delta",
        default=0.08,
        type=float,
        help="Maximum per-frame finger angle change in radians; 0 disables rate limiting.",
    )
    parser.add_argument(
        "--hand_deadband",
        default=0.005,
        type=float,
        help="Suppress finger angle changes smaller than this many radians after smoothing.",
    )
    parser.add_argument(
        "--hand_angle_scale",
        default=0.85,
        type=float,
        help="Scale raw MANO-to-robot finger angles before smoothing/clamping.",
    )
    parser.add_argument(
        "--hand_curl_power",
        default=1.0,
        type=float,
        help="Power-law exponent for amplifying small finger curl (0.5=sqrt boost, 1.0=linear).",
    )
    parser.add_argument(
        "--hand_wrist_orientation_mode",
        choices=["off", "diagnostic", "override_frames"],
        default="diagnostic",
        help=(
            "diagnostic stores sidecar palm/wrist orientation in the output PKL; "
            "override_frames replaces left/right wrist IK target orientations using that sidecar "
            "for explicitly supported experimental robots."
        ),
    )
    parser.add_argument(
        "--force_hand_wrist_orientation_override",
        action="store_true",
        help="Force wrist orientation override even for robots that are not in the supported whitelist.",
    )
    parser.add_argument(
        "--palm_roll_mode",
        choices=["auto", "on", "off"],
        default="auto",
        help="Map palm flip to wrist_roll only. auto enables this for G1 when hand sidecar data is available.",
    )
    parser.add_argument(
        "--palm_roll_gain",
        default=1.0,
        type=float,
        help="Scale for palm-normal roll mapped onto the robot wrist_roll joints.",
    )
    parser.add_argument(
        "--palm_roll_source",
        choices=["auto", "sidecar", "sidecar_joints", "smpl"],
        default="smpl",
        help=(
            "Palm-normal source for wrist-roll mapping. auto prefers the HaMeR sidecar "
            "21-joint palm geometry, then the sidecar palm normal, then SMPL MCP geometry."
        ),
    )
    parser.add_argument(
        "--palm_roll_normal_dot_min",
        default=-0.2,
        type=float,
        help=(
            "Minimum allowed dot product between adjacent sidecar palm normals before "
            "a sudden 180-degree ambiguity is repaired or rejected."
        ),
    )
    parser.add_argument(
        "--palm_roll_body_align",
        choices=["auto", "on", "off"],
        default="auto",
        help="Align sidecar palm normals to the SMPL MCP palm-normal sign before extracting palm roll.",
    )
    parser.add_argument(
        "--palm_roll_body_align_dot_min",
        default=0.0,
        type=float,
        help="Flip sidecar palm normals when their dot product with the SMPL MCP reference is below this value.",
    )
    parser.add_argument(
        "--left_palm_roll_sign",
        default=1.0,
        type=float,
        help="Sign for left-hand palm roll mapping; use -1 if the left hand flips the wrong way.",
    )
    parser.add_argument(
        "--right_palm_roll_sign",
        default=1.0,
        type=float,
        help="Sign for right-hand palm roll mapping; use -1 if the right hand flips the wrong way.",
    )
    parser.add_argument(
        "--palm_roll_smooth_window",
        default=15,
        type=int,
        help="Savgol smoothing window for palm roll signal; 0 disables smoothing.",
    )
    parser.add_argument(
        "--palm_roll_max_delta",
        default=0.10,
        type=float,
        help="Maximum per-frame palm roll change in radians after smoothing; 0 disables rate limiting.",
    )
    parser.add_argument(
        "--palm_roll_max_abs",
        default=0.0,
        type=float,
        help="Optional absolute palm-roll delta clamp in radians; 0 disables the clamp.",
    )
    parser.add_argument(
        "--palm_roll_branch_mode",
        choices=["off", "viterbi"],
        default="off",
        help="Stabilize equivalent full-turn palm-roll wrapping by selecting a temporal branch.",
    )
    parser.add_argument(
        "--palm_roll_branch_candidates",
        default=2,
        type=int,
        help="Number of +/- 2*pi palm-roll branches to consider on each side of the raw signal.",
    )
    parser.add_argument(
        "--palm_roll_branch_anchor_frames",
        default=45,
        type=int,
        help="Number of early frames used to prefer the initial visible palm-roll branch.",
    )
    parser.add_argument(
        "--palm_roll_branch_transition_weight",
        default=1.0,
        type=float,
        help="Viterbi transition weight for palm-roll branch stabilization.",
    )
    parser.add_argument(
        "--palm_roll_branch_penalty",
        default=0.001,
        type=float,
        help="Small per-frame cost for shifted palm-roll branches.",
    )
    parser.add_argument(
        "--palm_roll_branch_anchor_penalty",
        default=0.08,
        type=float,
        help="Cost for moving away from the initial palm-roll branch during anchor frames.",
    )
    parser.add_argument(
        "--palm_roll_branch_range_penalty",
        default=3.0,
        type=float,
        help="Cost for branch candidates outside --palm_roll_max_abs before final clipping.",
    )
    parser.add_argument(
        "--wrist_pitch_yaw_stabilize",
        choices=["auto", "on", "off", "soft_limit"],
        default="auto",
        help="Control relaxed G1 wrist pitch/yaw: auto uses soft_limit for G1, on keeps legacy neutral locking.",
    )
    parser.add_argument(
        "--wrist_pitch_neutral",
        default=0.0,
        type=float,
        help="Neutral value used by --wrist_pitch_yaw_stabilize for wrist pitch joints.",
    )
    parser.add_argument(
        "--wrist_yaw_neutral",
        default=0.0,
        type=float,
        help="Neutral value used by --wrist_pitch_yaw_stabilize for wrist yaw joints.",
    )
    parser.add_argument(
        "--wrist_pitch_yaw_soft_margin_ratio",
        default=0.15,
        type=float,
        help="Fraction of wrist pitch/yaw joint range used as the soft-limit transition band.",
    )
    parser.add_argument(
        "--retarget_mode",
        choices=["full", "upper_body"],
        default="full",
        help="upper_body locks leg/ankle/foot joints after IK, useful for robots or policies without foot motion.",
    )
    parser.add_argument(
        "--upper_body_root_mode",
        choices=["fixed", "fixed_xy", "free"],
        default="fixed",
        help="Root handling when --retarget_mode upper_body: fixed locks floating base, fixed_xy only locks XY, free keeps GMR root.",
    )
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.relax_orientation_bodies is None:
        hand_enabled = bool(args.hand_npz or args.auto_hand_npz)
        if hand_enabled and wrist_orientation_override_enabled(args):
            # Geometric wrist correction is active: keep wrist orientation
            # IK targets enabled so the IK can track the corrected orientation.
            args.relax_orientation_bodies = ""
        else:
            args.relax_orientation_bodies = "left_wrist,right_wrist"

    src_root = args.src_root.resolve()
    tgt_root = args.tgt_root.resolve()
    device = "cuda:0" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    files = sorted(src_root.glob(args.pattern))
    include_clips = {clip.strip() for clip in args.include_clips.split(",") if clip.strip()}
    if include_clips:
        before = len(files)
        filtered_files = []
        for path in files:
            try:
                rel_parts = path.relative_to(src_root).parts
            except ValueError:
                rel_parts = path.parts
            if any(part in include_clips for part in rel_parts[:-1]):
                filtered_files.append(path)
        files = filtered_files
        print(f"Filtered sources by include_clips: {len(files)}/{before}")

    print(f"Found {len(files)} files with pattern: {args.pattern}")
    print(f"Using body model: {args.model_type} at {args.body_model_path}")
    print(f"Using device: {device}")

    failed_files = []
    for src_file in tqdm(files, desc="Retargeting"):
        resolved_src_file = src_file.resolve()
        tgt_file = output_path_for(src_file, src_root, tgt_root, args.output_name or None)
        if tgt_file.exists() and not args.override:
            continue
        try:
            hand_npz = resolve_hand_npz_for_source(resolved_src_file, args)
            convert_file(resolved_src_file, tgt_file, args, device, hand_npz=hand_npz)
            print(f"Saved {tgt_file}")
        except Exception as exc:
            failed_files.append((src_file, exc))
            print(f"[red]Failed {src_file}: {exc}[/red]")

    if failed_files:
        print(f"[red]Failed to retarget {len(failed_files)}/{len(files)} files.[/red]")
        raise SystemExit(1)

    print(f"Done. Outputs saved to {tgt_root}")


if __name__ == "__main__":
    main()
