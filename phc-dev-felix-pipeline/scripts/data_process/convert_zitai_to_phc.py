import argparse
import os
import os.path as osp
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot


ROBOT_CFG = {
    "mesh": False,
    "model": "smpl",
    "upright_start": True,
    "body_params": {},
    "joint_params": {},
    "geom_params": {},
    "actuator_params": {},
}


def _axis_alignment_rotation(gravity_axis):
    if gravity_axis == "neg_z":
        return None
    if gravity_axis == "neg_y":
        return sRot.from_euler("x", 90, degrees=True)
    raise ValueError(f"Unsupported gravity axis: {gravity_axis}")


def apply_world_alignment_to_smpl(pose_aa, trans, gravity_axis="neg_z", inverse=False):
    pose_aa = np.asarray(pose_aa, dtype=np.float32)
    trans = np.asarray(trans, dtype=np.float32)
    align_rot = _axis_alignment_rotation(gravity_axis)
    if align_rot is None:
        return pose_aa.astype(np.float32), trans.astype(np.float32)

    world_rot = align_rot.inv() if inverse else align_rot
    pose_aa = pose_aa.reshape(-1, 24, 3).copy()
    root_rot = sRot.from_rotvec(pose_aa[:, 0, :])
    pose_aa[:, 0, :] = (world_rot * root_rot).as_rotvec().astype(np.float32)
    trans = world_rot.apply(trans).astype(np.float32)
    return pose_aa.reshape(-1, 72).astype(np.float32), trans.astype(np.float32)


def _load_scalar_npy(path):
    value = np.load(path, allow_pickle=True)
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    return value


def _normalize_gender(gender_value):
    if isinstance(gender_value, np.ndarray):
        gender_value = gender_value.item()
    if isinstance(gender_value, bytes):
        gender_value = gender_value.decode("utf-8")
    if gender_value is None:
        return "neutral", [0]

    gender = str(gender_value).lower()
    if gender in {"male", "m"}:
        return "male", [1]
    if gender in {"female", "f"}:
        return "female", [2]
    return "neutral", [0]


def _load_pose_aa(clip_dir):
    poses_path = osp.join(clip_dir, "poses.npy")
    if osp.isfile(poses_path):
        pose_aa = np.load(poses_path, allow_pickle=True)
    else:
        root_orient = np.load(osp.join(clip_dir, "root_orient.npy"), allow_pickle=True)
        pose_body = np.load(osp.join(clip_dir, "pose_body.npy"), allow_pickle=True)
        pose_aa = np.concatenate([root_orient, pose_body], axis=-1)

    pose_aa = np.asarray(pose_aa, dtype=np.float32)
    if pose_aa.ndim != 2:
        raise ValueError(f"Unexpected pose shape {pose_aa.shape} in {clip_dir}")

    if pose_aa.shape[1] < 66:
        raise ValueError(f"Pose dim {pose_aa.shape[1]} is too small in {clip_dir}")

    if pose_aa.shape[1] < 72:
        pose_aa = np.concatenate([pose_aa[:, :66], np.zeros((pose_aa.shape[0], 6), dtype=np.float32)], axis=1)
    else:
        pose_aa = pose_aa[:, :72]
    return pose_aa


def _load_pose_aa_npz(clip_file):
    with np.load(clip_file, allow_pickle=True) as data:
        if "poses" in data:
            pose_aa = data["poses"]
        else:
            root_orient = data["root_orient"]
            pose_body = data["pose_body"]
            pose_aa = np.concatenate([root_orient, pose_body], axis=-1)

    pose_aa = np.asarray(pose_aa, dtype=np.float32)
    if pose_aa.ndim == 3 and pose_aa.shape[1:] == (24, 3):
        pose_aa = pose_aa.reshape(pose_aa.shape[0], 72)
    if pose_aa.ndim != 2:
        raise ValueError(f"Unexpected pose shape {pose_aa.shape} in {clip_file}")

    if pose_aa.shape[1] < 66:
        raise ValueError(f"Pose dim {pose_aa.shape[1]} is too small in {clip_file}")

    if pose_aa.shape[1] < 72:
        pose_aa = np.concatenate([pose_aa[:, :66], np.zeros((pose_aa.shape[0], 6), dtype=np.float32)], axis=1)
    else:
        pose_aa = pose_aa[:, :72]
    return pose_aa


def _find_clip_dirs(input_root):
    input_root = Path(input_root)
    if input_root.is_file() and input_root.suffix == ".npz":
        return [input_root]
    if osp.isfile(input_root / "poses.npy") or osp.isfile(input_root / "root_orient.npy"):
        return [input_root]

    clip_dirs = []
    for path in input_root.rglob("*"):
        if path.is_file() and path.suffix == ".npz":
            clip_dirs.append(path)
            continue
        if path.is_dir() and ((path / "poses.npy").is_file() or (path / "root_orient.npy").is_file()):
            clip_dirs.append(path)
    return sorted(clip_dirs)


def _build_key(input_root, clip_dir):
    rel = clip_dir.relative_to(input_root)
    if str(rel) == ".":
        return clip_dir.stem if clip_dir.is_file() else clip_dir.name
    rel_str = str(rel).replace(os.sep, "__")
    return rel_str[:-4] if rel_str.endswith(".npz") else rel_str


def _load_source_fields(clip_path):
    clip_path = Path(clip_path)
    if clip_path.is_file() and clip_path.suffix == ".npz":
        with np.load(clip_path, allow_pickle=True) as data:
            pose_aa = _load_pose_aa_npz(clip_path)
            trans = np.asarray(data["trans"], dtype=np.float32)
            betas = np.asarray(data["betas"], dtype=np.float32).reshape(-1)
            fps = float(data["mocap_frame_rate"].item() if np.asarray(data["mocap_frame_rate"]).shape == () else data["mocap_frame_rate"])
            gender_value = data["gender"].item() if np.asarray(data["gender"]).shape == () else data["gender"]
        return pose_aa, trans, betas, fps, gender_value

    clip_dir = str(clip_path)
    pose_aa = _load_pose_aa(clip_dir)
    trans = np.asarray(np.load(osp.join(clip_dir, "trans.npy"), allow_pickle=True), dtype=np.float32)
    betas = np.asarray(np.load(osp.join(clip_dir, "betas.npy"), allow_pickle=True), dtype=np.float32).reshape(-1)
    fps = float(_load_scalar_npy(osp.join(clip_dir, "mocap_frame_rate.npy")))
    gender_value = _load_scalar_npy(osp.join(clip_dir, "gender.npy"))
    return pose_aa, trans, betas, fps, gender_value


def convert_clip(clip_dir, smpl_local_robot, force_neutral=True, target_fps=30, gravity_axis="neg_z"):
    clip_dir = str(clip_dir)
    pose_aa, trans, betas, fps, gender_value = _load_source_fields(clip_dir)
    pose_aa, trans = apply_world_alignment_to_smpl(pose_aa, trans, gravity_axis=gravity_axis, inverse=False)

    fps = target_fps if fps <= 0 else fps
    stride = max(int(round(fps / target_fps)), 1)

    pose_aa = pose_aa[::stride]
    trans = trans[::stride]
    fps_out = fps / stride

    if pose_aa.shape[0] != trans.shape[0]:
        raise ValueError(f"Frame mismatch in {clip_dir}: pose={pose_aa.shape[0]}, trans={trans.shape[0]}")

    if pose_aa.shape[0] < 10:
        raise ValueError(f"Clip too short after resampling in {clip_dir}: {pose_aa.shape[0]} frames")

    gender, gender_number = _normalize_gender(gender_value)
    beta = betas[:10].copy()

    if force_neutral:
        gender = "neutral"
        gender_number = [0]
        beta[:] = 0

    smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]
    pose_aa_mj = pose_aa.reshape(-1, 24, 3)[:, smpl_2_mujoco]
    pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(pose_aa.shape[0], 24, 4)

    smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None, ]), gender=gender_number, objs_info=None)
    xml_path = "phc/data/assets/mjcf/smpl_humanoid_1.xml"
    smpl_local_robot.write_xml(xml_path)
    skeleton_tree = SkeletonTree.from_mjcf(xml_path)
    root_trans_offset = torch.from_numpy(trans) + skeleton_tree.local_translation[0]

    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(pose_quat),
        root_trans_offset,
        is_local=True,
    )

    if ROBOT_CFG["upright_start"]:
        pose_quat_global = (
            sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy())
            * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
        ).as_quat().reshape(pose_aa.shape[0], -1, 4)
        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat_global),
            root_trans_offset,
            is_local=False,
        )

    return {
        "pose_quat_global": new_sk_state.global_rotation.numpy(),
        "pose_quat": new_sk_state.local_rotation.numpy(),
        "trans_orig": trans,
        "root_trans_offset": root_trans_offset,
        "beta": beta,
        "gender": gender,
        "pose_aa": pose_aa,
        "fps": fps_out,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, required=True, help="Single optimized clip dir or a directory containing many optimized clip dirs.")
    parser.add_argument("--output_file", type=str, required=True, help="Output PHC motion file (.pkl).")
    parser.add_argument("--target_fps", type=int, default=30)
    parser.add_argument("--keep_shape", action="store_true", help="Keep original betas and gender instead of forcing neutral SMPL.")
    parser.add_argument("--gravity_axis", type=str, default="neg_z", choices=["neg_z", "neg_y"], help="Gravity axis used by the source motion data.")
    args = parser.parse_args()

    clip_dirs = _find_clip_dirs(args.input_root)
    if not clip_dirs:
        raise FileNotFoundError(f"No clip directories found under {args.input_root}")

    smpl_local_robot = LocalRobot(ROBOT_CFG, data_dir="data/smpl")
    motion_dict = {}
    input_root = Path(args.input_root)

    for clip_dir in tqdm(clip_dirs):
        key = _build_key(input_root, clip_dir)
        motion_dict[key] = convert_clip(
            clip_dir=clip_dir,
            smpl_local_robot=smpl_local_robot,
            force_neutral=not args.keep_shape,
            target_fps=args.target_fps,
            gravity_axis=args.gravity_axis,
        )

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(motion_dict, args.output_file)
    print(f"Converted {len(motion_dict)} clips to {args.output_file}")


if __name__ == "__main__":
    main()
