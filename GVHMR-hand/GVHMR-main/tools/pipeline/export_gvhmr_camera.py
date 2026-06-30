#!/usr/bin/env python
"""Export a GVHMR/GVHMR-hand camera trajectory for Isaac Gym recording."""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from pytorch3d.transforms import matrix_to_axis_angle
from scipy.spatial.transform import Rotation as Rotation


GVHMR_ROOT = Path(__file__).resolve().parents[2]
if str(GVHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(GVHMR_ROOT))

from hmr4d.utils.geo.hmr_global import get_T_w2c_from_wcparams
from hmr4d.utils.smplx_utils import make_smplx


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
        return tensor

    if tensor.ndim == 3:
        looks_like_people = tensor.shape[0] <= 16 and tensor.shape[1] > 1
        if name == "global_orient":
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
        aa = matrix_to_axis_angle(tensor).float()
    else:
        aa = tensor.float()

    if aa.ndim == 1:
        aa = aa[None, :]
    if aa.ndim >= 3:
        aa = aa.reshape(aa.shape[0], -1)
    return aa[:, :3].contiguous()


def as_trans(value, person_idx):
    tensor = select_person(value, person_idx, "transl").float()
    if tensor.ndim == 1:
        tensor = tensor[None, :]
    return tensor.reshape(tensor.shape[0], -1)[:, :3].contiguous()


def as_betas(value, person_idx):
    tensor = select_person(value, person_idx, "betas").float()
    if tensor.ndim >= 2:
        tensor = tensor.reshape(tensor.shape[0], -1).mean(dim=0)
    return tensor.reshape(-1).contiguous()


def load_params(results, key, person_idx):
    if key not in results:
        raise KeyError(f"GVHMR results missing {key}")
    params = results[key]
    return {
        "global_orient": as_axis_angle(params["global_orient"], person_idx, "global_orient"),
        "transl": as_trans(params["transl"], person_idx),
        "betas": as_betas(params["betas"], person_idx),
    }


def root_offset_from_betas(betas):
    betas_0 = betas.reshape(1, -1)[:, :10]
    model = make_smplx("supermotion")
    with torch.no_grad():
        return model.get_skeleton(betas_0)[0, 0].detach().cpu().float()


def reference_alignment_offset(reference_npz, gvhmr_transl):
    if not reference_npz:
        return np.zeros(3, dtype=np.float32)
    ref_path = Path(reference_npz)
    if not ref_path.is_file():
        return np.zeros(3, dtype=np.float32)
    with np.load(ref_path, allow_pickle=True) as data:
        if "trans" not in data:
            return np.zeros(3, dtype=np.float32)
        ref_trans = np.asarray(data["trans"], dtype=np.float32)
    n = min(ref_trans.shape[0], gvhmr_transl.shape[0])
    if n <= 0:
        return np.zeros(3, dtype=np.float32)
    return np.median(ref_trans[:n] - gvhmr_transl[:n], axis=0).astype(np.float32)


def select_k_fullimg(k_fullimg, person_idx):
    k = to_tensor(k_fullimg).float()
    if k.ndim == 4:
        k = k[person_idx]
    return k.numpy().astype(np.float32)


def horizontal_fov_from_k(k_fullimg):
    k_arr = np.asarray(k_fullimg, dtype=np.float64)
    k0 = k_arr[0] if k_arr.ndim == 3 else k_arr
    fx = float(k0[0, 0])
    cx = float(k0[0, 2])
    width = max(1.0, 2.0 * cx)
    if fx <= 1e-6:
        return 70.0
    return float(np.degrees(2.0 * math.atan(width / (2.0 * fx))))


def world_to_isaac_rotation(gravity_axis):
    if gravity_axis == "neg_y":
        return Rotation.from_euler("x", 90.0, degrees=True).as_matrix().astype(np.float32)
    if gravity_axis == "neg_z":
        return np.eye(3, dtype=np.float32)
    raise ValueError(f"Unsupported gravity axis: {gravity_axis}")


def export_camera(args):
    results = torch.load(args.gvhmr_results, map_location="cpu", weights_only=False)
    params_w = load_params(results, "smpl_params_global", args.person_idx)
    params_c = load_params(results, "smpl_params_incam", args.person_idx)
    if params_w["transl"].shape[0] != params_c["transl"].shape[0]:
        raise ValueError("GVHMR global/incam frame counts differ")

    frame_count = min(
        params_w["global_orient"].shape[0],
        params_w["transl"].shape[0],
        params_c["global_orient"].shape[0],
        params_c["transl"].shape[0],
    )
    for params in (params_w, params_c):
        params["global_orient"] = params["global_orient"][:frame_count]
        params["transl"] = params["transl"][:frame_count]

    offset = root_offset_from_betas(params_w["betas"])
    t_w2c = get_T_w2c_from_wcparams(
        global_orient_w=params_w["global_orient"],
        transl_w=params_w["transl"],
        global_orient_c=params_c["global_orient"],
        transl_c=params_c["transl"],
        offset=offset,
    ).detach().cpu().numpy().astype(np.float32)

    r_w2c = t_w2c[:, :3, :3]
    t_vec = t_w2c[:, :3, 3]
    r_c2w = np.transpose(r_w2c, (0, 2, 1))
    camera_pos_world = -np.einsum("fij,fj->fi", r_c2w, t_vec)

    forward = np.einsum("fij,j->fi", r_c2w, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    subject_world = params_w["transl"].numpy().astype(np.float32)
    subject_world = subject_world + offset.numpy().astype(np.float32)[None, :]
    if np.median(np.sum((subject_world - camera_pos_world) * forward, axis=1)) < 0.0:
        forward *= -1.0

    align_offset = reference_alignment_offset(args.reference_npz, params_w["transl"].numpy())
    camera_pos_world = camera_pos_world + align_offset[None, :]
    subject_world = subject_world + align_offset[None, :]
    camera_target_world = camera_pos_world + forward

    world_to_isaac = world_to_isaac_rotation(args.gravity_axis)
    camera_pos_isaac = camera_pos_world @ world_to_isaac.T
    camera_target_isaac = camera_target_world @ world_to_isaac.T
    subject_isaac = subject_world @ world_to_isaac.T

    k_fullimg = select_k_fullimg(results["K_fullimg"], args.person_idx)
    horizontal_fov_deg = horizontal_fov_from_k(k_fullimg)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        camera_pos_world=camera_pos_world.astype(np.float32),
        camera_target_world=camera_target_world.astype(np.float32),
        camera_pos_isaac=camera_pos_isaac.astype(np.float32),
        camera_target_isaac=camera_target_isaac.astype(np.float32),
        subject_world=subject_world.astype(np.float32),
        subject_isaac=subject_isaac.astype(np.float32),
        T_w2c=t_w2c,
        K_fullimg=k_fullimg,
        horizontal_fov_deg=np.asarray(horizontal_fov_deg, dtype=np.float32),
        world_to_isaac=world_to_isaac.astype(np.float32),
        alignment_offset_world=align_offset.astype(np.float32),
        gravity_axis=np.asarray(args.gravity_axis),
        person_idx=np.asarray(args.person_idx, dtype=np.int32),
    )
    print(
        f"exported GVHMR camera: {output} "
        f"({camera_pos_world.shape[0]} frames, fov={horizontal_fov_deg:.2f} deg, "
        f"person_idx={args.person_idx}, offset={align_offset.tolist()})"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gvhmr_results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference_npz", default=None)
    parser.add_argument("--gravity_axis", default="neg_y", choices=["neg_y", "neg_z"])
    parser.add_argument("--person_idx", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    export_camera(parse_args())
