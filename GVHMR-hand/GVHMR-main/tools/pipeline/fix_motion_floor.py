#!/usr/bin/env python
"""Raise an SMPL NPZ so its mesh stays on or above a floor plane."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from smpl_sim.smpllib.smpl_parser import SMPL_Parser


def normalize_poses(poses):
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim == 3:
        poses = poses.reshape(poses.shape[0], -1)
    if poses.shape[1] < 72:
        poses = np.concatenate(
            [poses, np.zeros((poses.shape[0], 72 - poses.shape[1]), dtype=np.float32)],
            axis=1,
        )
    return poses[:, :72].astype(np.float32)


def normalize_betas(betas):
    betas = np.asarray(betas, dtype=np.float32).reshape(-1)
    if betas.shape[0] >= 17 and betas[0] in (0, 1, 2):
        betas = betas[1:]
    if betas.shape[0] < 10:
        betas = np.pad(betas, (0, 10 - betas.shape[0]))
    return betas[:10].astype(np.float32)


def mesh_min_heights(path, model_path, gender, up_axis, chunk_size):
    with np.load(path, allow_pickle=True) as data:
        poses = normalize_poses(data["poses"])
        trans = np.asarray(data["trans"], dtype=np.float32)
        betas = normalize_betas(data["betas"])

    smpl = SMPL_Parser(model_path=model_path, gender=gender)
    betas_t = torch.from_numpy(betas[None, :])
    mins = []
    with torch.no_grad():
        for start in range(0, poses.shape[0], chunk_size):
            end = min(start + chunk_size, poses.shape[0])
            verts, _ = smpl.get_joints_verts(
                pose=torch.from_numpy(poses[start:end]),
                th_trans=torch.from_numpy(trans[start:end]),
                th_betas=betas_t,
            )
            mins.append(verts[..., up_axis].amin(dim=1).detach().cpu().numpy())
    return np.concatenate(mins, axis=0)


def load_payload(path):
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def apply_offsets(payload, offsets, up_axis):
    out = dict(payload)
    trans = np.asarray(out["trans"], dtype=np.float32).copy()
    trans[:, up_axis] += offsets.astype(np.float32)
    out["trans"] = trans

    if "trans_original" in out:
        trans_original = np.asarray(out["trans_original"], dtype=np.float32)
        if trans_original.shape == trans.shape:
            trans_original = trans_original.copy()
            trans_original[:, up_axis] += offsets.astype(np.float32)
            out["trans_original"] = trans_original
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_path", default="data/smpl")
    parser.add_argument("--gender", default="neutral", choices=["neutral", "male", "female"])
    parser.add_argument("--up_axis", type=int, default=1)
    parser.add_argument("--floor", type=float, default=0.0)
    parser.add_argument("--clearance", type=float, default=0.0)
    parser.add_argument("--mode", default="global_min", choices=["global_min", "per_frame"])
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    min_heights = mesh_min_heights(input_path, args.model_path, args.gender, args.up_axis, args.chunk_size)
    target = float(args.floor) + float(args.clearance)
    if args.mode == "per_frame":
        offsets = np.maximum(0.0, target - min_heights).astype(np.float32)
    else:
        offsets = np.full(min_heights.shape, max(0.0, target - float(min_heights.min())), dtype=np.float32)

    payload = apply_offsets(load_payload(input_path), offsets, args.up_axis)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)

    after_min = mesh_min_heights(output_path, args.model_path, args.gender, args.up_axis, args.chunk_size)
    report = {
        "input": input_path.as_posix(),
        "output": output_path.as_posix(),
        "mode": args.mode,
        "floor": float(args.floor),
        "clearance": float(args.clearance),
        "before_min_height": float(min_heights.min()),
        "after_min_height": float(after_min.min()),
        "max_applied_offset": float(offsets.max()) if offsets.size else 0.0,
        "mean_applied_offset": float(offsets.mean()) if offsets.size else 0.0,
    }
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        "floor_fix: "
        f"before_min={report['before_min_height']:.4f}m "
        f"after_min={report['after_min_height']:.4f}m "
        f"offset_max={report['max_applied_offset']:.4f}m "
        f"mode={args.mode}"
    )


if __name__ == "__main__":
    main()
