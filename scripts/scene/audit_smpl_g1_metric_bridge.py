#!/usr/bin/env python3
"""Audit the one global SMPL-to-G1 metric scale without editing either input.

VideoMimic's scene is metrically tied to the reconstructed SMPL body, whereas
GMR contains a fixed-size G1.  A chair should therefore be transferred through
one static similarity transform, not fitted by a time-varying root or chair
offset.  This utility estimates the scale from corresponding lower-limb spans
and records all inputs and per-side measurements for review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import smplx
import torch


SMPL_LEG_CHAINS = {
    "left": (1, 4, 7),
    "right": (2, 5, 8),
}
G1_LEG_CHAINS = {
    "left": ("left_hip_pitch_link", "left_knee_link", "left_ankle_pitch_link"),
    "right": ("right_hip_pitch_link", "right_knee_link", "right_ankle_pitch_link"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def robust_summary(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite) & (finite > 1e-6)]
    if not len(finite):
        raise ValueError("no finite positive metric measurements")
    return {
        "count": int(len(finite)),
        "median_m": float(np.median(finite)),
        "p05_m": float(np.quantile(finite, 0.05)),
        "p95_m": float(np.quantile(finite, 0.95)),
    }


def smpl_joints_z_up(motion: Any, body_model_root: Path, device: str) -> np.ndarray:
    required = {"trans", "root_orient", "pose_body", "betas"}
    missing = required - set(motion.files)
    if missing:
        raise ValueError(f"SMPL motion misses fields: {sorted(missing)}")
    frames = len(motion["trans"])
    gender = str(motion["gender"].item()).lower() if "gender" in motion.files else "neutral"
    if not (body_model_root / "smplh" / f"SMPLH_{gender.upper()}.npz").is_file():
        gender = "neutral"
    model = smplx.create(
        str(body_model_root), model_type="smplh", gender=gender, ext="npz",
        num_betas=len(motion["betas"]), use_pca=False, batch_size=frames,
    ).to(device)
    betas = np.repeat(np.asarray(motion["betas"], dtype=np.float32)[None], frames, axis=0)
    with torch.no_grad():
        output = model(
            betas=torch.as_tensor(betas, dtype=torch.float32, device=device),
            global_orient=torch.as_tensor(motion["root_orient"], dtype=torch.float32, device=device),
            body_pose=torch.as_tensor(motion["pose_body"], dtype=torch.float32, device=device),
            transl=torch.as_tensor(motion["trans"], dtype=torch.float32, device=device),
            return_verts=False,
        )
    # VideoMimic stores the body in neg-Y-up coordinates; its physical scene is
    # the documented [x, z, -y] Z-up conversion.
    joints = output.joints.detach().cpu().numpy()[..., [0, 2, 1]]
    joints[..., 1] *= -1.0
    return joints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smpl-motion", required=True, type=Path)
    parser.add_argument("--robot-motion", required=True, type=Path)
    parser.add_argument("--body-model-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("refusing to overwrite an existing metric-bridge audit")
    if not args.body_model_root.is_dir():
        raise FileNotFoundError(args.body_model_root)

    smpl_motion = np.load(args.smpl_motion, allow_pickle=False)
    smpl_joints = smpl_joints_z_up(smpl_motion, args.body_model_root, args.device)
    with args.robot_motion.open("rb") as handle:
        robot_motion = pickle.load(handle)
    link_names = list(robot_motion.get("link_body_list", []))
    robot_local = np.asarray(robot_motion.get("local_body_pos"), dtype=np.float64)
    if robot_local.ndim != 3 or robot_local.shape[0] != len(smpl_joints) or robot_local.shape[2] != 3:
        raise ValueError("robot local_body_pos must be [same_frames, links, 3]")
    missing = sorted({name for chain in G1_LEG_CHAINS.values() for name in chain} - set(link_names))
    if missing:
        raise ValueError(f"robot link_body_list misses required links: {missing}")
    robot_index = {name: link_names.index(name) for name in link_names}

    sides: dict[str, dict[str, Any]] = {}
    all_ratios: list[np.ndarray] = []
    for side in ("left", "right"):
        s_hip, s_knee, s_ankle = SMPL_LEG_CHAINS[side]
        g_hip, g_knee, g_ankle = (robot_index[name] for name in G1_LEG_CHAINS[side])
        smpl_thigh = np.linalg.norm(smpl_joints[:, s_hip] - smpl_joints[:, s_knee], axis=1)
        smpl_shin = np.linalg.norm(smpl_joints[:, s_knee] - smpl_joints[:, s_ankle], axis=1)
        g1_thigh = np.linalg.norm(robot_local[:, g_hip] - robot_local[:, g_knee], axis=1)
        g1_shin = np.linalg.norm(robot_local[:, g_knee] - robot_local[:, g_ankle], axis=1)
        ratios = np.concatenate((g1_thigh / smpl_thigh, g1_shin / smpl_shin))
        all_ratios.append(ratios)
        sides[side] = {
            "smpl_thigh": robust_summary(smpl_thigh),
            "smpl_shin": robust_summary(smpl_shin),
            "g1_thigh": robust_summary(g1_thigh),
            "g1_shin": robust_summary(g1_shin),
            "g1_over_smpl_scale": robust_summary(ratios),
        }
    scale = robust_summary(np.concatenate(all_ratios))
    report = {
        "schema_version": 1,
        "purpose": "audit_static_smpl_to_g1_metric_similarity",
        "status": "measured_not_applied",
        "policy": {
            "allowed_next_operation": "one_global_similarity_transform_of_static_scene_and_camera_contract",
            "forbidden": ["per_frame_scene_motion", "per_frame_root_height_rewrite", "unrecorded_scale"],
        },
        "inputs": {
            "smpl_motion": str(args.smpl_motion), "smpl_motion_sha256": sha256(args.smpl_motion),
            "robot_motion": str(args.robot_motion), "robot_motion_sha256": sha256(args.robot_motion),
            "body_model_root": str(args.body_model_root),
        },
        "frame_count": int(len(smpl_joints)),
        "metric_basis": "SMPL-H hip-knee-ankle versus G1 hip_pitch-knee-ankle Euclidean spans",
        "sides": sides,
        "recommended_scene_scale_g1_over_smpl": scale,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "scale": scale}, ensure_ascii=False))


if __name__ == "__main__":
    main()
