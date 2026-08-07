#!/usr/bin/env python3
"""Export immutable SMPL-H joints in VideoMimic's documented Z-up frame."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import smplx
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smpl-motion", required=True, type=Path)
    parser.add_argument("--body-model-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite an existing SMPL-joint export")
    motion = np.load(args.smpl_motion, allow_pickle=False)
    required = {"trans", "root_orient", "pose_body", "betas"}
    missing = required - set(motion.files)
    if missing:
        raise ValueError(f"SMPL motion misses fields: {sorted(missing)}")
    frames = len(motion["trans"])
    gender = str(motion["gender"].item()).lower() if "gender" in motion.files else "neutral"
    if not (args.body_model_root / "smplh" / f"SMPLH_{gender.upper()}.npz").is_file():
        gender = "neutral"
    model = smplx.create(
        str(args.body_model_root), model_type="smplh", gender=gender, ext="npz",
        num_betas=len(motion["betas"]), use_pca=False, batch_size=frames,
    ).to(args.device)
    betas = np.repeat(np.asarray(motion["betas"], dtype=np.float32)[None], frames, axis=0)
    with torch.no_grad():
        output = model(
            betas=torch.as_tensor(betas, dtype=torch.float32, device=args.device),
            global_orient=torch.as_tensor(motion["root_orient"], dtype=torch.float32, device=args.device),
            body_pose=torch.as_tensor(motion["pose_body"], dtype=torch.float32, device=args.device),
            transl=torch.as_tensor(motion["trans"], dtype=torch.float32, device=args.device),
            return_verts=False,
        )
    joints_zup = output.joints.detach().cpu().numpy()[..., [0, 2, 1]]
    joints_zup[..., 1] *= -1.0  # [x,y,z] neg-Y-up -> [x,z,-y] Z-up
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, joints_zup=joints_zup.astype(np.float32))
    report = {
        "schema_version": 1,
        "purpose": "immutable_smplh_joint_export_for_metric_bridge",
        "status": "exported",
        "coordinate_system": "mujoco_world_z_up_from_neg_y_input",
        "inputs": {"smpl_motion": str(args.smpl_motion), "smpl_motion_sha256": sha256(args.smpl_motion), "body_model_root": str(args.body_model_root)},
        "output": str(args.output), "frame_count": int(frames), "joint_count": int(joints_zup.shape[1]),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
