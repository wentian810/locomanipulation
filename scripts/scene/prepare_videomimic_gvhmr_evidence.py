#!/usr/bin/env python3
"""Export active GVHMR/SMPL evidence to VideoMimic's retargeting input.

This is an evidence adapter, not a motion blend. It converts the same SMPL
motion used by GMR into VideoMimic's 45-keypoint contract in the project's Z-up
(Isaac/MuJoCo) world frame. The downstream optimizer must still pass visual and
MuJoCo gates before it can replace the visual GMR track.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation

# The documented GVHMR Y-up -> Isaac/MuJoCo Z-up change of basis used by GMR.
# It is a coordinate contract, never a fitted per-clip alignment.
GVHMR_TO_ZUP = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)


def _string(value: np.ndarray | str, default: str = "neutral") -> str:
    value = np.asarray(value)
    text = str(value.item()).lower() if value.shape == () else str(value).lower()
    return text if text in {"male", "female", "neutral"} else default


def _betas(value: np.ndarray, count: int = 10) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim > 1:
        value = value.mean(axis=0)
    value = value.reshape(-1)
    return np.pad(value, (0, max(0, count - value.size)))[:count]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smpl-npz", required=True, type=Path)
    parser.add_argument("--body-model-root", required=True, type=Path)
    parser.add_argument("--output-h5", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model-type", choices=("smpl", "smplh", "smplx"), default="smplh")
    parser.add_argument("--person-id", default="1")
    args = parser.parse_args()

    with np.load(args.smpl_npz, allow_pickle=True) as source:
        required = ("pose_body", "root_orient", "trans", "betas")
        absent = [key for key in required if key not in source.files]
        if absent:
            raise ValueError(f"SMPL evidence missing fields: {absent}")
        pose_body = np.asarray(source["pose_body"], dtype=np.float32)
        root_orient = np.asarray(source["root_orient"], dtype=np.float32)
        trans = np.asarray(source["trans"], dtype=np.float32)
        gender = _string(source["gender"] if "gender" in source.files else "neutral")
        fps = float(np.asarray(source["mocap_frame_rate"] if "mocap_frame_rate" in source.files else 30.0).reshape(-1)[0])
        betas = _betas(source["betas"])

    frames = int(trans.shape[0])
    if pose_body.shape != (frames, 63) or root_orient.shape != (frames, 3):
        raise ValueError(
            "Expected pose_body=(T,63), root_orient=(T,3), trans=(T,3); "
            f"got {pose_body.shape}, {root_orient.shape}, {trans.shape}"
        )

    body = smplx.create(
        str(args.body_model_root), args.model_type, gender=gender, use_pca=False,
        num_betas=int(betas.size),
    )
    kwargs: dict[str, torch.Tensor | bool] = {
        "betas": torch.from_numpy(betas).view(1, -1).repeat(frames, 1),
        "global_orient": torch.from_numpy(root_orient),
        "body_pose": torch.from_numpy(pose_body),
        "transl": torch.from_numpy(trans),
        "return_full_pose": False,
    }
    if args.model_type in {"smplh", "smplx"}:
        kwargs["left_hand_pose"] = torch.zeros(frames, 45)
        kwargs["right_hand_pose"] = torch.zeros(frames, 45)
    if args.model_type == "smplx":
        kwargs.update(
            jaw_pose=torch.zeros(frames, 3), leye_pose=torch.zeros(frames, 3),
            reye_pose=torch.zeros(frames, 3), expression=torch.zeros(frames, 10),
        )
    with torch.no_grad():
        output = body(**kwargs)
    smpl_joints = output.joints.detach().cpu().numpy().astype(np.float64)
    if smpl_joints.shape[1] < 24:
        raise RuntimeError(f"Body model returned {smpl_joints.shape[1]} joints; need 24")

    # VideoMimic's first 24 joints are standard SMPL. Face/toe/finger points
    # are unused by G1 factors, so they only preserve known wrist/hand evidence.
    keypoints = np.empty((frames, 45, 3), dtype=np.float64)
    keypoints[:, :24] = smpl_joints[:, :24]
    fallback_indices = np.array(
        [15, 15, 15, 15, 15, 15, 15, 15, 15, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 23, 23]
    )
    keypoints[:, 24:] = smpl_joints[:, fallback_indices]
    keypoints = keypoints @ GVHMR_TO_ZUP.T
    root_mats = np.einsum(
        "ij,tjk->tik", GVHMR_TO_ZUP, Rotation.from_rotvec(root_orient).as_matrix()
    )
    if not np.isfinite(keypoints).all() or not np.isfinite(root_mats).all():
        raise FloatingPointError("Non-finite keypoint or root-orientation evidence")

    args.output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output_h5, "w") as out:
        out.attrs["fps"] = fps
        out.attrs["source_type"] = "gvhmr_smpl_evidence"
        out.attrs["coordinate_contract"] = "gvhmr_y_up_to_isaac_mujoco_z_up"
        out.create_group("joints").create_dataset(args.person_id, data=keypoints)
        out.create_group("root_orient").create_dataset(args.person_id, data=root_mats[:, None])

    manifest = {
        "status": "evidence_exported",
        "source_type": "gvhmr_smpl_evidence",
        "source_npz": str(args.smpl_npz),
        "source_sha256": _sha256(args.smpl_npz),
        "body_model_root": str(args.body_model_root),
        "model_type": args.model_type,
        "person_id": str(args.person_id),
        "frames": frames,
        "fps": fps,
        "coordinate_contract": "gvhmr_y_up_to_isaac_mujoco_z_up",
        "unconstrained_keypoints": "face/toe/finger points preserve wrist/hand evidence and are unused by G1 factors",
        "warning": "Evidence only; candidate replacement requires visual, scene, and mj_step acceptance.",
    }
    manifest_path = args.manifest or args.output_h5.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_h5": str(args.output_h5), "manifest": str(manifest_path), "frames": frames}, ensure_ascii=False))


if __name__ == "__main__":
    main()

