#!/usr/bin/env python3
"""Audit sitting-phase PHC tracking without changing the chair or motion.

The report separates a reference-SMPL/PHC tracking error from a scene geometry
error.  It is deliberately diagnostic: a raised support is never moved merely
because a controller did not reproduce a crossed-leg pose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# SMPL body-pose joint indices after global orientation.
LOWER_BODY = {
    "left_hip": 1,
    "right_hip": 2,
    "left_knee": 4,
    "right_knee": 5,
    "left_ankle": 7,
    "right_ankle": 8,
}


def _load_motion(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
        if "trans" not in payload or "poses" not in payload:
            raise ValueError(f"{path} must contain trans and poses")
        trans = np.asarray(payload["trans"], dtype=np.float64)
        poses = np.asarray(payload["poses"], dtype=np.float64)
    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"{path}: trans must be [T,3]")
    if poses.ndim == 2:
        if poses.shape[1] % 3:
            raise ValueError(f"{path}: flattened poses must be divisible by 3")
        poses = poses.reshape(len(poses), -1, 3)
    if poses.ndim != 3 or poses.shape[0] != len(trans) or poses.shape[2] != 3:
        raise ValueError(f"{path}: poses must be [T,J,3]")
    return trans, poses


def _z_up(trans_neg_y: np.ndarray) -> np.ndarray:
    """Pipeline contract: neg-Y-up SMPL translation -> MuJoCo/PHC Z-up."""
    return np.stack(
        (trans_neg_y[:, 0], -trans_neg_y[:, 2], trans_neg_y[:, 1]),
        axis=1,
    )


def _rotmat(rotvec: np.ndarray) -> np.ndarray:
    """Vectorized Rodrigues formula for [...,3] axis-angle values."""
    theta = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    axis = np.divide(rotvec, theta, out=np.zeros_like(rotvec), where=theta > 1e-9)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zeros = np.zeros_like(x)
    k = np.stack(
        (
            zeros, -z, y,
            z, zeros, -x,
            -y, x, zeros,
        ),
        axis=-1,
    ).reshape(*axis.shape[:-1], 3, 3)
    identity = np.broadcast_to(np.eye(3), k.shape)
    sin_theta = np.sin(theta)[..., None]
    one_minus_cos = (1.0 - np.cos(theta))[..., None]
    return identity + sin_theta * k + one_minus_cos * (k @ k)


def _rotation_error_deg(reference: np.ndarray, replay: np.ndarray) -> np.ndarray:
    ref_mat = _rotmat(reference)
    replay_mat = _rotmat(replay)
    relative = np.swapaxes(ref_mat, -1, -2) @ replay_mat
    cosine = np.clip(
        (np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0
    )
    return np.degrees(np.arccos(cosine))


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit PHC sitting tracking against its pre-PHC SMPL reference."
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--contact-anchors", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--root-height-warn-m", type=float, default=0.10)
    parser.add_argument("--lower-body-warn-deg", type=float, default=25.0)
    args = parser.parse_args()

    ref_trans, ref_poses = _load_motion(args.reference)
    replay_trans, replay_poses = _load_motion(args.replay)
    if ref_trans.shape != replay_trans.shape or ref_poses.shape != replay_poses.shape:
        raise ValueError(
            "reference and replay must share [T,3] translations and [T,J,3] poses; "
            f"got {ref_trans.shape}/{ref_poses.shape} vs "
            f"{replay_trans.shape}/{replay_poses.shape}"
        )
    if ref_poses.shape[1] <= max(LOWER_BODY.values()):
        raise ValueError("motion does not contain the required SMPL lower-body joints")

    anchors = np.load(args.contact_anchors)
    sit_mask = np.asarray(anchors["sit_mask"], dtype=bool)
    if sit_mask.shape != (len(ref_trans),) or not np.any(sit_mask):
        raise ValueError("contact anchors must provide a non-empty sit_mask [T]")

    ref_z_up = _z_up(ref_trans)
    replay_z_up = _z_up(replay_trans)
    root_delta = replay_z_up - ref_z_up
    rotation_error = _rotation_error_deg(ref_poses, replay_poses)

    sit_root = root_delta[sit_mask]
    lower_by_joint = {
        name: _stats(rotation_error[sit_mask, index])
        for name, index in LOWER_BODY.items()
    }
    lower_values = rotation_error[sit_mask][:, list(LOWER_BODY.values())]

    vertical = sit_root[:, 2]
    horizontal = np.linalg.norm(sit_root[:, :2], axis=1)
    lower_p95 = float(np.quantile(lower_values, 0.95))
    warnings: list[str] = []
    if abs(float(np.median(vertical))) > args.root_height_warn_m:
        warnings.append("root_height_tracking_bias")
    if lower_p95 > args.lower_body_warn_deg:
        warnings.append("lower_body_pose_tracking_error")
    if not warnings:
        status = "pass"
    else:
        status = "warn"

    report = {
        "schema_version": 1,
        "purpose": "separate_phc_sitting_tracking_error_from_scene_height_error",
        "status": status,
        "reference": str(args.reference),
        "replay": str(args.replay),
        "contact_anchors": str(args.contact_anchors),
        "coordinate_contract": "input_neg_y_up_to_z_up=[x,-z,y]",
        "sit_frame_count": int(sit_mask.sum()),
        "sit_frame_range": [
            int(np.flatnonzero(sit_mask)[0]),
            int(np.flatnonzero(sit_mask)[-1]),
        ],
        "root_delta_z_up_m": {
            "vertical": _stats(vertical),
            "horizontal_norm": _stats(horizontal),
        },
        "lower_body_rotation_error_deg": {
            "all_selected": _stats(lower_values.reshape(-1)),
            "by_joint": lower_by_joint,
        },
        "thresholds": {
            "root_height_warn_m": args.root_height_warn_m,
            "lower_body_warn_deg": args.lower_body_warn_deg,
        },
        "warnings": warnings,
        "interpretation": (
            "A root-height warning identifies PHC reference/replay mismatch, "
            "not evidence that the reconstructed seat should be moved. "
            "A lower-body warning is expected to capture difficult poses such "
            "as crossed legs and should be addressed in PHC retargeting/control."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

