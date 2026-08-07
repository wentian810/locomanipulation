#!/usr/bin/env python3
"""Audit whether GVHMR and PHC can contribute compatible motion evidence.

This tool deliberately does *not* merge motion files and never edits either
input.  It estimates a robust global similarity only for comparison, measures
the disagreement by frame/joint, and routes each source to the narrowest
evidence role justified by the data.  Downstream scene fitting consumes these
roles as factors, never as a blind field-wise npz splice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


# SMPL pose_body order after the root: hips, spine, knees, ankles, feet, ...
# These indices are used only to characterize lower-limb agreement, not to
# overwrite a joint from PHC.
_SUPPORT_POSE_BODY_INDICES = np.asarray([0, 1, 3, 4, 6, 7, 9, 10], dtype=int)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rot_error(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    a = Rotation.from_rotvec(np.asarray(first).reshape(-1, 3))
    b = Rotation.from_rotvec(np.asarray(second).reshape(-1, 3))
    return (a.inv() * b).magnitude().reshape(np.asarray(first).shape[:-1])


def _fit_similarity(source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src, dst = source[mask], target[mask]
    if len(src) < 4:
        raise ValueError("similarity fit needs at least four inlier frames")
    src_center, dst_center = src.mean(axis=0), dst.mean(axis=0)
    src_zero, dst_zero = src - src_center, dst - dst_center
    left, _, right = np.linalg.svd(src_zero.T @ dst_zero)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0:
        right[-1] *= -1
        rotation = right.T @ left.T
    denominator = float(np.sum(src_zero * src_zero))
    if denominator < 1e-10:
        raise ValueError("source root trajectory is degenerate")
    scale = float(np.sum((src_zero @ rotation) * dst_zero) / denominator)
    translation = dst_center - scale * (src_center @ rotation)
    return scale, rotation, translation


def _robust_similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = np.ones(len(source), dtype=bool)
    for _ in range(4):
        scale, rotation, translation = _fit_similarity(source, target, mask)
        aligned = scale * (source @ rotation) + translation
        residual = np.linalg.norm(aligned - target, axis=1)
        # Trim only the worst fifth.  The retained frames are evidence used to
        # compare coordinate contracts, never an instruction to change motion.
        cutoff = float(np.quantile(residual, 0.80))
        next_mask = residual <= max(cutoff, 1e-6)
        if np.array_equal(next_mask, mask):
            break
        mask = next_mask
    scale, rotation, translation = _fit_similarity(source, target, mask)
    aligned = scale * (source @ rotation) + translation
    return scale, rotation, translation, aligned, mask


def _quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(values.max()),
    }


def _temporal_root_metrics(trans: np.ndarray, fps: float) -> dict[str, float]:
    velocity = np.diff(trans, axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    return {
        "speed_p50_m_s": float(np.quantile(np.linalg.norm(velocity, axis=1), 0.50)),
        "speed_p95_m_s": float(np.quantile(np.linalg.norm(velocity, axis=1), 0.95)),
        "acceleration_p95_m_s2": float(np.quantile(np.linalg.norm(acceleration, axis=1), 0.95)),
        "acceleration_max_m_s2": float(np.linalg.norm(acceleration, axis=1).max()),
    }


def _load_motion(path: Path) -> dict[str, np.ndarray]:
    archive = np.load(path, allow_pickle=True)
    required = ("trans", "root_orient", "pose_body")
    missing = [name for name in required if name not in archive]
    if missing:
        raise ValueError(f"{path} missing {missing}")
    data = {name: np.asarray(archive[name], dtype=np.float64) for name in required}
    data["fps"] = float(np.asarray(archive["mocap_frame_rate"]).item()) if "mocap_frame_rate" in archive else 30.0
    if data["trans"].ndim != 2 or data["trans"].shape[1] != 3:
        raise ValueError(f"{path} has invalid trans shape {data['trans'].shape}")
    if len(data["root_orient"]) != len(data["trans"]) or len(data["pose_body"]) != len(data["trans"]):
        raise ValueError(f"{path} has inconsistent frame counts")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gvhmr-motion", required=True, type=Path)
    parser.add_argument("--phc-motion", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--output-arrays", required=True, type=Path)
    # These are intentionally conservative provisional gates.  Dataset-level
    # calibration may tighten them; reports always retain the raw metrics.
    parser.add_argument("--full-pose-p95-rad", type=float, default=0.25)
    parser.add_argument("--root-orient-p95-rad", type=float, default=0.25)
    parser.add_argument("--root-path-p95-m", type=float, default=0.08)
    parser.add_argument("--support-joint-p95-rad", type=float, default=0.30)
    args = parser.parse_args()

    gvh = _load_motion(args.gvhmr_motion)
    phc = _load_motion(args.phc_motion)
    if len(gvh["trans"]) != len(phc["trans"]):
        raise ValueError("time resampling is an upstream responsibility; frame counts differ")
    if abs(gvh["fps"] - phc["fps"]) > 1e-6:
        raise ValueError("frame rates differ; frame correspondence is not established")
    frame_count = len(gvh["trans"])
    if gvh["pose_body"].shape != phc["pose_body"].shape:
        raise ValueError("pose_body layouts differ; no joint-level comparison is safe")

    scale, rotation, translation, phc_aligned, inliers = _robust_similarity(phc["trans"], gvh["trans"])
    root_path_error = np.linalg.norm(phc_aligned - gvh["trans"], axis=1)
    root_rotation_error = _rot_error(gvh["root_orient"], phc["root_orient"])
    body_shape = gvh["pose_body"].shape
    body_error = _rot_error(
        gvh["pose_body"].reshape(frame_count, -1, 3),
        phc["pose_body"].reshape(frame_count, -1, 3),
    )
    frame_body_error = np.median(body_error, axis=1)
    support_indices = _SUPPORT_POSE_BODY_INDICES[_SUPPORT_POSE_BODY_INDICES < body_error.shape[1]]
    support_error = body_error[:, support_indices]
    phc_pose_weight = np.exp(-(body_error / args.full_pose_p95_rad) ** 2)
    phc_frame_weight = np.median(phc_pose_weight, axis=1)

    full_pose_usable = bool(
        np.quantile(frame_body_error, 0.95) <= args.full_pose_p95_rad
        and np.quantile(root_rotation_error, 0.95) <= args.root_orient_p95_rad
        and np.quantile(root_path_error, 0.95) <= args.root_path_p95_m
    )
    root_path_usable = bool(
        np.quantile(root_path_error, 0.95) <= args.root_path_p95_m
        and np.quantile(root_rotation_error, 0.95) <= args.root_orient_p95_rad
    )
    # This is deliberately only a candidate role.  A later factor must verify
    # foot geometry/velocity against the floor and image before it can create a
    # support constraint.
    support_candidate = bool(np.quantile(support_error, 0.95) <= args.support_joint_p95_rad)

    report: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "audit_gvhmr_phc_motion_evidence_without_motion_fusion",
        "frame_count": int(frame_count),
        "fps": float(gvh["fps"]),
        "inputs": {
            "gvhmr_motion": str(args.gvhmr_motion),
            "gvhmr_sha256": _sha256(args.gvhmr_motion),
            "phc_motion": str(args.phc_motion),
            "phc_sha256": _sha256(args.phc_motion),
        },
        "comparison_similarity_phc_to_gvhmr": {
            "scale": scale,
            "rotation_matrix": rotation.tolist(),
            "translation_xyz": translation.tolist(),
            "inlier_frame_count": int(inliers.sum()),
            "inlier_ratio": float(inliers.mean()),
            "note": "comparison-only coordinate registration; never applied to either input file",
        },
        "disagreement": {
            "root_path_after_similarity_m": _quantiles(root_path_error),
            "root_orientation_rad": _quantiles(root_rotation_error),
            "body_orientation_frame_median_rad": _quantiles(frame_body_error),
            "support_joint_orientation_rad": _quantiles(support_error.reshape(-1)),
            "body_joint_median_rad": np.median(body_error, axis=0).tolist(),
            "gvhmr_temporal_root": _temporal_root_metrics(gvh["trans"], gvh["fps"]),
            "phc_temporal_root": _temporal_root_metrics(phc["trans"], phc["fps"]),
        },
        "provisional_thresholds": {
            "full_pose_p95_rad": args.full_pose_p95_rad,
            "root_orient_p95_rad": args.root_orient_p95_rad,
            "root_path_p95_m": args.root_path_p95_m,
            "support_joint_p95_rad": args.support_joint_p95_rad,
            "calibration_note": "thresholds must be calibrated on a labelled multi-action validation set",
        },
        "routing": {
            "relative_pose_provider": "phc" if full_pose_usable else "gvhmr",
            "phc_root_translation_usable": root_path_usable,
            "phc_full_pose_usable": full_pose_usable,
            "phc_support_event_candidate": support_candidate,
            "phc_support_rule": (
                "requires independent floor/foot-geometry and image-consistency gates"
                if support_candidate else
                "not admissible even as support evidence"
            ),
            "fusion_rule": (
                "factor-level only: retain GVHMR pose and consume admitted PHC observations as weighted constraints; never concatenate npz fields"
            ),
        },
        "output_arrays": str(args.output_arrays),
        "input_motion_modified": False,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_arrays,
        phc_to_gvhmr_scale=np.asarray(scale),
        phc_to_gvhmr_rotation=rotation,
        phc_to_gvhmr_translation=translation,
        robust_inlier_mask=inliers,
        root_path_error_m=root_path_error,
        root_orientation_error_rad=root_rotation_error,
        body_orientation_error_rad=body_error,
        phc_pose_weight=phc_pose_weight,
        phc_frame_weight=phc_frame_weight,
    )
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

