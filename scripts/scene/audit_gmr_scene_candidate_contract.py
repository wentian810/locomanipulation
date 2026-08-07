#!/usr/bin/env python3
# Audit a proposed robot-scene repair against immutable visual and geometry evidence.
# This tool never edits a motion file.  It is intended as the acceptance gate for
# batch candidates, not as a clip-specific repair.
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _motion_root(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    root = np.asarray(payload.get("root_pos"), dtype=np.float64)
    if root.ndim != 2 or root.shape[1] != 3:
        raise ValueError(f"{path} has no valid root_pos [T,3]")
    return root


def _metric(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def _world_from_isaac(root: np.ndarray, world_to_isaac: np.ndarray) -> np.ndarray:
    if world_to_isaac.shape != (3, 3):
        raise ValueError("camera world_to_isaac must be [3,3]")
    return (np.linalg.inv(world_to_isaac) @ root.T).T


def _project(world: np.ndarray, w2c: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    count = len(world)
    if len(w2c) not in {1, count} or len(intrinsics) not in {1, count}:
        raise ValueError("camera arrays must have one sample or one sample per frame")
    if len(w2c) == 1:
        w2c = np.repeat(w2c, count, axis=0)
    if len(intrinsics) == 1:
        intrinsics = np.repeat(intrinsics, count, axis=0)
    homogeneous = np.concatenate([world, np.ones((count, 1), dtype=np.float64)], axis=1)
    camera = np.einsum("tij,tj->ti", w2c, homogeneous)[:, :3]
    pixels_h = np.einsum("tij,tj->ti", intrinsics, camera)
    depth = pixels_h[:, 2]
    if np.any(np.abs(depth) < 1e-8):
        raise ValueError("robot projects at camera depth zero")
    return pixels_h[:, :2] / depth[:, None]


def _load_keypoints(path: Path) -> np.ndarray:
    import torch

    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("keypoints", "wholebody", "vitpose"):
            if key in payload:
                payload = payload[key]
                break
    keypoints = np.asarray(payload.detach().cpu().numpy() if hasattr(payload, "detach") else payload)
    if keypoints.ndim == 4 and keypoints.shape[0] == 1:
        keypoints = keypoints[0]
    if keypoints.ndim != 3 or keypoints.shape[2] < 3:
        raise ValueError("expected whole-body keypoints [T,J,>=3]")
    return keypoints.astype(np.float64, copy=False)


def _hips(keypoints: np.ndarray, left_index: int, right_index: int, confidence: float) -> tuple[np.ndarray, np.ndarray]:
    if max(left_index, right_index) >= keypoints.shape[1]:
        raise ValueError("configured hip keypoint index is unavailable")
    left, right = keypoints[:, left_index, :3], keypoints[:, right_index, :3]
    weight = np.clip(np.stack([left[:, 2], right[:, 2]], axis=1), 0.0, None)
    weight_sum = weight.sum(axis=1)
    observed = (left[:, :2] * weight[:, :1] + right[:, :2] * weight[:, 1:2]) / np.maximum(weight_sum[:, None], 1e-12)
    valid = np.isfinite(observed).all(axis=1) & (weight.min(axis=1) >= confidence)
    return observed, valid


def _audit_summary(path: Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Static audits report all-geometry results under min_robot_to_seat_distance_m,
    # while the driven mj_step validator reports the explicitly declared semantic
    # support scope under seat_contact.  Normalize both evidence schemas here.
    distance = payload.get("min_robot_to_seat_distance_m", {})
    required = ("penetration_frame_ratio", "minimum")
    if all(key in distance for key in required):
        return {
            "penetration_frame_ratio": float(distance["penetration_frame_ratio"]),
            "minimum_signed_distance_m": float(distance["minimum"]),
        }
    dynamic = payload.get("seat_contact", {})
    dynamic_required = ("penetration_frame_ratio", "minimum_signed_distance_m")
    if all(key in dynamic for key in dynamic_required):
        return {
            "penetration_frame_ratio": float(dynamic["penetration_frame_ratio"]),
            "minimum_signed_distance_m": float(dynamic["minimum_signed_distance_m"]),
        }
    raise ValueError(f"{path} lacks a seat penetration summary")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-motion", required=True, type=Path)
    parser.add_argument("--candidate-motion", required=True, type=Path)
    parser.add_argument("--camera-path", required=True, type=Path)
    parser.add_argument("--keypoints", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--baseline-contact-audit", type=Path)
    parser.add_argument("--candidate-contact-audit", type=Path)
    parser.add_argument("--hip-indices", default="11,12")
    parser.add_argument("--min-keypoint-confidence", type=float, default=0.50)
    parser.add_argument("--max-root-contract-p95-regression-m", type=float, default=0.03)
    parser.add_argument("--max-reprojection-p50-regression-px", type=float, default=0.0)
    parser.add_argument("--max-reprojection-p95-regression-px", type=float, default=0.0)
    parser.add_argument("--max-penetration-ratio-regression", type=float, default=0.0)
    args = parser.parse_args()

    baseline, candidate = _motion_root(args.baseline_motion), _motion_root(args.candidate_motion)
    if baseline.shape != candidate.shape:
        raise ValueError("baseline and candidate root trajectories must share shape")
    camera = np.load(args.camera_path, allow_pickle=True)
    required_camera = ("world_to_isaac", "subject_world", "T_w2c", "K_fullimg")
    missing = [key for key in required_camera if key not in camera]
    if missing:
        raise ValueError(f"camera path missing {missing}")
    subject = np.asarray(camera["subject_world"], dtype=np.float64)
    if subject.shape != baseline.shape:
        raise ValueError("camera subject trajectory does not match robot frame count")
    world_to_isaac = np.asarray(camera["world_to_isaac"], dtype=np.float64)
    baseline_world = _world_from_isaac(baseline, world_to_isaac)
    candidate_world = _world_from_isaac(candidate, world_to_isaac)
    baseline_contract = np.linalg.norm((baseline_world - baseline_world[0]) - (subject - subject[0]), axis=1)
    candidate_contract = np.linalg.norm((candidate_world - candidate_world[0]) - (subject - subject[0]), axis=1)

    keypoints = _load_keypoints(args.keypoints)
    if len(keypoints) != len(baseline):
        raise ValueError("keypoint and robot frame counts differ")
    indices = [int(value.strip()) for value in args.hip_indices.split(",")]
    if len(indices) != 2:
        raise ValueError("hip-indices must be two comma-separated whole-body indices")
    observed, visible = _hips(keypoints, indices[0], indices[1], args.min_keypoint_confidence)
    if int(visible.sum()) < max(10, len(baseline) // 3):
        raise RuntimeError("insufficient confident hip observations for visual-contract audit")
    w2c = np.asarray(camera["T_w2c"], dtype=np.float64)
    intrinsics = np.asarray(camera["K_fullimg"], dtype=np.float64)
    baseline_pixels = _project(baseline_world, w2c, intrinsics)
    candidate_pixels = _project(candidate_world, w2c, intrinsics)
    # Compensate the fixed pelvis/robot morphology offset once, using the
    # unmodified visual reference.  The same offset is applied to every candidate.
    fixed_offset = np.median(observed[visible] - baseline_pixels[visible], axis=0)
    baseline_reprojection = np.linalg.norm(baseline_pixels[visible] + fixed_offset - observed[visible], axis=1)
    candidate_reprojection = np.linalg.norm(candidate_pixels[visible] + fixed_offset - observed[visible], axis=1)

    baseline_contact = _audit_summary(args.baseline_contact_audit)
    candidate_contact = _audit_summary(args.candidate_contact_audit)
    if (baseline_contact is None) != (candidate_contact is None):
        raise ValueError("provide both contact audits or neither")

    base_contract_metric, candidate_contract_metric = _metric(baseline_contract), _metric(candidate_contract)
    base_projection_metric, candidate_projection_metric = _metric(baseline_reprojection), _metric(candidate_reprojection)
    reasons: list[str] = []
    if candidate_contract_metric["p95"] > base_contract_metric["p95"] + args.max_root_contract_p95_regression_m:
        reasons.append("root_trajectory_contract_regressed")
    if candidate_projection_metric["p50"] > base_projection_metric["p50"] + args.max_reprojection_p50_regression_px:
        reasons.append("median_hip_reprojection_regressed")
    if candidate_projection_metric["p95"] > base_projection_metric["p95"] + args.max_reprojection_p95_regression_px:
        reasons.append("p95_hip_reprojection_regressed")
    if baseline_contact is not None and candidate_contact is not None:
        if candidate_contact["penetration_frame_ratio"] > baseline_contact["penetration_frame_ratio"] + args.max_penetration_ratio_regression:
            reasons.append("seat_penetration_ratio_regressed")
        # Reducing a positive air gap is the intended outcome of a scene-side
        # support fit.  Only increased *negative* clearance (penetration depth),
        # not a smaller positive distance, is a regression.
        baseline_penetration_depth = max(0.0, -baseline_contact["minimum_signed_distance_m"])
        candidate_penetration_depth = max(0.0, -candidate_contact["minimum_signed_distance_m"])
        if candidate_penetration_depth > baseline_penetration_depth + 1e-6:
            reasons.append("worst_seat_penetration_regressed")

    report: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "batch_acceptance_gate_for_gmr_scene_repair_candidates",
        "inputs": {
            "baseline_motion": str(args.baseline_motion),
            "baseline_sha256": _sha256(args.baseline_motion),
            "candidate_motion": str(args.candidate_motion),
            "candidate_sha256": _sha256(args.candidate_motion),
            "camera_path": str(args.camera_path),
            "keypoints": str(args.keypoints),
        },
        "evidence_contract": {
            "root_trajectory": "GVHMR subject_world after fixed world_to_isaac conversion",
            "image": "confident whole-body hip observations with a baseline-only fixed morphology offset",
            "scene": "robot-seat signed-distance audit",
            "no_input_motion_modified": True,
        },
        "metrics": {
            "root_trajectory_contract_m": {"baseline": base_contract_metric, "candidate": candidate_contract_metric},
            "hip_reprojection_px": {"baseline": base_projection_metric, "candidate": candidate_projection_metric, "fixed_offset_px": fixed_offset.tolist(), "valid_frames": int(visible.sum())},
            "seat_contact": {"baseline": baseline_contact, "candidate": candidate_contact},
        },
        "thresholds": {
            "max_root_contract_p95_regression_m": args.max_root_contract_p95_regression_m,
            "max_reprojection_p50_regression_px": args.max_reprojection_p50_regression_px,
            "max_reprojection_p95_regression_px": args.max_reprojection_p95_regression_px,
            "max_penetration_ratio_regression": args.max_penetration_ratio_regression,
        },
        "status": "accepted_candidate" if not reasons else "rejected_candidate",
        "rejection_reasons": reasons,
        "routing": "candidate may enter mj_step validation only when accepted_candidate; otherwise retain immutable visual baseline",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
