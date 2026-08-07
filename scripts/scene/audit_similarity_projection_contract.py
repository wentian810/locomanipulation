#!/usr/bin/env python3
"""Verify that a static Sim(3) scene/camera pair preserves image projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BOX_EDGES = (
    (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
)


def corners(item: dict[str, object]) -> np.ndarray:
    center = np.asarray(item["center"], dtype=np.float64)
    rotation = np.asarray(item["rotation_matrix"], dtype=np.float64)
    half = 0.5 * np.asarray(item["extents"], dtype=np.float64)
    signs = np.asarray([
        (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
        (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1),
    ], dtype=np.float64)
    return center + (signs * half) @ rotation.T


def edge_points(items: list[dict[str, object]], samples_per_edge: int) -> np.ndarray:
    pieces = []
    for item in items:
        box = corners(item)
        for begin, end in BOX_EDGES:
            start, finish = box[begin], box[end]
            # Compare corresponding normalized points, rather than a fixed
            # metre spacing: a valid Sim(3) intentionally changes edge length.
            pieces.append(np.linspace(start, finish, samples_per_edge))
    return np.concatenate(pieces)


def project(points: np.ndarray, transform: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    homogeneous = np.column_stack((points, np.ones(len(points))))
    camera = (transform @ homogeneous.T).T[:, :3]
    positive = camera[:, 2] > 1e-6
    pixels_h = (intrinsic @ camera.T).T
    return pixels_h[:, :2] / np.maximum(pixels_h[:, 2:3], 1e-12), positive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-primitives", required=True, type=Path)
    parser.add_argument("--candidate-primitives", required=True, type=Path)
    parser.add_argument("--baseline-camera", required=True, type=Path)
    parser.add_argument("--candidate-camera", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--samples-per-edge", type=int, default=9)
    parser.add_argument("--max-p95-pixel-error", type=float, default=1e-4)
    parser.add_argument("--max-pixel-error", type=float, default=1e-3)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("refusing to overwrite an existing projection contract")
    base = json.loads(args.baseline_primitives.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate_primitives.read_text(encoding="utf-8"))
    base_by_name = {item["name"]: item for item in base.get("primitives", [])}
    candidate_by_name = {item["name"]: item for item in candidate.get("primitives", [])}
    if set(base_by_name) != set(candidate_by_name) or not base_by_name:
        raise ValueError("baseline and candidate must contain the same nonempty primitive names")
    if args.samples_per_edge < 2:
        raise ValueError("samples-per-edge must be at least two")
    baseline_points = edge_points([base_by_name[name] for name in sorted(base_by_name)], args.samples_per_edge)
    candidate_points = edge_points([candidate_by_name[name] for name in sorted(candidate_by_name)], args.samples_per_edge)
    if baseline_points.shape != candidate_points.shape:
        raise RuntimeError("static primitive edge sample counts differ")
    base_camera = np.load(args.baseline_camera)
    candidate_camera = np.load(args.candidate_camera)
    base_w2c, base_k = np.asarray(base_camera["T_w2c"]), np.asarray(base_camera["K_fullimg"])
    candidate_w2c, candidate_k = np.asarray(candidate_camera["T_w2c"]), np.asarray(candidate_camera["K_fullimg"])
    if base_w2c.shape != candidate_w2c.shape or base_k.shape != candidate_k.shape or len(base_w2c) != len(base_k):
        raise ValueError("baseline/candidate camera arrays must have matching frame shapes")
    errors: list[np.ndarray] = []
    positive_counts: list[int] = []
    for frame in range(len(base_w2c)):
        base_pixels, base_positive = project(baseline_points, base_w2c[frame], base_k[frame])
        candidate_pixels, candidate_positive = project(candidate_points, candidate_w2c[frame], candidate_k[frame])
        visible = base_positive & candidate_positive
        if np.any(visible):
            errors.append(np.linalg.norm(base_pixels[visible] - candidate_pixels[visible], axis=1))
            positive_counts.append(int(np.sum(visible)))
    values = np.concatenate(errors) if errors else np.empty(0)
    if not len(values):
        raise RuntimeError("no primitive edge samples are in front of both cameras")
    summary = {"count": int(len(values)), "p50_px": float(np.quantile(values, .5)), "p95_px": float(np.quantile(values, .95)), "max_px": float(np.max(values))}
    accepted = bool(summary["p95_px"] <= args.max_p95_pixel_error and summary["max_px"] <= args.max_pixel_error)
    report = {
        "schema_version": 1,
        "purpose": "static_similarity_scene_camera_projection_contract",
        "status": "accepted_projection_invariance" if accepted else "rejected_projection_invariance",
        "policy": "audit_only_no_scene_or_motion_modified",
        "inputs": {"baseline_primitives": str(args.baseline_primitives), "candidate_primitives": str(args.candidate_primitives), "baseline_camera": str(args.baseline_camera), "candidate_camera": str(args.candidate_camera)},
        "frames": int(len(base_w2c)), "visible_samples_per_frame_min": int(min(positive_counts)),
        "pixel_error": summary,
        "thresholds": {"max_p95_pixel_error": args.max_p95_pixel_error, "max_pixel_error": args.max_pixel_error},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
