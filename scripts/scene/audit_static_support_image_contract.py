#!/usr/bin/env python3
"""Audit whether a static semantic-support adjustment remains image-consistent.

A static scene correction is admissible only when it does not worsen agreement
between projected semantic support edges and image edges on frames where the
support is not substantially occluded by the person.  This is a gate, never a
scene or motion editor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch


_BOX_EDGES = (
    (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
)


def _load_keypoints(path):
    value = torch.load(path, map_location="cpu")
    if isinstance(value, dict):
        value = next((value[k] for k in ("keypoints", "wholebody", "vitpose") if k in value), value)
    value = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 3 or value.shape[2] < 3:
        raise ValueError("expected whole-body keypoints [T,J,>=3]")
    return value.astype(np.float64)


def _corners(item):
    center = np.asarray(item["center"], dtype=np.float64)
    rotation = np.asarray(item["rotation_matrix"], dtype=np.float64)
    half = 0.5 * np.asarray(item["extents"], dtype=np.float64)
    signs = np.asarray([
        (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
        (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1),
    ], dtype=np.float64)
    return center + (signs * half) @ rotation.T


def _sample_box_edges(item, step_m):
    corners = _corners(item)
    samples = []
    for begin, end in _BOX_EDGES:
        first, second = corners[begin], corners[end]
        count = max(2, int(np.ceil(np.linalg.norm(second - first) / step_m)) + 1)
        samples.append(np.linspace(first, second, count))
    return np.concatenate(samples)


def _points(primitives, names, step_m):
    selected = [item for item in primitives if item.get("name") in names]
    if not selected:
        raise ValueError("requested primitives are absent: " + ",".join(sorted(names)))
    return np.concatenate([_sample_box_edges(item, step_m) for item in selected])


def _project(points, w2c, intrinsic):
    homogeneous = np.column_stack([points, np.ones(len(points))])
    camera = (w2c @ homogeneous.T).T[:, :3]
    positive = camera[:, 2] > 1e-5
    projected = (intrinsic @ camera.T).T
    return projected[:, :2] / np.maximum(projected[:, 2:3], 1e-12), positive


def _human_box(keypoints, margin, width, height):
    confident = keypoints[:, 2] >= 0.35
    values = keypoints[confident, :2]
    if not len(values):
        return None
    lo = np.maximum(np.floor(values.min(axis=0) - margin), 0).astype(int)
    hi = np.minimum(np.ceil(values.max(axis=0) + margin), [width - 1, height - 1]).astype(int)
    return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])


def _edge_distance(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grey, 60, 140)
    return cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)


def _score(points, transform, intrinsic, distance, human_box, width, height):
    pixels, positive = _project(points, transform, intrinsic)
    inside = (
        positive
        & (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    )
    xi = np.clip(np.rint(pixels[:, 0]).astype(int), 0, width - 1)
    yi = np.clip(np.rint(pixels[:, 1]).astype(int), 0, height - 1)
    if human_box is not None:
        x0, y0, x1, y1 = human_box
        inside &= ~((xi >= x0) & (xi <= x1) & (yi >= y0) & (yi <= y1))
    values = distance[yi[inside], xi[inside]]
    return values, int(np.sum(inside))


def _summary(values):
    return {
        "count": int(len(values)),
        "p50_px": float(np.quantile(values, 0.5)),
        "p90_px": float(np.quantile(values, 0.9)),
        "mean_px": float(np.mean(values)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--camera-path", required=True, type=Path)
    p.add_argument("--keypoints", required=True, type=Path)
    p.add_argument("--baseline-primitives", required=True, type=Path)
    p.add_argument("--candidate-primitives", required=True, type=Path)
    p.add_argument("--report", required=True, type=Path)
    p.add_argument("--primitive-names", default="seat_support,backrest")
    p.add_argument("--max-frames", type=int, default=32)
    p.add_argument("--edge-step-m", type=float, default=0.025)
    p.add_argument("--human-margin-px", type=float, default=24.0)
    p.add_argument("--min-points-per-frame", type=int, default=24)
    p.add_argument("--min-total-points", type=int, default=400)
    p.add_argument("--max-p50-regression-px", type=float, default=0.75)
    p.add_argument("--max-p90-regression-px", type=float, default=1.5)
    a = p.parse_args()
    if a.max_frames < 1 or a.edge_step_m <= 0 or a.min_points_per_frame < 1:
        raise ValueError("invalid sampling arguments")
    camera = np.load(a.camera_path)
    for key in ("T_w2c", "K_fullimg"):
        if key not in camera:
            raise ValueError("camera path lacks " + key)
    w2c = np.asarray(camera["T_w2c"], dtype=np.float64)
    intrinsic = np.asarray(camera["K_fullimg"], dtype=np.float64)
    keypoints = _load_keypoints(a.keypoints)
    if len(w2c) != len(keypoints) or len(intrinsic) != len(keypoints):
        raise ValueError("camera and keypoint frame counts differ")
    names = set(item.strip() for item in a.primitive_names.split(",") if item.strip())
    baseline = json.loads(a.baseline_primitives.read_text(encoding="utf-8"))
    candidate = json.loads(a.candidate_primitives.read_text(encoding="utf-8"))
    base_points = _points(baseline["primitives"], names, a.edge_step_m)
    candidate_points = _points(candidate["primitives"], names, a.edge_step_m)
    cap = cv2.VideoCapture(str(a.video))
    if not cap.isOpened():
        raise RuntimeError("cannot read video " + str(a.video))
    try:
        count = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(keypoints))
        indices = np.unique(np.linspace(0, count - 1, min(a.max_frames, count)).round().astype(int))
        base_values, candidate_values, frames = [], [], []
        for frame in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
            ok, image = cap.read()
            if not ok:
                continue
            height, width = image.shape[:2]
            distance = _edge_distance(image)
            box = _human_box(keypoints[frame], a.human_margin_px, width, height)
            base, nbase = _score(base_points, w2c[frame], intrinsic[frame], distance, box, width, height)
            cand, ncand = _score(candidate_points, w2c[frame], intrinsic[frame], distance, box, width, height)
            if min(nbase, ncand) < a.min_points_per_frame:
                continue
            base_values.append(base)
            candidate_values.append(cand)
            frames.append({"frame": int(frame), "baseline_points": nbase, "candidate_points": ncand})
        base_values = np.concatenate(base_values) if base_values else np.empty(0)
        candidate_values = np.concatenate(candidate_values) if candidate_values else np.empty(0)
    finally:
        cap.release()
    report = {
        "schema_version": 1,
        "purpose": "static_semantic_support_image_edge_contract",
        "policy": "audit_only_no_motion_or_scene_modified",
        "inputs": {
            "video": str(a.video),
            "baseline_primitives": str(a.baseline_primitives),
            "candidate_primitives": str(a.candidate_primitives),
            "primitive_names": sorted(names),
        },
        "sampled_frames": frames,
        "baseline": _summary(base_values) if len(base_values) else None,
        "candidate": _summary(candidate_values) if len(candidate_values) else None,
        "thresholds": {
            "min_total_points": a.min_total_points,
            "max_p50_regression_px": a.max_p50_regression_px,
            "max_p90_regression_px": a.max_p90_regression_px,
        },
    }
    if min(len(base_values), len(candidate_values)) < a.min_total_points:
        report["status"] = "insufficient_unoccluded_scene_edge_evidence"
    else:
        base_metric, cand_metric = _summary(base_values), _summary(candidate_values)
        reasons = []
        if cand_metric["p50_px"] > base_metric["p50_px"] + a.max_p50_regression_px:
            reasons.append("median_edge_alignment_regressed")
        if cand_metric["p90_px"] > base_metric["p90_px"] + a.max_p90_regression_px:
            reasons.append("p90_edge_alignment_regressed")
        report["status"] = "accepted_image_contract" if not reasons else "rejected_image_contract"
        report["rejection_reasons"] = reasons
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

