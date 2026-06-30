#!/usr/bin/env python3
"""Build an image-space object proxy from wholebody hand keypoints.

This is a deliberately small bootstrap for interaction diagnostics. It does not
perform object recognition; it estimates a 2D proxy box around the region between
the two hands, which can be overlaid on GVHMR in-camera renders.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def as_np(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_vitpose(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    arr = as_np(data)
    if arr.ndim != 4 or arr.shape[-2] < 133:
        raise ValueError(f"Expected vitpose_wholebody shape (P,F,133,3), got {arr.shape}")
    return arr.astype(np.float32)


def choose_person(vitpose, person_idx):
    if person_idx < 0:
        scores = vitpose[:, :, :17, 2].mean(axis=(1, 2))
        person_idx = int(np.argmax(scores))
    return vitpose[person_idx]


def valid_points(keyp, conf_thr, low_conf_thr, hi_min_keypoints):
    finite_xy = np.isfinite(keyp[:, :2]).all(axis=1)
    conf = np.where(finite_xy & np.isfinite(keyp[:, 2]), keyp[:, 2], -np.inf)
    high = conf > conf_thr
    low = conf > low_conf_thr
    return high if int(high.sum()) >= hi_min_keypoints else low


def smooth_boxes(boxes, valid, window):
    if window <= 1:
        return boxes
    if window % 2 == 0:
        window += 1
    radius = window // 2
    out = boxes.copy()
    for col in range(4):
        values = boxes[:, col].copy()
        good = valid & np.isfinite(values)
        if not np.any(good):
            continue
        idx = np.arange(values.shape[0])
        values[~good] = np.interp(idx[~good], idx[good], values[good])
        padded = np.pad(values, (radius, radius), mode="edge")
        kernel = np.ones(window, dtype=np.float64) / float(window)
        out[:, col] = np.convolve(padded, kernel, mode="valid")
    return out


def build(args):
    vitpose = choose_person(load_vitpose(args.vitpose_wholebody), args.person_idx)
    frame_count = vitpose.shape[0]
    boxes = np.zeros((frame_count, 4), dtype=np.float32)
    valid = np.zeros(frame_count, dtype=bool)
    centers = np.zeros((frame_count, 2), dtype=np.float32)

    for idx in range(frame_count):
        left = vitpose[idx, -42:-21]
        right = vitpose[idx, -21:]
        left_valid = valid_points(left, args.conf_thr, args.low_conf_thr, args.hi_min_keypoints)
        right_valid = valid_points(right, args.conf_thr, args.low_conf_thr, args.hi_min_keypoints)
        pts = []
        if int(left_valid.sum()) >= args.min_keypoints:
            pts.append(left[left_valid, :2])
        if int(right_valid.sum()) >= args.min_keypoints:
            pts.append(right[right_valid, :2])
        if not pts:
            boxes[idx] = np.nan
            centers[idx] = np.nan
            continue
        pts = np.concatenate(pts, axis=0)
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        width = max(float(x2 - x1) * args.scale, args.min_size)
        height = max(float(y2 - y1) * args.scale, args.min_size * args.aspect)
        if args.square:
            side = max(width, height)
            width = height = side
        boxes[idx] = [cx - width * 0.5, cy - height * 0.5, cx + width * 0.5, cy + height * 0.5]
        centers[idx] = [cx, cy]
        valid[idx] = True

    boxes = smooth_boxes(boxes, valid, args.smooth_window).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        schema_version=np.asarray(1, dtype=np.int32),
        source="vitpose_between_hands_proxy",
        coordinate_system="image_xyxy",
        object_type=np.asarray(args.object_type),
        bbox_xyxy=boxes,
        center_2d=centers,
        valid=valid,
    )
    summary = {
        "output": str(args.output),
        "frames": int(frame_count),
        "valid_frames": int(valid.sum()),
        "object_type": args.object_type,
        "bbox_p50": np.nanpercentile(boxes, 50, axis=0).round(2).tolist(),
        "bbox_p05": np.nanpercentile(boxes, 5, axis=0).round(2).tolist(),
        "bbox_p95": np.nanpercentile(boxes, 95, axis=0).round(2).tolist(),
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved image object proxy: {args.output}")
    print(f"Saved summary: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vitpose_wholebody", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--person_idx", type=int, default=-1)
    parser.add_argument("--object_type", choices=["box", "sphere", "cylinder"], default="box")
    parser.add_argument("--conf_thr", type=float, default=0.5)
    parser.add_argument("--low_conf_thr", type=float, default=0.2)
    parser.add_argument("--hi_min_keypoints", type=int, default=6)
    parser.add_argument("--min_keypoints", type=int, default=4)
    parser.add_argument("--min_size", type=float, default=80.0)
    parser.add_argument("--scale", type=float, default=1.15)
    parser.add_argument("--aspect", type=float, default=0.75)
    parser.add_argument("--square", action="store_true")
    parser.add_argument("--smooth_window", type=int, default=9)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
