#!/usr/bin/env python3
"""Overlay an object proxy on an image-space video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def load_proxy(path):
    data = np.load(path, allow_pickle=True)
    if "bbox_xyxy" not in data:
        raise ValueError(f"{path} must contain bbox_xyxy")
    boxes = np.asarray(data["bbox_xyxy"], dtype=np.float32)
    valid = np.asarray(data["valid"], dtype=bool) if "valid" in data else np.isfinite(boxes).all(axis=1)
    object_type = str(np.asarray(data["object_type"] if "object_type" in data else "box").reshape(-1)[0])
    return boxes, valid, object_type


def parse_color(text):
    parts = [int(float(x)) for x in str(text).replace(",", " ").split()]
    if len(parts) != 3:
        raise ValueError(f"Expected BGR color with 3 values, got {text!r}")
    return tuple(int(np.clip(x, 0, 255)) for x in parts)


def draw_proxy(frame, bbox, object_type, color, alpha):
    if not np.isfinite(bbox).all():
        return frame
    x1, y1, x2, y2 = np.round(bbox).astype(int)
    h, w = frame.shape[:2]
    x1, x2 = int(np.clip(x1, 0, w - 1)), int(np.clip(x2, 0, w - 1))
    y1, y2 = int(np.clip(y1, 0, h - 1)), int(np.clip(y2, 0, h - 1))
    if x2 <= x1 or y2 <= y1:
        return frame
    overlay = frame.copy()
    if object_type == "sphere":
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
        cv2.ellipse(overlay, center, axes, 0, 0, 360, color, -1, cv2.LINE_AA)
        cv2.ellipse(frame, center, axes, 0, 0, 360, color, 3, cv2.LINE_AA)
    else:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, dst=frame)
    return frame


def overlay(args):
    boxes, valid, object_type = load_proxy(args.object_proxy)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {args.output}")

    color = parse_color(args.color)
    idx = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        proxy_idx = min(idx, boxes.shape[0] - 1)
        if proxy_idx >= 0 and proxy_idx < boxes.shape[0] and valid[proxy_idx]:
            draw_proxy(frame, boxes[proxy_idx], object_type, color, args.alpha)
        writer.write(frame)
        written += 1
        idx += 1
        if args.max_frames > 0 and written >= args.max_frames:
            break

    cap.release()
    writer.release()
    print(f"Saved object overlay video: {args.output}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--object_proxy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--color", default="32,154,238", help="BGR color, default orange")
    parser.add_argument("--alpha", type=float, default=0.26)
    parser.add_argument("--max_frames", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    overlay(parse_args())
