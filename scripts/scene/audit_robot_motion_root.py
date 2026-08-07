#!/usr/bin/env python3
"""Audit GMR root translation/orientation continuity before any controller use."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, nargs="+", required=True)
    parser.add_argument("--max-step-m", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def summary(values: np.ndarray) -> dict[str, float]:
    if not len(values):
        return {"p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "p50": float(np.quantile(values, 0.5)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def quaternion_step_rad(quaternions_xyzw: np.ndarray) -> np.ndarray:
    normalized = quaternions_xyzw / np.maximum(
        np.linalg.norm(quaternions_xyzw, axis=1, keepdims=True), 1e-12
    )
    cosine = np.abs(np.sum(normalized[1:] * normalized[:-1], axis=1))
    return 2.0 * np.arccos(np.clip(cosine, -1.0, 1.0))


def audit(path: Path, max_step_m: float) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        motion = pickle.load(handle)
    root = np.asarray(motion["root_pos"], dtype=np.float64)
    rotation = np.asarray(motion["root_rot"], dtype=np.float64)
    if root.ndim != 2 or root.shape[1] != 3 or rotation.shape != (len(root), 4):
        raise ValueError(f"invalid root arrays in {path}")
    fps = float(np.asarray(motion.get("fps", 30.0)).reshape(-1)[0])
    steps = np.diff(root, axis=0)
    total_step = np.linalg.norm(steps, axis=1)
    vertical_step = np.abs(steps[:, 2])
    acceleration = np.linalg.norm(np.diff(root, n=2, axis=0), axis=1) * fps * fps
    angular_step = quaternion_step_rad(rotation)
    bad_indices = (np.flatnonzero(total_step > max_step_m) + 1).tolist()
    root_height = root[:, 2]
    return {
        "motion": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "frames": int(len(root)),
        "fps": fps,
        "root_step_m": summary(total_step),
        "root_vertical_step_m": summary(vertical_step),
        "root_height_m": {
            "start": float(root_height[0]),
            "end": float(root_height[-1]),
            "min": float(np.min(root_height)),
            "min_frame": int(np.argmin(root_height)),
            "max": float(np.max(root_height)),
            "max_frame": int(np.argmax(root_height)),
        },
        "root_acceleration_mps2": summary(acceleration),
        "root_rotation_step_rad": summary(angular_step),
        "threshold_m": max_step_m,
        "large_step_frames": bad_indices,
        "status": "pass" if not bad_indices else "reject_root_jump",
    }


def main() -> None:
    args = parse_args()
    if args.max_step_m <= 0.0:
        raise ValueError("--max-step-m must be positive")
    report = {"audits": [audit(path, args.max_step_m) for path in args.motion]}
    encoded = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if args.require_pass and any(item["status"] != "pass" for item in report["audits"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
