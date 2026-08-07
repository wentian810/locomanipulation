#!/usr/bin/env python3
"""Measure body-position disagreement between two SONIC reference directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BODY_NAMES = (
    "pelvis", "left_hip_roll", "left_knee", "left_ankle", "right_hip_roll",
    "right_knee", "right_ankle", "torso", "left_shoulder_roll", "left_elbow",
    "left_wrist_yaw", "right_shoulder_roll", "right_elbow", "right_wrist_yaw",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    return parser.parse_args()


def read_body_positions(directory: Path) -> np.ndarray:
    path = directory / "body_pos.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[1] != 3 * len(BODY_NAMES):
        raise ValueError(f"unexpected body_pos shape {values.shape}")
    return values.reshape(len(values), len(BODY_NAMES), 3)


def main() -> None:
    args = parse_args()
    first = read_body_positions(args.first)
    second = read_body_positions(args.second)
    if first.shape != second.shape:
        raise ValueError(f"reference shape mismatch: {first.shape} vs {second.shape}")
    distances = np.linalg.norm(first - second, axis=2)
    report = {
        "frames": int(first.shape[0]),
        "global_max_m": float(np.max(distances)),
        "global_rms_m": float(np.sqrt(np.mean(distances ** 2))),
        "per_body_max_m": {
            name: float(np.max(distances[:, index])) for index, name in enumerate(BODY_NAMES)
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
