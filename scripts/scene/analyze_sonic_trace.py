#!/usr/bin/env python3
"""Summarise an audited SONIC MuJoCo trace without changing simulation state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.trace.is_file():
        raise FileNotFoundError(args.trace)
    with np.load(args.trace) as trace:
        time = np.asarray(trace["time"], dtype=np.float64)
        qpos = np.asarray(trace["qpos"], dtype=np.float64)
        qvel = np.asarray(trace["qvel"], dtype=np.float64)
        ctrl = np.asarray(trace["ctrl"], dtype=np.float64)
        ncon = np.asarray(trace["ncon"], dtype=np.int64)

    if qpos.ndim != 2 or qvel.ndim != 2 or ctrl.ndim != 2 or len(time) != len(qpos):
        raise ValueError("trace arrays have inconsistent dimensions")
    changed_ctrl = np.linalg.norm(np.diff(ctrl, axis=0), axis=1) if len(ctrl) > 1 else np.empty(0)
    report = {
        "trace": str(args.trace.resolve()),
        "steps": int(len(time)),
        "simulated_duration_s": float(time[-1] - time[0]) if len(time) else 0.0,
        "root_height_m": {
            "start": float(qpos[0, 2]) if len(qpos) else None,
            "end": float(qpos[-1, 2]) if len(qpos) else None,
            "min": float(np.min(qpos[:, 2])) if len(qpos) else None,
            "max": float(np.max(qpos[:, 2])) if len(qpos) else None,
        },
        "base_velocity_magnitude": {
            "max": float(np.max(np.linalg.norm(qvel[:, :6], axis=1))) if len(qvel) else 0.0,
            "end": float(np.linalg.norm(qvel[-1, :6])) if len(qvel) else 0.0,
        },
        "controls": {
            "actuators": int(ctrl.shape[1]),
            "max_abs": float(np.max(np.abs(ctrl))) if len(ctrl) else 0.0,
            "mean_abs": float(np.mean(np.abs(ctrl))) if len(ctrl) else 0.0,
            "nonzero_frame_fraction": float(np.mean(np.any(np.abs(ctrl) > 1e-7, axis=1))) if len(ctrl) else 0.0,
            "max_interstep_change_l2": float(np.max(changed_ctrl)) if len(changed_ctrl) else 0.0,
            "first_six": ctrl[0, :6].tolist() if len(ctrl) else [],
            "last_six": ctrl[-1, :6].tolist() if len(ctrl) else [],
        },
        "contacts": {
            "min": int(np.min(ncon)) if len(ncon) else 0,
            "max": int(np.max(ncon)) if len(ncon) else 0,
            "mean": float(np.mean(ncon)) if len(ncon) else 0.0,
        },
    }
    encoded = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
