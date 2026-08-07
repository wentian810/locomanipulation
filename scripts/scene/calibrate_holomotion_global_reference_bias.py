#!/usr/bin/env python3
"""Build one bounded outer-loop reference correction from an audited rollout.

HoloMotion's ONNX policy consumes global body references but advances physical
state only through MuJoCo.  When a first, fully physical rollout exposes a
stable global tracking lag, this tool applies the smooth negative of that lag
to the *controller observation reference* for one second pass.  It shifts all
robot bodies equally in selected world axes, so local joint/body geometry is
unchanged.  The original reference remains the audit reference; this file is
explicitly a controller setpoint, never a reported physical trajectory.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np


def _time_derivative(values: np.ndarray, timestep: float) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values, dtype=np.float64)
    return np.gradient(values, timestep, axis=0, edge_order=2 if len(values) >= 3 else 1)


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        raise ValueError("smoothing-window must be odd")
    padding = window // 2
    padded = np.pad(values, ((padding, padding), (0, 0)), mode="edge")
    kernel = np.full(window, 1.0 / window, dtype=np.float64)
    return np.stack([np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(3)], axis=1)


def _clip_norm(vectors: np.ndarray, maximum: float) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    scale = np.minimum(1.0, maximum / np.maximum(norms, 1e-12))
    return vectors * scale[:, None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-reference", required=True, type=Path)
    parser.add_argument("--rollout-npz", required=True, type=Path)
    parser.add_argument("--output-controller-reference", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--start-frame", required=True, type=int)
    parser.add_argument("--max-horizontal-bias-m", type=float, default=0.12)
    parser.add_argument(
        "--pre-seat-end-frame", type=int, default=None,
        help="exclusive frame bound for an additional conservative pre-seat bias cap",
    )
    parser.add_argument("--max-pre-seat-horizontal-bias-m", type=float, default=None)
    parser.add_argument("--smoothing-window", type=int, default=9)
    args = parser.parse_args()
    if args.output_controller_reference.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite controller-reference calibration")
    if args.max_horizontal_bias_m <= 0.0:
        raise ValueError("max horizontal bias must be positive")
    if args.max_pre_seat_horizontal_bias_m is not None and args.max_pre_seat_horizontal_bias_m <= 0.0:
        raise ValueError("max pre-seat horizontal bias must be positive")

    with np.load(args.physical_reference, allow_pickle=False) as reference, np.load(args.rollout_npz, allow_pickle=False) as rollout:
        required = {"metadata", "ref_global_translation", "ref_global_velocity"}
        missing = required.difference(reference.files)
        if missing:
            raise ValueError(f"physical reference lacks {sorted(missing)}")
        ref_translation = np.asarray(reference["ref_global_translation"], dtype=np.float64)
        actual_translation = np.asarray(rollout["robot_global_translation"], dtype=np.float64)
        if actual_translation.shape != ref_translation.shape:
            raise ValueError("rollout and physical reference body tensors differ")
        metadata_raw = reference["metadata"]
        metadata = json.loads(str(metadata_raw.item() if isinstance(metadata_raw, np.ndarray) else metadata_raw))
        payload = {key: reference[key] for key in reference.files}
    if not 0 <= args.start_frame < len(ref_translation):
        raise ValueError("start-frame is outside reference")
    if args.pre_seat_end_frame is not None and not args.start_frame <= args.pre_seat_end_frame <= len(ref_translation):
        raise ValueError("pre-seat-end-frame must lie in [start-frame, frame_count]")
    if (args.pre_seat_end_frame is None) != (args.max_pre_seat_horizontal_bias_m is None):
        raise ValueError("pre-seat-end-frame and max-pre-seat-horizontal-bias-m must be supplied together")
    if args.smoothing_window <= 0 or args.smoothing_window % 2 == 0:
        raise ValueError("smoothing-window must be a positive odd integer")
    fps = float(metadata["motion_fps"])
    root_error = actual_translation[:, 0] - ref_translation[:, 0]
    raw_bias = np.zeros_like(root_error)
    raw_bias[args.start_frame:, :2] = -root_error[args.start_frame:, :2]
    bias = _smooth(raw_bias, args.smoothing_window)
    bias[:args.start_frame] = 0.0
    bias[:, 2] = 0.0
    bias = _clip_norm(bias, args.max_horizontal_bias_m)
    if args.pre_seat_end_frame is not None:
        bias[:args.pre_seat_end_frame] = _clip_norm(
            bias[:args.pre_seat_end_frame], args.max_pre_seat_horizontal_bias_m
        )

    controller_translation = ref_translation + bias[:, None, :]
    output_metadata = copy.deepcopy(metadata)
    output_metadata["controller_reference_calibration"] = {
        "mode": "one_step_bounded_global_xy_tracking_bias_v1",
        "physical_audit_reference": str(args.physical_reference.resolve()),
        "calibration_rollout": str(args.rollout_npz.resolve()),
        "start_frame": args.start_frame,
        "axes": "xy_only",
        "max_horizontal_bias_m": args.max_horizontal_bias_m,
        "pre_seat_end_frame": args.pre_seat_end_frame,
        "max_pre_seat_horizontal_bias_m": args.max_pre_seat_horizontal_bias_m,
        "smoothing_window_frames": args.smoothing_window,
        "audit_contract": "audit actual state against physical_audit_reference, not this controller setpoint",
    }
    payload["metadata"] = np.asarray(json.dumps(output_metadata, ensure_ascii=False))
    payload["ref_global_translation"] = controller_translation.astype(np.float32)
    payload["ref_global_velocity"] = _time_derivative(controller_translation, 1.0 / fps).astype(np.float32)
    args.output_controller_reference.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_controller_reference, **payload)
    report = {
        "schema_version": 1,
        "status": "ready_for_one_bounded_controller_calibration_rollout",
        "physical_reference": str(args.physical_reference.resolve()),
        "rollout": str(args.rollout_npz.resolve()),
        "output_controller_reference": str(args.output_controller_reference.resolve()),
        "start_frame": args.start_frame,
        "root_error_before_m": {
            "p95_norm": float(np.quantile(np.linalg.norm(root_error, axis=1), 0.95)),
            "post_start_median_xy": np.median(root_error[args.start_frame:, :2], axis=0).tolist(),
        },
        "controller_xy_bias_m": {
            "max_norm": float(np.max(np.linalg.norm(bias[:, :2], axis=1))),
            "post_start_median": np.median(bias[args.start_frame:, :2], axis=0).tolist(),
        },
        "contract": {
            "runtime": "one initial qpos/qvel write and continuous mj_step only",
            "does_not_change": ["physical_audit_reference", "robot_joint_reference", "reference_rotations", "chair", "qpos_runtime_overwrite", "external_forces"],
        },
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
