#!/usr/bin/env python3
"""Audit the one static coordinate contract shared by VideoMimic and GMR.

This audit intentionally does *not* fit or modify a chair, robot root, or
reference pose.  The semantic chair is already emitted in MuJoCo Z-up by
VideoMimic and GMR applies the same GVHMR conversion, ``[x, -z, y]``, while
retargeting.  A G1 root is not an SMPL pelvis: their remaining static and
pose-dependent offset is an embodiment residual, not evidence for moving the
scene.

The output makes that distinction executable.  Downstream physical tracking
may consume an accepted report only with the identity scene transform.  A
separate, explicitly labelled physical-motion candidate is then free to adapt
joint controls in response to contact, but may not quietly absorb the error by
moving the chair or writing a per-frame root trajectory.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


GVHMR_TO_MUJOCO_Z_UP = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _as_finite_array(value: Any, *, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _rigid_planar_fit(source_xy: np.ndarray, robot_xy: np.ndarray) -> dict[str, Any]:
    """Fit only for diagnosis; never use this result to transform the scene."""
    if source_xy.shape != robot_xy.shape or source_xy.ndim != 2 or source_xy.shape[1] != 2:
        raise ValueError("planar trajectories must both have shape (T, 2)")
    source_center = source_xy.mean(axis=0)
    robot_center = robot_xy.mean(axis=0)
    source_zero = source_xy - source_center
    robot_zero = robot_xy - robot_center
    left, _, right_t = np.linalg.svd(source_zero.T @ robot_zero)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    translation = robot_center - source_center @ rotation
    residual = np.linalg.norm(source_xy @ rotation + translation - robot_xy, axis=1)
    yaw_deg = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
    return {
        "diagnostic_yaw_deg": yaw_deg,
        "diagnostic_translation_xy_m": translation.astype(float).tolist(),
        "rmse_m": float(np.sqrt(np.mean(np.square(residual)))),
        "p95_m": float(np.quantile(residual, 0.95)),
        "max_m": float(np.max(residual)),
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum_m": float(np.min(values)),
        "p05_m": float(np.quantile(values, 0.05)),
        "median_m": float(np.median(values)),
        "p95_m": float(np.quantile(values, 0.95)),
        "maximum_m": float(np.max(values)),
    }


def _seat_top(primitives: dict[str, Any]) -> float:
    if primitives.get("frame") != "mujoco_world_z_up":
        raise ValueError("semantic chair primitives are not in mujoco_world_z_up")
    for primitive in primitives.get("primitives", []):
        if primitive.get("name") == "seat_support":
            center = _as_finite_array(primitive.get("center"), name="seat center", shape=(3,))
            extents = _as_finite_array(primitive.get("extents"), name="seat extents", shape=(3,))
            if primitive.get("type") != "box" or extents[2] <= 0.0:
                raise ValueError("seat_support must be a positive-height box")
            # VideoMimic serializes full box dimensions as ``extents``.  MuJoCo
            # stores half-sizes internally, so the physical seat top is one
            # half-thickness above the serialized centre.
            return float(center[2] + 0.5 * extents[2])
    raise ValueError("semantic chair primitives have no seat_support")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-motion", type=Path, required=True)
    parser.add_argument("--robot-motion", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--chair-primitives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approach-end-frame", type=int, default=178)
    parser.add_argument("--max-planar-yaw-deg", type=float, default=3.0)
    parser.add_argument("--max-planar-translation-m", type=float, default=0.15)
    parser.add_argument("--max-planar-rmse-m", type=float, default=0.08)
    args = parser.parse_args()

    if args.approach_end_frame < 2:
        raise ValueError("approach-end-frame must leave at least three frames")
    if min(args.max_planar_yaw_deg, args.max_planar_translation_m, args.max_planar_rmse_m) < 0.0:
        raise ValueError("coordinate-contract tolerances must be non-negative")

    with np.load(args.human_motion, allow_pickle=False) as source:
        if "trans" not in source:
            raise ValueError("human motion lacks trans")
        human_translation = _as_finite_array(source["trans"], name="human trans")
    if human_translation.ndim != 2 or human_translation.shape[1] != 3:
        raise ValueError(f"human trans must have shape (T, 3), got {human_translation.shape}")

    with args.robot_motion.open("rb") as stream:
        robot_motion = pickle.load(stream)
    if not isinstance(robot_motion, dict):
        raise ValueError("robot motion must be a pickle dictionary")
    robot_root = _as_finite_array(robot_motion.get("root_pos"), name="robot root_pos")
    if robot_root.shape != human_translation.shape:
        raise ValueError(
            "human and robot motions must have the same (T, 3) shape, got "
            f"{human_translation.shape} and {robot_root.shape}"
        )

    manifest = _load_json(args.scene_manifest)
    primitives = _load_json(args.chair_primitives)
    if manifest.get("simulation_coordinate_system") != "mujoco_world_z_up":
        raise ValueError("scene manifest does not declare mujoco_world_z_up")
    if manifest.get("scene_coordinate_system") != "videomimic_gravity_calibrated":
        raise ValueError("scene manifest does not declare VideoMimic gravity calibration")

    frame_count = int(human_translation.shape[0])
    if args.approach_end_frame >= frame_count:
        raise ValueError("approach-end-frame is outside the source motion")
    approach = slice(0, args.approach_end_frame + 1)
    human_z_up = human_translation @ GVHMR_TO_MUJOCO_Z_UP.T
    planar_fit = _rigid_planar_fit(human_z_up[approach, :2], robot_root[approach, :2])
    vertical_offset = robot_root[:, 2] - human_z_up[:, 2]
    approach_vertical_offset = vertical_offset[approach]
    diagnostic_translation = np.asarray(planar_fit["diagnostic_translation_xy_m"], dtype=np.float64)

    fit_passed = bool(
        abs(float(planar_fit["diagnostic_yaw_deg"])) <= args.max_planar_yaw_deg
        and float(np.linalg.norm(diagnostic_translation)) <= args.max_planar_translation_m
        and float(planar_fit["rmse_m"]) <= args.max_planar_rmse_m
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "immutable_videomimic_gmr_coordinate_contract",
        "status": "accepted_identity_scene_contract" if fit_passed else "rejected_coordinate_contract",
        "frame_count": frame_count,
        "approach_frame_range": [0, args.approach_end_frame],
        "source_assets": {
            "human_motion": str(args.human_motion),
            "robot_motion": str(args.robot_motion),
            "scene_manifest": str(args.scene_manifest),
            "chair_primitives": str(args.chair_primitives),
        },
        "coordinate_contract": {
            "gvhmr_to_mujoco_formula": "[x, -z, y]",
            "gvhmr_to_mujoco_matrix": GVHMR_TO_MUJOCO_Z_UP.astype(float).tolist(),
            "scene_frame": "mujoco_world_z_up",
            "gmr_retarget_frame": "mujoco_world_z_up",
            "scene_to_gmr_transform": {
                "type": "identity",
                "translation_xyz_m": [0.0, 0.0, 0.0],
                "yaw_deg": 0.0,
            },
            "seat_top_z_m": _seat_top(primitives),
        },
        "embodiment_residual_not_a_scene_transform": {
            "planar_fit_on_contact_free_approach": planar_fit,
            "robot_root_minus_smpl_root_z_up_approach": _quantiles(approach_vertical_offset),
            "robot_root_minus_smpl_root_z_up_all_frames": _quantiles(vertical_offset),
            "explanation": (
                "The G1 free-root origin and the SMPL pelvis are different body points. "
                "These values diagnose retargeting morphology only and must not be "
                "applied to the chair or per-frame robot root."
            ),
        },
        "quality_gate": {
            "max_planar_yaw_deg": args.max_planar_yaw_deg,
            "max_planar_translation_m": args.max_planar_translation_m,
            "max_planar_rmse_m": args.max_planar_rmse_m,
            "passed": fit_passed,
        },
        "downstream_policy": {
            "static_chair_transform_allowed": False,
            "per_frame_root_write_allowed": False,
            "external_pelvis_force_allowed": False,
            "physical_candidate_must_use": "bounded_joint_actuator_control_plus_mj_step",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output)}, ensure_ascii=False))
    if not fit_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
