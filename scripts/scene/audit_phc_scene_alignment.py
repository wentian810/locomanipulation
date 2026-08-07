#!/usr/bin/env python3
"""Audit whether a PHC replay remains aligned with a packaged support.

The source SMPL motion from the VideoMimic bridge uses ``neg_y`` gravity.
PHC converts it to its Z-up simulation world using +90 degrees about X.  The
scene primitives are already stored in that Z-up world.  This utility compares
the requested source root trajectory and the *actual recorded Isaac Gym root
trajectory* in the support's local frame, rather than judging alignment from a
render alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.spatial.transform import Rotation


def _array(value: Any) -> np.ndarray:
    """Make a numeric array from numpy / torch / serialized values."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _z_up_from_neg_y(points: np.ndarray) -> np.ndarray:
    """Apply the exact world conversion used by convert_zitai_to_phc.py."""
    return Rotation.from_euler("x", 90.0, degrees=True).apply(points)


def _primitive_by_name(path: Path, requested_name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("frame") != "mujoco_world_z_up":
        raise ValueError(
            f"Expected mujoco_world_z_up primitives, got {payload.get('frame')!r}"
        )
    for primitive in payload.get("primitives", []):
        if primitive.get("name") == requested_name:
            return primitive
    names = [str(item.get("name")) for item in payload.get("primitives", [])]
    raise KeyError(f"Support {requested_name!r} not found; available: {names}")


def _local_points(points: np.ndarray, center: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    # The primitives use world_point = center + rotation @ local_point.
    return (points - center) @ rotation


def _find_sit_mask(local_target: np.ndarray, extents: np.ndarray) -> np.ndarray:
    """Conservative root-based sitting interval; root is normally above seat."""
    return (
        (np.abs(local_target[:, 0]) <= extents[0] / 2.0 + 0.15)
        & (np.abs(local_target[:, 1]) <= extents[1] / 2.0 + 0.15)
        & (local_target[:, 2] >= 0.25)
        & (local_target[:, 2] <= 0.65)
    )


def _summary(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "median_m": np.median(values, axis=0).round(6).tolist(),
        "mean_m": np.mean(values, axis=0).round(6).tolist(),
        "p05_m": np.percentile(values, 5, axis=0).round(6).tolist(),
        "p95_m": np.percentile(values, 95, axis=0).round(6).tolist(),
        "rmse_m": np.sqrt(np.mean(values**2, axis=0)).round(6).tolist(),
    }


def _read_root_offset(serialized_tree: Any) -> np.ndarray:
    """Read SkeletonTree.to_dict() without importing PHC / Isaac Gym."""
    if not isinstance(serialized_tree, dict):
        raise TypeError(f"Unexpected serialized skeleton tree: {type(serialized_tree)!r}")
    for key in ("local_translation", "local_translations"):
        if key in serialized_tree:
            # ``SkeletonTree.to_dict`` serializes tensors as
            # {"arr": ndarray, "context": {"dtype": ...}}.
            value = serialized_tree[key]
            if isinstance(value, dict) and "arr" in value:
                value = value["arr"]
            local_translation = _array(value)
            return local_translation[0]
    raise KeyError(f"Missing local translation in skeleton tree: {list(serialized_tree)}")


def _state_report(
    state_path: Path,
    target_z_up: np.ndarray,
    center: np.ndarray,
    rotation: np.ndarray,
    extents: np.ndarray,
    sit_mask: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, str]:
    states = joblib.load(state_path)
    if not isinstance(states, dict) or not states:
        raise ValueError(f"No state segments found in {state_path}")

    selected_key, selected = max(
        states.items(), key=lambda item: len(_array(item[1]["trans"]))
    )
    segments: dict[str, Any] = {}
    selected_actual = None
    for key, sequence in states.items():
        sim_root = _array(sequence["trans"])
        root_offset = _read_root_offset(sequence["skeleton_tree"])
        # PHC writes the MuJoCo skeleton root.  Its exported SMPL translation
        # subtracts this offset, so use the same point for source comparison.
        actual_root = sim_root - root_offset[None, :]
        count = min(len(actual_root), len(target_z_up))
        residual = actual_root[:count] - target_z_up[:count]
        local_actual = _local_points(actual_root[:count], center, rotation)
        common_sit = sit_mask[:count]
        segment: dict[str, Any] = {
            "frames": int(len(actual_root)),
            "root_offset_m": root_offset.round(6).tolist(),
            "all_frames_residual": _summary(residual),
            "final_root_z_up_m": actual_root[-1].round(6).tolist(),
            "final_root_support_local_m": local_actual[-1].round(6).tolist(),
        }
        if np.any(common_sit):
            sit_residual = residual[common_sit]
            segment["sit_phase"] = {
                "frames": np.flatnonzero(common_sit).astype(int).tolist(),
                "residual": _summary(sit_residual),
                "horizontal_median_error_m": float(
                    np.median(np.linalg.norm(sit_residual[:, :2], axis=1))
                ),
                "vertical_median_error_m": float(np.median(sit_residual[:, 2])),
                "actual_root_support_local_median": np.median(
                    local_actual[common_sit], axis=0
                ).round(6).tolist(),
            }
        segments[str(key)] = segment
        if key == selected_key:
            selected_actual = actual_root

    assert selected_actual is not None
    return {"selected_segment": str(selected_key), "segments": segments}, selected_actual, str(selected_key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-root", type=Path, required=True)
    parser.add_argument("--scene-subdir", default="scene_reconstruction_semantic_chair_v2")
    parser.add_argument(
        "--primitives-path",
        type=Path,
        default=None,
        help="Optional packaged MuJoCo primitive JSON; defaults to the semantic-chair asset.",
    )
    parser.add_argument("--source-npz", type=Path, default=None)
    parser.add_argument("--state-pkl", type=Path, default=None)
    parser.add_argument("--support-name", default="seat_support")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    clip_root = args.clip_root.resolve()
    source_path = args.source_npz or clip_root / "phc_in" / "001" / "001_optimized.npz"
    state_path = args.state_pkl or clip_root / "phc_states" / "001" / "001_optimized.pkl"
    scene_root = clip_root / args.scene_subdir
    primitives_path = args.primitives_path or (
        scene_root / "scene" / "semantic_chair_primitives_mujoco.json"
    )

    source = np.load(source_path, allow_pickle=False)
    source_z_up = _z_up_from_neg_y(_array(source["trans"]))
    primitive = _primitive_by_name(primitives_path, args.support_name)
    center = _array(primitive["center"])
    rotation = _array(primitive["rotation_matrix"])
    extents = _array(primitive["extents"])
    local_target = _local_points(source_z_up, center, rotation)
    sit_mask = _find_sit_mask(local_target, extents)

    state_info, selected_actual, selected_key = _state_report(
        state_path, source_z_up, center, rotation, extents, sit_mask
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "coordinate_contract": "neg_y_source_to_phc_z_up_x_plus_90",
        "source_npz": str(source_path),
        "state_pkl": str(state_path),
        "support": {
            "name": args.support_name,
            "center_z_up_m": center.round(6).tolist(),
            "extents_m": extents.round(6).tolist(),
        },
        "target_sit_phase_frames": np.flatnonzero(sit_mask).astype(int).tolist(),
        "target_root_support_local_median": np.median(local_target[sit_mask], axis=0)
        .round(6)
        .tolist(),
        "actual_state": state_info,
    }

    selected = report["actual_state"]["segments"][selected_key]
    sit = selected.get("sit_phase")
    warnings: list[str] = []
    if not sit:
        warnings.append("The selected PHC state segment does not cover the source sitting interval.")
    else:
        vertical = float(sit["vertical_median_error_m"])
        horizontal = float(sit["horizontal_median_error_m"])
        if vertical < -0.10:
            warnings.append(
                "PHC's recorded root is more than 10 cm below the source trajectory during sitting; "
                "do not use this replay as a contact-aligned result."
            )
        if horizontal > 0.10:
            warnings.append(
                "PHC's recorded root has more than 10 cm median horizontal error during sitting."
            )
    report["warnings"] = warnings

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
