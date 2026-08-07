#!/usr/bin/env python3
"""Certify an observed-support semantic chair for an isolated PHC replay.

The default scope is the conservative seat-only support proxy.  A full
semantic chair is admitted only when a separate world-space asset-contract
report proves that Isaac's primitives, MuJoCo's primitives, and MuJoCo's XML
geometries are the same named boxes.  Neither scope ever admits raw NKSR
triangles or promotes a PHC/GMR result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def summary(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum_m": float(np.min(values)),
        "p01_m": float(np.quantile(values, 0.01)),
        "p05_m": float(np.quantile(values, 0.05)),
        "median_m": float(np.median(values)),
        "p95_m": float(np.quantile(values, 0.95)),
        "maximum_m": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact-evidence", required=True, type=Path)
    parser.add_argument("--smpl-anchors", required=True, type=Path)
    parser.add_argument("--semantic-chair-report", required=True, type=Path)
    parser.add_argument("--phc-primitives", required=True, type=Path)
    parser.add_argument(
        "--collision-scope",
        choices=("semantic_seat_support_only", "semantic_primitives_only"),
        default="semantic_seat_support_only",
    )
    parser.add_argument(
        "--scene-asset-contract-report",
        type=Path,
        help="required for semantic_primitives_only; verifies the full shared chair",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-seat-penetration-m", type=float, default=0.005)
    parser.add_argument("--max-seat-clearance-m", type=float, default=0.005)
    parser.add_argument("--min-stable-frames", type=int, default=12)
    args = parser.parse_args()

    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite certificate: {args.output}")
    if args.max_seat_penetration_m < 0.0 or args.max_seat_clearance_m <= 0.0:
        raise ValueError("seat penetration/clearance thresholds must be non-negative/positive")
    if args.min_stable_frames < 4:
        raise ValueError("min-stable-frames must be at least four")

    evidence = load_json(args.contact_evidence)
    frames = np.asarray(
        [row["frame"] for row in evidence.get("frames", []) if row.get("seated_candidate")],
        dtype=np.int64,
    )
    if len(frames) < args.min_stable_frames:
        raise RuntimeError(
            f"Only {len(frames)} stable seated frames; need {args.min_stable_frames}"
        )

    with np.load(args.smpl_anchors, allow_pickle=False) as anchors:
        if "seat_signed_distance_m" not in anchors:
            raise KeyError("SMPL anchors are missing seat_signed_distance_m")
        distance = np.asarray(anchors["seat_signed_distance_m"], dtype=np.float64)
    if frames.min() < 0 or frames.max() >= len(distance):
        raise ValueError("stable seated frames do not match SMPL anchor length")
    stable_distance = distance[frames]
    if not np.all(np.isfinite(stable_distance)):
        raise RuntimeError("stable seated frames have non-finite SMPL seat distances")

    chair_report = load_json(args.semantic_chair_report)
    layout = chair_report.get("layout_inference")
    if not isinstance(layout, dict) or layout.get("method") != "seated_smpl_body_heading":
        raise RuntimeError("semantic chair is not backed by automatic seated-body heading")
    if int(chair_report.get("back_sign", 0)) not in {-1, 1}:
        raise RuntimeError("semantic chair has no valid backrest orientation")

    primitive_payload = load_json(args.phc_primitives)
    if primitive_payload.get("frame") != "mujoco_world_z_up":
        raise RuntimeError("PHC primitives are not in mujoco_world_z_up")
    primitives = primitive_payload.get("primitives")
    if not isinstance(primitives, list) or not primitives:
        raise RuntimeError("PHC primitives must be a non-empty list")
    names: set[str] = set()
    for primitive in primitives:
        if (
            not isinstance(primitive, dict)
            or primitive.get("type") != "box"
            or not isinstance(primitive.get("name"), str)
            or np.asarray(primitive.get("center"), dtype=np.float64).shape != (3,)
            or np.asarray(primitive.get("extents"), dtype=np.float64).shape != (3,)
            or np.asarray(primitive.get("rotation_matrix"), dtype=np.float64).shape != (3, 3)
        ):
            raise RuntimeError("PHC primitive is not an oriented semantic box")
        if primitive["name"] in names:
            raise RuntimeError("PHC primitives contain duplicate names")
        names.add(primitive["name"])
        rotation = np.asarray(primitive["rotation_matrix"], dtype=np.float64)
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4):
            raise RuntimeError("PHC primitive rotation is not orthonormal")
        if np.any(np.asarray(primitive["extents"], dtype=np.float64) <= 0.0):
            raise RuntimeError("PHC primitive extents must be positive")

    full_scene_contract: dict[str, object] | None = None
    primitive_hash = sha256(args.phc_primitives)
    if args.collision_scope == "semantic_seat_support_only":
        if names != {"seat_support"}:
            raise RuntimeError("seat-only certificate permits exactly seat_support")
    else:
        required_names = {
            "seat_support",
            "leg_front_left",
            "leg_front_right",
            "leg_back_left",
            "leg_back_right",
            "backrest",
        }
        if names != required_names:
            raise RuntimeError("full semantic chair must contain the six canonical chair boxes")
        if args.scene_asset_contract_report is None:
            raise RuntimeError("full semantic chair requires --scene-asset-contract-report")
        full_scene_contract = load_json(args.scene_asset_contract_report)
        if (
            full_scene_contract.get("schema_version") != 1
            or full_scene_contract.get("status") != "accepted_world_space_equivalent"
            or full_scene_contract.get("collision_representation") != "identical_named_semantic_boxes"
            or full_scene_contract.get("mujoco_primitives_sha256") != primitive_hash
            or full_scene_contract.get("isaac_primitives_sha256") != primitive_hash
            or float(full_scene_contract.get("max_world_vertex_error_m", float("inf"))) > 1e-6
        ):
            raise RuntimeError("full-chair asset contract does not bind Isaac and MuJoCo to this primitive file")

    contact_band = (
        (stable_distance >= -args.max_seat_penetration_m)
        & (stable_distance <= args.max_seat_clearance_m)
    )
    if not bool(np.all(contact_band)):
        raise RuntimeError(
            "stable SMPL seated contact does not fit the configured seat tolerance"
        )

    result = {
        "schema_version": 1,
        "status": "accepted_for_isolated_phc_candidate",
        "collision_scope": args.collision_scope,
        "raw_nksr_mesh_permitted": False,
        "semantic_layout": {
            "method": layout["method"],
            "heading_concentration": layout.get("heading_concentration"),
            "inferred_source_depth_axis": layout.get("inferred_source_depth_axis"),
            "inferred_back_sign": layout.get("inferred_back_sign"),
        },
        "stable_smpl_seat_evidence": {
            "frame_count": int(len(frames)),
            "frame_range": [int(frames.min()), int(frames.max())],
            "distance_m": summary(stable_distance),
            "contact_band_coverage": float(np.mean(contact_band)),
            "thresholds_m": {
                "max_penetration": float(args.max_seat_penetration_m),
                "max_clearance": float(args.max_seat_clearance_m),
            },
        },
        "inputs": {
            "contact_evidence": {"path": str(args.contact_evidence.resolve()), "sha256": sha256(args.contact_evidence)},
            "smpl_anchors": {"path": str(args.smpl_anchors.resolve()), "sha256": sha256(args.smpl_anchors)},
            "semantic_chair_report": {"path": str(args.semantic_chair_report.resolve()), "sha256": sha256(args.semantic_chair_report)},
            "phc_primitives": {"path": str(args.phc_primitives.resolve()), "sha256": primitive_hash},
        },
    }
    if full_scene_contract is not None:
        result["inputs"]["scene_asset_contract_report"] = {
            "path": str(args.scene_asset_contract_report.resolve()),
            "sha256": sha256(args.scene_asset_contract_report),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
