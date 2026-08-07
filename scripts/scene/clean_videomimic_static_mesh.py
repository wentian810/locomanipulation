#!/usr/bin/env python3
"""Conservatively clean VideoMimic's static NKSR visual mesh.

This is deliberately a geometry-only operation.  It never names an object,
fits a chair, adds legs/backrests, or smooths coordinates.  The SAM2-masked
first-round NKSR mesh remains the collision source.  The filled NKSR mesh is
used only as a visual completion after removing invalid faces, tiny detached
components, and fill faces too far from the observed first-round surface.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def _load(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"invalid mesh: {path}")
    mesh.remove_unreferenced_vertices()
    if hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()
    elif hasattr(mesh, "unique_faces"):
        mesh.update_faces(mesh.unique_faces())
    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    return mesh


def _retain_components(mesh: trimesh.Trimesh, minimum_faces: int, minimum_area: float) -> tuple[trimesh.Trimesh, dict]:
    components = mesh.split(only_watertight=False)
    if not components:
        raise ValueError("mesh has no connected components")
    ranked = sorted(components, key=lambda item: len(item.faces), reverse=True)
    kept = [item for index, item in enumerate(ranked) if index == 0 or len(item.faces) >= minimum_faces or float(item.area) >= minimum_area]
    if not kept:
        kept = [ranked[0]]
    result = trimesh.util.concatenate(kept)
    result.remove_unreferenced_vertices()
    return result, {
        "input_components": int(len(components)),
        "kept_components": int(len(kept)),
        "dropped_components": int(len(components) - len(kept)),
        "largest_component_faces": int(len(ranked[0].faces)),
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {name: float(np.quantile(values, q)) for name, q in (("p50_m", .50), ("p90_m", .90), ("p95_m", .95), ("p99_m", .99), ("max_m", 1.0))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-round-mesh", required=True, type=Path)
    parser.add_argument("--filled-mesh", required=True, type=Path)
    parser.add_argument("--output-mesh", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--min-component-faces", type=int, default=48)
    parser.add_argument("--min-component-area-m2", type=float, default=0.002)
    parser.add_argument("--completion-max-distance-m", type=float, default=0.35)
    parser.add_argument("--minimum-retained-face-ratio", type=float, default=0.90)
    args = parser.parse_args()
    if args.min_component_faces < 1 or args.min_component_area_m2 < 0:
        raise ValueError("component thresholds must be non-negative")
    if not 0.05 <= args.completion_max_distance_m <= 1.0:
        raise ValueError("completion-max-distance-m must be in [0.05, 1.0]")
    if not 0.5 <= args.minimum_retained_face_ratio <= 1.0:
        raise ValueError("minimum-retained-face-ratio must be in [0.5, 1.0]")

    observed = _load(args.first_round_mesh)
    filled = _load(args.filled_mesh)
    component_clean, component_report = _retain_components(filled, args.min_component_faces, args.min_component_area_m2)
    centers = component_clean.triangles_center
    distances = cKDTree(observed.vertices).query(centers, workers=-1)[0]
    keep_faces = distances <= args.completion_max_distance_m
    retained_ratio = float(keep_faces.mean())
    fallback_to_observed = retained_ratio < args.minimum_retained_face_ratio
    if fallback_to_observed:
        cleaned = observed.copy()
        reason = "filled_completion_rejected_insufficient_observed_proximity"
    else:
        cleaned = component_clean.copy()
        cleaned.update_faces(keep_faces)
        cleaned.remove_unreferenced_vertices()
        reason = "filled_completion_retained_with_observed_proximity_filter"
    args.output_mesh.parent.mkdir(parents=True, exist_ok=True)
    cleaned.export(args.output_mesh)
    report = {
        "schema_version": 1,
        "method": "sam2_masked_nksr_fill_geometry_cleaning",
        "semantic_template_used": False,
        "coordinate_smoothing_used": False,
        "collision_policy": "first_round_nksr_is_separate_collision_source",
        "inputs": {"first_round_mesh": str(args.first_round_mesh), "filled_mesh": str(args.filled_mesh)},
        "thresholds": {"min_component_faces": args.min_component_faces, "min_component_area_m2": args.min_component_area_m2, "completion_max_distance_m": args.completion_max_distance_m, "minimum_retained_face_ratio": args.minimum_retained_face_ratio},
        "filled_mesh": {"vertices": int(len(filled.vertices)), "faces": int(len(filled.faces))},
        "component_cleaning": component_report,
        "observed_proximity": {"face_center_distance": _quantiles(distances), "retained_face_ratio": retained_ratio},
        "output": {"mesh": str(args.output_mesh), "vertices": int(len(cleaned.vertices)), "faces": int(len(cleaned.faces)), "fallback_to_first_round": fallback_to_observed, "reason": reason},
        "verdict": "warn" if fallback_to_observed else "pass",
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": report["verdict"], "faces": report["output"]["faces"], "retained_face_ratio": retained_ratio, "reason": reason}, ensure_ascii=False))


if __name__ == "__main__":
    main()
