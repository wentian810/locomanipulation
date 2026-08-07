#!/usr/bin/env python3
"""Build a support-grounded Stage-2 terrain from VideoMimic semantic assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import trimesh


class ContractError(RuntimeError):
    """Raised when the semantic chair cannot define a stable terrain contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(path: Path, label: str) -> Path:
    value = path.expanduser().resolve()
    if not value.is_file():
        raise ContractError(f"missing {label}: {value}")
    return value


def _floor_z(primitives_path: Path) -> tuple[float, list[float]]:
    payload = json.loads(primitives_path.read_text(encoding="utf-8"))
    values: list[float] = []
    for item in payload.get("primitives", []):
        if not str(item.get("name", "")).startswith("leg_"):
            continue
        center = np.asarray(item.get("center"), dtype=np.float64)
        extents = np.asarray(item.get("extents"), dtype=np.float64)
        if center.shape != (3,) or extents.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(extents).all():
            raise ContractError(f"invalid leg primitive in {primitives_path}")
        values.append(float(center[2] - 0.5 * extents[2]))
    if len(values) != 4:
        raise ContractError(f"expected four semantic chair legs, found {len(values)}")
    if max(values) - min(values) > 0.003:
        raise ContractError(f"semantic chair leg bottoms disagree by more than 3 mm: {values}")
    return float(np.median(values)), values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-h5", required=True, type=Path)
    parser.add_argument("--semantic-chair-obj", required=True, type=Path)
    parser.add_argument("--semantic-primitives", required=True, type=Path)
    parser.add_argument("--output-obj", required=True, type=Path)
    parser.add_argument("--margin-m", type=float, default=1.0)
    args = parser.parse_args()
    if args.margin_m <= 0.0 or not np.isfinite(args.margin_m):
        parser.error("--margin-m must be positive and finite")
    reference = _require(args.reference_h5, "Stage-4 reference H5")
    chair_path = _require(args.semantic_chair_obj, "semantic chair OBJ")
    primitives = _require(args.semantic_primitives, "semantic chair primitives")
    output = args.output_obj.expanduser().resolve()
    if output.exists():
        raise ContractError(f"refusing to overwrite terrain candidate: {output}")
    floor_z, leg_bottoms = _floor_z(primitives)
    chair = trimesh.load(chair_path, force="mesh", process=False)
    if not isinstance(chair, trimesh.Trimesh) or len(chair.faces) == 0:
        raise ContractError(f"semantic chair is not a non-empty mesh: {chair_path}")
    with h5py.File(reference, "r") as archive:
        if "link_pos" not in archive:
            raise ContractError(f"Stage-4 H5 lacks link_pos: {reference}")
        link_pos = np.asarray(archive["link_pos"], dtype=np.float64)
    if link_pos.ndim != 3 or link_pos.shape[2] != 3 or not np.isfinite(link_pos).all():
        raise ContractError(f"invalid Stage-4 link positions: {reference}")
    xy = np.concatenate((link_pos[:, :, :2].reshape(-1, 2), chair.vertices[:, :2]), axis=0)
    lower = xy.min(axis=0) - args.margin_m
    upper = xy.max(axis=0) + args.margin_m
    vertices = np.array(
        [
            [lower[0], lower[1], floor_z],
            [upper[0], lower[1], floor_z],
            [upper[0], upper[1], floor_z],
            [lower[0], upper[1], floor_z],
        ],
        dtype=np.float64,
    )
    floor = trimesh.Trimesh(vertices=vertices, faces=np.array([[0, 1, 2], [0, 2, 3]]), process=False)
    terrain = trimesh.util.concatenate((floor, chair))
    output.parent.mkdir(parents=True, exist_ok=True)
    terrain.export(output)
    report = {
        "schema_version": 1,
        "status": "pass",
        "coordinate_frame": "mujoco_world_z_up",
        "method": "semantic_chair_leg_bottom_observed_ground_plane_plus_semantic_chair",
        "reference_h5": str(reference),
        "reference_h5_sha256": _sha256(reference),
        "semantic_chair_obj": str(chair_path),
        "semantic_chair_obj_sha256": _sha256(chair_path),
        "semantic_primitives": str(primitives),
        "semantic_primitives_sha256": _sha256(primitives),
        "floor_z_m": floor_z,
        "chair_leg_bottoms_m": leg_bottoms,
        "ground_xy_bounds_m": [[float(lower[0]), float(lower[1])], [float(upper[0]), float(upper[1])]],
        "margin_m": args.margin_m,
        "floor_face_count": int(len(floor.faces)),
        "semantic_chair_face_count": int(len(chair.faces)),
        "terrain_face_count": int(len(terrain.faces)),
        "output_obj": str(output),
        "output_sha256": _sha256(output),
    }
    output.with_suffix(".terrain.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
