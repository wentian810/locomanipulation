#!/usr/bin/env python3
"""Build MuJoCo collision meshes from a metric visual mesh.

The visual mesh stays untouched. Collision geometry is exported as one or more
convex OBJ files in the same object-local coordinate system. CoACD is used when
installed; otherwise a single convex hull is a safe (but coarse) fallback.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_mesh(path: Path):
    import trimesh

    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"mesh has no geometry: {path}")
        mesh = (
            loaded.to_geometry()
            if hasattr(loaded, "to_geometry")
            else loaded.dump(concatenate=True)
        )
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) < 4:
        raise ValueError(f"mesh is not a usable triangle mesh: {path}")
    mesh = mesh.copy()
    mesh.remove_infinite_values()
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def _coacd_parts(mesh, args):
    try:
        import coacd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "CoACD is not installed. Install `coacd`, or use "
            "`--method convex-hull` for the coarse fallback."
        ) from exc

    source = coacd.Mesh(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int32),
    )
    parts = coacd.run_coacd(
        source,
        threshold=args.threshold,
        max_convex_hull=args.max_convex_hulls,
        max_ch_vertex=args.max_hull_vertices,
        preprocess_resolution=args.preprocess_resolution,
        decimate=True,
    )
    return [
        (
            np.asarray(vertices, dtype=np.float64),
            np.asarray(faces, dtype=np.int64),
        )
        for vertices, faces in parts
    ]


def _convex_hull_parts(mesh):
    hull = mesh.convex_hull
    return [
        (
            np.asarray(hull.vertices, dtype=np.float64),
            np.asarray(hull.faces, dtype=np.int64),
        )
    ]


def build(args) -> None:
    import trimesh

    visual_mesh = args.mesh.resolve()
    mesh = _load_mesh(visual_mesh)
    extents = np.asarray(mesh.extents, dtype=np.float64)
    if np.any(extents <= 0):
        raise ValueError(f"mesh has invalid extents: {extents.tolist()}")
    if extents.max() > args.max_extent_m or extents.min() < args.min_extent_m:
        raise ValueError(
            "mesh dimensions are implausible for metre units: "
            f"{extents.tolist()} m. Fix GLB scaling before collision generation, "
            "or adjust --min_extent_m/--max_extent_m deliberately."
        )

    method = args.method
    if method == "auto":
        try:
            import coacd  # noqa: F401
        except ModuleNotFoundError:
            method = "convex-hull"
        else:
            method = "coacd"

    if method == "coacd":
        parts = _coacd_parts(mesh, args)
    else:
        parts = _convex_hull_parts(mesh)
    if not parts:
        raise RuntimeError("convex decomposition produced no parts")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.output_dir.glob("collision_*.obj"):
        stale.unlink()

    collision_paths: list[str] = []
    collision_volume = 0.0
    for index, (vertices, faces) in enumerate(parts):
        part = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        if len(part.vertices) < 4 or len(part.faces) < 4:
            continue
        if not part.is_convex:
            part = part.convex_hull
        path = args.output_dir / f"collision_{index:03d}.obj"
        part.export(path, file_type="obj")
        collision_paths.append(str(path.resolve()))
        collision_volume += abs(float(part.volume))

    if not collision_paths:
        raise RuntimeError("all convex parts were empty")

    visual_volume = abs(float(mesh.volume)) if mesh.is_volume else 0.0
    reference_volume = visual_volume if visual_volume > 1e-10 else collision_volume
    density = float(args.density)
    if args.mass is not None:
        density = float(args.mass) / max(reference_volume, 1e-10)
    estimated_mass = density * reference_volume

    manifest = {
        "schema_version": 1,
        "units": "meter",
        "visual_mesh_path": str(visual_mesh),
        "collision_mesh_paths": collision_paths,
        "decomposition_method": method,
        "mesh_extents_m": extents.tolist(),
        "visual_volume_m3": visual_volume,
        "collision_volume_m3": collision_volume,
        "density_kg_m3": density,
        "estimated_mass_kg": estimated_mass,
        "friction": [
            float(args.sliding_friction),
            float(args.torsional_friction),
            float(args.rolling_friction),
        ],
        "solref": [float(args.solref_timeconst), float(args.solref_dampratio)],
        "part_count": len(collision_paths),
    }
    manifest_path = args.output_dir / "collision_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved {len(collision_paths)} collision mesh(es): {args.output_dir}")
    print(f"Saved collision manifest: {manifest_path}")
    if method == "convex-hull":
        print(
            "[WARN] Using a single convex hull. Install coacd and rerun with "
            "--method coacd for concave objects such as chairs or handles."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate convex MuJoCo collision meshes from a metric object mesh."
    )
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=["auto", "coacd", "convex-hull"],
        default="auto",
    )
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--max_convex_hulls", type=int, default=32)
    parser.add_argument("--max_hull_vertices", type=int, default=128)
    parser.add_argument("--preprocess_resolution", type=int, default=100)
    parser.add_argument("--density", type=float, default=600.0)
    parser.add_argument("--mass", type=float, default=None)
    parser.add_argument("--sliding_friction", type=float, default=1.0)
    parser.add_argument("--torsional_friction", type=float, default=0.05)
    parser.add_argument("--rolling_friction", type=float, default=0.005)
    parser.add_argument("--solref_timeconst", type=float, default=0.01)
    parser.add_argument("--solref_dampratio", type=float, default=1.0)
    parser.add_argument("--min_extent_m", type=float, default=0.003)
    parser.add_argument("--max_extent_m", type=float, default=5.0)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
