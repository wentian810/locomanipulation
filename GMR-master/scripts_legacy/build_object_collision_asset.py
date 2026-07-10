#!/usr/bin/env python3
"""Build fail-closed MuJoCo collision and inertial assets.

``physics`` mode requires CoACD and a defensible explicit/category mass.
``visual_only`` may use one convex hull, but that output is clearly marked and
must not be treated as dynamic validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


STRONGLY_CONCAVE_CATEGORIES = {
    "chair",
    "stool",
    "suitcase",
    "backpack",
    "basket",
}


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
            if args.mode == "physics":
                raise RuntimeError(
                    "physics mode requires CoACD; convex-hull fallback is "
                    "only allowed in visual_only mode"
                )
            method = "convex-hull"
        else:
            method = "coacd"
    if args.mode == "physics" and method != "coacd":
        category = args.category.lower()
        if (
            category in STRONGLY_CONCAVE_CATEGORIES
            or not args.allow_single_hull_physics
        ):
            raise RuntimeError(
                "single convex-hull collision is forbidden in physics mode "
                f"for category={args.category!r}"
            )

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
    mass = None
    mass_source = "unavailable"
    if args.mass is not None:
        mass = float(args.mass)
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError("--mass must be positive and finite")
        mass_source = "explicit"
    elif args.category_mass_range is not None:
        lower, upper = (float(value) for value in args.category_mass_range)
        if lower <= 0 or upper < lower:
            raise ValueError("category mass range must be positive and ordered")
        mass = 0.5 * (lower + upper)
        mass_source = "category_prior_midpoint"
    elif mesh.is_volume and args.allow_watertight_volume_mass:
        mass = float(args.density) * visual_volume
        mass_source = "watertight_visual_mesh_volume"
    elif args.mode == "physics":
        raise RuntimeError(
            "physics mode requires --mass or --category_mass_range; "
            "non-watertight generated meshes cannot use hull volume as mass"
        )

    canonicalization_path = (
        args.canonicalization
        if args.canonicalization is not None
        else visual_mesh.parent / "canonicalization.json"
    )
    canonicalization = {}
    if canonicalization_path.is_file():
        canonicalization = json.loads(
            canonicalization_path.read_text(encoding="utf-8")
        )
    center_of_mass = np.asarray(
        canonicalization.get(
            "center_of_mass_m",
            np.asarray(mesh.bounds, dtype=np.float64).mean(axis=0).tolist(),
        ),
        dtype=np.float64,
    )
    if center_of_mass.shape != (3,) or not np.isfinite(center_of_mass).all():
        raise ValueError("center of mass must contain three finite values")
    diaginertia = None
    inertia_source = "unavailable"
    if mass is not None:
        x, y, z = extents
        diaginertia = np.asarray(
            [
                mass * (y * y + z * z) / 12.0,
                mass * (x * x + z * z) / 12.0,
                mass * (x * x + y * y) / 12.0,
            ],
            dtype=np.float64,
        )
        inertia_source = "canonical_bbox_uniform_density"
    density = (
        float(mass / visual_volume)
        if mass is not None and visual_volume > 1e-10
        else 0.0
    )

    manifest = {
        "schema_version": 2,
        "validation_mode": args.mode,
        "units": "meter",
        "visual_mesh_path": str(visual_mesh),
        "collision_mesh_paths": collision_paths,
        "decomposition_method": method,
        "mesh_extents_m": extents.tolist(),
        "visual_volume_m3": visual_volume,
        "collision_volume_m3": collision_volume,
        "density_kg_m3": density,
        "mass_kg": mass,
        "estimated_mass_kg": mass,
        "mass_source": mass_source,
        "center_of_mass_m": center_of_mass.tolist(),
        "diaginertia_kg_m2": (
            diaginertia.tolist() if diaginertia is not None else None
        ),
        "inertia_source": inertia_source,
        "category": args.category,
        "visual_mesh_watertight": bool(mesh.is_watertight),
        "mass_uses_collision_hull_volume": False,
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
    parser.add_argument(
        "--mode",
        choices=["visual_only", "physics"],
        default="physics",
    )
    parser.add_argument("--category", default="object")
    parser.add_argument("--canonicalization", type=Path)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--max_convex_hulls", type=int, default=32)
    parser.add_argument("--max_hull_vertices", type=int, default=128)
    parser.add_argument("--preprocess_resolution", type=int, default=100)
    parser.add_argument("--density", type=float, default=600.0)
    parser.add_argument("--mass", type=float, default=None)
    parser.add_argument(
        "--category_mass_range",
        type=float,
        nargs=2,
        default=None,
    )
    parser.add_argument(
        "--allow_watertight_volume_mass",
        action="store_true",
    )
    parser.add_argument(
        "--allow_single_hull_physics",
        action="store_true",
    )
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
