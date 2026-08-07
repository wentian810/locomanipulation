#!/usr/bin/env python3
"""Materialise a rejected static-chair fit for diagnostic physics only.

The output is deliberately labelled non-promotable.  It lets a controller
test whether the fitted stable pose can bear gravity, while preserving the
quality gate's decision that the complete clip must not be delivered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from fit_static_chair_to_frozen_gmr import (
    _transform_mjcf,
    _transform_obj,
    _transform_primitives,
    _yaw_matrix,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scene-dir", required=True, type=Path)
    parser.add_argument("--fit-report", required=True, type=Path)
    parser.add_argument("--output-scene-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_scene_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_scene_dir}")
    report = json.loads(args.fit_report.read_text(encoding="utf-8"))
    if report.get("status") == "accepted":
        raise ValueError("accepted fits must use the normal promotion path, not this diagnostic materialiser")
    transform = report.get("transform", {})
    translation = np.asarray(transform.get("translation_xyz_m"), dtype=np.float64)
    pivot = np.asarray(transform.get("pivot_xyz_m"), dtype=np.float64)
    yaw_deg = float(transform.get("yaw_deg"))
    if translation.shape != (3,) or pivot.shape != (3,) or not np.isfinite(translation).all() or not np.isfinite(pivot).all():
        raise ValueError("fit report lacks a finite 3D static chair transform")
    primitives_path = args.source_scene_dir / "scene/semantic_chair_primitives_mujoco.json"
    primitive_payload = json.loads(primitives_path.read_text(encoding="utf-8"))
    body_names = {str(item.get("name")) for item in primitive_payload.get("primitives", [])}
    if not body_names:
        raise ValueError("semantic chair primitive payload has no bodies")
    rotation = _yaw_matrix(np.deg2rad(yaw_deg))
    stage = Path(tempfile.mkdtemp(prefix=f".{args.output_scene_dir.name}.stage-", dir=args.output_scene_dir.parent))
    try:
        shutil.copytree(args.source_scene_dir, stage, dirs_exist_ok=True)
        for relative in (
            "scene/semantic_chair_primitives_mujoco.json",
            "scene/primitives_mujoco.json",
            "scene/semantic_chair_primitives_phc.json",
        ):
            path = stage / relative
            if path.is_file():
                _transform_primitives(path, rotation, translation, pivot)
        _transform_mjcf(stage / "scene/scene_mujoco_semantic_chair.xml", body_names, rotation, translation, pivot)
        _transform_obj(stage / "scene/background_mesh_semantic_chair.obj", rotation, translation, pivot)
        manifest = {
            "schema_version": 1,
            "status": "rejected_diagnostic_only",
            "promotion_forbidden": True,
            "source_scene_dir": str(args.source_scene_dir),
            "fit_report": str(args.fit_report),
            "fit_report_sha256": _sha256(args.fit_report),
            "transform": transform,
            "reason": "parent static chair fit did not pass all-frame quality gates",
        }
        (stage / "rejected_diagnostic_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        stage.replace(args.output_scene_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(json.dumps({"status": "rejected_diagnostic_only", "output_scene_dir": str(args.output_scene_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
