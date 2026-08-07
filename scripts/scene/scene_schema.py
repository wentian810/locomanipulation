"""Paths and JSON helpers shared by the scene-sidecar stages.

The initial implementation deliberately keeps this module dependency-free so
configuration and package validation can run before the separate CRISP Conda
environment has been installed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


SCENE_SCHEMA_VERSION = 2
VIDEOMIMIC_FIXED_CAMERA_MODES = frozenset(
    ("static_exact_reprojection", "static_optimized")
)
FINAL_REQUIRED_FILES = (
    "scene_mesh.obj",
    "scene.urdf",
    "scene_mujoco.xml",
    "sqs_params.npz",
)
ROOT_REQUIRED_FILES = (
    "scene_manifest.json",
    "scene_quality.json",
    "alignment.npz",
)
VIDEOMIMIC_REQUIRED_FILES = (
    "scene/background_mesh_raw.obj",
    "scene/background_mesh_visual_clean.obj",
    "scene/background_mesh_collision.obj",
    "scene/background_mesh_mujoco.obj",
    "scene/background_mesh_visual_mujoco.obj",
    "scene/scene.urdf",
    "scene/scene_mujoco.xml",
    "scene/primitives.json",
    "scene/primitives_mujoco.json",
    "scene/simulation_frame.json",
    "alignment.npz",
    "quality/human_scene_contacts.npz",
    "quality/mesh_cleaning_quality.json",
    "human/human_motion_static_hard.npz",
    "human/gvhmr_camera_static_hard.npz",
    "checksums.sha256",
)


def scene_root(clip_root: Path, output_subdir: str = "scene_reconstruction") -> Path:
    """Return the declared per-clip scene directory without resolving symlinks."""
    candidate = Path(output_subdir)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("scene output_subdir must be a safe relative path")
    return clip_root / candidate


def scene_package_paths(root: Path, backend: str = "crisp") -> dict[str, Path]:
    """Return canonical paths for a declared backend."""
    paths = {name: root / name for name in ROOT_REQUIRED_FILES}
    if backend == "crisp":
        paths.update({f"final/{name}": root / "final" / name for name in FINAL_REQUIRED_FILES})
    elif backend == "videomimic_nksr":
        paths.update({name: root / name for name in VIDEOMIMIC_REQUIRED_FILES})
    else:
        raise ValueError(f"unsupported scene backend: {backend}")
    return paths


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a UTF-8 JSON object and reject arrays/scalars at the boundary."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _safe_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    candidate = Path(value)
    return not candidate.is_absolute() and bool(candidate.parts) and ".." not in candidate.parts


def _validate_file_provenance(
    record: Any,
    *,
    label: str,
    require_temporal_metadata: bool,
) -> list[str]:
    if not isinstance(record, dict):
        return [f"input_provenance.{label} must be an object"]
    errors: list[str] = []
    if not _safe_relative_path(record.get("relative_path")):
        errors.append(f"input_provenance.{label}.relative_path must be safe and relative")
    if not isinstance(record.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
        errors.append(f"input_provenance.{label}.sha256 must be a SHA-256 hex digest")
    if not isinstance(record.get("bytes"), int) or record["bytes"] <= 0:
        errors.append(f"input_provenance.{label}.bytes must be a positive integer")
    if require_temporal_metadata:
        frame_count = record.get("frame_count")
        shapes = record.get("required_shapes")
        if not isinstance(frame_count, int) or frame_count <= 0:
            errors.append(f"input_provenance.{label}.frame_count must be positive")
        if not isinstance(shapes, dict) or not shapes:
            errors.append(f"input_provenance.{label}.required_shapes must be a non-empty object")
        elif isinstance(frame_count, int) and frame_count > 0:
            for name, shape in shapes.items():
                if (
                    not isinstance(name, str)
                    or not isinstance(shape, list)
                    or not shape
                    or any(not isinstance(size, int) or size <= 0 for size in shape)
                    or shape[0] != frame_count
                ):
                    errors.append(
                        f"input_provenance.{label}.required_shapes must use positive shapes with frame_count first"
                    )
                    break
    return errors


def validate_input_provenance(manifest: dict[str, Any], *, required: bool) -> list[str]:
    """Validate the portable provenance shape without requiring NumPy or a GPU."""
    provenance = manifest.get("input_provenance")
    if provenance is None:
        return ["scene_manifest.input_provenance is required"] if required else []
    if not isinstance(provenance, dict):
        return ["scene_manifest.input_provenance must be an object"]
    errors: list[str] = []
    if provenance.get("schema_version") != 1:
        errors.append("input_provenance.schema_version must equal 1")
    clip_id = provenance.get("clip_id")
    if not isinstance(clip_id, str) or not clip_id or Path(clip_id).name != clip_id:
        errors.append("input_provenance.clip_id must be one path component")
    errors.extend(_validate_file_provenance(provenance.get("source_video"), label="source_video", require_temporal_metadata=False))
    for label in ("source_motion", "source_camera", "canonical_motion", "canonical_camera"):
        errors.extend(_validate_file_provenance(provenance.get(label), label=label, require_temporal_metadata=True))
    assets = provenance.get("collision_assets")
    if not isinstance(assets, dict):
        errors.append("input_provenance.collision_assets must be an object")
    else:
        for label in ("phc_mesh", "phc_primitives", "mujoco_xml"):
            errors.extend(_validate_file_provenance(assets.get(label), label=f"collision_assets.{label}", require_temporal_metadata=False))
    canonicalization = provenance.get("canonicalization")
    if not isinstance(canonicalization, dict):
        errors.append("input_provenance.canonicalization must be an object")
    else:
        if canonicalization.get("mode") not in VIDEOMIMIC_FIXED_CAMERA_MODES:
            errors.append(
                "input_provenance.canonicalization.mode must be one of "
                f"{sorted(VIDEOMIMIC_FIXED_CAMERA_MODES)}"
            )
        if not _safe_relative_path(canonicalization.get("quality_path")):
            errors.append("input_provenance.canonicalization.quality_path must be safe and relative")
        if not isinstance(canonicalization.get("quality_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", canonicalization["quality_sha256"]):
            errors.append("input_provenance.canonicalization.quality_sha256 must be a SHA-256 hex digest")
    return errors


def validate_manifest_basics(
    manifest: dict[str, Any],
    *,
    require_input_provenance: bool = False,
) -> list[str]:
    """Return schema-level errors that need no CRISP/MuJoCo/Isaac imports."""
    errors: list[str] = []
    backend = manifest.get("backend")
    expected_schema = 1 if backend == "crisp" else SCENE_SCHEMA_VERSION
    if manifest.get("schema_version") != expected_schema:
        errors.append(
            "scene_manifest.schema_version must equal "
            f"{expected_schema}"
        )
    if backend not in {"crisp", "videomimic_nksr"}:
        errors.append("scene_manifest.backend must be 'crisp' or 'videomimic_nksr'")
    if backend == "crisp" and manifest.get("camera_mode") != "fixed":
        errors.append("CRISP scene_manifest.camera_mode must be 'fixed'")
    if backend == "videomimic_nksr" and manifest.get("camera_mode") not in VIDEOMIMIC_FIXED_CAMERA_MODES:
        errors.append(
            "VideoMimic scene_manifest.camera_mode must be one of "
            f"{sorted(VIDEOMIMIC_FIXED_CAMERA_MODES)}"
        )
    if manifest.get("status") not in {"ok", "pass"}:
        errors.append("scene_manifest.status must be 'ok' or 'pass'")
    if backend == "videomimic_nksr":
        errors.extend(validate_input_provenance(manifest, required=require_input_provenance))
    return errors
