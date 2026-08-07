#!/usr/bin/env python3
"""Stage isolated VideoMimic policy inputs from a reproducible JSON manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np


class ContractError(RuntimeError):
    """Raised when a Stage-2 batch input is incomplete or ambiguous."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(project: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project / path).resolve()


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ContractError("manifest.schema_version must equal 1")
    if value.get("task") != "g1_deepmimic_proj_heightfield":
        raise ContractError("only g1_deepmimic_proj_heightfield is supported")
    clips = value.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ContractError("manifest.clips must be a non-empty list")
    return value


def _stage_reference(
    source: Path, destination: Path, reference_time_scale: float
) -> dict[str, Any]:
    if not source.is_file():
        raise ContractError(f"missing Stage-4 H5: {source}")
    shutil.copy2(source, destination)
    with h5py.File(destination, "r+") as archive:
        required = ("root_pos", "root_quat", "joints", "link_pos", "link_quat", "contacts")
        missing = [name for name in required if name not in archive]
        if missing:
            raise ContractError(f"Stage-4 H5 lacks datasets {missing}: {source}")
        frames = int(archive["root_pos"].shape[0])
        if frames <= 1 or archive["joints"].shape[0] != frames or archive["link_pos"].shape[0] != frames:
            raise ContractError(f"inconsistent Stage-4 frame contract: {source}")
        source_fps = float(archive.attrs["fps"])
        if not np.isfinite(source_fps) or source_fps <= 0.0:
            raise ContractError(f"Stage-4 H5 has invalid fps metadata: {source}")
        effective_fps = source_fps / reference_time_scale
        for plain_name in ("fps", "joint_names", "link_names"):
            slash_name = f"/{plain_name}"
            if plain_name not in archive.attrs:
                raise ContractError(f"Stage-4 H5 lacks attribute {plain_name}: {source}")
            if plain_name == "fps":
                archive.attrs[plain_name] = effective_fps
                archive.attrs[slash_name] = effective_fps
                continue
            if slash_name in archive.attrs and not np.array_equal(
                archive.attrs[slash_name], archive.attrs[plain_name]
            ):
                raise ContractError(f"conflicting H5 attribute alias {slash_name}: {source}")
            archive.attrs[slash_name] = archive.attrs[plain_name]
        if not np.isfinite(archive["root_pos"][:]).all() or not np.isfinite(archive["joints"][:]).all():
            raise ContractError(f"Stage-4 H5 contains non-finite motion values: {source}")
        fps = float(archive.attrs["fps"])
    return {
        "frames": frames,
        "source_fps": source_fps,
        "effective_fps": fps,
        "reference_time_scale": reference_time_scale,
        "sha256": _sha256(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    project = args.project_root.resolve()
    manifest_path = args.manifest.resolve()
    config = _load_manifest(manifest_path)
    reference_time_scale = float(config.get("reference_time_scale", 1.0))
    if not np.isfinite(reference_time_scale) or reference_time_scale < 1.0:
        raise ContractError("manifest.reference_time_scale must be finite and at least 1.0")
    output_root = _path(project, str(config.get("output_root", "")))
    if output_root.exists():
        raise ContractError(f"refusing to mix an existing Stage-2 batch: {output_root}")

    staged: list[dict[str, Any]] = []
    names: set[str] = set()
    for item in config["clips"]:
        if not isinstance(item, dict):
            raise ContractError("every manifest.clips entry must be an object")
        clip = item.get("clip")
        if not isinstance(clip, str) or not clip or "/" in clip or "\\" in clip or clip in names:
            raise ContractError(f"invalid or duplicate clip name: {clip!r}")
        names.add(clip)
        reference = _path(project, str(item.get("reference_h5", "")))
        terrain = _path(project, str(item.get("terrain_obj", "")))
        if not terrain.is_file() or terrain.suffix.lower() != ".obj":
            raise ContractError(f"missing terrain OBJ for {clip}: {terrain}")
        target_dir = output_root / "data" / clip
        target_dir.mkdir(parents=True, exist_ok=False)
        target_h5 = target_dir / "retarget_poses_g1.h5"
        reference_info = _stage_reference(reference, target_h5, reference_time_scale)
        target_terrain = target_dir / "background_mesh.obj"
        target_terrain.symlink_to(terrain)
        staged.append(
            {
                "clip": clip,
                "source_reference_h5": str(reference),
                "source_reference_sha256": _sha256(reference),
                "staged_reference_h5": str(target_h5),
                "staged_reference": reference_info,
                "source_terrain_obj": str(terrain),
                "source_terrain_sha256": _sha256(terrain),
                "staged_terrain_obj": str(target_terrain),
            }
        )

    motion_list = [
        {
            "folder_path": item["clip"],
            "human_video_data_pattern": "retarget_poses_g1.h5",
            "human_video_terrain_pattern": "background_mesh.obj",
        }
        for item in staged
    ]
    motion_list_path = output_root / "human_motion_list.yaml"
    # JSON is a valid YAML subset, avoiding a training-environment dependency here.
    motion_list_path.write_text(json.dumps(motion_list, indent=2) + "\n", encoding="utf-8")
    report = {
        "schema_version": 1,
        "status": "pass",
        "task": config["task"],
        "reference_time_scale": reference_time_scale,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "output_root": str(output_root),
        "motion_list": str(motion_list_path),
        "clips": staged,
        "release_status": "staged_for_training_not_rollout_accepted",
    }
    (output_root / "stage2_input_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
