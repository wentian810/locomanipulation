"""Quality-gate and publish a VideoMimic/NKSR scene sidecar package.

The reconstruction work directory is intentionally separate from the human
pipeline output because it includes H5 caches and other large intermediate
artifacts.  This program runs the Phase-2 bridge/contacts checks, builds the
conservative collision representation, then publishes only portable assets.
It refuses to overwrite an existing package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
import numpy as np


def _run(command: list[str], *, cwd: Path) -> None:
    print("[RUN] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _portable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable(item) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        return Path(value).name
    return value


def _copy(source: Path, target: Path) -> None:
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"required scene artifact is missing or empty: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(path: Path, root: Path, *, label: str) -> str:
    """Return a portable path and refuse inputs outside their declared root."""
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must live below {root}: {path}") from exc
    if not relative.parts or ".." in relative.parts:
        raise ValueError(f"{label} is not a safe relative path: {path}")
    return relative.as_posix()


def _file_provenance(path: Path, root: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} is missing or empty: {path}")
    return {
        "relative_path": _relative_path(path, root, label=label),
        "sha256": _sha256(path),
        "bytes": int(path.stat().st_size),
    }


def _npz_provenance(
    path: Path,
    root: Path,
    *,
    label: str,
    required_keys: tuple[str, ...],
) -> dict[str, Any]:
    record = _file_provenance(path, root, label=label)
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in required_keys if key not in archive.files]
        if missing:
            raise ValueError(f"{label} is missing required arrays: {missing}")
        shapes: dict[str, list[int]] = {}
        frame_count: int | None = None
        for key in required_keys:
            value = np.asarray(archive[key])
            if value.ndim < 1 or value.shape[0] <= 0:
                raise ValueError(f"{label}.{key} must be a non-empty time series")
            if frame_count is None:
                frame_count = int(value.shape[0])
            elif int(value.shape[0]) != frame_count:
                raise ValueError(f"{label} has inconsistent temporal lengths")
            shapes[key] = [int(size) for size in value.shape]
    record["frame_count"] = int(frame_count or 0)
    record["required_shapes"] = shapes
    return record


def _sample_time_series_npz(source: Path, target: Path, indices: np.ndarray) -> int:
    with np.load(source, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    if "poses" in payload:
        count = int(len(payload["poses"]))
    else:
        temporal_lengths = [int(value.shape[0]) for value in payload.values() if value.ndim >= 1]
        if not temporal_lengths:
            raise ValueError(f"scene evidence source has no temporal arrays: {source}")
        count = max(temporal_lengths)
    if indices.ndim != 1 or not len(indices) or indices[0] < 0 or indices[-1] >= count:
        raise ValueError("scene evidence frame map is outside the canonical motion")
    sampled = {
        key: value[indices] if value.ndim >= 1 and value.shape[0] == count else value
        for key, value in payload.items()
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **sampled)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--human-dir", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--video-path", required=True, type=Path)
    parser.add_argument("--motion-npz", required=True, type=Path)
    parser.add_argument("--camera-npz", required=True, type=Path)
    parser.add_argument(
        "--canonicalization-report",
        type=Path,
        default=None,
        help="Accepted report for a pre-canonicalized motion/camera pair.  When supplied, the pair is copied rather than canonicalized again.",
    )
    parser.add_argument("--gender", choices=("male", "female", "neutral"), required=True)
    parser.add_argument(
        "--camera-mode",
        choices=("static_exact_reprojection", "static_optimized"),
        default="static_optimized",
    )
    parser.add_argument(
        "--freeze-method",
        choices=("reference_frame", "robust_median"),
        default="robust_median",
    )
    args = parser.parse_args()

    project = args.project_root.resolve()
    human_dir = args.human_dir.resolve()
    work = args.work_dir.resolve()
    output = args.output_root.resolve()
    if not args.clip_id or Path(args.clip_id).name != args.clip_id:
        raise ValueError("clip-id must be a single, non-empty path component")
    if human_dir.name != args.clip_id:
        raise ValueError(f"clip-id does not match human-dir: {args.clip_id!r} != {human_dir.name!r}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing scene package: {output}")
    calibrated = work / "videomimic" / "output_calib_mesh"
    h5 = calibrated / "gravity_calibrated_megahunter.h5"
    first = calibrated / "background_mesh_nksr_first_round.obj"
    filled = calibrated / "background_mesh_nksr_filled.obj"
    frame_map = work / "inputs" / "frame_map.npz"
    scripts = project / "scripts" / "scene"
    for path in (h5, first, filled, frame_map, args.video_path, args.motion_npz, args.camera_npz, args.model_root):
        if not path.exists():
            raise FileNotFoundError(path)
    source_provenance = {
        "schema_version": 1,
        "clip_id": args.clip_id,
        "source_video": _file_provenance(args.video_path, human_dir, label="source video"),
        "source_motion": _npz_provenance(
            args.motion_npz,
            human_dir,
            label="source motion",
            required_keys=("poses", "trans"),
        ),
        "source_camera": _npz_provenance(
            args.camera_npz,
            human_dir,
            label="source camera",
            required_keys=("T_w2c",),
        ),
    }
    with np.load(frame_map, allow_pickle=False) as archive:
        if "human_frame_index" not in archive.files:
            raise ValueError("scene adapter frame_map is missing human_frame_index")
        evidence_indices = np.asarray(archive["human_frame_index"], dtype=np.int64)
    if evidence_indices.ndim != 1 or not len(evidence_indices):
        raise ValueError("scene adapter frame_map has no evidence frames")

    phase = calibrated / "phase2_gvhmr_static_hard"
    contact = calibrated / "phase2_contact"
    alignment = calibrated / "phase2_alignment_static_hard"
    collision = calibrated / "collision_static_hard"
    visual_clean = calibrated / "visual_clean_static_hard"
    for path in (phase, contact, alignment, collision, visual_clean):
        if path.exists():
            raise FileExistsError(f"refusing to mix prior Phase-2 artifacts: {path}")

    canonical_report_path = phase / "canonicalization_quality.json"
    if args.canonicalization_report is None:
        _run(
            [
                sys.executable, str(scripts / "canonicalize_gvhmr_static_camera.py"),
                "--motion-npz", str(args.motion_npz), "--camera-npz", str(args.camera_npz),
                "--output-motion-npz", str(phase / "human_motion_static_hard.npz"),
                "--output-camera-npz", str(phase / "gvhmr_camera_static_hard.npz"),
                "--report-json", str(canonical_report_path),
                "--mode", args.camera_mode,
                "--freeze-method", args.freeze_method,
            ],
            cwd=project,
        )
    else:
        report_path = args.canonicalization_report.resolve()
        report = _read_json(report_path)
        if report.get("status") != "accepted":
            raise RuntimeError(f"pre-canonicalization report is not accepted: {report_path}")
        expected_motion = Path(str(report.get("output_motion", ""))).resolve()
        expected_camera = Path(str(report.get("output_camera", ""))).resolve()
        if expected_motion != args.motion_npz.resolve() or expected_camera != args.camera_npz.resolve():
            raise RuntimeError("pre-canonicalization report does not describe the supplied motion/camera pair")
        _copy(args.motion_npz, phase / "human_motion_static_hard.npz")
        _copy(args.camera_npz, phase / "gvhmr_camera_static_hard.npz")
        _copy(report_path, canonical_report_path)
    full_frame_count = _sample_time_series_npz(
        phase / "human_motion_static_hard.npz",
        phase / "human_motion_static_hard_scene_evidence.npz",
        evidence_indices,
    )
    _sample_time_series_npz(phase / "gvhmr_camera_static_hard.npz", phase / "gvhmr_camera_static_hard_scene_evidence.npz", evidence_indices)

    _run(
        [
            sys.executable, str(scripts / "assess_videomimic_contact.py"),
            "--calibrated-h5", str(h5), "--first-round-mesh", str(first),
            "--model-root", str(args.model_root), "--output-dir", str(contact),
            "--gender", args.gender,
            "--gravity-axis", "z",
        ],
        cwd=project,
    )
    _run(
        [
            sys.executable, str(scripts / "fit_videomimic_alignment.py"),
            "--calibrated-h5", str(h5),
            "--motion-npz", str(phase / "human_motion_static_hard_scene_evidence.npz"),
            "--camera-npz", str(phase / "gvhmr_camera_static_hard_scene_evidence.npz"),
            "--contact-evidence", str(contact / "contact_evidence.json"),
            "--model-root", str(args.model_root), "--output-dir", str(alignment),
            "--gender", args.gender,
            "--source-coordinate-system", "gvhmr_static_camera_reference_neg_y",
            "--camera-mode", args.camera_mode,
        ],
        cwd=project,
    )
    alignment_quality = _read_json(alignment / "alignment_quality.json")
    if alignment_quality.get("status") != "pass":
        raise RuntimeError(
            "VideoMimic-to-GVHMR alignment failed; refusing to publish a scene package: "
            + ", ".join(str(item) for item in alignment_quality.get("failure_codes", []))
        )
    _run(
        [
            sys.executable, str(scripts / "clean_videomimic_static_mesh.py"),
            "--first-round-mesh", str(first), "--filled-mesh", str(filled),
            "--output-mesh", str(visual_clean / "background_mesh_visual_clean.obj"),
            "--report-json", str(visual_clean / "mesh_cleaning_quality.json"),
        ],
        cwd=project,
    )
    _run(
        [
            sys.executable, str(scripts / "build_contact_collision.py"),
            "--calibrated-h5", str(h5), "--first-round-mesh", str(first),
            "--contact-evidence", str(contact / "contact_evidence.json"),
            "--model-root", str(args.model_root), "--output-dir", str(collision),
            "--gender", args.gender,
        ],
        cwd=project,
    )

    canonical_quality = _read_json(canonical_report_path)
    if canonical_quality.get("status") != "accepted":
        raise RuntimeError(
            "fixed-camera canonicalization was not accepted; refusing to publish "
            "a scene package"
        )
    contact_quality = _read_json(contact / "contact_evidence.json")
    collision_quality = _read_json(collision / "quality.json")
    mesh_cleaning_quality = _read_json(visual_clean / "mesh_cleaning_quality.json")
    primitive = _read_json(collision / "primitives.json")
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.stage-", dir=output.parent) as temporary:
        staging = Path(temporary)
        scene_dir = staging / "scene"
        quality_dir = staging / "quality"
        human_dir_out = staging / "human"
        review_dir = staging / "review"
        _copy(visual_clean / "background_mesh_visual_clean.obj", scene_dir / "background_mesh_visual_clean.obj")
        _copy(visual_clean / "mesh_cleaning_quality.json", quality_dir / "mesh_cleaning_quality.json")
        _copy(collision / "residual_mesh.obj", scene_dir / "background_mesh_collision.obj")
        _copy(collision / "residual_mesh.obj", scene_dir / "residual_mesh.obj")
        _copy(alignment / "scene_alignment.npz", staging / "alignment.npz")
        _copy(alignment / "human_scene_contacts.npz", quality_dir / "human_scene_contacts.npz")
        _copy(phase / "human_motion_static_hard.npz", human_dir_out / "human_motion_static_hard.npz")
        _copy(phase / "gvhmr_camera_static_hard.npz", human_dir_out / "gvhmr_camera_static_hard.npz")
        _copy(contact / "first_round_mesh_plus_human.glb", review_dir / "first_round_mesh_plus_human.glb")
        (scene_dir / "primitives.json").parent.mkdir(parents=True, exist_ok=True)
        (scene_dir / "primitives.json").write_text(
            json.dumps(_portable(primitive), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _run(
            [
                sys.executable, str(scripts / "prepare_videomimic_sim_assets.py"),
                "--input-mesh", str(scene_dir / "residual_mesh.obj"),
                "--input-visual-mesh", str(scene_dir / "background_mesh_visual_clean.obj"),
                "--input-primitives", str(scene_dir / "primitives.json"),
                "--input-evidence-mesh", str(first),
                "--output-evidence-mesh", str(scene_dir / "background_mesh_raw.obj"),
                "--alignment-npz", str(staging / "alignment.npz"),
                "--output-mesh", str(scene_dir / "background_mesh_mujoco.obj"),
                "--output-visual-mesh", str(scene_dir / "background_mesh_visual_mujoco.obj"),
                "--output-mujoco", str(scene_dir / "scene_mujoco.xml"),
                "--output-urdf", str(scene_dir / "scene.urdf"),
                "--output-primitives", str(scene_dir / "primitives_mujoco.json"),
                "--report-json", str(scene_dir / "simulation_frame.json"),
            ],
            cwd=project,
        )
        (quality_dir / "alignment_quality.json").write_text(
            json.dumps(_portable(alignment_quality), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (quality_dir / "canonicalization_quality.json").write_text(
            json.dumps(_portable(canonical_quality), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (quality_dir / "contact_evidence.json").write_text(
            json.dumps(_portable(contact_quality), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        canonical_motion = _npz_provenance(
            human_dir_out / "human_motion_static_hard.npz",
            staging,
            label="canonical motion",
            required_keys=("poses", "trans"),
        )
        canonical_camera = _npz_provenance(
            human_dir_out / "gvhmr_camera_static_hard.npz",
            staging,
            label="canonical camera",
            required_keys=("T_w2c",),
        )
        if canonical_motion["frame_count"] != source_provenance["source_motion"]["frame_count"]:
            raise ValueError("canonical motion frame count differs from source motion")
        if canonical_camera["frame_count"] != source_provenance["source_camera"]["frame_count"]:
            raise ValueError("canonical camera frame count differs from source camera")
        input_provenance = {
            **source_provenance,
            "canonical_motion": canonical_motion,
            "canonical_camera": canonical_camera,
            "collision_assets": {
                "phc_mesh": _file_provenance(
                    scene_dir / "background_mesh_mujoco.obj", staging, label="PHC collision mesh"
                ),
                "phc_primitives": _file_provenance(
                    scene_dir / "primitives_mujoco.json", staging, label="PHC primitives"
                ),
                "mujoco_xml": _file_provenance(
                    scene_dir / "scene_mujoco.xml", staging, label="MuJoCo scene XML"
                ),
            },
            "canonicalization": {
                "mode": args.camera_mode,
                "quality_path": "quality/canonicalization_quality.json",
                "quality_sha256": _sha256(quality_dir / "canonicalization_quality.json"),
            },
        }
        aggregate_quality = {
            "schema_version": 2,
            "verdict": collision_quality["verdict"],
            "alignment": {
                "status": alignment_quality["status"],
                "scale_sim3_diagnostic": alignment_quality["scale_sim3_diagnostic"],
                "joint_p90_m": alignment_quality["joint_alignment_error_m"]["p90"],
            },
            "camera": {
                "mode": args.camera_mode,
                "canonicalization": canonical_quality["canonicalization_reprojection_invariance"],
            },
            "mesh_cleaning": _portable(mesh_cleaning_quality),
            "contact": {
                "seated_frame_count": contact_quality["seated_frame_count"],
                "collision": _portable(collision_quality),
            },
            "crisp": collision_quality["fallback_policy"]["crisp"],
        }
        (staging / "scene_quality.json").write_text(
            json.dumps(aggregate_quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifest = {
            "schema_version": 2,
            "status": "pass",
            "backend": "videomimic_nksr",
            "camera_mode": args.camera_mode,
            "frame_count": full_frame_count,
            "scene_evidence_frame_count": int(alignment_quality["frame_count"]),
            "units": "meter",
            "coordinate_system": "gvhmr_static_camera_reference_neg_y",
            "scene_coordinate_system": "videomimic_gravity_calibrated",
            "simulation_coordinate_system": "mujoco_world_z_up",
            "input_provenance": input_provenance,
            "assets": {
                "raw_mesh": "scene/background_mesh_raw.obj",
                "visual_clean_mesh": "scene/background_mesh_visual_clean.obj",
                "collision_mesh": "scene/background_mesh_collision.obj",
                "simulation_collision_mesh": "scene/background_mesh_mujoco.obj",
                "simulation_visual_mesh": "scene/background_mesh_visual_mujoco.obj",
                "urdf": "scene/scene.urdf",
                "mujoco": "scene/scene_mujoco.xml",
                "primitives": "scene/primitives.json",
                "simulation_primitives": "scene/primitives_mujoco.json",
                "simulation_frame": "scene/simulation_frame.json",
                "alignment": "alignment.npz",
                "contacts": "quality/human_scene_contacts.npz",
                "canonical_human": "human/human_motion_static_hard.npz",
            },
        }
        (staging / "scene_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        checksums = []
        for item in sorted(path for path in staging.rglob("*") if path.is_file()):
            checksums.append(f"{_sha256(item)}  {item.relative_to(staging).as_posix()}")
        (staging / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
        staging.replace(output)
    print(f"[PASS] published VideoMimic scene package -> {output}")


if __name__ == "__main__":
    main()
