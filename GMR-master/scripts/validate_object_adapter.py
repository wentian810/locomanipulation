#!/usr/bin/env python3
"""Validate the monocular_adapter contract before feeding it into the object bridge.

Checks every requirement listed in the unified contract:

* pose_raw.npy / pose_optimized.npy: (T,4,4) T_camera_object
* pose_valid.npy: bool array, sufficient coverage
* pose_observed.npy: bool array, minimum observation ratio
* pose_predicted/confidence/covariance: optimized trajectory provenance
* raw_candidates.npz: numeric top-K archive loadable without pickle
* mesh/mesh.obj: exists, reasonable extents
* cam_K.json: 3x3 intrinsics, resolution, convention metadata
* adapter_manifest.json: source, scale, warnings

Usage::

  python GMR-master/scripts/validate_object_adapter.py \
    --adapter_dir <clip_dir>/object_reconstruction/monocular_adapter \
    --clip_dir <clip_dir> \
    --video <work_video_or_input_video> \
    --output <clip_dir>/object_reconstruction/object_adapter_validation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _longest_false_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in np.asarray(values, dtype=bool).reshape(-1):
        current = 0 if value else current + 1
        longest = max(longest, current)
    return longest


def _expected_frame_count(video_path: Path | None) -> int | None:
    if video_path is None or not video_path.exists():
        return None
    import subprocess

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        value = result.stdout.strip()
        if value and value.isdigit():
            return int(value)
    except Exception:
        pass
    # Fallback: count all_frames/*.png
    for parent in (video_path.parent / "all_frames", video_path.parent):
        pngs = sorted(parent.glob("*.png"))
        if pngs:
            return len(pngs)
    return None


def validate(args) -> None:
    adapter_dir = args.adapter_dir.resolve()
    errors: list[str] = []
    warnings: list[str] = []

    # --- raw and optimized trajectories ---
    raw_pose_path = adapter_dir / "pose_raw.npy"
    pose_path = adapter_dir / "pose_optimized.npy"
    if not raw_pose_path.exists():
        errors.append("pose_raw.npy missing")
    if not pose_path.exists():
        errors.append("pose_optimized.npy missing")
        _write_report(args, adapter_dir, errors, warnings)
        return
    poses = np.load(pose_path, allow_pickle=False)
    if raw_pose_path.exists():
        raw_poses = np.load(raw_pose_path, allow_pickle=False)
        if raw_poses.shape != poses.shape:
            errors.append(
                "pose_raw.npy shape does not match pose_optimized.npy"
            )

    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        errors.append(
            f"pose_optimized.npy shape must be (T,4,4), got {poses.shape}"
        )
    else:
        if not np.isfinite(poses).all():
            errors.append("pose_optimized.npy contains NaN / Inf")
        bottom = poses[:, 3]
        expected = np.asarray([0.0, 0.0, 0.0, 1.0])
        if not np.allclose(bottom, expected, atol=1e-4):
            errors.append(
                "pose_optimized.npy has invalid homogeneous bottom rows"
            )
        z_depths = poses[:, 2, 3]
        bad_z = ~np.isfinite(z_depths) | (z_depths <= 0.001)
        if bad_z.mean() > 0.6:
            errors.append(
                "pose_optimized.npy: "
                f"{bad_z.mean():.1%} frames have z <= 0.001"
            )
        elif bad_z.mean() > 0.1:
            warnings.append(
                "pose_optimized.npy: "
                f"{bad_z.mean():.1%} frames have z <= 0.001"
            )

    # --- pose_valid.npy ---
    valid_path = adapter_dir / "pose_valid.npy"
    if not valid_path.exists():
        errors.append("pose_valid.npy missing")
    else:
        valid = np.asarray(
            np.load(valid_path, allow_pickle=False),
            dtype=bool,
        ).reshape(-1)
        if len(valid) != len(poses):
            errors.append(
                "pose_valid.npy length "
                f"{len(valid)} != pose_optimized.npy length {len(poses)}"
            )
        else:
            valid_ratio = float(valid.mean())
            if valid_ratio < args.min_valid_ratio:
                errors.append(
                    f"valid_ratio={valid_ratio:.3f} < {args.min_valid_ratio}"
                )
            if valid_ratio < 0.5:
                warnings.append(f"valid_ratio only {valid_ratio:.3f}")
            longest_invalid_gap = _longest_false_run(valid)
            if longest_invalid_gap > args.max_invalid_gap:
                errors.append(
                    "longest invalid gap "
                    f"{longest_invalid_gap} > {args.max_invalid_gap} frames"
                )

    # --- pose_observed.npy ---
    observed_path = adapter_dir / "pose_observed.npy"
    observed_ratio = None
    if observed_path.exists():
        observed = np.asarray(
            np.load(observed_path, allow_pickle=False),
            dtype=bool,
        ).reshape(-1)
        if len(observed) == len(poses):
            observed_ratio = float(observed.mean())
            if observed_ratio < args.min_observed_ratio:
                errors.append(
                    f"observed_ratio={observed_ratio:.3f} < {args.min_observed_ratio}"
                )
        else:
            warnings.append(
                "pose_observed.npy length "
                f"{len(observed)} != pose_optimized.npy length {len(poses)}"
            )
    else:
        warnings.append("pose_observed.npy missing; cannot distinguish observed vs interpolated")

    predicted_path = adapter_dir / "pose_predicted.npy"
    confidence_path = adapter_dir / "pose_confidence.npy"
    covariance_path = adapter_dir / "pose_covariance.npy"
    if not predicted_path.exists():
        errors.append("pose_predicted.npy missing")
    else:
        predicted = np.asarray(
            np.load(predicted_path, allow_pickle=False),
            dtype=bool,
        ).reshape(-1)
        if len(predicted) != len(poses):
            errors.append("pose_predicted.npy length mismatch")
        elif "valid" in locals() and np.any(predicted & ~valid):
            errors.append("pose_predicted contains frames not marked valid")
        elif "observed" in locals() and np.any(predicted & observed):
            errors.append("pose_predicted overlaps pose_observed")
        elif (
            "valid" in locals()
            and "observed" in locals()
            and not np.array_equal(predicted, valid & ~observed)
        ):
            errors.append(
                "pose_predicted must equal pose_valid AND NOT pose_observed"
            )
    if not confidence_path.exists():
        errors.append("pose_confidence.npy missing")
    else:
        confidence = np.asarray(
            np.load(confidence_path, allow_pickle=False),
            dtype=np.float64,
        ).reshape(-1)
        if len(confidence) != len(poses):
            errors.append("pose_confidence.npy length mismatch")
        elif (
            not np.isfinite(confidence).all()
            or np.any(confidence < 0)
            or np.any(confidence > 1)
        ):
            errors.append("pose_confidence.npy must be finite and in [0,1]")
        elif "valid" in locals() and np.any(valid):
            confidence_p10 = float(np.percentile(confidence[valid], 10))
            if confidence_p10 < args.min_confidence_p10:
                errors.append(
                    "pose confidence P10 "
                    f"{confidence_p10:.3f} < {args.min_confidence_p10}"
                )
    if not covariance_path.exists():
        errors.append("pose_covariance.npy missing")
    else:
        covariance = np.asarray(
            np.load(covariance_path, allow_pickle=False),
            dtype=np.float64,
        )
        if covariance.shape != (len(poses), 6, 6):
            errors.append(
                "pose_covariance.npy must have shape "
                f"({len(poses)},6,6), got {covariance.shape}"
            )
        elif not np.isfinite(covariance).all():
            errors.append("pose_covariance.npy contains NaN / Inf")
        elif not np.allclose(
            covariance,
            np.swapaxes(covariance, 1, 2),
            atol=1e-7,
        ):
            errors.append("pose_covariance.npy is not symmetric")

    candidates_path = adapter_dir / "raw_candidates.npz"
    if not candidates_path.exists():
        errors.append("raw_candidates.npz missing")
    else:
        try:
            with np.load(candidates_path, allow_pickle=False) as candidates:
                required_candidate_keys = {
                    "pose_camera_object",
                    "candidate_valid",
                    "silhouette_iou",
                    "boundary_chamfer_px",
                    "area_ratio",
                    "total_score",
                }
                missing_keys = required_candidate_keys - set(candidates.files)
                if missing_keys:
                    errors.append(
                        "raw_candidates.npz missing keys: "
                        + ", ".join(sorted(missing_keys))
                    )
                elif candidates["pose_camera_object"].shape[0] != len(poses):
                    errors.append(
                        "raw_candidates.npz frame count does not match poses"
                    )
        except (OSError, ValueError) as exc:
            errors.append(f"raw_candidates.npz is not safely loadable: {exc}")

    interaction_path = adapter_dir / "interaction_state.npy"
    anchors_path = adapter_dir / "contact_anchors.npz"
    if not interaction_path.exists():
        errors.append("interaction_state.npy missing")
    else:
        interaction = np.asarray(
            np.load(interaction_path, allow_pickle=False),
            dtype=np.int64,
        ).reshape(-1)
        if len(interaction) != len(poses):
            errors.append("interaction_state.npy length mismatch")
        elif np.any((interaction < 0) | (interaction > 5)):
            errors.append("interaction_state.npy contains unknown codes")
    if not anchors_path.exists():
        errors.append("contact_anchors.npz missing")
    else:
        try:
            with np.load(anchors_path, allow_pickle=False) as anchors:
                for key in (
                    "left_object_local",
                    "right_object_local",
                    "left_active",
                    "right_active",
                ):
                    if key not in anchors.files:
                        errors.append(
                            f"contact_anchors.npz missing {key}"
                        )
        except (OSError, ValueError) as exc:
            errors.append(f"contact_anchors.npz is unsafe: {exc}")

    # --- frame count vs video ---
    expected = _expected_frame_count(args.video)
    if expected is not None:
        diff = abs(len(poses) - expected)
        if diff > 1:
            errors.append(
                "pose_optimized.npy has "
                f"{len(poses)} frames but video/all_frames has {expected} "
                f"(diff={diff})"
            )
        elif diff == 1:
            warnings.append(
                "pose_optimized.npy has "
                f"{len(poses)} frames, video has {expected} (off by 1)"
            )

    # --- cam_K.json ---
    cam_path = adapter_dir / "cam_K.json"
    if not cam_path.exists():
        errors.append("cam_K.json missing")
    else:
        cam = json.loads(cam_path.read_text(encoding="utf-8"))
        K = np.asarray(cam.get("K", []))
        if K.shape != (3, 3):
            errors.append(f"cam_K.json K shape is {K.shape}, expected (3,3)")
        elif not np.isfinite(K).all():
            errors.append("cam_K.json K contains NaN / Inf")
        convention = str(cam.get("camera_convention", cam.get("convention", "")))
        if "opencv" not in convention.lower():
            warnings.append(f"camera convention is not OpenCV: {convention}")
        scale_source = str(cam.get("scale_source", ""))
        if not scale_source:
            warnings.append("cam_K.json does not document scale_source")
        if cam.get("width") is None or cam.get("height") is None:
            warnings.append("cam_K.json missing width/height")
        if cam.get("data_source") is None:
            warnings.append("cam_K.json missing data_source")

    # --- mesh/mesh.obj ---
    mesh_path = adapter_dir / "mesh" / "mesh.obj"
    canonicalization_path = (
        adapter_dir / "mesh" / "canonicalization.json"
    )
    mesh_extents = None
    if not mesh_path.exists():
        errors.append(f"mesh missing: {mesh_path}")
    else:
        try:
            import trimesh

            mesh = trimesh.load(mesh_path, process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = (
                    mesh.to_geometry()
                    if hasattr(mesh, "to_geometry")
                    else mesh.dump(concatenate=True)
                )
            extents = np.asarray(mesh.extents)
            mesh_extents = extents.tolist()
            if not np.isfinite(extents).all():
                errors.append("mesh extents contain NaN / Inf")
            else:
                max_extent = float(extents.max())
                if max_extent <= 0:
                    errors.append(f"mesh max extent is {max_extent:.5f}")
                elif (
                    args.min_extent_m is not None
                    and max_extent < args.min_extent_m
                ):
                    errors.append(
                        f"mesh max extent {max_extent:.3f}m < {args.min_extent_m}m"
                    )
                elif (
                    args.max_extent_m is not None
                    and max_extent > args.max_extent_m
                ):
                    errors.append(
                        f"mesh max extent {max_extent:.3f}m > {args.max_extent_m}m"
                    )
        except ImportError:
            warnings.append("trimesh not available; skipping mesh extent check")
        except Exception as exc:
            errors.append(f"mesh load failed: {exc}")
    if not canonicalization_path.exists():
        errors.append("mesh/canonicalization.json missing")
    else:
        canonicalization = json.loads(
            canonicalization_path.read_text(encoding="utf-8")
        )
        transform = np.asarray(
            canonicalization.get("T_source_to_canonical"),
            dtype=np.float64,
        )
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            errors.append("canonical transform must be finite (4,4)")
        if canonicalization.get("up_axis") != "z":
            errors.append("canonical mesh up_axis must be z")
        if abs(float(canonicalization.get("support_plane_z", np.inf))) > 1e-6:
            errors.append("canonical support_plane_z must be zero")

    # --- adapter_manifest.json ---
    manifest_path = adapter_dir / "adapter_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("scale_warning"):
            warnings.append(f"scale: {manifest['scale_warning']}")
        interpolated = manifest.get("interpolated_valid_frames", 0)
        total_valid = manifest.get("valid_frames", 1)
        if total_valid > 0 and interpolated / total_valid > 0.5:
            warnings.append(
                f"interpolated frames ({interpolated}) > 50% of valid ({total_valid})"
            )
        if manifest.get("scale_mode") != "global_fixed":
            errors.append("adapter manifest scale_mode must be global_fixed")
        if manifest.get("per_frame_scale_present") is not False:
            errors.append(
                "adapter manifest must explicitly reject per-frame scale"
            )
    else:
        warnings.append("adapter_manifest.json missing")

    ok = len(errors) == 0
    report = {
        "schema_version": 2,
        "ok": ok,
        "adapter_dir": str(adapter_dir),
        "frame_count": len(poses),
        "valid_ratio": (
            round(float(valid.mean()), 4) if "valid" in dir() else None
        ),
        "observed_ratio": (
            round(observed_ratio, 4) if observed_ratio is not None else None
        ),
        "longest_invalid_gap_frames": (
            _longest_false_run(valid) if "valid" in dir() else None
        ),
        "mesh_extents_m": mesh_extents,
        "warnings": warnings,
        "errors": errors,
    }
    _write_report(args, adapter_dir, errors, warnings, report)


def _write_report(
    args,
    adapter_dir: Path,
    errors: list[str],
    warnings: list[str],
    report: dict | None = None,
) -> None:
    if report is None:
        report = {
            "schema_version": 2,
            "ok": len(errors) == 0,
            "adapter_dir": str(adapter_dir),
            "frame_count": None,
            "valid_ratio": None,
            "observed_ratio": None,
            "mesh_extents_m": None,
            "warnings": warnings,
            "errors": errors,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(
            f"Adapter validation FAILED ({len(errors)} errors, {len(warnings)} warnings)"
        )
    if warnings:
        print(f"Adapter validation OK with {len(warnings)} warning(s)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate the monocular_adapter object contract."
    )
    parser.add_argument("--adapter_dir", type=Path, required=True)
    parser.add_argument("--clip_dir", type=Path, default=None)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min_valid_ratio", type=float, default=0.35)
    parser.add_argument("--min_observed_ratio", type=float, default=0.50)
    parser.add_argument("--max_invalid_gap", type=int, default=15)
    parser.add_argument("--min_confidence_p10", type=float, default=0.20)
    parser.add_argument("--min_extent_m", type=float, default=None)
    parser.add_argument("--max_extent_m", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
