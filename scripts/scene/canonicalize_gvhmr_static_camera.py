"""Publish a fixed-camera GVHMR coordinate contract without hiding motion defects.

``static_exact_reprojection`` preserves the historical transformation

    p_ref[t] = inv(T_w2c[reference]) @ T_w2c[t] @ p_world[t]

exactly.  It is useful to diagnose camera estimation drift, but it can inject a
large root discontinuity into an otherwise smooth GVHMR sequence.  The
production ``static_optimized`` mode instead chooses one robust camera pose,
freezes camera metadata to it, and preserves the native human trajectory.  It
therefore makes the reprojection trade-off explicit and refuses outputs that
amplify root motion beyond the configured quality gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _require_new(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _step_lengths(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return np.zeros(0, dtype=np.float64)
    return np.linalg.norm(np.diff(points, axis=0), axis=1)


def _acceleration_lengths(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return np.zeros(0, dtype=np.float64)
    return np.linalg.norm(np.diff(points, n=2, axis=0), axis=1)


def _rotation_angles(reference: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    relative = rotations @ reference.T
    trace = np.trace(relative, axis1=1, axis2=2)
    return np.degrees(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))


def _as_rotvec(matrices: np.ndarray) -> np.ndarray:
    return np.stack(
        [cv2.Rodrigues(matrix)[0].reshape(3) for matrix in matrices]
    ).astype(np.float32)


def _camera_centers(w2c: np.ndarray) -> np.ndarray:
    return np.linalg.inv(w2c)[:, :3, 3]


def _robust_reference_frame(w2c: np.ndarray, fallback: int) -> int:
    """Choose an observed camera pose close to median translation and rotation.

    Selecting an observed pose, instead of independently averaging rotation and
    translation, keeps the frozen extrinsic a valid calibrated camera transform.
    """
    centers = _camera_centers(w2c)
    rotations = w2c[:, :3, :3]
    median_center = np.median(centers, axis=0)
    translation_error = np.linalg.norm(centers - median_center, axis=1)
    rotation_error = _rotation_angles(rotations[fallback], rotations)

    def normalize(values: np.ndarray) -> np.ndarray:
        scale = float(np.median(values))
        return values / scale if scale > 1e-8 else np.zeros_like(values)

    score = normalize(translation_error) + normalize(rotation_error)
    return int(np.argmin(score))


def _translation_for_row_vector_bridge(camera: dict[str, np.ndarray]) -> np.ndarray:
    rotation = np.asarray(camera["world_to_isaac"], dtype=np.float64)
    source = np.asarray(camera["subject_world"], dtype=np.float64)
    target = np.asarray(camera["subject_isaac"], dtype=np.float64)
    return np.median(target - source @ rotation.T, axis=0)


def _fixed_camera_payload(
    camera: dict[str, np.ndarray],
    *,
    fixed_w2c: np.ndarray,
    subject_world: np.ndarray,
    fixed_frame: int,
    mode: str,
) -> dict[str, np.ndarray]:
    frame_count = len(subject_world)
    output = dict(camera)
    output["T_w2c"] = np.repeat(fixed_w2c[None], frame_count, axis=0).astype(np.float32)
    output["subject_world"] = np.asarray(subject_world, dtype=np.float32)
    isaac_translation = _translation_for_row_vector_bridge(camera)
    output["subject_isaac"] = (
        output["subject_world"]
        @ np.asarray(camera["world_to_isaac"], dtype=np.float32).T
        + isaac_translation
    ).astype(np.float32)
    for name in ("camera_pos_world", "camera_target_world"):
        if name in camera:
            output[name] = np.repeat(
                np.asarray(camera[name])[fixed_frame][None], frame_count, axis=0
            )
    for source_name, target_name in (
        ("camera_pos_world", "camera_pos_isaac"),
        ("camera_target_world", "camera_target_isaac"),
    ):
        if source_name in output and target_name in output:
            output[target_name] = (
                output[source_name]
                @ np.asarray(camera["world_to_isaac"], dtype=np.float32).T
                + isaac_translation
            ).astype(np.float32)
    output["camera_canonicalization_reference_frame"] = np.array(
        fixed_frame, dtype=np.int32
    )
    output["camera_canonicalization_mode"] = np.array(mode, dtype="<U32")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-npz", required=True, type=Path)
    parser.add_argument("--camera-npz", required=True, type=Path)
    parser.add_argument("--output-motion-npz", required=True, type=Path)
    parser.add_argument("--output-camera-npz", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--reference-frame", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("static_exact_reprojection", "static_optimized"),
        default="static_optimized",
    )
    parser.add_argument(
        "--freeze-method",
        choices=("reference_frame", "robust_median"),
        default="robust_median",
    )
    parser.add_argument("--max-root-step-m", type=float, default=0.05)
    parser.add_argument("--max-root-step-amplification", type=float, default=2.0)
    args = parser.parse_args()

    if args.max_root_step_m <= 0.0 or args.max_root_step_amplification <= 0.0:
        raise ValueError("root continuity thresholds must be positive")
    for path in (
        args.output_motion_npz,
        args.output_camera_npz,
        args.report_json,
    ):
        _require_new(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    with np.load(args.motion_npz, allow_pickle=False) as archive:
        motion = {key: archive[key] for key in archive.files}
    with np.load(args.camera_npz, allow_pickle=False) as archive:
        camera = {key: archive[key] for key in archive.files}

    poses = np.asarray(motion["poses"], dtype=np.float32)
    translation = np.asarray(motion["trans"], dtype=np.float32)
    w2c = np.asarray(camera["T_w2c"], dtype=np.float64)
    frame_count = len(poses)
    if poses.shape != (frame_count, 24, 3) or translation.shape != (frame_count, 3):
        raise ValueError("motion poses/trans must have shapes (T,24,3)/(T,3)")
    if w2c.shape != (frame_count, 4, 4):
        raise ValueError(f"T_w2c must have shape ({frame_count},4,4), got {w2c.shape}")
    if not 0 <= args.reference_frame < frame_count:
        raise ValueError(f"reference frame must be within [0,{frame_count - 1}]")
    if np.asarray(camera.get("subject_world")).shape != (frame_count, 3):
        raise ValueError("camera.subject_world must have shape (T,3)")

    fixed_frame = (
        args.reference_frame
        if args.freeze_method == "reference_frame"
        else _robust_reference_frame(w2c, args.reference_frame)
    )
    fixed_w2c = w2c[fixed_frame]
    translation_camera = (
        np.einsum("tij,tj->ti", w2c[:, :3, :3], translation) + w2c[:, :3, 3]
    )
    root_world = np.stack([cv2.Rodrigues(item)[0] for item in poses[:, 0]])
    root_camera = np.einsum("tij,tjk->tik", w2c[:, :3, :3], root_world)

    if args.mode == "static_exact_reprojection":
        c2w_fixed = np.linalg.inv(fixed_w2c)
        output_translation = (
            np.einsum("ij,tj->ti", c2w_fixed[:3, :3], translation_camera)
            + c2w_fixed[:3, 3]
        ).astype(np.float32)
        root_reference = np.einsum(
            "ij,tjk->tik", c2w_fixed[:3, :3], root_camera
        )
        output_poses = poses.copy()
        output_root_orient = _as_rotvec(root_reference)
        output_poses[:, 0] = output_root_orient
        subject_camera = (
            np.einsum(
                "tij,tj->ti", w2c[:, :3, :3], camera["subject_world"]
            )
            + w2c[:, :3, 3]
        )
        output_subject_world = (
            np.einsum("ij,tj->ti", c2w_fixed[:3, :3], subject_camera)
            + c2w_fixed[:3, 3]
        )
    else:
        output_translation = translation.copy()
        output_poses = poses.copy()
        output_root_orient = np.asarray(poses[:, 0], dtype=np.float32)
        output_subject_world = np.asarray(camera["subject_world"], dtype=np.float32)

    output_motion = dict(motion)
    output_motion["poses"] = output_poses
    output_motion["root_orient"] = output_root_orient
    output_motion["trans"] = output_translation
    output_camera = _fixed_camera_payload(
        camera,
        fixed_w2c=fixed_w2c,
        subject_world=output_subject_world,
        fixed_frame=fixed_frame,
        mode=args.mode,
    )
    np.savez_compressed(args.output_motion_npz, **output_motion)
    np.savez_compressed(args.output_camera_npz, **output_camera)

    output_camera_translation = (
        fixed_w2c[:3, :3] @ output_translation.T
    ).T + fixed_w2c[:3, 3]
    projection_error = np.linalg.norm(
        output_camera_translation - translation_camera, axis=1
    )
    output_root_world = np.stack(
        [cv2.Rodrigues(item)[0] for item in output_poses[:, 0]]
    )
    rotation_error = np.linalg.norm(
        fixed_w2c[:3, :3] @ output_root_world - root_camera, axis=(1, 2)
    )
    native_steps = _step_lengths(translation)
    output_steps = _step_lengths(output_translation)
    native_max_step = float(native_steps.max()) if native_steps.size else 0.0
    output_max_step = float(output_steps.max()) if output_steps.size else 0.0
    amplification = output_max_step / max(native_max_step, 1e-8)
    camera_centers = _camera_centers(w2c)
    report = {
        "schema_version": 2,
        "mode": args.mode,
        "freeze_method": args.freeze_method,
        "reference_frame": int(args.reference_frame),
        "fixed_camera_frame": int(fixed_frame),
        "frame_count": frame_count,
        "source_motion": str(args.motion_npz.resolve()),
        "source_camera": str(args.camera_npz.resolve()),
        "output_motion": str(args.output_motion_npz.resolve()),
        "output_camera": str(args.output_camera_npz.resolve()),
        "camera_center_step_m": {
            "before": _summary(_step_lengths(camera_centers)),
            "after": _summary(_step_lengths(_camera_centers(output_camera["T_w2c"]))),
        },
        "camera_center_drift_m": {
            "before": float(np.linalg.norm(camera_centers[-1] - camera_centers[0])),
            "after": 0.0,
        },
        "root_step_m": {
            "before": _summary(native_steps),
            "after": _summary(output_steps),
            "amplification_ratio": amplification,
        },
        "root_acceleration_step_m": {
            "before": _summary(_acceleration_lengths(translation)),
            "after": _summary(_acceleration_lengths(output_translation)),
        },
        "fixed_camera_reprojection_error": {
            "root_translation_p50_m": float(np.quantile(projection_error, 0.50)),
            "root_translation_p90_m": float(np.quantile(projection_error, 0.90)),
            "root_rotation_frobenius_p90": float(np.quantile(rotation_error, 0.90)),
        },
        # Kept for downstream packages produced by the previous schema.
        "canonicalization_reprojection_invariance": {
            "root_translation_p90_m": float(np.quantile(projection_error, 0.90)),
            "root_rotation_frobenius_p90": float(np.quantile(rotation_error, 0.90)),
        },
        "quality_gate": {
            "max_root_step_m": float(args.max_root_step_m),
            "max_root_step_amplification": float(args.max_root_step_amplification),
        },
    }
    failures = []
    if output_max_step > args.max_root_step_m:
        failures.append("root_max_step_exceeded")
    if amplification > args.max_root_step_amplification:
        failures.append("root_step_amplification_exceeded")
    report["status"] = "accepted" if not failures else "rejected"
    report["failure_codes"] = failures
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if failures:
        raise RuntimeError(
            "fixed-camera canonicalization rejected: " + ", ".join(failures)
        )
    print(
        "[PASS] fixed-camera canonicalization: "
        f"mode={args.mode}, fixed_frame={fixed_frame}, "
        f"root_step_max={output_max_step:.4f} m",
        flush=True,
    )


if __name__ == "__main__":
    main()
