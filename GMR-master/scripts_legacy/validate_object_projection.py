#!/usr/bin/env python3
"""Validate object pose by projecting the mesh onto original video frames.

Uses the monocular_adapter contract
(pose_optimized.npy + cam_K.json + mesh/mesh.obj)
to project the object mesh vertices onto selected video frames and compute
quality metrics against available segmentation masks.

Outputs:
  projection_validation/
    projection_report.json
    overlay_sample.mp4
    sampled_frames/

Usage::

  python GMR-master/scripts/validate_object_projection.py \
    --adapter_dir <clip_dir>/object_reconstruction/monocular_adapter \
    --frames_dir <raw_dir>/all_frames \
    --masks_dir <raw_dir>/video_segmentation/masks \
    --object_name chair \
    --output_dir <clip_dir>/object_reconstruction/projection_validation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


RECONSTRUCTION_SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "do-as-i-do-main"
    / "reconstruction"
    / "scripts"
)
if str(RECONSTRUCTION_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(RECONSTRUCTION_SCRIPTS))

from render_object_candidates import rasterize_silhouette  # noqa: E402
from score_object_candidates import silhouette_metrics  # noqa: E402

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import trimesh

    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


def _load_adapter(adapter_dir: Path) -> dict:
    """Load all adapter contract files."""
    data = {}
    pose_path = adapter_dir / "pose_optimized.npy"
    if not pose_path.exists():
        raise FileNotFoundError(pose_path)
    data["poses"] = np.load(pose_path, allow_pickle=False)

    valid_path = adapter_dir / "pose_valid.npy"
    if valid_path.exists():
        data["valid"] = np.asarray(
            np.load(valid_path, allow_pickle=False),
            dtype=bool,
        ).reshape(-1)
    else:
        data["valid"] = np.ones(len(data["poses"]), dtype=bool)
    observed_path = adapter_dir / "pose_observed.npy"
    if observed_path.exists():
        data["observed"] = np.asarray(
            np.load(observed_path, allow_pickle=False),
            dtype=bool,
        ).reshape(-1)
    else:
        data["observed"] = data["valid"].copy()
    observed_6d_path = adapter_dir / "pose_observed_6d.npy"
    if observed_6d_path.exists():
        data["quality_observed"] = np.asarray(
            np.load(observed_6d_path, allow_pickle=False),
            dtype=bool,
        ).reshape(-1)
    else:
        data["quality_observed"] = data["observed"].copy()

    cam_path = adapter_dir / "cam_K.json"
    if cam_path.exists():
        cam = json.loads(cam_path.read_text(encoding="utf-8"))
        data["K"] = np.asarray(cam.get("K"), dtype=np.float64)
        data["width"] = cam.get("width")
        data["height"] = cam.get("height")
    else:
        raise FileNotFoundError(cam_path)

    mesh_path = adapter_dir / "mesh" / "mesh.obj"
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)
    if HAS_TRIMESH:
        mesh = trimesh.load(mesh_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = (
                mesh.to_geometry()
                if hasattr(mesh, "to_geometry")
                else mesh.dump(concatenate=True)
            )
        data["mesh_vertices"] = np.asarray(mesh.vertices, dtype=np.float64)
        data["mesh_faces"] = np.asarray(mesh.faces, dtype=np.int64)
    else:
        data["mesh_vertices"] = None

    return data


def _project_vertices(vertices, pose, K):
    """Project vertices to image plane. Returns (N,2) pixel coordinates."""
    verts_h = np.hstack([vertices, np.ones((len(vertices), 1))])
    verts_cam_h = verts_h @ pose.T
    verts_cam = verts_cam_h[:, :3]
    in_front = verts_cam[:, 2] > 0.001
    verts_img = verts_cam @ K.T
    u = verts_img[:, 0] / np.clip(verts_img[:, 2], 1e-12, None)
    v = verts_img[:, 1] / np.clip(verts_img[:, 2], 1e-12, None)
    return np.column_stack([u, v]), in_front


def _compute_projection_bbox(uv, in_front):
    """Compute bounding box from projected valid vertices."""
    valid = uv[in_front]
    if len(valid) < 3:
        return None
    x_min, y_min = valid.min(axis=0)
    x_max, y_max = valid.max(axis=0)
    return np.asarray([x_min, y_min, x_max, y_max])


def _load_mask_for_frame(masks_dir: Path, object_name: str, frame_idx: int) -> np.ndarray | None:
    """Try to load a segmentation mask for a given frame."""
    if masks_dir is None or not masks_dir.exists():
        return None
    patterns = [
        masks_dir / f"frame_{frame_idx:06d}_masks" / f"{object_name}.png",
        masks_dir / f"frame_{frame_idx:06d}_masks" / object_name / "mask.png",
        masks_dir / f"frame_{frame_idx:06d}_masks" / object_name / "mask.npy",
        masks_dir / f"{frame_idx:06d}.png",
        masks_dir / f"{frame_idx:04d}.png",
    ]
    for path in patterns:
        if path.suffix == ".npy" and path.exists():
            return np.load(path, allow_pickle=False)
        if path.exists() and HAS_CV2:
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                return mask > 127
    return None


def _mask_bbox(mask: np.ndarray) -> np.ndarray | None:
    """Compute bbox from a binary mask."""
    ys, xs = np.where(mask)
    if len(ys) < 3:
        return None
    return np.asarray([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two [x1,y1,x2,y2] boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def validate(args) -> None:
    if not HAS_CV2:
        raise ImportError("opencv-python (cv2) is required")
    if not HAS_TRIMESH:
        raise ImportError("trimesh is required for mesh loading")

    adapter = _load_adapter(args.adapter_dir)
    poses = adapter["poses"]
    valid = adapter["valid"]
    observed = adapter["observed"]
    quality_observed = adapter["quality_observed"]
    K = adapter["K"]
    vertices = adapter.get("mesh_vertices")
    faces = adapter.get("mesh_faces")
    width = adapter.get("width") or args.default_width
    height = adapter.get("height") or args.default_height

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sampled_dir = args.output_dir / "sampled_frames"
    sampled_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = args.frames_dir

    # Select sample frames to evaluate.
    valid_indices = np.where(valid)[0]
    if len(valid_indices) == 0:
        raise ValueError("no valid frames to evaluate")
    observed_indices = np.where(valid & quality_observed)[0]
    observed_sample_count = min(
        max(args.max_samples // 2, 1),
        len(observed_indices),
    )
    valid_sample_count = min(
        args.max_samples - observed_sample_count,
        len(valid_indices),
    )
    sampled_observed = (
        observed_indices[
            np.linspace(
                0,
                len(observed_indices) - 1,
                observed_sample_count,
                dtype=int,
            )
        ]
        if observed_sample_count
        else np.asarray([], dtype=int)
    )
    sampled_valid = valid_indices[
        np.linspace(
            0,
            len(valid_indices) - 1,
            valid_sample_count,
            dtype=int,
        )
    ]
    sampled = np.unique(np.r_[sampled_observed, sampled_valid])

    metrics = {
        "mask_iou": [],
        "centroid_error_px": [],
        "bbox_scale_ratio": [],
        "in_front_ratio": [],
        "silhouette_iou": [],
        "boundary_chamfer_px": [],
        "area_ratio": [],
    }
    render_frames = []

    writer = None
    if HAS_CV2:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for sample_idx in sampled:
        pose = poses[sample_idx]
        frame_name = f"{sample_idx:06d}"

        # Load frame image.
        img = None
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = frames_dir / f"{frame_name}{ext}"
            if candidate.exists():
                img = cv2.imread(str(candidate))
                break
        if img is None:
            continue

        if writer is None and HAS_CV2:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(
                str(args.output_dir / "overlay_sample.mp4"),
                fourcc,
                args.video_fps,
                (w, h),
            )

        # Project vertices.
        if vertices is not None:
            uv, in_front = _project_vertices(vertices, pose, K)
            proj_bbox = _compute_projection_bbox(uv, in_front)
            in_front_ratio = float(in_front.mean())
            metrics["in_front_ratio"].append(in_front_ratio)

            # Draw projected vertex bbox.
            if proj_bbox is not None:
                cv2.rectangle(
                    img,
                    (int(proj_bbox[0]), int(proj_bbox[1])),
                    (int(proj_bbox[2]), int(proj_bbox[3])),
                    (0, 255, 0),
                    2,
                )

            # Load mask and compare.
            mask = _load_mask_for_frame(
                args.masks_dir, args.object_name, sample_idx
            )
            if mask is not None and proj_bbox is not None:
                if mask.shape != (height, width):
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    ) > 0
                mask_bb = _mask_bbox(mask)
                if mask_bb is not None:
                    cv2.rectangle(
                        img,
                        (int(mask_bb[0]), int(mask_bb[1])),
                        (int(mask_bb[2]), int(mask_bb[3])),
                        (255, 0, 0),
                        2,
                    )
                    silhouette = rasterize_silhouette(
                        vertices,
                        faces,
                        pose,
                        adapter["K"],
                        width,
                        height,
                    )
                    full = silhouette_metrics(silhouette, mask)
                    if quality_observed[sample_idx]:
                        iou = _bbox_iou(proj_bbox, mask_bb)
                        metrics["mask_iou"].append(iou)
                        proj_center = proj_bbox[:2] + (
                            proj_bbox[2:] - proj_bbox[:2]
                        ) / 2
                        mask_center = mask_bb[:2] + (
                            mask_bb[2:] - mask_bb[:2]
                        ) / 2
                        centroid_err = float(
                            np.linalg.norm(proj_center - mask_center)
                        )
                        metrics["centroid_error_px"].append(centroid_err)
                        proj_size = max(
                            proj_bbox[2] - proj_bbox[0],
                            proj_bbox[3] - proj_bbox[1],
                        )
                        mask_size = max(
                            mask_bb[2] - mask_bb[0],
                            mask_bb[3] - mask_bb[1],
                        )
                        if mask_size > 0:
                            metrics["bbox_scale_ratio"].append(
                                float(proj_size / mask_size)
                            )
                        metrics["silhouette_iou"].append(
                            full["silhouette_iou"]
                        )
                        metrics["boundary_chamfer_px"].append(
                            full["boundary_chamfer_px"]
                        )
                        metrics["area_ratio"].append(full["area_ratio"])
                    overlay = img.copy()
                    overlay[silhouette] = (0, 255, 0)
                    img = cv2.addWeighted(img, 0.65, overlay, 0.35, 0)

        # Save sample frame.
        out_path = sampled_dir / f"frame_{frame_name}.png"
        cv2.imwrite(str(out_path), img)
        render_frames.append(str(out_path))

        if writer is not None:
            writer.write(img)

    if writer is not None:
        writer.release()

    # Build report.
    warnings = []
    errors = []

    mask_iou_mean = (
        float(np.mean(metrics["mask_iou"])) if metrics["mask_iou"] else None
    )
    centroid_error_median = (
        float(np.median(metrics["centroid_error_px"]))
        if metrics["centroid_error_px"]
        else None
    )
    bbox_scale_ratio_median = (
        float(np.median(metrics["bbox_scale_ratio"]))
        if metrics["bbox_scale_ratio"]
        else None
    )
    valid_projection_ratio = (
        float(np.mean(metrics["in_front_ratio"]))
        if metrics["in_front_ratio"]
        else None
    )
    silhouette_iou_p10 = (
        float(np.percentile(metrics["silhouette_iou"], 10))
        if metrics["silhouette_iou"]
        else None
    )
    boundary_chamfer_p90_px = (
        float(np.percentile(metrics["boundary_chamfer_px"], 90))
        if metrics["boundary_chamfer_px"]
        else None
    )
    projected_area_ratio_p05 = (
        float(np.percentile(metrics["area_ratio"], 5))
        if metrics["area_ratio"]
        else None
    )
    projected_area_ratio_p95 = (
        float(np.percentile(metrics["area_ratio"], 95))
        if metrics["area_ratio"]
        else None
    )
    projected_area_log_error_p95 = (
        float(
            np.percentile(
                np.abs(
                    np.log(
                        np.clip(
                            np.asarray(metrics["area_ratio"]),
                            1e-12,
                            None,
                        )
                    )
                ),
                95,
            )
        )
        if metrics["area_ratio"]
        else None
    )

    if mask_iou_mean is not None and mask_iou_mean < 0.15:
        errors.append(f"mask IoU mean {mask_iou_mean:.3f} < 0.15")
    elif mask_iou_mean is not None and mask_iou_mean < 0.30:
        warnings.append(f"mask IoU mean {mask_iou_mean:.3f} is low")

    if centroid_error_median is not None:
        max_centroid = args.max_centroid_error_px
        if centroid_error_median > max_centroid:
            warnings.append(
                f"centroid error median {centroid_error_median:.1f}px > {max_centroid}px"
            )

    if bbox_scale_ratio_median is not None:
        lo, hi = args.bbox_scale_ratio_range
        if bbox_scale_ratio_median < lo:
            warnings.append(
                f"bbox scale ratio median {bbox_scale_ratio_median:.3f} < {lo}"
            )
        elif bbox_scale_ratio_median > hi:
            warnings.append(
                f"bbox scale ratio median {bbox_scale_ratio_median:.3f} > {hi}"
            )

    if valid_projection_ratio is not None and valid_projection_ratio < 0.3:
        errors.append(
            f"valid projection ratio {valid_projection_ratio:.3f} < 0.3"
        )
    if (
        silhouette_iou_p10 is None
        or silhouette_iou_p10 < args.min_silhouette_iou_hard
    ):
        errors.append(
            "silhouette IoU P10 is missing or below "
            f"{args.min_silhouette_iou_hard:.2f}"
        )
    elif silhouette_iou_p10 < args.min_silhouette_iou_warn:
        warnings.append(
            f"silhouette IoU P10 {silhouette_iou_p10:.3f} < "
            f"{args.min_silhouette_iou_warn:.2f}"
        )
    if (
        projected_area_ratio_p95 is None
        or projected_area_ratio_p95 > 3.0
        or projected_area_ratio_p05 is None
        or projected_area_ratio_p05 < 0.30
    ):
        errors.append("projected area ratio is outside hard limits")
    elif (
        projected_area_ratio_p95 > 1.8
        or projected_area_ratio_p05 < 0.55
    ):
        warnings.append("projected area ratio is outside warning limits")
    if (
        boundary_chamfer_p90_px is None
        or boundary_chamfer_p90_px > 40.0
    ):
        errors.append("boundary Chamfer P90 is missing or above 40 px")
    elif boundary_chamfer_p90_px > 20.0:
        warnings.append(
            f"boundary Chamfer P90 {boundary_chamfer_p90_px:.1f}px > 20px"
        )

    ok = len(errors) == 0
    report = {
        "schema_version": 2,
        "ok": ok,
        "adapter_dir": str(args.adapter_dir),
        "frames_sampled": len(sampled),
        "observed_frames_sampled": int(
            np.count_nonzero(observed[sampled])
        ),
        "quality_observed_frames_sampled": int(
            np.count_nonzero(quality_observed[sampled])
        ),
        "mask_iou_mean": mask_iou_mean,
        "centroid_error_px_median": centroid_error_median,
        "bbox_scale_ratio_median": bbox_scale_ratio_median,
        "valid_projection_ratio": valid_projection_ratio,
        "silhouette_iou_p10": silhouette_iou_p10,
        "boundary_chamfer_p90_px": boundary_chamfer_p90_px,
        "projected_area_ratio_p05": projected_area_ratio_p05,
        "projected_area_ratio_p95": projected_area_ratio_p95,
        "projected_area_log_error_p95": projected_area_log_error_p95,
        "depth_positive_ratio": (
            valid_projection_ratio  # same as in_front_ratio for our projection
        ),
        "overlay_video": str(args.output_dir / "overlay_sample.mp4"),
        "sampled_frames": render_frames[:10],
        "warnings": warnings,
        "errors": errors,
        "thresholds": {
            "max_centroid_error_px": args.max_centroid_error_px,
            "bbox_scale_ratio_range": args.bbox_scale_ratio_range,
            "min_silhouette_iou_hard": args.min_silhouette_iou_hard,
            "min_silhouette_iou_warn": args.min_silhouette_iou_warn,
        },
    }

    report_path = args.output_dir / "projection_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(
            f"Projection validation FAILED ({len(errors)} errors, {len(warnings)} warnings)"
        )
    if warnings:
        print(f"Projection validation OK with {len(warnings)} warning(s)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate object pose projection against video frames."
    )
    parser.add_argument("--adapter_dir", type=Path, required=True)
    parser.add_argument("--frames_dir", type=Path, required=True)
    parser.add_argument("--masks_dir", type=Path, default=None)
    parser.add_argument("--object_name", default="object")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_samples", type=int, default=30)
    parser.add_argument("--video_fps", type=float, default=10.0)
    parser.add_argument("--default_width", type=int, default=1280)
    parser.add_argument("--default_height", type=int, default=960)
    parser.add_argument("--max_centroid_error_px", type=float, default=80.0)
    parser.add_argument(
        "--min_silhouette_iou_hard",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--min_silhouette_iou_warn",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--bbox_scale_ratio_range",
        type=float,
        nargs=2,
        default=[0.35, 2.8],
    )
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
