"""Predict gravity for a static-camera video given a directory of frames.

Runs GeoCalib per-frame, then aggregates using confidence-weighted spherical
mean with MAD-based outlier rejection.

Usage:
    python predict_video_gravity.py <frames_dir> [--camera_model pinhole|simple_radial|simple_divisional]
"""

import argparse
import json
import math
from pathlib import Path

import torch
import numpy as np

from geocalib import GeoCalib
from geocalib.gravity import Gravity
from geocalib.utils import print_calibration


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def angle_between(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """Angular distance (radians) between two unit vectors."""
    dot = (v1 * v2).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.acos(dot)


def spherical_mean(vecs: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """Compute the weighted spherical mean of unit vectors."""
    if weights is not None:
        weights = weights / weights.sum()
        mean = (vecs * weights.unsqueeze(-1)).sum(dim=0)
    else:
        mean = vecs.mean(dim=0)
    return torch.nn.functional.normalize(mean, dim=-1)


def predict_video_gravity(
    frames_dir: str,
    camera_model: str = "pinhole",
    mad_threshold: float = 3.0,
    max_frames: int | None = None,
    device: str | None = None,
) -> dict:
    """Predict gravity for a static-camera video.

    Args:
        frames_dir: Directory containing image frames.
        camera_model: GeoCalib camera model ('pinhole', 'simple_radial', 'simple_divisional').
        mad_threshold: Outlier rejection threshold in units of MAD (higher = less aggressive).
        max_frames: If set, subsample at most this many frames.
        device: Torch device string. Defaults to CUDA if available.

    Returns:
        dict with keys:
            gravity: Gravity object (final estimate)
            roll_deg, pitch_deg: angles in degrees
            n_frames: total frames processed
            n_inliers: frames kept after outlier rejection
            per_frame: list of per-frame dicts
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    frames_dir = Path(frames_dir)
    image_paths = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    if not image_paths:
        raise ValueError(f"No images found in {frames_dir}")

    if max_frames is not None and len(image_paths) > max_frames:
        # Uniformly subsample
        indices = np.round(np.linspace(0, len(image_paths) - 1, max_frames)).astype(int)
        image_paths = [image_paths[i] for i in indices]

    weights = "pinhole" if camera_model == "pinhole" else "distorted"
    model = GeoCalib(weights=weights).to(device)
    model.eval()

    print(f"Running GeoCalib on {len(image_paths)} frames...")

    gravity_vecs = []   # [N, 3] unit vectors
    confidences = []    # [N] scalar confidence per frame
    per_frame = []

    with torch.no_grad():
        for i, path in enumerate(image_paths):
            try:
                img = model.load_image(str(path)).to(device)
                results = model.calibrate(img, camera_model=camera_model)

                grav = results["gravity"]           # Gravity object, shape [1]
                vec = grav.vec3d.squeeze(0).cpu()  # [3]

                # Use mean of up- and latitude-confidence as frame weight
                up_conf = results["up_confidence"].mean().item()
                lat_conf = results["latitude_confidence"].mean().item()
                conf = (up_conf + lat_conf) / 2.0

                gravity_vecs.append(vec)
                confidences.append(conf)
                per_frame.append({
                    "path": str(path),
                    "vec": vec,
                    "roll_deg": math.degrees(grav.roll.item()),
                    "pitch_deg": math.degrees(grav.pitch.item()),
                    "confidence": conf,
                    "outlier": False,
                })

                if (i + 1) % 10 == 0 or i == len(image_paths) - 1:
                    print(f"  [{i+1}/{len(image_paths)}] roll={per_frame[-1]['roll_deg']:.1f}°  "
                          f"pitch={per_frame[-1]['pitch_deg']:.1f}°  conf={conf:.3f}")

            except Exception as e:
                print(f"  Warning: failed on {path.name}: {e}")

    if not gravity_vecs:
        raise RuntimeError("No frames successfully processed.")

    vecs = torch.stack(gravity_vecs)        # [N, 3]
    confs = torch.tensor(confidences)       # [N]

    # --- Step 1: Compute initial unweighted spherical mean ---
    mean_vec = spherical_mean(vecs)

    # --- Step 2: Compute per-frame angle to the mean ---
    angles = angle_between(vecs, mean_vec.unsqueeze(0).expand_as(vecs))  # [N]

    # --- Step 3: MAD-based outlier rejection ---
    median_angle = angles.median()
    mad = (angles - median_angle).abs().median()
    mad = mad.clamp(min=1e-6)  # avoid division by zero

    threshold = median_angle + mad_threshold * mad
    inlier_mask = angles <= threshold

    n_inliers = inlier_mask.sum().item()
    print(f"\nOutlier rejection: {n_inliers}/{len(vecs)} frames kept "
          f"(threshold={math.degrees(threshold.item()):.2f}°, "
          f"median angle={math.degrees(median_angle.item()):.2f}°, "
          f"MAD={math.degrees(mad.item()):.2f}°)")

    for i, is_outlier in enumerate((~inlier_mask).tolist()):
        per_frame[i]["outlier"] = is_outlier

    # --- Step 4: Confidence-weighted spherical mean over inliers ---
    inlier_vecs = vecs[inlier_mask]
    inlier_confs = confs[inlier_mask]

    final_vec = spherical_mean(inlier_vecs, weights=inlier_confs)
    final_gravity = Gravity(final_vec.unsqueeze(0).to(device))

    roll_deg = math.degrees(final_gravity.roll.item())
    pitch_deg = math.degrees(final_gravity.pitch.item())

    return {
        "gravity": final_gravity,
        "vec": final_vec,
        "roll_deg": roll_deg,
        "pitch_deg": pitch_deg,
        "n_frames": len(vecs),
        "n_inliers": n_inliers,
        "per_frame": per_frame,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict gravity for a static-camera video.")
    parser.add_argument("frames_dir", help="Directory of image frames.")
    parser.add_argument(
        "--camera_model",
        default="pinhole",
        choices=["pinhole", "simple_radial", "simple_divisional"],
    )
    parser.add_argument(
        "--mad_threshold",
        type=float,
        default=3.0,
        help="MAD multiplier for outlier rejection (default: 3.0).",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Uniformly subsample to at most this many frames.",
    )
    parser.add_argument("--device", default=None, help="Torch device (default: auto).")
    parser.add_argument("--output_path", default=None, help="Save results as JSON to this path.")
    args = parser.parse_args()

    result = predict_video_gravity(
        args.frames_dir,
        camera_model=args.camera_model,
        mad_threshold=args.mad_threshold,
        max_frames=args.max_frames,
        device=args.device,
    )

    print("\n=== Final Gravity Estimate ===")
    print(f"  Roll:   {result['roll_deg']:+.2f}°")
    print(f"  Pitch:  {result['pitch_deg']:+.2f}°")
    print(f"  Vec3D:  {result['vec'].tolist()}")
    print(f"  Frames: {result['n_inliers']} inliers / {result['n_frames']} total")

    if args.output_path is not None:
        output = {
            "roll_deg": result["roll_deg"],
            "pitch_deg": result["pitch_deg"],
            "vec3d": result["vec"].tolist(),
            "n_frames": result["n_frames"],
            "n_inliers": result["n_inliers"],
            "per_frame": [
                {
                    "path": f["path"],
                    "roll_deg": f["roll_deg"],
                    "pitch_deg": f["pitch_deg"],
                    "confidence": f["confidence"],
                    "outlier": f["outlier"],
                }
                for f in result["per_frame"]
            ],
        }
        Path(args.output_path).write_text(json.dumps(output, indent=2))
        print(f"\nSaved to {args.output_path}")


if __name__ == "__main__":
    main()
