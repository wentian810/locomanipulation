#!/usr/bin/env python3
"""BootsTAPIR per-frame velocity tracking.

Tracks an object across video frames using per-frame mask re-sampling,
estimates translation + rotation velocity via SVD rigid fitting.

Usage:
  python tapir_velocity_tracking.py \
    --video /path/to/video.mp4 \
    --mask-dir /path/to/video_dir/video_segmentation/masks \
    --object mug
"""

import argparse
import glob
import json
import os
import time

import mediapy as media
import numpy as np
import torch
import torch.nn.functional as F

from tapnet.torch import tapir_model
from tapnet.utils import transforms
from tapnet.utils import viz_utils

# NOTE: do NOT sys.path.append("…/tapnet-src/tapnet") here — that dir contains a
# `torch/` subdir which shadows the real PyTorch package and breaks all torch
# imports. The pip-editable install of tapnet (from tapnet-src/tapnet) makes
# `tapnet.torch` importable as a submodule with no path hackery required.


def sample_points_from_mask(
    frame_idx: int,
    mask_path: str,
    num_points: int,
) -> np.ndarray:
    """Sample query points uniformly along the mask's principal axis."""
    from PIL import Image

    mask = np.array(Image.open(mask_path).convert("L"))
    ys, xs = np.where(mask > 128)
    if len(ys) == 0:
        raise ValueError(f"No foreground pixels found in mask: {mask_path}")

    coords = np.stack([xs, ys], axis=1).astype(np.float64)
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    projections = centered @ Vt[0]
    order = np.argsort(projections)
    indices = np.linspace(0, len(order) - 1, min(num_points, len(order))).astype(np.int32)
    sel_x, sel_y = xs[order[indices]], ys[order[indices]]

    n = len(sel_x)
    t = np.full((n, 1), frame_idx, dtype=np.int32)
    y = sel_y.reshape(-1, 1)
    x = sel_x.reshape(-1, 1)
    return np.concatenate((t, y, x), axis=-1).astype(np.float32)


def find_first_mask_frame(mask_dir: str, object_name: str) -> int:
    """Find the first frame that has a mask for the given object."""
    pattern = os.path.join(mask_dir, f"frame_*_masks/{object_name}.png")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise ValueError(f"No masks found for object '{object_name}' in {mask_dir}")
    frame_dir = os.path.basename(os.path.dirname(matches[0]))
    return int(frame_dir.split("_")[1])


def print_motion_stats(stats: dict, fps: float = 10.0):
    """Print summary of motion statistics."""
    tv = stats["translation_speed"]
    rv = np.abs(stats["rotation_vel"])
    valid_tv = tv[~np.isnan(tv)]
    valid_rv = rv[~np.isnan(rv)]
    n_total = len(tv)
    n_valid = len(valid_tv)

    print(f"\nMotion statistics ({n_valid}/{n_total} valid frame transitions):")
    if n_valid > 0:
        print(f"  Translation speed (px/frame): "
              f"mean={valid_tv.mean():.2f}, max={valid_tv.max():.2f}, std={valid_tv.std():.2f}")
        print(f"  Translation speed (px/sec):   "
              f"mean={valid_tv.mean()*fps:.1f}, max={valid_tv.max()*fps:.1f}")
        print(f"  Rotation speed (deg/frame):   "
              f"mean={np.degrees(valid_rv.mean()):.3f}, max={np.degrees(valid_rv.max()):.3f}")
        print(f"  Rotation speed (deg/sec):     "
              f"mean={np.degrees(valid_rv.mean())*fps:.2f}, max={np.degrees(valid_rv.max())*fps:.2f}")
    else:
        print("  Not enough visible points for reliable estimates.")
    print(f"  Visible points: min={stats['num_visible'].min()}, "
          f"max={stats['num_visible'].max()}")


def save_motion_plot(stats: dict, output_path: str, fps: float = 10.0):
    """Save a plot of translation and rotation velocity over time."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    frames = np.arange(len(stats["translation_speed"]))

    axes[0].plot(frames, stats["translation_speed"], "b-", linewidth=1.5)
    axes[0].set_ylabel("Translation\nspeed (px/frame)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(frames, np.degrees(stats["rotation_vel"]), "r-", linewidth=1.5)
    axes[1].set_ylabel("Rotation\nvelocity (deg/frame)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(np.arange(len(stats["num_visible"])), stats["num_visible"], "g-", linewidth=1.5)
    axes[2].set_ylabel("Visible\npoints")
    axes[2].set_xlabel("Frame")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Object Motion Stats from Point Tracks")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved motion plot to {output_path}")


def overlay_motion_stats(
    video: np.ndarray, stats: dict, alpha: float = 0.6
) -> np.ndarray:
    """Overlay motion stats plots onto the bottom of each video frame."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from io import BytesIO

    num_frames, vid_h, vid_w, _ = video.shape
    chart_h = vid_h // 3
    chart_w = vid_w
    dpi = 150

    trans_speed = stats["translation_speed"]
    rot_vel = stats["rotation_vel"]

    def render_chart_frame(current_frame):
        fig, axes = plt.subplots(
            2, 1, figsize=(chart_w / dpi, chart_h / dpi), dpi=dpi,
            gridspec_kw={"hspace": 0.4},
        )
        fig.patch.set_alpha(0.0)

        frames_x = np.arange(len(trans_speed))

        ax = axes[0]
        ax.set_facecolor((0, 0, 0, 0.5))
        ax.fill_between(frames_x, trans_speed, alpha=0.3, color="cyan")
        ax.plot(frames_x, trans_speed, "c-", linewidth=1.5)
        ax.axvline(x=current_frame, color="yellow", linewidth=2, alpha=0.9)
        ax.set_ylabel("Translation\n(px/frame)", fontsize=9, color="white", fontweight="bold")
        ax.tick_params(colors="white", labelsize=7)
        ax.set_xticklabels([])
        for spine in ax.spines.values():
            spine.set_color("white")
            spine.set_alpha(0.4)
        ax.set_xlim(0, len(trans_speed))
        valid = trans_speed[~np.isnan(trans_speed)]
        if len(valid) > 0:
            ax.set_ylim(0, np.nanmax(valid) * 1.2 + 1)

        ax = axes[1]
        ax.set_facecolor((0, 0, 0, 0.5))
        rot_deg = np.degrees(rot_vel)
        ax.fill_between(frames_x, rot_deg, alpha=0.3, color="magenta")
        ax.plot(frames_x, rot_deg, "m-", linewidth=1.5)
        ax.axvline(x=current_frame, color="yellow", linewidth=2, alpha=0.9)
        ax.axhline(y=0, color="white", linewidth=0.5, alpha=0.3)
        ax.set_ylabel("Rotation\n(deg/frame)", fontsize=9, color="white", fontweight="bold")
        ax.set_xlabel("Frame", fontsize=9, color="white")
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("white")
            spine.set_alpha(0.4)
        ax.set_xlim(0, len(rot_vel))
        valid = np.degrees(rot_vel[~np.isnan(rot_vel)])
        if len(valid) > 0:
            margin = max(np.nanmax(np.abs(valid)) * 1.2, 1)
            ax.set_ylim(-margin, margin)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.tight_layout(pad=0.8)

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, transparent=True,
                    bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        buf.seek(0)

        from PIL import Image
        chart_img = np.array(Image.open(buf).convert("RGBA"))
        chart_img_pil = Image.fromarray(chart_img).resize((chart_w, chart_h), Image.LANCZOS)
        return np.array(chart_img_pil)

    output = video.copy()
    for t in range(num_frames):
        chart_rgba = render_chart_frame(t)
        chart_rgb = chart_rgba[:, :, :3].astype(np.float32)
        chart_alpha = (chart_rgba[:, :, 3:4].astype(np.float32) / 255.0) * alpha

        region = output[t, vid_h - chart_h:vid_h, :chart_w, :].astype(np.float32)
        blended = region * (1 - chart_alpha) + chart_rgb * chart_alpha
        output[t, vid_h - chart_h:vid_h, :chart_w, :] = blended.astype(np.uint8)

    return output


def _save_pair_viz(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    tracks_pair: np.ndarray,
    vis: np.ndarray,
    vis_both: np.ndarray,
    frame_idx: int,
    output_path: str,
):
    """Save a side-by-side visualization of a 2-frame tracking pair."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colormap = viz_utils.get_colors(len(tracks_pair))

    for ax_idx, (frame, fidx_label) in enumerate(
        [(frame_a, f"Frame {frame_idx}"), (frame_b, f"Frame {frame_idx + 1}")]
    ):
        ax = axes[ax_idx]
        ax.imshow(frame)
        ax.set_title(fidx_label, fontsize=12, color="white", fontweight="bold",
                     bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))
        ax.axis("off")

        for i in range(len(tracks_pair)):
            x, y = tracks_pair[i, ax_idx]
            color = tuple(np.array(colormap[i]) / 255.0)
            if vis_both[i]:
                ax.plot(x, y, "o", color=color, markersize=6, markeredgecolor="white",
                        markeredgewidth=0.5)
            elif vis[i, ax_idx]:
                ax.plot(x, y, "x", color=color, markersize=6, alpha=0.5)

    n_vis = vis_both.sum()
    axes[0].set_title(
        f"Frame {frame_idx} ({n_vis} matched)", fontsize=12, color="white",
        fontweight="bold", bbox=dict(boxstyle="round", facecolor="black", alpha=0.7)
    )

    fig.tight_layout(pad=1)
    fig.savefig(output_path, dpi=100, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close(fig)


def compute_perframe_motion_stats(
    video_path: str,
    mask_dir: str,
    mask_name: str,
    model: torch.nn.Module,
    device: torch.device,
    num_points: int = 20,
    resize_hw: tuple = (256, 256),
    threshold: float = 0.5,
    min_visible: int = 5,
    save_dir: str = None,
) -> dict:
    """Estimate motion by re-sampling points from per-frame masks.

    For each consecutive frame pair (t, t+1):
      1. Load mask at frame t, sample points on the object
      2. Run 2-frame TAPIR inference
      3. Fit rigid transform (SVD) to get translation + rotation
    """
    from PIL import Image

    video = media.read_video(video_path)
    num_frames, vid_h, vid_w = video.shape[:3]
    resize_h, resize_w = resize_hw
    frames_resized = media.resize_video(video, (resize_h, resize_w))
    frames_preprocessed = torch.tensor(frames_resized).float().to(device)
    frames_preprocessed = frames_preprocessed / 255 * 2 - 1

    if save_dir:
        viz_dir = os.path.join(save_dir, "pair_visualizations")
        os.makedirs(viz_dir, exist_ok=True)

    centroids = np.full((num_frames, 2), np.nan)
    translation_vel = np.full((num_frames - 1, 2), np.nan)
    rotation_vel = np.full(num_frames - 1, np.nan)
    num_visible = np.zeros(num_frames, dtype=np.int32)
    per_frame_data = []

    t_start = time.time()

    for t in range(num_frames - 1):
        mask_path = os.path.join(mask_dir, f"frame_{t:06d}_masks", mask_name)
        if not os.path.exists(mask_path):
            per_frame_data.append({
                "frame": t, "status": "no_mask",
                "num_sampled": 0, "num_visible_both": 0,
                "translation_dx": None, "translation_dy": None,
                "translation_speed": None, "rotation_deg": None,
            })
            continue

        mask = np.array(Image.open(mask_path).convert("L"))
        ys, xs = np.where(mask > 128)
        if len(ys) < min_visible:
            per_frame_data.append({
                "frame": t, "status": "mask_too_small",
                "num_sampled": 0, "num_visible_both": 0,
                "translation_dx": None, "translation_dy": None,
                "translation_speed": None, "rotation_deg": None,
            })
            continue

        qp = sample_points_from_mask(t, mask_path, num_points)
        n_sampled = len(qp)
        if n_sampled < min_visible:
            per_frame_data.append({
                "frame": t, "status": "too_few_sampled",
                "num_sampled": n_sampled, "num_visible_both": 0,
                "translation_dx": None, "translation_dy": None,
                "translation_speed": None, "rotation_deg": None,
            })
            continue

        qp_resized = transforms.convert_grid_coordinates(
            qp, (1, vid_h, vid_w), (1, resize_h, resize_w),
            coordinate_format="tyx",
        )

        pair = frames_preprocessed[t:t + 2][None]
        qp_tensor = torch.tensor(qp_resized).float().to(device)
        qp_tensor[:, 0] = 0
        qp_tensor = qp_tensor[None]

        with torch.no_grad():
            outputs = model(pair, qp_tensor)

        tracks_pair = outputs["tracks"][0].cpu().numpy()
        occ = outputs["occlusion"][0]
        exp_dist = outputs["expected_dist"][0]
        vis = ((1 - F.sigmoid(occ)) * (1 - F.sigmoid(exp_dist)) > threshold).cpu().numpy()

        tracks_pair = transforms.convert_grid_coordinates(
            tracks_pair, (resize_w, resize_h), (vid_w, vid_h)
        )

        mask_next_path = os.path.join(mask_dir, f"frame_{t+1:06d}_masks", mask_name)
        if os.path.exists(mask_next_path):
            mask_next = np.array(Image.open(mask_next_path).convert("L"))
            on_mask_next = np.array([
                mask_next[
                    int(np.clip(tracks_pair[i, 1, 1], 0, vid_h - 1)),
                    int(np.clip(tracks_pair[i, 1, 0], 0, vid_w - 1)),
                ] > 128
                for i in range(len(tracks_pair))
            ])
        else:
            on_mask_next = np.ones(len(tracks_pair), dtype=bool)

        vis_both = vis[:, 0] & vis[:, 1] & on_mask_next
        n_vis = vis_both.sum()
        num_visible[t] = max(num_visible[t], vis[:, 0].sum())

        if save_dir:
            _save_pair_viz(
                video[t], video[t + 1], tracks_pair, vis, vis_both,
                t, os.path.join(viz_dir, f"pair_{t:04d}_{t+1:04d}.png"),
            )

        if n_vis < min_visible:
            per_frame_data.append({
                "frame": t, "status": "too_few_visible",
                "num_sampled": n_sampled, "num_visible_both": int(n_vis),
                "translation_dx": None, "translation_dy": None,
                "translation_speed": None, "rotation_deg": None,
            })
            continue

        pts_a = tracks_pair[vis_both, 0, :]
        pts_b = tracks_pair[vis_both, 1, :]

        centroid_a = pts_a.mean(axis=0)
        centroid_b = pts_b.mean(axis=0)
        centroids[t] = centroid_a
        if t == num_frames - 2:
            centroids[t + 1] = centroid_b

        translation_vel[t] = centroid_b - centroid_a

        pa = pts_a - centroid_a
        pb = pts_b - centroid_b
        H = pa.T @ pb
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        rotation_vel[t] = np.arctan2(R[1, 0], R[0, 0])

        t_speed = float(np.linalg.norm(translation_vel[t]))
        per_frame_data.append({
            "frame": t, "status": "ok",
            "num_sampled": n_sampled, "num_visible_both": int(n_vis),
            "translation_dx": float(translation_vel[t, 0]),
            "translation_dy": float(translation_vel[t, 1]),
            "translation_speed": t_speed,
            "rotation_deg": float(np.degrees(rotation_vel[t])),
        })

    last_mask = os.path.join(mask_dir, f"frame_{num_frames-1:06d}_masks", mask_name)
    if os.path.exists(last_mask):
        m = np.array(Image.open(last_mask).convert("L"))
        num_visible[num_frames - 1] = (m > 128).sum() > 0

    elapsed = time.time() - t_start

    # Interpolate failed frames from neighbors
    n_interpolated = 0
    for i, entry in enumerate(per_frame_data):
        if entry["status"] == "ok":
            continue
        prev_val = next_val = None
        for j in range(i - 1, -1, -1):
            if per_frame_data[j]["status"] in ("ok", "interpolated"):
                prev_val = per_frame_data[j]
                break
        for j in range(i + 1, len(per_frame_data)):
            if per_frame_data[j]["status"] == "ok":
                next_val = per_frame_data[j]
                break

        if prev_val and next_val:
            alpha = (entry["frame"] - prev_val["frame"]) / (next_val["frame"] - prev_val["frame"])
            for key in ("translation_dx", "translation_dy", "translation_speed", "rotation_deg"):
                entry[key] = prev_val[key] * (1 - alpha) + next_val[key] * alpha
        elif prev_val:
            for key in ("translation_dx", "translation_dy", "translation_speed", "rotation_deg"):
                entry[key] = prev_val[key]
        elif next_val:
            for key in ("translation_dx", "translation_dy", "translation_speed", "rotation_deg"):
                entry[key] = next_val[key]
        else:
            continue

        entry["status"] = "interpolated"
        t = entry["frame"]
        translation_vel[t] = [entry["translation_dx"], entry["translation_dy"]]
        rotation_vel[t] = np.radians(entry["rotation_deg"])
        n_interpolated += 1

    translation_speed = np.linalg.norm(translation_vel, axis=1)

    if n_interpolated > 0:
        print(f"  Interpolated {n_interpolated} failed frame(s)")

    print(f"  Per-frame motion estimation: {elapsed:.2f}s "
          f"({elapsed/(num_frames-1)*1000:.1f}ms/pair)")

    if save_dir:
        json_path = os.path.join(save_dir, "motion_stats.json")
        with open(json_path, "w") as f:
            json.dump(per_frame_data, f, indent=2)
        print(f"  Saved motion JSON to {json_path}")
        print(f"  Saved {len(os.listdir(viz_dir))} pair visualizations to {viz_dir}/")

    return {
        "centroids": centroids,
        "translation_vel": translation_vel,
        "rotation_vel": rotation_vel,
        "translation_speed": translation_speed,
        "num_visible": num_visible,
    }


def load_model(checkpoint: str, device: torch.device) -> torch.nn.Module:
    model = tapir_model.TAPIR(pyramid_level=1)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="BootsTAPIR per-frame velocity tracking"
    )
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--mask-dir", required=True,
                        help="Directory with per-frame masks (frame_NNNNNN_masks/ subdirs)")
    parser.add_argument("--object", required=True,
                        help="Object name (matches <object>.png masks in each frame subdir)")
    parser.add_argument("--num-points", type=int, default=20,
                        help="Number of query points (default: 20)")
    parser.add_argument("--min-visible", type=int, default=5,
                        help="Min visible points for reliable motion estimates (default: 5)")
    parser.add_argument("--checkpoint", default="checkpoints/bootstapir_checkpoint_v2.pt",
                        help="Path to BootsTAPIR checkpoint")
    parser.add_argument("--resize", type=int, nargs=2, default=[256, 256],
                        metavar=("H", "W"),
                        help="Resize resolution for inference (default: 256 256)")
    parser.add_argument("--vis-threshold", type=float, default=0.5,
                        help="Visibility threshold (default: 0.5)")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Video FPS for per-second conversion (default: 10)")
    args = parser.parse_args()

    # Infer paths from video + object name
    video_dir = os.path.dirname(args.video)
    mask_name = f"{args.object}.png"
    output = os.path.join(video_dir, f"output_tapir_{args.object}.mp4")
    save_dir = os.path.join(video_dir, f"perframe_tracking_{args.object}")

    # Find the first frame with a mask for this object
    first_frame = find_first_mask_frame(args.mask_dir, args.object)
    first_mask = os.path.join(args.mask_dir, f"frame_{first_frame:06d}_masks", mask_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.checkpoint, device)

    # Full-video tracking visualization
    video = media.read_video(args.video)
    height, width = video.shape[1:3]
    resize_h, resize_w = args.resize

    query_points = sample_points_from_mask(first_frame, first_mask, args.num_points)
    query_points = transforms.convert_grid_coordinates(
        query_points, (1, height, width), (1, resize_h, resize_w),
        coordinate_format="tyx",
    )
    print(f"Sampled {args.num_points} points from mask: {first_mask}")

    frames = media.resize_video(video, (resize_h, resize_w))
    frames_t = torch.tensor(frames).to(device)
    query_points_t = torch.tensor(query_points).to(device)

    with torch.no_grad():
        frames_pre = frames_t.float() / 255 * 2 - 1
        outputs = model(frames_pre[None], query_points_t.float()[None])
        tracks = outputs["tracks"][0]
        occ = outputs["occlusion"][0]
        exp_dist = outputs["expected_dist"][0]
        visibles = ((1 - F.sigmoid(occ)) * (1 - F.sigmoid(exp_dist)) > args.vis_threshold)

    tracks = tracks.cpu().numpy()
    visibles = visibles.cpu().numpy()
    tracks = transforms.convert_grid_coordinates(
        tracks, (resize_w, resize_h), (width, height)
    )

    video_viz = viz_utils.paint_point_track(video, tracks, visibles)
    media.write_video(output, video_viz, fps=10)
    print(f"Saved visualization to {output}")
    print(f"  Video: {video.shape[0]} frames, {height}x{width}")
    print(f"  Tracked {args.num_points} points")

    # Per-frame motion stats
    print("Computing per-frame motion stats (re-sampling points each frame)...")
    stats = compute_perframe_motion_stats(
        args.video, args.mask_dir, mask_name, model, device,
        num_points=args.num_points, resize_hw=tuple(args.resize),
        threshold=args.vis_threshold, min_visible=args.min_visible,
        save_dir=save_dir,
    )
    print_motion_stats(stats, fps=args.fps)

    plot_path = output.rsplit(".", 1)[0] + "_motion.png"
    save_motion_plot(stats, plot_path, fps=args.fps)

    print("Rendering overlay video...")
    overlay_vid = overlay_motion_stats(video, stats)
    overlay_path = output.rsplit(".", 1)[0] + "_overlay.mp4"
    media.write_video(overlay_path, overlay_vid, fps=10)
    print(f"  Saved overlay video to {overlay_path}")


if __name__ == "__main__":
    main()
