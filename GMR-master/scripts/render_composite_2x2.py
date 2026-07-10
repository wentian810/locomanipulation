#!/usr/bin/env python
"""2x2 composite: Original | GVHMR | Isaac/hand diagnostic | GMR Robot.
Uses ffmpeg H.264 for high-quality output."""

import argparse
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np


def count_video(path):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return n, fps, w, h


def read_all_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f[:, :, ::-1])  # BGR -> RGB
    cap.release()
    return frames


def open_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return cap


def read_frame_rgb(cap, fallback_shape=None):
    ok, frame = cap.read()
    if ok:
        return frame[:, :, ::-1]
    if fallback_shape is None:
        return None
    return np.full(fallback_shape, 200, dtype=np.uint8)


def normalize_frame(frame):
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=2)
    if frame.shape[2] > 3:
        frame = frame[:, :, :3]
    return frame.astype(np.uint8, copy=False)


def resample_frames(frames, target_n):
    if not frames:
        return []
    n = len(frames)
    if n == target_n:
        return frames
    indices = np.linspace(0, n - 1, target_n).round().astype(int)
    return [frames[i] for i in indices]


def letterbox(frame, width, height, bg=(248, 249, 251)):
    h, w = frame.shape[:2]
    s = min(width / float(w), height / float(h))
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), bg, dtype=np.uint8)
    y0, x0 = (height - nh) // 2, (width - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def write_composite_frames(args, frame_sources, target_n, panel_w, panel_h):
    divider_w = 6
    divider_color = (229, 231, 235)
    h_divider = np.full((divider_w, panel_w * 2 + divider_w, 3), divider_color, dtype=np.uint8)
    v_divider = np.full((panel_h, divider_w, 3), divider_color, dtype=np.uint8)

    total_w = panel_w * 2 + divider_w
    total_w -= total_w % 2
    total_h = panel_h * 2 + divider_w
    total_h -= total_h % 2
    labels = [args.label_original, args.label_gvhmr, args.label_isaac, args.label_gmr]

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{total_w}x{total_h}",
        "-pix_fmt", "bgr24",
        "-r", str(args.fps),
        "-i", "-",
        "-c:v", "libx264",
        "-crf", str(args.crf),
        "-preset", args.preset,
        "-pix_fmt", "yuv420p",
        args.output,
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    try:
        for idx in range(target_n):
            orig, gvhmr, isaac, gmr = frame_sources(idx)

            tl = letterbox(normalize_frame(orig), panel_w, panel_h)
            tr = letterbox(normalize_frame(gvhmr), panel_w, panel_h)
            bl = letterbox(normalize_frame(isaac), panel_w, panel_h)
            br = letterbox(normalize_frame(gmr), panel_w, panel_h)

            top_row = np.concatenate([tl, v_divider, tr], axis=1)
            bottom_row = np.concatenate([bl, v_divider, br], axis=1)
            combined = np.concatenate([top_row, h_divider, bottom_row], axis=0)

            if not args.hide_labels:
                add_labels(combined, labels, panel_w, panel_h, divider_w)

            proc.stdin.write(combined[:, :, ::-1].tobytes())

            if (idx + 1) % 100 == 0:
                print(f"[2x2] Encoded {idx + 1}/{target_n} frames", flush=True)
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")
    print(f"[2x2] Done: {args.output}", flush=True)


def build_streaming(args, counts):
    orig_meta, gvhmr_meta, isaac_meta, gmr_meta = counts
    orig_n, _, w0, h0 = orig_meta
    target_n = orig_n
    if args.max_frames > 0:
        target_n = min(target_n, args.max_frames)

    if args.panel_width is None:
        panel_w = max(2, int(round(args.panel_height * (w0 / float(h0)))))
        panel_w -= panel_w % 2
    else:
        panel_w = args.panel_width
    panel_h = args.panel_height
    panel_h -= panel_h % 2

    print(f"[2x2] Streaming 4 videos frame-by-frame", flush=True)
    print(f"[2x2] Output: {target_n} frames @{args.fps}fps, panel={panel_w}x{panel_h}", flush=True)

    fourth_source = args.diagnostic or args.isaac
    caps = [
        open_video(args.original),
        open_video(args.gvhmr),
        open_video(fourth_source),
        open_video(args.gmr),
    ]
    fallback_shape = (h0, w0, 3)

    def next_frames(_idx):
        frames = [read_frame_rgb(cap, fallback_shape) for cap in caps]
        if frames[0] is None:
            raise RuntimeError(f"Original video ended before frame {_idx}")
        return frames

    try:
        write_composite_frames(args, next_frames, target_n, panel_w, panel_h)
    finally:
        for cap in caps:
            cap.release()


def add_labels(canvas, labels, panel_w, panel_h, divider):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, label in enumerate(labels):
        col = i % 2
        row = i // 2
        x = col * (panel_w + divider) + 12
        y = row * (panel_h + divider) + 36
        cv2.putText(canvas, label, (x, y), font, 0.7, (31, 41, 55), 2, cv2.LINE_AA)


def build(args):
    third_source = args.diagnostic or args.isaac
    if not third_source:
        raise ValueError("Provide either --isaac or --diagnostic for the lower-left panel")
    counts = [
        count_video(args.original),
        count_video(args.gvhmr),
        count_video(third_source),
        count_video(args.gmr),
    ]
    frame_counts = [item[0] for item in counts]
    if frame_counts and all(n == frame_counts[0] and n > 0 for n in frame_counts):
        return build_streaming(args, counts)

    # --- Read all sources ---
    print(f"[2x2] Reading original: {args.original}", flush=True)
    orig_frames = read_all_frames(args.original)
    if not orig_frames:
        raise RuntimeError(f"No frames in {args.original}")

    target_n = len(orig_frames)
    print(f"[2x2] Original frames: {target_n}", flush=True)

    print(f"[2x2] Reading GVHMR: {args.gvhmr}", flush=True)
    gvhmr_frames = read_all_frames(args.gvhmr)
    if not gvhmr_frames:
        gvhmr_frames = [np.full(orig_frames[0].shape, 200, dtype=np.uint8)]
    print(f"[2x2] GVHMR frames: {len(gvhmr_frames)}", flush=True)

    print(f"[2x2] Reading panel 3: {third_source}", flush=True)
    isaac_frames = read_all_frames(third_source)
    if not isaac_frames:
        isaac_frames = [np.full(orig_frames[0].shape, 200, dtype=np.uint8)]
    print(f"[2x2] ISSAC frames: {len(isaac_frames)}", flush=True)

    print(f"[2x2] Reading GMR: {args.gmr}", flush=True)
    gmr_frames = read_all_frames(args.gmr)
    if not gmr_frames:
        gmr_frames = [np.full(orig_frames[0].shape, 200, dtype=np.uint8)]
    print(f"[2x2] GMR frames: {len(gmr_frames)}", flush=True)

    # --- Determine panel size ---
    h0, w0 = orig_frames[0].shape[:2]
    if args.panel_width is None:
        panel_w = max(2, int(round(args.panel_height * (w0 / float(h0)))))
        panel_w -= panel_w % 2
    else:
        panel_w = args.panel_width
    panel_h = args.panel_height
    panel_h -= panel_h % 2

    divider_w = 6
    divider_color = (229, 231, 235)
    h_divider = np.full((divider_w, panel_w * 2 + divider_w, 3), divider_color, dtype=np.uint8)
    v_divider = np.full((panel_h, divider_w, 3), divider_color, dtype=np.uint8)

    # --- Resample all to same length ---
    orig_frames = resample_frames(orig_frames, target_n)
    gvhmr_frames = resample_frames(gvhmr_frames, target_n)
    isaac_frames = resample_frames(isaac_frames, target_n)
    gmr_frames = resample_frames(gmr_frames, target_n)

    if args.max_frames > 0 and target_n > args.max_frames:
        orig_frames = resample_frames(orig_frames, args.max_frames)
        gvhmr_frames = resample_frames(gvhmr_frames, args.max_frames)
        isaac_frames = resample_frames(isaac_frames, args.max_frames)
        gmr_frames = resample_frames(gmr_frames, args.max_frames)
        target_n = args.max_frames

    print(f"[2x2] Output: {target_n} frames @{args.fps}fps, panel={panel_w}x{panel_h}", flush=True)

    def frame_sources(idx):
        return orig_frames[idx], gvhmr_frames[idx], isaac_frames[idx], gmr_frames[idx]

    write_composite_frames(args, frame_sources, target_n, panel_w, panel_h)


def parse_args():
    p = argparse.ArgumentParser(description="2x2 video composite")
    p.add_argument("--original", required=True, help="Original input video")
    p.add_argument("--gvhmr", required=True, help="GVHMR incam render MP4")
    panel3 = p.add_mutually_exclusive_group(required=True)
    panel3.add_argument("--isaac", default="", help="Isaac Gym physics MP4")
    panel3.add_argument(
        "--diagnostic",
        default="",
        help="Hand diagnostic MP4 (for a no-PHC, hand-focused composite)",
    )
    p.add_argument("--gmr", required=True, help="GMR robot retargeting MP4")
    p.add_argument("--output", required=True, help="Output MP4 path")
    p.add_argument("--panel_width", type=int, default=None)
    p.add_argument("--panel_height", type=int, default=360)
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--label_original", default="Original Video")
    p.add_argument("--label_gvhmr", default="GVHMR")
    p.add_argument("--label_isaac", default="Isaac Gym / Hand Diagnostics")
    p.add_argument("--label_gmr", default="GMR Robot")
    p.add_argument("--hide_labels", action="store_true")
    p.add_argument("--crf", type=int, default=17, help="ffmpeg H.264 CRF (lower=better, 17-23 typical)")
    p.add_argument("--preset", default="medium", help="ffmpeg libx264 preset (medium/fast/slow)")
    return p.parse_args()


if __name__ == "__main__":
    build(parse_args())
