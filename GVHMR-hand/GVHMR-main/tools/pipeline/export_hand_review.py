#!/usr/bin/env python3
"""Export a close-up, evidence-aware review video for reconstructed hands.

The normal 2x2 pipeline preview is deliberately compact and therefore not a
valid way to judge a palm/back or wrist-orientation error.  This CPU-only tool
keeps inference untouched and writes one panel per hand:

    source crop + ViTPose evidence | final GVHMR camera-render crop

The crop is driven by the same MANO/DWPose hand bbox used by the reconstruction
track.  Labels distinguish direct source observations from repaired/infilled
frames so a smooth rendered hand is not accidentally treated as visual proof.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch


HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
FINGERTIPS = {4, 8, 12, 16, 20}


def as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_vitpose(path: Path, person_idx: int) -> np.ndarray:
    value = as_numpy(torch.load(path, map_location="cpu", weights_only=True))
    if value.ndim != 4 or value.shape[-2:] != (133, 3):
        raise ValueError(
            "Expected vitpose_wholebody shape (people, frames, 133, 3), "
            f"got {value.shape} from {path}"
        )
    if person_idx < 0:
        person_idx = int(np.argmax(value[:, :, :17, 2].mean(axis=(1, 2))))
    if person_idx >= value.shape[0]:
        raise ValueError(
            f"person_idx={person_idx} is unavailable; VitPose has {value.shape[0]} people"
        )
    return value[person_idx].astype(np.float32)


def person_track(value, person_idx: int, key: str) -> np.ndarray:
    array = as_numpy(value)
    if array.ndim < 2:
        raise ValueError(f"{key} must have person and frame dimensions, got {array.shape}")
    if person_idx < 0:
        person_idx = 0
    if person_idx >= array.shape[0]:
        raise ValueError(f"{key} has only {array.shape[0]} people, requested {person_idx}")
    return np.asarray(array[person_idx])


def bool_track(data, side: str, suffix: str, person_idx: int, frame_count: int, default=False):
    key = f"{side}_hand_{suffix}"
    if key not in data:
        return np.full(frame_count, bool(default), dtype=bool)
    track = person_track(data[key], person_idx, key).astype(bool)
    return track[:frame_count]


def float_track(data, side: str, suffix: str, person_idx: int, frame_count: int, default=np.nan):
    key = f"{side}_hand_{suffix}"
    if key not in data:
        return np.full(frame_count, default, dtype=np.float32)
    track = person_track(data[key], person_idx, key).astype(np.float32)
    return track[:frame_count]


def load_mano_tracks(path: Path, person_idx: int):
    data = torch.load(path, map_location="cpu", weights_only=False)
    result = {}
    frame_count = None
    for side in ("left", "right"):
        bbox_key = f"{side}_hand_bbox_xyxy"
        valid_key = f"{side}_hand_valid"
        if bbox_key not in data or valid_key not in data:
            raise KeyError(f"{path} must contain {bbox_key} and {valid_key}")
        valid = person_track(data[valid_key], person_idx, valid_key).astype(bool)
        if frame_count is None:
            frame_count = int(valid.shape[0])
        elif frame_count != int(valid.shape[0]):
            raise ValueError("Left/right MANO track frame counts differ")
        result[side] = {
            "bbox": person_track(data[bbox_key], person_idx, bbox_key).astype(np.float32),
            "valid": valid,
            # Reliable is the key distinction for review.  A temporal fill can
            # be valid but is not an original image observation.
            "reliable": bool_track(data, side, "reliable_mask", person_idx, len(valid), default=True),
            "temporal_fixed": bool_track(data, side, "temporal_fixed_mask", person_idx, len(valid)),
            "low_evidence": bool_track(data, side, "low_evidence_mask", person_idx, len(valid)),
            "finger_low_evidence": bool_track(data, side, "finger_low_evidence_mask", person_idx, len(valid)),
            "finger_fixed": bool_track(data, side, "finger_fixed_mask", person_idx, len(valid)),
            "bbox_shrink": bool_track(data, side, "bbox_shrink_mask", person_idx, len(valid)),
            "bbox_jump": bool_track(data, side, "bbox_jump_mask", person_idx, len(valid)),
            "bbox_overlap": bool_track(data, side, "bbox_overlap_mask", person_idx, len(valid)),
            "wrist_changed": bool_track(
                data, side, "wrist_global_orient_changed_mask", person_idx, len(valid)
            ),
            "reproj": float_track(data, side, "reproj_error", person_idx, len(valid)),
        }
    return result, int(frame_count)


def draw_text(frame, text, org, color=(255, 255, 255), scale=0.45):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def bbox_from_keypoints(keypoints, min_size=96.0):
    keypoints = np.asarray(keypoints, dtype=np.float32)
    valid = np.isfinite(keypoints[:, :2]).all(axis=1) & (keypoints[:, 2] > 0.2)
    if int(valid.sum()) < 3:
        return None
    points = keypoints[valid, :2]
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    side = max(float(x2 - x1), float(y2 - y1), float(min_size))
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    return np.asarray([cx - side * 0.5, cy - side * 0.5, cx + side * 0.5, cy + side * 0.5], dtype=np.float32)


def valid_bbox(bbox):
    bbox = np.asarray(bbox, dtype=np.float32)
    return bool(
        bbox.shape == (4,)
        and np.isfinite(bbox).all()
        and (bbox[2] - bbox[0]) > 2.0
        and (bbox[3] - bbox[1]) > 2.0
    )


def normalized_crop_box(bbox, source_w, source_h, context_scale):
    bbox = np.asarray(bbox, dtype=np.float32)
    cx = float((bbox[0] + bbox[2]) * 0.5) / max(source_w, 1)
    cy = float((bbox[1] + bbox[3]) * 0.5) / max(source_h, 1)
    side = max(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])) * float(context_scale)
    return cx, cy, side / max(float(source_w), float(source_h), 1.0)


def crop_square(frame, normalized_box, output_size):
    height, width = frame.shape[:2]
    cx, cy, normalized_side = normalized_box
    side = max(4.0, normalized_side * max(width, height))
    cx *= width
    cy *= height
    x1 = int(np.floor(cx - side * 0.5))
    y1 = int(np.floor(cy - side * 0.5))
    x2 = int(np.ceil(cx + side * 0.5))
    y2 = int(np.ceil(cy + side * 0.5))
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - width)
    pad_bottom = max(0, y2 - height)
    cropped = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
    cropped = cv2.copyMakeBorder(
        cropped,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(12, 12, 12),
    )
    if cropped.size == 0:
        cropped = np.zeros((int(side), int(side), 3), dtype=np.uint8)
    crop = cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
    return crop, (x1, y1, side)


def draw_local_keypoints(crop, keypoints, crop_geometry, color):
    x1, y1, side = crop_geometry
    if side <= 0:
        return
    points = np.asarray(keypoints, dtype=np.float32).copy()
    points[:, 0] = (points[:, 0] - x1) / side * crop.shape[1]
    points[:, 1] = (points[:, 1] - y1) / side * crop.shape[0]
    for first, second in HAND_EDGES:
        if points[first, 2] > 0.2 and points[second, 2] > 0.2:
            cv2.line(
                crop,
                tuple(np.round(points[first, :2]).astype(int)),
                tuple(np.round(points[second, :2]).astype(int)),
                color,
                2,
                cv2.LINE_AA,
            )
    for index, point in enumerate(points):
        if point[2] <= 0.2:
            continue
        radius = 4 if index in FINGERTIPS else 3
        cv2.circle(crop, tuple(np.round(point[:2]).astype(int)), radius, color, -1, cv2.LINE_AA)


def flags_for(tracks, frame_idx):
    flags = []
    if tracks["temporal_fixed"][frame_idx]:
        flags.append("TEMP_FILL")
    if tracks["low_evidence"][frame_idx] or tracks["finger_low_evidence"][frame_idx]:
        flags.append("LOW_EVIDENCE")
    if tracks["finger_fixed"][frame_idx]:
        flags.append("FINGER_FIXED")
    if tracks["bbox_shrink"][frame_idx]:
        flags.append("BBOX_SHRINK")
    if tracks["bbox_jump"][frame_idx]:
        flags.append("BBOX_JUMP")
    if tracks["bbox_overlap"][frame_idx]:
        flags.append("BBOX_OVERLAP")
    if tracks["wrist_changed"][frame_idx]:
        flags.append("WRIST_CHANGED")
    return flags


def annotate(panel, side, frame_idx, tracks, rendered=False):
    observed = bool(tracks["reliable"][frame_idx])
    status = "OBSERVED" if observed else "REPAIRED / LOW EVIDENCE"
    status_color = (80, 235, 100) if observed else (50, 120, 255)
    draw_text(panel, f"{side.upper()} {'GVHMR RENDER' if rendered else 'SOURCE'} | f={frame_idx}", (8, 18))
    draw_text(panel, status, (8, 38), color=status_color, scale=0.42)
    reproj = float(tracks["reproj"][frame_idx])
    reproj_text = f"reproj={reproj:.1f}px" if np.isfinite(reproj) else "reproj=n/a"
    draw_text(panel, reproj_text, (8, 58), color=(220, 220, 220), scale=0.40)
    flags = flags_for(tracks, frame_idx)
    if flags:
        text = ",".join(flags[:3])
        if len(flags) > 3:
            text += ",..."
        draw_text(panel, text, (8, panel.shape[0] - 10), color=(80, 180, 255), scale=0.36)


def open_video(path: Path, name: str):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {name}: {path}")
    return cap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_video", type=Path, required=True)
    parser.add_argument("--render_video", type=Path, required=True)
    parser.add_argument("--mano_params", type=Path, required=True)
    parser.add_argument("--vitpose_wholebody", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--person_idx", type=int, default=0)
    parser.add_argument("--crop_size", type=int, default=384)
    parser.add_argument("--context_scale", type=float, default=2.5)
    parser.add_argument("--skip", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0)
    args = parser.parse_args()
    if args.crop_size < 96 or args.context_scale <= 1.0 or args.skip < 1:
        raise ValueError("crop_size must be >=96, context_scale >1, and skip >=1")

    vitpose = load_vitpose(args.vitpose_wholebody, args.person_idx)
    tracks, mano_frames = load_mano_tracks(args.mano_params, args.person_idx)
    source = open_video(args.source_video, "source video")
    rendered = open_video(args.render_video, "render video")
    source_w = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    render_w = int(rendered.get(cv2.CAP_PROP_FRAME_WIDTH))
    render_h = int(rendered.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(source.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = min(mano_frames, vitpose.shape[0])
    if args.max_frames > 0:
        frame_count = min(frame_count, args.max_frames)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / args.skip,
        (args.crop_size * 2, args.crop_size * 2),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not write {args.output}")

    fallback = {"left": None, "right": None}
    fallback_count = {"left": 0, "right": 0}
    written = 0
    for frame_idx in range(frame_count):
        source_ok, source_frame = source.read()
        render_ok, render_frame = rendered.read()
        if not source_ok or not render_ok:
            break
        if frame_idx % args.skip:
            continue
        panels = []
        for side, hand_keypoints in (
            ("left", vitpose[frame_idx, -42:-21]),
            ("right", vitpose[frame_idx, -21:]),
        ):
            track = tracks[side]
            bbox = track["bbox"][frame_idx]
            if valid_bbox(bbox):
                selected_bbox = bbox
            else:
                selected_bbox = bbox_from_keypoints(hand_keypoints)
                if selected_bbox is None:
                    selected_bbox = fallback[side]
                else:
                    fallback[side] = selected_bbox
                fallback_count[side] += 1
            if selected_bbox is None:
                selected_bbox = np.asarray(
                    [source_w * 0.45, source_h * 0.45, source_w * 0.55, source_h * 0.55],
                    dtype=np.float32,
                )
            norm_box = normalized_crop_box(
                selected_bbox,
                source_w,
                source_h,
                args.context_scale,
            )
            source_crop, source_geometry = crop_square(source_frame, norm_box, args.crop_size)
            render_crop, _ = crop_square(render_frame, norm_box, args.crop_size)
            draw_local_keypoints(source_crop, hand_keypoints, source_geometry, (0, 230, 255))
            annotate(source_crop, side, frame_idx, track, rendered=False)
            annotate(render_crop, side, frame_idx, track, rendered=True)
            panels.append(np.concatenate([source_crop, render_crop], axis=1))
        writer.write(np.concatenate(panels, axis=0))
        written += 1

    source.release()
    rendered.release()
    writer.release()
    summary_path = args.summary or args.output.with_suffix(".json")
    summary = {
        "schema_version": 1,
        "purpose": "visual review only; not a hand-pose ground-truth metric",
        "source_video": str(args.source_video),
        "render_video": str(args.render_video),
        "mano_params": str(args.mano_params),
        "vitpose_wholebody": str(args.vitpose_wholebody),
        "source_size": [source_w, source_h],
        "render_size": [render_w, render_h],
        "input_frames": int(frame_count),
        "written_frames": int(written),
        "skip": int(args.skip),
        "crop_size": int(args.crop_size),
        "context_scale": float(args.context_scale),
        "bbox_fallback_frames": fallback_count,
        "sides": {
            side: {
                "source_reliable_frames": int(np.sum(track["reliable"][:frame_count])),
                "repaired_or_low_evidence_frames": int(np.sum(~track["reliable"][:frame_count])),
                "temporal_fill_frames": int(np.sum(track["temporal_fixed"][:frame_count])),
                "finger_fixed_frames": int(np.sum(track["finger_fixed"][:frame_count])),
            }
            for side, track in tracks.items()
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved hand review video: {args.output}")
    print(f"Saved hand review summary: {summary_path}")


if __name__ == "__main__":
    main()
