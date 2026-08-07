#!/usr/bin/env python3
"""Compose source/GVHMR/PHC/actual-GMR panels on a common time axis.

Unlike the legacy compositor, this tool does not assume all sources have the
same frame rate.  Each output time ``t`` samples every panel at its own
``round(t * source_fps)`` frame, preventing a 25 Hz actual MuJoCo recording
from being stretched to match 30 Hz input frames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


PANEL_ORDER = (
    ("original", "Original video"),
    ("gvhmr", "GVHMR"),
    ("phc", "PHC (historical comparison)"),
    ("gmr", "GMR actual (audited mj_step)"),
)


def _read_video(path: Path) -> tuple[list[np.ndarray], float]:
    reader = imageio.get_reader(str(path))
    try:
        metadata = reader.get_meta_data()
        fps = float(metadata.get("fps", 0.0))
        frames = [np.asarray(frame) for frame in reader]
    finally:
        reader.close()
    if not frames or not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"cannot read non-empty positive-FPS video {path}")
    return frames, fps


def _resize_letterbox(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized = cv2.resize(frame, (round(source_width * scale), round(source_height * scale)))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    top = (height - resized.shape[0]) // 2
    left = (width - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized[:, :, :3]
    return canvas


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    result = frame.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 30), (0, 0, 0), thickness=-1)
    cv2.putText(result, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key, _ in PANEL_ORDER:
        parser.add_argument(f"--{key}", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--panel-height", type=int, default=360)
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite time-synchronised composite/report")
    if args.panel_height <= 0 or args.fps <= 0.0:
        raise ValueError("panel height and fps must be positive")

    streams: dict[str, tuple[list[np.ndarray], float]] = {}
    for key, _ in PANEL_ORDER:
        streams[key] = _read_video(getattr(args, key))
    durations = {key: len(frames) / fps for key, (frames, fps) in streams.items()}
    output_duration = min(durations.values())
    output_count = int(np.floor(output_duration * args.fps + 1e-9))
    if output_count < 1:
        raise ValueError("videos have no shared nonzero duration")
    panel_width = max(
        round(frames[0].shape[1] * args.panel_height / frames[0].shape[0])
        for frames, _ in streams.values()
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(args.output), format="FFMPEG", fps=args.fps, macro_block_size=1) as writer:
        for output_frame in range(output_count):
            time_s = output_frame / args.fps
            panels = []
            for key, label in PANEL_ORDER:
                frames, source_fps = streams[key]
                source_index = min(len(frames) - 1, int(np.rint(time_s * source_fps)))
                panels.append(_label(_resize_letterbox(frames[source_index], args.panel_height, panel_width), label))
            writer.append_data(np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:]))))

    report = {
        "schema_version": 1,
        "purpose": "time_synchronised_v24_actual_2x2_composite",
        "output": str(args.output.resolve()),
        "output_fps": args.fps,
        "output_frame_count": output_count,
        "output_duration_s": output_count / args.fps,
        "selection": "shortest source duration; all panels sampled by wall-clock time",
        "panels": {
            key: {
                "label": label,
                "path": str(getattr(args, key).resolve()),
                "frame_count": len(streams[key][0]),
                "fps": streams[key][1],
                "duration_s": durations[key],
            }
            for key, label in PANEL_ORDER
        },
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
