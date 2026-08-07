#!/usr/bin/env python3
"""Smooth an evidence-backed GMR root XY correction over contact episodes."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def _load_motion(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    root = np.asarray(payload.get("root_pos"), dtype=np.float64)
    if root.ndim != 2 or root.shape[1] != 3:
        raise ValueError(f"{path} must contain root_pos [T,3]")
    return payload


def _components(indices: np.ndarray) -> list[np.ndarray]:
    if not len(indices):
        return []
    return list(np.split(indices, np.where(np.diff(indices) > 1)[0] + 1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a bounded temporal ramp to a contact-evidenced GMR root XY correction."
    )
    parser.add_argument("--reference-motion", required=True, type=Path)
    parser.add_argument("--contact-motion", required=True, type=Path)
    parser.add_argument("--contact-anchors", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--contact-key", default="back_contact")
    parser.add_argument("--ramp-frames", type=int, default=6)
    args = parser.parse_args()
    if args.ramp_frames < 0:
        raise ValueError("ramp-frames must be non-negative")

    reference = _load_motion(args.reference_motion)
    corrected = _load_motion(args.contact_motion)
    reference_root = np.asarray(reference["root_pos"], dtype=np.float64)
    contact_root = np.asarray(corrected["root_pos"], dtype=np.float64)
    if contact_root.shape != reference_root.shape:
        raise ValueError("reference and contact motion root_pos shapes differ")
    anchors = np.load(args.contact_anchors)
    if args.contact_key not in anchors or "sit_mask" not in anchors:
        raise ValueError("anchors must contain sit_mask and the requested contact key")
    contact_mask = np.asarray(anchors[args.contact_key], dtype=bool)
    sit_mask = np.asarray(anchors["sit_mask"], dtype=bool)
    if contact_mask.shape != (len(reference_root),) or sit_mask.shape != contact_mask.shape:
        raise ValueError("contact anchors do not match motion length")
    contact_indices = np.flatnonzero(contact_mask & sit_mask)
    sit_indices = np.flatnonzero(sit_mask)
    if not len(contact_indices) or not len(sit_indices):
        raise ValueError("no evidence-backed contact episode to ramp")

    raw_delta = contact_root[:, :2] - reference_root[:, :2]
    ramped_delta = np.zeros_like(raw_delta)
    episodes: list[dict[str, int]] = []
    sit_first, sit_last = int(sit_indices[0]), int(sit_indices[-1])
    for component in _components(contact_indices):
        first, last = int(component[0]), int(component[-1])
        start = max(sit_first, first - args.ramp_frames)
        end = min(sit_last, last + args.ramp_frames)
        if first > start:
            for frame in range(start, first):
                ramped_delta[frame] = raw_delta[first] * (frame - start) / (first - start)
        ramped_delta[first : last + 1] = raw_delta[first : last + 1]
        if end > last:
            for frame in range(last + 1, end + 1):
                ramped_delta[frame] = raw_delta[last] * (end - frame) / (end - last)
        episodes.append({"contact_first": first, "contact_last": last, "ramp_first": start, "ramp_last": end})

    output = dict(corrected)
    root_out = contact_root.copy()
    root_out[:, :2] = reference_root[:, :2] + ramped_delta
    output["root_pos"] = root_out.astype(np.asarray(corrected["root_pos"]).dtype, copy=False)
    output["scene_contact_temporal_ramp"] = {
        "contact_key": args.contact_key,
        "ramp_frames": args.ramp_frames,
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    report = {
        "schema_version": 1,
        "purpose": "gmr_contact_evidenced_root_xy_temporal_ramp",
        "status": "accepted",
        "reference_motion": str(args.reference_motion),
        "contact_motion": str(args.contact_motion),
        "contact_key": args.contact_key,
        "contact_frames": int(len(contact_indices)),
        "ramp_frames": args.ramp_frames,
        "episodes": episodes,
        "peak_xy_translation_m": float(np.max(np.linalg.norm(ramped_delta, axis=1))),
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
