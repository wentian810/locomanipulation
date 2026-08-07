#!/usr/bin/env python3
"""Build one provenance-checked GMR sitting-event anchor archive.

The seat event comes only from VideoMimic's gravity-aligned human-scene
contact evidence.  Older GMR support masks may provide backrest evidence, but
can never widen the sitting interval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _event_summary(mask: np.ndarray) -> dict[str, object]:
    frames = np.flatnonzero(mask)
    if len(frames) == 0:
        return {"count": 0, "frame_range": None, "contiguous": False}
    return {
        "count": int(len(frames)),
        "frame_range": [int(frames[0]), int(frames[-1])],
        "contiguous": bool(np.all(np.diff(frames) == 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-contacts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--sit-key", default="pelvis_contact")
    parser.add_argument("--fallback-anchors", type=Path)
    parser.add_argument("--back-key", default="back_contact")
    args = parser.parse_args()

    scene = np.load(args.scene_contacts)
    if args.sit_key not in scene:
        raise ValueError(f"scene contact archive has no {args.sit_key!r}; refusing manual sit event")
    sit_mask = np.asarray(scene[args.sit_key], dtype=bool)
    if sit_mask.ndim != 1:
        raise ValueError("sitting event must be one-dimensional")
    sit_summary = _event_summary(sit_mask)
    if sit_summary["count"] < 4 or not sit_summary["contiguous"]:
        raise ValueError("VideoMimic sitting event is insufficient or non-contiguous")

    back_mask = np.zeros_like(sit_mask)
    fallback_metadata: dict[str, object] | None = None
    if args.fallback_anchors is not None:
        fallback = np.load(args.fallback_anchors)
        if args.back_key not in fallback:
            raise ValueError(f"fallback archive has no {args.back_key!r}")
        candidate_back = np.asarray(fallback[args.back_key], dtype=bool)
        if candidate_back.shape != sit_mask.shape:
            raise ValueError("fallback back-contact length differs from VideoMimic event")
        back_mask = candidate_back & sit_mask
        fallback_metadata = {
            "path": str(args.fallback_anchors), "sha256": _sha256(args.fallback_anchors),
            "back_key": args.back_key,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, sit_mask=sit_mask, back_contact=back_mask)
    report = {
        "schema_version": 1,
        "purpose": "authoritative_gmr_sitting_event_anchors",
        "status": "ready",
        "policy": {
            "sit_mask": "VideoMimic human-scene pelvis contact only",
            "back_contact": "optional fallback evidence intersected with authoritative sit_mask",
            "manual_frame_override": "forbidden",
        },
        "scene_contacts": {"path": str(args.scene_contacts), "sha256": _sha256(args.scene_contacts), "sit_key": args.sit_key},
        "fallback_back_contact": fallback_metadata,
        "sit_event": sit_summary,
        "back_event": _event_summary(back_mask),
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
