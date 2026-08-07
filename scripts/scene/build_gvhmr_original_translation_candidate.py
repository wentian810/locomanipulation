#!/usr/bin/env python3
"""Stage a provenance-recorded GVHMR translation-only postprocess candidate.

The candidate keeps the smoothed pose, shape, rate, and metadata intact.  It
only restores the original GVHMR translation already preserved in the input
NPZ as ``trans_original``.  It is intentionally generic: every selected clip
is handled identically and its source digest is recorded for downstream gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.stem}.", suffix=".npz", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **values)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--clips", required=True, help="Comma-separated clip names")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    clips = [item.strip() for item in args.clips.split(",") if item.strip()]
    if not clips:
        raise ValueError("--clips must name at least one clip")
    report: dict[str, object] = {
        "schema_version": 1,
        "kind": "gvhmr_original_translation_only_postprocess",
        "weights_modified": False,
        "pose_modified": False,
        "translation_source": "trans_original",
        "clips": {},
    }
    for clip in clips:
        source = args.input_root / clip / "001_smoothed.npz"
        target = args.output_root / clip / "001_smoothed.npz"
        if not source.is_file():
            raise FileNotFoundError(source)
        if target.exists() and not args.overwrite:
            raise FileExistsError(target)
        with np.load(source, allow_pickle=False) as archive:
            values = {key: np.asarray(archive[key]) for key in archive.files}
        required = {"trans", "trans_original", "root_orient", "pose_body", "betas"}
        missing = required - values.keys()
        if missing:
            raise KeyError(f"{source} missing {sorted(missing)}")
        current = np.asarray(values["trans"], dtype=np.float32)
        original = np.asarray(values["trans_original"], dtype=np.float32)
        if current.shape != original.shape or current.ndim != 2 or current.shape[1] != 3:
            raise ValueError(f"incompatible translation arrays in {source}")
        if values["root_orient"].shape[0] != len(current) or values["pose_body"].shape[0] != len(current):
            raise ValueError(f"motion arrays have inconsistent frame counts in {source}")
        values["trans"] = original
        atomic_npz(target, values)
        delta = np.linalg.norm(current - original, axis=1)
        report["clips"][clip] = {
            "source": str(source),
            "source_sha256": sha256(source),
            "output": str(target),
            "output_sha256": sha256(target),
            "frame_count": int(len(current)),
            "translation_delta_m": {
                "median": float(np.median(delta)),
                "p95": float(np.percentile(delta, 95)),
                "maximum": float(np.max(delta)),
            },
        }
    report_path = args.output_root / "gvhmr_original_translation_postprocess_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
