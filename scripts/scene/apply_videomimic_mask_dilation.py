"""Create a derived, conservatively dilated VideoMimic human-mask sequence.

The original SAM2 masks are copied once beside ``mask_data`` before the derived
masks replace the active inputs consumed by MegaSAM.  This prevents human-depth
spill from entering the static point cloud while retaining reproducible image
evidence for later inspection.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--extra-radius-px", required=True, type=int)
    parser.add_argument("--report-json", required=True, type=Path)
    args = parser.parse_args()
    if args.extra_radius_px < 1:
        raise ValueError("--extra-radius-px must be positive")

    mask_dir = args.mask_dir.resolve()
    report_path = args.report_json.resolve()
    masks = sorted(mask_dir.glob("mask_*.npz"))
    if not masks:
        raise FileNotFoundError(f"no VideoMimic masks in {mask_dir}")
    backup_dir = mask_dir.parent / f"{mask_dir.name}_sam2_original"
    if backup_dir.exists():
        raise FileExistsError(f"refusing to overwrite SAM2 mask backup: {backup_dir}")
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite mask-dilation report: {report_path}")

    shutil.copytree(mask_dir, backup_dir)
    kernel_size = 2 * args.extra_radius_px + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    original_coverages: list[float] = []
    dilated_coverages: list[float] = []
    shapes: set[tuple[int, int]] = set()
    for path in masks:
        with np.load(path, allow_pickle=False) as archive:
            payload = {key: archive[key] for key in archive.files}
        if "mask" not in payload:
            raise KeyError(f"mask archive is missing 'mask': {path}")
        mask = np.asarray(payload["mask"])
        if mask.ndim != 2:
            raise ValueError(f"mask must be two-dimensional: {path} has {mask.shape}")
        labels = np.unique(mask[mask > 0])
        if labels.size != 1:
            raise ValueError(f"mask must contain exactly one foreground label: {path}")
        original = mask > 0
        dilated = cv2.dilate(original.astype(np.uint8), kernel, iterations=1) > 0
        payload["mask"] = np.where(dilated, labels[0], 0).astype(mask.dtype)
        temporary = path.with_name(f"{path.stem}.dilation_tmp.npz")
        if temporary.exists():
            raise FileExistsError(f"temporary mask path already exists: {temporary}")
        np.savez_compressed(temporary, **payload)
        temporary.replace(path)
        original_coverages.append(float(original.mean()))
        dilated_coverages.append(float(dilated.mean()))
        shapes.add((int(mask.shape[0]), int(mask.shape[1])))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "pass",
                "mask_dir": str(mask_dir),
                "sam2_original_mask_dir": str(backup_dir),
                "mask_count": len(masks),
                "mask_shapes": sorted([list(shape) for shape in shapes]),
                "extra_radius_px": args.extra_radius_px,
                "kernel_size_px": kernel_size,
                "original_coverage": _percentiles(original_coverages),
                "dilated_coverage": _percentiles(dilated_coverages),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[PASS] dilated {len(masks)} SAM2 masks by radius={args.extra_radius_px}px")


if __name__ == "__main__":
    main()
