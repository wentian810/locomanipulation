"""Audit an isolated terminal SAM2/GVHMR disagreement without losing evidence.

The SAM2 mask is image evidence; a GVHMR 2-D box is only an auxiliary pose
association signal.  When they disagree in one eligible terminal frame, the
scene path must preserve the SAM2 mask and its label rather than replacing or
enlarging it with the GVHMR box.  Otherwise a visible static object (for
example a chair revealed as the person leaves) can be erased from the only
useful frame.  The disagreement is recorded for downstream review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    lower = np.maximum(first[:2], second[:2])
    upper = np.minimum(first[2:], second[2:])
    intersection = float(np.prod(np.maximum(upper - lower, 0.0)))
    union = float(np.prod(first[2:] - first[:2]) + np.prod(second[2:] - second[:2]) - intersection)
    return intersection / max(union, 1.0)


def _label_box(label: dict[str, object]) -> np.ndarray:
    return np.asarray([label[key] for key in ("x1", "y1", "x2", "y2")], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-iou", type=float, default=0.35)
    parser.add_argument("--max-terminal-frames", type=int, default=1)
    args = parser.parse_args()
    if not 0.0 < args.min_iou <= 1.0 or args.max_terminal_frames < 1:
        raise ValueError("invalid terminal SAM2 repair thresholds")

    root = args.output.resolve()
    report = json.loads((root / "adapter_report.json").read_text(encoding="utf-8"))
    if report.get("mask_source") != "videomimic_official_sam2_person_mask":
        print("[SKIP] terminal SAM2 repair is not applicable to this mask source")
        return
    count = int(report["frame_count"])
    person_id = str(int(report["person_id"]))
    outputs = report["outputs"]
    mask_root = Path(outputs["masks"])
    pose_root = Path(outputs["pose2d"])
    entries: list[dict[str, object]] = []
    bad: list[int] = []
    for frame in range(count):
        json_path = mask_root / "json_data" / f"mask_{frame:05d}.json"
        pose_path = pose_root / f"pose_{frame:05d}.json"
        label = json.loads(json_path.read_text(encoding="utf-8"))["labels"][person_id]
        pose = json.loads(pose_path.read_text(encoding="utf-8"))[person_id]
        iou = _iou(_label_box(label), np.asarray(pose["bbox"][:4], dtype=np.float64))
        entries.append(
            {
                "frame": frame,
                "iou": iou,
                "mask_json": json_path,
                "sam2_bbox": _label_box(label).tolist(),
                "pose_bbox": pose["bbox"][:4],
            }
        )
        if iou < args.min_iou:
            bad.append(frame)

    repair_report = root / "terminal_sam2_mask_repair.json"
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "not_needed" if not bad else "rejected",
        "min_iou": args.min_iou,
        "max_terminal_frames": args.max_terminal_frames,
        "frame_count": count,
        "bad_frames": bad,
        "iou_summary": {
            "min": float(min(entry["iou"] for entry in entries)),
            "p01": float(np.quantile([entry["iou"] for entry in entries], 0.01)),
            "p50": float(np.quantile([entry["iou"] for entry in entries], 0.50)),
        },
    }
    if not bad:
        repair_report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("[PASS] terminal SAM2 repair not needed")
        return

    first_terminal = count - args.max_terminal_frames
    if (
        len(bad) > args.max_terminal_frames
        or any(frame < first_terminal for frame in bad)
        or (bad[0] > 0 and entries[bad[0] - 1]["iou"] < args.min_iou)
    ):
        repair_report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError("SAM2 association failures are not an isolated terminal event")

    preserved: list[dict[str, object]] = []
    for frame in bad:
        entry = entries[frame]
        mask_path = mask_root / "mask_data" / f"mask_{frame:05d}.npz"
        with np.load(mask_path, allow_pickle=False) as archive:
            sam2_mask = np.asarray(archive["mask"]) > 0
        if not np.any(sam2_mask):
            raise RuntimeError(f"terminal SAM2 mask is empty at frame {frame}")
        preserved.append(
            {
                "frame": frame,
                "original_iou": entry["iou"],
                "mask_policy": "preserve_sam2_mask_and_label",
                "sam2_coverage": float(sam2_mask.mean()),
                "sam2_bbox": entry["sam2_bbox"],
                "gvhmr_pose_bbox": list(np.asarray(entry["pose_bbox"], dtype=float)),
            }
        )

    payload["status"] = "preserved_terminal_sam2_over_gvhmr_bbox"
    payload["preserved_frames"] = preserved
    repair_report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[PASS] preserved isolated terminal SAM2 masks: {bad}")


if __name__ == "__main__":
    main()
