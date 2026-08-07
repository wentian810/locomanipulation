"""Validate a generated VideoMimic adapter package without running inference."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def _fail(message: str) -> None:
    raise ValueError(message)


def validate(root: Path) -> dict:
    report_path = root / "adapter_report.json"
    if not report_path.is_file():
        _fail(f"missing adapter report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("status") != "pass":
        _fail("adapter report is not a passing schema-v1 report")
    count = int(report.get("frame_count", 0))
    if count <= 0:
        _fail("adapter report has no frames")
    outputs = report.get("outputs", {})
    required = ("frames", "masks", "pose2d", "smpl", "frame_map")
    if any(name not in outputs for name in required):
        _fail("adapter report lacks an output path")
    frame_map = np.load(outputs["frame_map"], allow_pickle=False)
    videomimic_index = np.asarray(frame_map["videomimic_frame_index"])
    if videomimic_index.shape != (count,) or not np.array_equal(videomimic_index, np.arange(count)):
        _fail("invalid frame map: videomimic_frame_index")
    source_count = int(report.get("source_frame_count", count))
    source_indices: list[np.ndarray] = []
    for key in ("source_video_frame_index", "work_video_frame_index", "human_frame_index"):
        value = np.asarray(frame_map[key])
        if value.shape != (count,) or not np.issubdtype(value.dtype, np.integer):
            _fail(f"invalid frame map: {key}")
        if value.min() < 0 or value.max() >= source_count:
            _fail(f"out-of-range frame map: {key}")
        if count > 1 and not np.all(np.diff(value) > 0):
            _fail(f"non-monotonic frame map: {key}")
        source_indices.append(value)
    if any(not np.array_equal(source_indices[0], value) for value in source_indices[1:]):
        _fail("source, work and human frame maps disagree")
    frames = sorted(Path(outputs["frames"]).glob("frame_*.jpg"))
    masks = sorted((Path(outputs["masks"]) / "mask_data").glob("mask_*.npz"))
    mask_json = sorted((Path(outputs["masks"]) / "json_data").glob("mask_*.json"))
    poses = sorted(Path(outputs["pose2d"]).glob("pose_*.json"))
    smpls = sorted(Path(outputs["smpl"]).glob("smpl_params_*.pkl"))
    if [len(frames), len(masks), len(mask_json), len(poses), len(smpls)] != [count] * 5:
        _fail("adapter outputs have inconsistent frame counts")
    person_id = int(report["person_id"])
    if report.get("mask_source") == "videomimic_official_sam2_person_mask":
        for index in range(count):
            meta = json.loads(mask_json[index].read_text(encoding="utf-8"))
            label = meta.get("labels", {}).get(str(person_id))
            if not isinstance(label, dict):
                _fail(f"SAM2 mask has no label for external person {person_id} at frame {index}")
            mask = np.load(masks[index], allow_pickle=False)["mask"]
            if not np.any(mask == person_id):
                _fail(f"SAM2 mask has no pixels for external person {person_id} at frame {index}")
            pose = json.loads(poses[index].read_text(encoding="utf-8"))[str(person_id)]["bbox"]
            ref = np.asarray(pose[:4], dtype=np.float64)
            sam = np.asarray([label["x1"], label["y1"], label["x2"], label["y2"]], dtype=np.float64)
            left, top = np.maximum(ref[:2], sam[:2])
            inter = max(0.0, min(ref[2], sam[2]) - left) * max(0.0, min(ref[3], sam[3]) - top)
            union = (ref[2] - ref[0]) * (ref[3] - ref[1]) + (sam[2] - sam[0]) * (sam[3] - sam[1]) - inter
            if inter / max(union, 1.0) < 0.35:
                _fail(f"SAM2/GVHMR person association IoU is below 0.35 at frame {index}")
            if float((mask == person_id).sum()) / max((sam[2] - sam[0]) * (sam[3] - sam[1]), 1.0) < 0.04:
                _fail(f"SAM2 person mask is implausibly sparse at frame {index}")
    for index in (0, count - 1):
        with smpls[index].open("rb") as handle:
            data = pickle.load(handle)
        params = data[person_id]["smpl_params"]
        if np.asarray(params["global_orient"]).shape != (1, 3, 3):
            _fail("invalid global_orient shape")
        if np.asarray(params["body_pose"]).shape != (23, 3, 3):
            _fail("invalid body_pose shape")
        if np.asarray(params["betas"]).shape != (10,):
            _fail("invalid beta shape")
        rotation = np.asarray(params["global_orient"])[0]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
            _fail("global orientation is not a rotation matrix")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = validate(args.output)
    except (OSError, ValueError, KeyError, pickle.UnpicklingError) as exc:
        parser.error(str(exc))
    print(f"[PASS] adapter validation: {report['frame_count']} aligned frames")


if __name__ == "__main__":
    main()
