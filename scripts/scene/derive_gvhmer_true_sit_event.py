#!/usr/bin/env python3
"""Gate chair-contact control from a settled GVHMR sitting event.

``sit_mask`` and an SMPL-to-seat distance alone are intentionally insufficient:
they also hold while a person is descending into a chair.  This utility keeps
the scene unchanged and derives a conservative event from the GVHMR SMPL-H
motion, its semantic-seat anchors, and the temporal settling of the torso.

The resulting JSON is an input contract for a physical controller.  A failed
gate means that the controller must *not* invent a seated transition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import smplx
import torch


Z_UP_FROM_NEG_Y = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _z_up_from_neg_y(points: np.ndarray) -> np.ndarray:
    """Convert the GVHMR neg-Y-up world convention to the chair's Z-up frame."""
    return np.asarray(points, dtype=np.float64) @ Z_UP_FROM_NEG_Y.T


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    edges = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _containing_run(runs: list[tuple[int, int]], frame: int) -> tuple[int, int]:
    for start, end in runs:
        if start <= frame <= end:
            return start, end
    raise ValueError("frame %d is not covered by the supplied runs" % frame)


def _forward_span(values: np.ndarray, valid: np.ndarray, window: int) -> np.ndarray:
    """Maximum-minus-minimum over a future confirmation window.

    A forward window prevents a transient during descent from being called
    settled merely because preceding frames happened to be quiet.
    """
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    for frame in range(values.shape[0]):
        stop = min(frame + window, values.shape[0])
        if stop - frame < window or not np.all(valid[frame:stop]):
            continue
        sample = values[frame:stop]
        if np.isfinite(sample).all():
            result[frame] = float(np.max(sample) - np.min(sample))
    return result


def _load_smpl_torso(
    motion_path: Path,
    body_model_root: Path,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return neck height and pelvis-to-neck tilt in the shared Z-up frame."""
    motion = np.load(motion_path, allow_pickle=False)
    required = {"trans", "root_orient", "pose_body", "betas"}
    missing = required.difference(motion.files)
    if missing:
        raise KeyError("human motion missing keys: %s" % sorted(missing))
    frame_count = len(motion["trans"])
    requested_gender = (
        str(motion["gender"].item()).lower() if "gender" in motion.files else "neutral"
    )
    gender = requested_gender
    expected_model = body_model_root / "smplh" / ("SMPLH_%s.npz" % gender.upper())
    if not expected_model.is_file():
        gender = "neutral"
    model = smplx.create(
        str(body_model_root),
        model_type="smplh",
        gender=gender,
        ext="npz",
        num_betas=len(motion["betas"]),
        use_pca=False,
        batch_size=frame_count,
    ).to(device)
    betas = np.repeat(np.asarray(motion["betas"], dtype=np.float32)[None], frame_count, axis=0)
    with torch.no_grad():
        output = model(
            betas=torch.as_tensor(betas, dtype=torch.float32, device=device),
            global_orient=torch.as_tensor(motion["root_orient"], dtype=torch.float32, device=device),
            body_pose=torch.as_tensor(motion["pose_body"], dtype=torch.float32, device=device),
            transl=torch.as_tensor(motion["trans"], dtype=torch.float32, device=device),
            return_verts=False,
        )
    joints = _z_up_from_neg_y(output.joints.detach().cpu().numpy())
    # SMPL-H body joints use pelvis=0 and neck=12.  Do not use the head: head
    # motion would cause a false transition during a perfectly valid seated pose.
    if joints.shape[1] <= 12:
        raise ValueError("SMPL-H output has no neck joint at index 12")
    pelvis = joints[:, 0]
    neck = joints[:, 12]
    torso = neck - pelvis
    torso_norm = np.linalg.norm(torso, axis=1)
    if np.any(torso_norm < 1e-8):
        raise ValueError("degenerate pelvis-to-neck vector in GVHMR motion")
    tilt_rad = np.arccos(np.clip(torso[:, 2] / torso_norm, -1.0, 1.0))
    return neck[:, 2], tilt_rad


def _video_metadata(path: Path | None) -> dict | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(path)
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,nb_frames,duration,width,height",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    return {
        "path": str(path),
        "sha256": _file_sha256(path),
        "fps": stream.get("avg_frame_rate"),
        "frame_count": stream.get("nb_frames"),
        "duration_s": stream.get("duration"),
        "width": stream.get("width"),
        "height": stream.get("height"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--human-motion", type=Path, required=True)
    parser.add_argument("--body-model-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-video", type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--settle-window-frames", type=int, default=18)
    parser.add_argument("--min-run-frames", type=int, default=12)
    parser.add_argument("--max-seat-gap-m", type=float, default=0.012)
    parser.add_argument("--max-seat-height-span-m", type=float, default=0.006)
    parser.add_argument("--max-neck-height-span-m", type=float, default=0.020)
    parser.add_argument("--max-torso-tilt-span-deg", type=float, default=8.0)
    args = parser.parse_args()
    if args.settle_window_frames < 2 or args.min_run_frames < 1:
        raise ValueError("settling window must be >=2 and minimum run must be >=1")

    anchors = np.load(args.anchors, allow_pickle=False)
    required = {
        "frame_index",
        "sit_mask",
        "seat_anchor_z_up",
        "seat_signed_distance_m",
        "seat_contact",
    }
    missing = required.difference(anchors.files)
    if missing:
        raise KeyError("anchors missing keys: %s" % sorted(missing))
    frame_index = np.asarray(anchors["frame_index"], dtype=np.int32)
    sit_mask = np.asarray(anchors["sit_mask"], dtype=bool)
    seat_contact = np.asarray(anchors["seat_contact"], dtype=bool)
    seat_height = np.asarray(anchors["seat_anchor_z_up"], dtype=np.float64)[:, 2]
    seat_gap = np.asarray(anchors["seat_signed_distance_m"], dtype=np.float64)
    neck_height, torso_tilt = _load_smpl_torso(
        args.human_motion, args.body_model_root, args.device
    )
    if not (len(frame_index) == len(neck_height) == len(sit_mask)):
        raise ValueError(
            "anchor and GVHMR motion frame counts disagree: %d versus %d"
            % (len(frame_index), len(neck_height))
        )

    near_surface = np.isfinite(seat_gap) & (np.abs(seat_gap) <= args.max_seat_gap_m)
    contact_candidate = sit_mask & seat_contact & near_surface & np.isfinite(seat_height)
    seat_span = _forward_span(seat_height, contact_candidate, args.settle_window_frames)
    neck_span = _forward_span(neck_height, contact_candidate, args.settle_window_frames)
    tilt_span = _forward_span(torso_tilt, contact_candidate, args.settle_window_frames)
    settled = (
        contact_candidate
        & (seat_span <= args.max_seat_height_span_m)
        & (neck_span <= args.max_neck_height_span_m)
        & (tilt_span <= np.deg2rad(args.max_torso_tilt_span_deg))
    )
    sit_runs = _runs(sit_mask)
    candidate_runs = _runs(contact_candidate)
    all_settled_runs = _runs(settled)
    accepted_runs = [
        (start, end)
        for start, end in all_settled_runs
        if end - start + 1 >= args.min_run_frames
    ]
    events = []
    for start, end in accepted_runs:
        # A run touching frame zero is a supported initial state, not proof of a
        # sit-down transition within this clip.  It remains useful for static
        # seated replay but must never trigger a manufactured descent.
        transition_observed = start > 0 and bool(np.any(~seat_contact[:start]))
        candidate_start, candidate_end = _containing_run(candidate_runs, start)
        sit_start, sit_end = _containing_run(sit_runs, start)
        events.append(
            {
                "frame_range": [int(start), int(end)],
                "seat_approach_reference_range": [int(sit_start), int(sit_end)],
                "near_surface_contact_reference_range": [
                    int(candidate_start),
                    int(candidate_end),
                ],
                "physical_support_validation_range": [
                    int(start),
                    int(candidate_end),
                ],
                "settlement_confirmed_at_frame": int(
                    start + args.settle_window_frames - 1
                ),
                "duration_s": float((end - start + 1) / args.fps),
                "transition_observed_in_clip": transition_observed,
                "role": "physical_sit_transition" if transition_observed else "initial_or_continuing_seated_state",
                "seat_gap_median_mm": float(np.median(seat_gap[start : end + 1]) * 1000.0),
                "seat_gap_p95_mm": float(np.percentile(seat_gap[start : end + 1], 95) * 1000.0),
                "neck_height_span_mm": float(
                    np.nanmax(neck_height[start : end + 1])
                    - np.nanmin(neck_height[start : end + 1])
                )
                * 1000.0,
                "torso_tilt_span_deg": float(
                    np.rad2deg(
                        np.nanmax(torso_tilt[start : end + 1])
                        - np.nanmin(torso_tilt[start : end + 1])
                    )
                ),
            }
        )
    physical_events = [event for event in events if event["transition_observed_in_clip"]]
    report = {
        "schema_version": 1,
        "decision": "pass" if physical_events else "no_physical_sit_transition",
        "purpose": "gate physical chair-contact control from GVHMR evidence",
        "inputs": {
            "anchors": str(args.anchors),
            "anchors_sha256": _file_sha256(args.anchors),
            "human_motion": str(args.human_motion),
            "human_motion_sha256": _file_sha256(args.human_motion),
            "source_video": _video_metadata(args.source_video),
        },
        "coordinate_contract": {
            "human_motion_input": "GVHMR neg-Y-up",
            "anchors": "semantic chair mujoco_world_z_up",
            "conversion": "[x, y, z] -> [x, -z, y]",
        },
        "criteria": {
            "requires": [
                "GVHMR pelvis is in the semantic-seat footprint",
                "GVHMR lower body is within the signed seat contact band",
                "seat anchor remains settled over a future confirmation window",
                "GVHMR neck height and pelvis-to-neck tilt are settled over the same window",
            ],
            "settle_window_frames": args.settle_window_frames,
            "min_run_frames": args.min_run_frames,
            "max_seat_gap_m": args.max_seat_gap_m,
            "max_seat_height_span_m": args.max_seat_height_span_m,
            "max_neck_height_span_m": args.max_neck_height_span_m,
            "max_torso_tilt_span_deg": args.max_torso_tilt_span_deg,
        },
        "candidate_runs": [[start, end] for start, end in candidate_runs],
        "settled_runs_before_minimum_length": [
            [start, end] for start, end in all_settled_runs
        ],
        "events": events,
        "physical_transition_events": physical_events,
        "controller_contract": {
            "when_passed": "Use the GMR trajectory only as a reference to a free-base, torque-limited policy stepped with mj_step; chair collision remains active.",
            "when_not_passed": "Do not add a seated transition, root wrench, qpos reset, or synthetic chair contact.",
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
