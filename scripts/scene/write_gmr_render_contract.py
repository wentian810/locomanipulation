#!/usr/bin/env python3
"""Bind one rendered GMR video to its audited robot-motion input."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values: np.ndarray) -> dict[str, float]:
    if not len(values):
        return {"p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "p50": float(np.quantile(values, 0.5)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def root_audit(motion_path: Path, max_step_m: float) -> dict[str, object]:
    with motion_path.open("rb") as handle:
        motion = pickle.load(handle)
    root = np.asarray(motion["root_pos"], dtype=np.float64)
    rotation = np.asarray(motion["root_rot"], dtype=np.float64)
    if root.ndim != 2 or root.shape[1] != 3 or rotation.shape != (len(root), 4):
        raise ValueError("robot_motion root_pos/root_rot arrays are invalid")
    steps = np.diff(root, axis=0)
    step_length = np.linalg.norm(steps, axis=1)
    vertical_step = np.abs(steps[:, 2])
    normalized_rotation = rotation / np.maximum(
        np.linalg.norm(rotation, axis=1, keepdims=True), 1e-12
    )
    rotation_step = 2.0 * np.arccos(np.clip(
        np.abs(np.sum(normalized_rotation[1:] * normalized_rotation[:-1], axis=1)), -1.0, 1.0
    ))
    height = root[:, 2]
    rejected_frames = (np.flatnonzero(step_length > max_step_m) + 1).tolist()
    return {
        "motion_sha256": sha256(motion_path),
        "frames": int(len(root)),
        "fps": float(np.asarray(motion.get("fps", 30.0)).reshape(-1)[0]),
        "max_root_step_threshold_m": max_step_m,
        "root_step_m": summary(step_length),
        "root_vertical_step_m": summary(vertical_step),
        "root_rotation_step_rad": summary(rotation_step),
        "root_height_m": {
            "start": float(height[0]), "end": float(height[-1]),
            "min": float(np.min(height)), "min_frame": int(np.argmin(height)),
            "max": float(np.max(height)), "max_frame": int(np.argmax(height)),
        },
        "status": "pass" if not rejected_frames else "reject_root_jump",
        "rejected_frames": rejected_frames,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "final"), required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-root-step-m", type=float, default=0.05)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument(
        "--run-manifest",
        type=Path,
        help="Optional pipeline manifest updated only after a bound render succeeds.",
    )
    parser.add_argument("--render-mode")
    parser.add_argument("--camera-source")
    return parser.parse_args()


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if not args.motion.is_file():
        raise FileNotFoundError(args.motion)
    if args.max_root_step_m <= 0.0:
        raise ValueError("--max-root-step-m must be positive")
    audit = root_audit(args.motion, args.max_root_step_m)
    if audit["status"] != "pass":
        write_json(args.output, {"phase": args.phase, "motion": str(args.motion.resolve()), "audit": audit})
        raise SystemExit("refusing GMR render contract: root continuity gate failed")

    contract: dict[str, object] = {
        "schema_version": 1,
        "phase": args.phase,
        "motion": str(args.motion.resolve()),
        "motion_sha256": audit["motion_sha256"],
        "root_audit": audit,
    }
    if args.phase == "preflight":
        write_json(args.output, contract)
        return

    if args.video is None or not args.video.is_file():
        raise FileNotFoundError("final contract requires an existing --video")
    if args.preflight is None or not args.preflight.is_file():
        raise FileNotFoundError("final contract requires an existing --preflight")
    previous = json.loads(args.preflight.read_text(encoding="utf-8"))
    if previous.get("motion_sha256") != audit["motion_sha256"]:
        raise RuntimeError("motion changed after render preflight; refuse to bind stale video")
    contract.update({
        "video": str(args.video.resolve()),
        "video_sha256": sha256(args.video),
        "render": {"mode": args.render_mode, "camera_source": args.camera_source},
        "preflight": str(args.preflight.resolve()),
        "preflight_sha256": sha256(args.preflight),
        "eligible_for_composite": True,
    })
    write_json(args.output, contract)
    if args.run_manifest is not None:
        if not args.run_manifest.is_file():
            raise FileNotFoundError(args.run_manifest)
        manifest = json.loads(args.run_manifest.read_text(encoding="utf-8"))
        manifest.update({
            "robot_motion_sha256": audit["motion_sha256"],
            "gmr_video": str(args.video.resolve()),
            "gmr_video_sha256": contract["video_sha256"],
            "gmr_render_contract": str(args.output.resolve()),
            "gmr_render_contract_sha256": sha256(args.output),
        })
        write_json(args.run_manifest, manifest)


if __name__ == "__main__":
    main()
