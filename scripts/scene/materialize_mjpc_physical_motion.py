#!/usr/bin/env python3
"""Materialize a continuous MJPC roll-out as a GMR-compatible motion file.

The input CSV is read-only evidence from the runner: its robot qpos values
were produced by bounded actuators and ``mj_step``.  This tool copies the
reference metadata/contact labels, replaces only the robot state arrays, and
records checksums so a downstream static-scene fit cannot silently consume a
kinematic GMR reference instead of the physical trajectory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_trajectory(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.reader(line for line in handle if not line.startswith("#"))]
    if len(rows) < 2:
        raise ValueError(f"trajectory has no state rows: {path}")
    header = rows[0]
    qpos_columns = [name for name in header if name.startswith("qpos_")]
    if qpos_columns != [f"qpos_{index}" for index in range(len(qpos_columns))]:
        raise ValueError("trajectory qpos columns are missing or reordered")
    values = np.asarray(rows[1:], dtype=np.float64)
    if values.shape[1] != len(header):
        raise ValueError("trajectory rows have inconsistent width")
    qpos = values[:, 1 : 1 + len(qpos_columns)]
    if qpos.shape[1] != 36:
        raise ValueError(f"expected a floating-base 29-DoF G1 qpos of length 36, got {qpos.shape[1]}")
    if not np.isfinite(qpos).all():
        raise ValueError("trajectory qpos contains NaN or Inf")
    return qpos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--reference-motion", required=True, type=Path)
    parser.add_argument("--output-motion", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    qpos = _load_trajectory(args.trajectory)
    with args.reference_motion.open("rb") as handle:
        reference: dict[str, Any] = pickle.load(handle)
    expected_frames = len(np.asarray(reference["root_pos"], dtype=np.float64))
    if len(qpos) != expected_frames:
        raise ValueError(
            f"trajectory frame count {len(qpos)} does not match reference motion {expected_frames}; "
            "a partial roll-out cannot be materialized as a full physical reference"
        )

    motion = dict(reference)
    motion["root_pos"] = qpos[:, :3].copy()
    # MuJoCo free-joint qpos stores WXYZ; the GMR contract stores XYZW.
    motion["root_rot"] = qpos[:, [4, 5, 6, 3]].copy()
    motion["dof_pos"] = qpos[:, 7:].copy()
    motion["physical_motion_provenance"] = {
        "schema_version": 1,
        "producer": "materialize_mjpc_physical_motion",
        "trajectory": str(args.trajectory),
        "trajectory_sha256": _sha256(args.trajectory),
        "reference_motion": str(args.reference_motion),
        "reference_motion_sha256": _sha256(args.reference_motion),
        "state_contract": "bounded_actuator_then_mj_step",
        "root_rotation_storage": "xyzw",
    }

    args.output_motion.parent.mkdir(parents=True, exist_ok=True)
    with args.output_motion.open("wb") as handle:
        pickle.dump(motion, handle, protocol=pickle.HIGHEST_PROTOCOL)
    report = {
        "schema_version": 1,
        "purpose": "materialize_continuous_mjpc_rollout_for_static_scene_fit",
        "status": "ready",
        "physical_contract": "bounded_actuator_then_mj_step",
        "frame_count": int(len(qpos)),
        "trajectory": str(args.trajectory),
        "trajectory_sha256": _sha256(args.trajectory),
        "reference_motion": str(args.reference_motion),
        "reference_motion_sha256": _sha256(args.reference_motion),
        "output_motion": str(args.output_motion),
        "output_motion_sha256": _sha256(args.output_motion),
        "root_rotation_output_order": "xyzw",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
