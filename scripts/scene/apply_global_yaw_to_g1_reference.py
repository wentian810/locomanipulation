#!/usr/bin/env python3
"""Apply one rigid world-yaw correction to a MuJoCo G1 qpos reference.

This is for testing an inter-model root-axis convention.  It never modifies
joint angles, root translations, timing, or individual frames; every root
quaternion receives the same left-multiplied world rotation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product for one wxyz quaternion and an array of wxyz quaternions."""
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right.T
    return np.column_stack((
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--yaw-degrees", type=float, required=True)
    args = parser.parse_args()
    if args.output_csv.exists() or args.manifest_json.exists():
        raise FileExistsError("refusing to overwrite an existing alignment diagnostic")
    qpos = np.loadtxt(args.input_csv, delimiter=",")
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"expected [T,36] qpos data, got {qpos.shape}")
    half_angle = np.deg2rad(args.yaw_degrees) / 2.0
    yaw_wxyz = np.array([np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)])
    corrected = qpos.copy()
    corrected[:, 3:7] = multiply_wxyz(yaw_wxyz, qpos[:, 3:7])
    corrected[:, 3:7] /= np.linalg.norm(corrected[:, 3:7], axis=1, keepdims=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output_csv, corrected, delimiter=",")
    manifest = {
        "schema_version": 1,
        "purpose": "global_root_yaw_coordinate_contract_diagnostic",
        "input_csv": str(args.input_csv),
        "output_csv": str(args.output_csv),
        "frame_count": int(qpos.shape[0]),
        "yaw_degrees": float(args.yaw_degrees),
        "operation": "q_world_corrected = q_yaw_world * q_original",
        "unchanged": ["root_translation", "29_joint_angles", "frame_timing"],
        "promotion_forbidden_until_static_chair_contact_audit_passes": True,
    }
    args.manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
