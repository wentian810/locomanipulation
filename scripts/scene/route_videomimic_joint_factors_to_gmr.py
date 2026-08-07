#!/usr/bin/env python3
"""Build an auditable GMR candidate from VideoMimic's mapped joint factors.

The visual GMR track owns root translation and orientation. VideoMimic owns only
the named joint factors it actually optimizes. Unrepresented joints may be
retained only when they are whitelisted non-load-bearing wrist joints. This is a
factor router with an explicit contract, never an NPZ splice.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import h5py
import numpy as np


DEFAULT_ALLOWED_UNMAPPED = {
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _names(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-motion", required=True, type=Path)
    parser.add_argument("--videomimic-h5", required=True, type=Path)
    parser.add_argument("--output-motion", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--person-id", default=None)
    parser.add_argument(
        "--allow-unmapped", default=",".join(sorted(DEFAULT_ALLOWED_UNMAPPED)),
        help="Only named factors allowed to remain from reference motion.",
    )
    args = parser.parse_args()

    with args.reference_motion.open("rb") as f:
        reference = pickle.load(f)
    root_pos = np.asarray(reference.get("root_pos"), dtype=np.float64)
    root_rot = np.asarray(reference.get("root_rot"), dtype=np.float64)
    dof_pos = np.asarray(reference.get("dof_pos"), dtype=np.float64)
    dof_names = [str(name) for name in reference.get("dof_names", [])]
    if root_pos.ndim != 2 or root_pos.shape[1] != 3 or root_rot.shape != (len(root_pos), 4):
        raise ValueError("reference motion must provide root_pos [T,3] and root_rot [T,4]")
    if dof_pos.shape != (len(root_pos), len(dof_names)):
        raise ValueError("reference dof_pos/dof_names shape mismatch")

    with h5py.File(args.videomimic_h5, "r") as source:
        optimized_node = source["joints"]
        if isinstance(optimized_node, h5py.Group):
            person_ids = list(optimized_node.keys())
            if not person_ids:
                raise ValueError("VideoMimic file contains no people")
            person_id = args.person_id or person_ids[0]
            if person_id not in optimized_node:
                raise ValueError(f"person-id {person_id!r} unavailable; found {person_ids}")
            optimized = np.asarray(optimized_node[person_id], dtype=np.float64)
        else:
            if args.person_id is not None:
                raise ValueError("person-id is only valid for grouped VideoMimic inputs")
            optimized = np.asarray(optimized_node, dtype=np.float64)
        vm_names = _names(np.asarray(source.attrs["joint_names"]))
        vm_root_pos = np.asarray(source["root_pos"], dtype=np.float64)
        vm_root_quat = np.asarray(source["root_quat"], dtype=np.float64)
    if optimized.shape != (len(root_pos), len(vm_names)):
        raise ValueError(
            f"VideoMimic joints shape {optimized.shape} incompatible with {len(root_pos)} frames and {len(vm_names)} names"
        )
    if vm_root_pos.shape != root_pos.shape or vm_root_quat.shape != root_rot.shape:
        raise ValueError("VideoMimic root arrays do not share reference frame count")

    allowed = {name.strip() for name in args.allow_unmapped.split(",") if name.strip()}
    unknown_vm = [name for name in vm_names if name not in dof_names]
    unmapped_reference = [name for name in dof_names if name not in vm_names]
    unsafe_unmapped = [name for name in unmapped_reference if name not in allowed]
    if unknown_vm or unsafe_unmapped:
        raise ValueError(
            "Cannot make a factor-routed candidate: "
            f"unknown VideoMimic joints={unknown_vm}; unrepresented non-whitelisted reference joints={unsafe_unmapped}"
        )

    candidate = dict(reference)
    candidate_dof = dof_pos.copy()
    mapped_indices: dict[str, int] = {}
    for vm_index, name in enumerate(vm_names):
        reference_index = dof_names.index(name)
        candidate_dof[:, reference_index] = optimized[:, vm_index]
        mapped_indices[name] = reference_index

    candidate["root_pos"] = root_pos.copy()
    candidate["root_rot"] = root_rot.copy()
    candidate["dof_pos"] = candidate_dof.astype(np.float32)
    candidate["source_motion"] = str(args.reference_motion)
    candidate["factor_provenance"] = {
        "schema_version": 1,
        "route": "visual_root_and_orientation_from_reference__named_joint_factors_from_videomimic",
        "root_pos": "reference_motion_immutable",
        "root_rot": "reference_motion_immutable",
        "mapped_joint_names": vm_names,
        "unmapped_joint_names": unmapped_reference,
        "unmapped_policy": "retain_reference_only_for_explicit_whitelisted_non_load_bearing_factors",
        "videomimic_h5": str(args.videomimic_h5),
        "videomimic_sha256": _sha256(args.videomimic_h5),
    }
    candidate["retarget_mode"] = "factor_routed_videomimic_temporal_candidate"
    candidate.pop("local_body_pos", None)

    args.output_motion.parent.mkdir(parents=True, exist_ok=True)
    with args.output_motion.open("wb") as f:
        pickle.dump(candidate, f)

    difference = candidate_dof - dof_pos
    report = {
        "status": "candidate_prepared_for_contract_audit",
        "purpose": "batch_factor_router_not_raw_npz_merge",
        "reference_motion": str(args.reference_motion),
        "reference_sha256": _sha256(args.reference_motion),
        "videomimic_h5": str(args.videomimic_h5),
        "videomimic_sha256": _sha256(args.videomimic_h5),
        "output_motion": str(args.output_motion),
        "frames": int(len(root_pos)),
        "root_contract": "root_pos/root_rot copied byte-for-byte from visual reference",
        "mapped_joints": vm_names,
        "retained_unmapped_joints": unmapped_reference,
        "mapped_joint_delta_rad": {
            "p50_abs": float(np.quantile(np.abs(difference[:, list(mapped_indices.values())]), 0.50)),
            "p95_abs": float(np.quantile(np.abs(difference[:, list(mapped_indices.values())]), 0.95)),
            "maximum_abs": float(np.max(np.abs(difference[:, list(mapped_indices.values())]))),
        },
        "routing": "must pass visual/scene contract before mj_step; rejection retains the reference motion",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

