#!/usr/bin/env python3
"""Create a bounded, policy-attainable Stage-4 G1 reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


class ContractError(RuntimeError):
    """Raised when a projected reference would not be batch-safe."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strings(values: Any) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def _resample(values: np.ndarray, frames: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1 or values.shape[0] < 2:
        raise ContractError(f"cannot resample shape {values.shape}")
    time_in = np.linspace(0.0, 1.0, values.shape[0])
    time_out = np.linspace(0.0, 1.0, frames)
    flat = values.reshape(values.shape[0], -1)
    result = np.empty((frames, flat.shape[1]), dtype=np.float64)
    for column in range(flat.shape[1]):
        result[:, column] = np.interp(time_out, time_in, flat[:, column])
    return result.reshape((frames,) + values.shape[1:])


def _normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ContractError("encountered a zero-length quaternion")
    return quaternions / norms


def _resample_quaternions(quaternions: np.ndarray, frames: int) -> np.ndarray:
    values = _normalize_quaternions(quaternions).copy()
    flat = values.reshape(values.shape[0], -1, 4)
    for index in range(1, flat.shape[0]):
        flip = np.sum(flat[index - 1] * flat[index], axis=-1) < 0.0
        flat[index, flip] *= -1.0
    return _normalize_quaternions(_resample(flat.reshape(values.shape), frames))


def _blend_quaternions(source: np.ndarray, actual: np.ndarray, alpha: float) -> np.ndarray:
    source = _normalize_quaternions(source)
    actual = _normalize_quaternions(actual)
    actual = actual.copy()
    actual[np.sum(source * actual, axis=-1) < 0.0] *= -1.0
    return _normalize_quaternions((1.0 - alpha) * source + alpha * actual)


def _motion_group(name: str) -> str:
    """Return a robot-anatomy group without depending on a source clip."""
    token = name.lower()
    if any(part in token for part in ("hip", "knee", "ankle", "foot")):
        return "leg"
    if any(part in token for part in ("shoulder", "elbow", "wrist", "hand")):
        return "arm"
    return "torso"


def _group_alpha(name: str, args: argparse.Namespace) -> float:
    group = _motion_group(name)
    value = getattr(args, f"{group}_alpha")
    return float(args.alpha if value is None else value)


def _canonical_body_name(name: str) -> str:
    """Normalize the one conventional difference between H5 and simulator links."""
    return name[:-5] if name.endswith("_link") else name

def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", required=True, type=Path)
    parser.add_argument("--capture-npz", required=True, type=Path)
    parser.add_argument("--output-h5", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--root-alpha", type=float, default=None)
    parser.add_argument("--leg-alpha", type=float, default=None)
    parser.add_argument("--arm-alpha", type=float, default=None)
    parser.add_argument("--torso-alpha", type=float, default=None)
    parser.add_argument("--max-root-deviation-m", type=float, default=0.16)
    parser.add_argument("--max-link-deviation-m", type=float, default=0.25)
    parser.add_argument("--max-joint-deviation-rad", type=float, default=0.50)
    parser.add_argument("--max-violation-frame-ratio", type=float, default=0.0)
    args = parser.parse_args()
    if not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must be in (0, 1)")
    for name in ("root_alpha", "leg_alpha", "arm_alpha", "torso_alpha"):
        value = getattr(args, name)
        if value is not None and not 0.0 <= value < 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1)")
    if min(args.max_root_deviation_m, args.max_link_deviation_m, args.max_joint_deviation_rad) <= 0.0:
        parser.error("source-deviation limits must be positive")
    if not 0.0 <= args.max_violation_frame_ratio <= 1.0:
        parser.error("--max-violation-frame-ratio must be in [0, 1]")
    return args


def main() -> None:
    args = _parse_args()
    source = args.source_h5.expanduser().resolve()
    capture_path = args.capture_npz.expanduser().resolve()
    output = args.output_h5.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    if not source.is_file() or not capture_path.is_file():
        raise ContractError("source H5 and capture NPZ must exist")
    if output.exists():
        raise ContractError(f"refusing to overwrite output H5: {output}")

    capture = np.load(capture_path)
    required = {
        "root_states", "dof_positions", "dof_names", "target_root_positions",
        "target_dof_positions", "actual_body_positions", "actual_body_quaternions",
        "body_names",
    }
    missing = sorted(required.difference(capture.files))
    if missing:
        raise ContractError(f"capture lacks policy-state fields: {missing}")

    with h5py.File(source, "r") as archive:
        root_pos = np.asarray(archive["root_pos"], dtype=np.float64)
        root_quat = np.asarray(archive["root_quat"], dtype=np.float64)
        joints = np.asarray(archive["joints"], dtype=np.float64)
        link_pos = np.asarray(archive["link_pos"], dtype=np.float64)
        link_quat = np.asarray(archive["link_quat"], dtype=np.float64)
        joint_names = _strings(archive.attrs["joint_names"])
        link_names = _strings(archive.attrs["link_names"])
    frames = root_pos.shape[0]
    if not (root_quat.shape[0] == joints.shape[0] == link_pos.shape[0] == link_quat.shape[0] == frames):
        raise ContractError("source H5 arrays do not share a frame dimension")

    root_states = np.asarray(capture["root_states"], dtype=np.float64)
    actual_root_pos = _resample(root_states[:, :3], frames)
    actual_root_quat = _resample_quaternions(root_states[:, 3:7], frames)
    target_root_pos = _resample(np.asarray(capture["target_root_positions"]), frames)
    actual_dofs = _resample(np.asarray(capture["dof_positions"]), frames)
    target_dofs = _resample(np.asarray(capture["target_dof_positions"]), frames)
    body_pos = _resample(np.asarray(capture["actual_body_positions"]), frames)
    body_quat = _resample_quaternions(np.asarray(capture["actual_body_quaternions"]), frames)
    dof_names = _strings(capture["dof_names"])
    body_names = _strings(capture["body_names"])
    if not (len(dof_names) == actual_dofs.shape[1] == target_dofs.shape[1]):
        raise ContractError("capture DOF-name contract is inconsistent")
    if not (len(body_names) == body_pos.shape[1] == body_quat.shape[1]):
        raise ContractError("capture body-name contract is inconsistent")

    root_alpha = float(args.alpha if args.root_alpha is None else args.root_alpha)
    new_root_pos = root_pos + root_alpha * (actual_root_pos - target_root_pos)
    new_root_quat = _blend_quaternions(root_quat, actual_root_quat, root_alpha)
    new_joints = joints.copy()
    for dof_index, name in enumerate(dof_names):
        if name not in joint_names:
            raise ContractError(f"capture DOF missing from source H5: {name}")
        joint_index = joint_names.index(name)
        new_joints[:, joint_index] += _group_alpha(name, args) * (
            actual_dofs[:, dof_index] - target_dofs[:, dof_index]
        )

    root_offset = target_root_pos - root_pos
    new_link_pos = link_pos.copy()
    new_link_quat = link_quat.copy()
    body_index = {name: index for index, name in enumerate(body_names)}
    canonical_body_index: dict[str, list[int]] = defaultdict(list)
    for name, index in body_index.items():
        canonical_body_index[_canonical_body_name(name)].append(index)
    updated_links: list[str] = []
    for link_index, name in enumerate(link_names):
        if name in body_index:
            index = body_index[name]
        else:
            candidates = canonical_body_index[_canonical_body_name(name)]
            if len(candidates) > 1:
                raise ContractError(f"ambiguous simulator body alias for H5 link: {name}")
            if not candidates:
                continue
            index = candidates[0]
        alpha = _group_alpha(name, args)
        expected_world = link_pos[:, link_index] + root_offset
        new_link_pos[:, link_index] += alpha * (body_pos[:, index] - expected_world)
        new_link_quat[:, link_index] = _blend_quaternions(
            link_quat[:, link_index], body_quat[:, index], alpha
        )
        updated_links.append(name)
    if not updated_links:
        raise ContractError("none of the source H5 links map to the captured G1 bodies")

    root_deviation = np.linalg.norm(new_root_pos - root_pos, axis=-1)
    link_deviation = np.linalg.norm(new_link_pos - link_pos, axis=-1).max(axis=-1)
    joint_deviation = np.abs(new_joints - joints).max(axis=-1)
    violations = (
        (root_deviation > args.max_root_deviation_m)
        | (link_deviation > args.max_link_deviation_m)
        | (joint_deviation > args.max_joint_deviation_rad)
    )
    violation_ratio = float(np.mean(violations))
    deviation = {
        "root_p95_m": float(np.quantile(root_deviation, 0.95)),
        "root_max_m": float(root_deviation.max()),
        "link_p95_m": float(np.quantile(link_deviation, 0.95)),
        "link_max_m": float(link_deviation.max()),
        "joint_p95_rad": float(np.quantile(joint_deviation, 0.95)),
        "joint_max_rad": float(joint_deviation.max()),
        "violation_frame_ratio": violation_ratio,
    }
    accepted = violation_ratio <= args.max_violation_frame_ratio
    report = {
        "status": "pass" if accepted else "rejected",
        "contract": "bounded_policy_attainability_projection_v1",
        "source_h5": str(source),
        "source_h5_sha256": _sha256(source),
        "capture_npz": str(capture_path),
        "capture_npz_sha256": _sha256(capture_path),
        "output_h5": str(output),
        "alpha": float(args.alpha),
        "gain_profile": {
            "root": root_alpha,
            "leg": float(args.alpha if args.leg_alpha is None else args.leg_alpha),
            "arm": float(args.alpha if args.arm_alpha is None else args.arm_alpha),
            "torso": float(args.alpha if args.torso_alpha is None else args.torso_alpha),
        },
        "source_deviation_limits": {
            "max_root_m": float(args.max_root_deviation_m),
            "max_link_m": float(args.max_link_deviation_m),
            "max_joint_rad": float(args.max_joint_deviation_rad),
            "max_violation_frame_ratio": float(args.max_violation_frame_ratio),
        },
        "source_deviation": deviation,
        "updated_link_count": len(updated_links),
        "updated_links": updated_links,
        "contacts_preserved": True,
    }
    if not accepted:
        _write_report(report_path, report)
        raise ContractError(f"projection exceeds source-deviation contract: {deviation}")

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    with h5py.File(output, "r+") as archive:
        archive["root_pos"][...] = new_root_pos
        archive["root_quat"][...] = new_root_quat
        archive["joints"][...] = new_joints.astype(archive["joints"].dtype)
        archive["link_pos"][...] = new_link_pos
        archive["link_quat"][...] = new_link_quat
    report["output_h5_sha256"] = _sha256(output)
    _write_report(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": 
    main()
