#!/usr/bin/env python3
"""Batch-safe VideoMimic Stage-4 retargeting with real contact-surface recovery.

This sidecar is deliberately non-semantic.  It keeps the foreground-cleaned
first-round NKSR residual, then restores only local, near-horizontal raw NKSR
triangles selected by official BSTRO foot-contact labels.  The recovered mesh,
GVHMR evidence, and Stage-4 output all share the MuJoCo Z-up contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import trimesh
from scipy.spatial import cKDTree


class ContractError(RuntimeError):
    """Raised when a cross-stage coordinate or identity contract is violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ContractError(f"missing {label}: {resolved}")
    return resolved


def _link(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        if target.resolve() != source.resolve():
            raise ContractError(f"refusing to mix existing input at {target}")
        return
    target.symlink_to(source)


def _frame_names(megahunter_h5: Path) -> list[str]:
    with h5py.File(megahunter_h5, "r") as archive:
        key = "our_pred_world_cameras_and_structure"
        if key not in archive:
            raise ContractError(f"frame-index H5 lacks {key}: {megahunter_h5}")
        names = list(archive[key].keys())
    if not names:
        raise ContractError("frame-index H5 has no frames")
    return names


def _load_contacts(contact_dir: Path, names: list[str], person_id: str) -> tuple[np.ndarray, np.ndarray]:
    numeric_id = int(person_id)
    left: list[bool] = []
    right: list[bool] = []
    for name in names:
        path = contact_dir / f"{name}.pkl"
        if not path.is_file():
            raise ContractError(f"BSTRO contact file missing: {path}")
        with path.open("rb") as stream:
            record = pickle.load(stream)
        if numeric_id not in record:
            raise ContractError(f"BSTRO person id {numeric_id} absent in {path.name}")
        value = record[numeric_id]
        left.append(bool(value.get("left_foot_contact", False)))
        right.append(bool(value.get("right_foot_contact", False)))
    return np.asarray(left, dtype=bool), np.asarray(right, dtype=bool)


def _load_keypoints(path: Path, person_id: str, frames: int) -> np.ndarray:
    with h5py.File(path, "r") as archive:
        try:
            keypoints = np.asarray(archive["joints"][person_id], dtype=np.float64)
        except KeyError as exc:
            available = list(archive.get("joints", {}).keys())
            raise ContractError(f"evidence person id {person_id!r} unavailable; found {available}") from exc
    if keypoints.shape != (frames, 45, 3):
        raise ContractError(f"expected evidence keypoints {(frames, 45, 3)}, got {keypoints.shape}")
    if not np.isfinite(keypoints).all():
        raise ContractError("evidence keypoints contain non-finite values")
    return keypoints



def _vertical_face(mesh: trimesh.Trimesh, point: np.ndarray) -> int:
    """Nearest triangle on a vertical line, downward first, without rtree."""
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    x, y, origin_z = map(float, point)
    xy = triangles[:, :, :2]
    candidate = (xy[:, :, 0].min(axis=1) <= x + 1e-9) & (xy[:, :, 0].max(axis=1) >= x - 1e-9) & (xy[:, :, 1].min(axis=1) <= y + 1e-9) & (xy[:, :, 1].max(axis=1) >= y - 1e-9)
    indices = np.flatnonzero(candidate)
    if not len(indices):
        return -1
    tri = triangles[indices]
    edge_a = tri[:, 1, :2] - tri[:, 0, :2]
    edge_b = tri[:, 2, :2] - tri[:, 0, :2]
    delta = np.array([x, y]) - tri[:, 0, :2]
    determinant = edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]
    nonvertical = np.abs(determinant) > 1e-12
    u = np.zeros(len(indices))
    v = np.zeros(len(indices))
    u[nonvertical] = (delta[nonvertical, 0] * edge_b[nonvertical, 1] - edge_b[nonvertical, 0] * delta[nonvertical, 1]) / determinant[nonvertical]
    v[nonvertical] = (edge_a[nonvertical, 0] * delta[nonvertical, 1] - delta[nonvertical, 0] * edge_a[nonvertical, 1]) / determinant[nonvertical]
    inside = nonvertical & (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1.0 + 1e-8)
    if not inside.any():
        return -1
    indices = indices[inside]
    tri = tri[inside]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    valid = np.abs(normal[:, 2]) > 1e-12
    if not valid.any():
        return -1
    indices = indices[valid]
    tri = tri[valid]
    normal = normal[valid]
    z = tri[:, 0, 2] - (normal[:, 0] * (x - tri[:, 0, 0]) + normal[:, 1] * (y - tri[:, 0, 1])) / normal[:, 2]
    downward = z <= origin_z + 1e-8
    if downward.any():
        return int(indices[downward][np.argmax(z[downward])])
    return int(indices[np.argmin(z)])


def _ray_hits(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    return np.asarray([_vertical_face(mesh, np.array([point[0], point[1], point[2] + 0.10])) >= 0 for point in points], dtype=bool)


def _coverage(mesh: trimesh.Trimesh, keypoints: np.ndarray, left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, index, labels in (("left", 10, left), ("right", 11, right)):
        hits = _ray_hits(mesh, keypoints[:, index])
        count = int(labels.sum())
        contact_hits = int((hits & labels).sum())
        result[name] = {
            "contact_frames": count,
            "contact_ray_hits": contact_hits,
            "contact_ray_coverage": float(contact_hits / count) if count else None,
            "all_frame_ray_hits": int(hits.sum()),
            "all_frame_ray_coverage": float(hits.mean()),
        }
    return result


def _recover(mesh_raw: trimesh.Trimesh, mesh_residual: trimesh.Trimesh, keypoints: np.ndarray, left: np.ndarray, right: np.ndarray, *, normal_threshold: float, radius: float, height_band: float) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    raw_centers = np.asarray(mesh_raw.triangles_center, dtype=np.float64)
    raw_normals = np.asarray(mesh_raw.face_normals, dtype=np.float64)
    residual_centers = np.asarray(mesh_residual.triangles_center, dtype=np.float64)
    retained_distance = cKDTree(residual_centers).query(raw_centers, workers=-1)[0]
    removed = retained_distance > 1e-5
    horizontal_tree = cKDTree(raw_centers[:, :2])
    selected: set[int] = set()
    seeds: set[int] = set()
    examined = 0
    for foot_index, labels in ((10, left), (11, right)):
        for frame in np.flatnonzero(labels):
            examined += 1
            point = keypoints[frame, foot_index]
            face = _vertical_face(mesh_raw, np.array([point[0], point[1], point[2] + 0.10]))
            if face < 0 or abs(float(raw_normals[face, 2])) < normal_threshold:
                continue
            seeds.add(face)
            center = raw_centers[face]
            local = np.asarray(horizontal_tree.query_ball_point(center[:2], radius), dtype=np.int64)
            if not len(local):
                continue
            local = local[(np.abs(raw_centers[local, 2] - center[2]) <= height_band) & (np.abs(raw_normals[local, 2]) >= normal_threshold) & removed[local]]
            selected.update(local.tolist())
    before = _coverage(mesh_residual, keypoints, left, right)
    if selected:
        patch = mesh_raw.submesh([np.asarray(sorted(selected), dtype=np.int64)], append=True, repair=False)
        recovered = trimesh.util.concatenate([mesh_residual, patch])
    else:
        recovered = mesh_residual.copy()
    after = _coverage(recovered, keypoints, left, right)
    return recovered, {
        "status": "recovered" if selected else "no_recoverable_contact_faces",
        "method": "bstro_contact_ray_local_horizontal_raw_nksr_face_recovery",
        "parameters": {
            "ray_origin_offset_m": 0.10,
            "min_abs_vertical_normal": normal_threshold,
            "horizontal_patch_radius_m": radius,
            "vertical_patch_half_band_m": height_band,
            "raw_to_residual_face_match_tolerance_m": 1e-5,
        },
        "labeled_contact_frames": int(examined),
        "seed_face_count": int(len(seeds)),
        "recovered_face_count": int(len(selected)),
        "raw_face_count": int(len(mesh_raw.faces)),
        "residual_face_count": int(len(mesh_residual.faces)),
        "fused_face_count": int(len(recovered.faces)),
        "coverage_before": before,
        "coverage_after": after,
    }


def _validate_coverage(report: dict[str, Any], minimum: float) -> None:
    values = [item["contact_ray_coverage"] for item in report["coverage_after"].values() if item["contact_ray_coverage"] is not None]
    if not values:
        return
    aggregate_hits = sum(int(item["contact_ray_hits"]) for item in report["coverage_after"].values())
    aggregate_frames = sum(int(item["contact_frames"]) for item in report["coverage_after"].values())
    aggregate = aggregate_hits / aggregate_frames if aggregate_frames else 1.0
    report["minimum_required_contact_coverage"] = minimum
    report["aggregate_contact_ray_coverage"] = aggregate
    if aggregate < minimum:
        raise ContractError(f"contact-surface recovery coverage {aggregate:.3f} below required {minimum:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smpl-npz", required=True, type=Path)
    parser.add_argument("--body-model-root", required=True, type=Path)
    parser.add_argument("--prepare-evidence-script", required=True, type=Path)
    parser.add_argument("--evidence-python", required=True, type=Path)
    parser.add_argument("--raw-mesh", required=True, type=Path)
    parser.add_argument("--residual-mesh", required=True, type=Path)
    parser.add_argument("--megahunter-h5", required=True, type=Path)
    parser.add_argument("--contact-dir", required=True, type=Path)
    parser.add_argument("--stage4-script", required=True, type=Path)
    parser.add_argument("--stage4-python", required=True, type=Path)
    parser.add_argument("--output-h5", required=True, type=Path)
    parser.add_argument("--person-id", default="1")
    parser.add_argument("--stage4-cudnn-lib", type=Path, default=None)
    parser.add_argument("--minimum-contact-coverage", type=float, default=0.75)
    parser.add_argument("--normal-threshold", type=float, default=0.65)
    parser.add_argument("--patch-radius-m", type=float, default=0.08)
    parser.add_argument("--patch-height-band-m", type=float, default=0.035)
    args = parser.parse_args()
    if not 0.0 <= args.minimum_contact_coverage <= 1.0:
        parser.error("--minimum-contact-coverage must be in [0,1]")
    if not 0.0 < args.normal_threshold <= 1.0 or args.patch_radius_m <= 0.0 or args.patch_height_band_m <= 0.0:
        parser.error("invalid recovery thresholds")
    inputs = {
        "smpl_npz": _require(args.smpl_npz, "SMPL motion"),
        "body_model_root": args.body_model_root.expanduser().resolve(),
        "prepare_evidence_script": _require(args.prepare_evidence_script, "evidence adapter"),
        "evidence_python": _require(args.evidence_python, "evidence Python"),
        "raw_mesh": _require(args.raw_mesh, "raw MuJoCo-frame NKSR mesh"),
        "residual_mesh": _require(args.residual_mesh, "cleaned MuJoCo-frame NKSR mesh"),
        "megahunter_h5": _require(args.megahunter_h5, "frame-index MegaHunter H5"),
        "stage4_script": _require(args.stage4_script, "VideoMimic Stage-4 script"),
        "stage4_python": _require(args.stage4_python, "Stage-4 Python"),
    }
    if not inputs["body_model_root"].is_dir() or not args.contact_dir.is_dir():
        raise ContractError("body-model root or BSTRO contact directory is missing")
    output = args.output_h5.expanduser().resolve()
    if output.exists():
        raise ContractError(f"refusing to overwrite existing Stage-4 output: {output}")
    frames = _frame_names(inputs["megahunter_h5"])
    stage_dir = output.parent / f"{output.stem}_input_frame_0_{len(frames)-1}_subsample_1"
    if stage_dir.exists():
        raise ContractError(f"refusing to mix existing Stage-4 input directory: {stage_dir}")
    stage_dir.mkdir(parents=True)
    evidence = stage_dir / "gravity_calibrated_keypoints.h5"
    evidence_manifest = stage_dir / "evidence_manifest.json"
    prepare_command = [str(inputs["evidence_python"]), str(inputs["prepare_evidence_script"]), "--smpl-npz", str(inputs["smpl_npz"]), "--body-model-root", str(inputs["body_model_root"]), "--person-id", str(args.person_id), "--output-h5", str(evidence), "--manifest", str(evidence_manifest)]
    prepare_env = os.environ.copy()
    prepare_env["PYTHONNOUSERSITE"] = "1"
    subprocess.run(prepare_command, check=True, env=prepare_env)
    keypoints = _load_keypoints(evidence, str(args.person_id), len(frames))
    left, right = _load_contacts(args.contact_dir.resolve(), frames, str(args.person_id))
    raw = trimesh.load(inputs["raw_mesh"], force="mesh", process=False)
    residual = trimesh.load(inputs["residual_mesh"], force="mesh", process=False)
    if not isinstance(raw, trimesh.Trimesh) or not isinstance(residual, trimesh.Trimesh):
        raise ContractError("raw or residual mesh is not a triangle mesh")
    recovered, recovery = _recover(raw, residual, keypoints, left, right, normal_threshold=args.normal_threshold, radius=args.patch_radius_m, height_band=args.patch_height_band_m)
    _validate_coverage(recovery, args.minimum_contact_coverage)
    recovered_path = stage_dir / "background_mesh_contact_recovered.obj"
    recovered.export(recovered_path)
    _link(recovered_path, stage_dir / "background_mesh.obj")
    _link(inputs["megahunter_h5"], stage_dir / "gravity_calibrated_megahunter.h5")
    stage_env = os.environ.copy()
    stage_env["PYTHONNOUSERSITE"] = "1"
    if args.stage4_cudnn_lib is not None:
        cudnn = args.stage4_cudnn_lib.expanduser().resolve()
        if not cudnn.is_dir():
            raise ContractError(f"missing Stage-4 cuDNN library directory: {cudnn}")
        stage_env["LD_LIBRARY_PATH"] = str(cudnn) + ":" + str(args.stage4_python.parent.parent / "lib")
    stage_log = output.with_suffix(".stage4.log")
    with stage_log.open("w", encoding="utf-8") as log:
        subprocess.run([str(inputs["stage4_python"]), str(inputs["stage4_script"]), "--src-dir", str(stage_dir), "--contact-dir", str(args.contact_dir.resolve())], check=True, cwd=inputs["stage4_script"].parent.parent, env=stage_env, stdout=log, stderr=subprocess.STDOUT)
    generated = stage_dir / "retarget_poses_g1.h5"
    if not generated.is_file() or generated.stat().st_size == 0:
        raise ContractError("Stage-4 did not create retarget_poses_g1.h5")
    with h5py.File(generated, "r") as archive:
        required = ("root_pos", "root_quat", "joints", "link_pos", "link_quat", "contacts")
        missing = [name for name in required if name not in archive]
        if missing or archive["root_pos"].shape[0] != len(frames):
            raise ContractError(f"invalid Stage-4 output; missing={missing}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    shutil.copy2(generated, temporary)
    temporary.replace(output)
    manifest = {
        "schema_version": 1,
        "status": "pass",
        "coordinate_contract": "gvhmr_static_hard_and_nksr_mujoco_z_up",
        "person_id": str(args.person_id),
        "frames": len(frames),
        "inputs": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in inputs.items() if path.is_file()},
        "bstro_contact_frames": {"left": int(left.sum()), "right": int(right.sum())},
        "contact_surface_recovery": recovery,
        "recovered_mesh": str(recovered_path),
        "stage4_log": str(stage_log),
        "output_h5": str(output),
        "output_sha256": _sha256(output),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "output_h5": str(output), "coverage": recovery.get("aggregate_contact_ray_coverage"), "recovered_faces": recovery["recovered_face_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
