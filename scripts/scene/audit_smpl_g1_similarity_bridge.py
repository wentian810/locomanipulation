#!/usr/bin/env python3
"""Audit one global Sim(3) bridge from the VideoMimic SMPL scene to G1.

The result is deliberately an audit artifact.  It fits a *single* scale,
rotation and translation from matched lower-body landmarks, and derives the
equivalent camera extrinsics.  It never changes a scene, a robot motion, or a
runtime state.  A candidate camera file is emitted only after the residual
gate passes, so a metric hypothesis cannot silently enter rendering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import smplx
import torch


# SMPL-H joint indices and their closest G1 link origins.  These are used only
# to establish a static coordinate/metric bridge, never as a contact proxy.
LANDMARKS = (
    ("pelvis", 0, "pelvis"),
    ("left_hip", 1, "left_hip_pitch_link"),
    ("right_hip", 2, "right_hip_pitch_link"),
    ("left_knee", 4, "left_knee_link"),
    ("right_knee", 5, "right_knee_link"),
    ("left_ankle", 7, "left_ankle_pitch_link"),
    ("right_ankle", 8, "right_ankle_pitch_link"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def qpos_from_motion(motion: dict[str, Any]) -> np.ndarray:
    root = np.asarray(motion["root_pos"], dtype=np.float64)
    quaternion_xyzw = np.asarray(motion["root_rot"], dtype=np.float64)
    dof = np.asarray(motion["dof_pos"], dtype=np.float64)
    if root.ndim != 2 or root.shape[1] != 3 or quaternion_xyzw.shape != (len(root), 4):
        raise ValueError("robot root fields are malformed")
    if dof.ndim != 2 or len(dof) != len(root):
        raise ValueError("robot dof_pos is malformed")
    qpos = np.empty((len(root), 7 + dof.shape[1]), dtype=np.float64)
    qpos[:, :3] = root
    qpos[:, 3:7] = quaternion_xyzw[:, [3, 0, 1, 2]]
    qpos[:, 7:] = dof
    return qpos


def smpl_joints_z_up(motion: Any, body_model_root: Path, device: str) -> np.ndarray:
    required = {"trans", "root_orient", "pose_body", "betas"}
    missing = required - set(motion.files)
    if missing:
        raise ValueError(f"SMPL motion misses fields: {sorted(missing)}")
    frames = len(motion["trans"])
    gender = str(motion["gender"].item()).lower() if "gender" in motion.files else "neutral"
    if not (body_model_root / "smplh" / f"SMPLH_{gender.upper()}.npz").is_file():
        gender = "neutral"
    model = smplx.create(
        str(body_model_root), model_type="smplh", gender=gender, ext="npz",
        num_betas=len(motion["betas"]), use_pca=False, batch_size=frames,
    ).to(device)
    betas = np.repeat(np.asarray(motion["betas"], dtype=np.float32)[None], frames, axis=0)
    with torch.no_grad():
        output = model(
            betas=torch.as_tensor(betas, dtype=torch.float32, device=device),
            global_orient=torch.as_tensor(motion["root_orient"], dtype=torch.float32, device=device),
            body_pose=torch.as_tensor(motion["pose_body"], dtype=torch.float32, device=device),
            transl=torch.as_tensor(motion["trans"], dtype=torch.float32, device=device),
            return_verts=False,
        )
    joints = output.joints.detach().cpu().numpy()[..., [0, 2, 1]]
    joints[..., 1] *= -1.0  # neg-Y-up -> MuJoCo Z-up: [x, z, -y]
    return joints


def fit_similarity(
    source: np.ndarray, target: np.ndarray, fixed_scale: float | None = None
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return y = scale * R @ x + translation for column-vector points."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("similarity fit needs matched [N,3] landmarks")
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    # Row-vector Kabsch first, then transpose to the documented column form.
    u, _, vt = np.linalg.svd(source_zero.T @ target_zero)
    row_rotation = u @ vt
    if np.linalg.det(row_rotation) < 0.0:
        u[:, -1] *= -1.0
        row_rotation = u @ vt
    rotation = row_rotation.T
    denominator = float(np.sum(source_zero * source_zero))
    if denominator <= 1e-12:
        raise ValueError("degenerate source landmark cloud")
    measured_scale = float(np.sum((source_zero @ row_rotation) * target_zero) / denominator)
    scale = measured_scale if fixed_scale is None else float(fixed_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("similarity fit produced an invalid scale")
    translation = target_center - scale * (rotation @ source_center)
    return scale, rotation, translation


def residuals(source: np.ndarray, target: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    predicted = scale * (source @ rotation.T) + translation
    return np.linalg.norm(predicted - target, axis=1)


def summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0, "p50_m": float("nan"), "p95_m": float("nan"), "max_m": float("nan")}
    return {
        "count": int(len(values)), "p50_m": float(np.quantile(values, 0.5)),
        "p95_m": float(np.quantile(values, 0.95)), "max_m": float(np.max(values)),
    }


def transformed_w2c(camera: dict[str, np.ndarray], scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transforms = np.asarray(camera["T_w2c"], dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError("camera T_w2c must be [T,4,4]")
    result = np.empty_like(transforms)
    for index, transform in enumerate(transforms):
        source_rotation = transform[:3, :3]
        source_translation = transform[:3, 3]
        # q_g = s R q_s + t.  The transformed camera has R_c R^T and
        # -R'_c t + s t_c; the common scale cancels in perspective division.
        target_rotation = source_rotation @ rotation.T
        target_translation = -target_rotation @ translation + scale * source_translation
        result[index] = np.eye(4)
        result[index, :3, :3] = target_rotation
        result[index, :3, 3] = target_translation
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smpl-motion", type=Path)
    parser.add_argument(
        "--smpl-joints-zup", type=Path,
        help="pre-exported joints_zup [T,J,3], used when SMPL and MuJoCo live in separate environments",
    )
    parser.add_argument("--robot-motion", required=True, type=Path)
    parser.add_argument("--task-xml", required=True, type=Path)
    parser.add_argument("--camera", required=True, type=Path)
    parser.add_argument("--body-model-root", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--camera-output", required=True, type=Path)
    parser.add_argument("--inlier-residual-m", type=float, default=0.10)
    parser.add_argument("--min-inlier-fraction", type=float, default=0.80)
    parser.add_argument("--max-inlier-p95-m", type=float, default=0.075)
    parser.add_argument(
        "--fixed-scale", type=float, default=None,
        help="use an independently audited global metric scale while fitting only rotation/translation",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.report.exists() or args.camera_output.exists():
        raise FileExistsError("refusing to overwrite a similarity-bridge artifact")
    if args.inlier_residual_m <= 0.0 or not 0.0 < args.min_inlier_fraction <= 1.0 or args.max_inlier_p95_m <= 0.0:
        raise ValueError("invalid similarity acceptance thresholds")
    if args.fixed_scale is not None and not 0.0 < args.fixed_scale < 2.0:
        raise ValueError("fixed-scale must be positive and finite")

    if (args.smpl_motion is None) == (args.smpl_joints_zup is None):
        raise ValueError("provide exactly one of --smpl-motion and --smpl-joints-zup")
    if args.smpl_motion is not None:
        if args.body_model_root is None:
            raise ValueError("--body-model-root is required with --smpl-motion")
        smpl_motion = np.load(args.smpl_motion, allow_pickle=False)
        smpl_joints = smpl_joints_z_up(smpl_motion, args.body_model_root, args.device)
        smpl_input = {
            "smpl_motion": str(args.smpl_motion),
            "smpl_motion_sha256": sha256(args.smpl_motion),
            "smpl_joints_zup": None,
        }
    else:
        joints_payload = np.load(args.smpl_joints_zup, allow_pickle=False)
        if "joints_zup" not in joints_payload.files:
            raise ValueError("--smpl-joints-zup must contain joints_zup")
        smpl_joints = np.asarray(joints_payload["joints_zup"], dtype=np.float64)
        if smpl_joints.ndim != 3 or smpl_joints.shape[2] != 3:
            raise ValueError("joints_zup must be [T,J,3]")
        smpl_input = {
            "smpl_motion": None, "smpl_motion_sha256": None,
            "smpl_joints_zup": str(args.smpl_joints_zup),
            "smpl_joints_zup_sha256": sha256(args.smpl_joints_zup),
        }
    with args.robot_motion.open("rb") as handle:
        robot_motion = pickle.load(handle)
    qpos = qpos_from_motion(robot_motion)
    model = mujoco.MjModel.from_xml_path(str(args.task_xml))
    if model.nq != qpos.shape[1] or model.nkey != len(qpos):
        raise ValueError("task XML dimensions do not exactly match robot motion")
    if not np.allclose(model.key_qpos, qpos, atol=1e-6, rtol=0.0):
        raise ValueError("task XML keyframes do not exactly match robot motion")
    if len(smpl_joints) != len(qpos):
        raise ValueError("SMPL and robot frame counts differ")
    body_ids = []
    for _, _, body in LANDMARKS:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if body_id < 0:
            raise ValueError(f"task XML has no required G1 body {body!r}")
        body_ids.append(body_id)
    data = mujoco.MjData(model)
    g1_points = np.empty((len(qpos), len(body_ids), 3), dtype=np.float64)
    for frame, pose in enumerate(qpos):
        data.qpos[:] = pose
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        g1_points[frame] = data.xpos[body_ids]
    smpl_indices = [index for _, index, _ in LANDMARKS]
    source_points = smpl_joints[:, smpl_indices].reshape(-1, 3)
    target_points = g1_points.reshape(-1, 3)
    scale, rotation, translation = fit_similarity(source_points, target_points, args.fixed_scale)
    first_residuals = residuals(source_points, target_points, scale, rotation, translation)
    inliers = first_residuals <= args.inlier_residual_m
    if int(np.sum(inliers)) < 3:
        raise RuntimeError("similarity fit has fewer than three residual inliers")
    scale, rotation, translation = fit_similarity(source_points[inliers], target_points[inliers], args.fixed_scale)
    all_residuals = residuals(source_points, target_points, scale, rotation, translation)
    inliers = all_residuals <= args.inlier_residual_m
    inlier_residuals = all_residuals[inliers]
    inlier_fraction = float(np.mean(inliers))
    all_summary = summary(all_residuals)
    inlier_summary = summary(inlier_residuals)
    accepted = bool(
        inlier_fraction >= args.min_inlier_fraction
        and inlier_summary["p95_m"] <= args.max_inlier_p95_m
        and 0.65 <= scale <= 0.95
    )
    camera = np.load(args.camera)
    camera_payload = {key: camera[key] for key in camera.files}
    if len(camera_payload.get("T_w2c", [])) != len(qpos):
        raise ValueError("camera and motion frame counts differ")
    if accepted:
        camera_payload["T_w2c"] = transformed_w2c(camera_payload, scale, rotation, translation)
        args.camera_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.camera_output, **camera_payload)
    report = {
        "schema_version": 1,
        "purpose": "audit_static_smpl_to_g1_similarity_bridge",
        "status": "accepted_similarity_bridge_candidate" if accepted else "rejected_similarity_bridge",
        "policy": {
            "transform": "one_global_scale_rotation_translation",
            "scene_motion": "static_only",
            "camera_output_written": accepted,
            "forbidden": ["per_frame_scene_motion", "per_frame_root_rewrite", "dynamic_chair"],
        },
        "inputs": {
            **smpl_input,
            "robot_motion": str(args.robot_motion), "robot_motion_sha256": sha256(args.robot_motion),
            "task_xml": str(args.task_xml), "camera": str(args.camera), "camera_sha256": sha256(args.camera),
        },
        "landmarks": [{"name": name, "smpl_joint": joint, "g1_body": body} for name, joint, body in LANDMARKS],
        "similarity_smpl_zup_to_g1": {
            "scale": scale, "scale_mode": "fixed_independent_metric_audit" if args.fixed_scale is not None else "joint_landmark_fit",
            "rotation_matrix": rotation.tolist(), "translation_xyz_m": translation.tolist(),
        },
        "residuals": {"all": all_summary, "inliers": inlier_summary, "inlier_fraction": inlier_fraction},
        "thresholds": {
            "inlier_residual_m": args.inlier_residual_m,
            "min_inlier_fraction": args.min_inlier_fraction,
            "max_inlier_p95_m": args.max_inlier_p95_m,
            "allowed_scale_range": [0.65, 0.95],
        },
        "camera_output": str(args.camera_output) if accepted else None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "scale": scale, "residuals": report["residuals"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
