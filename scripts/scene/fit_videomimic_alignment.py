"""Fit and quality-gate the VideoMimic-world to GVHMR-world coordinate bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import smplx


JOINT_IDS = np.array([0, 1, 2, 4, 5, 7, 8, 16, 17, 18, 19, 20, 21], dtype=np.int64)
JOINT_NAMES = (
    "pelvis", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
)


def _fit_similarity(source: np.ndarray, target: np.ndarray, fixed_scale: float | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    if len(source) < 3:
        raise ValueError("at least three point correspondences are required")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    u, singular, vt = np.linalg.svd((target_zero.T @ source_zero) / len(source))
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    if fixed_scale is None:
        variance = float((source_zero * source_zero).sum() / len(source))
        scale = float((singular * np.diag(correction)).sum() / max(variance, 1e-12))
    else:
        scale = float(fixed_scale)
    translation = target_center - scale * rotation @ source_center
    return scale, rotation, translation


def _apply(source: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return scale * source @ rotation.T + translation


def _ransac_similarity(source: np.ndarray, target: np.ndarray, threshold: float, trials: int) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260724)
    best_mask: np.ndarray | None = None
    best_score = -1
    count = len(source)
    for _ in range(trials):
        sample = rng.choice(count, size=4, replace=False)
        try:
            scale, rotation, translation = _fit_similarity(source[sample], target[sample])
        except np.linalg.LinAlgError:
            continue
        mask = np.linalg.norm(_apply(source, scale, rotation, translation) - target, axis=1) <= threshold
        score = int(mask.sum())
        if score > best_score:
            best_mask, best_score = mask, score
    if best_mask is None or int(best_mask.sum()) < 12:
        best_mask = np.ones(count, dtype=bool)
    scale, rotation, translation = _fit_similarity(source[best_mask], target[best_mask])
    return scale, rotation, translation, best_mask


def _load_vm_human(path: Path, person_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        root = handle["our_pred_humans_smplx_params"][person_id]
        return (
            np.asarray(root["betas"], dtype=np.float32),
            np.asarray(root["body_pose"], dtype=np.float32),
            np.asarray(root["global_orient"], dtype=np.float32),
            np.asarray(root["root_transl"], dtype=np.float32).reshape(-1, 3),
        )


def _smpl_joints_vm(model: torch.nn.Module, betas: np.ndarray, body: np.ndarray, orient: np.ndarray, transl: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        output = model(
            betas=torch.as_tensor(betas, dtype=torch.float32),
            body_pose=torch.as_tensor(body, dtype=torch.float32),
            global_orient=torch.as_tensor(orient, dtype=torch.float32),
            transl=torch.as_tensor(transl, dtype=torch.float32),
            pose2rot=False,
        )
    return output.joints.cpu().numpy()


def _smpl_joints_gv(model: torch.nn.Module, motion: np.lib.npyio.NpzFile) -> np.ndarray:
    poses = np.asarray(motion["poses"], dtype=np.float32)
    translation = np.asarray(motion["trans"], dtype=np.float32)
    betas = np.asarray(motion["betas"], dtype=np.float32).reshape(-1)[:10]
    if poses.ndim != 3 or poses.shape[1:] != (24, 3):
        raise ValueError(f"expected poses with shape (T, 24, 3), received {poses.shape}")
    with torch.no_grad():
        output = model(
            betas=torch.as_tensor(np.repeat(betas[None], len(poses), axis=0), dtype=torch.float32),
            body_pose=torch.as_tensor(poses[:, 1:24].reshape(len(poses), -1), dtype=torch.float32),
            global_orient=torch.as_tensor(poses[:, 0], dtype=torch.float32),
            transl=torch.as_tensor(translation, dtype=torch.float32),
        )
    return output.joints.cpu().numpy()


def _camera_transform(camera: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray, float]:
    rotation = np.asarray(camera["world_to_isaac"], dtype=np.float64)
    source = np.asarray(camera["subject_world"], dtype=np.float64)
    target = np.asarray(camera["subject_isaac"], dtype=np.float64)
    translation = np.median(target - source @ rotation.T, axis=0)
    error = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
    return rotation, translation, float(np.quantile(error, 0.90))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrated-h5", required=True, type=Path)
    parser.add_argument("--motion-npz", required=True, type=Path)
    parser.add_argument("--camera-npz", required=True, type=Path)
    parser.add_argument("--contact-evidence", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--person-id", default="1")
    parser.add_argument("--gender", choices=("male", "female", "neutral"), default="neutral")
    parser.add_argument("--ransac-threshold-m", type=float, default=0.08)
    parser.add_argument(
        "--source-coordinate-system",
        default="gvhmr_world_gravity_negative_y",
        help="Name recorded for the motion/camera coordinate system being aligned.",
    )
    parser.add_argument(
        "--camera-mode",
        default="native",
        help="Camera treatment recorded in the alignment artifact (for example static_hard).",
    )
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    motion = np.load(args.motion_npz, allow_pickle=False)
    camera = np.load(args.camera_npz, allow_pickle=False)
    evidence = json.loads(args.contact_evidence.read_text(encoding="utf-8"))
    vm_betas, vm_body, vm_orient, vm_transl = _load_vm_human(args.calibrated_h5, args.person_id)
    frame_count = len(vm_body)
    if not (len(motion["poses"]) == len(camera["subject_world"]) == frame_count):
        raise ValueError("motion, camera, and calibrated human frame counts must match exactly")

    model = smplx.create(str(args.model_root), model_type="smpl", gender=args.gender, num_betas=10, batch_size=frame_count)
    vm_joints = _smpl_joints_vm(model, vm_betas, vm_body, vm_orient, vm_transl)
    gv_joints = _smpl_joints_gv(model, motion)
    source = vm_joints[:, JOINT_IDS].reshape(-1, 3).astype(np.float64)
    target = gv_joints[:, JOINT_IDS].reshape(-1, 3).astype(np.float64)

    scale, rotation, translation, ransac_mask = _ransac_similarity(source, target, args.ransac_threshold_m, trials=256)
    ransac_inlier_ratio = float(ransac_mask.mean())
    # A partial RANSAC consensus is a useful failure signal, not a license to
    # align a static scene to only half of a moving body.  Fall back to the
    # all-correspondence least-squares estimate unless the consensus is broad.
    if ransac_inlier_ratio < 0.80:
        scale, rotation, translation = _fit_similarity(source, target)
        fitting_mask = np.ones(len(source), dtype=bool)
        ransac_consensus = "insufficient_for_static_scene_used_all_correspondences"
    else:
        fitting_mask = ransac_mask
        ransac_consensus = "accepted"
    # A scale close to one is expected after MegaHunter.  Fit the rigid version
    # on the same inliers so its residual is directly comparable.
    rigid_rotation, rigid_translation = rotation, translation
    if 0.90 <= scale <= 1.10:
        _, rigid_rotation, rigid_translation = _fit_similarity(source[fitting_mask], target[fitting_mask], fixed_scale=1.0)
        chosen_scale, chosen_rotation, chosen_translation = 1.0, rigid_rotation, rigid_translation
        transform_mode = "rigid_refit_after_sim3_diagnostic"
    else:
        chosen_scale, chosen_rotation, chosen_translation = scale, rotation, translation
        transform_mode = "sim3_retained_scale_outside_expected_range"
    residual = np.linalg.norm(_apply(source, chosen_scale, chosen_rotation, chosen_translation) - target, axis=1)
    residual_by_joint = residual.reshape(frame_count, len(JOINT_IDS))

    t_gv_from_vm = np.eye(4, dtype=np.float64)
    t_gv_from_vm[:3, :3] = chosen_scale * chosen_rotation
    t_gv_from_vm[:3, 3] = chosen_translation
    world_to_isaac, isaac_translation, isaac_fit_p90 = _camera_transform(camera)
    t_mujoco_from_gv = np.eye(4, dtype=np.float64)
    t_mujoco_from_gv[:3, :3] = world_to_isaac
    t_mujoco_from_gv[:3, 3] = isaac_translation
    t_mujoco_from_vm = t_mujoco_from_gv @ t_gv_from_vm

    seated = np.zeros(frame_count, dtype=bool)
    for item in evidence["frames"]:
        if item.get("seated_candidate"):
            seated[int(item["frame"])] = True
    contacts = {
        "left_foot_contact": np.zeros(frame_count, dtype=bool),
        "right_foot_contact": np.zeros(frame_count, dtype=bool),
        "left_foot_point_gvhmr": gv_joints[:, 7].astype(np.float32),
        "right_foot_point_gvhmr": gv_joints[:, 8].astype(np.float32),
        "pelvis_contact": seated,
        "back_contact": np.zeros(frame_count, dtype=bool),
        "contact_confidence": np.column_stack((np.zeros(frame_count), np.zeros(frame_count), seated.astype(np.float32) * 0.55, np.zeros(frame_count))).astype(np.float32),
        "contact_source": np.array("kinematic_seated_pose_plus_first_round_nksr_proximity", dtype="<U64"),
    }
    np.savez_compressed(output / "human_scene_contacts.npz", **contacts)
    np.savez_compressed(
        output / "scene_alignment.npz",
        schema_version=np.array(1, dtype=np.int32),
        units=np.array("meter", dtype="<U16"),
        source_coordinate_system=np.array("videomimic_gravity_calibrated", dtype="<U40"),
        intermediate_coordinate_system=np.array(args.source_coordinate_system, dtype="<U96"),
        target_coordinate_system=np.array("mujoco_world_z_up", dtype="<U24"),
        T_gvhmr_from_videomimic=t_gv_from_vm.astype(np.float32),
        T_mujoco_from_gvhmr=t_mujoco_from_gv.astype(np.float32),
        T_mujoco_from_videomimic=t_mujoco_from_vm.astype(np.float32),
        scale_videomimic_to_gvhmr=np.array(chosen_scale, dtype=np.float32),
        scale_sim3_diagnostic=np.array(scale, dtype=np.float32),
        human_yaw_offset_deg=np.array(0.0, dtype=np.float32),
        root_bridge_translation=np.zeros(3, dtype=np.float32),
        root_bridge_error_p90_m=np.array(-1.0, dtype=np.float32),
        root_bridge_available=np.array(False, dtype=bool),
        camera_mode=np.array("native", dtype="<U16"),
        reconstruction_backend=np.array("videomimic_nksr", dtype="<U32"),
    )

    sim_scale_ok = 0.90 <= scale <= 1.10
    p90 = float(np.quantile(residual, 0.90))
    failure_codes: list[str] = []
    if not sim_scale_ok:
        failure_codes.append("scale_mismatch")
    if p90 >= 0.05:
        failure_codes.append("human_adapter_mismatch")
    verdict = "pass" if not failure_codes else "fail"
    quality = {
        "schema_version": 1,
        "status": verdict,
        "failure_codes": failure_codes,
        "frame_count": frame_count,
        "joint_set": list(JOINT_NAMES),
        "transform_mode": transform_mode,
        "scale_sim3_diagnostic": scale,
        "scale_expected_range": [0.90, 1.10],
        "ransac_inlier_count": int(ransac_mask.sum()),
        "ransac_total_correspondences": int(len(ransac_mask)),
        "ransac_inlier_ratio": ransac_inlier_ratio,
        "ransac_consensus": ransac_consensus,
        "joint_alignment_error_m": {
            "mean": float(residual.mean()),
            "p50": float(np.quantile(residual, 0.50)),
            "p90": p90,
            "p99": float(np.quantile(residual, 0.99)),
            "required_p90_lt": 0.05,
            "by_joint_p90": {name: float(np.quantile(residual_by_joint[:, index], 0.90)) for index, name in enumerate(JOINT_NAMES)},
        },
        "gvhmr_to_mujoco_camera_bridge_p90_m": isaac_fit_p90,
        "camera_mode": args.camera_mode,
        "robot_root_bridge": "unavailable: human_pre run intentionally skipped PHC/GMR",
        "artifacts": {
            "scene_alignment": str(output / "scene_alignment.npz"),
            "human_scene_contacts": str(output / "human_scene_contacts.npz"),
        },
    }
    (output / "alignment_quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[{verdict.upper()}] Sim(3) scale={scale:.5f}; joint p90={p90:.4f}m; codes={failure_codes}")


if __name__ == "__main__":
    main()
