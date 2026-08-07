#!/usr/bin/env python3
"""Refine GVHMR SMPL contact windows from 2-D evidence without training.

The optimizer is deliberately an inference-time post-process.  It keeps shape,
global orientation, hands, camera, and every non-leg body joint fixed.  Only
root translation and the hip/knee/ankle rotations of visually planted feet are
allowed to move, and every proposed result is rejected when it regresses the
held-out full-body reprojection or exceeds temporal/motion bounds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import smplx
import torch
import torch.nn.functional as F
import mujoco


SIDES = ("left", "right")
COCO_BODY_MAP = (
    (5, "left_shoulder"),
    (6, "right_shoulder"),
    (7, "left_elbow"),
    (8, "right_elbow"),
    (9, "left_wrist"),
    (10, "right_wrist"),
    (11, "left_hip"),
    (12, "right_hip"),
    (13, "left_knee"),
    (14, "right_knee"),
    (15, "left_ankle"),
    (16, "right_ankle"),
)
FOOT_KEYPOINTS = {
    "left": (15, 17, 18, 19),
    "right": (16, 20, 21, 22),
}
LEG_POSE_ROWS = {
    "left": (0, 3, 6),
    "right": (1, 4, 7),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_save_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.stem}.", suffix=".npz", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **values)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _runs(mask: np.ndarray, minimum: int) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    first: int | None = None
    for index, enabled in enumerate(mask.tolist() + [False]):
        if enabled and first is None:
            first = index
        elif not enabled and first is not None:
            if index - first >= minimum:
                output.append(np.arange(first, index, dtype=np.int64))
            first = None
    return output


def _merge_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    for first, last in sorted(windows):
        if not output or first > output[-1][1] + 1:
            output.append((first, last))
        else:
            output[-1] = (output[-1][0], max(output[-1][1], last))
    return output


def _load_keypoints(path: Path) -> np.ndarray:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise ValueError("vitpose_wholebody input is not a tensor")
    array = value.detach().cpu().numpy()
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[1:] != (133, 3):
        raise ValueError(f"unexpected VitPose shape: {array.shape}")
    return np.asarray(array, dtype=np.float32)


def _normal_gender(value: np.ndarray) -> str:
    gender = str(np.asarray(value).reshape(-1)[0]).lower()
    return gender if gender in {"female", "male", "neutral"} else "neutral"


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values):
        return {"count": 0, "median": 0.0, "p95": 0.0, "maximum": 0.0}
    return {
        "count": int(len(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(np.max(values)),
    }


def _torch_stats(values: torch.Tensor) -> dict[str, float]:
    return _stats(values.detach().cpu().numpy())


def _gmr_near_ground(
    motion_path: Path,
    robot_xml: Path,
    frames: int,
    contact_height_m: float,
) -> np.ndarray:
    """Return feet whose visible GMR mesh is inside the ground contact band."""
    with motion_path.open("rb") as stream:
        motion = pickle.load(stream)
    if not isinstance(motion, dict):
        raise ValueError("GMR contact-evidence motion is not a dictionary")
    root = np.asarray(motion["root_pos"], dtype=np.float64)
    rotation = np.asarray(motion["root_rot"], dtype=np.float64)
    dof = np.asarray(motion["dof_pos"], dtype=np.float64)
    if root.shape != (frames, 3) or rotation.shape != (frames, 4) or dof.shape[0] != frames:
        raise ValueError("GMR contact-evidence motion does not match SMPL frame count")
    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    if model.nq - 7 != dof.shape[1]:
        raise ValueError("GMR contact-evidence motion does not match robot XML")
    feet: dict[str, list[tuple[int, np.ndarray]]] = {}
    for side in SIDES:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link")
        if body < 0:
            raise ValueError(f"robot XML has no {side} foot body")
        meshes: list[tuple[int, np.ndarray]] = []
        for geom in range(model.ngeom):
            if (
                int(model.geom_bodyid[geom]) == body
                and int(model.geom_type[geom]) == int(mujoco.mjtGeom.mjGEOM_MESH)
                and int(model.geom_group[geom]) == 1
            ):
                mesh = int(model.geom_dataid[geom])
                start = int(model.mesh_vertadr[mesh])
                count = int(model.mesh_vertnum[mesh])
                meshes.append((geom, model.mesh_vert[start : start + count].copy()))
        if not meshes:
            raise ValueError(f"robot XML has no visible mesh for {side} foot")
        feet[side] = meshes
    heights = np.empty((frames, 2), dtype=np.float64)
    data = mujoco.MjData(model)
    for frame in range(frames):
        data.qpos[:3] = root[frame]
        data.qpos[3:7] = rotation[frame][[3, 0, 1, 2]]
        data.qpos[7:] = dof[frame]
        mujoco.mj_forward(model, data)
        for side_index, side in enumerate(SIDES):
            low = float("inf")
            for geom, vertices in feet[side]:
                matrix = data.geom_xmat[geom].reshape(3, 3)
                low = min(low, float(np.min((vertices @ matrix.T + data.geom_xpos[geom])[:, 2])))
            heights[frame, side_index] = low
    return (heights >= -0.005) & (heights <= contact_height_m)


class SMPLProjector:
    def __init__(
        self,
        values: dict[str, np.ndarray],
        camera: dict[str, np.ndarray],
        model_root: Path,
        device: torch.device,
    ) -> None:
        required = {"trans", "root_orient", "pose_body", "betas", "gender"}
        missing = required - values.keys()
        if missing:
            raise KeyError(f"motion is missing {sorted(missing)}")
        self.frames = int(len(values["trans"]))
        self.device = device
        self.root_base = torch.as_tensor(values["trans"], dtype=torch.float32, device=device)
        self.orient = torch.as_tensor(values["root_orient"], dtype=torch.float32, device=device)
        self.pose_base = torch.as_tensor(values["pose_body"], dtype=torch.float32, device=device).reshape(self.frames, 21, 3)
        if self.root_base.shape != (self.frames, 3) or self.orient.shape != (self.frames, 3) or self.pose_base.shape != (self.frames, 21, 3):
            raise ValueError("incompatible SMPL motion array shapes")
        if "T_w2c" not in camera or "K_fullimg" not in camera:
            raise KeyError("GVHMR camera is missing T_w2c or K_fullimg")
        self.w2c = torch.as_tensor(camera["T_w2c"], dtype=torch.float32, device=device)
        self.intrinsics = torch.as_tensor(camera["K_fullimg"], dtype=torch.float32, device=device)
        if self.w2c.shape != (self.frames, 4, 4) or self.intrinsics.shape != (self.frames, 3, 3):
            raise ValueError("camera frame count does not match SMPL motion")
        betas = np.asarray(values["betas"], dtype=np.float32).reshape(-1)
        requested_gender = _normal_gender(values["gender"])
        if (model_root / "smpl").is_dir():
            model_type = "smpl"
            extension = "pkl"
            max_betas = 10
        elif (model_root / "smplh" / "SMPLH_NEUTRAL.npz").is_file():
            # The released GMR asset pack contains SMPL-H.  Its body joints
            # share SMPL's first 24-joint ordering, while hands remain fixed
            # at zero and are never part of this contact optimizer.
            model_type = "smplh"
            extension = "npz"
            max_betas = 16
        else:
            raise FileNotFoundError(
                "body-model root must contain SMPL or SMPL-H assets: " + str(model_root)
            )
        self.model_type = model_type
        gender = requested_gender
        if model_type == "smplh" and not (model_root / "smplh" / f"SMPLH_{gender.upper()}.npz").is_file():
            gender = "neutral"
        self.model = smplx.create(
            str(model_root),
            model_type=model_type,
            gender=gender,
            ext=extension,
            num_betas=min(max_betas, len(betas)),
            use_pca=False,
        ).to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.betas = torch.as_tensor(betas[: self.model.num_betas], dtype=torch.float32, device=device).unsqueeze(0)
        # SMPL's first 24 joints follow this standard ordering.  The model may
        # expose additional regression joints, but the contact terms only rely
        # on the stable ankle/foot pair.
        self.joint_ids = {
            "left_hip": 1,
            "right_hip": 2,
            "left_knee": 4,
            "right_knee": 5,
            "left_ankle": 7,
            "right_ankle": 8,
            "left_foot": 10,
            "right_foot": 11,
            "left_shoulder": 16,
            "right_shoulder": 17,
            "left_elbow": 18,
            "right_elbow": 19,
            "left_wrist": 20,
            "right_wrist": 21,
        }

    def forward(
        self,
        frame_ids: torch.Tensor,
        root_delta: torch.Tensor | None = None,
        leg_deltas: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        count = int(len(frame_ids))
        root = self.root_base[frame_ids]
        pose = self.pose_base[frame_ids]
        if root_delta is not None:
            root = root + root_delta
        if leg_deltas:
            pose = pose.clone()
            for side, delta in leg_deltas.items():
                pose[:, list(LEG_POSE_ROWS[side]), :] += delta
        arguments: dict[str, torch.Tensor] = {
            "betas": self.betas.expand(count, -1),
            "global_orient": self.orient[frame_ids],
            "body_pose": pose.reshape(count, -1),
            "transl": root,
        }
        if self.model_type == "smplh":
            arguments["left_hand_pose"] = torch.zeros((count, 45), dtype=torch.float32, device=self.device)
            arguments["right_hand_pose"] = torch.zeros((count, 45), dtype=torch.float32, device=self.device)
        output = self.model(
            **arguments,
        )
        if output.joints.shape[1] < 22:
            raise RuntimeError("SMPL model did not expose the required body joints")
        return output.joints

    def project(self, frame_ids: torch.Tensor, joints: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rotation = self.w2c[frame_ids, :3, :3]
        translation = self.w2c[frame_ids, :3, 3]
        camera = torch.einsum("fij,fkj->fki", rotation, joints) + translation[:, None, :]
        homogeneous = torch.einsum("fij,fkj->fki", self.intrinsics[frame_ids], camera)
        depth = homogeneous[..., 2]
        pixels = homogeneous[..., :2] / depth.clamp_min(1e-4).unsqueeze(-1)
        return pixels, depth


def _projection_errors(
    projector: SMPLProjector,
    joints: torch.Tensor,
    keypoints: torch.Tensor,
    confidence: float,
    holdout: bool,
) -> dict[str, float]:
    frame_ids = torch.arange(projector.frames, device=projector.device)
    pixels, depth = projector.project(frame_ids, joints)
    errors: list[torch.Tensor] = []
    for keypoint, name in COCO_BODY_MAP:
        if holdout and keypoint in FOOT_KEYPOINTS["left"] + FOOT_KEYPOINTS["right"]:
            continue
        joint = projector.joint_ids[name]
        observed = keypoints[:, keypoint, :2]
        valid = (keypoints[:, keypoint, 2] >= confidence) & (depth[:, joint] > 1e-4)
        if torch.any(valid):
            errors.append(torch.linalg.norm(pixels[valid, joint] - observed[valid], dim=1))
    if not errors:
        return _stats(np.empty(0))
    return _torch_stats(torch.cat(errors))


def _clip_root(delta: torch.Tensor, maximum: float) -> None:
    with torch.no_grad():
        norm = torch.linalg.norm(delta, dim=1, keepdim=True).clamp_min(1e-12)
        delta.mul_(torch.clamp(maximum / norm, max=1.0))


def _clip_leg(delta: torch.Tensor, maximum: float) -> None:
    with torch.no_grad():
        norm = torch.linalg.norm(delta, dim=2, keepdim=True).clamp_min(1e-12)
        delta.mul_(torch.clamp(maximum / norm, max=1.0))


def _motion_metrics(
    root_delta: np.ndarray,
    pose_delta: np.ndarray,
    fps: float,
) -> dict[str, float]:
    root_speed = np.linalg.norm(np.diff(root_delta, axis=0), axis=1) * fps
    root_acc = np.linalg.norm(np.diff(root_delta, n=2, axis=0), axis=1) * fps * fps
    leg = pose_delta[:, [*LEG_POSE_ROWS["left"], *LEG_POSE_ROWS["right"]], :]
    leg_speed = np.linalg.norm(np.diff(leg, axis=0), axis=2) * fps
    leg_acc = np.linalg.norm(np.diff(leg, n=2, axis=0), axis=2) * fps * fps
    return {
        "root_translation_delta_m": _stats(np.linalg.norm(root_delta, axis=1)),
        "leg_joint_delta_rad": _stats(np.linalg.norm(leg, axis=2)),
        "root_delta_speed_m_s": _stats(root_speed),
        "root_delta_acceleration_m_s2": _stats(root_acc),
        "leg_delta_speed_rad_s": _stats(leg_speed),
        "leg_delta_acceleration_rad_s2": _stats(leg_acc),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-motion", required=True, type=Path)
    parser.add_argument("--camera", required=True, type=Path)
    parser.add_argument("--vitpose", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--gmr-motion", required=True, type=Path)
    parser.add_argument("--robot-xml", required=True, type=Path)
    parser.add_argument("--body-model-root", required=True, type=Path)
    parser.add_argument("--output-motion", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--minimum-episode-frames", type=int, default=5)
    parser.add_argument("--window-padding-frames", type=int, default=3)
    parser.add_argument("--min-keypoint-confidence", type=float, default=0.45)
    parser.add_argument("--gmr-contact-height-m", type=float, default=0.015)
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--temporal-weight", type=float, default=0.80)
    parser.add_argument("--second-difference-weight", type=float, default=0.25)
    parser.add_argument("--max-root-translation-delta-m", type=float, default=0.06)
    parser.add_argument("--max-leg-joint-delta-rad", type=float, default=0.20)
    parser.add_argument("--max-anchor-error-m", type=float, default=0.025)
    parser.add_argument("--max-source-anchor-drift-m", type=float, default=0.35)
    parser.add_argument("--max-holdout-p95-regression-px", type=float, default=1.5)
    parser.add_argument("--max-all-p95-regression-px", type=float, default=4.0)
    parser.add_argument("--max-root-delta-speed-m-s", type=float, default=0.80)
    parser.add_argument("--max-root-delta-acceleration-m-s2", type=float, default=12.0)
    parser.add_argument("--max-leg-delta-speed-rad-s", type=float, default=5.0)
    parser.add_argument("--max-leg-delta-acceleration-rad-s2", type=float, default=100.0)
    args = parser.parse_args()
    if args.minimum_episode_frames < 3 or args.window_padding_frames < 0 or args.iterations < 1:
        raise ValueError("episode/window/iteration settings are invalid")
    if min(args.max_root_translation_delta_m, args.max_leg_joint_delta_rad, args.max_anchor_error_m, args.max_source_anchor_drift_m, args.learning_rate, args.gmr_contact_height_m, args.temporal_weight) <= 0.0 or args.second_difference_weight < 0.0:
        raise ValueError("optimization bounds and learning rate must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for SMPL refinement but is unavailable")
    device = torch.device(args.device)
    with np.load(args.input_motion, allow_pickle=False) as archive:
        values = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(args.camera, allow_pickle=False) as archive:
        camera = {key: np.asarray(archive[key]) for key in archive.files}
    keypoints_np = _load_keypoints(args.vitpose)
    with np.load(args.evidence, allow_pickle=False) as archive:
        phases = {side: np.asarray(archive[f"{side}_phase"], dtype=np.int8) for side in SIDES}
    frames = int(len(values["trans"]))
    if len(keypoints_np) != frames or any(value.shape != (frames,) for value in phases.values()):
        raise ValueError("motion, VitPose, and visual contact evidence must have identical frame counts")
    if not args.body_model_root.is_dir():
        raise FileNotFoundError(args.body_model_root)
    if not args.gmr_motion.is_file() or not args.robot_xml.is_file():
        raise FileNotFoundError("GMR contact-evidence motion or robot XML is missing")
    fps = float(np.asarray(values.get("mocap_frame_rate", 30.0)).reshape(-1)[0])
    if fps <= 0.0:
        raise ValueError("motion frame rate must be positive")
    projector = SMPLProjector(values, camera, args.body_model_root, device)
    keypoints = torch.as_tensor(keypoints_np, dtype=torch.float32, device=device)
    all_ids = torch.arange(frames, device=device)
    with torch.no_grad():
        original_joints = projector.forward(all_ids)
    gmr_near_ground = _gmr_near_ground(
        args.gmr_motion, args.robot_xml, frames, args.gmr_contact_height_m
    )
    anchors = np.full((frames, 2, 2, 3), np.nan, dtype=np.float32)
    support = np.zeros((frames, 2), dtype=bool)
    episodes: list[dict[str, Any]] = []
    ignored_episodes: list[dict[str, Any]] = []
    windows: list[tuple[int, int]] = []
    for side_index, side in enumerate(SIDES):
        eligible = (phases[side] == 1) & gmr_near_ground[:, side_index]
        for run in _runs(eligible, args.minimum_episode_frames):
            first, last = int(run[0]), int(run[-1])
            joint_ids = [projector.joint_ids[f"{side}_ankle"], projector.joint_ids[f"{side}_foot"]]
            source_displacement = torch.linalg.norm(
                original_joints[run][:, joint_ids] - original_joints[first, joint_ids],
                dim=2,
            ).detach().cpu().numpy()
            source_stats = _stats(source_displacement)
            if source_stats["p95"] > args.max_source_anchor_drift_m:
                ignored_episodes.append({
                    "side": side,
                    "frame_range": [first, last],
                    "reason": "source_anchor_drift_exceeds_recoverable_limit",
                    "source_anchor_drift_m": source_stats,
                })
                continue
            target = original_joints[first, joint_ids].detach().cpu().numpy()
            anchors[run, side_index] = target
            support[run, side_index] = True
            windows.append((max(0, first - args.window_padding_frames), min(frames - 1, last + args.window_padding_frames)))
            episodes.append({
                "side": side,
                "frame_range": [first, last],
                "anchor_frame": first,
                "anchor_joint_names": [f"{side}_ankle", f"{side}_foot"],
                "source_anchor_drift_m": source_stats,
            })
    windows = _merge_windows(windows)
    root_delta_total = np.zeros((frames, 3), dtype=np.float32)
    pose_delta_total = np.zeros((frames, 21, 3), dtype=np.float32)
    window_reports: list[dict[str, Any]] = []
    anchor_errors: list[np.ndarray] = []
    for first, last in windows:
        ids_np = np.arange(first, last + 1, dtype=np.int64)
        ids = torch.as_tensor(ids_np, dtype=torch.long, device=device)
        active_sides = [side for side_index, side in enumerate(SIDES) if np.any(support[ids_np, side_index])]
        root_delta = torch.zeros((len(ids), 3), dtype=torch.float32, device=device, requires_grad=True)
        leg_deltas = {
            side: torch.zeros((len(ids), 3, 3), dtype=torch.float32, device=device, requires_grad=True)
            for side in active_sides
        }
        optimizer = torch.optim.Adam([root_delta, *leg_deltas.values()], lr=args.learning_rate)
        anchor_tensor = torch.as_tensor(anchors[ids_np], dtype=torch.float32, device=device)
        support_tensor = torch.as_tensor(support[ids_np], dtype=torch.bool, device=device)
        initial_loss = None
        for iteration in range(args.iterations):
            optimizer.zero_grad(set_to_none=True)
            joints = projector.forward(ids, root_delta, leg_deltas)
            pixels, depth = projector.project(ids, joints)
            reprojection_terms: list[torch.Tensor] = []
            for keypoint, name in COCO_BODY_MAP:
                joint = projector.joint_ids[name]
                valid = (keypoints[ids, keypoint, 2] >= args.min_keypoint_confidence) & (depth[:, joint] > 1e-4)
                if torch.any(valid):
                    difference = (pixels[valid, joint] - keypoints[ids[valid], keypoint, :2]) / 12.0
                    reprojection_terms.append(F.smooth_l1_loss(difference, torch.zeros_like(difference), reduction="mean"))
            reprojection = torch.stack(reprojection_terms).mean() if reprojection_terms else torch.zeros((), device=device)
            contact_terms: list[torch.Tensor] = []
            for side_index, side in enumerate(SIDES):
                active = support_tensor[:, side_index]
                if not torch.any(active):
                    continue
                joint_ids = [projector.joint_ids[f"{side}_ankle"], projector.joint_ids[f"{side}_foot"]]
                difference = (joints[active][:, joint_ids] - anchor_tensor[active, side_index]) / 0.012
                contact_terms.append(torch.mean(difference.square()))
            contact = torch.stack(contact_terms).mean() if contact_terms else torch.zeros((), device=device)
            root_prior = torch.mean((root_delta / 0.025).square())
            leg_prior = torch.stack([torch.mean((value / 0.10).square()) for value in leg_deltas.values()]).mean()
            temporal = torch.mean((torch.diff(root_delta, dim=0) / 0.015).square()) if len(ids) > 1 else torch.zeros((), device=device)
            acceleration = torch.mean((torch.diff(root_delta, n=2, dim=0) / 0.020).square()) if len(ids) > 2 else torch.zeros((), device=device)
            # A true support run may reach either video boundary.  There is no
            # adjacent source frame outside that boundary, so forcing the
            # correction back to zero there creates an artificial velocity
            # spike.  Only padded interior window edges are anchored to zero.
            boundary_ids: list[int] = []
            if first > 0:
                boundary_ids.append(0)
            if last < frames - 1:
                boundary_ids.append(-1)
            boundary = (
                torch.mean((root_delta[boundary_ids] / 0.010).square())
                if boundary_ids
                else torch.zeros((), device=device)
            )
            for value in leg_deltas.values():
                if len(ids) > 1:
                    temporal = temporal + 0.35 * torch.mean((torch.diff(value, dim=0) / 0.06).square())
                if len(ids) > 2:
                    acceleration = acceleration + 0.15 * torch.mean((torch.diff(value, n=2, dim=0) / 0.08).square())
                if boundary_ids:
                    boundary = boundary + 0.30 * torch.mean((value[boundary_ids] / 0.025).square())
            loss = 1.0 * reprojection + 7.5 * contact + 0.35 * root_prior + 0.18 * leg_prior + args.temporal_weight * temporal + args.second_difference_weight * acceleration + 0.60 * boundary
            if initial_loss is None:
                initial_loss = float(loss.detach().cpu())
            loss.backward()
            optimizer.step()
            _clip_root(root_delta, args.max_root_translation_delta_m)
            for value in leg_deltas.values():
                _clip_leg(value, args.max_leg_joint_delta_rad)
        with torch.no_grad():
            joints = projector.forward(ids, root_delta, leg_deltas)
            for side_index, side in enumerate(SIDES):
                active = support_tensor[:, side_index]
                if not torch.any(active):
                    continue
                joint_ids = [projector.joint_ids[f"{side}_ankle"], projector.joint_ids[f"{side}_foot"]]
                error = torch.linalg.norm(joints[active][:, joint_ids] - anchor_tensor[active, side_index], dim=2)
                anchor_errors.append(error.detach().cpu().numpy())
            root_delta_total[ids_np] = root_delta.detach().cpu().numpy()
            for side, value in leg_deltas.items():
                pose_delta_total[ids_np[:, None], list(LEG_POSE_ROWS[side])] = value.detach().cpu().numpy()
            window_reports.append({
                "frame_range": [first, last],
                "active_sides": active_sides,
                "initial_loss": initial_loss,
                "final_loss": float(loss.detach().cpu()),
                "root_translation_delta_m": _stats(torch.linalg.norm(root_delta, dim=1).detach().cpu().numpy()),
                "leg_joint_delta_rad": {side: _stats(torch.linalg.norm(value, dim=2).detach().cpu().numpy()) for side, value in leg_deltas.items()},
            })
    with torch.no_grad():
        root_delta_tensor = torch.as_tensor(root_delta_total, dtype=torch.float32, device=device)
        leg_delta_tensor = {
            side: torch.as_tensor(pose_delta_total[:, list(LEG_POSE_ROWS[side])], dtype=torch.float32, device=device)
            for side in SIDES
        }
        candidate_joints = projector.forward(all_ids, root_delta_tensor, leg_delta_tensor)
    baseline_all = _projection_errors(projector, original_joints, keypoints, args.min_keypoint_confidence, holdout=False)
    candidate_all = _projection_errors(projector, candidate_joints, keypoints, args.min_keypoint_confidence, holdout=False)
    baseline_holdout = _projection_errors(projector, original_joints, keypoints, args.min_keypoint_confidence, holdout=True)
    candidate_holdout = _projection_errors(projector, candidate_joints, keypoints, args.min_keypoint_confidence, holdout=True)
    anchors_error = np.concatenate(anchor_errors) if anchor_errors else np.empty(0)
    metrics = _motion_metrics(root_delta_total, pose_delta_total, fps)
    # Derive the untouched pose rows from the exact same definition used when
    # applying the correction.  This makes the preservation guarantee a real
    # acceptance condition rather than a duplicated, fallible index list.
    leg_pose_rows = sorted({row for rows in LEG_POSE_ROWS.values() for row in rows})
    non_leg_pose_rows = [
        row for row in range(pose_delta_total.shape[1]) if row not in leg_pose_rows
    ]
    non_leg_delta_max_rad = float(
        np.max(np.abs(pose_delta_total[:, non_leg_pose_rows, :]))
    ) if non_leg_pose_rows else 0.0
    failures: list[str] = []
    if not episodes:
        failures.append("no_visual_planted_contact_episode")
    if not len(anchors_error) or float(np.percentile(anchors_error, 95)) > args.max_anchor_error_m:
        failures.append("support_anchor_error_exceeds_limit")
    if candidate_holdout["p95"] > baseline_holdout["p95"] + args.max_holdout_p95_regression_px:
        failures.append("holdout_reprojection_regression")
    if candidate_all["p95"] > baseline_all["p95"] + args.max_all_p95_regression_px:
        failures.append("all_keypoint_reprojection_regression")
    if metrics["root_translation_delta_m"]["maximum"] > args.max_root_translation_delta_m + 1e-6:
        failures.append("root_translation_bound_exceeded")
    if metrics["leg_joint_delta_rad"]["maximum"] > args.max_leg_joint_delta_rad + 1e-6:
        failures.append("leg_joint_bound_exceeded")
    if metrics["root_delta_speed_m_s"]["maximum"] > args.max_root_delta_speed_m_s:
        failures.append("root_delta_speed_exceeded")
    if metrics["root_delta_acceleration_m_s2"]["maximum"] > args.max_root_delta_acceleration_m_s2:
        failures.append("root_delta_acceleration_exceeded")
    if metrics["leg_delta_speed_rad_s"]["maximum"] > args.max_leg_delta_speed_rad_s:
        failures.append("leg_delta_speed_exceeded")
    if metrics["leg_delta_acceleration_rad_s2"]["maximum"] > args.max_leg_delta_acceleration_rad_s2:
        failures.append("leg_delta_acceleration_exceeded")
    if non_leg_delta_max_rad > 1e-7:
        failures.append("non_leg_body_pose_changed")
    accepted = not failures
    result = dict(values)
    if accepted:
        root = np.asarray(values["trans"], dtype=np.float32) + root_delta_total
        pose = np.asarray(values["pose_body"], dtype=np.float32).reshape(frames, 21, 3) + pose_delta_total
        result["trans"] = root.astype(np.asarray(values["trans"]).dtype, copy=False)
        result["pose_body"] = pose.reshape(frames, 63).astype(np.asarray(values["pose_body"]).dtype, copy=False)
        if "poses" in result:
            poses = np.asarray(result["poses"]).copy()
            if poses.shape == (frames, 24, 3):
                poses[:, 0] = result["root_orient"]
                poses[:, 1:22] = pose
            else:
                failures.append("poses_array_shape_incompatible")
                accepted = False
                result = dict(values)
            result["poses"] = poses.astype(np.asarray(values["poses"]).dtype, copy=False)
    _atomic_save_npz(args.output_motion, result)
    report = {
        "schema_version": 1,
        "kind": "gvhmr_smpl_2d_contact_window_refinement",
        "weights_modified": False,
        "input_motion": str(args.input_motion),
        "input_motion_sha256": _sha256(args.input_motion),
        "camera": str(args.camera),
        "camera_sha256": _sha256(args.camera),
        "vitpose": str(args.vitpose),
        "vitpose_sha256": _sha256(args.vitpose),
        "evidence": str(args.evidence),
        "evidence_sha256": _sha256(args.evidence),
        "gmr_contact_evidence_motion": str(args.gmr_motion),
        "gmr_contact_evidence_motion_sha256": _sha256(args.gmr_motion),
        "robot_xml": str(args.robot_xml),
        "gmr_contact_height_m": args.gmr_contact_height_m,
        "gmr_near_ground_frame_count": {
            side: int(np.count_nonzero(gmr_near_ground[:, index]))
            for index, side in enumerate(SIDES)
        },
        "visual_and_gmr_eligible_frame_count": {
            side: int(np.count_nonzero((phases[side] == 1) & gmr_near_ground[:, index]))
            for index, side in enumerate(SIDES)
        },
        "output_motion": str(args.output_motion),
        "output_motion_sha256": _sha256(args.output_motion),
        "frame_count": frames,
        "fps": fps,
        "global_orientation_unchanged": True,
        "shape_unchanged": True,
        "non_leg_body_pose_unchanged": bool(non_leg_delta_max_rad <= 1e-7),
        "non_leg_body_pose_delta_max_rad": non_leg_delta_max_rad,
        "episodes": episodes,
        "ignored_episodes": ignored_episodes,
        "windows": window_reports,
        "anchor_error_m": _stats(anchors_error),
        "reprojection_error_px": {
            "all_before": baseline_all,
            "all_after": candidate_all,
            "holdout_before": baseline_holdout,
            "holdout_after": candidate_holdout,
        },
        "motion_delta": metrics,
        "limits": {
            "max_root_translation_delta_m": args.max_root_translation_delta_m,
            "max_leg_joint_delta_rad": args.max_leg_joint_delta_rad,
            "max_anchor_error_m": args.max_anchor_error_m,
            "max_source_anchor_drift_m": args.max_source_anchor_drift_m,
            "max_holdout_p95_regression_px": args.max_holdout_p95_regression_px,
            "max_all_p95_regression_px": args.max_all_p95_regression_px,
            "max_root_delta_speed_m_s": args.max_root_delta_speed_m_s,
            "max_root_delta_acceleration_m_s2": args.max_root_delta_acceleration_m_s2,
            "max_leg_delta_speed_rad_s": args.max_leg_delta_speed_rad_s,
            "max_leg_delta_acceleration_rad_s2": args.max_leg_delta_acceleration_rad_s2,
            "temporal_weight": args.temporal_weight,
            "second_difference_weight": args.second_difference_weight,
        },
        "failures": sorted(set(failures)),
        "accepted": accepted,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "accepted": accepted,
        "episodes": len(episodes),
        "windows": len(windows),
        "anchor_error_m": report["anchor_error_m"],
        "failures": report["failures"],
    }))


if __name__ == "__main__":
    main()
