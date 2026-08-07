#!/usr/bin/env python3
"""Audit frozen GMR foot contact segments without modifying the motion."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mujoco
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runs(mask: np.ndarray, minimum_frames: int) -> List[np.ndarray]:
    runs: List[np.ndarray] = []
    start = None
    for index, active in enumerate(mask.tolist() + [False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            if index - start >= minimum_frames:
                runs.append(np.arange(start, index, dtype=np.int64))
            start = None
    return runs


def _summary(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-motion", required=True, type=Path)
    parser.add_argument("--robot-xml", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--left-foot-body", default="left_ankle_roll_link")
    parser.add_argument("--right-foot-body", default="right_ankle_roll_link")
    parser.add_argument("--ground-z-m", type=float, default=0.0)
    parser.add_argument("--min-contact-frames", type=int, default=3)
    parser.add_argument("--max-slip-p95-m", type=float, default=0.02)
    parser.add_argument("--max-stable-speed-m-s", type=float, default=0.08)
    parser.add_argument("--support-contact-height-m", type=float, default=0.015)
    parser.add_argument("--max-support-clearance-m", type=float, default=0.015)
    parser.add_argument("--max-support-penetration-m", type=float, default=0.005)
    parser.add_argument("--min-ground-skate-frames", type=int, default=3)
    parser.add_argument("--require-support-for-each-foot", action="store_true")
    args = parser.parse_args()
    if args.min_contact_frames < 2:
        parser.error("--min-contact-frames must be at least two")
    for option in (
        args.max_slip_p95_m,
        args.max_stable_speed_m_s,
        args.support_contact_height_m,
        args.max_support_clearance_m,
        args.max_support_penetration_m,
    ):
        if option < 0.0 or not np.isfinite(option):
            parser.error("quality thresholds must be finite and non-negative")
    if args.min_ground_skate_frames < 1:
        parser.error("--min-ground-skate-frames must be positive")
    return args


def _load_motion(path: Path) -> Dict[str, Any]:
    with path.open("rb") as stream:
        motion = pickle.load(stream)
    if not isinstance(motion, dict):
        raise ValueError("robot motion must be a dictionary")
    return motion


def _foot_meshes(
    model: mujoco.MjModel, body_name: str
) -> Tuple[int, List[Tuple[int, np.ndarray]]]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError("foot body is missing: " + body_name)
    meshes: List[Tuple[int, np.ndarray]] = []
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != body_id:
            continue
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        if int(model.geom_group[geom_id]) != 1:
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        meshes.append((geom_id, model.mesh_vert[start : start + count].copy()))
    if not meshes:
        raise ValueError("foot body has no visible mesh: " + body_name)
    return body_id, meshes


def _set_pose(
    data: mujoco.MjData,
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
) -> None:
    data.qpos[:3] = root_pos
    data.qpos[3:7] = root_rot_xyzw[[3, 0, 1, 2]]
    data.qpos[7:] = dof_pos


def _foot_state(
    data: mujoco.MjData,
    body_id: int,
    meshes: List[Tuple[int, np.ndarray]],
) -> Tuple[np.ndarray, float]:
    min_z = float("inf")
    for geom_id, vertices in meshes:
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        world = vertices @ rotation.T + data.geom_xpos[geom_id]
        min_z = min(min_z, float(np.min(world[:, 2])))
    return data.xpos[body_id, :2].copy(), min_z


def _segment_report(
    frames: np.ndarray,
    positions_xy: np.ndarray,
    clearances: np.ndarray,
    fps: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    origin = positions_xy[0]
    displacement = np.linalg.norm(positions_xy - origin, axis=1)
    if len(positions_xy) > 1:
        speed = np.linalg.norm(np.diff(positions_xy, axis=0), axis=1) * fps
    else:
        speed = np.zeros(1, dtype=np.float64)
    slip = _summary(displacement)
    clearance = _summary(clearances)
    reasons: List[str] = []
    if slip["p95"] > args.max_slip_p95_m:
        reasons.append("foot_slip")
    if clearance["maximum"] > args.max_support_clearance_m:
        reasons.append("foot_hover")
    if clearance["minimum"] < -args.max_support_penetration_m:
        reasons.append("foot_penetration")
    return {
        "frame_range": [int(frames[0]), int(frames[-1])],
        "frame_count": int(len(frames)),
        "duration_s": float((len(frames) - 1) / fps),
        "displacement_from_segment_start_m": slip,
        "horizontal_speed_m_s": _summary(speed),
        "sole_clearance_to_ground_m": clearance,
        "accepted": not reasons,
        "rejection_reasons": reasons,
    }


def _episode_report(
    frames: np.ndarray,
    positions_xy: np.ndarray,
    clearances: np.ndarray,
    speeds: np.ndarray,
    fps: float,
) -> Dict[str, Any]:
    displacement = np.linalg.norm(positions_xy - positions_xy[0], axis=1)
    return {
        "frame_range": [int(frames[0]), int(frames[-1])],
        "frame_count": int(len(frames)),
        "duration_s": float((len(frames) - 1) / fps),
        "displacement_from_episode_start_m": _summary(displacement),
        "horizontal_speed_m_s": _summary(speeds),
        "sole_clearance_to_ground_m": _summary(clearances),
    }


def main() -> None:
    args = _parse_args()
    motion_path = args.robot_motion.resolve()
    robot_xml = args.robot_xml.resolve()
    if not motion_path.is_file() or not robot_xml.is_file():
        raise FileNotFoundError("robot motion or robot XML is missing")

    motion = _load_motion(motion_path)
    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_rot = np.asarray(motion["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
    frames = len(root_pos)
    if root_pos.shape != (frames, 3) or root_rot.shape != (frames, 4):
        raise ValueError("root pose arrays have inconsistent shapes")
    if dof_pos.ndim != 2 or dof_pos.shape[0] != frames:
        raise ValueError("dof_pos does not match the root frame count")
    left_mask = np.asarray(motion.get("support_left_contact", []), dtype=bool)
    right_mask = np.asarray(motion.get("support_right_contact", []), dtype=bool)
    if left_mask.shape != (frames,) or right_mask.shape != (frames,):
        raise ValueError("motion lacks per-frame left/right support contact labels")
    fps_value = motion.get("fps")
    fps = float(fps_value) if fps_value is not None else 30.0
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("motion fps must be positive")

    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    if model.nq - 7 != dof_pos.shape[1]:
        raise ValueError("motion DoF count does not match the GMR robot XML")
    data = mujoco.MjData(model)
    left_body, left_meshes = _foot_meshes(model, args.left_foot_body)
    right_body, right_meshes = _foot_meshes(model, args.right_foot_body)
    xy = np.zeros((frames, 2, 2), dtype=np.float64)
    clearance = np.zeros((frames, 2), dtype=np.float64)
    for frame in range(frames):
        _set_pose(data, root_pos[frame], root_rot[frame], dof_pos[frame])
        mujoco.mj_forward(model, data)
        xy[frame, 0], clearance[frame, 0] = _foot_state(data, left_body, left_meshes)
        xy[frame, 1], clearance[frame, 1] = _foot_state(data, right_body, right_meshes)
    clearance -= args.ground_z_m

    speed = np.zeros((frames, 2), dtype=np.float64)
    if frames > 1:
        speed[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=2) * fps
        speed[:-1] = np.maximum(speed[:-1], speed[1:])

    side_data = (("left", left_mask, 0), ("right", right_mask, 1))
    reports: Dict[str, Any] = {}
    all_reasons: List[str] = []
    for side, mask, index in side_data:
        near_ground = (
            (clearance[:, index] >= -args.max_support_penetration_m)
            & (clearance[:, index] <= args.support_contact_height_m)
        )
        stable = mask & near_ground & (speed[:, index] <= args.max_stable_speed_m_s)
        skating = mask & near_ground & (speed[:, index] > args.max_stable_speed_m_s)
        # This is deliberately independent of the support label.  A candidate
        # must not pass merely by relabeling a visibly ground-skating foot as
        # flight after it has been moved close to the floor.
        near_ground_motion = near_ground & (speed[:, index] > args.max_stable_speed_m_s)
        hovering = mask & (clearance[:, index] > args.max_support_clearance_m)
        penetrating = mask & (clearance[:, index] < -args.max_support_penetration_m)
        segments = [
            _segment_report(segment, xy[segment, index], clearance[segment, index], fps, args)
            for segment in _runs(stable, args.min_contact_frames)
        ]
        skate_episodes = [
            _episode_report(
                episode,
                xy[episode, index],
                clearance[episode, index],
                speed[episode, index],
                fps,
            )
            for episode in _runs(skating, args.min_ground_skate_frames)
        ]
        near_ground_motion_episodes = [
            _episode_report(
                episode,
                xy[episode, index],
                clearance[episode, index],
                speed[episode, index],
                fps,
            )
            for episode in _runs(near_ground_motion, args.min_ground_skate_frames)
        ]
        reasons = sorted({reason for segment in segments for reason in segment["rejection_reasons"]})
        if skate_episodes:
            reasons.append("foot_ground_skate")
        if near_ground_motion_episodes:
            reasons.append("near_ground_foot_motion")
        if np.any(hovering):
            reasons.append("foot_hover")
        if np.any(penetrating):
            reasons.append("foot_penetration")
        if args.require_support_for_each_foot and not segments:
            reasons.append("no_stable_support_segment")
        reasons = sorted(set(reasons))
        reports[side] = {
            "raw_support_frame_count": int(np.count_nonzero(mask)),
            "near_ground_support_frame_count": int(np.count_nonzero(mask & near_ground)),
            "stable_support_frame_count": int(np.count_nonzero(stable)),
            "ground_skate_frame_count": int(np.count_nonzero(skating)),
            "near_ground_motion_frame_count": int(np.count_nonzero(near_ground_motion)),
            "hover_frame_count": int(np.count_nonzero(hovering)),
            "penetration_frame_count": int(np.count_nonzero(penetrating)),
            "eligible_segment_count": int(len(segments)),
            "segments": segments,
            "ground_skate_episodes": skate_episodes,
            "near_ground_motion_episodes": near_ground_motion_episodes,
            "accepted": not reasons,
            "rejection_reasons": reasons,
        }
        all_reasons.extend(side + ":" + reason for reason in reasons)

    report = {
        "schema_version": 1,
        "purpose": "frozen_gmr_foot_support_audit",
        "robot_motion": str(motion_path),
        "robot_motion_sha256": _sha256(motion_path),
        "robot_motion_modified": False,
        "robot_xml": str(robot_xml),
        "frame_count": int(frames),
        "fps": fps,
        "ground_z_m": float(args.ground_z_m),
        "quality_gate": {
            "max_slip_p95_m": args.max_slip_p95_m,
            "max_stable_speed_m_s": args.max_stable_speed_m_s,
            "support_contact_height_m": args.support_contact_height_m,
            "max_support_clearance_m": args.max_support_clearance_m,
            "max_support_penetration_m": args.max_support_penetration_m,
            "min_contact_frames": args.min_contact_frames,
            "min_ground_skate_frames": args.min_ground_skate_frames,
        },
        "feet": reports,
        "accepted": not all_reasons,
        "rejection_reasons": sorted(all_reasons),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"accepted": report["accepted"], "rejection_reasons": report["rejection_reasons"]}))


if __name__ == "__main__":
    main()
