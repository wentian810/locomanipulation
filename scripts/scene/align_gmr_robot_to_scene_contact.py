#!/usr/bin/env python3
"""Apply a bounded, contact-evidenced GMR root translation toward a chair backrest.

This is intentionally a robot-side morphology compensation.  It does not move the
VideoMimic/NKSR scene, alter the PHC motion, or overwrite its input trajectory.
The required translation is estimated from MuJoCo torso collision geoms on frames
where the source SMPL body has a verified backrest contact.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _as_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


def _load_primitives(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = _load_json(path)
    primitives = payload.get("primitives", [])
    by_name = {entry.get("name"): entry for entry in primitives}
    seat = by_name.get("seat") or by_name.get("seat_support")
    backrest = by_name.get("backrest")
    if seat is None or backrest is None:
        raise ValueError(f"{path} must contain visual seat and backrest primitives")

    seat_center = _as_array(seat["center"], (3,), "seat.center")
    rotation = _as_array(seat.get("rotation_matrix", seat.get("rotation")),
                         (3, 3), "seat.rotation")
    back_center = _as_array(backrest["center"], (3,), "backrest.center")
    back_extents = _as_array(backrest["extents"], (3,), "backrest.extents")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("chair rotation is not orthonormal")
    if back_extents[1] <= 0:
        raise ValueError("backrest local thickness must be positive")
    return seat_center, rotation, back_center, back_extents


def _chair_back_frame(rotation: np.ndarray, seat_center: np.ndarray, back_center: np.ndarray,
                      back_extents: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return world direction toward the backrest, its front-face local y and seat rear y."""
    center_local = (back_center - seat_center) @ rotation
    sign = 1.0 if center_local[1] >= 0.0 else -1.0
    direction = rotation[:, 1] * sign
    front_face_y = float(center_local[1] - sign * back_extents[1] * 0.5)
    # nearest seat edge on the backrest side; seat extents are not needed for shift itself.
    return direction, front_face_y, sign


def _load_contact_anchors(path: Path, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    anchors = np.load(path)
    for key in ("sit_mask", "back_contact"):
        if key not in anchors:
            raise ValueError(f"{path} is missing {key}")
    sit_mask = np.asarray(anchors["sit_mask"], dtype=bool)
    back_contact = np.asarray(anchors["back_contact"], dtype=bool)
    if sit_mask.shape != (n_frames,) or back_contact.shape != (n_frames,):
        raise ValueError(
            f"anchor length mismatch: expected {n_frames}, got "
            f"sit={sit_mask.shape}, back={back_contact.shape}"
        )
    return sit_mask, back_contact


def _body_id(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"body {name!r} not found in {model}")
    return int(body_id)


def _root_qpos(robot: dict[str, Any], frame: int) -> np.ndarray:
    root_pos = np.asarray(robot["root_pos"], dtype=np.float64)
    root_rot_xyzw = np.asarray(robot["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(robot["dof_pos"], dtype=np.float64)
    qpos = np.zeros(7 + dof_pos.shape[1], dtype=np.float64)
    qpos[:3] = root_pos[frame]
    # MuJoCo free-joint quaternion is wxyz; robot payload stores xyzw.
    qpos[3:7] = root_rot_xyzw[frame][[3, 0, 1, 2]]
    qpos[7:] = dof_pos[frame]
    return qpos


def _torso_back_surface_local_y(model: mujoco.MjModel, data: mujoco.MjData, torso_body_id: int,
                                chair_center: np.ndarray, chair_rotation: np.ndarray,
                                back_sign: float) -> float:
    geom_ids = np.flatnonzero(model.geom_bodyid == torso_body_id)
    if len(geom_ids) == 0:
        raise ValueError("torso body has no directly attached collision geoms")
    values: list[float] = []
    for geom_id in geom_ids:
        local_center = (data.geom_xpos[geom_id] - chair_center) @ chair_rotation
        values.append(float(local_center[1] + back_sign * model.geom_rbound[geom_id]))
    return max(values)


def _contact_shift_signal(
    n_frames: int,
    contact_indices: np.ndarray,
    required: np.ndarray,
    sit_mask: np.ndarray,
    ramp_frames: int,
    bridge_max_gap: int,
) -> tuple[np.ndarray, list[list[int]]]:
    """Build a smooth, evidence-limited shift signal.

    Small holes in a contact detector are linearly bridged.  Separate contact
    episodes are only joined by short fade ramps, so the correction never
    silently extends over a long interval without source contact evidence.
    """
    if len(contact_indices) == 0:
        raise ValueError("no back-contact frames")
    if bridge_max_gap < 0:
        raise ValueError("bridge_max_gap must be non-negative")

    sit_indices = np.flatnonzero(sit_mask)
    if len(sit_indices) == 0:
        raise ValueError("sit_mask has no positive frame")
    sit_start, sit_end = int(sit_indices[0]), int(sit_indices[-1])
    signal = np.zeros(n_frames, dtype=np.float64)
    components: list[list[int]] = []

    split_points = np.flatnonzero(np.diff(contact_indices) > bridge_max_gap + 1) + 1
    for indices, values in zip(np.split(contact_indices, split_points),
                               np.split(required, split_points)):
        start, end = int(indices[0]), int(indices[-1])
        interpolated = np.interp(np.arange(start, end + 1), indices, values)
        signal[start : end + 1] = np.maximum(signal[start : end + 1], interpolated)
        components.append([start, end])
        for offset in range(1, ramp_frames + 1):
            weight = 1.0 - offset / (ramp_frames + 1.0)
            before, after = start - offset, end + offset
            if before >= sit_start:
                signal[before] = max(signal[before], values[0] * weight)
            if after <= sit_end:
                signal[after] = max(signal[after], values[-1] * weight)
    return signal, components


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Contact-evidenced GMR root correction against a semantic chair backrest."
    )
    parser.add_argument("--robot-motion", required=True, type=Path)
    parser.add_argument("--chair-primitives", required=True, type=Path)
    parser.add_argument("--contact-anchors", required=True, type=Path)
    parser.add_argument("--robot-xml", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--torso-body", default="torso_link")
    parser.add_argument("--clearance-m", type=float, default=0.025)
    parser.add_argument("--max-shift-m", type=float, default=0.30)
    parser.add_argument("--min-back-contact-coverage", type=float, default=0.30)
    parser.add_argument("--ramp-frames", type=int, default=6)
    parser.add_argument("--bridge-max-gap", type=int, default=2)
    args = parser.parse_args()

    if (args.clearance_m < 0.0 or args.max_shift_m <= 0.0 or args.ramp_frames < 0
            or args.bridge_max_gap < 0):
        raise ValueError("clearance, max shift, ramp frames and bridge gap must be non-negative (max shift > 0)")

    with args.robot_motion.open("rb") as handle:
        robot = pickle.load(handle)
    root_pos = np.asarray(robot.get("root_pos"), dtype=np.float64)
    root_rot = np.asarray(robot.get("root_rot"), dtype=np.float64)
    dof_pos = np.asarray(robot.get("dof_pos"), dtype=np.float64)
    if root_pos.ndim != 2 or root_pos.shape[1] != 3 or root_rot.shape != (len(root_pos), 4):
        raise ValueError("robot motion must contain root_pos [T,3] and root_rot [T,4]")
    if dof_pos.ndim != 2 or len(dof_pos) != len(root_pos):
        raise ValueError("robot motion must contain dof_pos [T,D]")

    chair_center, chair_rotation, back_center, back_extents = _load_primitives(args.chair_primitives)
    back_direction, back_front_y, back_sign = _chair_back_frame(
        chair_rotation, chair_center, back_center, back_extents
    )
    sit_mask, back_contact = _load_contact_anchors(args.contact_anchors, len(root_pos))
    sit_count = int(sit_mask.sum())
    contact_count = int((sit_mask & back_contact).sum())
    contact_coverage = contact_count / max(sit_count, 1)
    if contact_coverage < args.min_back_contact_coverage:
        raise RuntimeError(
            f"back-contact evidence too weak: {contact_coverage:.3f} < "
            f"{args.min_back_contact_coverage:.3f}"
        )

    model = mujoco.MjModel.from_xml_path(str(args.robot_xml))
    data = mujoco.MjData(model)
    if model.nq != 7 + dof_pos.shape[1]:
        raise ValueError(
            f"robot DOF mismatch: XML nq={model.nq}, motion requires {7 + dof_pos.shape[1]}"
        )
    torso_body_id = _body_id(model, args.torso_body)

    contact_indices = np.flatnonzero(sit_mask & back_contact)
    pre_surface_y: list[float] = []
    required_shift: list[float] = []
    for frame in contact_indices:
        data.qpos[:] = _root_qpos(robot, int(frame))
        mujoco.mj_forward(model, data)
        surface_y = _torso_back_surface_local_y(
            model, data, torso_body_id, chair_center, chair_rotation, back_sign
        )
        pre_surface_y.append(surface_y)
        required_shift.append(back_front_y - args.clearance_m - surface_y)

    required = np.asarray(required_shift, dtype=np.float64)
    estimated_shift = float(np.median(required))
    if estimated_shift < -0.01:
        raise RuntimeError(
            f"robot is already too far toward the backrest (median desired shift={estimated_shift:.4f} m)"
        )
    contact_shift = np.clip(required, 0.0, args.max_shift_m)
    if float(np.max(contact_shift)) <= 1e-4:
        raise RuntimeError("computed root correction is negligible; refusing to create a misleading variant")
    shift_signal, contact_components = _contact_shift_signal(
        len(root_pos), contact_indices, contact_shift, sit_mask,
        args.ramp_frames, args.bridge_max_gap,
    )

    corrected = dict(robot)
    corrected_root_pos = root_pos.copy()
    corrected_root_pos += shift_signal[:, None] * back_direction[None, :]
    corrected["root_pos"] = corrected_root_pos.astype(np.asarray(robot["root_pos"]).dtype, copy=False)

    alignment = {
        "method": "crisp_style_smpl_back_contact_to_gmr_torso_proxy",
        "enabled": True,
        "source_robot_motion": str(args.robot_motion),
        "chair_primitives": str(args.chair_primitives),
        "contact_anchors": str(args.contact_anchors),
        "robot_xml": str(args.robot_xml),
        "torso_body": args.torso_body,
        "back_contact_frames": contact_count,
        "sit_frames": sit_count,
        "back_contact_coverage": contact_coverage,
        "clearance_m": args.clearance_m,
        "max_shift_m": args.max_shift_m,
        "estimated_shift_m": estimated_shift,
        "applied_shift_m": float(np.median(contact_shift)),
        "applied_contact_shift_range_m": [
            float(np.min(contact_shift)), float(np.max(contact_shift))
        ],
        "nonzero_shift_frame_range": [
            int(np.flatnonzero(shift_signal > 1e-6)[0]),
            int(np.flatnonzero(shift_signal > 1e-6)[-1]),
        ],
        "contact_components": contact_components,
        "back_direction_world": back_direction.tolist(),
        "backrest_front_local_y_m": back_front_y,
        "ramp_frames": args.ramp_frames,
        "bridge_max_gap": args.bridge_max_gap,
    }
    corrected["scene_contact_alignment"] = alignment

    pre_surface = np.asarray(pre_surface_y, dtype=np.float64)
    post_surface = pre_surface + shift_signal[contact_indices]
    pre_gap = back_front_y - pre_surface
    post_gap = back_front_y - post_surface
    report = {
        **alignment,
        "status": "accepted",
        "input_frames": int(len(root_pos)),
        "sit_frame_range": [int(np.flatnonzero(sit_mask)[0]), int(np.flatnonzero(sit_mask)[-1])],
        "pre_backrest_gap_m": {
            "median": float(np.median(pre_gap)),
            "p05": float(np.quantile(pre_gap, 0.05)),
            "p95": float(np.quantile(pre_gap, 0.95)),
        },
        "post_backrest_gap_m": {
            "median": float(np.median(post_gap)),
            "p05": float(np.quantile(post_gap, 0.05)),
            "p95": float(np.quantile(post_gap, 0.95)),
        },
        "target_clearance_m": args.clearance_m,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(corrected, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

