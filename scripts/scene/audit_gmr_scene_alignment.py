#!/usr/bin/env python3
"""Measure GMR and source-human placement in a semantic chair local frame.

This report is diagnostic only. It does not move the robot or scene: root points
are insufficient to justify a correction, so every reported offset remains
traceable before V20's SMPL posterior-contact refinement is enabled.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    x, y, z, w = (quat[:, index] for index in range(4))
    return np.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ),
        axis=1,
    ).reshape(-1, 3, 3)


def summary(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "median_m": np.median(values, axis=0).round(6).tolist(),
        "p05_m": np.percentile(values, 5, axis=0).round(6).tolist(),
        "p95_m": np.percentile(values, 95, axis=0).round(6).tolist(),
    }


def primitive(items: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for item in items:
        if item.get("name") == name:
            return item
    raise KeyError(f"Missing primitive {name!r}; available={[x.get('name') for x in items]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chair-primitives", type=Path, required=True)
    parser.add_argument("--robot-motion", type=Path, required=True)
    parser.add_argument("--human-motion", type=Path, default=None)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    chair = json.loads(args.chair_primitives.read_text(encoding="utf-8"))
    if chair.get("frame") != "mujoco_world_z_up":
        raise ValueError("chair primitives must use mujoco_world_z_up")
    seat = primitive(chair["primitives"], "seat_support")
    backrest = primitive(chair["primitives"], "backrest")
    center = np.asarray(seat["center"], dtype=np.float64)
    rotation = np.asarray(seat["rotation_matrix"], dtype=np.float64)
    extents = np.asarray(seat["extents"], dtype=np.float64)
    back_center_local = (np.asarray(backrest["center"], dtype=np.float64) - center) @ rotation

    with args.robot_motion.open("rb") as stream:
        robot = pickle.load(stream)
    root_pos = np.asarray(robot["root_pos"], dtype=np.float64)
    root_rot = np.asarray(robot["root_rot"], dtype=np.float64)
    local_body_pos = np.asarray(robot.get("local_body_pos"), dtype=np.float64)
    body_names = list(robot.get("link_body_list", []))
    if local_body_pos.shape[:2] != (len(root_pos), len(body_names)):
        raise ValueError("robot motion lacks world-reconstructable local_body_pos")
    body_world = root_pos[:, None, :] + np.einsum(
        "tij,tbj->tbi", quat_xyzw_to_matrix(root_rot), local_body_pos
    )
    pelvis_index = body_names.index("pelvis") if "pelvis" in body_names else None
    torso_index = body_names.index("torso_link") if "torso_link" in body_names else None
    pelvis_world = body_world[:, pelvis_index] if pelvis_index is not None else root_pos
    torso_world = body_world[:, torso_index] if torso_index is not None else pelvis_world
    pelvis_local = (pelvis_world - center) @ rotation
    torso_local = (torso_world - center) @ rotation
    sit_mask = (
        (np.abs(pelvis_local[:, 0]) <= 0.5 * extents[0] + 0.18)
        & (np.abs(pelvis_local[:, 1]) <= 0.5 * extents[1] + 0.30)
        & (pelvis_local[:, 2] >= 0.02)
        & (pelvis_local[:, 2] <= 0.55)
    )
    frames = np.flatnonzero(sit_mask)
    if not len(frames):
        raise RuntimeError("No conservative robot sitting interval found")

    human_info: dict[str, Any] | None = None
    if args.human_motion:
        human = np.load(args.human_motion, allow_pickle=False)
        # The bridged source is neg-Y-up; PHC/GMR use +90 degrees about X.
        trans = np.asarray(human["trans"], dtype=np.float64)
        human_world = np.column_stack((trans[:, 0], -trans[:, 2], trans[:, 1]))
        count = min(len(human_world), len(pelvis_local))
        human_local = (human_world[:count] - center) @ rotation
        common = frames[frames < count]
        human_info = {
            "coordinate_contract": "neg_y_source_to_z_up_x_plus_90",
            "root_local_sit": summary(human_local[common]),
            "robot_pelvis_minus_human_root_local_sit": summary(
                pelvis_local[common] - human_local[common]
            ),
            "limitation": (
                "human root is a proxy; V20 must replace it with posterior pelvis, "
                "thigh, and torso SMPL vertices before any correction is applied"
            ),
        }

    report = {
        "schema_version": 1,
        "mode": "diagnostic_no_motion_adjustment",
        "chair": {
            "seat_center_world_m": center.round(6).tolist(),
            "seat_extents_m": extents.round(6).tolist(),
            "backrest_center_local_m": back_center_local.round(6).tolist(),
        },
        "robot": {
            "frames": int(len(root_pos)),
            "sit_frames": frames.astype(int).tolist(),
            "pelvis_local_sit": summary(pelvis_local[sit_mask]),
            "torso_local_sit": summary(torso_local[sit_mask]),
            "pelvis_clearance_above_seat_top_m": summary(
                (pelvis_local[sit_mask, 2] - 0.5 * extents[2])[:, None]
            ),
            "torso_signed_distance_to_backrest_midplane_m": summary(
                (torso_local[sit_mask, 1] - back_center_local[1])[:, None]
            ),
            "body_landmarks": {
                "pelvis": "pelvis" if pelvis_index is not None else "root_fallback",
                "torso": "torso_link" if torso_index is not None else "pelvis_fallback",
            },
        },
        "human_source": human_info,
        "next_gate": (
            "Do not tune GMR root translation from this report alone. "
            "Use it to trigger V20 posterior-contact fitting and then replay both simulators."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

