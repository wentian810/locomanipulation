#!/usr/bin/env python3
"""Apply one measured world-Z correction to a G1 motion reference.

The correction is estimated from a short initial standing window and the
official Unitree foot collision geoms in a compiled MuJoCo task.  It is a
single coordinate-contract translation for the entire motion -- never a
per-frame root repair -- so later execution can still initialize once and use
only bounded actuator torques with ``mj_step``.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np


def _foot_collision_geoms(model: mujoco.MjModel) -> list[int]:
    result: list[int] = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        body = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])
        ) or ""
        if name.startswith("gmr_official_collision_") and body.endswith("ankle_roll_link"):
            result.append(geom_id)
    if not result:
        raise ValueError("task has no official collision geoms on either ankle-roll link")
    return result


def _qpos_from_motion(motion: dict[str, object], frame: int) -> np.ndarray:
    root_position = np.asarray(motion["root_pos"], dtype=np.float64)[frame]
    root_quaternion_xyzw = np.asarray(motion["root_rot"], dtype=np.float64)[frame]
    joints = np.asarray(motion["dof_pos"], dtype=np.float64)[frame]
    return np.concatenate((root_position, root_quaternion_xyzw[[3, 0, 1, 2]], joints))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-motion", required=True, type=Path)
    parser.add_argument("--probe-task-xml", required=True, type=Path)
    parser.add_argument("--output-motion", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--window-start-frame", type=int, default=0)
    parser.add_argument("--window-end-frame", type=int, required=True,
                        help="exclusive calibration frame bound")
    parser.add_argument("--target-foot-penetration-m", type=float, default=0.001)
    parser.add_argument("--max-window-foot-gap-spread-m", type=float, default=0.025)
    args = parser.parse_args()
    if args.output_motion.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite a ground-aligned reference")
    if args.target_foot_penetration_m < 0.0:
        raise ValueError("target-foot-penetration-m must be non-negative")
    if args.max_window_foot_gap_spread_m <= 0.0:
        raise ValueError("max-window-foot-gap-spread-m must be positive")

    with args.input_motion.open("rb") as handle:
        motion = pickle.load(handle)
    for key in ("root_pos", "root_rot", "dof_pos"):
        if key not in motion:
            raise ValueError(f"input motion lacks {key!r}")
    frame_count = len(np.asarray(motion["root_pos"]))
    if not 0 <= args.window_start_frame < args.window_end_frame <= frame_count:
        raise ValueError("calibration window must be non-empty and inside the motion")

    model = mujoco.MjModel.from_xml_path(str(args.probe_task_xml))
    if model.nq != 7 + np.asarray(motion["dof_pos"]).shape[1]:
        raise ValueError("probe task qpos layout does not match motion DOF count")
    floor_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_geom < 0:
        raise ValueError("probe task lacks a floor geom")
    foot_geoms = _foot_collision_geoms(model)
    data = mujoco.MjData(model)
    gaps: list[float] = []
    for frame in range(args.window_start_frame, args.window_end_frame):
        data.qpos[:] = _qpos_from_motion(motion, frame)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        # Foot-vs-plane is a specialised collider; the positive cutoff is used
        # only to estimate a standing gap.  Penetration validation remains the
        # separate zero-cutoff audit used by the v24 pipeline.
        gaps.append(min(float(mujoco.mj_geomDistance(
            model, data, geom_id, floor_geom, 0.25, None
        )) for geom_id in foot_geoms))
    gap_array = np.asarray(gaps, dtype=np.float64)
    median_gap = float(np.median(gap_array))
    p10_gap, p90_gap = (float(value) for value in np.percentile(gap_array, (10.0, 90.0)))
    if p90_gap - p10_gap > args.max_window_foot_gap_spread_m:
        raise ValueError(
            "initial window is not a stable stance: foot-gap spread "
            f"{p90_gap - p10_gap:.6f} m exceeds {args.max_window_foot_gap_spread_m:.6f} m"
        )
    root_z_translation = -median_gap - args.target_foot_penetration_m
    result = copy.deepcopy(motion)
    root_position = np.asarray(motion["root_pos"], dtype=np.float64).copy()
    root_position[:, 2] += root_z_translation
    result["root_pos"] = root_position.astype(np.float32)
    result["ground_alignment"] = {
        "mode": "single_global_root_z_translation_from_initial_foot_stance",
        "root_z_translation_m": root_z_translation,
        "calibration_window_frames": [args.window_start_frame, args.window_end_frame],
        "median_foot_gap_m": median_gap,
    }
    report = {
        "schema_version": 1,
        "purpose": "single_global_g1_ground_coordinate_alignment",
        "status": "ready_for_new_mj_step_validation",
        "physical_contract": {
            "offline_change": "one constant world-Z translation of every root keyframe",
            "unchanged": ["root_orientation", "all_joint_angles", "frame_timing", "chair_scene_pose"],
            "forbidden_runtime_mechanisms": ["root_state_reimposition", "xfrc_applied", "mocap_weld"],
        },
        "inputs": {"motion": str(args.input_motion), "probe_task_xml": str(args.probe_task_xml)},
        "calibration": {
            "window_frames": [args.window_start_frame, args.window_end_frame],
            "foot_gap_m": {
                "min": float(np.min(gap_array)), "p10": p10_gap,
                "median": median_gap, "p90": p90_gap, "max": float(np.max(gap_array)),
            },
            "target_foot_penetration_m": args.target_foot_penetration_m,
            "root_z_translation_m": root_z_translation,
        },
        "output_motion": str(args.output_motion),
    }
    args.output_motion.parent.mkdir(parents=True, exist_ok=True)
    with args.output_motion.open("wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
