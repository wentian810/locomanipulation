#!/usr/bin/env python3
"""Audit whether a GMR G1 reference is statically supported by its feet.

This is a read-only diagnostic.  It forwards each GMR qpos through the same
G1 MJCF and compares the whole-body center of mass projection to the convex
hull of the foot contact-cap centers that are near the ground.  A negative
margin does not declare an action impossible: it says that the frame needs a
dynamic step/contact transition and cannot be held statically by a simple
posture controller.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np


FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Return a counter-clockwise 2-D convex hull without duplicate points."""
    unique = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(unique) <= 1:
        return np.asarray(unique, dtype=np.float64)

    def cross(origin: tuple[float, float], first: tuple[float, float], second: tuple[float, float]) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    edge = end - start
    squared_length = float(np.dot(edge, edge))
    if squared_length <= 1e-14:
        return float(np.linalg.norm(point - start))
    interpolation = float(np.clip(np.dot(point - start, edge) / squared_length, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + interpolation * edge)))


def _signed_hull_margin(point: np.ndarray, hull: np.ndarray) -> float | None:
    """Positive means inside; negative is the distance outside the hull."""
    if len(hull) < 3:
        return None
    inward_distances = []
    outside = False
    for index, start in enumerate(hull):
        end = hull[(index + 1) % len(hull)]
        edge = end - start
        edge_length = float(np.linalg.norm(edge))
        relative = point - start
        signed_distance = float((edge[0] * relative[1] - edge[1] * relative[0]) / edge_length)
        inward_distances.append(signed_distance)
        outside |= signed_distance < 0.0
    if not outside:
        return min(inward_distances)
    return -min(
        _point_segment_distance(point, hull[index], hull[(index + 1) % len(hull)])
        for index in range(len(hull))
    )


def _contact_cap_geoms(model: mujoco.MjModel) -> dict[str, np.ndarray]:
    """Locate the four collision caps on each named G1 foot body."""
    result: dict[str, np.ndarray] = {}
    for body_name in FOOT_BODY_NAMES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"missing expected G1 foot body: {body_name}")
        geoms = [
            geom_id
            for geom_id in range(model.ngeom)
            if model.geom_bodyid[geom_id] == body_id
            and model.geom_contype[geom_id] != 0
            and model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE
        ]
        if len(geoms) != 4:
            raise ValueError(f"expected four collision caps on {body_name}, found {geoms}")
        result[body_name] = np.asarray(geoms, dtype=np.int32)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-pkl", required=True, type=Path)
    parser.add_argument("--robot-mjcf", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--ground-tolerance-m", type=float, default=0.04)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite audit artifact: {args.output_json}")
    if args.source_fps <= 0.0 or args.ground_tolerance_m <= 0.0:
        raise ValueError("source_fps and ground_tolerance_m must be positive")

    with args.motion_pkl.open("rb") as handle:
        motion = pickle.load(handle)
    qpos = np.concatenate(
        [
            np.asarray(motion["root_pos"], dtype=np.float64),
            # GMR persists root rotations as XYZW, whereas MuJoCo free-joint
            # qpos requires WXYZ.  This conversion is deliberately explicit
            # because an order error fabricates foot-clearance evidence.
            np.asarray(motion["root_rot"], dtype=np.float64)[:, [3, 0, 1, 2]],
            np.asarray(motion["dof_pos"], dtype=np.float64),
        ],
        axis=1,
    )
    if qpos.ndim != 2 or qpos.shape[1] != 36 or not np.isfinite(qpos).all():
        raise ValueError(f"expected finite GMR qpos [T,36], got {qpos.shape}")

    model = mujoco.MjModel.from_xml_path(str(args.robot_mjcf))
    data = mujoco.MjData(model)
    if model.nq != 36:
        raise ValueError(f"expected 36-DoF free-base G1 qpos, got {model.nq}")
    caps_by_foot = _contact_cap_geoms(model)
    movable_body_ids = np.arange(1, model.nbody, dtype=np.int32)
    movable_masses = model.body_mass[movable_body_ids]
    total_mass = float(np.sum(movable_masses))
    if total_mass <= 0.0:
        raise ValueError("G1 model has non-positive movable mass")

    frames = []
    foot_height_series = {name: [] for name in FOOT_BODY_NAMES}
    for frame_index, frame_qpos in enumerate(qpos):
        data.qpos[:] = frame_qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        center_of_mass = np.einsum("b,bi->i", movable_masses, data.xipos[movable_body_ids]) / total_mass
        foot_caps = {name: data.geom_xpos[ids].copy() for name, ids in caps_by_foot.items()}
        foot_min_height = {name: float(np.min(caps[:, 2])) for name, caps in foot_caps.items()}
        for name, height in foot_min_height.items():
            foot_height_series[name].append(height)
        active_feet = [
            name for name in FOOT_BODY_NAMES if foot_min_height[name] <= args.ground_tolerance_m
        ]
        support_points = (
            np.concatenate([foot_caps[name][:, :2] for name in active_feet], axis=0)
            if active_feet
            else np.empty((0, 2), dtype=np.float64)
        )
        support_hull = _convex_hull(support_points)
        margin = _signed_hull_margin(center_of_mass[:2], support_hull)
        frames.append(
            {
                "frame": frame_index,
                "time_s": frame_index / args.source_fps,
                "root_z_m": float(frame_qpos[2]),
                "com_xy_m": [float(value) for value in center_of_mass[:2]],
                "foot_min_cap_height_m": foot_min_height,
                "active_feet": active_feet,
                "support_margin_m": margin,
            }
        )

    margins = [frame["support_margin_m"] for frame in frames if frame["support_margin_m"] is not None]
    violations = [frame for frame in frames if frame["support_margin_m"] is not None and frame["support_margin_m"] < 0.0]
    no_support = [frame for frame in frames if frame["support_margin_m"] is None]
    foot_height_arrays = {name: np.asarray(values, dtype=np.float64) for name, values in foot_height_series.items()}
    all_foot_heights = np.concatenate(list(foot_height_arrays.values()))
    nearest_foot, nearest_frame = min(
        (
            (name, int(np.argmin(values)))
            for name, values in foot_height_arrays.items()
        ),
        key=lambda item: foot_height_arrays[item[0]][item[1]],
    )
    summary = {
        "schema_version": 1,
        "purpose": "read_only_gmr_static_support_feasibility_audit",
        "inputs": {
            "motion_pkl": str(args.motion_pkl),
            "robot_mjcf": str(args.robot_mjcf),
            "source_fps": args.source_fps,
            "ground_tolerance_m": args.ground_tolerance_m,
        },
        "model_movable_mass_kg": total_mass,
        "frame_count": len(frames),
        "frames_with_geometric_support": len(margins),
        "frames_without_geometric_support": len(no_support),
        "frames_with_com_outside_support": len(violations),
        "foot_clearance": {
            "global_min_cap_height_m": float(np.min(all_foot_heights)),
            "nearest_ground_foot": nearest_foot,
            "nearest_ground_frame": nearest_frame,
            "nearest_ground_time_s": nearest_frame / args.source_fps,
            "nearest_ground_root_z_m": float(qpos[nearest_frame, 2]),
            "per_foot_min_cap_height_m": {
                name: float(np.min(values)) for name, values in foot_height_arrays.items()
            },
            "frames_at_ground_tolerance": {
                name: int(np.sum(values <= args.ground_tolerance_m))
                for name, values in foot_height_arrays.items()
            },
            "single_global_root_z_shift_to_nearest_cap_contact_m": float(
                -np.min(all_foot_heights)
            ),
            "note": (
                "This is a diagnostic lower bound only.  A production correction must use "
                "validated foot-contact intervals, never a per-frame root shift."
            ),
        },
        "support_margin_min_m": float(np.min(margins)) if margins else None,
        "support_margin_median_m": float(np.median(margins)) if margins else None,
        "initial_frames": frames[:10],
        "first_outside_support_frame": violations[0] if violations else None,
        "first_no_support_frame": no_support[0] if no_support else None,
        "interpretation": (
            "Negative support margin means the GMR frame requires a dynamic step/contact transition; "
            "it cannot be held by a static posture controller."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
