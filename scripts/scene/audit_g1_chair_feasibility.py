#!/usr/bin/env python3
"""Measure G1/semantic-chair geometry in task keyframes, without simulating.

This is a diagnostic for the static scene contract.  It deliberately uses the
same compiled task XML as MPC, so scene, G1, and reference keyframes cannot be
silently drawn from different pipeline runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link", "right_hip_pitch_link",
    "left_knee_link", "right_knee_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_toe_link", "right_toe_link",
)


def _has_prefix(value: str | None, prefix: str) -> bool:
    return value is not None and value.startswith(prefix)


def _body_id(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"required G1 body is missing: {name}")
    return body_id


def _descendants(model: mujoco.MjModel, root_body: int) -> set[int]:
    result = {root_body}
    changed = True
    while changed:
        changed = False
        for body_id in range(1, model.nbody):
            if body_id not in result and int(model.body_parentid[body_id]) in result:
                result.add(body_id)
                changed = True
    return result


def _lowest_geom_z(model: mujoco.MjModel, data: mujoco.MjData, body_ids: set[int]) -> float:
    geom_ids = [geom_id for geom_id in range(model.ngeom) if int(model.geom_bodyid[geom_id]) in body_ids]
    if not geom_ids:
        raise ValueError("foot subtree has no geometry")
    # geom_rbound is a conservative enclosing radius.  It avoids assuming that
    # an arbitrary mesh/capsule has a particular local axis.
    return float(min(data.geom_xpos[geom_id, 2] - model.geom_rbound[geom_id] for geom_id in geom_ids))


def _collision_geom_ids(model: mujoco.MjModel, body_id: int) -> list[int]:
    return [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == body_id
        and (int(model.geom_contype[geom_id]) != 0 or int(model.geom_conaffinity[geom_id]) != 0)
    ]


def _collision_vertical_interval(
    model: mujoco.MjModel, data: mujoco.MjData, body_id: int
) -> tuple[float, float] | None:
    """Conservative vertical extent of collision-enabled geoms on one link."""
    geom_ids = _collision_geom_ids(model, body_id)
    if not geom_ids:
        return None
    lower = min(data.geom_xpos[geom_id, 2] - model.geom_rbound[geom_id] for geom_id in geom_ids)
    upper = max(data.geom_xpos[geom_id, 2] + model.geom_rbound[geom_id] for geom_id in geom_ids)
    return float(lower), float(upper)


def _minimum_distance_to_seat(
    model: mujoco.MjModel, data: mujoco.MjData, body_id: int, seat_geom: int
) -> float | None:
    distances: list[float] = []
    for geom_id in _collision_geom_ids(model, body_id):
        nearest_points = np.empty(6, dtype=np.float64)
        distances.append(float(mujoco.mj_geomDistance(model, data, geom_id, seat_geom, 2.0, nearest_points)))
    return min(distances) if distances else None


def _minimum_distance_to_chair(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_geom_ids: list[int],
    chair_geom_ids: list[int],
) -> tuple[float, int, int]:
    best_distance = np.inf
    best_pair = (-1, -1)
    nearest_points = np.empty(6, dtype=np.float64)
    for robot_geom in robot_geom_ids:
        for chair_geom in chair_geom_ids:
            distance = float(mujoco.mj_geomDistance(model, data, robot_geom, chair_geom, 2.0, nearest_points))
            if distance < best_distance:
                best_distance = distance
                best_pair = (robot_geom, chair_geom)
    if best_pair[0] < 0:
        raise ValueError("cannot form robot-chair distance pairs")
    return float(best_distance), best_pair[0], best_pair[1]


def _actual_chair_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_geom_ids: set[int],
    chair_geom_ids: set[int],
) -> dict[str, object]:
    """Read MuJoCo's generated contacts, rather than extrapolating mesh SDFs.

    ``mj_geomDistance`` is useful for separated primitive pairs, but its signed
    result for deeply interpenetrating mesh pairs is not a penetration-depth
    measurement.  The simulator's contact list is the quantity relevant to a
    subsequent ``mj_step`` rollout, so it is reported separately here.
    """
    minimum_distance = float("inf")
    contacts = 0
    pairs: dict[tuple[str, str], float] = {}
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        first, second = int(contact.geom1), int(contact.geom2)
        if first in robot_geom_ids and second in chair_geom_ids:
            robot_geom, chair_geom = first, second
        elif second in robot_geom_ids and first in chair_geom_ids:
            robot_geom, chair_geom = second, first
        else:
            continue
        distance = float(contact.dist)
        minimum_distance = min(minimum_distance, distance)
        contacts += 1
        robot_body = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[robot_geom])
        ) or ""
        chair_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, chair_geom) or ""
        key = (robot_body, chair_name)
        pairs[key] = min(pairs.get(key, distance), distance)
    return {
        "contact_count": contacts,
        "minimum_distance_m": None if not contacts else minimum_distance,
        "maximum_penetration_m": max(0.0, -minimum_distance) if contacts else 0.0,
        "pairs": pairs,
    }


def _root_height_sweep(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_geom_ids: list[int],
    chair_geom_ids: list[int],
    minimum_m: float,
    maximum_m: float,
    step_m: float,
) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for offset in np.arange(minimum_m, maximum_m + 0.5 * step_m, step_m):
        distances: list[float] = []
        worst_pair = (-1, -1)
        worst_key = -1
        for key_id in range(model.nkey):
            data.qpos[:] = model.key_qpos[key_id]
            data.qpos[2] += offset
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            distance, robot_geom, chair_geom = _minimum_distance_to_chair(model, data, robot_geom_ids, chair_geom_ids)
            distances.append(distance)
            if distance <= min(distances):
                worst_pair = (robot_geom, chair_geom)
                worst_key = key_id
        candidates.append({
            "root_z_offset_m": float(offset),
            "minimum_distance_m": float(min(distances)),
            "p05_distance_m": float(np.percentile(distances, 5)),
            "collision_frame_ratio": float(np.mean(np.asarray(distances) < -0.005)),
            "worst_reference_key": float(worst_key),
            "worst_robot_geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, worst_pair[0]) or "",
            "worst_robot_body": mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[worst_pair[0]])
            ) or "",
            "worst_chair_geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, worst_pair[1]) or "",
        })
    best = max(candidates, key=lambda item: (item["minimum_distance_m"], item["p05_distance_m"]))
    return {
        "range_m": [minimum_m, maximum_m],
        "step_m": step_m,
        "best_candidate": best,
        "height_only_feasible": bool(best["minimum_distance_m"] >= -0.005),
        "candidates": candidates,
    }


def _seat_top_z(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
        raise ValueError("semantic seat support must be a box collision primitive")
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    half_size = model.geom_size[geom_id]
    return float(data.geom_xpos[geom_id, 2] + np.abs(rotation[2]) @ half_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-xml", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--root-z-scan-min-m", type=float, default=-0.15)
    parser.add_argument("--root-z-scan-max-m", type=float, default=0.25)
    parser.add_argument("--root-z-scan-step-m", type=float, default=0.005)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.task_xml))
    if model.nkey < 2:
        raise ValueError("task must contain multiple reference keyframes")
    seat_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "seat_support_geom")
    back_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "backrest_geom")
    if seat_geom < 0 or back_geom < 0:
        raise ValueError("task lacks semantic chair collision geometry")
    body_ids = {name: _body_id(model, name) for name in BODY_NAMES}
    left_foot = _descendants(model, body_ids["left_ankle_roll_link"])
    right_foot = _descendants(model, body_ids["right_ankle_roll_link"])
    robot_bodies = _descendants(model, body_ids["pelvis"])
    robot_geoms = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in robot_bodies
        and (int(model.geom_contype[geom_id]) != 0 or int(model.geom_conaffinity[geom_id]) != 0)
    ]
    chair_geoms = [
        geom_id
        for geom_id in range(model.ngeom)
        if _has_prefix(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id), "seat_support_geom")
        or _has_prefix(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id), "backrest_geom")
        or _has_prefix(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id), "leg_front_")
        or _has_prefix(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id), "leg_back_")
    ]
    data = mujoco.MjData(model)
    frames: list[dict[str, float]] = []
    contact_frames: list[dict[str, object]] = []
    contact_pair_depths: dict[tuple[str, str], float] = {}
    for key_id in range(model.nkey):
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)
        left_foot_z = _lowest_geom_z(model, data, left_foot)
        right_foot_z = _lowest_geom_z(model, data, right_foot)
        seat_top_z = _seat_top_z(model, data, seat_geom)
        frame: dict[str, float] = {
            "keyframe": float(key_id),
            "seat_top_z_m": seat_top_z,
            "left_foot_lowest_z_m": left_foot_z,
            "right_foot_lowest_z_m": right_foot_z,
            "seat_height_above_lowest_foot_m": seat_top_z - min(left_foot_z, right_foot_z),
        }
        for name, body_id in body_ids.items():
            frame[f"{name}_z_m"] = float(data.xpos[body_id, 2])
            frame[f"{name}_above_seat_m"] = float(data.xpos[body_id, 2] - seat_top_z)
            interval = _collision_vertical_interval(model, data, body_id)
            if interval is not None:
                lower, upper = interval
                frame[f"{name}_collision_lowest_z_m"] = lower
                frame[f"{name}_collision_highest_z_m"] = upper
                frame[f"{name}_collision_lowest_above_seat_m"] = lower - seat_top_z
            seat_distance = _minimum_distance_to_seat(model, data, body_id, seat_geom)
            if seat_distance is not None:
                frame[f"{name}_minimum_distance_to_seat_m"] = seat_distance
        frames.append(frame)
        contact = _actual_chair_contact(model, data, set(robot_geoms), set(chair_geoms))
        for pair, depth in contact.pop("pairs").items():
            contact_pair_depths[pair] = min(contact_pair_depths.get(pair, depth), depth)
        contact_frames.append({"keyframe": key_id, **contact})
    numeric_keys = [key for key in frames[0] if key != "keyframe"]
    median = {key: float(np.median([frame[key] for frame in frames])) for key in numeric_keys}
    root_height_sweep = _root_height_sweep(
        model, data, robot_geoms, chair_geoms,
        args.root_z_scan_min_m, args.root_z_scan_max_m, args.root_z_scan_step_m,
    )
    contact_penetrations = np.asarray(
        [float(item["maximum_penetration_m"]) for item in contact_frames], dtype=np.float64
    )
    contact_summary = {
        "contact_frame_count": int(sum(int(item["contact_count"]) > 0 for item in contact_frames)),
        "penetrating_frame_count": int(np.count_nonzero(contact_penetrations > 0.0)),
        "maximum_penetration_m": float(np.max(contact_penetrations)),
        "p95_penetration_m": float(np.quantile(contact_penetrations, 0.95)),
        "first_penetrating_frames": [
            int(item["keyframe"]) for item in contact_frames
            if float(item["maximum_penetration_m"]) > 0.0
        ][:32],
        "pairs_by_max_penetration": [
            {"robot_body": pair[0], "chair_geom": pair[1], "minimum_contact_distance_m": depth}
            for pair, depth in sorted(contact_pair_depths.items(), key=lambda item: item[1])
        ],
    }
    report = {
        "schema_version": 1,
        "purpose": "g1_chair_static_geometry_feasibility",
        "task_xml": str(args.task_xml),
        "frame_count": model.nkey,
        "coordinate_contract": "compiled_MuJoCo_world_z_up; all values are metres",
        "measurements": {"median": median, "per_reference_frame": frames},
        "actual_mujoco_chair_contacts": {
            "summary": contact_summary,
            "per_reference_frame": contact_frames,
            "definition": "Generated MuJoCo robot-chair contacts after mj_forward; this is the contact-depth source for physics decisions.",
        },
        "root_height_only_sweep": root_height_sweep,
        "interpretation": {
            "seat_height": "seat_top_z_m minus lowest foot geometry is the chair-height evidence.",
            "body_clearance": "negative *_above_seat_m means that body origin lies below the seat top; it is not a mesh penetration test.",
            "limitation": "This report measures geometry only. Promotion still requires continuous bounded-actuator contact validation and visible-mesh penetration checks.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"frame_count": model.nkey, "median": median, "root_height_only_sweep": root_height_sweep["best_candidate"], "height_only_feasible": root_height_sweep["height_only_feasible"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
