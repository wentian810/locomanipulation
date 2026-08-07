#!/usr/bin/env python3
"""Construct a collision-feasible HoloMotion reference for a static chair.

This is deliberately a *reference* repair, not a runtime physics shortcut.
It keeps the original HoloMotion reference until the last collision-free
approach frame, then smoothly re-times the motion into a separately verified
gravity-supported seated qpos.  At runtime the tracker still receives one
initial state and advances only through MuJoCo ``mj_step``; no state is
reimposed and the chair remains a static world geometry.

Why this exists: a GMR trajectory can encode the human pelvis passing through
an estimated chair.  Asking a physical tracker to follow such a trajectory
creates apparent root jumps when contact correctly rejects the reference.
The transition endpoint is therefore detected from actual MuJoCo geometry,
not hand-picked per clip.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import mujoco
import numpy as np


SEMANTIC_CHAIR_GEOMS = frozenset((
    "seat_support_geom", "backrest_geom", "leg_front_left_geom",
    "leg_front_right_geom", "leg_back_left_geom", "leg_back_right_geom",
))


def _smoothstep5(values: np.ndarray) -> np.ndarray:
    """Quintic blend with zero velocity/acceleration at both endpoints."""
    return values**3 * (10.0 + values * (-15.0 + 6.0 * values))


def _normalise_continuous_xyzw(quaternions: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternions, dtype=np.float64).copy()
    result /= np.linalg.norm(result, axis=1, keepdims=True)
    for index in range(1, len(result)):
        if np.dot(result[index - 1], result[index]) < 0.0:
            result[index] *= -1.0
    return result


def _slerp_xyzw(left: np.ndarray, right: np.ndarray, fraction: float) -> np.ndarray:
    cosine = float(np.clip(np.dot(left, right), -1.0, 1.0))
    if cosine < 0.0:
        right, cosine = -right, -cosine
    if cosine > 0.9995:
        blended = (1.0 - fraction) * left + fraction * right
        return blended / np.linalg.norm(blended)
    theta = float(np.arccos(cosine))
    sine = float(np.sin(theta))
    return (
        np.sin((1.0 - fraction) * theta) / sine * left
        + np.sin(fraction * theta) / sine * right
    )


def _time_derivative(values: np.ndarray, timestep: float) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values, dtype=np.float64)
    return np.gradient(values, timestep, axis=0, edge_order=2 if len(values) >= 3 else 1)


def _quat_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.array((
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ), dtype=np.float64)


def _world_angular_velocity_xyzw(quaternions: np.ndarray, timestep: float) -> np.ndarray:
    output = np.zeros((len(quaternions), 3), dtype=np.float64)
    for index in range(len(quaternions)):
        left, right = max(0, index - 1), min(len(quaternions) - 1, index + 1)
        if left == right:
            continue
        inverse_left = quaternions[left].copy()
        inverse_left[:3] *= -1.0
        delta = _quat_multiply_xyzw(quaternions[right], inverse_left)
        if delta[3] < 0.0:
            delta *= -1.0
        vector_norm = float(np.linalg.norm(delta[:3]))
        if vector_norm > 1e-10:
            angle = 2.0 * np.arctan2(vector_norm, float(np.clip(delta[3], -1.0, 1.0)))
            output[index] = delta[:3] / vector_norm * (angle / ((right - left) * timestep))
    return output


def _chair_contact_distances(model: mujoco.MjModel, data: mujoco.MjData) -> list[float]:
    distances: list[float] = []
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        first = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        second = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        if first in SEMANTIC_CHAIR_GEOMS or second in SEMANTIC_CHAIR_GEOMS:
            distances.append(float(contact.dist))
    return distances


def _set_qpos(data: mujoco.MjData, qpos: np.ndarray) -> None:
    # HoloMotion globals are XYZW; MuJoCo free-joint qpos stores WXYZ.
    data.qpos[:] = qpos
    data.qvel[:] = 0.0


def _minimum_chair_distance(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray) -> float | None:
    _set_qpos(data, qpos)
    mujoco.mj_forward(model, data)
    distances = _chair_contact_distances(model, data)
    return min(distances) if distances else None


def _raise_to_chair_clearance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    tolerance_m: float,
    max_lift_m: float,
) -> tuple[np.ndarray, float]:
    """Raise the whole free base only when an interpolated frame intersects chair.

    This is a conservative fallback for an interpolation path whose endpoints
    are feasible but whose intermediate link sweep clips the seat.  It is not
    applied to the held gravity-verified endpoint.
    """
    baseline = _minimum_chair_distance(model, data, qpos)
    if baseline is None or baseline >= -tolerance_m:
        return qpos, 0.0
    lifted = qpos.copy()
    lifted[2] += max_lift_m
    high_distance = _minimum_chair_distance(model, data, lifted)
    if high_distance is not None and high_distance < -tolerance_m:
        raise RuntimeError(
            f"even {max_lift_m:.3f} m base lift leaves chair penetration {high_distance:.6f} m"
        )
    low, high = 0.0, max_lift_m
    for _ in range(32):
        middle = 0.5 * (low + high)
        probe = qpos.copy()
        probe[2] += middle
        distance = _minimum_chair_distance(model, data, probe)
        if distance is None or distance >= -tolerance_m:
            high = middle
        else:
            low = middle
    qpos = qpos.copy()
    qpos[2] += high
    return qpos, high


def _fill_global_arrays(model: mujoco.MjModel, qposes: np.ndarray, timestep: float) -> dict[str, np.ndarray]:
    data = mujoco.MjData(model)
    count, body_count = len(qposes), model.nbody - 1
    translation = np.empty((count, body_count, 3), dtype=np.float64)
    rotation_xyzw = np.empty((count, body_count, 4), dtype=np.float64)
    for frame, qpos in enumerate(qposes):
        _set_qpos(data, qpos)
        mujoco.mj_forward(model, data)
        translation[frame] = data.xpos[1:]
        rotation_xyzw[frame] = data.xquat[1:, [1, 2, 3, 0]]
    return {
        "ref_dof_pos": qposes[:, 7:],
        "ref_dof_vel": _time_derivative(qposes[:, 7:], timestep),
        "ref_global_translation": translation,
        "ref_global_rotation_quat": rotation_xyzw,
        "ref_global_velocity": _time_derivative(translation, timestep),
        "ref_global_angular_velocity": np.stack(
            [_world_angular_velocity_xyzw(rotation_xyzw[:, body], timestep) for body in range(body_count)],
            axis=1,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-reference", required=True, type=Path)
    parser.add_argument("--static-scene", required=True, type=Path)
    parser.add_argument("--landing-report", required=True, type=Path)
    parser.add_argument("--output-reference", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--transition-duration-s", type=float, default=1.0)
    parser.add_argument(
        "--seat-cue-frame", type=int, default=None,
        help=(
            "optional independent semantic-contact cue in HoloMotion (not source-video) frames. "
            "Use this when the source robot reference hovers above the seat instead of penetrating it. "
            "If geometry penetrates earlier, that earlier collision remains authoritative."
        ),
    )
    parser.add_argument("--penetration-tolerance-m", type=float, default=0.003)
    parser.add_argument("--max-interpolation-lift-m", type=float, default=0.5)
    args = parser.parse_args()
    if args.output_reference.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite an existing physical-seat reference/report")
    if args.transition_duration_s <= 0.0:
        raise ValueError("transition duration must be positive")
    if args.penetration_tolerance_m < 0.0 or args.max_interpolation_lift_m <= 0.0:
        raise ValueError("clearance arguments must be non-negative/positive")

    source = np.load(args.source_reference, allow_pickle=False)
    required = {
        "metadata", "ref_dof_pos", "ref_global_translation", "ref_global_rotation_quat",
    }
    missing = required.difference(source.files)
    if missing:
        raise ValueError(f"source reference is missing {sorted(missing)}")
    metadata = json.loads(str(source["metadata"]))
    fps = float(metadata["motion_fps"])
    model = mujoco.MjModel.from_xml_path(str(args.static_scene))
    if model.nq != 7 + source["ref_dof_pos"].shape[1]:
        raise ValueError("static scene qpos layout does not match reference DoF count")
    landing = json.loads(args.landing_report.read_text(encoding="utf-8"))
    if landing.get("status") != "accepted_gravity_seat_landing":
        raise ValueError("landing report must be an accepted gravity-seat baseline")
    target_qpos = np.asarray(landing["summary"]["settled_final_qpos_wxyz"], dtype=np.float64)
    if target_qpos.shape != (model.nq,):
        raise ValueError("landing qpos shape does not match static scene")

    root_translation = np.asarray(source["ref_global_translation"], dtype=np.float64)[:, 0]
    root_rotation = _normalise_continuous_xyzw(np.asarray(source["ref_global_rotation_quat"], dtype=np.float64)[:, 0])
    source_qposes = np.concatenate((
        root_translation,
        root_rotation[:, [3, 0, 1, 2]],
        np.asarray(source["ref_dof_pos"], dtype=np.float64),
    ), axis=1)
    data = mujoco.MjData(model)
    source_distances = [_minimum_chair_distance(model, data, qpos) for qpos in source_qposes]
    colliding = [
        index for index, distance in enumerate(source_distances)
        if distance is not None and distance < -args.penetration_tolerance_m
    ]
    if args.seat_cue_frame is not None and not 0 <= args.seat_cue_frame < len(source_qposes):
        raise ValueError("seat-cue-frame is outside source reference")
    if colliding:
        seat_frame = colliding[0]
        seat_cue_source = "first_geometric_penetration"
    elif args.seat_cue_frame is not None:
        seat_frame = args.seat_cue_frame
        seat_cue_source = "independent_semantic_contact"
    else:
        raise RuntimeError(
            "source reference neither penetrates the static chair nor supplies an independent seat cue"
        )
    if args.seat_cue_frame is not None and seat_frame < args.seat_cue_frame:
        seat_cue_source = "first_geometric_penetration_precedes_semantic_cue"
    transition_frames = int(np.rint(args.transition_duration_s * fps))
    transition_start = seat_frame - transition_frames
    if transition_start < 0:
        raise RuntimeError(
            f"need {transition_frames} clean transition frames before first collision {seat_frame}, only {seat_frame} exist"
        )
    prefix_penetration = [
        (index, distance) for index, distance in enumerate(source_distances[:transition_start + 1])
        if distance is not None and distance < -args.penetration_tolerance_m
    ]
    if prefix_penetration:
        raise RuntimeError(f"preserved approach already penetrates chair: {prefix_penetration[:3]}")

    qposes = source_qposes.copy()
    start_qpos = source_qposes[transition_start]
    start_quat_xyzw = start_qpos[3:7][[1, 2, 3, 0]]
    target_quat_xyzw = target_qpos[3:7][[1, 2, 3, 0]]
    for frame in range(transition_start, seat_frame + 1):
        fraction = (frame - transition_start) / transition_frames
        weight = float(_smoothstep5(np.asarray(fraction)))
        qposes[frame, :3] = (1.0 - weight) * start_qpos[:3] + weight * target_qpos[:3]
        qposes[frame, 3:7] = _slerp_xyzw(start_quat_xyzw, target_quat_xyzw, weight)[[3, 0, 1, 2]]
        qposes[frame, 7:] = (1.0 - weight) * start_qpos[7:] + weight * target_qpos[7:]
    qposes[seat_frame:] = target_qpos

    lifts = np.zeros(len(qposes), dtype=np.float64)
    for frame in range(transition_start, seat_frame):
        qposes[frame], lifts[frame] = _raise_to_chair_clearance(
            model, data, qposes[frame], args.penetration_tolerance_m, args.max_interpolation_lift_m
        )
    target_distance = _minimum_chair_distance(model, data, target_qpos)
    if target_distance is not None and target_distance < -args.penetration_tolerance_m:
        raise RuntimeError(
            f"accepted landing itself penetrates chair by {target_distance:.6f} m; fix landing, not reference"
        )
    final_distances = [_minimum_chair_distance(model, data, qpos) for qpos in qposes]
    final_penetrations = [
        (index, distance) for index, distance in enumerate(final_distances)
        if distance is not None and distance < -args.penetration_tolerance_m
    ]
    if final_penetrations:
        raise RuntimeError(f"constructed reference remains chair-penetrating: {final_penetrations[:3]}")

    arrays = _fill_global_arrays(model, qposes, 1.0 / fps)
    output_metadata = copy.deepcopy(metadata)
    output_metadata.update({
        "physical_reference": {
            "mode": "collision_aware_retimed_gravity_seat_hold_v1",
            "source_reference": str(args.source_reference.resolve()),
            "static_scene": str(args.static_scene.resolve()),
            "accepted_landing_report": str(args.landing_report.resolve()),
            "seat_cue_frame": seat_frame,
            "seat_cue_source": seat_cue_source,
            "transition_start_frame": transition_start,
            "transition_duration_s": args.transition_duration_s,
            "penentration_tolerance_m": args.penetration_tolerance_m,
        },
        "conversion": "source prefix preserved; collision-free transition to verified gravity seat; globals regenerated by MuJoCo FK",
        "quaternion_convention": "reference globals are XYZW; MuJoCo qpos uses WXYZ",
    })
    payload = {"metadata": np.asarray(json.dumps(output_metadata, ensure_ascii=False))}
    payload.update({key: value.astype(np.float32) for key, value in arrays.items()})
    args.output_reference.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_reference, **payload)
    report = {
        "schema_version": 1,
        "status": "ready_for_frozen_holomotion_mj_step_validation",
        "mode": "collision_aware_retimed_gravity_seat_hold_v1",
        "inputs": {
            "source_reference": str(args.source_reference.resolve()),
            "static_scene": str(args.static_scene.resolve()),
            "accepted_landing_report": str(args.landing_report.resolve()),
        },
        "timing": {
            "fps": fps,
            "seat_cue_frame": seat_frame,
            "seat_cue_source": seat_cue_source,
            "transition_start_frame": transition_start,
            "transition_duration_frames": transition_frames,
            "held_seat_frame_range": [seat_frame, len(qposes) - 1],
        },
        "geometry": {
            "source_min_chair_distance_m": (
                None if all(distance is None for distance in source_distances)
                else float(min(distance for distance in source_distances if distance is not None))
            ),
            "target_min_chair_distance_m": target_distance,
            "constructed_min_chair_distance_m": None if all(distance is None for distance in final_distances) else float(min(distance for distance in final_distances if distance is not None)),
            "max_interpolation_root_lift_m": float(np.max(lifts)),
            "lifted_transition_frame_count": int(np.count_nonzero(lifts > 1e-9)),
        },
        "physical_contract": {
            "runtime": "one initial qpos/qvel write, policy torque within official limits, then continuous mj_step",
            "chair": "static world geoms; no body added to policy observation",
            "forbidden": ["root_state_reimposition", "xfrc_applied", "mocap_weld", "moving_scene_geometry"],
        },
        "output_reference": str(args.output_reference.resolve()),
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
