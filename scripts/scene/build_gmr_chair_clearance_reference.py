#!/usr/bin/env python3
"""Build a continuous GMR reference that respects a fixed MuJoCo chair.

This is an offline reference-construction step.  It never steps MuJoCo or
changes the chair.  For each source frame it evaluates *actual* MuJoCo
contacts, then uses dynamic programming to select a small, continuous free
base translation.  The generated reference is intended for a later
joint-torque + ``mj_step`` rollout; it is not a kinematic playback result.
"""

from __future__ import print_function

import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np


CHAIR_PREFIXES = ("seat_support_geom", "backrest_geom", "leg_front_", "leg_back_")
SUPPORT_BODIES = frozenset((
    "pelvis",
    "left_hip_pitch_link", "right_hip_pitch_link",
    "left_hip_roll_link", "right_hip_roll_link",
    "left_hip_yaw_link", "right_hip_yaw_link",
))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def motion_to_qpos(motion):
    root = np.asarray(motion["root_pos"], dtype=np.float64)
    rotation = np.asarray(motion["root_rot"], dtype=np.float64)
    dof = np.asarray(motion["dof_pos"], dtype=np.float64)
    if root.ndim != 2 or root.shape[1] != 3 or rotation.shape != (len(root), 4):
        raise ValueError("expected root_pos [T,3] and root_rot [T,4]")
    qpos = np.empty((len(root), 7 + dof.shape[1]), dtype=np.float64)
    qpos[:, :3] = root
    # GMR stores xyzw while MuJoCo free joints use wxyz.
    qpos[:, 3:7] = rotation[:, (3, 0, 1, 2)]
    qpos[:, 7:] = dof
    return qpos


def object_name(model, object_type, object_id):
    return mujoco.mj_id2name(model, object_type, object_id) or ""


def robot_bodies(model):
    free_joint = next(
        joint_id for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
    )
    bodies = {int(model.jnt_bodyid[free_joint])}
    changed = True
    while changed:
        changed = False
        for body_id in range(1, model.nbody):
            if body_id not in bodies and int(model.body_parentid[body_id]) in bodies:
                bodies.add(body_id)
                changed = True
    return bodies


def chair_contact_summary(model, data, robot_body_ids):
    """Classify real contact records, not arbitrary mesh-distance queries."""
    support_depths = []
    forbidden = []
    contact_count = 0
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom_a, geom_b = int(contact.geom1), int(contact.geom2)
        name_a = object_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_a)
        name_b = object_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_b)
        if name_a.startswith(CHAIR_PREFIXES):
            chair_name, robot_geom = name_a, geom_b
        elif name_b.startswith(CHAIR_PREFIXES):
            chair_name, robot_geom = name_b, geom_a
        else:
            continue
        if int(model.geom_bodyid[robot_geom]) not in robot_body_ids:
            continue
        contact_count += 1
        body_name = object_name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[robot_geom])
        )
        depth = float(contact.dist)
        if body_name in SUPPORT_BODIES and chair_name == "seat_support_geom":
            support_depths.append(depth)
        else:
            forbidden.append({
                "robot_body": body_name,
                "chair_geom": chair_name,
                "distance_m": depth,
            })
    return {
        "contact_count": contact_count,
        "support_count": len(support_depths),
        "support_min_distance_m": min(support_depths) if support_depths else None,
        "forbidden_count": len(forbidden),
        "forbidden_min_distance_m": min(
            (item["distance_m"] for item in forbidden), default=None
        ),
    }


def accepted(summary, frame, preseat_end, max_support_penetration,
             require_seat_support_from_frame):
    if frame <= preseat_end:
        return summary["contact_count"] == 0
    support_depth = summary["support_min_distance_m"]
    has_required_support = (
        require_seat_support_from_frame is None
        or frame < require_seat_support_from_frame
        or support_depth is not None
    )
    return (
        summary["forbidden_count"] == 0
        and (support_depth is None or support_depth >= -max_support_penetration)
        and has_required_support
    )


def make_grid(args):
    return [
        (round(float(x), 6), round(float(y), 6), round(float(z), 6))
        for x in np.arange(args.min_offset_x, args.max_offset_x + 0.5 * args.grid_step, args.grid_step)
        for y in np.arange(args.min_offset_y, args.max_offset_y + 0.5 * args.grid_step, args.grid_step)
        for z in np.arange(args.min_offset_z, args.max_offset_z + 0.5 * args.grid_step, args.grid_step)
    ]


def choose_continuous_path(candidates, frames, args):
    """Choose valid offsets while bounding the change in every coordinate."""
    states = {}
    for candidate in candidates[frames[0]]:
        offset = np.asarray(candidate, dtype=np.float64)
        states[candidate] = (args.offset_weight * float(np.dot(offset, offset)), [candidate])
    for frame in frames[1:]:
        next_states = {}
        for candidate in candidates[frame]:
            candidate_array = np.asarray(candidate, dtype=np.float64)
            best = None
            for previous, (cost, path) in states.items():
                delta = candidate_array - np.asarray(previous, dtype=np.float64)
                if np.max(np.abs(delta)) > args.max_component_step + 1e-12:
                    continue
                score = (
                    cost
                    + args.offset_weight * float(np.dot(candidate_array, candidate_array))
                    + args.delta_weight * float(np.dot(delta, delta))
                )
                if best is None or score < best[0]:
                    best = (score, path + [candidate])
            if best is not None:
                next_states[candidate] = best
        states = next_states
        if not states:
            raise RuntimeError("no continuous collision-free path at frame {}".format(frame))
    return min(states.values(), key=lambda item: item[0])[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-motion", required=True, type=Path)
    parser.add_argument("--task-xml", required=True, type=Path)
    parser.add_argument("--output-motion", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--start-frame", type=int, default=95)
    parser.add_argument("--preseat-end-frame", type=int, default=186)
    parser.add_argument("--end-frame", type=int, default=221)
    parser.add_argument(
        "--require-seat-support-from-frame", type=int, default=None,
        help=(
            "from this source frame onward, require at least one shallow "
            "pelvis/upper-leg-to-seat contact in every static reference frame"
        ),
    )
    parser.add_argument("--min-offset-x", type=float, default=-0.14)
    parser.add_argument("--max-offset-x", type=float, default=0.14)
    parser.add_argument("--min-offset-y", type=float, default=-0.14)
    parser.add_argument("--max-offset-y", type=float, default=0.14)
    parser.add_argument("--min-offset-z", type=float, default=0.0)
    parser.add_argument("--max-offset-z", type=float, default=0.12)
    parser.add_argument("--grid-step", type=float, default=0.02)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--max-component-step", type=float, default=0.025)
    parser.add_argument("--max-root-step", type=float, default=0.05)
    parser.add_argument("--max-support-penetration", type=float, default=0.003)
    parser.add_argument("--offset-weight", type=float, default=40.0)
    parser.add_argument("--delta-weight", type=float, default=240.0)
    args = parser.parse_args()

    if args.output_motion.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite a reference or audit report")
    if not (0 <= args.start_frame <= args.preseat_end_frame < args.end_frame):
        raise ValueError("require start <= preseat_end < end")
    if (
        args.require_seat_support_from_frame is not None
        and not args.preseat_end_frame < args.require_seat_support_from_frame <= args.end_frame
    ):
        raise ValueError("required seat support must begin after the pre-seat interval")
    if args.grid_step <= 0 or args.top_k <= 0 or args.max_component_step <= 0:
        raise ValueError("grid and continuity settings must be positive")

    with args.input_motion.open("rb") as handle:
        motion = pickle.load(handle)
    qpos = motion_to_qpos(motion)
    model = mujoco.MjModel.from_xml_path(str(args.task_xml))
    if model.nkey != len(qpos) or model.nq != qpos.shape[1]:
        raise ValueError("task keyframes and input motion have incompatible dimensions")
    if not np.allclose(model.key_qpos, qpos, atol=1e-6, rtol=0.0):
        raise ValueError("task keyframes must exactly match the input motion")
    if args.end_frame >= len(qpos):
        raise ValueError("end frame exceeds input motion")

    frames = list(range(args.start_frame, args.end_frame + 1))
    grid = make_grid(args)
    data = mujoco.MjData(model)
    robot_body_ids = robot_bodies(model)
    candidates = {}
    for frame in frames:
        valid = []
        for offset in grid:
            data.qpos[:] = qpos[frame]
            data.qpos[:3] += offset
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            summary = chair_contact_summary(model, data, robot_body_ids)
            if accepted(
                summary, frame, args.preseat_end_frame,
                args.max_support_penetration, args.require_seat_support_from_frame,
            ):
                valid.append(offset)
        valid.sort(key=lambda value: float(np.dot(value, value)))
        candidates[frame] = valid[:args.top_k]
        if not candidates[frame]:
            raise RuntimeError("no valid static-chair offset at frame {}".format(frame))

    zero = (0.0, 0.0, 0.0)
    if zero not in candidates[frames[0]] or zero not in candidates[frames[-1]]:
        raise RuntimeError("the selected range must start and end at an already valid raw frame")
    candidates[frames[0]] = [zero]
    candidates[frames[-1]] = [zero]
    path = np.asarray(choose_continuous_path(candidates, frames, args), dtype=np.float64)

    root = np.asarray(motion["root_pos"], dtype=np.float64)
    corrected_root = root.copy()
    corrected_root[args.start_frame:args.end_frame + 1] += path
    root_steps = np.linalg.norm(np.diff(corrected_root, axis=0), axis=1)
    if float(np.max(root_steps)) > args.max_root_step + 1e-9:
        raise RuntimeError(
            "corrected root step {:.6f} exceeds gate {:.6f}".format(
                float(np.max(root_steps)), args.max_root_step
            )
        )

    # Re-run the exact static acceptance condition on the materialised path.
    audits = []
    for frame, offset in zip(frames, path):
        data.qpos[:] = qpos[frame]
        data.qpos[:3] += offset
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        summary = chair_contact_summary(model, data, robot_body_ids)
        if not accepted(
            summary, frame, args.preseat_end_frame,
            args.max_support_penetration, args.require_seat_support_from_frame,
        ):
            raise RuntimeError("post-materialisation gate failed at frame {}".format(frame))
        audits.append({"frame": frame, "offset_xyz_m": offset.tolist(), **summary})

    output_motion = copy.deepcopy(motion)
    output_motion["root_pos"] = corrected_root.astype(
        np.asarray(motion["root_pos"]).dtype, copy=False
    )
    args.output_motion.parent.mkdir(parents=True, exist_ok=True)
    with args.output_motion.open("wb") as handle:
        pickle.dump(output_motion, handle, protocol=pickle.HIGHEST_PROTOCOL)
    report = {
        "schema_version": 1,
        "status": "accepted_static_chair_clearance_reference",
        "physical_contract": {
            "offline_operation": "reference_only_no_mj_step_or_external_force",
            "runtime_requirement": "initialise_once_then_joint_torque_plus_mj_step",
            "forbidden_runtime_mechanisms": ["root_state_reimposition", "xfrc_applied", "mocap_weld"],
        },
        "inputs": {
            "input_motion": str(args.input_motion),
            "input_motion_sha256": sha256(args.input_motion),
            "task_xml": str(args.task_xml),
            "task_xml_sha256": sha256(args.task_xml),
        },
        "output_motion": str(args.output_motion),
        "settings": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "metrics": {
            "maximum_offset_norm_m": float(np.max(np.linalg.norm(path, axis=1))),
            "maximum_offset_component_step_m": float(np.max(np.abs(np.diff(path, axis=0)))),
            "maximum_corrected_root_step_m": float(np.max(root_steps)),
            "maximum_corrected_root_step_frame": int(np.argmax(root_steps) + 1),
        },
        "frame_audits": audits,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "metrics": report["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
