#!/usr/bin/env python3
"""Make a retargeted HoloMotion reference continuously feasible above its floor.

Human-to-G1 retargeting can place the G1 collision sole a few millimetres below
the world plane even when the source SMPL feet are visually correct.  A tracker
cannot physically follow such a target.  This preflight applies the *smallest
continuous vertical root lift* that makes every active G1 sole patch clear the
floor.  It operates before simulation; the evaluator still sets its initial
state once and subsequently uses only continuous ``mj_step`` integration.

The lift is the Lipschitz majorant of the measured per-frame requirement.  It
therefore cannot introduce a root-height jump and is not a per-frame runtime
state overwrite.  The same script and parameters are used for every clip.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import mujoco
import numpy as np


ANKLE_BODY_BY_SIDE = {
    "left": "left_ankle_roll_link",
    "right": "right_ankle_roll_link",
}


def _read_metadata(archive: np.lib.npyio.NpzFile) -> dict[str, object]:
    raw = archive["metadata"]
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    parsed = json.loads(str(raw))
    if not isinstance(parsed, dict):
        raise ValueError("reference metadata is not a JSON object")
    return parsed


def _qpos_from_reference(
    model: mujoco.MjModel,
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos_reference_order: np.ndarray,
    dof_names: list[str],
) -> np.ndarray:
    frame_count = len(root_pos)
    if root_rot_xyzw.shape != (frame_count, 4):
        raise ValueError("reference root rotations must be [T,4]")
    if dof_pos_reference_order.shape != (frame_count, len(dof_names)):
        raise ValueError("reference DoF positions do not match metadata dof_names")
    free_joint_ids = [
        joint_id for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
    ]
    if len(free_joint_ids) != 1:
        raise ValueError("task must contain exactly one free base")
    qpos = np.zeros((frame_count, model.nq), dtype=np.float64)
    root_address = int(model.jnt_qposadr[free_joint_ids[0]])
    qpos[:, root_address:root_address + 3] = root_pos
    # HoloMotion serialises global rotations as XYZW; MuJoCo qpos is WXYZ.
    qpos[:, root_address + 3:root_address + 7] = root_rot_xyzw[:, [3, 0, 1, 2]]
    for column, name in enumerate(dof_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"task XML lacks reference joint {name!r}")
        qpos[:, int(model.jnt_qposadr[joint_id])] = dof_pos_reference_order[:, column]
    return qpos


def _active_sole_spheres(model: mujoco.MjModel) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for side, body_name in ANKLE_BODY_BY_SIDE.items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"task XML lacks {body_name!r}")
        geom_ids = [
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) == body_id
            and int(model.geom_type[geom_id]) == mujoco.mjtGeom.mjGEOM_SPHERE
            and int(model.geom_contype[geom_id]) != 0
            and int(model.geom_conaffinity[geom_id]) != 0
        ]
        if not geom_ids:
            raise ValueError(f"no active collision-sphere sole patches on {body_name!r}")
        result[side] = geom_ids
    return result


def _sole_floor_clearance(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    floor_id: int,
    sole_spheres: dict[str, list[int]],
) -> dict[str, np.ndarray]:
    data = mujoco.MjData(model)
    values = {side: np.empty(len(qpos), dtype=np.float64) for side in sole_spheres}
    for frame, state in enumerate(qpos):
        data.qpos[:] = state
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for side, geom_ids in sole_spheres.items():
            values[side][frame] = min(
                float(mujoco.mj_geomDistance(model, data, geom_id, floor_id, 0.25, None))
                for geom_id in geom_ids
            )
    return values


def _minimal_lipschitz_majorant(required_lift: np.ndarray, max_step: float) -> np.ndarray:
    """Smallest sampled sequence >= requirement with adjacent change <= max_step."""
    lift = np.asarray(required_lift, dtype=np.float64).copy()
    if max_step <= 0.0:
        raise ValueError("max root-lift step must be positive")
    # Two directional relaxation passes are sufficient for the 1-D cone envelope;
    # loop defensively to make the sampled bound explicit and verifiable.
    for _ in range(3):
        for frame in range(1, len(lift)):
            lift[frame] = max(lift[frame], lift[frame - 1] - max_step)
        for frame in range(len(lift) - 2, -1, -1):
            lift[frame] = max(lift[frame], lift[frame + 1] - max_step)
    if np.any(lift + 1e-12 < required_lift):
        raise RuntimeError("lift envelope no longer covers required floor clearance")
    if len(lift) > 1 and float(np.max(np.abs(np.diff(lift)))) > max_step + 1e-12:
        raise RuntimeError("lift envelope violates requested continuity bound")
    return lift


def _time_derivative(values: np.ndarray, timestep_s: float) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values)
    return np.gradient(values, timestep_s, axis=0, edge_order=1)


def _clearance_summary(values: dict[str, np.ndarray]) -> dict[str, object]:
    return {
        side: {
            "minimum_m": float(np.min(distances)),
            "p01_m": float(np.percentile(distances, 1.0)),
            "worst_frame": int(np.argmin(distances)),
        }
        for side, distances in values.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-reference", required=True, type=Path)
    parser.add_argument("--task-xml", required=True, type=Path)
    parser.add_argument("--output-reference", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--target-sole-clearance-m", type=float, default=0.001,
        help="minimum desired signed sole-floor distance in the kinematic reference",
    )
    parser.add_argument(
        "--max-root-lift-step-m", type=float, default=0.002,
        help="strict bound on the offline root-Z correction between adjacent 50 Hz frames",
    )
    parser.add_argument(
        "--end-frame-exclusive", type=int, default=None,
        help=(
            "optionally apply the floor-feasibility projection only before this frame. "
            "For a semantically evidenced chair-contact cue, the seated segment already has "
            "a separate gravity-validated target and should not be vertically retuned."
        ),
    )
    args = parser.parse_args()
    if args.output_reference.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite floor-cleared reference/report")
    if args.target_sole_clearance_m < 0.0:
        raise ValueError("target-sole-clearance-m must be non-negative")

    with np.load(args.input_reference, allow_pickle=False) as archive:
        required = {
            "metadata", "ref_dof_pos", "ref_global_translation",
            "ref_global_rotation_quat", "ref_global_velocity",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"input reference lacks {sorted(missing)}")
        metadata = _read_metadata(archive)
        dof_names = metadata.get("dof_names")
        if not isinstance(dof_names, list) or not all(isinstance(name, str) for name in dof_names):
            raise ValueError("reference metadata lacks ordered dof_names")
        payload = {key: np.asarray(archive[key]).copy() for key in archive.files if key != "metadata"}
    fps = float(metadata.get("motion_fps", 0.0))
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("reference metadata motion_fps must be positive")
    translation = np.asarray(payload["ref_global_translation"], dtype=np.float64)
    rotation_xyzw = np.asarray(payload["ref_global_rotation_quat"], dtype=np.float64)
    dof_pos = np.asarray(payload["ref_dof_pos"], dtype=np.float64)
    if translation.ndim != 3 or translation.shape[1:] != (30, 3):
        raise ValueError("expected HoloMotion ref_global_translation shape [T,30,3]")
    if rotation_xyzw.shape != (len(translation), 30, 4):
        raise ValueError("reference global rotations do not match translations")

    model = mujoco.MjModel.from_xml_path(str(args.task_xml))
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0 or int(model.geom_type[floor_id]) != mujoco.mjtGeom.mjGEOM_PLANE:
        raise ValueError("task must have a horizontal plane geom named 'floor'")
    sole_spheres = _active_sole_spheres(model)
    before_qpos = _qpos_from_reference(model, translation[:, 0], rotation_xyzw[:, 0], dof_pos, dof_names)
    before = _sole_floor_clearance(model, before_qpos, floor_id, sole_spheres)
    minimum = np.minimum.reduce([before[side] for side in ANKLE_BODY_BY_SIDE])
    end_frame = len(translation) if args.end_frame_exclusive is None else args.end_frame_exclusive
    if not 1 <= end_frame <= len(translation):
        raise ValueError("end-frame-exclusive must be in [1, frame_count]")
    required_lift = np.zeros(len(translation), dtype=np.float64)
    required_lift[:end_frame] = np.maximum(
        0.0, args.target_sole_clearance_m - minimum[:end_frame]
    )
    lift = _minimal_lipschitz_majorant(required_lift, args.max_root_lift_step_m)

    corrected_translation = translation.copy()
    corrected_translation[:, :, 2] += lift[:, None]
    payload["ref_global_translation"] = corrected_translation.astype(np.float32)
    corrected_velocity = np.asarray(payload["ref_global_velocity"], dtype=np.float64).copy()
    if corrected_velocity.shape != corrected_translation.shape:
        raise ValueError("reference global velocity does not match global translation shape")
    corrected_velocity[:, :, 2] = _time_derivative(corrected_translation[:, :, 2], 1.0 / fps)
    payload["ref_global_velocity"] = corrected_velocity.astype(np.float32)

    after_qpos = _qpos_from_reference(
        model, corrected_translation[:, 0], rotation_xyzw[:, 0], dof_pos, dof_names
    )
    after = _sole_floor_clearance(model, after_qpos, floor_id, sole_spheres)
    after_minimum = float(min(np.min(after[side][:end_frame]) for side in after))
    if after_minimum < args.target_sole_clearance_m - 2e-6:
        raise RuntimeError("corrected reference does not satisfy requested sole clearance")

    output_metadata = copy.deepcopy(metadata)
    output_metadata["floor_clearance_projection"] = {
        "mode": "offline_minimal_lipschitz_root_z_majorant",
        "task_xml": str(args.task_xml.resolve()),
        "target_sole_clearance_m": args.target_sole_clearance_m,
        "max_root_lift_step_m": args.max_root_lift_step_m,
        "max_lift_m": float(np.max(lift)),
        "mean_lift_m": float(np.mean(lift)),
        "frames_with_nonzero_lift": int(np.count_nonzero(lift > 1e-9)),
        "projection_frame_range": [0, int(end_frame - 1)],
    }
    payload["metadata"] = np.asarray(json.dumps(output_metadata, ensure_ascii=False))
    args.output_reference.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_reference, **payload)
    report = {
        "schema_version": 1,
        "purpose": "shared_holomotion_reference_floor_clearance_preflight",
        "input_reference": str(args.input_reference.resolve()),
        "output_reference": str(args.output_reference.resolve()),
        "task_xml": str(args.task_xml.resolve()),
        "frame_count": int(len(translation)),
        "projection_frame_range": [0, int(end_frame - 1)],
        "before": _clearance_summary(before),
        "after": _clearance_summary(after),
        "after_projected_range": _clearance_summary(
            {side: distances[:end_frame] for side, distances in after.items()}
        ),
        "root_z_lift": {
            "max_m": float(np.max(lift)),
            "mean_m": float(np.mean(lift)),
            "max_frame_step_m": float(np.max(np.abs(np.diff(lift))) if len(lift) > 1 else 0.0),
            "nonzero_frame_count": int(np.count_nonzero(lift > 1e-9)),
        },
        "physical_contract": {
            "offline_change": "minimal continuous world-Z translation of all reference body globals in the selected approach range",
            "unchanged": ["joint_angles", "root_orientation", "frame_timing", "chair_scene", "actuators"],
            "runtime_forbidden": ["per_frame_qpos_reset", "xfrc_applied", "mocap_weld", "moving_scene_geometry"],
        },
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
