#!/usr/bin/env python3
"""Create a smooth GMR-derived reference ending at an accepted physical seat pose.

This converts an accepted one-time-initialisation gravity landing into a
*reference only*.  It neither simulates nor overwrites the source GMR motion.
The caller must still validate the output with a controller that uses bounded
joint torques and ``mj_step``.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np


def _smoothstep5(progress: np.ndarray) -> np.ndarray:
    progress = np.clip(progress, 0.0, 1.0)
    return progress**3 * (10.0 - 15.0 * progress + 6.0 * progress**2)


def _slerp_xyzw(first: np.ndarray, second: np.ndarray, weight: float) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if float(np.dot(first, second)) < 0.0:
        second = -second
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    if cosine > 0.9995:
        result = first + weight * (second - first)
        return result / np.linalg.norm(result)
    angle = float(np.arccos(cosine))
    sine = float(np.sin(angle))
    result = (np.sin((1.0 - weight) * angle) / sine) * first + (np.sin(weight * angle) / sine) * second
    return result / np.linalg.norm(result)


def _quat_multiply_xyzw(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Compose two unit quaternions stored as ``[x, y, z, w]``."""
    x1, y1, z1, w1 = np.asarray(first, dtype=np.float64)
    x2, y2, z2, w2 = np.asarray(second, dtype=np.float64)
    result = np.asarray((
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ))
    return result / np.linalg.norm(result)


def _quat_inverse_xyzw(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm_squared = float(np.dot(quaternion, quaternion))
    if norm_squared <= 0.0:
        raise ValueError("cannot invert a zero quaternion")
    return np.asarray((-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3])) / norm_squared


def _qpos_to_motion_fields(qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return qpos[:3].copy(), qpos[[4, 5, 6, 3]].copy(), qpos[7:].copy()


def _rebuild_local_body_positions(
    robot_xml: Path, root_pos: np.ndarray, root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray, body_names: np.ndarray,
) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    data = mujoco.MjData(model)
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id < 0:
        raise ValueError("robot XML lacks pelvis")
    body_ids = np.asarray([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(name)) for name in body_names
    ], dtype=np.int32)
    if np.any(body_ids < 0):
        missing = [str(body_names[index]) for index in np.flatnonzero(body_ids < 0)]
        raise ValueError(f"link_body_list has no matching G1 body: {missing}")
    output = np.empty((len(root_pos), len(body_ids), 3), dtype=np.float32)
    for frame in range(len(root_pos)):
        data.qpos[:3] = root_pos[frame]
        data.qpos[3:7] = root_rot_xyzw[frame, [3, 0, 1, 2]]
        data.qpos[7:] = dof_pos[frame]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        root_rotation = data.xmat[pelvis_id].reshape(3, 3)
        output[frame] = (root_rotation.T @ (data.xpos[body_ids] - data.xpos[pelvis_id]).T).T
    return output


def _joint_limit_arrays(
    mjcf_path: Path, dof_names: list[str], margin_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return named hinge limits in the motion-file DOF ordering.

    The reference writer normally preserves its input exactly.  Once it adds a
    terminal physical-seat correction, however, a changing source joint can
    make ``source + smooth_correction`` exceed a limit *between* two feasible
    endpoints.  Query the same MuJoCo model used by the eventual task rather
    than relying on a duplicated limits table.
    """
    if margin_rad < 0.0:
        raise ValueError("joint-limit margin must be non-negative")
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    lower = np.empty(len(dof_names), dtype=np.float64)
    upper = np.empty(len(dof_names), dtype=np.float64)
    for index, name in enumerate(dof_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
        if joint_id < 0:
            raise ValueError(f"joint-limit MJCF lacks reference DOF {name!r}")
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"reference DOF {name!r} is not a hinge in the joint-limit MJCF")
        joint_range = model.jnt_range[joint_id]
        if not bool(model.jnt_limited[joint_id]):
            raise ValueError(f"reference DOF {name!r} has no finite MuJoCo joint limit")
        lower[index] = joint_range[0] + margin_rad
        upper[index] = joint_range[1] - margin_rad
    if np.any(lower > upper):
        raise ValueError("joint-limit margin leaves an empty feasible interval")
    return lower, upper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-motion", required=True, type=Path)
    parser.add_argument("--landing-report", required=True, type=Path)
    parser.add_argument("--robot-xml", required=True, type=Path)
    parser.add_argument("--output-motion", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--prepare-start-frame", type=int, default=90)
    parser.add_argument("--landing-start-frame", type=int, default=179)
    parser.add_argument("--terminal-frame", type=int, default=221)
    parser.add_argument(
        "--joint-limit-mjcf", type=Path,
        help=(
            "MJCF that defines the physical joint limits for a terminal-corrected "
            "reference.  When provided, only the corrected interval is projected "
            "onto these limits; uncorrected source frames are left untouched."
        ),
    )
    parser.add_argument(
        "--joint-limit-margin-rad", type=float, default=0.0,
        help="Optional inward margin applied to --joint-limit-mjcf bounds (default: 0).",
    )
    parser.add_argument(
        "--trajectory-mode",
        choices=(
            "legacy_anchor_blend_v1",
            "source_preserving_correction_v2",
            "source_preserving_tail_correction_v3",
        ),
        default="legacy_anchor_blend_v1",
        help=(
            "legacy_anchor_blend_v1 replaces the tail with two anchor blends; "
            "source_preserving_correction_v2 also matches the verified "
            "landing seed at the sit event; source_preserving_tail_correction_v3 "
            "keeps the sit-event frame unchanged and ramps only the verified "
            "terminal correction across the observed sitting tail"
        ),
    )
    args = parser.parse_args()
    with args.source_motion.open("rb") as handle:
        source = pickle.load(handle)
    for key in ("fps", "root_pos", "root_rot", "dof_pos", "link_body_list"):
        if key not in source:
            raise ValueError(f"source motion is missing {key!r}")
    frame_count = len(source["root_pos"])
    if not 0 <= args.prepare_start_frame < args.landing_start_frame < args.terminal_frame < frame_count:
        raise ValueError("require 0 <= prepare_start < landing_start < terminal < frame_count")
    landing = json.loads(args.landing_report.read_text(encoding="utf-8"))
    if landing.get("status") != "accepted_gravity_seat_landing":
        raise ValueError("landing report must be an accepted gravity-seat baseline")
    summary = landing.get("summary", {})
    initial_qpos = np.asarray(summary.get("initial_qpos_wxyz"), dtype=np.float64)
    settled_qpos = np.asarray(summary.get("settled_final_qpos_wxyz"), dtype=np.float64)
    expected_qpos = 7 + np.asarray(source["dof_pos"]).shape[1]
    if initial_qpos.shape != (expected_qpos,) or settled_qpos.shape != (expected_qpos,):
        raise ValueError("landing qpos dimensionality does not match source GMR motion")

    result = copy.deepcopy(source)
    root_pos = np.asarray(source["root_pos"], dtype=np.float64).copy()
    root_rot = np.asarray(source["root_rot"], dtype=np.float64).copy()
    dof_pos = np.asarray(source["dof_pos"], dtype=np.float64).copy()
    source_root_pos = root_pos.copy()
    source_root_rot = root_rot.copy()
    source_dof_pos = dof_pos.copy()
    initial_root, initial_quat, initial_joint = _qpos_to_motion_fields(initial_qpos)
    settled_root, settled_quat, settled_joint = _qpos_to_motion_fields(settled_qpos)

    def write_segment(first_frame: int, last_frame: int, first: np.ndarray, second: np.ndarray) -> None:
        weights = _smoothstep5(np.linspace(0.0, 1.0, last_frame - first_frame + 1))
        first_root, first_quat, first_joint = _qpos_to_motion_fields(first)
        second_root, second_quat, second_joint = _qpos_to_motion_fields(second)
        for offset, weight in enumerate(weights):
            frame = first_frame + offset
            root_pos[frame] = (1.0 - weight) * first_root + weight * second_root
            root_rot[frame] = _slerp_xyzw(first_quat, second_quat, float(weight))
            dof_pos[frame] = (1.0 - weight) * first_joint + weight * second_joint

    if args.trajectory_mode == "legacy_anchor_blend_v1":
        start_qpos = np.r_[
            root_pos[args.prepare_start_frame],
            root_rot[args.prepare_start_frame, [3, 0, 1, 2]],
            dof_pos[args.prepare_start_frame],
        ]
        # The original draft mode replaces the tail by two endpoint blends.
        write_segment(args.prepare_start_frame, args.landing_start_frame, start_qpos, initial_qpos)
        write_segment(args.landing_start_frame, args.terminal_frame, initial_qpos, settled_qpos)
        anchor_corrections: dict[str, object] = {}
        reference_mode = "smooth_reference_to_accepted_gravity_seat_terminal_v1"
    else:
        identity_quat = np.asarray((0.0, 0.0, 0.0, 1.0))

        def anchor_correction(target_root: np.ndarray, target_quat: np.ndarray,
                              target_joint: np.ndarray, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return (
                target_root - source_root_pos[frame],
                _quat_multiply_xyzw(target_quat, _quat_inverse_xyzw(source_root_rot[frame])),
                target_joint - source_dof_pos[frame],
            )

        if args.trajectory_mode == "source_preserving_correction_v2":
            landing_correction = anchor_correction(
                initial_root, initial_quat, initial_joint, args.landing_start_frame
            )
            reference_mode = "smooth_source_preserving_correction_to_accepted_gravity_seat_terminal_v2"
        else:
            # The verified gravity landing establishes the terminal state, but
            # its initial qpos is sampled from the terminal GMR key.  Applying
            # that terminal joint vector at the observed sit-event frame would
            # overwrite the last part of the original sit-down.  Preserve the
            # source at the event and ramp only the small terminal correction.
            landing_correction = (
                np.zeros(3, dtype=np.float64), identity_quat,
                np.zeros_like(source_dof_pos[0]),
            )
            reference_mode = "smooth_source_preserving_tail_correction_to_accepted_gravity_seat_terminal_v3"
        terminal_correction = anchor_correction(
            settled_root, settled_quat, settled_joint, args.terminal_frame
        )

        def apply_correction_segment(
            first_frame: int,
            last_frame: int,
            first: tuple[np.ndarray, np.ndarray, np.ndarray],
            second: tuple[np.ndarray, np.ndarray, np.ndarray],
        ) -> None:
            weights = _smoothstep5(np.linspace(0.0, 1.0, last_frame - first_frame + 1))
            for offset, weight in enumerate(weights):
                frame = first_frame + offset
                root_delta = (1.0 - weight) * first[0] + weight * second[0]
                joint_delta = (1.0 - weight) * first[2] + weight * second[2]
                delta_quat = _slerp_xyzw(first[1], second[1], float(weight))
                root_pos[frame] = source_root_pos[frame] + root_delta
                root_rot[frame] = _quat_multiply_xyzw(delta_quat, source_root_rot[frame])
                dof_pos[frame] = source_dof_pos[frame] + joint_delta

        zero_correction = (
            np.zeros(3, dtype=np.float64), identity_quat, np.zeros_like(source_dof_pos[0]),
        )
        apply_correction_segment(
            args.prepare_start_frame, args.landing_start_frame,
            zero_correction, landing_correction,
        )
        apply_correction_segment(
            args.landing_start_frame, args.terminal_frame,
            landing_correction, terminal_correction,
        )
        anchor_corrections = {
            "landing_start": {
                "root_translation_m": landing_correction[0].tolist(),
                "joint_l2_delta_rad": float(np.linalg.norm(landing_correction[2])),
            },
            "terminal": {
                "root_translation_m": terminal_correction[0].tolist(),
                "joint_l2_delta_rad": float(np.linalg.norm(terminal_correction[2])),
            },
        }

    joint_limit_projection: dict[str, object] | None = None
    if args.joint_limit_mjcf is not None:
        lower, upper = _joint_limit_arrays(
            args.joint_limit_mjcf,
            [str(name) for name in source["dof_names"]],
            args.joint_limit_margin_rad,
        )
        # Do not silently change the observed GVHMR/GMR prefix.  The only
        # states this writer owns are the tail states it constructed above.
        corrected = slice(args.prepare_start_frame, args.terminal_frame + 1)
        before_projection = dof_pos[corrected].copy()
        dof_pos[corrected] = np.clip(before_projection, lower, upper)
        adjustment = dof_pos[corrected] - before_projection
        changed = np.abs(adjustment) > 1e-10
        changed_frames = np.flatnonzero(np.any(changed, axis=1)) + args.prepare_start_frame
        changed_dofs = np.flatnonzero(np.any(changed, axis=0))
        joint_limit_projection = {
            "mjcf": str(args.joint_limit_mjcf),
            "margin_rad": float(args.joint_limit_margin_rad),
            "corrected_frame_range": [args.prepare_start_frame, args.terminal_frame],
            "changed_frame_count": int(len(changed_frames)),
            "changed_frames": changed_frames.tolist(),
            "changed_dofs": [str(source["dof_names"][index]) for index in changed_dofs],
            "max_abs_adjustment_rad": float(np.max(np.abs(adjustment))) if adjustment.size else 0.0,
        }
    local_body_pos = _rebuild_local_body_positions(
        args.robot_xml, root_pos, root_rot, dof_pos, np.asarray(source["link_body_list"])
    )
    result["root_pos"] = root_pos
    result["root_rot"] = root_rot
    result["dof_pos"] = dof_pos.astype(np.float32)
    result["local_body_pos"] = local_body_pos
    result["physics_reference_mode"] = reference_mode
    result["physics_reference_source_motion"] = str(args.source_motion)
    result["physics_reference_landing_report"] = str(args.landing_report)
    result["physics_reference_frames"] = {
        "prepare_start": args.prepare_start_frame,
        "authoritative_sit_start": args.landing_start_frame,
        "terminal": args.terminal_frame,
    }
    root_step = np.linalg.norm(np.diff(root_pos, axis=0), axis=1)
    report = {
        "schema_version": 1,
        "purpose": "gmr_reference_warp_to_accepted_physical_seat_terminal",
        "status": "draft_reference_requires_mj_step_validation",
        "trajectory_mode": args.trajectory_mode,
        "physical_contract": {
            "changed": "reference_targets_only",
            "unchanged": ["VideoMimic_scene_pose", "source_motion_file"],
            "future_runtime_requirement": "initialize_once_then_bounded_joint_torque_and_mj_step",
            "forbidden_runtime_mechanisms": ["root_state_reimposition", "xfrc_applied", "mocap_weld"],
        },
        "inputs": {"source_motion": str(args.source_motion), "accepted_landing_report": str(args.landing_report)},
        "frames": result["physics_reference_frames"],
        "root_continuity": {"max_step_m": float(np.max(root_step)), "p95_step_m": float(np.quantile(root_step, 0.95))},
        "terminal_qpos_wxyz": settled_qpos.tolist(),
        "anchor_corrections": anchor_corrections,
        "joint_limit_projection": joint_limit_projection,
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
