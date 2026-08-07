#!/usr/bin/env python3
"""Replay a GMR reference against a semantic scene in MuJoCo.

The input GMR motion is a reference, never a state that may be reimposed on a
free base during a physical pass.  Diagnostic modes are retained to compare
legacy renders, but only a torque-only, free-base ``mj_step`` replay can be
reported as physically eligible.  This module is an acceptance harness for a
contact-aware controller; it must not manufacture chair support by applying a
pelvis wrench or a mocap weld.
"""

from __future__ import annotations

import argparse
import json
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from evaluate_gmr_chair_contacts import (
    _body_name,
    _combine_mjcf,
    _load_motion,
    _robot_chair_contact_geoms,
    _robot_collision_geoms,
)
from fit_static_chair_to_frozen_gmr import (
    _render_mesh_geoms,
    _render_mesh_seat_clearance,
)


def _normalise(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    return quat / max(float(np.linalg.norm(quat)), 1e-12)


def _slerp(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    first, second = _normalise(first), _normalise(second)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second, dot = -second, -dot
    if dot > 0.9995:
        return _normalise(first + fraction * (second - first))
    angle = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sine = max(float(np.sin(angle)), 1e-12)
    return _normalise(
        np.sin((1.0 - fraction) * angle) / sine * first
        + np.sin(fraction * angle) / sine * second
    )


def _enable_render_ground_mesh_contacts(mjcf_path: Path, body_names: set[str]) -> int:
    """Mark visible foot meshes before MuJoCo compiles their floor pairs."""
    if not body_names:
        return 0
    tree = ET.parse(mjcf_path)
    matched = 0
    for body in tree.getroot().findall(".//body"):
        if body.get("name") not in body_names:
            continue
        for geom in body.findall("geom"):
            if geom.get("type") == "mesh" and geom.get("group") == "1":
                # Runtime categories below narrow this compiled pair to floor-only.
                geom.set("contype", "1")
                geom.set("conaffinity", "1")
                matched += 1
    if matched:
        tree.write(mjcf_path, encoding="unicode")
    return matched


def _quat_mul(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = first
    w2, x2, y2, z2 = second
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _rotation_error(desired: np.ndarray, actual: np.ndarray) -> np.ndarray:
    error = _quat_mul(_normalise(desired), _normalise(actual) * [1.0, -1.0, -1.0, -1.0])
    if error[0] < 0.0:
        error = -error
    sin_half = float(np.linalg.norm(error[1:]))
    if sin_half < 1e-9:
        return 2.0 * error[1:]
    angle = 2.0 * np.arctan2(sin_half, max(float(error[0]), 1e-12))
    return error[1:] / sin_half * angle


def _reference_qpos(
    model: mujoco.MjModel,
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
) -> np.ndarray:
    qpos = np.empty(model.nq, dtype=np.float64)
    qpos[:3] = root_pos
    qpos[3:7] = root_rot_xyzw[[3, 0, 1, 2]]
    qpos[7:] = dof_pos
    return qpos


def _add_root_mocap_weld(
    combined_xml: Path,
    root_body: str,
    time_constant_s: float,
    damping_ratio: float,
) -> str:
    """Attach a trajectory-driven mocap body to the free pelvis.

    A force-limited pelvis tracker deliberately allows contact forces to alter
    the global trajectory. That is useful for a free-base stress test, but it
    is not valid when the visual reconstruction owns the root trajectory. A
    MuJoCo weld makes that ownership explicit while contacts and all joint
    dynamics are still advanced by ``mj_step``.

    ``combined_xml`` is a per-run temporary MJCF assembled by _combine_mjcf, so
    editing it cannot alter the source robot or scene assets.
    """
    tree = ET.parse(combined_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("combined MJCF has no worldbody")
    if not any(node.get("name") == root_body for node in root.findall(".//body")):
        raise ValueError(f"root body {root_body!r} is absent from combined MJCF")
    mocap_name = "gmr_visual_root_mocap"
    if any(node.get("name") == mocap_name for node in root.findall(".//body")):
        raise ValueError("temporary MJCF already contains the visual-root mocap")
    ET.SubElement(
        worldbody,
        "body",
        {"name": mocap_name, "mocap": "true", "pos": "0 0 0", "quat": "1 0 0 0"},
    )
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    ET.SubElement(
        equality,
        "weld",
        {
            "name": "gmr_visual_root_weld",
            "body1": root_body,
            "body2": mocap_name,
            "solref": f"{time_constant_s:.8g} {damping_ratio:.8g}",
            "solimp": "0.999 0.999 0.001 0.5 2",
        },
    )
    tree.write(combined_xml, encoding="unicode")
    return mocap_name


def _subtree_mass(model: mujoco.MjModel, body_id: int) -> float:
    """Mass of a free-base robot subtree, excluding fixed scene bodies."""
    is_robot_body = np.zeros(model.nbody, dtype=bool)
    for candidate_id in range(1, model.nbody):
        ancestor_id = candidate_id
        while ancestor_id > 0:
            if ancestor_id == body_id:
                is_robot_body[candidate_id] = True
                break
            ancestor_id = int(model.body_parentid[ancestor_id])
    return float(np.sum(model.body_mass[is_robot_body]))


def _set_initial_state(
    data: mujoco.MjData,
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
    initial_qvel: np.ndarray | None = None,
) -> None:
    data.qpos[:3] = root_pos
    data.qpos[3:7] = root_rot_xyzw[[3, 0, 1, 2]]
    data.qpos[7:] = dof_pos
    if initial_qvel is None:
        data.qvel[:] = 0.0
    else:
        if initial_qvel.shape != data.qvel.shape:
            raise ValueError(
                f"initial qvel shape {initial_qvel.shape} does not match {data.qvel.shape}"
            )
        data.qvel[:] = initial_qvel


def _record_state(data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        data.qpos[:3].copy(),
        data.qpos[3:7][[1, 2, 3, 0]].copy(),
        data.qpos[7:].copy(),
    )


def _chair_contact_force(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    chair_geom_ids: set[int],
) -> tuple[int, float]:
    count, peak = 0, 0.0
    force = np.zeros(6, dtype=np.float64)
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        if int(contact.geom1) not in chair_geom_ids and int(contact.geom2) not in chair_geom_ids:
            continue
        mujoco.mj_contactForce(model, data, contact_id, force)
        count += 1
        peak = max(peak, float(np.linalg.norm(force[:3])))
    return count, peak


def _semantic_seat_support_contact_depth(
    data: mujoco.MjData,
    seat_geom_id: int,
    support_geom_ids: set[int],
) -> float | None:
    """Smallest signed narrow-phase depth for an actual seat-support contact.

    This is deliberately read from ``data.contact`` rather than
    ``mj_geomDistance``.  The latter produced a false large negative distance
    for a distant G1 wrist proxy in this model, while the contact solver had no
    such pair.  A missing contact is represented as ``None`` (not as a guessed
    positive gap), because it cannot certify that the robot is sitting.
    """
    depths: list[float] = []
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        pair = {int(contact.geom1), int(contact.geom2)}
        if seat_geom_id in pair and pair & support_geom_ids:
            depths.append(float(contact.dist))
    return min(depths) if depths else None


def _semantic_seat_support_normal_force(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    seat_geom_id: int,
    support_geom_ids: set[int],
) -> float:
    """Return the normal load of permitted robot--seat contacts in Newtons.

    ``mj_contactForce`` is queried only after MuJoCo has formed its narrow
    phase contacts.  This is a measurement, never a commanded downward force;
    it distinguishes a robot that is really supported from a visual contact
    with nearly zero reaction load.
    """
    total = 0.0
    force = np.empty(6, dtype=np.float64)
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        pair = {int(contact.geom1), int(contact.geom2)}
        if seat_geom_id not in pair or not (pair & support_geom_ids):
            continue
        mujoco.mj_contactForce(model, data, contact_id, force)
        total += max(0.0, float(force[0]))
    return total


def _semantic_chair_min_contact_depth(
    data: mujoco.MjData,
    chair_geom_ids: set[int],
    robot_geom_ids: set[int],
) -> float | None:
    """Return the worst actual robot--chair overlap from MuJoCo's contacts."""
    depths: list[float] = []
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        pair = {int(contact.geom1), int(contact.geom2)}
        if pair & chair_geom_ids and pair & robot_geom_ids:
            depths.append(float(contact.dist))
    return min(depths) if depths else None


def _has_semantic_seat_support_contact(
    data: mujoco.MjData,
    seat_geom_id: int,
    support_geom_ids: set[int],
) -> bool:
    """Return whether a permitted support link is already touching the seat."""
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        pair = {int(contact.geom1), int(contact.geom2)}
        if seat_geom_id in pair and pair & support_geom_ids:
            return True
    return False


def _chair_contacted_robot_geom_ids(
    data: mujoco.MjData,
    chair_geom_ids: set[int],
    robot_geom_ids: set[int],
) -> set[int]:
    """Return robot collision proxies in an actual semantic-chair contact."""
    contacted: set[int] = set()
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        first, second = int(contact.geom1), int(contact.geom2)
        if first in chair_geom_ids and second in robot_geom_ids:
            contacted.add(second)
        elif second in chair_geom_ids and first in robot_geom_ids:
            contacted.add(first)
    return contacted


def _body_is_ancestor(
    model: mujoco.MjModel,
    possible_ancestor: int,
    body_id: int,
) -> bool:
    """Whether a joint child body can kinematically move ``body_id``."""
    current = int(body_id)
    while current > 0:
        if current == possible_ancestor:
            return True
        current = int(model.body_parentid[current])
    return False


def _configure_joint_torque_actuators(
    model: mujoco.MjModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Expose each direct motor's declared joint-force range as its ctrl range.

    The GMR visual MJCF intentionally uses ``[-1, 1]`` controls because it is
    normally driven by direct qpos rendering.  Its joints nevertheless declare
    physical actuator-force limits.  A dynamics validator must use those limits;
    otherwise a requested 80 Nm torque is silently clipped to 1 Nm by MuJoCo.
    """
    command_scale = np.empty(model.nu, dtype=np.float64)
    torque_lower = np.empty(model.nu, dtype=np.float64)
    torque_upper = np.empty(model.nu, dtype=np.float64)
    source_ctrlrange = model.actuator_ctrlrange.copy()
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0 or not bool(model.jnt_actfrclimited[joint_id]):
            raise ValueError(
                "each GMR actuator must target a joint with an actuatorfrcrange"
            )
        gear = float(model.actuator_gear[actuator_id, 0])
        gain = float(model.actuator_gainprm[actuator_id, 0])
        if abs(gear) < 1e-9 or abs(gain) < 1e-9:
            raise ValueError(f"actuator {actuator_id} has zero torque scale")
        scale = gear * gain
        lower, upper = np.asarray(model.jnt_actfrcrange[joint_id], dtype=np.float64)
        ctrl_bounds = np.sort(np.asarray([lower / scale, upper / scale]))
        model.actuator_ctrllimited[actuator_id] = 1
        model.actuator_ctrlrange[actuator_id] = ctrl_bounds
        command_scale[actuator_id] = scale
        torque_lower[actuator_id] = lower
        torque_upper[actuator_id] = upper
    details = {
        "mode": "direct_joint_torque_with_declared_joint_actuatorfrcrange",
        "source_ctrlrange_min": float(np.min(source_ctrlrange)),
        "source_ctrlrange_max": float(np.max(source_ctrlrange)),
        "effective_joint_torque_min_nm": float(np.min(torque_lower)),
        "effective_joint_torque_max_nm": float(np.max(torque_upper)),
    }
    return command_scale, torque_lower, torque_upper, details


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Advance a GMR trajectory with MuJoCo mj_step and scene contacts."
    )
    parser.add_argument("--robot-motion", required=True, type=Path)
    parser.add_argument("--scene-mujoco-xml", required=True, type=Path)
    parser.add_argument("--robot-xml", required=True, type=Path)
    parser.add_argument(
        "--contact-anchors", type=Path, default=None,
        help="Required only for semantic-seat validation; omit for generic scene replay.",
    )
    parser.add_argument("--output-motion", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--joint-kp", type=float, default=100.0)
    parser.add_argument("--joint-kd", type=float, default=5.0)
    parser.add_argument("--joint-torque-limit", type=float, default=80.0)
    parser.add_argument("--root-pos-kp", type=float, default=3000.0)
    parser.add_argument("--root-pos-kd", type=float, default=180.0)
    parser.add_argument("--root-force-limit", type=float, default=1600.0)
    parser.add_argument("--root-rot-kp", type=float, default=500.0)
    parser.add_argument("--root-rot-kd", type=float, default=45.0)
    parser.add_argument("--root-torque-limit", type=float, default=320.0)
    parser.add_argument(
        "--root-tracking-mode",
        choices=("none", "force", "mocap_weld", "kinematic"),
        default="force",
        help=(
            "none uses only joint torques and physical contacts; force is a "
            "bounded diagnostic tracking harness; mocap_weld and kinematic own "
            "the visual root and are not autonomous dynamics."
        ),
    )
    parser.add_argument(
        "--joint-tracking-mode",
        choices=("pd", "kinematic"),
        default="pd",
        help=(
            "pd emits a free joint response; kinematic treats the visual joint "
            "trajectory as a boundary while still advancing contacts with mj_step."
        ),
    )
    parser.add_argument("--root-weld-time-constant-s", type=float, default=0.002)
    parser.add_argument("--root-weld-damping-ratio", type=float, default=1.0)
    parser.add_argument("--gravity-compensation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--root-gravity-compensation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override the legacy gravity-compensation switch for the external pelvis force only",
    )
    parser.add_argument(
        "--joint-gravity-compensation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override the legacy gravity-compensation switch for actuated joint torque only",
    )
    parser.add_argument("--initialize-reference-velocity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-ground-contact", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ground-geom", default="floor")
    parser.add_argument("--ground-height-m", type=float, default=None)
    parser.add_argument(
        "--ground-contact-surface", choices=("collision_proxy", "render_mesh", "both"),
        default="collision_proxy", help="floor contact uses proxies, visible meshes, or both",
    )
    parser.add_argument("--ground-contact-bodies", default="left_ankle_roll_link,right_ankle_roll_link")
    parser.add_argument(
        "--stance-task-mode",
        choices=("none", "reference_stance"),
        default="none",
        help=(
            "reference_stance adds joint-space task torques that hold source-"
            "detected ground feet in place; it requires a free base with no "
            "external root tracker"
        ),
    )
    parser.add_argument("--stance-foot-bodies", default="left_ankle_roll_link,right_ankle_roll_link")
    parser.add_argument("--stance-reference-height-m", type=float, default=0.14)
    parser.add_argument("--stance-reference-speed-mps", type=float, default=0.25)
    parser.add_argument("--stance-position-kp", type=float, default=600.0)
    parser.add_argument("--stance-position-kd", type=float, default=55.0)
    parser.add_argument("--stance-force-limit-n", type=float, default=500.0)
    parser.add_argument("--inertia-scaled-joint-gains", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--joint-inertia-scale-min", type=float, default=0.02)
    parser.add_argument("--joint-inertia-scale-max", type=float, default=25.0)
    parser.add_argument("--seat-geom", default="seat_support_geom")
    parser.add_argument(
        "--chair-geom-names",
        default=(
            "seat_support_geom,backrest_geom,leg_front_left_geom,"
            "leg_front_right_geom,leg_back_left_geom,leg_back_right_geom"
        ),
        help="comma-separated semantic chair geoms that must be collidable in full-chair mode",
    )
    parser.add_argument(
        "--full-chair-collision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="make every robot collision proxy collide with seat, backrest, and chair legs",
    )
    parser.add_argument(
        "--prevent-downward-root-force-on-seat-contact",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "after a permitted support link reaches the semantic seat, do not let "
            "the virtual pelvis tracker push the robot further downward"
        ),
    )
    parser.add_argument(
        "--seat-contact-joint-gain-scale",
        type=float,
        default=1.0,
        help="scale PD torque only for joints whose child body is a permitted seat-support body",
    )
    parser.add_argument(
        "--chair-contact-joint-gain-scale",
        type=float,
        default=1.0,
        help=(
            "scale PD torque for every actuator upstream of a robot proxy that "
            "is actually contacting any semantic chair component"
        ),
    )
    parser.add_argument(
        "--freeze-root-horizontal-after-seat-contact",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "hold the virtual pelvis tracker's x/y target at the first actual "
            "semantic-seat support contact instead of driving the reference base "
            "through the chair"
        ),
    )
    parser.add_argument(
        "--enable-seat-contact", action=argparse.BooleanOptionalAction, default=True,
        help="Require a named semantic seat/backrest and evaluate seat contact.",
    )
    parser.add_argument("--seat-contact-bodies", default="pelvis,left_hip_pitch_link,left_hip_roll_link,right_hip_pitch_link,right_hip_roll_link")
    parser.add_argument(
        "--seat-contact-surface", choices=("collision_proxy", "render_mesh"), default="collision_proxy",
        help="validate semantic seat clearance with conservative boxes or visible render meshes",
    )
    parser.add_argument("--support-footprint-margin-m", type=float, default=0.01)
    parser.add_argument(
        "--min-seat-support-frame-ratio", type=float, default=0.5,
        help="minimum fraction of validated support frames that must project onto the semantic seat in render-mesh mode",
    )
    parser.add_argument(
        "--min-seat-support-normal-force-n", type=float, default=20.0,
        help=(
            "minimum measured normal reaction on permitted robot--seat contacts "
            "for a stable sit frame; this is an acceptance threshold, never a command"
        ),
    )
    parser.add_argument(
        "--settled-tail-frames", type=int, default=20,
        help="for render-mesh replay, validate the stable tail of the final sit segment rather than descent frames",
    )
    parser.add_argument("--backrest-geom", default="backrest_geom")
    parser.add_argument(
        "--max-penetration-m",
        type=float,
        default=0.005,
        help=(
            "Material penetration tolerance.  Sub-millimetre signed-distance "
            "noise is reported separately and must not be counted as a full "
            "penetration-frame failure."
        ),
    )
    parser.add_argument(
        "--max-penetration-frame-ratio", type=float, default=0.0,
        help="Maximum fraction of sit frames with negative semantic-seat clearance.",
    )
    parser.add_argument("--max-root-rmse-m", type=float, default=0.08)
    parser.add_argument("--max-joint-median-rad", type=float, default=0.20)
    parser.add_argument(
        "--max-root-z-frame-jump-m", type=float, default=0.04,
        help="maximum consecutive rendered root-height change allowed in a physical pass",
    )
    args = parser.parse_args()
    if args.substeps < 1:
        raise ValueError("substeps must be positive")
    if min(args.joint_kp, args.joint_kd, args.root_pos_kp, args.root_pos_kd) < 0.0:
        raise ValueError("PD gains must be non-negative")
    if args.joint_inertia_scale_min <= 0.0 or args.joint_inertia_scale_max < args.joint_inertia_scale_min:
        raise ValueError("joint inertia gain scale bounds must satisfy 0 < min <= max")
    if args.root_weld_time_constant_s <= 0.0 or args.root_weld_damping_ratio <= 0.0:
        raise ValueError("root weld parameters must be positive")
    if not 0.0 <= args.max_penetration_frame_ratio <= 1.0:
        raise ValueError("max-penetration-frame-ratio must be in [0, 1]")
    if args.support_footprint_margin_m < 0.0:
        raise ValueError("support-footprint-margin-m must be non-negative")
    if not 0.0 <= args.min_seat_support_frame_ratio <= 1.0:
        raise ValueError("min-seat-support-frame-ratio must be in [0, 1]")
    if args.min_seat_support_normal_force_n < 0.0:
        raise ValueError("min-seat-support-normal-force-n must be non-negative")
    if args.max_root_z_frame_jump_m <= 0.0:
        raise ValueError("max-root-z-frame-jump-m must be positive")
    if args.settled_tail_frames < 4:
        raise ValueError("settled-tail-frames must be at least 4")
    if not 0.0 < args.seat_contact_joint_gain_scale <= 1.0:
        raise ValueError("seat-contact-joint-gain-scale must be in (0, 1]")
    if not 0.0 <= args.chair_contact_joint_gain_scale <= 1.0:
        raise ValueError("chair-contact-joint-gain-scale must be in [0, 1]")
    if args.stance_task_mode != "none":
        if args.root_tracking_mode != "none":
            raise ValueError("stance-task-mode requires root-tracking-mode=none")
        if not args.enable_ground_contact:
            raise ValueError("stance-task-mode requires --enable-ground-contact")
        if (
            args.stance_reference_height_m <= 0.0
            or args.stance_reference_speed_mps <= 0.0
            or args.stance_position_kp < 0.0
            or args.stance_position_kd < 0.0
            or args.stance_force_limit_n <= 0.0
        ):
            raise ValueError("invalid stance-task threshold, gain, or force limit")
    root_gravity_compensation = (
        bool(args.gravity_compensation)
        if args.root_gravity_compensation is None
        else bool(args.root_gravity_compensation)
    )
    joint_gravity_compensation = (
        bool(args.gravity_compensation)
        if args.joint_gravity_compensation is None
        else bool(args.joint_gravity_compensation)
    )

    motion = _load_motion(args.robot_motion)
    root_target = np.asarray(motion["root_pos"], dtype=np.float64)
    root_rot_target_xyzw = np.asarray(motion["root_rot"], dtype=np.float64)
    joint_target = np.asarray(motion["dof_pos"], dtype=np.float64)
    if root_target.ndim != 2 or root_target.shape[1] != 3:
        raise ValueError("root_pos must be [T,3]")
    if root_rot_target_xyzw.shape != (len(root_target), 4):
        raise ValueError("root_rot must be [T,4]")
    if joint_target.shape[0] != len(root_target):
        raise ValueError("dof_pos frame count differs from root_pos")
    fps = float(np.asarray(motion.get("fps", 30.0)).reshape(-1)[0])
    if fps <= 0.0:
        raise ValueError("motion fps must be positive")

    if args.contact_anchors is None:
        if args.enable_seat_contact:
            raise ValueError("--contact-anchors is required when --enable-seat-contact")
        sit_mask = np.zeros(len(root_target), dtype=bool)
    else:
        anchors = np.load(args.contact_anchors)
        if "sit_mask" not in anchors:
            raise ValueError("contact anchors are missing sit_mask")
        sit_mask = np.asarray(anchors["sit_mask"], dtype=bool)
        if sit_mask.shape != (len(root_target),):
            raise ValueError("contact anchors do not match robot motion")

    combined_xml = _combine_mjcf(args.robot_xml, args.scene_mujoco_xml)
    ground_render_body_names = {name.strip() for name in args.ground_contact_bodies.split(",") if name.strip()}
    compiled_ground_render_meshes = 0
    if args.enable_ground_contact and args.ground_contact_surface in {"render_mesh", "both"}:
        compiled_ground_render_meshes = _enable_render_ground_mesh_contacts(combined_xml, ground_render_body_names)
    mocap_body_name = None
    if args.root_tracking_mode == "mocap_weld":
        mocap_body_name = _add_root_mocap_weld(
            combined_xml, "pelvis", args.root_weld_time_constant_s, args.root_weld_damping_ratio
        )
    try:
        model = mujoco.MjModel.from_xml_path(str(combined_xml))
        data = mujoco.MjData(model)
        if model.nq != 7 + joint_target.shape[1] or model.nu != joint_target.shape[1]:
            raise ValueError(
                "this validator requires direct one-motor-per-GMR-DOF support: "
                f"nq={model.nq}, nu={model.nu}, GMR DOFs={joint_target.shape[1]}"
            )
        pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if pelvis_id < 0:
            raise ValueError("pelvis body missing from robot model")
        mocap_id = None
        if mocap_body_name is not None:
            mocap_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mocap_body_name)
            mocap_id = int(model.body_mocapid[mocap_body_id]) if mocap_body_id >= 0 else -1
            if mocap_id < 0:
                raise ValueError("visual-root mocap body was not compiled as a mocap body")
        seat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, args.seat_geom) if args.enable_seat_contact else -1
        back_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, args.backrest_geom) if args.enable_seat_contact else -1
        if args.enable_seat_contact and (seat_id < 0 or back_id < 0):
            raise ValueError("chair seat/backrest collision geom missing")
        chair_geom_names = {
            name.strip() for name in args.chair_geom_names.split(",") if name.strip()
        }
        if args.enable_seat_contact and args.full_chair_collision:
            if args.seat_geom not in chair_geom_names or args.backrest_geom not in chair_geom_names:
                raise ValueError("full-chair collision names must include the seat and backrest")
            chair_ids = {
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                for name in chair_geom_names
            }
            missing_chair_geoms = [
                name
                for name in chair_geom_names
                if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) < 0
            ]
            if missing_chair_geoms:
                raise ValueError(
                    "full-chair collision geoms are absent: "
                    + ", ".join(sorted(missing_chair_geoms))
                )
            chair_ids = {int(geom_id) for geom_id in chair_ids}
        else:
            chair_ids = {int(seat_id), int(back_id)} if args.enable_seat_contact else set()
        robot_geoms = sorted(
            set(_robot_chair_contact_geoms(model)) | set(_robot_collision_geoms(model))
        )
        seat_contact_bodies = {name.strip() for name in args.seat_contact_bodies.split(",") if name.strip()}
        seat_support_geoms = [
            geom_id for geom_id in robot_geoms
            if _body_name(model, geom_id) in seat_contact_bodies
        ]
        if args.enable_seat_contact and not seat_support_geoms:
            raise ValueError("seat-contact-bodies selects no robot collision geometry")
        render_support_geoms = [
            geom_id for geom_id in _render_mesh_geoms(model)
            if _body_name(model, geom_id) in seat_contact_bodies
        ]
        if args.enable_seat_contact and args.seat_contact_surface == "render_mesh" and not render_support_geoms:
            raise ValueError("seat-contact-bodies selects no visible render mesh")
        seat_excluded_geoms = [geom_id for geom_id in robot_geoms if geom_id not in seat_support_geoms]
        render_ground_geoms = [geom_id for geom_id in _render_mesh_geoms(model) if _body_name(model, geom_id) in ground_render_body_names]
        if args.enable_ground_contact and args.ground_contact_surface in {"render_mesh", "both"} and not render_ground_geoms:
            raise ValueError("ground-contact-bodies selects no visible robot mesh")

        seat_bit, support_bit, back_bit, other_bit, floor_bit, render_ground_bit, chair_bit = 1, 2, 4, 8, 16, 32, 64
        if args.enable_seat_contact:
            if args.full_chair_collision:
                if args.seat_contact_surface != "collision_proxy":
                    raise ValueError(
                        "full-chair collision requires seat-contact-surface=collision_proxy"
                    )
                for geom_id in chair_ids:
                    model.geom_contype[geom_id] = chair_bit
                    model.geom_conaffinity[geom_id] = support_bit | other_bit
                for geom_id in seat_support_geoms:
                    model.geom_contype[geom_id] = support_bit
                    model.geom_conaffinity[geom_id] = chair_bit | floor_bit
                for geom_id in seat_excluded_geoms:
                    model.geom_contype[geom_id] = other_bit
                    model.geom_conaffinity[geom_id] = chair_bit | floor_bit
            else:
                model.geom_contype[seat_id] = seat_bit
                model.geom_conaffinity[seat_id] = 0 if args.seat_contact_surface == "render_mesh" else support_bit
                model.geom_contype[back_id] = back_bit
                model.geom_conaffinity[back_id] = support_bit | other_bit
                for geom_id in seat_support_geoms:
                    model.geom_contype[geom_id] = support_bit
                    model.geom_conaffinity[geom_id] = seat_bit | back_bit | floor_bit
                for geom_id in seat_excluded_geoms:
                    model.geom_contype[geom_id] = other_bit
                    model.geom_conaffinity[geom_id] = back_bit | floor_bit
        else:
            for geom_id in robot_geoms:
                model.geom_contype[geom_id] = other_bit
                model.geom_conaffinity[geom_id] = seat_bit | floor_bit
        if args.ground_contact_surface == "render_mesh":
            for geom_id in robot_geoms:
                model.geom_conaffinity[geom_id] &= ~floor_bit
        for geom_id in render_ground_geoms:
            model.geom_contype[geom_id] = render_ground_bit
            model.geom_conaffinity[geom_id] = floor_bit
        (
            torque_command_scale,
            actuator_torque_lower,
            actuator_torque_upper,
            actuator_details,
        ) = _configure_joint_torque_actuators(model)
        actuator_dof_ids = np.asarray(
            [
                model.jnt_dofadr[int(model.actuator_trnid[actuator_id, 0])]
                for actuator_id in range(model.nu)
            ],
            dtype=np.int32,
        )
        support_actuator_mask = np.asarray(
            [
                mujoco.mj_id2name(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    int(model.jnt_bodyid[int(model.actuator_trnid[actuator_id, 0])]),
                ) in seat_contact_bodies
                for actuator_id in range(model.nu)
            ],
            dtype=bool,
        )
        chair_contact_actuator_masks = {
            int(geom_id): np.asarray(
                [
                    _body_is_ancestor(
                        model,
                        int(model.jnt_bodyid[int(model.actuator_trnid[actuator_id, 0])]),
                        int(model.geom_bodyid[geom_id]),
                    )
                    for actuator_id in range(model.nu)
                ],
                dtype=bool,
            )
            for geom_id in robot_geoms
        }
        if (
            args.enable_seat_contact
            and args.seat_contact_joint_gain_scale < 1.0
            and not bool(np.any(support_actuator_mask))
        ):
            raise ValueError(
                "seat-contact-joint-gain-scale selects no actuated support joints"
            )
        if (
            args.enable_seat_contact
            and args.chair_contact_joint_gain_scale < 1.0
            and not any(bool(np.any(mask)) for mask in chair_contact_actuator_masks.values())
        ):
            raise ValueError("chair-contact gain scaling selects no robot actuator")
        full_mass_matrix = np.empty((model.nv, model.nv), dtype=np.float64)
        model.opt.timestep = 1.0 / (fps * args.substeps)

        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, args.ground_geom)
        if args.enable_ground_contact:
            if floor_id < 0:
                raise ValueError(f"ground collision requested but geom {args.ground_geom!r} is absent")
            model.geom_contype[floor_id] = floor_bit
            model.geom_conaffinity[floor_id] = support_bit | other_bit | render_ground_bit
            if args.ground_height_m is not None:
                model.geom_pos[floor_id, 2] = args.ground_height_m
        ground_report = {
            "requested": bool(args.enable_ground_contact),
            "geom": args.ground_geom,
            "geom_found": bool(floor_id >= 0),
            "enabled": bool(args.enable_ground_contact and floor_id >= 0),
            "height_m": float(model.geom_pos[floor_id, 2]) if floor_id >= 0 else None,
            "surface": args.ground_contact_surface,
            "render_mesh_bodies": sorted(ground_render_body_names),
            "compiled_render_mesh_count": int(compiled_ground_render_meshes),
        }
        stance_body_names = {
            name.strip() for name in args.stance_foot_bodies.split(",") if name.strip()
        }
        stance_body_ids: list[int] = []
        reference_stance_mask = np.zeros((len(root_target), 0), dtype=bool)
        reference_stance_targets = np.empty((len(root_target), 0, 3), dtype=np.float64)
        if args.stance_task_mode == "reference_stance":
            for name in sorted(stance_body_names):
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                if body_id < 0:
                    raise ValueError(f"stance-foot body {name!r} is absent")
                stance_body_ids.append(int(body_id))
            reference_data = mujoco.MjData(model)
            reference_foot_pos = np.empty(
                (len(root_target), len(stance_body_ids), 3), dtype=np.float64
            )
            for frame in range(len(root_target)):
                _set_initial_state(
                    reference_data, root_target[frame], root_rot_target_xyzw[frame],
                    joint_target[frame],
                )
                mujoco.mj_forward(model, reference_data)
                reference_foot_pos[frame] = reference_data.xpos[stance_body_ids]
            reference_foot_velocity = np.gradient(reference_foot_pos, 1.0 / fps, axis=0)
            floor_height = float(model.geom_pos[floor_id, 2])
            reference_stance_mask = (
                (reference_foot_pos[:, :, 2] <= floor_height + args.stance_reference_height_m)
                & (np.linalg.norm(reference_foot_velocity, axis=2) <= args.stance_reference_speed_mps)
            )
            reference_stance_targets = np.full_like(reference_foot_pos, np.nan)
            for foot_index in range(len(stance_body_ids)):
                segment_target: np.ndarray | None = None
                for frame in range(len(root_target)):
                    if reference_stance_mask[frame, foot_index]:
                        if not reference_stance_mask[max(frame - 1, 0), foot_index]:
                            segment_target = reference_foot_pos[frame, foot_index].copy()
                        if segment_target is None:
                            segment_target = reference_foot_pos[frame, foot_index].copy()
                        reference_stance_targets[frame, foot_index] = segment_target
                    else:
                        segment_target = None
        stance_report = {
            "mode": args.stance_task_mode,
            "body_names": sorted(stance_body_names),
            "body_ids": stance_body_ids,
            "source_reference_stance_frames_per_body": (
                reference_stance_mask.sum(axis=0).astype(int).tolist()
                if args.stance_task_mode == "reference_stance" else []
            ),
            "position_kp": args.stance_position_kp if args.stance_task_mode != "none" else None,
            "position_kd": args.stance_position_kd if args.stance_task_mode != "none" else None,
            "force_limit_n": args.stance_force_limit_n if args.stance_task_mode != "none" else None,
        }
        initial_qvel = np.zeros(model.nv, dtype=np.float64)
        if args.initialize_reference_velocity and len(root_target) > 1:
            mujoco.mj_differentiatePos(
                model, initial_qvel, 1.0 / fps,
                _reference_qpos(model, root_target[0], root_rot_target_xyzw[0], joint_target[0]),
                _reference_qpos(model, root_target[1], root_rot_target_xyzw[1], joint_target[1]),
            )
        _set_initial_state(
            data, root_target[0], root_rot_target_xyzw[0], joint_target[0], initial_qvel
        )
        if mocap_id is not None:
            data.mocap_pos[mocap_id] = root_target[0]
            data.mocap_quat[mocap_id] = root_rot_target_xyzw[0][[3, 0, 1, 2]]
        mujoco.mj_forward(model, data)
        robot_mass_kg = _subtree_mass(model, int(pelvis_id))
        gravity_support_force = -robot_mass_kg * model.opt.gravity
        mujoco.mj_fullM(model, full_mass_matrix, data.qM)
        reference_joint_inertia = np.diag(full_mass_matrix)[actuator_dof_ids]
        inertia_reference = float(np.median(reference_joint_inertia))
        if inertia_reference <= 0.0:
            raise ValueError("non-positive reference joint inertia")
        gain_scale_min_seen, gain_scale_max_seen = 1.0, 1.0
        actual_root = np.empty_like(root_target)
        actual_rot = np.empty_like(root_rot_target_xyzw)
        actual_joint = np.empty_like(joint_target)
        actual_root[0], actual_rot[0], actual_joint[0] = _record_state(data)
        seat_min_distance = np.full(len(root_target), np.nan, dtype=np.float64)
        proxy_min_distance = np.full(len(root_target), np.nan, dtype=np.float64)
        contact_counts = np.zeros(len(root_target), dtype=np.int32)
        contact_force_peak = np.zeros(len(root_target), dtype=np.float64)
        seat_support_normal_force = np.zeros(len(root_target), dtype=np.float64)
        chair_min_contact_depth = np.full(len(root_target), np.nan, dtype=np.float64)
        seat_support_contact_substeps = 0
        root_downward_force_suppressed_substeps = 0
        support_joint_gain_scaled_substeps = 0
        chair_contact_joint_gain_scaled_substeps = 0
        root_xy_contact_anchor: np.ndarray | None = None
        root_xy_freeze_frame: int | None = None
        stance_task_substeps = 0
        stance_task_peak_force_n = 0.0
        stance_jacobian = np.empty((3, model.nv), dtype=np.float64)
        max_external_wrench_norm_n = 0.0
        root_state_reimposition_substeps = 0
        joint_state_reimposition_substeps = 0

        def sample(frame: int) -> None:
            if not args.enable_seat_contact or not sit_mask[frame]:
                return
            proxy_distances = [
                float(mujoco.mj_geomDistance(model, data, geom_id, int(seat_id), 2.0, None))
                for geom_id in seat_support_geoms
            ]
            proxy_min_distance[frame] = min(proxy_distances)
            if args.seat_contact_surface == "render_mesh":
                try:
                    seat_min_distance[frame] = _render_mesh_seat_clearance(
                        model, data, render_support_geoms, int(seat_id), args.support_footprint_margin_m,
                    )
                except RuntimeError as error:
                    if "no vertices above the seat footprint" not in str(error):
                        raise
                    # Sit labels can include the descent onto a chair.  A mesh
                    # that has not yet projected over the finite seat is not a
                    # support measurement, not a negative clearance.
                    seat_min_distance[frame] = np.nan
            else:
                contact_depth = _semantic_seat_support_contact_depth(
                    data, int(seat_id), set(seat_support_geoms)
                )
                seat_min_distance[frame] = (
                    float(contact_depth) if contact_depth is not None else np.nan
                )
            contact_counts[frame], contact_force_peak[frame] = _chair_contact_force(
                model, data, chair_ids
            )
            seat_support_normal_force[frame] = _semantic_seat_support_normal_force(
                model, data, int(seat_id), set(seat_support_geoms)
            )
            depth = _semantic_chair_min_contact_depth(
                data, chair_ids, set(robot_geoms)
            )
            chair_min_contact_depth[frame] = float(depth) if depth is not None else np.nan

        sample(0)
        frame_period = 1.0 / fps
        for frame in range(len(root_target) - 1):
            linear_velocity = (root_target[frame + 1] - root_target[frame]) / frame_period
            joint_velocity = (joint_target[frame + 1] - joint_target[frame]) / frame_period
            start_quat = root_rot_target_xyzw[frame][[3, 0, 1, 2]]
            end_quat = root_rot_target_xyzw[frame + 1][[3, 0, 1, 2]]
            for substep in range(args.substeps):
                alpha = (substep + 1.0) / args.substeps
                desired_pos = (1.0 - alpha) * root_target[frame] + alpha * root_target[frame + 1]
                desired_joint = (1.0 - alpha) * joint_target[frame] + alpha * joint_target[frame + 1]
                desired_quat = _slerp(start_quat, end_quat, alpha)
                control_linear_velocity = linear_velocity.copy()
                chair_contact_robot_geoms = _chair_contacted_robot_geom_ids(
                    data, chair_ids, set(robot_geoms)
                ) if args.enable_seat_contact else set()
                if args.root_tracking_mode == "kinematic":
                    data.qpos[:3] = desired_pos
                    data.qpos[3:7] = desired_quat
                    data.qvel[:3] = linear_velocity
                    data.qvel[3:6] = 0.0
                    root_state_reimposition_substeps += 1
                if args.joint_tracking_mode == "kinematic":
                    data.qpos[7:] = desired_joint
                    data.qvel[6:] = joint_velocity
                    joint_state_reimposition_substeps += 1
                if (
                    args.root_tracking_mode == "kinematic"
                    or args.joint_tracking_mode == "kinematic"
                ):
                    mujoco.mj_forward(model, data)
                seat_support_active = bool(
                    args.enable_seat_contact
                    and (sit_mask[frame] or sit_mask[frame + 1])
                    and _has_semantic_seat_support_contact(
                        data,
                        int(seat_id),
                        set(seat_support_geoms),
                    )
                )
                if seat_support_active:
                    seat_support_contact_substeps += 1
                    if (
                        args.freeze_root_horizontal_after_seat_contact
                        and root_xy_contact_anchor is None
                    ):
                        root_xy_contact_anchor = data.xpos[pelvis_id, :2].copy()
                        root_xy_freeze_frame = int(frame)
                if root_xy_contact_anchor is not None:
                    desired_pos = desired_pos.copy()
                    desired_pos[:2] = root_xy_contact_anchor
                    control_linear_velocity[:2] = 0.0
                data.xfrc_applied[:] = 0.0
                if mocap_id is not None:
                    data.mocap_pos[mocap_id] = desired_pos
                    data.mocap_quat[mocap_id] = desired_quat
                joint_error = desired_joint - data.qpos[7:]
                joint_kp, joint_kd = args.joint_kp, args.joint_kd
                if args.inertia_scaled_joint_gains:
                    mujoco.mj_fullM(model, full_mass_matrix, data.qM)
                    joint_inertia = np.diag(full_mass_matrix)[actuator_dof_ids]
                    gain_scale = np.clip(
                        joint_inertia / inertia_reference,
                        args.joint_inertia_scale_min,
                        args.joint_inertia_scale_max,
                    )
                    gain_scale_min_seen = min(gain_scale_min_seen, float(np.min(gain_scale)))
                    gain_scale_max_seen = max(gain_scale_max_seen, float(np.max(gain_scale)))
                    joint_kp = args.joint_kp * gain_scale
                    joint_kd = args.joint_kd * gain_scale
                torque = (
                    joint_kp * joint_error
                    + joint_kd * (joint_velocity - data.qvel[6:])
                )
                if (
                    seat_support_active
                    and args.seat_contact_joint_gain_scale < 1.0
                ):
                    torque[support_actuator_mask] *= args.seat_contact_joint_gain_scale
                    support_joint_gain_scaled_substeps += 1
                if (
                    chair_contact_robot_geoms
                    and args.chair_contact_joint_gain_scale < 1.0
                ):
                    chair_contact_actuator_mask = np.zeros(model.nu, dtype=bool)
                    for geom_id in chair_contact_robot_geoms:
                        chair_contact_actuator_mask |= chair_contact_actuator_masks[geom_id]
                    torque[chair_contact_actuator_mask] *= args.chair_contact_joint_gain_scale
                    chair_contact_joint_gain_scaled_substeps += 1
                if joint_gravity_compensation:
                    torque += data.qfrc_bias[6:]
                if args.stance_task_mode == "reference_stance":
                    for foot_index, body_id in enumerate(stance_body_ids):
                        if not reference_stance_mask[frame, foot_index]:
                            continue
                        target_position = reference_stance_targets[frame, foot_index]
                        mujoco.mj_jacBody(model, data, stance_jacobian, None, body_id)
                        foot_velocity = stance_jacobian @ data.qvel
                        foot_force = (
                            args.stance_position_kp
                            * (target_position - data.xpos[body_id])
                            - args.stance_position_kd * foot_velocity
                        )
                        force_norm = float(np.linalg.norm(foot_force))
                        if force_norm > args.stance_force_limit_n:
                            foot_force *= args.stance_force_limit_n / force_norm
                            force_norm = args.stance_force_limit_n
                        torque += stance_jacobian[:, 6:].T @ foot_force
                        stance_task_substeps += 1
                        stance_task_peak_force_n = max(stance_task_peak_force_n, force_norm)
                torque_lower = np.maximum(
                    actuator_torque_lower, -args.joint_torque_limit
                )
                torque_upper = np.minimum(
                    actuator_torque_upper, args.joint_torque_limit
                )
                torque = np.clip(torque, torque_lower, torque_upper)
                if args.joint_tracking_mode == "kinematic":
                    torque.fill(0.0)
                data.ctrl[:] = torque / torque_command_scale
                root_force = (
                    args.root_pos_kp * (desired_pos - data.xpos[pelvis_id])
                    + args.root_pos_kd * (control_linear_velocity - data.qvel[:3])
                )
                if root_gravity_compensation:
                    root_force += gravity_support_force
                if (
                    seat_support_active
                    and args.prevent_downward_root_force_on_seat_contact
                    and root_force[2] < 0.0
                ):
                    root_force[2] = 0.0
                    root_downward_force_suppressed_substeps += 1
                force_norm = float(np.linalg.norm(root_force))
                if force_norm > args.root_force_limit:
                    root_force *= args.root_force_limit / force_norm
                root_torque = (
                    args.root_rot_kp * _rotation_error(desired_quat, data.xquat[pelvis_id])
                    - args.root_rot_kd * data.qvel[3:6]
                )
                torque_norm = float(np.linalg.norm(root_torque))
                if torque_norm > args.root_torque_limit:
                    root_torque *= args.root_torque_limit / torque_norm
                if args.root_tracking_mode != "force":
                    root_force.fill(0.0)
                    root_torque.fill(0.0)
                data.xfrc_applied[pelvis_id, :3] = root_force
                data.xfrc_applied[pelvis_id, 3:] = root_torque
                max_external_wrench_norm_n = max(
                    max_external_wrench_norm_n,
                    float(np.linalg.norm(data.xfrc_applied[pelvis_id])),
                )
                mujoco.mj_step(model, data)
            if args.root_tracking_mode == "kinematic":
                data.qpos[:3] = root_target[frame + 1]
                data.qpos[3:7] = end_quat
                data.qvel[:3] = linear_velocity
                data.qvel[3:6] = 0.0
                root_state_reimposition_substeps += 1
            if args.joint_tracking_mode == "kinematic":
                data.qpos[7:] = joint_target[frame + 1]
                data.qvel[6:] = joint_velocity
                joint_state_reimposition_substeps += 1
            # mj_step integrates qpos after evaluating contacts.  Re-run the
            # position stage before recording so distance/contact metrics refer
            # to the same state written to the output trajectory.
            mujoco.mj_forward(model, data)
            actual_root[frame + 1], actual_rot[frame + 1], actual_joint[frame + 1] = _record_state(data)
            sample(frame + 1)

        root_error = np.linalg.norm(actual_root - root_target, axis=1)
        joint_error = np.abs(actual_joint - joint_target)
        sit_indices = np.flatnonzero(sit_mask)
        if args.seat_contact_surface == "render_mesh":
            tail_count = min(len(sit_indices), args.settled_tail_frames)
            validation_indices = sit_indices[-tail_count:]
            validation_policy = "stable_tail_of_final_sit_segment"
        else:
            validation_indices = sit_indices
            validation_policy = "all_sit_labelled_frames"
        validation_mask = np.zeros(len(root_target), dtype=bool)
        validation_mask[validation_indices] = True
        valid_seat = seat_min_distance[validation_mask]
        valid_seat = valid_seat[np.isfinite(valid_seat)]
        support_frame_ratio = float(len(valid_seat) / max(len(validation_indices), 1))
        valid_support_force = seat_support_normal_force[validation_mask]
        loaded_support_frame_ratio = float(
            np.mean(valid_support_force >= args.min_seat_support_normal_force_n)
        ) if len(valid_support_force) else 0.0
        all_sit_visual = seat_min_distance[sit_mask]
        all_sit_visual = all_sit_visual[np.isfinite(all_sit_visual)]
        min_distance = float(np.min(valid_seat)) if len(valid_seat) else float("nan")
        valid_chair_depth = chair_min_contact_depth[validation_mask]
        valid_chair_depth = valid_chair_depth[np.isfinite(valid_chair_depth)]
        min_full_chair_distance = (
            float(np.min(valid_chair_depth)) if len(valid_chair_depth) else float("nan")
        )
        full_chair_material_penetration_frame_ratio = float(
            np.mean(valid_chair_depth < -args.max_penetration_m)
        ) if len(valid_chair_depth) else 0.0
        root_z_frame_delta = np.abs(np.diff(actual_root[:, 2]))
        max_root_z_frame_delta = float(np.max(root_z_frame_delta)) if len(root_z_frame_delta) else 0.0
        root_height_continuous = max_root_z_frame_delta <= args.max_root_z_frame_jump_m
        bad_qacc_warning_count = int(
            data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number
        )
        numerical_negative_clearance_frame_ratio = float(
            np.mean(valid_seat < -1e-5)
        )
        material_penetration_frame_ratio = float(
            np.mean(valid_seat < -args.max_penetration_m)
        )
        valid_proxy = proxy_min_distance[validation_mask]
        valid_proxy = valid_proxy[np.isfinite(valid_proxy)]
        root_tracking_required = True
        root_tracking_ok = float(np.sqrt(np.mean(root_error ** 2))) <= args.max_root_rmse_m
        physical_control_contract_ok = bool(
            args.root_tracking_mode == "none"
            and args.joint_tracking_mode == "pd"
            and max_external_wrench_norm_n <= 1e-9
            and root_state_reimposition_substeps == 0
            and joint_state_reimposition_substeps == 0
        )
        physical_control_contract_failures = []
        if args.root_tracking_mode != "none":
            physical_control_contract_failures.append("free_base_is_not_autonomous")
        if args.joint_tracking_mode != "pd":
            physical_control_contract_failures.append("joint_state_is_reimposed")
        if max_external_wrench_norm_n > 1e-9:
            physical_control_contract_failures.append("external_base_wrench_applied")
        if root_state_reimposition_substeps:
            physical_control_contract_failures.append("root_state_reimposed_after_initialization")
        if joint_state_reimposition_substeps:
            physical_control_contract_failures.append("joint_state_reimposed_after_initialization")
        status = (
            "pass"
            if (
                physical_control_contract_ok
                and (not args.enable_seat_contact or (
                    min_distance >= -args.max_penetration_m
                    and material_penetration_frame_ratio <= args.max_penetration_frame_ratio
                    and support_frame_ratio >= args.min_seat_support_frame_ratio
                    and loaded_support_frame_ratio >= args.min_seat_support_frame_ratio
                    and full_chair_material_penetration_frame_ratio <= args.max_penetration_frame_ratio
                ))
                and root_tracking_ok
                and root_height_continuous
                and float(np.median(joint_error)) <= args.max_joint_median_rad
                and bad_qacc_warning_count == 0
            )
            else "fail"
        )
        physical_interpretation = (
            (
                "mj_step_driven_render_mesh_contact_replay_with_kinematic_visual_state_boundaries"
                if args.seat_contact_surface == "render_mesh" else
                "mj_step_driven_contact_replay_with_kinematic_visual_state_boundaries"
            )
            if (
                args.root_tracking_mode == "kinematic"
                and args.joint_tracking_mode == "kinematic"
            )
            else (
                "autonomous_free_base_mj_step_contact_replay_with_joint_pd_tracking"
                if args.root_tracking_mode == "none"
                else (
                "mj_step_joint_and_contact_replay_with_kinematic_visual_root_boundary"
                if args.root_tracking_mode == "kinematic"
                else "contact_constrained_mj_step_replay_with_external_pelvis_tracking_harness"
                )
            )
        )
        report = {
            "schema_version": 4,
            "purpose": "mujoco_contact_constrained_tracking_replay",
            "status": status,
            "physical_interpretation": physical_interpretation,
            "external_root_tracking_harness": args.root_tracking_mode == "force",
            "autonomous_free_base": args.root_tracking_mode == "none",
            "physical_pass_eligibility": {
                "eligible": physical_control_contract_ok,
                "contract": (
                    "one_initial_state_then_torque_only_mj_step; no mocap/equality "
                    "base ownership, no external base wrench, no state reimposition"
                ),
                "state_initialization_count": 1,
                "root_state_reimposition_substeps": int(root_state_reimposition_substeps),
                "joint_state_reimposition_substeps": int(joint_state_reimposition_substeps),
                "max_external_base_wrench_norm": float(max_external_wrench_norm_n),
                "failure_reasons": physical_control_contract_failures,
            },
            "root_trajectory_contract": {
                "mode": args.root_tracking_mode,
                "owner": (
                    "mj_step_free_base_response"
                    if args.root_tracking_mode == "none"
                    else "input_visual_motion"
                ),
                "reimposed_each_mj_step_substep": args.root_tracking_mode == "kinematic",
                "rationale": (
                    "No external root force is applied; base motion follows joint torques "
                    "and scene contacts."
                    if args.root_tracking_mode == "none"
                    else "GVHMR global root is a visual-coordinate boundary; scene contacts "
                    "may not translate the person in x/y."
                ),
            },
            "joint_trajectory_contract": {
                "mode": args.joint_tracking_mode,
                "owner": "input_visual_motion" if args.joint_tracking_mode == "kinematic" else "mj_step_pd_response",
                "reimposed_each_mj_step_substep": args.joint_tracking_mode == "kinematic",
                "rationale": (
                    "For GVHMR-only clips, retain visually observed crossed-leg and "
                    "other non-PHC poses; use mj_step to validate contact rather than "
                    "replace the observed joint trajectory with a free response."
                ),
            },
            "contact_model": (
                "visible_robot_mesh_directional_clearance_to_semantic_seat; conservative_box_proxies_for_non-seat_diagnostics"
                if args.seat_contact_surface == "render_mesh" else
                "conservative_robot_link_box_proxies_against_semantic_scene_collision_geometries"
            ),
            "robot_motion_reference": str(args.robot_motion),
            "scene_mujoco_xml": str(args.scene_mujoco_xml),
            "frames": int(len(root_target)),
            "fps": fps,
            "mj_step_substeps_per_video_frame": args.substeps,
            "physics_timestep_s": float(model.opt.timestep),
            "actuator_control": actuator_details,
            "initialization": {
                "reference_velocity_enabled": bool(args.initialize_reference_velocity),
                "initial_linear_velocity_m_per_s": initial_qvel[:3].tolist(),
                "initial_angular_velocity_rad_per_s": initial_qvel[3:6].tolist(),
                "initial_joint_velocity_l2_rad_per_s": float(np.linalg.norm(initial_qvel[6:])),
            },
            "gravity_compensation": {
                "root_enabled": root_gravity_compensation,
                "joint_enabled": joint_gravity_compensation,
                "robot_mass_kg": robot_mass_kg,
                "world_support_force_N": gravity_support_force.tolist(),
                "joint_generalized_bias_compensation": joint_gravity_compensation,
            },
            "ground_contact": ground_report,
            "stance_task": {
                **stance_report,
                "active_substeps": int(stance_task_substeps),
                "peak_requested_force_n": float(stance_task_peak_force_n),
                "actuation_contract": (
                    "joint_torque_only; no pelvis/root external force"
                    if args.stance_task_mode != "none" else "not_enabled"
                ),
            },
            "joint_pd_gain_schedule": {
                "mode": "inertia_normalized" if args.inertia_scaled_joint_gains else "uniform",
                "reference_joint_inertia_kg_m2": inertia_reference,
                "scale_bounds": [args.joint_inertia_scale_min, args.joint_inertia_scale_max],
                "observed_scale_min": gain_scale_min_seen,
                "observed_scale_max": gain_scale_max_seen,
            },
            "simulation": {
                "bad_qacc_warning_count": bad_qacc_warning_count,
                "numerically_stable": bad_qacc_warning_count == 0,
            },
            "seat_contact": {
                "scope": "semantic_support_bodies_only",
                "surface": args.seat_contact_surface,
                "gap_definition": (
                    "minimum visible-mesh clearance along upward semantic-seat normal within seat footprint"
                    if args.seat_contact_surface == "render_mesh" else
                    "MuJoCo narrow-phase contact.dist for an actual permitted seat-support pair; no-contact frames are not guessed as a positive gap"
                ),
                "support_footprint_margin_m": args.support_footprint_margin_m if args.seat_contact_surface == "render_mesh" else None,
                "allowed_bodies": sorted(seat_contact_bodies),
                "full_chair_collision": bool(args.full_chair_collision),
                "collidable_chair_geoms": sorted(
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                    for geom_id in chair_ids
                ),
                "excluded_bodies": sorted({
                    _body_name(model, geom_id) for geom_id in seat_excluded_geoms
                }),
                "support_contact_substeps": int(seat_support_contact_substeps),
                "minimum_support_normal_force_N": float(np.min(valid_support_force)) if len(valid_support_force) else float("nan"),
                "median_support_normal_force_N": float(np.median(valid_support_force)) if len(valid_support_force) else float("nan"),
                "loaded_support_frame_ratio": loaded_support_frame_ratio,
                "root_downward_force_suppressed_substeps": int(
                    root_downward_force_suppressed_substeps
                ),
                "support_joint_gain_scaled_substeps": int(
                    support_joint_gain_scaled_substeps
                ),
                "chair_contact_joint_gain_scaled_substeps": int(
                    chair_contact_joint_gain_scaled_substeps
                ),
                "support_joint_gain_scale": float(args.seat_contact_joint_gain_scale),
                "chair_contact_joint_gain_scale": float(
                    args.chair_contact_joint_gain_scale
                ),
                "prevent_downward_root_force_on_contact": bool(
                    args.prevent_downward_root_force_on_seat_contact
                ),
                "root_horizontal_reference_frozen_after_seat_contact": bool(
                    root_xy_contact_anchor is not None
                ),
                "root_horizontal_reference_freeze_frame": root_xy_freeze_frame,
                "root_horizontal_reference_anchor_world_xy_m": (
                    root_xy_contact_anchor.tolist()
                    if root_xy_contact_anchor is not None else None
                ),
                "sit_frames": int(sit_mask.sum()),
                "validation_frame_policy": validation_policy,
                "validation_frame_range": [int(validation_indices[0]), int(validation_indices[-1])] if len(validation_indices) else [],
                "validation_frames": int(len(validation_indices)),
                "evaluated_support_frames": int(len(valid_seat)),
                "evaluated_support_frame_ratio": support_frame_ratio,
                "minimum_signed_distance_m": min_distance,
                "median_signed_distance_m": float(np.median(valid_seat)) if len(valid_seat) else float("nan"),
                "minimum_full_chair_contact_distance_m": min_full_chair_distance,
                "full_chair_material_penetration_frame_ratio": full_chair_material_penetration_frame_ratio,
                # Kept for consumers of schema v3; it is diagnostic only.
                "penetration_frame_ratio": numerical_negative_clearance_frame_ratio,
                "numerical_negative_clearance_frame_ratio": numerical_negative_clearance_frame_ratio,
                "material_penetration_frame_ratio": material_penetration_frame_ratio,
                "chair_contact_frame_ratio": float(np.mean(contact_counts[sit_mask] > 0)),
                "peak_contact_force_N": float(np.max(contact_force_peak)),
                "conservative_proxy_diagnostic": {
                    "minimum_signed_distance_m": float(np.min(valid_proxy)) if len(valid_proxy) else float("nan"),
                    "median_signed_distance_m": float(np.median(valid_proxy)) if len(valid_proxy) else float("nan"),
                    "note": "mj_geomDistance proxy diagnostic only; never an acceptance surface in collision_proxy mode",
                },
                "transition_diagnostic_all_sit_frames": {
                    "evaluated_frames": int(len(all_sit_visual)),
                    "minimum_signed_distance_m": float(np.min(all_sit_visual)) if len(all_sit_visual) else float("nan"),
                    "penetration_frame_ratio": (
                        float(np.mean(all_sit_visual < -1e-5))
                        if len(all_sit_visual)
                        else float("nan")
                    ),
                    "numerical_negative_clearance_frame_ratio": (
                        float(np.mean(all_sit_visual < -1e-5))
                        if len(all_sit_visual)
                        else float("nan")
                    ),
                    "material_penetration_frame_ratio": (
                        float(np.mean(all_sit_visual < -args.max_penetration_m))
                        if len(all_sit_visual)
                        else float("nan")
                    ),
                    "note": "reported but not an acceptance surface during sit descent; final support is validated on the stable tail",
                },
            },
            "tracking": {
                "root_tracking_required": root_tracking_required,
                "root_tracking_ok": root_tracking_ok,
                "root_position_rmse_m": float(np.sqrt(np.mean(root_error ** 2))),
                "root_position_p95_m": float(np.quantile(root_error, 0.95)),
                "root_height_continuous": root_height_continuous,
                "maximum_root_z_frame_delta_m": max_root_z_frame_delta,
                "joint_absolute_error_median_rad": float(np.median(joint_error)),
                "joint_absolute_error_p95_rad": float(np.quantile(joint_error, 0.95)),
            },
            "thresholds": {
                "max_penetration_m": args.max_penetration_m,
                "max_penetration_frame_ratio": args.max_penetration_frame_ratio,
                "min_seat_support_frame_ratio": args.min_seat_support_frame_ratio,
                "min_seat_support_normal_force_N": args.min_seat_support_normal_force_n,
                "settled_tail_frames": args.settled_tail_frames,
                "max_root_rmse_m": args.max_root_rmse_m,
                "max_joint_median_rad": args.max_joint_median_rad,
                "max_root_z_frame_jump_m": args.max_root_z_frame_jump_m,
            },
            "decision": (
                "diagnostic_only_nonphysical_state_or_force_ownership"
                if not physical_control_contract_ok
                else (
                    "simulation_numerically_unstable"
                    if bad_qacc_warning_count > 0
                    else (
                        "dynamic_track_is_not_physically_acceptable"
                        if status == "fail"
                        else "dynamic_track_meets_physical_acceptance_thresholds"
                    )
                )
            ),
        }
        output = dict(motion)
        output["root_pos"] = actual_root.astype(np.asarray(motion["root_pos"]).dtype, copy=False)
        output["root_rot"] = actual_rot.astype(np.asarray(motion["root_rot"]).dtype, copy=False)
        output["dof_pos"] = actual_joint.astype(np.asarray(motion["dof_pos"]).dtype, copy=False)
        output["mujoco_dynamic_validation"] = report
        args.output_motion.parent.mkdir(parents=True, exist_ok=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.output_motion.open("wb") as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        combined_xml.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
