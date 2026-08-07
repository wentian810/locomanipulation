#!/usr/bin/env python3
"""Create a MuJoCo MPC task model from one GMR motion and one fixed scene.

This is a format bridge, not a motion repair.  It writes the original GMR
trajectory as reference keyframes, full G1 posture targets, and marker targets.
The resulting model has no welds, no applied root force, and no robot-state
callback: the companion ``g1_mpc_runner`` may initialize keyframe zero once and
then use bounded physical actuators and ``mj_step`` only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from evaluate_gmr_chair_contacts import _combine_mjcf, _load_motion


MARKER_NAMES = (
    "pelvis", "head", "ltoe", "rtoe", "lheel", "rheel", "lknee", "rknee",
    "lhand", "rhand", "lelbow", "relbow", "lshoulder", "rshoulder", "lhip", "rhip",
)

# The only chair geometry accepted by the contact-aware profile.  Keeping this
# explicit prevents a task bundle from silently optimizing against a visual
# overlay, an unrelated scene object, or a dynamically duplicated chair.
CHAIR_COLLISION_GEOM_NAMES = (
    "seat_support_geom",
    "leg_front_left_geom",
    "leg_front_right_geom",
    "leg_back_left_geom",
    "leg_back_right_geom",
    "backrest_geom",
)

# Official G1 collision geoms are assigned to contact type 2 by
# ``official_unitree_mjcf``.  A scene floor with ``conaffinity=0`` is visual
# only to that robot: it silently permits a free-base fall through z=0 even
# though the static chair may still collide.  Physical task bundles therefore
# make the floor explicitly receive contact type 2.
G1_COLLISION_CONTACT_TYPE = "2"

DEFAULT_UNITREE_ACTUATOR_LIMITS = Path(__file__).with_name("g1_unitree_actuator_limits.json")

# These numeric ids are defined by MJPC's ``kPlannerNames`` declaration.  Keep
# the mapping here, rather than exposing an unvalidated integer flag: a task
# report must say which contact-regime solver generated it.  The default stays
# iLQG so established V24 bundles are bit-for-bit unchanged unless the caller
# explicitly requests a sampling-based contact search.
MJPC_PLANNER_IDS = {
    "sampling": 0,
    "gradient": 1,
    "ilqg": 2,
    "ilqs": 3,
    "robust_sampling": 4,
    "cross_entropy": 5,
    "sample_gradient": 6,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_vector(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite [3] vector")
    return vector


def _load_marker_map(path: Path) -> dict[str, tuple[str, np.ndarray]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != set(MARKER_NAMES):
        missing = sorted(set(MARKER_NAMES) - set(raw) if isinstance(raw, dict) else MARKER_NAMES)
        extra = sorted(set(raw) - set(MARKER_NAMES) if isinstance(raw, dict) else [])
        raise ValueError(f"marker map must contain exactly {len(MARKER_NAMES)} markers; missing={missing}, extra={extra}")
    result: dict[str, tuple[str, np.ndarray]] = {}
    for name in MARKER_NAMES:
        entry = raw[name]
        if isinstance(entry, str):
            body, offset = entry, np.zeros(3, dtype=np.float64)
        elif isinstance(entry, dict) and isinstance(entry.get("body"), str):
            body = entry["body"]
            offset = _as_vector(entry.get("pos", [0.0, 0.0, 0.0]), f"{name}.pos")
        else:
            raise ValueError(f"{name} must be a body name or {{'body': ..., 'pos': [...]}}")
        if not body:
            raise ValueError(f"{name}.body must not be empty")
        result[name] = (body, offset)
    return result


def _format(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.10g}" for value in np.asarray(values).reshape(-1))


def _child(parent: ET.Element, tag: str) -> ET.Element:
    found = parent.find(tag)
    return found if found is not None else ET.SubElement(parent, tag)


def _remove_named(parent: ET.Element, tag: str, name: str) -> None:
    for child in list(parent):
        if child.tag == tag and child.get("name") == name:
            parent.remove(child)


def _enable_g1_floor_contact(root: ET.Element) -> dict[str, str]:
    """Turn the uniquely named scene floor into a physical G1 support plane.

    This changes no scene pose, dimensions, material, or visual geometry.  It
    only closes the collision-mask contract required by the injected official
    Unitree collision geoms (``contype=2``).
    """
    floors = [
        geom for geom in root.iter("geom")
        if geom.get("name") == "floor"
    ]
    if len(floors) != 1:
        raise ValueError(f"physical task requires exactly one named floor, found {len(floors)}")
    floor = floors[0]
    if floor.get("type") != "plane":
        raise ValueError("named floor must be a MuJoCo plane")
    floor.set("contype", "1")
    floor.set("conaffinity", G1_COLLISION_CONTACT_TYPE)
    return {
        "name": "floor",
        "contype": floor.get("contype", ""),
        "conaffinity": floor.get("conaffinity", ""),
    }


def _validate_g1_floor_contact(model: mujoco.MjModel) -> dict[str, Any]:
    """Fail task creation unless compiled floor/G1 collision masks overlap."""
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0:
        raise ValueError("compiled physical task lacks floor")
    floor_contype = int(model.geom_contype[floor_id])
    floor_conaffinity = int(model.geom_conaffinity[floor_id])
    eligible: list[str] = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if not name.startswith("gmr_official_collision_"):
            continue
        robot_contype = int(model.geom_contype[geom_id])
        robot_conaffinity = int(model.geom_conaffinity[geom_id])
        if (floor_contype & robot_conaffinity) or (robot_contype & floor_conaffinity):
            eligible.append(name)
    if not eligible:
        raise ValueError(
            "compiled floor collision mask cannot contact any official G1 collision geom: "
            f"floor contype={floor_contype}, conaffinity={floor_conaffinity}"
        )
    return {
        "floor_name": "floor",
        "floor_contype": floor_contype,
        "floor_conaffinity": floor_conaffinity,
        "eligible_official_g1_collision_geom_count": len(eligible),
    }


def _set_task_metadata(
    root: ET.Element,
    fps: float,
    agent_horizon_s: float,
    chair_avoidance_until_s: float | None,
    phase_sync: dict[str, float] | None,
    planner_contact_model: str,
    agent_planner: str,
    sampling_trajectories: int,
) -> None:
    custom = _child(root, "custom")
    # These are the exact planner defaults from MJPC's upstream humanoid
    # tracking task.  Keeping them together avoids a misleading "MPC" smoke
    # test that merely runs an unconfigured sampling controller.
    if planner_contact_model not in {"exact", "smoothed"}:
        raise ValueError(f"unsupported planner contact model: {planner_contact_model}")
    if agent_planner not in MJPC_PLANNER_IDS:
        raise ValueError(f"unsupported MJPC planner: {agent_planner}")
    if sampling_trajectories < 2:
        raise ValueError("sampling_trajectories must be at least 2")
    planner_defaults = {
        "agent_planner": str(MJPC_PLANNER_IDS[agent_planner]),
        # Gradient planners normally use a differentiable contact surrogate.
        # With exact mode, every supported planner evaluates the same hard
        # contact and limit model as the following runtime ``mj_step``.  This
        # is particularly important for CEM/sampling comparisons: they must
        # not win merely because they planned in a different physics model.
        "agent_differentiable": "0" if planner_contact_model == "exact" else "1",
        "sampling_representation": "2",
        "sampling_spline_points": "16",
        "sampling_exploration": "0.15",
        "sampling_trajectories": str(sampling_trajectories),
        "gradient_spline_points": "5",
        "ilqg_num_rollouts": "16",
        "ilqg_regularization_type": "1",
        "ilqg_representation": "2",
    }
    phase_numeric_names = (
        "gmr_phase_sync_enabled",
        "gmr_phase_sync_start_reference_frame",
        "gmr_phase_sync_seat_reference_frame",
        "gmr_phase_sync_max_root_xy_error_m",
        "gmr_phase_sync_max_root_z_error_m",
        "gmr_phase_sync_max_pre_sit_penetration_m",
        "gmr_phase_sync_min_seat_support_force_n",
        "gmr_phase_sync_seat_debounce_s",
        "gmr_phase_sync_max_hold_s",
    )
    for name in (
        "gmr_reference_fps", "gmr_chair_avoidance_until_s", "agent_horizon",
        "agent_timestep", *phase_numeric_names, *planner_defaults,
    ):
        _remove_named(custom, "numeric", name)
    ET.SubElement(custom, "numeric", {"name": "gmr_reference_fps", "data": f"{fps:.10g}"})
    ET.SubElement(custom, "numeric", {"name": "agent_horizon", "data": f"{agent_horizon_s:.10g}"})
    ET.SubElement(custom, "numeric", {"name": "agent_timestep", "data": "0.005"})
    if chair_avoidance_until_s is not None:
        ET.SubElement(custom, "numeric", {
            "name": "gmr_chair_avoidance_until_s",
            "data": f"{chair_avoidance_until_s:.10g}",
        })
    if phase_sync is not None:
        for name in phase_numeric_names:
            if name not in phase_sync:
                raise ValueError(f"phase-sync configuration lacks {name}")
            ET.SubElement(custom, "numeric", {
                "name": name, "data": f"{float(phase_sync[name]):.10g}",
            })
    for name, value in planner_defaults.items():
        ET.SubElement(custom, "numeric", {"name": name, "data": value})


def _load_unitree_actuator_limits(path: Path) -> tuple[dict[str, float], dict[str, Any], dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1 or raw.get("profile") != "unitree_g1_29dof_position_servo_v1":
        raise ValueError(f"{path} is not a supported Unitree G1 actuator-limit profile")
    limits_raw = raw.get("joint_torque_limits_nm")
    if not isinstance(limits_raw, dict) or not limits_raw:
        raise ValueError(f"{path} has no joint_torque_limits_nm mapping")
    limits: dict[str, float] = {}
    for joint_name, limit in limits_raw.items():
        if not isinstance(joint_name, str) or not joint_name:
            raise ValueError("actuator-limit joint names must be non-empty strings")
        limit_float = float(limit)
        if not np.isfinite(limit_float) or limit_float <= 0:
            raise ValueError(f"invalid torque limit for {joint_name!r}: {limit!r}")
        limits[joint_name] = limit_float
    provenance = raw.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{path} has no provenance object")
    servo = raw.get("low_level_position_servo")
    if not isinstance(servo, dict) or servo.get("actuator_type") != "position":
        raise ValueError(f"{path} has no supported low_level_position_servo")
    for name in ("kp_nm_per_rad", "dampratio"):
        value = float(servo.get(name, 0.0))
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"invalid low-level servo {name}: {servo.get(name)!r}")
    joint_dynamics = servo.get("joint_dynamics")
    if not isinstance(joint_dynamics, dict):
        raise ValueError(f"{path} has no joint_dynamics profile")
    for name in ("damping_nms_per_rad", "armature_kgm2", "frictionloss_nm"):
        value = float(joint_dynamics.get(name, -1.0))
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"invalid joint dynamics {name}: {joint_dynamics.get(name)!r}")
    return limits, provenance, servo


def _load_official_joint_ranges(path: Path) -> dict[str, tuple[float, float]]:
    """Read scalar mechanical joint limits from the revision-matched G1 MJCF."""
    root = ET.parse(path).getroot()
    ranges: dict[str, tuple[float, float]] = {}
    for joint in root.findall(".//joint"):
        name = joint.get("name")
        range_text = joint.get("range")
        if not name or range_text is None:
            continue
        values = tuple(float(value) for value in range_text.split())
        if len(values) != 2 or not all(np.isfinite(values)) or values[0] >= values[1]:
            raise ValueError(f"official joint {name!r} has invalid range {range_text!r}")
        if name in ranges and ranges[name] != values:
            raise ValueError(f"official MJCF defines conflicting limits for {name!r}")
        ranges[name] = values
    if not ranges:
        raise ValueError(f"official MJCF has no finite joint ranges: {path}")
    return ranges


def _apply_official_joint_ranges(
    root: ET.Element,
    official_ranges: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    """Apply only official limits for actuated G1 joints in a temporary task.

    The source GMR XML and the input motion remain untouched.  This closes the
    physical-model contract before MPC is allowed to exploit a G1 joint's true
    mechanical range for balance/contact corrections.
    """
    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError("combined model has no actuator section")
    actuator_joint_names = {
        element.get("joint")
        for element in actuator
        if element.get("joint")
    }
    joints_by_name = {
        joint.get("name"): joint
        for joint in root.findall(".//joint")
        if joint.get("name")
    }
    missing_in_model = sorted(actuator_joint_names - set(joints_by_name))
    missing_official = sorted(actuator_joint_names - set(official_ranges))
    if missing_in_model or missing_official:
        raise ValueError(
            "cannot establish official G1 joint-limit contract: "
            f"missing_in_model={missing_in_model}, missing_official={missing_official}"
        )
    changed: dict[str, dict[str, list[float]]] = {}
    for name in sorted(actuator_joint_names):
        joint = joints_by_name[name]
        old_text = joint.get("range")
        if old_text is None:
            raise ValueError(f"actuated G1 joint {name!r} has no native range")
        old_values = tuple(float(value) for value in old_text.split())
        if len(old_values) != 2:
            raise ValueError(f"actuated G1 joint {name!r} has invalid native range")
        new_values = official_ranges[name]
        joint.set("range", _format(np.asarray(new_values)))
        if old_values != new_values:
            changed[name] = {
                "gmr_native_range_rad": [float(value) for value in old_values],
                "official_range_rad": [float(value) for value in new_values],
            }
    return {
        "status": "official_limits_applied_to_temporary_physical_task",
        "actuated_joint_count": len(actuator_joint_names),
        "changed_joint_ranges": changed,
    }


def _apply_constraint_profile(
    root: ET.Element,
    profile: str,
    joint_limit_margin_rad: float,
) -> dict[str, Any]:
    """Set the explicit contact/limit compliance contract for this task only.

    The stock Unitree MJCF uses the same intentionally soft values for joint
    limits and contacts.  They are suitable for broad simulation, but the
    recorded v9 rollout proves that they allow a torque-driven G1 to exceed a
    mechanical range and enter the chair before a seated contact is intended.
    ``rigid_static_support_v1`` keeps the 2 ms integration step and all source
    poses unchanged.  It only makes the existing mechanical stops and the
    static support surfaces respond faster and with higher impedance.
    """
    if not np.isfinite(joint_limit_margin_rad) or joint_limit_margin_rad < 0.0:
        raise ValueError("joint-limit margin must be finite and non-negative")
    if profile == "official_default_v1":
        if joint_limit_margin_rad != 0.0:
            raise ValueError("joint-limit margin requires a rigid_static_support profile")
        return {
            "name": profile,
            "status": "source_constraint_parameters_retained",
        }
    if profile not in {"rigid_static_support_v1", "rigid_static_support_v2"}:
        raise ValueError(f"unsupported physical constraint profile: {profile}")

    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError("combined model has no actuator section")
    actuator_joint_names = {
        element.get("joint") for element in actuator if element.get("joint")
    }
    joints_by_name = {
        joint.get("name"): joint for joint in root.findall(".//joint")
        if joint.get("name")
    }
    missing_joints = sorted(actuator_joint_names - set(joints_by_name))
    if missing_joints:
        raise ValueError(f"constraint profile cannot find actuated joints: {missing_joints}")

    if profile == "rigid_static_support_v1":
        # The 8 ms, critically damped response is 2.5x faster than the official
        # 20 ms default while remaining safely above MuJoCo's 2 * timestep floor.
        limit_solref = "0.008 1"
        limit_solimp = "0.98 0.995 0.001 0.5 2"
        limit_solref_values = [0.008, 1.0]
        limit_solimp_values = [0.98, 0.995, 0.001, 0.5, 2.0]
        support_solref = limit_solref
        support_solimp = limit_solimp
        support_solref_values = limit_solref_values
        support_solimp_values = limit_solimp_values
    else:
        # This is the stiffest critically damped time-constant accepted by the
        # unchanged 2 ms MuJoCo step (the solver's reference-safety floor is
        # 2 * timestep).  It models the G1's mechanical stops and the static
        # chair as contacts, rather than asking a trajectory cost to repair a
        # collision after it has happened.
        limit_solref = "0.004 1"
        limit_solimp = "0.995 0.999 0.001 0.5 2"
        limit_solref_values = [0.004, 1.0]
        limit_solimp_values = [0.995, 0.999, 0.001, 0.5, 2.0]
        support_solref = limit_solref
        support_solimp = limit_solimp
        support_solref_values = limit_solref_values
        support_solimp_values = limit_solimp_values
    for joint_name in actuator_joint_names:
        joint = joints_by_name[joint_name]
        joint.set("solreflimit", limit_solref)
        joint.set("solimplimit", limit_solimp)
        if joint_limit_margin_rad > 0.0:
            # MuJoCo activates an existing limit constraint once its distance
            # drops below margin.  The official range and every GMR keyframe
            # remain unchanged; this only gives bounded torque control a
            # physically modelled braking band before the hard stop.
            joint.set("margin", f"{joint_limit_margin_rad:.10g}")

    support_geom_names = set(CHAIR_COLLISION_GEOM_NAMES) | {"floor"}
    hardened_support_geoms: list[str] = []
    for geom in root.iter("geom"):
        name = geom.get("name", "")
        if name.startswith("gmr_official_collision_") or name in support_geom_names:
            geom.set("solref", support_solref)
            geom.set("solimp", support_solimp)
            hardened_support_geoms.append(name)
    if not hardened_support_geoms:
        raise ValueError("constraint profile found no robot or support collision geoms")
    return {
        "name": profile,
        "status": "stiff_joint_limits_and_static_support_applied",
        "joint_limit_solref": limit_solref_values,
        "joint_limit_solimp": limit_solimp_values,
        "joint_limit_margin_rad": joint_limit_margin_rad,
        "static_support_solref": support_solref_values,
        "static_support_solimp": support_solimp_values,
        "actuated_joint_count": len(actuator_joint_names),
        "hardened_collision_geom_count": len(hardened_support_geoms),
    }


def _project_reference_to_joint_limits(
    qpos: np.ndarray,
    model: mujoco.MjModel,
    max_allowed_correction_rad: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project only tiny scalar-joint reference violations into model limits.

    A keyframe beyond a mechanical limit is not a valid physical target.  The
    caller chooses a strict maximum correction; larger violations are rejected
    rather than hidden by a motion rewrite.
    """
    if not np.isfinite(max_allowed_correction_rad) or max_allowed_correction_rad < 0.0:
        raise ValueError("reference joint-limit correction tolerance must be finite and non-negative")
    result = qpos.copy()
    corrections: dict[str, dict[str, Any]] = {}
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        if not model.jnt_limited[joint_id]:
            continue
        qpos_address = int(model.jnt_qposadr[joint_id])
        lower, upper = (float(value) for value in model.jnt_range[joint_id])
        original = result[:, qpos_address]
        projected = np.clip(original, lower, upper)
        delta = projected - original
        maximum = float(np.max(np.abs(delta)))
        if maximum <= 0.0:
            continue
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name is None:
            raise ValueError(f"limited scalar joint {joint_id} has no name")
        if maximum > max_allowed_correction_rad:
            raise ValueError(
                f"reference exceeds physical limit at {joint_name!r} by {maximum:.6g} rad, "
                f"above allowed correction {max_allowed_correction_rad:.6g} rad"
            )
        frames = np.flatnonzero(np.abs(delta) > 0.0)
        result[:, qpos_address] = projected
        corrections[joint_name] = {
            "range_rad": [lower, upper],
            "max_abs_correction_rad": maximum,
            "frames": [int(frame) for frame in frames],
        }
    return result, {
        "status": "bounded_projection_applied" if corrections else "already_within_physical_joint_limits",
        "max_allowed_correction_rad": max_allowed_correction_rad,
        "corrections": corrections,
    }


def _apply_unitree_actuator_limits(root: ET.Element, limits: dict[str, float], servo: dict[str, Any]) -> None:
    """Install bounded MuJoCo position servos without mutating the GMR XML."""
    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError("combined model has no actuator section")
    joints_by_name = {
        joint.get("name"): joint
        for joint in root.findall(".//joint")
        if joint.get("name")
    }
    kp = float(servo["kp_nm_per_rad"])
    dampratio = float(servo["dampratio"])
    joint_dynamics = servo["joint_dynamics"]
    actuator_joints: list[str] = []
    for motor in actuator.findall("motor"):
        joint_name = motor.get("joint")
        if not joint_name:
            raise ValueError("every G1 actuator must be a joint motor")
        if joint_name in actuator_joints:
            raise ValueError(f"duplicate G1 motor for joint {joint_name!r}")
        actuator_joints.append(joint_name)
        if joint_name not in limits:
            raise ValueError(f"no official Unitree torque limit for motor joint {joint_name!r}")
        joint = joints_by_name.get(joint_name)
        joint_range = joint.get("range") if joint is not None else None
        if joint_range is None or len(joint_range.split()) != 2:
            raise ValueError(f"G1 motor joint {joint_name!r} has no finite position range")
        limit = limits[joint_name]
        torque_range = f"{-limit:.10g} {limit:.10g}"
        # The legacy GMR XML exposes unscaled ±1 Nm motors.  Change only the
        # temporary combined task: command is a target angle, force remains
        # bounded by official G1 limits, and MuJoCo integrates the result.
        motor.tag = "position"
        motor.set("ctrllimited", "true")
        motor.set("ctrlrange", joint_range)
        motor.set("kp", f"{kp:.10g}")
        motor.set("dampratio", f"{dampratio:.10g}")
        motor.set("forcelimited", "true")
        motor.set("forcerange", torque_range)
        joint.set("damping", f"{float(joint_dynamics['damping_nms_per_rad']):.10g}")
        joint.set("armature", f"{float(joint_dynamics['armature_kgm2']):.10g}")
        joint.set("frictionloss", f"{float(joint_dynamics['frictionloss_nm']):.10g}")
    missing_actuators = sorted(set(limits) - set(actuator_joints))
    if missing_actuators:
        raise ValueError(f"official torque profile has joints absent from G1 model: {missing_actuators}")


def _apply_unitree_torque_motor_limits(root: ET.Element, limits: dict[str, float], servo: dict[str, Any]) -> None:
    """Install direct, torque-limited motors for MuJoCo MPC.

    Unlike the position-servo profile, an MPC action is directly a joint torque
    in N m.  This keeps the optimization variable aligned with the upstream
    MJPC humanoid task while retaining the exact same official force limits.
    """
    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError("combined model has no actuator section")
    joints_by_name = {
        joint.get("name"): joint
        for joint in root.findall(".//joint")
        if joint.get("name")
    }
    joint_dynamics = servo["joint_dynamics"]
    actuator_joints: list[str] = []
    for motor in actuator.findall("motor"):
        joint_name = motor.get("joint")
        if not joint_name:
            raise ValueError("every G1 actuator must be a joint motor")
        if joint_name in actuator_joints:
            raise ValueError(f"duplicate G1 motor for joint {joint_name!r}")
        actuator_joints.append(joint_name)
        if joint_name not in limits:
            raise ValueError(f"no official Unitree torque limit for motor joint {joint_name!r}")
        joint = joints_by_name.get(joint_name)
        if joint is None:
            raise ValueError(f"G1 motor joint {joint_name!r} does not exist")
        limit = limits[joint_name]
        torque_range = f"{-limit:.10g} {limit:.10g}"
        # A MuJoCo motor with unit gear maps ctrl directly to joint torque.
        # Remove all position-servo fields in case this function is reused on
        # a temporary model that has previously been transformed.
        for attribute in ("kp", "dampratio"):
            motor.attrib.pop(attribute, None)
        motor.set("gear", "1")
        motor.set("ctrllimited", "true")
        motor.set("ctrlrange", torque_range)
        motor.set("forcelimited", "true")
        motor.set("forcerange", torque_range)
        joint.set("damping", f"{float(joint_dynamics['damping_nms_per_rad']):.10g}")
        joint.set("armature", f"{float(joint_dynamics['armature_kgm2']):.10g}")
        joint.set("frictionloss", f"{float(joint_dynamics['frictionloss_nm']):.10g}")
    missing_actuators = sorted(set(limits) - set(actuator_joints))
    if missing_actuators:
        raise ValueError(f"official torque profile has joints absent from G1 model: {missing_actuators}")


def _absolutize_compiler_directories(root: ET.Element, source_directory: Path) -> None:
    """Keep merged robot assets resolvable after writing beside the scene XML."""
    compiler = root.find("compiler")
    if compiler is None:
        return
    for attribute in ("meshdir", "texturedir"):
        value = compiler.get(attribute)
        if value and not Path(value).is_absolute():
            compiler.set(attribute, str((source_directory / value).resolve()))


def _set_cost_sensors(
    root: ET.Element,
    joint_dim: int,
    control_dim: int,
    tracking_profile: str,
) -> None:
    sensor = _child(root, "sensor")
    # MJPC requires every residual user sensor to be first and consecutive.
    # Preserve non-cost scene sensors after them; discard pre-existing user
    # residual slots because their callback contract is unrelated to this task.
    preserved = [child for child in list(sensor) if child.tag != "user"]
    sensor[:] = []
    residual_sensors: list[ET.Element] = []
    residual_sensors.append(ET.Element("user", {
        "name": "gmr_cost_joint_velocity", "dim": str(joint_dim), "user": "0 0.001 0 0.01",
    }))
    # Preserve the upstream humanoid tracker’s free-root velocity objective.
    # Without it, an MPC rollout can match a slowly moving root position while
    # still accumulating a large downward pelvis velocity before seat contact.
    residual_sensors.append(ET.Element("user", {
        "name": "gmr_cost_root_velocity", "dim": "3", "user": "6 0.1 0 1.0 0.3",
    }))
    residual_sensors.append(ET.Element("user", {
        "name": "gmr_cost_joint_limit_approach_velocity", "dim": str(joint_dim),
        "user": "6 0.1 0 1.0 0.3",
    }))
    residual_sensors.append(ET.Element("user", {"name": "gmr_cost_control", "dim": str(control_dim), "user": "3 0.1 0 1.0 0.3"}))
    has_global_posture = tracking_profile in {
        "marker_and_posture_v1",
        "marker_and_posture_chair_avoidance_v1",
        "marker_and_posture_chair_contact_v2",
        "marker_and_posture_chair_phase_v3",
    }
    has_pre_sit_clearance_cost = tracking_profile in {
        "marker_only_chair_avoidance_v1",
        "marker_chair_contact_seated_posture_v1",
        "marker_and_posture_chair_avoidance_v1",
    }
    has_chair_penetration_cost = has_pre_sit_clearance_cost or tracking_profile == (
        "marker_and_posture_chair_contact_v2"
    ) or tracking_profile == "marker_and_posture_chair_phase_v3"
    has_seated_posture = tracking_profile == "marker_chair_contact_seated_posture_v1"

    if has_global_posture:
        # This stricter profile keeps the GMR joint configuration and torso
        # attitude close to the source, while still leaving them as soft costs.
        residual_sensors.append(ET.Element("user", {"name": "gmr_cost_joint_position", "dim": str(joint_dim), "user": "6 8.0 0.0 40.0 0.1"}))
        residual_sensors.append(ET.Element("user", {"name": "gmr_cost_base_orientation", "dim": "3", "user": "6 20.0 0.0 60.0 0.1"}))
    if has_pre_sit_clearance_cost:
        residual_sensors.append(ET.Element("user", {
            "name": "gmr_cost_pre_sit_chair_penetration",
            "dim": "1",
            "user": "6 30.0 0.0 50.0 0.1",
        }))
    if has_chair_penetration_cost:
        residual_sensors.append(ET.Element("user", {
            "name": "gmr_cost_chair_penetration_excess",
            "dim": "1",
            "user": "6 30.0 0.0 50.0 0.1",
        }))
        if has_seated_posture:
            # The approach remains free for balance corrections.  Only after
            # the authoritative VideoMimic sit event do these preserve the
            # observed GMR lower-body/torso configuration as soft references.
            residual_sensors.append(ET.Element("user", {
                "name": "gmr_cost_seated_joint_position",
                "dim": str(joint_dim), "user": "6 8.0 0.0 40.0 0.1",
            }))
            residual_sensors.append(ET.Element("user", {
                "name": "gmr_cost_seated_base_orientation",
                "dim": "3", "user": "6 20.0 0.0 60.0 0.1",
            }))
    if tracking_profile not in {
        "marker_and_posture_v1",
        "marker_and_posture_chair_avoidance_v1",
        "marker_and_posture_chair_contact_v2",
        "marker_and_posture_chair_phase_v3",
        "marker_only_chair_avoidance_v1",
        "marker_chair_contact_seated_posture_v1",
        "upstream_marker_only_v1",
    }:
        raise ValueError(f"unsupported tracking profile: {tracking_profile}")
    residual_sensors.append(ET.Element("user", {"name": "gmr_cost_position_average", "dim": "3", "user": "6 100.0 0.0 100.0 0.1"}))
    for marker in MARKER_NAMES:
        residual_sensors.append(ET.Element("user", {"name": f"gmr_cost_position_{marker}", "dim": "3", "user": "6 30.0 0.0 100.0 0.1"}))
    for marker in MARKER_NAMES:
        residual_sensors.append(ET.Element("user", {"name": f"gmr_cost_velocity_{marker}", "dim": "3", "user": "6 0.1 0.0 1.0 0.3"}))
    for child in residual_sensors:
        sensor.append(child)
    for child in preserved:
        if not child.get("name", "").startswith(("tracking_pos[", "tracking_linvel[")):
            sensor.append(child)
    for marker in MARKER_NAMES:
        ET.SubElement(sensor, "framepos", {"name": f"tracking_pos[{marker}]", "objtype": "site", "objname": f"tracking[{marker}]"})
        ET.SubElement(sensor, "framelinvel", {"name": f"tracking_linvel[{marker}]", "objtype": "site", "objname": f"tracking[{marker}]"})


def _add_markers(root: ET.Element, marker_map: dict[str, tuple[str, np.ndarray]]) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("combined model has no worldbody")
    bodies = {body.get("name"): body for body in root.findall(".//body") if body.get("name")}
    for marker, (body_name, offset) in marker_map.items():
        body = bodies.get(body_name)
        if body is None:
            raise ValueError(f"marker {marker} refers to absent body {body_name!r}")
        site_name = f"tracking[{marker}]"
        for site in list(body.findall("site")):
            if site.get("name") == site_name:
                body.remove(site)
        ET.SubElement(body, "site", {
            "name": site_name, "type": "sphere", "size": "0.008",
            "pos": _format(offset), "rgba": "1 0.3 0 1", "group": "3",
        })
        mocap_name = f"mocap[{marker}]"
        for candidate in list(worldbody.findall("body")):
            if candidate.get("name") == mocap_name:
                worldbody.remove(candidate)
        mocap = ET.SubElement(worldbody, "body", {"name": mocap_name, "mocap": "true"})
        ET.SubElement(mocap, "site", {
            "name": mocap_name, "type": "sphere", "size": "0.012",
            "rgba": "0 0.3 1 0.55", "group": "3",
        })


def _qpos_trajectory(motion: dict[str, Any], model: mujoco.MjModel, start: int, end: int) -> np.ndarray:
    root = np.asarray(motion["root_pos"], dtype=np.float64)[start:end]
    rotation_xyzw = np.asarray(motion["root_rot"], dtype=np.float64)[start:end]
    dof = np.asarray(motion["dof_pos"], dtype=np.float64)[start:end]
    if root.shape != (len(root), 3) or rotation_xyzw.shape != (len(root), 4):
        raise ValueError("unexpected GMR root trajectory shape")
    if dof.shape != (len(root), model.nq - 7):
        raise ValueError(f"GMR has {dof.shape[1]} joints but task model requires {model.nq - 7}")
    qpos = np.empty((len(root), model.nq), dtype=np.float64)
    qpos[:, :3] = root
    qpos[:, 3:7] = rotation_xyzw[:, [3, 0, 1, 2]]
    qpos[:, 7:] = dof
    return qpos


def _contiguous_intervals(frames: list[int]) -> list[dict[str, int]]:
    """Summarize sorted local-frame indices without inventing a contact phase."""
    if not frames:
        return []
    intervals: list[dict[str, int]] = []
    start = previous = frames[0]
    for frame in frames[1:]:
        if frame == previous + 1:
            previous = frame
            continue
        intervals.append({"start_local_frame": start, "end_local_frame": previous,
                          "frame_count": previous - start + 1})
        start = previous = frame
    intervals.append({"start_local_frame": start, "end_local_frame": previous,
                      "frame_count": previous - start + 1})
    return intervals


def _detect_reference_seat_contact(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    source_start_frame: int,
    min_consecutive_frames: int,
) -> dict[str, Any]:
    """Infer a per-video chair-contact schedule from its own physical reference.

    This is intentionally geometric and deterministic: every GMR reference
    keyframe is forwarded once against the reconstructed *static* chair.  The
    result is only used to schedule a pre-contact clearance objective; it never
    changes a source pose, scene transform, or robot state at runtime.
    """
    if min_consecutive_frames < 1:
        raise ValueError("chair-contact-min-consecutive-frames must be positive")
    seat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "seat_support_geom")
    if seat_id < 0:
        raise ValueError("automatic chair schedule requires seat_support_geom")
    robot_collision_ids = {
        geom_id for geom_id in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").startswith(
            "gmr_official_collision_"
        )
    }
    if not robot_collision_ids:
        raise ValueError(
            "automatic chair schedule requires official_unitree_mjcf collision geoms"
        )
    data = mujoco.MjData(model)
    any_seat_contact: list[int] = []
    pelvis_seat_contact: list[int] = []
    for local_frame, frame_qpos in enumerate(qpos):
        data.qpos[:] = frame_qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        observed_any = False
        observed_pelvis = False
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            if contact.geom1 == seat_id and contact.geom2 in robot_collision_ids:
                robot_geom = contact.geom2
            elif contact.geom2 == seat_id and contact.geom1 in robot_collision_ids:
                robot_geom = contact.geom1
            else:
                continue
            observed_any = True
            robot_body = int(model.geom_bodyid[robot_geom])
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, robot_body) or "") == "pelvis":
                observed_pelvis = True
        if observed_any:
            any_seat_contact.append(local_frame)
        if observed_pelvis:
            pelvis_seat_contact.append(local_frame)

    any_intervals = _contiguous_intervals(any_seat_contact)
    pelvis_intervals = _contiguous_intervals(pelvis_seat_contact)
    robust_intervals = [
        interval for interval in any_intervals
        if interval["frame_count"] >= min_consecutive_frames
    ]
    if not robust_intervals:
        raise ValueError(
            "reference contains no robust robot-seat contact interval; cannot "
            "automatically schedule chair clearance"
        )
    first_robust = robust_intervals[0]["start_local_frame"]
    terminal_pelvis_intervals = [
        interval for interval in pelvis_intervals
        if interval["frame_count"] >= min_consecutive_frames
    ]
    return {
        "status": "detected_from_reference_static_contact",
        "minimum_consecutive_frames": min_consecutive_frames,
        "any_seat_contact_intervals_local": any_intervals,
        "pelvis_seat_contact_intervals_local": pelvis_intervals,
        "first_robust_seat_contact_source_frame": source_start_frame + first_robust,
        "terminal_robust_pelvis_contact_source_frame": (
            source_start_frame + terminal_pelvis_intervals[-1]["start_local_frame"]
            if terminal_pelvis_intervals else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-motion", required=True, type=Path)
    parser.add_argument("--robot-xml", required=True, type=Path)
    parser.add_argument("--scene-mujoco-xml", required=True, type=Path)
    parser.add_argument("--marker-map", required=True, type=Path)
    parser.add_argument("--output-xml", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None, help="exclusive GMR frame bound")
    parser.add_argument("--agent-horizon-s", type=float, default=0.5)
    parser.add_argument(
        "--agent-planner",
        choices=tuple(MJPC_PLANNER_IDS),
        default="ilqg",
        help=(
            "MJPC solver.  ilqg preserves V24's default; sampling-based "
            "solvers evaluate full rollouts and are useful for a free-base "
            "contact transition where local contact derivatives are unreliable"
        ),
    )
    parser.add_argument(
        "--sampling-trajectories",
        type=int,
        default=32,
        help=(
            "number of rollouts used by sampling, robust-sampling, CEM, and "
            "sample-gradient planners (ignored by iLQG/gradient/iLQS)"
        ),
    )
    parser.add_argument(
        "--planner-contact-model",
        choices=("exact", "smoothed"),
        default="exact",
        help=(
            "exact keeps MuJoCo contact and joint-limit solver parameters "
            "identical in planning rollouts and mj_step execution; smoothed "
            "reproduces upstream's differentiable contact rollout model"
        ),
    )
    parser.add_argument(
        "--tracking-profile",
        choices=(
            "marker_and_posture_v1",
            "marker_and_posture_chair_avoidance_v1",
            "marker_and_posture_chair_contact_v2",
            "marker_and_posture_chair_phase_v3",
            "upstream_marker_only_v1",
            "marker_only_chair_avoidance_v1",
            "marker_chair_contact_seated_posture_v1",
        ),
        default="marker_and_posture_v1",
        help=(
            "marker_and_posture_v1 adds GMR joint/base-orientation residuals; "
            "marker_and_posture_chair_avoidance_v1 combines full-motion "
            "posture tracking with actual MuJoCo chair-penetration costs; "
            "marker_and_posture_chair_contact_v2 keeps the same full-motion "
            "posture tracking but uses only the fast actual-contact excess "
            "residual, for a reference already cleared offline; "
            "marker_and_posture_chair_phase_v3 adds an adaptive physical "
            "phase gate: reference targets pause until root tracking/clearance "
            "are valid, and the seated phase requires measured seat support; "
            "upstream_marker_only_v1 matches MuJoCo MPC's humanoid tracking "
            "residual structure and leaves balance corrections unconstrained; "
            "marker_only_chair_avoidance_v1 adds a pre-sit, actual-contact "
            "penetration cost without changing the robot state or chair pose; "
            "marker_chair_contact_seated_posture_v1 additionally preserves "
            "GMR posture only after the authoritative sit event"
        ),
    )
    parser.add_argument(
        "--chair-avoidance-until-source-frame",
        type=int,
        default=None,
        help=(
            "exclusive global GMR frame bound for the chair-penetration cost; "
            "required by marker_only_chair_avoidance_v1 and normally equals "
            "the first authoritative seated-contact frame"
        ),
    )
    parser.add_argument(
        "--auto-chair-avoidance-until-source-frame",
        action="store_true",
        help=(
            "derive the clearance deadline from this video's own static G1/seat "
            "contact sequence instead of accepting a video-specific frame number"
        ),
    )
    parser.add_argument(
        "--chair-contact-min-consecutive-frames",
        type=int,
        default=2,
        help="minimum consecutive reference frames required for automatic seat-contact detection",
    )
    parser.add_argument(
        "--detect-reference-chair-contact",
        action="store_true",
        help=(
            "write this video's automatic static seat-contact schedule to the "
            "build report without enabling the predictive clearance residual"
        ),
    )
    parser.add_argument(
        "--phase-sync-start-source-frame",
        type=int,
        default=None,
        help="global GMR frame at which adaptive physical phase gating begins",
    )
    parser.add_argument(
        "--phase-sync-seat-source-frame",
        type=int,
        default=None,
        help="global GMR frame that may advance only after measured seat support",
    )
    parser.add_argument(
        "--auto-phase-sync",
        action="store_true",
        help=(
            "derive phase-gate start and seated frames from this video's own "
            "robust static G1/chair contact schedule; valid only with "
            "marker_and_posture_chair_phase_v3"
        ),
    )
    parser.add_argument("--phase-sync-max-root-xy-error-m", type=float, default=0.07)
    parser.add_argument("--phase-sync-max-root-z-error-m", type=float, default=0.04)
    parser.add_argument("--phase-sync-max-pre-sit-penetration-m", type=float, default=0.005)
    parser.add_argument("--phase-sync-min-seat-support-force-n", type=float, default=60.0)
    parser.add_argument("--phase-sync-seat-debounce-s", type=float, default=0.10)
    parser.add_argument(
        "--phase-sync-max-hold-s", type=float, default=2.0,
        help="maximum additional physical simulation time before an incomplete phase-gated run is rejected",
    )
    parser.add_argument(
        "--actuator-profile",
        choices=(
            "unitree_g1_29dof_position_servo_v1",
            "unitree_g1_29dof_torque_motor_v1",
            "legacy_unit_torque",
        ),
        default="unitree_g1_29dof_torque_motor_v1",
        help=(
            "select a bounded physical G1 actuator interface: position servo "
            "or direct torque motor using official limits; legacy_unit_torque "
            "exists only for historical reproduction"
        ),
    )
    parser.add_argument(
        "--unitree-actuator-limits",
        type=Path,
        default=DEFAULT_UNITREE_ACTUATOR_LIMITS,
        help="versioned official G1 torque-limit mapping used by the physical actuator profile",
    )
    parser.add_argument(
        "--collision-proxy-strategy",
        choices=(
            "legacy_aabb", "hybrid_capsule", "visual_mesh",
            "custom_urdf_cylinders", "official_unitree_mjcf",
        ),
        default="legacy_aabb",
        help=(
            "G1-to-scene collision representation: legacy AABB, hybrid "
            "capsule, visual mesh convex hull, GMR's hand-authored URDF "
            "cylinders, or revision-matched official Unitree MJCF collisions"
        ),
    )
    parser.add_argument(
        "--custom-collision-urdf",
        type=Path,
        default=None,
        help="optional G1 custom-collision URDF; defaults beside --robot-xml",
    )
    parser.add_argument(
        "--official-collision-mjcf",
        type=Path,
        default=None,
        help=(
            "revision-matched official Unitree G1 MJCF used only by the "
            "official_unitree_mjcf collision strategy"
        ),
    )
    parser.add_argument(
        "--joint-limit-contract",
        choices=("gmr_native", "official_g1"),
        default="gmr_native",
        help=(
            "joint-limit source for the temporary physical task. official_g1 "
            "copies only revision-matched official limits for the actuated G1 "
            "joints; it never changes the input GMR motion file."
        ),
    )
    parser.add_argument(
        "--reference-joint-limit-tolerance-rad",
        type=float,
        default=0.005,
        help=(
            "maximum explicit reference projection allowed when official_g1 "
            "limits expose a tiny out-of-range source keyframe"
        ),
    )
    parser.add_argument(
        "--constraint-profile",
        choices=(
            "official_default_v1", "rigid_static_support_v1",
            "rigid_static_support_v2",
        ),
        default="official_default_v1",
        help=(
            "physical compliance for temporary joint limits and static G1/scene "
            "contacts; does not alter GMR qpos, root poses, or scene geometry"
        ),
    )
    parser.add_argument(
        "--joint-limit-margin-rad",
        type=float,
        default=0.0,
        help=(
            "activate each existing actuated-joint limit this many radians "
            "before its official range; does not change the range or reference"
        ),
    )
    args = parser.parse_args()

    motion = _load_motion(args.robot_motion)
    frame_count = len(np.asarray(motion["root_pos"]))
    fps = float(np.asarray(motion.get("fps", 30.0)).reshape(-1)[0])
    end = frame_count if args.end_frame is None else args.end_frame
    if not 0 <= args.start_frame < end <= frame_count or end - args.start_frame < 2:
        raise ValueError("selected GMR frame range must contain at least two frames")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("GMR fps must be positive")
    if not np.isfinite(args.agent_horizon_s) or args.agent_horizon_s <= 0.0:
        raise ValueError("agent-horizon-s must be positive")
    if args.joint_limit_contract == "official_g1":
        if args.official_collision_mjcf is None:
            raise ValueError(
                "official_g1 joint-limit contract requires --official-collision-mjcf"
            )
        if not np.isfinite(args.reference_joint_limit_tolerance_rad) or args.reference_joint_limit_tolerance_rad < 0.0:
            raise ValueError("reference-joint-limit-tolerance-rad must be finite and non-negative")
    chair_tracking_profiles = {
        "marker_only_chair_avoidance_v1",
        "marker_chair_contact_seated_posture_v1",
        "marker_and_posture_chair_avoidance_v1",
    }
    static_chair_profiles = chair_tracking_profiles | {
        "marker_and_posture_chair_contact_v2",
        "marker_and_posture_chair_phase_v3",
    }
    if (args.detect_reference_chair_contact and
            args.tracking_profile not in static_chair_profiles):
        raise ValueError(
            "--detect-reference-chair-contact requires a static-chair tracking profile"
        )
    if args.tracking_profile in chair_tracking_profiles:
        if (args.chair_avoidance_until_source_frame is not None and
                args.auto_chair_avoidance_until_source_frame):
            raise ValueError(
                "choose either a manual or automatic chair-avoidance deadline, not both"
            )
        if (args.chair_avoidance_until_source_frame is None and
                not args.auto_chair_avoidance_until_source_frame):
            raise ValueError(
                "chair-contact tracking profiles require "
                "--chair-avoidance-until-source-frame or "
                "--auto-chair-avoidance-until-source-frame"
            )
        if (args.chair_avoidance_until_source_frame is not None and
                not args.start_frame < args.chair_avoidance_until_source_frame <= end):
            raise ValueError(
                "chair-avoidance-until-source-frame must be in "
                "(start-frame, end-frame]"
            )
    else:
        if (args.chair_avoidance_until_source_frame is not None or
                args.auto_chair_avoidance_until_source_frame):
            raise ValueError(
                "chair-avoidance deadline options are only valid with a "
                "chair-contact tracking profile"
            )
    if args.chair_contact_min_consecutive_frames < 1:
        raise ValueError("chair-contact-min-consecutive-frames must be positive")
    chair_avoidance_until_source_frame = args.chair_avoidance_until_source_frame
    chair_avoidance_until_s: float | None = None
    auto_chair_contact_report: dict[str, Any] = {"status": "not_requested"}
    phase_sync: dict[str, float] | None = None
    resolved_phase_sync_start_source_frame: int | None = None
    resolved_phase_sync_seat_source_frame: int | None = None
    if args.tracking_profile == "marker_and_posture_chair_phase_v3":
        manual_phase_frame_requested = (
            args.phase_sync_start_source_frame is not None or
            args.phase_sync_seat_source_frame is not None
        )
        if args.auto_phase_sync and manual_phase_frame_requested:
            raise ValueError(
                "choose either --auto-phase-sync or explicit phase-sync source frames, not both"
            )
        if (not args.auto_phase_sync and
                (args.phase_sync_start_source_frame is None or
                 args.phase_sync_seat_source_frame is None)):
            raise ValueError(
                "marker_and_posture_chair_phase_v3 requires --auto-phase-sync "
                "or both explicit phase-sync source frames"
            )
        if not args.auto_phase_sync:
            assert args.phase_sync_start_source_frame is not None
            assert args.phase_sync_seat_source_frame is not None
            if not (args.start_frame <= args.phase_sync_start_source_frame <=
                    args.phase_sync_seat_source_frame < end):
                raise ValueError(
                    "phase-sync source frames must satisfy "
                    "start-frame <= start <= seat < end-frame"
                )
            resolved_phase_sync_start_source_frame = args.phase_sync_start_source_frame
            resolved_phase_sync_seat_source_frame = args.phase_sync_seat_source_frame
        positive_values = {
            "phase-sync-max-root-xy-error-m": args.phase_sync_max_root_xy_error_m,
            "phase-sync-max-root-z-error-m": args.phase_sync_max_root_z_error_m,
            "phase-sync-min-seat-support-force-n": args.phase_sync_min_seat_support_force_n,
            "phase-sync-max-hold-s": args.phase_sync_max_hold_s,
        }
        if any(not np.isfinite(value) or value <= 0.0 for value in positive_values.values()):
            raise ValueError(f"phase-sync values must be finite and positive: {positive_values}")
        if (not np.isfinite(args.phase_sync_max_pre_sit_penetration_m) or
                args.phase_sync_max_pre_sit_penetration_m < 0.0 or
                not np.isfinite(args.phase_sync_seat_debounce_s) or
                args.phase_sync_seat_debounce_s < 0.0):
            raise ValueError("phase-sync penetration and debounce values must be finite and non-negative")
    elif (args.auto_phase_sync or args.phase_sync_start_source_frame is not None or
          args.phase_sync_seat_source_frame is not None):
        raise ValueError("phase-sync options are only valid with marker_and_posture_chair_phase_v3")
    marker_map = _load_marker_map(args.marker_map)

    combined = _combine_mjcf(
        args.robot_xml, args.scene_mujoco_xml,
        collision_proxy_strategy=args.collision_proxy_strategy,
        custom_collision_urdf=args.custom_collision_urdf,
        official_collision_mjcf=args.official_collision_mjcf,
    )
    temporary_xml: Path | None = None
    try:
        tree = ET.parse(combined)
        root = tree.getroot()
        _absolutize_compiler_directories(root, combined.parent)
        floor_collision_contract = _enable_g1_floor_contact(root)
        joint_limit_report: dict[str, Any] = {
            "status": "gmr_native_limits_retained",
            "requested_contract": args.joint_limit_contract,
        }
        if args.joint_limit_contract == "official_g1":
            assert args.official_collision_mjcf is not None
            joint_limit_report = _apply_official_joint_ranges(
                root, _load_official_joint_ranges(args.official_collision_mjcf)
            )
            joint_limit_report["requested_contract"] = args.joint_limit_contract
            joint_limit_report["source_mjcf"] = str(args.official_collision_mjcf)
        constraint_profile_report = _apply_constraint_profile(
            root, args.constraint_profile, args.joint_limit_margin_rad
        )
        actuator_profile_report: dict[str, Any] = {"profile": args.actuator_profile}
        runtime_actuation = "bounded_position_actuator_then_mj_step"
        if args.actuator_profile in {
            "unitree_g1_29dof_position_servo_v1",
            "unitree_g1_29dof_torque_motor_v1",
        }:
            torque_limits, provenance, servo = _load_unitree_actuator_limits(args.unitree_actuator_limits)
            if args.actuator_profile == "unitree_g1_29dof_position_servo_v1":
                _apply_unitree_actuator_limits(root, torque_limits, servo)
                runtime_interface = "joint_angle_target_with_torque_saturation"
            else:
                _apply_unitree_torque_motor_limits(root, torque_limits, servo)
                runtime_actuation = "bounded_torque_actuator_then_mj_step"
                runtime_interface = "direct_joint_torque_nm"
            actuator_profile_report.update({
                "limit_config": str(args.unitree_actuator_limits),
                "limit_config_sha256": _sha256(args.unitree_actuator_limits),
                "provenance": provenance,
                "low_level_position_servo": servo,
                "joint_count": len(torque_limits),
                "simulation_control_interface": runtime_interface,
            })
        else:
            actuator_profile_report["warning"] = "historical ±1 Nm GMR motor limits; not valid for physical acceptance"
        _add_markers(root, marker_map)
        # Compile once to resolve all joint/site addresses before keyframes are
        # emitted.  The temporary file lives beside the final XML so mesh paths
        # retain the same resolution base.
        args.output_xml.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".xml", dir=args.output_xml.parent, delete=False) as handle:
            temporary_xml = Path(handle.name)
        tree.write(temporary_xml, encoding="unicode")
        model = mujoco.MjModel.from_xml_path(str(temporary_xml))
        floor_collision_contract.update(_validate_g1_floor_contact(model))
        if args.tracking_profile in static_chair_profiles:
            missing_chair_geoms = [
                name for name in CHAIR_COLLISION_GEOM_NAMES
                if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) < 0
            ]
            if missing_chair_geoms:
                raise ValueError(
                    "chair-avoidance profile requires one static physical chair; "
                    f"missing geoms: {missing_chair_geoms}"
                )
        data = mujoco.MjData(model)
        qpos = _qpos_trajectory(motion, model, args.start_frame, end)
        reference_joint_limit_report: dict[str, Any] = {
            "status": "not_requested_gmr_native_contract"
        }
        if args.joint_limit_contract == "official_g1":
            qpos, reference_joint_limit_report = _project_reference_to_joint_limits(
                qpos, model, args.reference_joint_limit_tolerance_rad
            )
        if (args.tracking_profile == "marker_and_posture_chair_phase_v3" and
                args.auto_phase_sync):
            auto_chair_contact_report = _detect_reference_seat_contact(
                model, qpos, args.start_frame,
                args.chair_contact_min_consecutive_frames,
            )
            resolved_phase_sync_start_source_frame = int(
                auto_chair_contact_report["first_robust_seat_contact_source_frame"]
            )
            resolved_phase_sync_seat_source_frame = int(
                auto_chair_contact_report["terminal_robust_pelvis_contact_source_frame"]
            )
        elif args.tracking_profile in chair_tracking_profiles:
            if args.auto_chair_avoidance_until_source_frame:
                auto_chair_contact_report = _detect_reference_seat_contact(
                    model, qpos, args.start_frame,
                    args.chair_contact_min_consecutive_frames,
                )
                chair_avoidance_until_source_frame = int(
                    auto_chair_contact_report["first_robust_seat_contact_source_frame"]
                )
            else:
                assert chair_avoidance_until_source_frame is not None
                auto_chair_contact_report = {
                    "status": "manual_deadline_requested",
                    "first_robust_seat_contact_source_frame": chair_avoidance_until_source_frame,
                }
            chair_avoidance_until_s = (
                (chair_avoidance_until_source_frame - args.start_frame) / fps
            )
        elif args.detect_reference_chair_contact:
            auto_chair_contact_report = _detect_reference_seat_contact(
                model, qpos, args.start_frame,
                args.chair_contact_min_consecutive_frames,
            )
        if args.tracking_profile == "marker_and_posture_chair_phase_v3":
            if (resolved_phase_sync_start_source_frame is None or
                    resolved_phase_sync_seat_source_frame is None):
                raise AssertionError("phase-sync source frames were not resolved")
            if not (args.start_frame <= resolved_phase_sync_start_source_frame <=
                    resolved_phase_sync_seat_source_frame < end):
                raise ValueError(
                    "resolved phase-sync source frames must satisfy "
                    "start-frame <= start <= seat < end-frame"
                )
            phase_sync = {
                "gmr_phase_sync_enabled": 1.0,
                "gmr_phase_sync_start_reference_frame": float(
                    resolved_phase_sync_start_source_frame - args.start_frame),
                "gmr_phase_sync_seat_reference_frame": float(
                    resolved_phase_sync_seat_source_frame - args.start_frame),
                "gmr_phase_sync_max_root_xy_error_m": args.phase_sync_max_root_xy_error_m,
                "gmr_phase_sync_max_root_z_error_m": args.phase_sync_max_root_z_error_m,
                "gmr_phase_sync_max_pre_sit_penetration_m": args.phase_sync_max_pre_sit_penetration_m,
                "gmr_phase_sync_min_seat_support_force_n": args.phase_sync_min_seat_support_force_n,
                "gmr_phase_sync_seat_debounce_s": args.phase_sync_seat_debounce_s,
                "gmr_phase_sync_max_hold_s": args.phase_sync_max_hold_s,
            }
        _set_task_metadata(
            root, fps, args.agent_horizon_s, chair_avoidance_until_s, phase_sync,
            args.planner_contact_model, args.agent_planner,
            args.sampling_trajectories,
        )
        _set_cost_sensors(
            root, model.nv - 6, model.nu, args.tracking_profile,
        )
        keyframe = _child(root, "keyframe")
        for key in list(keyframe):
            keyframe.remove(key)
        site_ids = {
            marker: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"tracking[{marker}]")
            for marker in MARKER_NAMES
        }
        mocap_ids = {
            marker: int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"mocap[{marker}]")])
            for marker in MARKER_NAMES
        }
        if any(site_id < 0 for site_id in site_ids.values()) or any(mocap_id < 0 for mocap_id in mocap_ids.values()):
            raise RuntimeError("generated marker sites or mocap bodies did not compile")
        qvel = np.zeros((len(qpos), model.nv), dtype=np.float64)
        for frame in range(len(qpos) - 1):
            mujoco.mj_differentiatePos(model, qvel[frame], 1.0 / fps, qpos[frame], qpos[frame + 1])
        qvel[-1] = qvel[-2]
        for frame, frame_qpos in enumerate(qpos):
            data.qpos[:] = frame_qpos
            data.qvel[:] = qvel[frame]
            mujoco.mj_forward(model, data)
            mpos = np.zeros((model.nmocap, 3), dtype=np.float64)
            for marker in MARKER_NAMES:
                mpos[mocap_ids[marker]] = data.site_xpos[site_ids[marker]]
            ET.SubElement(keyframe, "key", {
                "name": f"gmr_{frame:05d}", "qpos": _format(frame_qpos),
                "qvel": _format(qvel[frame]), "mpos": _format(mpos),
            })
        tree.write(args.output_xml, encoding="unicode")
        final_model = mujoco.MjModel.from_xml_path(str(args.output_xml))
        residual_dim = 2 * (final_model.nv - 6) + 3 + final_model.nu + 3 + 6 * len(MARKER_NAMES)
        if args.tracking_profile in {
            "marker_and_posture_v1",
            "marker_and_posture_chair_avoidance_v1",
            "marker_and_posture_chair_contact_v2",
            "marker_and_posture_chair_phase_v3",
        }:
            residual_dim += (final_model.nq - 7) + 3
        if args.tracking_profile in {
            "marker_only_chair_avoidance_v1",
            "marker_chair_contact_seated_posture_v1",
            "marker_and_posture_chair_avoidance_v1",
        }:
            residual_dim += 2
        elif args.tracking_profile in {
            "marker_and_posture_chair_contact_v2",
            "marker_and_posture_chair_phase_v3",
        }:
            residual_dim += 1
        if args.tracking_profile == "marker_chair_contact_seated_posture_v1":
            residual_dim += (final_model.nq - 7) + 3
        report = {
            "schema_version": 1,
            "purpose": "gmr_to_mjpc_reference_task_bundle",
            "status": "ready",
            "physical_contract": {
                "reference_owner": "mocap_targets_only",
                "initial_robot_state": "keyframe_zero_once",
                "allowed_runtime_actuation": runtime_actuation,
                "agent_planner": {
                    "name": args.agent_planner,
                    "mjpc_id": MJPC_PLANNER_IDS[args.agent_planner],
                    "sampling_trajectories": args.sampling_trajectories,
                },
                "planner_contact_model": args.planner_contact_model,
                "constraint_profile": constraint_profile_report,
                "ground_collision": floor_collision_contract,
                "forbidden_runtime_mechanisms": ["xfrc_applied", "mocap_weld", "post_initialization_qpos_or_qvel_write"],
            },
            "inputs": {
                "robot_motion": str(args.robot_motion), "robot_motion_sha256": _sha256(args.robot_motion),
                "robot_xml": str(args.robot_xml), "scene_mujoco_xml": str(args.scene_mujoco_xml),
                "marker_map": str(args.marker_map),
                "collision_proxy_strategy": args.collision_proxy_strategy,
                "custom_collision_urdf": (
                    str(args.custom_collision_urdf) if args.custom_collision_urdf else None
                ),
                "official_collision_mjcf": (
                    str(args.official_collision_mjcf) if args.official_collision_mjcf else None
                ),
                "actuator_profile": actuator_profile_report,
                "joint_limit_contract": joint_limit_report,
            },
            "reference": {
                "source_frame_range": [args.start_frame, end - 1], "frame_count": int(len(qpos)),
                "fps": fps, "agent_horizon_s": args.agent_horizon_s,
                "tracking_profile": args.tracking_profile,
                "chair_avoidance_until_source_frame": chair_avoidance_until_source_frame,
                "chair_avoidance_until_s": chair_avoidance_until_s,
                "auto_chair_contact_detection": auto_chair_contact_report,
                "phase_sync": (
                    {
                        "resolution": "automatic_static_contact" if args.auto_phase_sync else "explicit_source_frames",
                        "start_source_frame": resolved_phase_sync_start_source_frame,
                        "seat_source_frame": resolved_phase_sync_seat_source_frame,
                        "max_root_xy_error_m": args.phase_sync_max_root_xy_error_m,
                        "max_root_z_error_m": args.phase_sync_max_root_z_error_m,
                        "max_pre_sit_penetration_m": args.phase_sync_max_pre_sit_penetration_m,
                        "min_seat_support_force_n": args.phase_sync_min_seat_support_force_n,
                        "seat_debounce_s": args.phase_sync_seat_debounce_s,
                        "max_hold_s": args.phase_sync_max_hold_s,
                    }
                    if phase_sync is not None else None
                ),
                "reference_joint_limit_projection": reference_joint_limit_report,
            },
            "model": {"nq": int(final_model.nq), "nv": int(final_model.nv), "nu": int(final_model.nu), "nmocap": int(final_model.nmocap), "nkey": int(final_model.nkey), "nsensordata": int(final_model.nsensordata), "residual_dim": int(residual_dim)},
            "output_xml": str(args.output_xml),
        }
        if final_model.nkey != len(qpos) or final_model.nsensordata < residual_dim:
            raise RuntimeError("compiled task model violates the reference or residual dimension contract")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False))
    finally:
        combined.unlink(missing_ok=True)
        if temporary_xml is not None:
            temporary_xml.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
