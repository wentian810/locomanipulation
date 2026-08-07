#!/usr/bin/env python3
"""Render and audit a GMR trajectory with a reconstructed static MuJoCo scene.

The physical mode initializes the floating base once, then writes only bounded
joint position targets and advances continuous MuJoCo time with ``mj_step``.
It never restores root ``qpos``/``qvel`` frame by frame, so a fall or contact
response is measured rather than hidden by a kinematic reset.  The legacy
``kinematic_reference`` mode remains available only for comparison.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np


def _merge_scene(robot_xml: Path, scene_xml: Path) -> Path:
    tree = ET.parse(robot_xml)
    root = tree.getroot()
    world = root.find("worldbody")
    if world is None:
        raise ValueError("robot MJCF has no worldbody")
    assets = root.find("asset")
    if assets is None:
        assets = ET.Element("asset")
        root.insert(0, assets)
    scene_root = ET.parse(scene_xml).getroot()
    scene_assets = scene_root.find("asset")
    if scene_assets is not None:
        for item in scene_assets:
            clone = copy.deepcopy(item)
            if clone.tag == "mesh" and clone.get("file"):
                mesh = Path(clone.get("file"))
                if not mesh.is_absolute():
                    mesh = (scene_xml.parent / mesh).resolve()
                clone.set("file", mesh.as_posix())
            assets.append(clone)
    scene_world = scene_root.find("worldbody")
    if scene_world is None:
        raise ValueError("scene MJCF has no worldbody")
    for item in scene_world:
        world.append(copy.deepcopy(item))
    handle = tempfile.NamedTemporaryFile(prefix="gmr_scene_mjstep_", suffix=".xml", dir=str(robot_xml.parent), delete=False)
    handle.close()
    tree.write(handle.name, encoding="unicode")
    return Path(handle.name)


def _configure_bounded_position_actuators(xml_path: Path, profile_path: Path) -> dict[str, object]:
    """Replace unit-torque motors with bounded MuJoCo position servos."""
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    limits = profile.get("joint_torque_limits_nm")
    servo = profile.get("low_level_position_servo", {})
    if not isinstance(limits, dict) or not isinstance(servo, dict):
        raise ValueError(f"invalid actuator profile: {profile_path}")
    kp = float(servo.get("kp_nm_per_rad", 400.0))
    dampratio = float(servo.get("dampratio", 1.0))
    joint_dynamics = servo.get("joint_dynamics")
    if not isinstance(joint_dynamics, dict):
        raise ValueError(f"actuator profile has no joint dynamics: {profile_path}")
    damping = float(joint_dynamics.get("damping_nms_per_rad", -1.0))
    armature = float(joint_dynamics.get("armature_kgm2", -1.0))
    frictionloss = float(joint_dynamics.get("frictionloss_nm", -1.0))
    if damping < 0.0 or armature < 0.0 or frictionloss < 0.0:
        raise ValueError(f"invalid joint dynamics in profile: {profile_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    joint_ranges: dict[str, str] = {}
    joints_by_name: dict[str, ET.Element] = {}
    for joint in root.findall(".//joint"):
        name = joint.get("name")
        if name:
            joints_by_name[name] = joint
            if joint.get("range"):
                joint_ranges[name] = joint.get("range")
    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError("robot MJCF has no actuator section")
    actuators = list(actuator)
    if len(actuators) != len(limits):
        raise ValueError(f"actuator/profile mismatch: xml={len(actuators)} profile={len(limits)}")
    names: list[str] = []
    for item in actuators:
        joint_name = item.get("joint") or item.get("name")
        if joint_name not in limits:
            raise ValueError(f"actuator joint is missing from profile: {joint_name}")
        item.tag = "position"
        item.attrib.clear()
        item.set("name", joint_name)
        item.set("joint", joint_name)
        item.set("kp", f"{kp:.9g}")
        item.set("dampratio", f"{dampratio:.9g}")
        item.set("ctrlrange", joint_ranges.get(joint_name, "-3.14159 3.14159"))
        item.set("ctrllimited", "true")
        item.set("forcerange", f"{-float(limits[joint_name]):.9g} {float(limits[joint_name]):.9g}")
        item.set("forcelimited", "true")
        joint = joints_by_name.get(joint_name)
        if joint is None:
            raise ValueError(f"actuator joint is absent from model: {joint_name}")
        joint.set("damping", f"{damping:.9g}")
        joint.set("armature", f"{armature:.9g}")
        joint.set("frictionloss", f"{frictionloss:.9g}")
        names.append(joint_name)
    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="unicode")
    return {
        "profile": str(profile_path.resolve()),
        "actuator_type": "position",
        "kp_nm_per_rad": kp,
        "dampratio": dampratio,
        "joint_dynamics": {
            "damping_nms_per_rad": damping,
            "armature_kgm2": armature,
            "frictionloss_nm": frictionloss,
        },
        "joint_names": names,
    }


def _robot_descendant(model: mujoco.MjModel, body_id: int, pelvis_id: int) -> bool:
    current = int(body_id)
    while current > 0:
        if current == pelvis_id:
            return True
        parent = int(model.body_parentid[current])
        if parent == current:
            break
        current = parent
    return current == pelvis_id


def _summary(values: np.ndarray) -> dict[str, float]:
    if not len(values):
        return {"min": float("nan"), "p50": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {"min": float(np.min(values)), "p50": float(np.quantile(values, .5)), "p95": float(np.quantile(values, .95)), "max": float(np.max(values))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-motion", required=True, type=Path)
    parser.add_argument("--robot-xml", required=True, type=Path)
    parser.add_argument("--scene-mujoco-xml", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--mujoco-gl", default="egl")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument(
        "--control-mode", choices=("bounded_reference_pd", "kinematic_reference"),
        default="bounded_reference_pd",
        help="continuous physical PD mode (default) or legacy per-frame qpos replay",
    )
    parser.add_argument(
        "--actuator-profile", type=Path,
        default=Path(__file__).with_name("g1_unitree_actuator_limits.json"),
    )
    parser.add_argument("--min-root-height-m", type=float, default=0.25)
    parser.add_argument("--max-root-step-m", type=float, default=0.08)
    parser.add_argument(
        "--trajectory-npz", type=Path,
        help="optional continuous physical state log for root-failure diagnosis",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--camera-radius", type=float, default=3.8)
    parser.add_argument("--camera-elevation", type=float, default=12.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--max-negative-contact-distance-m", type=float, default=0.012)
    parser.add_argument(
        "--max-negative-visible-mesh-distance-m", type=float, default=0.012,
        help="Maximum permitted visible robot-mesh penetration into a collidable scene geom.",
    )
    args = parser.parse_args()
    if args.substeps < 1:
        raise ValueError("substeps must be positive")
    if args.trajectory_npz is not None and args.trajectory_npz.exists():
        raise FileExistsError(f"refusing to overwrite physical trajectory log: {args.trajectory_npz}")
    if min(
        args.max_negative_contact_distance_m,
        args.max_negative_visible_mesh_distance_m,
    ) < 0.0:
        raise ValueError("negative-distance thresholds must be non-negative")
    if args.mujoco_gl:
        os.environ["MUJOCO_GL"] = args.mujoco_gl
        os.environ.setdefault("PYOPENGL_PLATFORM", args.mujoco_gl)
    with args.robot_motion.open("rb") as stream:
        motion = pickle.load(stream)
    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_rot = np.asarray(motion["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
    fps = float(np.asarray(motion.get("fps", 30.0)).reshape(-1)[0])
    if root_pos.ndim != 2 or root_pos.shape[1] != 3 or root_rot.shape != (len(root_pos), 4) or dof_pos.shape[0] != len(root_pos):
        raise ValueError("robot motion must contain aligned root_pos/root_rot/dof_pos arrays")
    merged = _merge_scene(args.robot_xml.resolve(), args.scene_mujoco_xml.resolve())
    try:
        actuation = {"mode": args.control_mode}
        if args.control_mode == "bounded_reference_pd":
            actuation.update(_configure_bounded_position_actuators(merged, args.actuator_profile.resolve()))
        model = mujoco.MjModel.from_xml_path(str(merged))
        if model.nq != 7 + dof_pos.shape[1]:
            raise ValueError(f"robot dof count {dof_pos.shape[1]} does not match compiled MJCF nq={model.nq}")
        model.opt.timestep = 1.0 / (fps * args.substeps)
        data = mujoco.MjData(model)
        pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if pelvis_id < 0:
            raise ValueError("robot MJCF must expose a pelvis body")
        robot_geom_ids = {i for i in range(model.ngeom) if _robot_descendant(model, int(model.geom_bodyid[i]), pelvis_id)}
        visible_robot_mesh_ids = {
            geom_id for geom_id in robot_geom_ids
            if int(model.geom_type[geom_id]) == mujoco.mjtGeom.mjGEOM_MESH
        }
        scene_geom_ids = set(range(model.ngeom)) - robot_geom_ids
        collision_scene_ids = {i for i in scene_geom_ids if int(model.geom_contype[i]) != 0}
        if not collision_scene_ids:
            raise ValueError("scene MJCF has no collidable static geoms")
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.distance = max(args.camera_radius, float(np.quantile(np.linalg.norm(root_pos[:, :2] - np.median(root_pos[:, :2], axis=0), axis=1), .95)) + 2.0)
        camera.elevation = args.camera_elevation
        camera.azimuth = args.camera_azimuth
        camera.lookat[:] = np.median(root_pos, axis=0) + np.array([0.0, 0.0, 0.55])
        args.video.parent.mkdir(parents=True, exist_ok=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        contact_count, min_distance, root_delta = [], [], []
        actual_root, actual_root_height = [], []
        actual_qpos, actual_qvel = [], []
        contact_frames = []
        pair_distances: dict[tuple[int, int], list[float]] = {}
        pair_frames: dict[tuple[int, int], list[int]] = {}
        visible_mesh_min_distance = []
        visible_mesh_pairs: dict[tuple[int, int], list[float]] = {}
        frame_count = len(root_pos) if args.max_frames <= 0 else min(len(root_pos), args.max_frames)
        with imageio.get_writer(str(args.video), format="FFMPEG", fps=fps, macro_block_size=1) as writer:
            if args.control_mode == "bounded_reference_pd":
                # The only direct state write in physical mode.  All later
                # motion comes from bounded actuator targets and mj_step.
                data.qpos[:3] = root_pos[0]
                data.qpos[3:7] = root_rot[0][[3, 0, 1, 2]]
                data.qpos[7:] = dof_pos[0]
                data.qvel[:] = 0.0
                data.ctrl[:] = dof_pos[0]
                # MuJoCo requires one forward pass after the initial free-joint
                # state write.  This is initialization only; no later frame
                # restores qpos/qvel or calls mj_forward.
                mujoco.mj_forward(model, data)
            for frame in range(frame_count):
                if args.control_mode == "kinematic_reference":
                    data.qpos[:3] = root_pos[frame]
                    data.qpos[3:7] = root_rot[frame][[3, 0, 1, 2]]
                    data.qpos[7:] = dof_pos[frame]
                    data.qvel[:] = 0.0
                else:
                    # Root position/orientation remain dynamical.  Only the
                    # bounded joint targets are refreshed at video rate.
                    data.ctrl[:] = dof_pos[frame]
                for _ in range(args.substeps):
                    mujoco.mj_step(model, data)
                frame_contacts = []
                for index in range(data.ncon):
                    contact = data.contact[index]
                    pair = {int(contact.geom1), int(contact.geom2)}
                    if pair & robot_geom_ids and pair & collision_scene_ids:
                        robot_geom = next(geom for geom in pair if geom in robot_geom_ids)
                        scene_geom = next(geom for geom in pair if geom in collision_scene_ids)
                        key = (robot_geom, scene_geom)
                        distance = float(contact.dist)
                        frame_contacts.append(distance)
                        pair_distances.setdefault(key, []).append(distance)
                        pair_frames.setdefault(key, []).append(frame)
                contact_count.append(len(frame_contacts))
                min_distance.append(min(frame_contacts) if frame_contacts else np.nan)
                root_delta.append(float(np.linalg.norm(data.qpos[:3] - root_pos[frame])))
                actual_root.append(data.qpos[:3].copy())
                actual_root_height.append(float(data.qpos[2]))
                actual_qpos.append(data.qpos.copy())
                actual_qvel.append(data.qvel.copy())
                if frame_contacts:
                    contact_frames.append(frame)
                frame_visible_distances = []
                for robot_geom in visible_robot_mesh_ids:
                    for scene_geom in collision_scene_ids:
                        distance = float(
                            mujoco.mj_geomDistance(
                                model, data, robot_geom, scene_geom, 2.0, None
                            )
                        )
                        frame_visible_distances.append(distance)
                        visible_mesh_pairs.setdefault(
                            (robot_geom, scene_geom), []
                        ).append(distance)
                visible_mesh_min_distance.append(
                    min(frame_visible_distances) if frame_visible_distances else np.nan
                )
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())
        renderer.close()
        distances = np.asarray(min_distance, dtype=np.float64)
        finite_distances = distances[np.isfinite(distances)]
        minimum_distance = float(np.min(finite_distances)) if len(finite_distances) else 0.0
        collision_acceptance = minimum_distance >= -args.max_negative_contact_distance_m
        pair_summary = []
        for (robot_geom, scene_geom), values in pair_distances.items():
            pair_summary.append({
                "robot_geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom),
                "robot_body": mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[robot_geom])
                ),
                "scene_geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, scene_geom),
                "minimum_distance_m": float(np.min(values)),
                "distance_m": _summary(np.asarray(values, dtype=np.float64)),
                "contact_frame_count": int(len(set(pair_frames[(robot_geom, scene_geom)]))),
                "first_contact_frames": pair_frames[(robot_geom, scene_geom)][:32],
            })
        pair_summary.sort(key=lambda item: item["minimum_distance_m"])
        visible_mesh_values = np.asarray(visible_mesh_min_distance, dtype=np.float64)
        finite_visible_mesh_values = visible_mesh_values[np.isfinite(visible_mesh_values)]
        minimum_visible_mesh_distance = (
            float(np.min(finite_visible_mesh_values))
            if len(finite_visible_mesh_values) else 0.0
        )
        visible_mesh_acceptance = (
            minimum_visible_mesh_distance
            >= -args.max_negative_visible_mesh_distance_m
        )
        actual_root_array = np.asarray(actual_root, dtype=np.float64)
        actual_qpos_array = np.asarray(actual_qpos, dtype=np.float64)
        actual_qvel_array = np.asarray(actual_qvel, dtype=np.float64)
        actual_root_steps = (
            np.linalg.norm(np.diff(actual_root_array, axis=0), axis=1)
            if len(actual_root_array) > 1 else np.zeros(0, dtype=np.float64)
        )
        root_stability_acceptance = True
        if args.control_mode == "bounded_reference_pd":
            root_stability_acceptance = bool(
                np.min(actual_root_array[:, 2]) >= args.min_root_height_m
                and (not len(actual_root_steps) or np.max(actual_root_steps) <= args.max_root_step_m)
            )
        acceptance = bool(collision_acceptance and visible_mesh_acceptance and root_stability_acceptance)
        visible_pair_summary = []
        for (robot_geom, scene_geom), values in visible_mesh_pairs.items():
            robot_body = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[robot_geom])
            )
            visible_pair_summary.append({
                "robot_geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom),
                "robot_body": robot_body,
                "scene_geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, scene_geom),
                "minimum_distance_m": float(np.min(values)),
                "distance_m": _summary(np.asarray(values, dtype=np.float64)),
            })
        visible_pair_summary.sort(key=lambda item: item["minimum_distance_m"])
        if args.trajectory_npz is not None:
            args.trajectory_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.trajectory_npz,
                source_frame=np.arange(frame_count, dtype=np.int64),
                reference_root_pos=root_pos[:frame_count],
                reference_joint_target=dof_pos[:frame_count],
                actual_qpos=actual_qpos_array,
                actual_qvel=actual_qvel_array,
                root_reference_error=np.asarray(root_delta, dtype=np.float64),
            )
        report = {
            "schema_version": 1,
            "verdict": "pass" if acceptance else "fail",
            "mode": "mj_step_reference_control_contact_replay",
            "control_mode": args.control_mode,
            "actuation": actuation,
            "uses_mujoco_mj_step": True,
            "uses_mujoco_mj_forward": args.control_mode == "bounded_reference_pd",
            "initial_mj_forward_count": 1 if args.control_mode == "bounded_reference_pd" else 0,
            "trajectory_adjustment": "none",
            "robot_motion": str(args.robot_motion.resolve()),
            "scene_mujoco_xml": str(args.scene_mujoco_xml.resolve()),
            "frames": int(frame_count),
            "fps": fps,
            "substeps_per_video_frame": args.substeps,
            "compiled": {"ngeom": int(model.ngeom), "robot_geoms": int(len(robot_geom_ids)), "collidable_scene_geoms": int(len(collision_scene_ids))},
            "acceptance": {
                "max_negative_contact_distance_m": args.max_negative_contact_distance_m,
                "minimum_distance_m": minimum_distance,
                "collision_proxy_pass": collision_acceptance,
                "max_negative_visible_mesh_distance_m": args.max_negative_visible_mesh_distance_m,
                "minimum_visible_mesh_distance_m": minimum_visible_mesh_distance,
                "visible_mesh_pass": visible_mesh_acceptance,
                "min_root_height_m": args.min_root_height_m,
                "observed_root_height_min_m": float(np.min(actual_root_array[:, 2])),
                "max_root_step_m": args.max_root_step_m,
                "observed_root_step_max_m": float(np.max(actual_root_steps)) if len(actual_root_steps) else 0.0,
                "root_stability_pass": root_stability_acceptance,
                "pass": acceptance,
            },
            "robot_scene_contact": {
                "contact_frame_count": int(np.count_nonzero(np.asarray(contact_count))),
                "contact_frame_ratio": float(np.mean(np.asarray(contact_count) > 0)),
                "minimum_contact_distance_m": _summary(finite_distances),
                "negative_distance_frame_count": int(np.count_nonzero(finite_distances < 0.0)),
                "negative_distance_frame_ratio_over_contact_frames": float(np.mean(finite_distances < 0.0)) if len(finite_distances) else 0.0,
                "first_contact_frames": contact_frames[:32],
                "pairs_by_minimum_distance": pair_summary,
            },
            "visible_mesh_scene_distance": {
                "mesh_geom_count": int(len(visible_robot_mesh_ids)),
                "minimum_distance_m": _summary(finite_visible_mesh_values),
                "negative_distance_frame_count": int(
                    np.count_nonzero(finite_visible_mesh_values < 0.0)
                ),
                "negative_distance_frame_ratio": float(
                    np.mean(finite_visible_mesh_values < 0.0)
                ) if len(finite_visible_mesh_values) else 0.0,
                "pairs_by_minimum_distance": visible_pair_summary[:24],
            },
            "one_frame_mj_step_root_response_m": _summary(np.asarray(root_delta)),
            "limitations": ["The bounded_reference_pd mode is a deterministic reference controller, not an autonomous learned policy.", "The legacy kinematic_reference mode restores GMR qpos every frame and is not eligible for physical promotion.", "One mj_forward is used only after the single initial state write; all subsequent frames use mj_step."],
            "artifacts": {
                "video": str(args.video),
                "report": str(args.report),
                "trajectory_npz": str(args.trajectory_npz) if args.trajectory_npz is not None else None,
            },
        }
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"video": str(args.video), "contact_frames": report["robot_scene_contact"]["contact_frame_count"], "negative_distance_frames": report["robot_scene_contact"]["negative_distance_frame_count"], "verdict": report["verdict"]}, ensure_ascii=False))
    finally:
        merged.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
