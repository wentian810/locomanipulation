#!/usr/bin/env python3
"""Validate robot-object contact with kinematic or finite-torque PD control.

The default ``pd_dynamic_validation`` mode drives existing MuJoCo motors with
finite PD torques.  A tracked mocap target and soft weld hold the object during
approach; release is selected from a stable multi-contact window.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import tempfile
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import numpy as np

from render_robot_motion_headless import (
    DEFAULT_SHARPA_ROOT,
    ROBOT_BASE_DICT,
    ROBOT_XML_DICT,
    assign_named_qpos,
    build_h1_sharpa_visual_xml,
    camera_sample_index,
    hide_original_g1_rubber_hands,
    hide_original_h1_hands,
    joint_qpos_map,
    load_external_hand_motion,
    load_motion,
    load_object_motion,
    parse_vec,
    qpos_joint_names,
)


def _mesh_asset(asset, name, path, scale):
    ET.SubElement(
        asset,
        "mesh",
        {
            "name": name,
            "file": str(path),
            "scale": " ".join(f"{value:.9g}" for value in scale),
        },
    )


def _pair_friction(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    sliding = float(values[0]) if len(values) else 1.0
    torsional = float(values[1]) if len(values) > 1 else 0.05
    rolling = float(values[2]) if len(values) > 2 else 0.005
    return [sliding, sliding, torsional, rolling, rolling]


def build_dynamic_object_xml(base_xml, object_motion, object_floor_collision):
    base_xml = pathlib.Path(base_xml)
    tree = ET.parse(base_xml)
    root = tree.getroot()
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(0, asset)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"worldbody not found in {base_xml}")

    body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "tracked_dynamic_object",
            "pos": "0 0 0",
            "quat": "1 0 0 0",
        },
    )
    ET.SubElement(
        body,
        "joint",
        {
            "name": "tracked_object_joint",
            "type": "free",
            "damping": "0.001",
            "armature": "0.0001",
        },
    )
    target_body_name = "tracked_object_mocap_target"
    ET.SubElement(
        worldbody,
        "body",
        {
            "name": target_body_name,
            "mocap": "true",
            "pos": "0 0 0",
            "quat": "1 0 0 0",
        },
    )
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    weld_name = "tracked_object_soft_weld"
    ET.SubElement(
        equality,
        "weld",
        {
            "name": weld_name,
            "body1": target_body_name,
            "body2": "tracked_dynamic_object",
            "solref": "0.01 1",
            "solimp": "0.95 0.99 0.001",
            "active": "true",
        },
    )
    if object_motion.get("physics_asset_mode") != "physics":
        raise ValueError(
            "dynamic validation requires a physics collision asset"
        )
    mass = float(object_motion.get("mass_kg", 0.0))
    center_of_mass = np.asarray(
        object_motion.get("center_of_mass_m", []),
        dtype=np.float64,
    )
    diaginertia = np.asarray(
        object_motion.get("diaginertia_kg_m2", []),
        dtype=np.float64,
    )
    if (
        mass <= 0
        or center_of_mass.shape != (3,)
        or diaginertia.shape != (3,)
        or np.any(diaginertia <= 0)
    ):
        raise ValueError("dynamic object mass/COM/inertia is invalid")
    ET.SubElement(
        body,
        "inertial",
        {
            "mass": f"{mass:.9g}",
            "pos": " ".join(f"{value:.9g}" for value in center_of_mass),
            "diaginertia": " ".join(
                f"{value:.9g}" for value in diaginertia
            ),
        },
    )

    collision_geom_names = []
    if object_motion["object_type"] == "mesh":
        _mesh_asset(
            asset,
            "tracked_object_visual_mesh",
            object_motion["visual_mesh_path"],
            object_motion["mesh_scale"],
        )
        ET.SubElement(
            body,
            "geom",
            {
                "name": "tracked_object_visual",
                "type": "mesh",
                "mesh": "tracked_object_visual_mesh",
                "density": "0",
                "contype": "0",
                "conaffinity": "0",
                "group": "0",
                "rgba": " ".join(
                    f"{value:.8g}" for value in object_motion["rgba"]
                ),
            },
        )
        collision_paths = object_motion["collision_mesh_paths"]
        if not collision_paths:
            raise ValueError(
                "physics object has no decomposed collision meshes"
            )
        for index, path in enumerate(collision_paths):
            asset_name = f"tracked_object_collision_mesh_{index:03d}"
            geom_name = f"tracked_object_collision_{index:03d}"
            _mesh_asset(
                asset,
                asset_name,
                path,
                object_motion["mesh_scale"],
            )
            ET.SubElement(
                body,
                "geom",
                {
                    "name": geom_name,
                    "type": "mesh",
                    "mesh": asset_name,
                    "density": "0",
                    "contype": "4",
                    "conaffinity": "3",
                    "group": "3",
                    "rgba": "0.2 1 0.2 0.18",
                },
            )
            collision_geom_names.append(geom_name)
    else:
        geom_name = "tracked_object_collision_000"
        attrs = {
            "name": geom_name,
            "type": object_motion["object_type"],
            "size": " ".join(
                f"{value:.8g}" for value in object_motion["geom_size"]
            ),
            "density": "0",
            "contype": "4",
            "conaffinity": "3",
            "rgba": " ".join(
                f"{value:.8g}" for value in object_motion["rgba"]
            ),
        }
        ET.SubElement(body, "geom", attrs)
        collision_geom_names.append(geom_name)

    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    hand_geoms = [
        geom.attrib["name"]
        for geom in root.iter("geom")
        if geom.attrib.get("name", "").startswith("collision_hand_")
    ]
    pair_friction = _pair_friction(object_motion["friction"])
    solref = np.asarray(object_motion["solref"], dtype=np.float64).reshape(-1)
    if len(solref) < 2:
        solref = np.asarray([0.01, 1.0])
    for hand_name in hand_geoms:
        for object_name in collision_geom_names:
            ET.SubElement(
                contact,
                "pair",
                {
                    "name": f"{hand_name}_{object_name}",
                    "geom1": hand_name,
                    "geom2": object_name,
                    "condim": "6",
                    "friction": " ".join(
                        f"{value:.8g}" for value in pair_friction
                    ),
                    "solref": f"{solref[0]:.8g} {solref[1]:.8g}",
                    "solimp": "0.998 0.998 0.001 0.5 2",
                },
            )

    if object_floor_collision:
        floor_names = {
            "floor",
            "ground",
            "plane",
        }
        existing = {
            geom.attrib.get("name")
            for geom in root.iter("geom")
            if geom.attrib.get("name") in floor_names
        }
        for floor_name in sorted(existing):
            for object_name in collision_geom_names:
                ET.SubElement(
                    contact,
                    "pair",
                    {
                        "name": f"{floor_name}_{object_name}",
                        "geom1": floor_name,
                        "geom2": object_name,
                        "condim": "6",
                        "friction": " ".join(
                            f"{value:.8g}" for value in pair_friction
                        ),
                    },
                )

    tmp = tempfile.NamedTemporaryFile(
        prefix="gmr_dynamic_object_",
        suffix=".xml",
        dir=str(base_xml.parent),
        delete=False,
    )
    tmp.close()
    tree.write(tmp.name, encoding="unicode")
    return (
        pathlib.Path(tmp.name),
        hand_geoms,
        collision_geom_names,
        target_body_name,
        weld_name,
    )


def _set_pd_controls(
    model,
    data,
    target_qpos,
    target_qvel,
    *,
    kp,
    kd,
):
    """Set finite motor torques and return (saturated, controlled)."""
    saturated = 0
    controlled = 0
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0:
            continue
        joint_type = int(model.jnt_type[joint_id])
        # mjJNT_SLIDE=2, mjJNT_HINGE=3. Ball/free joints are not scalar.
        if joint_type not in (2, 3):
            continue
        qpos_address = int(model.jnt_qposadr[joint_id])
        dof_address = int(model.jnt_dofadr[joint_id])
        torque = (
            float(kp) * (target_qpos[qpos_address] - data.qpos[qpos_address])
            + float(kd)
            * (target_qvel[dof_address] - data.qvel[dof_address])
        )
        if bool(model.actuator_ctrllimited[actuator_id]):
            lower, upper = model.actuator_ctrlrange[actuator_id]
            clipped = float(np.clip(torque, lower, upper))
            saturated += int(abs(clipped - torque) > 1e-9)
            torque = clipped
        data.ctrl[actuator_id] = torque
        controlled += 1
    return saturated, controlled


def _stable_multicontact(
    geom_names,
    normals,
    *,
    opposed_normal_dot,
):
    distinct = set(geom_names)
    semantic = any(
        "palm" in name.lower() or "thumb" in name.lower()
        for name in distinct
    )
    opposed = False
    for first in range(len(normals)):
        for second in range(first + 1, len(normals)):
            if float(np.dot(normals[first], normals[second])) <= opposed_normal_dot:
                opposed = True
                break
        if opposed:
            break
    return len(distinct) >= 2 and semantic and opposed, len(distinct), opposed


def _longest_true_run(values):
    longest = 0
    current = 0
    for value in np.asarray(values, dtype=bool).reshape(-1):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _set_robot_frame(
    target,
    frame_idx,
    root_pos,
    root_rot_xyzw,
    dof_pos,
    qpos_map,
    source_dof_names,
    external_motion,
):
    target[:3] = root_pos[frame_idx]
    target[3:7] = root_rot_xyzw[frame_idx][[3, 0, 1, 2]]
    if external_motion is None:
        target[7 : 7 + dof_pos.shape[1]] = dof_pos[frame_idx]
        return

    class QposView:
        pass

    view = QposView()
    view.qpos = target
    assign_named_qpos(view, qpos_map, source_dof_names, dof_pos[frame_idx])
    hand_idx = camera_sample_index(
        frame_idx,
        root_pos.shape[0],
        external_motion["left_qpos"].shape[0],
    )
    assign_named_qpos(
        view,
        qpos_map,
        external_motion["left_names"],
        external_motion["left_qpos"][hand_idx],
    )
    assign_named_qpos(
        view,
        qpos_map,
        external_motion["right_names"],
        external_motion["right_qpos"][hand_idx],
    )


def simulate(args):
    if args.video_path and args.mujoco_gl:
        os.environ["MUJOCO_GL"] = args.mujoco_gl
    import mujoco as mj

    _, fps, root_pos, root_rot_xyzw, dof_pos, _, _ = load_motion(
        args.robot_motion_path
    )
    object_motion = load_object_motion(args.object_motion_path)
    base_xml = (
        pathlib.Path(args.robot_xml)
        if args.robot_xml
        else ROBOT_XML_DICT[args.robot]
    )
    external_motion = load_external_hand_motion(args.sharpa_hand_npz)
    source_dof_names = None
    xml_path = base_xml
    temp_paths = []
    if external_motion is not None:
        source_model = mj.MjModel.from_xml_path(str(base_xml))
        source_dof_names = qpos_joint_names(source_model)
        legacy_quat = parse_vec(
            args.sharpa_mount_quat,
            4,
            [0.5, 0.5, 0.5, 0.5],
        )
        mount_pos = parse_vec(args.sharpa_mount_pos, 3, [0.055, 0.0, 0.0])
        xml_path = build_h1_sharpa_visual_xml(
            base_xml,
            args.sharpa_root,
            mount_pos,
            parse_vec(
                args.sharpa_left_mount_quat,
                4,
                [0.5, -0.5, 0.5, -0.5]
                if not args.sharpa_mount_quat
                else legacy_quat,
            ),
            parse_vec(
                args.sharpa_right_mount_quat,
                4,
                legacy_quat,
            ),
            left_mount_pos=parse_vec(args.sharpa_left_mount_pos, 3, mount_pos),
            right_mount_pos=parse_vec(args.sharpa_right_mount_pos, 3, mount_pos),
        )
        temp_paths.append(xml_path)

    (
        xml_path,
        hand_geom_names,
        object_geom_names,
        target_body_name,
        weld_name,
    ) = build_dynamic_object_xml(
        xml_path,
        object_motion,
        args.object_floor_collision,
    )
    temp_paths.append(xml_path)
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)
    if external_motion is not None:
        hide_original_h1_hands(model)
        hide_original_g1_rubber_hands(model)
    qpos_map = joint_qpos_map(model)
    weld_id = mj.mj_name2id(
        model,
        mj.mjtObj.mjOBJ_EQUALITY,
        weld_name,
    )
    target_body_id = mj.mj_name2id(
        model,
        mj.mjtObj.mjOBJ_BODY,
        target_body_name,
    )
    if weld_id < 0 or target_body_id < 0:
        raise RuntimeError("soft-weld object target was not compiled")
    target_mocap_id = int(model.body_mocapid[target_body_id])
    if target_mocap_id < 0:
        raise RuntimeError("object target body is not mocap-enabled")

    joint_id = mj.mj_name2id(
        model,
        mj.mjtObj.mjOBJ_JOINT,
        "tracked_object_joint",
    )
    if joint_id < 0:
        raise RuntimeError("tracked_object_joint was not compiled")
    object_qpos_adr = int(model.jnt_qposadr[joint_id])
    object_dof_adr = int(model.jnt_dofadr[joint_id])
    object_slice = slice(object_qpos_adr, object_qpos_adr + 7)
    object_dof_slice = slice(object_dof_adr, object_dof_adr + 6)
    object_body_id = mj.mj_name2id(
        model,
        mj.mjtObj.mjOBJ_BODY,
        "tracked_dynamic_object",
    )

    frame_dt = 1.0 / float(fps)
    substeps = max(1, int(args.substeps))
    model.opt.timestep = frame_dt / substeps
    automatic_release = str(args.release_frame).lower() == "auto"
    requested_release_frame = (
        None
        if automatic_release
        else max(0, int(args.release_frame))
    )
    release_start_frame = None
    release_frame = None
    object_count = len(object_motion["position"])

    valid_object_indices = np.where(object_motion["valid"])[0]
    if not len(valid_object_indices):
        raise ValueError("object motion has no valid frame")
    initial_index = camera_sample_index(
        (
            0
            if automatic_release
            else min(requested_release_frame, len(root_pos) - 1)
        ),
        len(root_pos),
        object_count,
    )
    if not object_motion["valid"][initial_index]:
        initial_index = int(
            valid_object_indices[
                np.argmin(np.abs(valid_object_indices - initial_index))
            ]
        )
    data.qpos[object_qpos_adr : object_qpos_adr + 3] = object_motion[
        "position"
    ][initial_index]
    data.qpos[object_qpos_adr + 3 : object_qpos_adr + 7] = object_motion[
        "quat_wxyz"
    ][initial_index]
    initial_robot_target = data.qpos.copy()
    _set_robot_frame(
        initial_robot_target,
        0,
        root_pos,
        root_rot_xyzw,
        dof_pos,
        qpos_map,
        source_dof_names,
        external_motion,
    )
    # One initialization write is allowed; PD mode never overwrites robot
    # joint qpos during playback.
    data.qpos[: object_qpos_adr] = initial_robot_target[:object_qpos_adr]
    data.mocap_pos[target_mocap_id] = object_motion["position"][initial_index]
    data.mocap_quat[target_mocap_id] = object_motion["quat_wxyz"][initial_index]
    mj.mj_forward(model, data)

    renderer = None
    writer = None
    camera = None
    if args.video_path:
        renderer = mj.Renderer(model, height=args.height, width=args.width)
        writer = imageio.get_writer(args.video_path, fps=fps)
        camera = mj.MjvCamera()
        mj.mjv_defaultCamera(camera)
        camera.distance = args.radius
        camera.elevation = args.elevation
        camera.azimuth = args.azimuth

    object_geom_ids = {
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
        for name in object_geom_names
    }
    hand_geom_ids = {
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
        for name in hand_geom_names
    }
    positions = []
    quaternions = []
    contact_counts = []
    max_contact_forces = []
    distinct_contact_geoms = []
    opposed_contact = []
    max_penetrations = []
    max_contact_impulses = []
    floor_contacts = []
    object_com_positions = []
    hand_centers = []
    relative_hand_object_rotations = []
    actuator_saturated = 0
    actuator_commands = 0
    robot_tracking_errors = []
    stable_contact_run = 0
    floor_geom_ids = {
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
        for name in ("floor", "ground", "plane")
    }
    floor_geom_ids.discard(-1)

    try:
        frame_count = len(root_pos)
        if args.max_frames > 0:
            frame_count = min(frame_count, args.max_frames)
        for frame_idx in range(frame_count):
            current_target = data.qpos.copy()
            next_target = data.qpos.copy()
            _set_robot_frame(
                current_target,
                frame_idx,
                root_pos,
                root_rot_xyzw,
                dof_pos,
                qpos_map,
                source_dof_names,
                external_motion,
            )
            next_idx = min(frame_idx + 1, len(root_pos) - 1)
            _set_robot_frame(
                next_target,
                next_idx,
                root_pos,
                root_rot_xyzw,
                dof_pos,
                qpos_map,
                source_dof_names,
                external_motion,
            )

            tracked_idx = camera_sample_index(
                frame_idx,
                len(root_pos),
                object_count,
            )
            if object_motion["valid"][tracked_idx]:
                data.mocap_pos[target_mocap_id] = object_motion[
                    "position"
                ][tracked_idx]
                data.mocap_quat[target_mocap_id] = object_motion[
                    "quat_wxyz"
                ][tracked_idx]

            if (
                release_start_frame is None
                and requested_release_frame is not None
                and frame_idx >= requested_release_frame
                and (
                    args.mode == "kinematic_contact_check"
                    or stable_contact_run >= args.min_contact_frames
                )
            ):
                release_start_frame = frame_idx
            if release_start_frame is not None:
                ramp_offset = frame_idx - release_start_frame
                if ramp_offset >= args.release_ramp_frames:
                    data.eq_active[weld_id] = 0
                    if release_frame is None:
                        release_frame = frame_idx
                else:
                    fraction = (ramp_offset + 1) / max(
                        args.release_ramp_frames,
                        1,
                    )
                    model.eq_solref[weld_id, 0] = (
                        (1.0 - fraction) * 0.01 + fraction * 0.30
                    )

            object_qpos = data.qpos[object_slice].copy()
            object_qvel = data.qvel[object_dof_slice].copy()
            velocity = np.zeros(model.nv, dtype=np.float64)
            mj.mj_differentiatePos(
                model,
                velocity,
                frame_dt,
                current_target,
                next_target,
            )
            frame_contacts = 0
            frame_force = 0.0
            frame_impulse = 0.0
            frame_penetration = 0.0
            frame_hand_normals = {}
            frame_hand_ids = set()
            frame_floor_contact = False
            for substep in range(substeps):
                desired = current_target.copy()
                mj.mj_integratePos(
                    model,
                    desired,
                    velocity,
                    model.opt.timestep * substep,
                )
                desired[object_slice] = object_qpos
                if args.mode == "kinematic_contact_check":
                    data.qpos[:] = desired
                    data.qvel[:] = velocity
                    data.qvel[object_dof_slice] = object_qvel
                else:
                    # The initial validation tier keeps only the floating base
                    # on its reference; every scalar robot joint is finite-PD.
                    data.qpos[:7] = desired[:7]
                    data.qvel[:6] = velocity[:6]
                    saturated, controlled = _set_pd_controls(
                        model,
                        data,
                        desired,
                        velocity,
                        kp=args.kp,
                        kd=args.kd,
                    )
                    actuator_saturated += saturated
                    actuator_commands += controlled
                mj.mj_step(model, data)
                object_qpos = data.qpos[object_slice].copy()
                object_qvel = data.qvel[object_dof_slice].copy()
                for contact_index in range(data.ncon):
                    contact = data.contact[contact_index]
                    pair = {int(contact.geom1), int(contact.geom2)}
                    if pair & object_geom_ids and pair & hand_geom_ids:
                        frame_contacts += 1
                        hand_id = next(iter(pair & hand_geom_ids))
                        frame_hand_ids.add(hand_id)
                        hand_name = mj.mj_id2name(
                            model,
                            mj.mjtObj.mjOBJ_GEOM,
                            hand_id,
                        )
                        normal = np.asarray(
                            contact.frame[:3],
                            dtype=np.float64,
                        )
                        if int(contact.geom1) not in object_geom_ids:
                            normal = -normal
                        frame_hand_normals.setdefault(
                            hand_name,
                            [],
                        ).append(normal)
                        force = np.zeros(6, dtype=np.float64)
                        mj.mj_contactForce(model, data, contact_index, force)
                        frame_force = max(frame_force, float(np.linalg.norm(force[:3])))
                        frame_impulse = max(
                            frame_impulse,
                            float(np.linalg.norm(force[:3]))
                            * model.opt.timestep,
                        )
                        frame_penetration = max(
                            frame_penetration,
                            max(0.0, -float(contact.dist)),
                        )
                    elif pair & object_geom_ids and pair & floor_geom_ids:
                        frame_floor_contact = True
                        frame_penetration = max(
                            frame_penetration,
                            max(0.0, -float(contact.dist)),
                        )

            averaged_normals = []
            for values in frame_hand_normals.values():
                normal = np.mean(np.asarray(values), axis=0)
                norm = np.linalg.norm(normal)
                if norm > 1e-9:
                    averaged_normals.append(normal / norm)
            stable, distinct_count, has_opposed = _stable_multicontact(
                list(frame_hand_normals),
                averaged_normals,
                opposed_normal_dot=args.opposed_normal_dot,
            )
            if stable:
                stable_contact_run += 1
            else:
                stable_contact_run = 0
            if (
                automatic_release
                and release_start_frame is None
                and stable_contact_run >= args.min_contact_frames
            ):
                release_start_frame = frame_idx + 1

            positions.append(object_qpos[:3].copy())
            quaternions.append(object_qpos[3:7].copy())
            contact_counts.append(frame_contacts)
            max_contact_forces.append(frame_force)
            distinct_contact_geoms.append(distinct_count)
            opposed_contact.append(has_opposed)
            max_penetrations.append(frame_penetration)
            max_contact_impulses.append(frame_impulse)
            floor_contacts.append(frame_floor_contact)
            object_com_positions.append(data.xipos[object_body_id].copy())
            if frame_hand_ids:
                hand_centers.append(
                    np.mean(
                        [data.geom_xpos[geom_id] for geom_id in frame_hand_ids],
                        axis=0,
                    )
                )
                selected_hand_id = sorted(
                    frame_hand_ids,
                    key=lambda geom_id: (
                        "palm"
                        not in (
                            mj.mj_id2name(
                                model,
                                mj.mjtObj.mjOBJ_GEOM,
                                geom_id,
                            )
                            or ""
                        ).lower(),
                        "thumb"
                        not in (
                            mj.mj_id2name(
                                model,
                                mj.mjtObj.mjOBJ_GEOM,
                                geom_id,
                            )
                            or ""
                        ).lower(),
                        geom_id,
                    ),
                )[0]
                hand_rotation = data.geom_xmat[selected_hand_id].reshape(3, 3)
                object_rotation_flat = np.zeros(9, dtype=np.float64)
                mj.mju_quat2Mat(
                    object_rotation_flat,
                    object_qpos[3:7],
                )
                relative_hand_object_rotations.append(
                    hand_rotation.T
                    @ object_rotation_flat.reshape(3, 3)
                )
            else:
                hand_centers.append(np.full(3, np.nan))
                relative_hand_object_rotations.append(
                    np.full((3, 3), np.nan)
                )
            scalar_errors = []
            for actuator_id in range(model.nu):
                joint_id = int(model.actuator_trnid[actuator_id, 0])
                if joint_id < 0 or int(model.jnt_type[joint_id]) not in (2, 3):
                    continue
                qpos_address = int(model.jnt_qposadr[joint_id])
                scalar_errors.append(
                    abs(desired[qpos_address] - data.qpos[qpos_address])
                )
            if scalar_errors:
                robot_tracking_errors.extend(scalar_errors)

            if renderer is not None:
                base_name = ROBOT_BASE_DICT.get(args.robot)
                if base_name:
                    base_id = mj.mj_name2id(
                        model,
                        mj.mjtObj.mjOBJ_BODY,
                        base_name,
                    )
                    if base_id >= 0:
                        camera.lookat[:] = data.xpos[base_id]
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        for path in temp_paths:
            try:
                pathlib.Path(path).unlink()
            except OSError:
                pass

    positions = np.asarray(positions, dtype=np.float32)
    quaternions = np.asarray(quaternions, dtype=np.float32)
    contact_counts = np.asarray(contact_counts, dtype=np.int32)
    max_contact_forces = np.asarray(max_contact_forces, dtype=np.float32)
    distinct_contact_geoms = np.asarray(
        distinct_contact_geoms,
        dtype=np.int16,
    )
    opposed_contact = np.asarray(opposed_contact, dtype=bool)
    max_penetrations = np.asarray(max_penetrations, dtype=np.float32)
    max_contact_impulses = np.asarray(
        max_contact_impulses,
        dtype=np.float32,
    )
    floor_contacts = np.asarray(floor_contacts, dtype=bool)
    object_com_positions = np.asarray(
        object_com_positions,
        dtype=np.float32,
    )
    hand_centers = np.asarray(hand_centers, dtype=np.float32)
    relative_hand_object_rotations = np.asarray(
        relative_hand_object_rotations,
        dtype=np.float32,
    )
    effective_release_frame = (
        int(release_frame) if release_frame is not None else -1
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        schema_version=np.asarray(2, dtype=np.int32),
        source=np.asarray(args.mode),
        fps=np.asarray(fps, dtype=np.float32),
        position=positions,
        center_of_mass_position=object_com_positions,
        quat_wxyz=quaternions,
        contact_count=contact_counts,
        distinct_hand_contact_geoms=distinct_contact_geoms,
        opposed_contact=opposed_contact,
        floor_contact=floor_contacts,
        max_normal_force=max_contact_forces,
        max_penetration=max_penetrations,
        max_contact_impulse=max_contact_impulses,
        release_start_frame=np.asarray(
            release_start_frame
            if release_start_frame is not None
            else -1,
            dtype=np.int32,
        ),
        release_frame=np.asarray(
            effective_release_frame,
            dtype=np.int32,
        ),
        visual_mesh_path=np.asarray(str(object_motion["visual_mesh_path"])),
        collision_mesh_paths=np.asarray(
            [str(path) for path in object_motion["collision_mesh_paths"]],
            dtype=str,
        ),
    )

    metric_release_frame = (
        effective_release_frame
        if effective_release_frame >= 0
        else len(positions) - 1
    )
    release_z = float(
        object_com_positions[
            min(metric_release_frame, len(object_com_positions) - 1),
            2,
        ]
    )
    lift = object_com_positions[:, 2] - release_z
    lifted = lift >= args.lift_threshold
    contact_after_release = contact_counts.copy()
    post_release = (
        np.arange(len(positions)) >= effective_release_frame
        if effective_release_frame >= 0
        else np.zeros(len(positions), dtype=bool)
    )
    contact_after_release[~post_release] = 0

    # Extended metrics.
    contact_any = contact_counts > 0
    first_contact_frame = int(np.argmax(contact_any)) if contact_any.any() else -1
    # First lift frame: first frame after release where lift >= threshold.
    first_lift_frame = -1
    lift_after = np.where(lifted & post_release)[0]
    if len(lift_after) > 0:
        first_lift_frame = int(lift_after[0])
    # Horizontal drift after release.
    if 0 <= effective_release_frame < len(positions):
        release_pos = positions[effective_release_frame, :2].copy()
        horizontal_drift = np.linalg.norm(
            positions[effective_release_frame:, :2] - release_pos[None, :],
            axis=1,
        )
        max_horizontal_drift = float(horizontal_drift.max())
    else:
        max_horizontal_drift = 0.0
    # Object explosion check: velocity spikes.
    if len(positions) > 2:
        velocities = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        object_exploded = bool((velocities > 5.0).any())  # 5m/frame is ~150m/s at 30fps
    else:
        object_exploded = False
    # Invalid force spike.
    invalid_force_spike = float(max_contact_forces.max(initial=0.0)) > 500.0
    stable_contact = (
        (distinct_contact_geoms >= 2)
        & opposed_contact
        & (contact_counts > 0)
    )
    contact_duration_frames = _longest_true_run(stable_contact)
    opposed_contact_ratio = (
        float(np.mean(opposed_contact[contact_any]))
        if np.any(contact_any)
        else 0.0
    )
    finite_relative = (
        np.isfinite(hand_centers).all(axis=1)
        & (np.arange(len(hand_centers)) >= max(effective_release_frame, 0))
    )
    relative_drift_m = None
    relative_rotation_deg = None
    if np.any(finite_relative):
        indices = np.where(finite_relative)[0]
        relative_position = (
            object_com_positions[indices] - hand_centers[indices]
        )
        relative_drift_m = float(
            np.max(
                np.linalg.norm(
                    relative_position - relative_position[0],
                    axis=1,
                )
            )
        )
        rotations = relative_hand_object_rotations[indices]
        rotations = rotations[
            np.isfinite(rotations).all(axis=(1, 2))
        ]
        if len(rotations):
            relative_rotation_deg = float(
                max(
                    np.degrees(
                        np.arccos(
                            np.clip(
                                (
                                    np.trace(rotations[0].T @ rotation)
                                    - 1.0
                                )
                                * 0.5,
                                -1.0,
                                1.0,
                            )
                        )
                    )
                    for rotation in rotations
                )
            )
    floor_contact_after_lift_ratio = (
        float(np.mean(floor_contacts[lifted]))
        if np.any(lifted)
        else 1.0
    )
    object_drop_frame = -1
    if len(lift_after):
        after_lift = np.where(
            (np.arange(len(lift)) > int(lift_after[0]))
            & (lift < 0.5 * args.lift_threshold)
        )[0]
        if len(after_lift):
            object_drop_frame = int(after_lift[0])
    actuator_saturation_ratio = (
        float(actuator_saturated / actuator_commands)
        if actuator_commands
        else 0.0
    )
    robot_tracking_error_p95 = (
        float(np.percentile(robot_tracking_errors, 95))
        if robot_tracking_errors
        else None
    )
    dynamic_success = bool(
        args.mode == "pd_dynamic_validation"
        and effective_release_frame >= 0
        and float(lift.max(initial=0.0)) >= args.lift_threshold
        and contact_duration_frames >= args.min_stable_contact_frames
        and relative_drift_m is not None
        and relative_drift_m <= args.max_relative_drift
        and float(max_penetrations.max(initial=0.0)) <= args.max_penetration
        and not invalid_force_spike
        and floor_contact_after_lift_ratio <= args.max_floor_contact_after_lift_ratio
        and actuator_saturation_ratio <= args.max_actuator_saturation_ratio
    )
    failure_reasons = []
    if effective_release_frame < 0:
        failure_reasons.append("no_stable_multicontact_release_window")
    if contact_duration_frames < args.min_stable_contact_frames:
        failure_reasons.append(
            "stable_contact_duration_below_threshold"
        )
    if relative_drift_m is None or relative_drift_m > args.max_relative_drift:
        failure_reasons.append("relative_hand_object_drift_too_large")
    if float(max_penetrations.max(initial=0.0)) > args.max_penetration:
        failure_reasons.append("hand_object_penetration_too_large")
    if invalid_force_spike:
        failure_reasons.append("invalid_contact_force_spike")
    if actuator_saturation_ratio > args.max_actuator_saturation_ratio:
        failure_reasons.append("actuator_saturation_too_high")
    if float(lift.max(initial=0.0)) < args.lift_threshold:
        failure_reasons.append("object_lift_below_threshold")

    report = {
        "schema_version": 2,
        "validation_level": args.mode,
        "same_physics_world": True,
        "object_body_type": "free_joint_dynamic",
        "tracked_reference_drive": (
            "soft_weld_to_mocap_until_release"
        ),
        "output": str(args.output),
        "video": str(args.video_path) if args.video_path else None,
        "frames": len(positions),
        "release_mode": "auto_contact_window" if automatic_release else "explicit",
        "release_start_frame": release_start_frame,
        "release_frame": effective_release_frame,
        "collision_parts": len(object_geom_names),
        "hand_collision_geoms": len(hand_geom_names),
        "contact_frames_after_release": int(
            np.count_nonzero(contact_after_release)
        ),
        "max_contact_force_n": float(max_contact_forces.max(initial=0.0)),
        "max_contact_impulse_ns": float(
            max_contact_impulses.max(initial=0.0)
        ),
        "max_penetration_m": float(max_penetrations.max(initial=0.0)),
        "distinct_hand_contact_geoms": int(
            distinct_contact_geoms.max(initial=0)
        ),
        "contact_duration_frames": int(contact_duration_frames),
        "opposed_contact_ratio": opposed_contact_ratio,
        "relative_hand_object_drift_m": relative_drift_m,
        "relative_hand_object_rotation_deg": relative_rotation_deg,
        "floor_contact_after_lift_ratio": floor_contact_after_lift_ratio,
        "object_com_lift_m": float(lift.max(initial=0.0)),
        "object_drop_frame": object_drop_frame,
        "actuator_saturation_ratio": actuator_saturation_ratio,
        "robot_tracking_error_p95": robot_tracking_error_p95,
        "max_lift_m": float(lift.max(initial=0.0)),
        "lift_threshold_m": float(args.lift_threshold),
        "dynamic_lift_success": dynamic_success,
        "failure_reasons": failure_reasons,
        "contact_success": bool(
            contact_any.any()
        ),
        "object_dropped": bool(
            not np.any(lifted & (contact_after_release > 0))
            and contact_any.any()
        ),
        "first_contact_frame": first_contact_frame,
        "first_lift_frame": first_lift_frame,
        "max_horizontal_drift_m": round(max_horizontal_drift, 4),
        "object_exploded": object_exploded,
        "invalid_force_spike": invalid_force_spike,
        "interpretation": (
            "pd_dynamic_validation uses finite motor torques and releases a "
            "soft weld only after stable multi-contact; "
            "kinematic_contact_check is not dynamic proof."
        ),
    }
    if object_exploded:
        report[
            "interpretation"
        ] += " OBJECT EXPLODED: velocity spike detected, likely collision/scale failure."
    if invalid_force_spike:
        report[
            "interpretation"
        ] += " INVALID_FORCE: contact force > 500N, physics anomaly."
    report_path = args.output.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate a free object with kinematic or finite-PD robot control."
    )
    parser.add_argument("--robot_motion_path", type=pathlib.Path, required=True)
    parser.add_argument("--object_motion_path", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--video_path", type=pathlib.Path, default=None)
    parser.add_argument("--robot", default="unitree_h1_with_hand")
    parser.add_argument("--robot_xml", default="")
    parser.add_argument("--sharpa_hand_npz", default="")
    parser.add_argument("--sharpa_root", default=str(DEFAULT_SHARPA_ROOT))
    parser.add_argument("--sharpa_mount_pos", default="0.055,0,0")
    parser.add_argument("--sharpa_left_mount_pos", default="")
    parser.add_argument("--sharpa_right_mount_pos", default="")
    parser.add_argument("--sharpa_mount_quat", default="")
    parser.add_argument("--sharpa_left_mount_quat", default="")
    parser.add_argument("--sharpa_right_mount_quat", default="")
    parser.add_argument(
        "--mode",
        choices=["kinematic_contact_check", "pd_dynamic_validation"],
        default="pd_dynamic_validation",
    )
    parser.add_argument("--release_frame", default="auto")
    parser.add_argument("--release_ramp_frames", type=int, default=10)
    parser.add_argument("--min_contact_frames", type=int, default=5)
    parser.add_argument("--min_stable_contact_frames", type=int, default=10)
    parser.add_argument("--opposed_normal_dot", type=float, default=-0.2)
    parser.add_argument("--kp", type=float, default=80.0)
    parser.add_argument("--kd", type=float, default=4.0)
    parser.add_argument("--max_relative_drift", type=float, default=0.05)
    parser.add_argument("--max_penetration", type=float, default=0.01)
    parser.add_argument(
        "--max_floor_contact_after_lift_ratio",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--max_actuator_saturation_ratio",
        type=float,
        default=0.10,
    )
    parser.add_argument("--substeps", type=int, default=16)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--lift_threshold", type=float, default=0.05)
    parser.add_argument("--object_floor_collision", action="store_true")
    parser.add_argument("--mujoco_gl", default="osmesa")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--elevation", type=float, default=-12.0)
    parser.add_argument("--azimuth", type=float, default=135.0)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    simulate(parse_args())
