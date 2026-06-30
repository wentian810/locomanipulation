#!/usr/bin/env python3
"""Replay a GMR robot kinematically while simulating a free dynamic object.

The robot follows the retargeted qpos trajectory. The object is *not* attached
to the hands and is not overwritten after ``release_frame``; it can move only
through gravity and MuJoCo contacts. This is a useful grasp/lift gate before a
full torque-controlled or MPC optimization stage.
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
            collision_paths = [object_motion["visual_mesh_path"]]
            print(
                "[WARN] No decomposed collision meshes; MuJoCo will use one "
                "convex hull of the visual mesh.",
                flush=True,
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
                    "density": f"{object_motion['density']:.9g}",
                    "contype": "0",
                    "conaffinity": "0",
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
            "density": f"{object_motion['density']:.9g}",
            "contype": "0",
            "conaffinity": "0",
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
    return pathlib.Path(tmp.name), hand_geoms, collision_geom_names


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
        xml_path = build_h1_sharpa_visual_xml(
            base_xml,
            args.sharpa_root,
            parse_vec(args.sharpa_mount_pos, 3, [0.055, 0.0, 0.0]),
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
        )
        temp_paths.append(xml_path)

    xml_path, hand_geom_names, object_geom_names = build_dynamic_object_xml(
        xml_path,
        object_motion,
        args.object_floor_collision,
    )
    temp_paths.append(xml_path)
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)
    if external_motion is not None:
        hide_original_h1_hands(model)
    qpos_map = joint_qpos_map(model)

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

    frame_dt = 1.0 / float(fps)
    substeps = max(1, int(args.substeps))
    model.opt.timestep = frame_dt / substeps
    release_frame = max(0, int(args.release_frame))
    object_count = len(object_motion["position"])

    initial_index = camera_sample_index(
        min(release_frame, len(root_pos) - 1),
        len(root_pos),
        object_count,
    )
    if not object_motion["valid"][initial_index]:
        raise ValueError(
            f"object pose is invalid at release_frame={release_frame} "
            f"(object index {initial_index})"
        )
    data.qpos[object_qpos_adr : object_qpos_adr + 3] = object_motion[
        "position"
    ][initial_index]
    data.qpos[object_qpos_adr + 3 : object_qpos_adr + 7] = object_motion[
        "quat_wxyz"
    ][initial_index]

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

            if frame_idx < release_frame:
                tracked_idx = camera_sample_index(
                    frame_idx,
                    len(root_pos),
                    object_count,
                )
                data.qpos[object_qpos_adr : object_qpos_adr + 3] = (
                    object_motion["position"][tracked_idx]
                )
                data.qpos[object_qpos_adr + 3 : object_qpos_adr + 7] = (
                    object_motion["quat_wxyz"][tracked_idx]
                )
                data.qvel[object_dof_slice] = 0.0

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
            for substep in range(substeps):
                desired = current_target.copy()
                mj.mj_integratePos(
                    model,
                    desired,
                    velocity,
                    model.opt.timestep * substep,
                )
                desired[object_slice] = object_qpos
                data.qpos[:] = desired
                data.qvel[:] = velocity
                data.qvel[object_dof_slice] = object_qvel
                mj.mj_step(model, data)
                object_qpos = data.qpos[object_slice].copy()
                object_qvel = data.qvel[object_dof_slice].copy()
                for contact_index in range(data.ncon):
                    contact = data.contact[contact_index]
                    pair = {int(contact.geom1), int(contact.geom2)}
                    if pair & object_geom_ids and pair & hand_geom_ids:
                        frame_contacts += 1
                        force = np.zeros(6, dtype=np.float64)
                        mj.mj_contactForce(model, data, contact_index, force)
                        frame_force = max(frame_force, float(np.linalg.norm(force[:3])))

            positions.append(object_qpos[:3].copy())
            quaternions.append(object_qpos[3:7].copy())
            contact_counts.append(frame_contacts)
            max_contact_forces.append(frame_force)

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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        schema_version=np.asarray(1, dtype=np.int32),
        source=np.asarray("mujoco_free_object_contact_simulation"),
        fps=np.asarray(fps, dtype=np.float32),
        position=positions,
        quat_wxyz=quaternions,
        contact_count=contact_counts,
        max_normal_force=max_contact_forces,
        release_frame=np.asarray(release_frame, dtype=np.int32),
        visual_mesh_path=np.asarray(str(object_motion["visual_mesh_path"])),
        collision_mesh_paths=np.asarray(
            [str(path) for path in object_motion["collision_mesh_paths"]],
            dtype=str,
        ),
    )

    release_z = float(positions[min(release_frame, len(positions) - 1), 2])
    lift = positions[:, 2] - release_z
    lifted = lift >= args.lift_threshold
    contact_after_release = contact_counts.copy()
    contact_after_release[:release_frame] = 0
    report = {
        "output": str(args.output),
        "video": str(args.video_path) if args.video_path else None,
        "frames": len(positions),
        "release_frame": release_frame,
        "collision_parts": len(object_geom_names),
        "hand_collision_geoms": len(hand_geom_names),
        "contact_frames_after_release": int(
            np.count_nonzero(contact_after_release)
        ),
        "max_contact_force_n": float(max_contact_forces.max(initial=0.0)),
        "max_lift_m": float(lift.max(initial=0.0)),
        "lift_threshold_m": float(args.lift_threshold),
        "dynamic_lift_success": bool(
            np.any(lifted & (contact_after_release > 0))
        ),
        "interpretation": (
            "Success means a free object was lifted while hand-object contact "
            "was active; the robot itself is still kinematically replayed."
        ),
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate a free object against a kinematically replayed GMR robot."
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
    parser.add_argument("--sharpa_mount_quat", default="")
    parser.add_argument("--sharpa_left_mount_quat", default="")
    parser.add_argument("--sharpa_right_mount_quat", default="")
    parser.add_argument("--release_frame", type=int, default=0)
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
