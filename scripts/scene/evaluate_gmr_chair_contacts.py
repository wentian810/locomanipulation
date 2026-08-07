#!/usr/bin/env python3
"""Audit real G1-to-chair signed distances for a kinematic GMR motion.

This prevents a false fix: lowering the root because the torso appears high,
while a crossed leg is already intersecting the seat.  Every distance comes
from compiled MuJoCo collision geoms, never from a root-node proxy.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


# A shared robot category excludes robot self-collision while allowing every
# official collision geom to meet the imported floor/chair category.  The old
# one-bit-per-geom scheme was a workaround for proxies and cannot represent
# Unitree's 36 official collision elements.
ROBOT_SCENE_CONTACT_BIT = 2


def _format_vector(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.10g}" for value in values)


def _quat_multiply_wxyz(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Compose two MuJoCo wxyz quaternions without changing frame order."""
    first_w, first_x, first_y, first_z = np.asarray(first, dtype=np.float64)
    second_w, second_x, second_y, second_z = np.asarray(second, dtype=np.float64)
    return np.asarray((
        first_w * second_w - first_x * second_x - first_y * second_y - first_z * second_z,
        first_w * second_x + first_x * second_w + first_y * second_z - first_z * second_y,
        first_w * second_y - first_x * second_z + first_y * second_w + first_z * second_x,
        first_w * second_z + first_x * second_y - first_y * second_x + first_z * second_w,
    ), dtype=np.float64)


def _tight_proxy_spec(
    half_extents: np.ndarray,
    base_quat_wxyz: np.ndarray,
    strategy: str,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Return a contact proxy in the original mesh geom frame.

    MuJoCo's single mesh AABB leaves substantial empty corner volume around a
    diagonal shank or forearm.  A capsule is a tighter, still conservative
    primitive for only clearly elongated links.  The visual mesh remains
    unchanged and must still pass the independent visible-mesh gate.
    """
    if strategy == "legacy_aabb":
        return "box", half_extents, base_quat_wxyz
    if strategy != "hybrid_capsule":
        raise ValueError(f"unknown collision proxy strategy: {strategy}")
    longest_axis = int(np.argmax(half_extents))
    ordered = np.sort(half_extents)
    if ordered[-1] < 2.5 * ordered[-2]:
        return "box", half_extents, base_quat_wxyz
    transverse_axes = [axis for axis in range(3) if axis != longest_axis]
    radius = float(max(half_extents[axis] for axis in transverse_axes))
    half_length = float(half_extents[longest_axis] - radius)
    if half_length <= 1e-5:
        return "box", half_extents, base_quat_wxyz
    # MuJoCo capsules are aligned with local +Z.  Rotate their local Z axis
    # onto the elongated mesh AABB axis before composing with the mesh pose.
    half_sqrt = float(np.sqrt(0.5))
    z_to_axis = (
        np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
        if longest_axis == 2
        else np.asarray((half_sqrt, 0.0, half_sqrt, 0.0), dtype=np.float64)
        if longest_axis == 0
        else np.asarray((half_sqrt, -half_sqrt, 0.0, 0.0), dtype=np.float64)
    )
    return "capsule", np.asarray((radius, half_length), dtype=np.float64), _quat_multiply_wxyz(base_quat_wxyz, z_to_axis)


def _rpy_to_quat_wxyz(rpy: np.ndarray) -> np.ndarray:
    """Convert URDF fixed-axis roll/pitch/yaw to a MuJoCo wxyz quaternion."""
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    half_roll, half_pitch, half_yaw = roll / 2.0, pitch / 2.0, yaw / 2.0
    cr, sr = math.cos(half_roll), math.sin(half_roll)
    cp, sp = math.cos(half_pitch), math.sin(half_pitch)
    cy, sy = math.cos(half_yaw), math.sin(half_yaw)
    return np.asarray((
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
    ), dtype=np.float64)


def _parse_custom_cylinders(path: Path) -> list[tuple[str, np.ndarray, np.ndarray, float, float]]:
    """Read the hand-authored G1 collision cylinders without changing URDF data."""
    root = ET.parse(path).getroot()
    result: list[tuple[str, np.ndarray, np.ndarray, float, float]] = []
    for link in root.findall("link"):
        link_name = link.get("name")
        if not link_name:
            continue
        for collision in link.findall("collision"):
            cylinder = collision.find("geometry/cylinder")
            if cylinder is None:
                continue
            radius = float(cylinder.get("radius", "nan"))
            length = float(cylinder.get("length", "nan"))
            origin = collision.find("origin")
            position = _format_vector(np.zeros(3)) if origin is None else origin.get("xyz", "0 0 0")
            rpy = "0 0 0" if origin is None else origin.get("rpy", "0 0 0")
            xyz = np.asarray([float(value) for value in position.split()], dtype=np.float64)
            angles = np.asarray([float(value) for value in rpy.split()], dtype=np.float64)
            if xyz.shape != (3,) or angles.shape != (3,) or not np.isfinite([radius, length, *xyz, *angles]).all():
                raise ValueError(f"invalid custom collision on {link_name}")
            if radius <= 0.0 or length <= 0.0:
                raise ValueError(f"non-positive custom cylinder on {link_name}")
            result.append((link_name, xyz, _rpy_to_quat_wxyz(angles), radius, length))
    if not result:
        raise ValueError(f"no active cylinder collision geometry found in {path}")
    return result


def _parse_official_unitree_collisions(
    path: Path,
) -> list[tuple[str, str, dict[str, str], np.ndarray, np.ndarray]]:
    """Parse collision geometry from Unitree's revision-matched G1 URDF.

    The returned geometry stays link-local.  In particular, it is *not*
    fitted to this clip or to an observed chair contact.  That distinction is
    essential: geometry provenance belongs to the robot model, whereas scene
    placement belongs to the immutable VideoMimic coordinate contract.
    """
    root = ET.parse(path).getroot()
    result: list[tuple[str, str, dict[str, str], np.ndarray, np.ndarray]] = []
    for link in root.findall("link"):
        link_name = link.get("name")
        if not link_name:
            continue
        for collision in link.findall("collision"):
            origin = collision.find("origin")
            xyz_text = "0 0 0" if origin is None else origin.get("xyz", "0 0 0")
            rpy_text = "0 0 0" if origin is None else origin.get("rpy", "0 0 0")
            xyz = np.asarray([float(value) for value in xyz_text.split()], dtype=np.float64)
            rpy = np.asarray([float(value) for value in rpy_text.split()], dtype=np.float64)
            geometry = collision.find("geometry")
            if geometry is None or len(geometry) != 1:
                raise ValueError(f"invalid official collision geometry on {link_name}")
            shape = geometry[0]
            if shape.tag not in {"mesh", "sphere", "cylinder"}:
                raise ValueError(
                    f"unsupported official collision type {shape.tag!r} on {link_name}"
                )
            if xyz.shape != (3,) or rpy.shape != (3,) or not np.isfinite([*xyz, *rpy]).all():
                raise ValueError(f"invalid official collision pose on {link_name}")
            result.append((link_name, shape.tag, dict(shape.attrib), xyz, _rpy_to_quat_wxyz(rpy)))
    if not result:
        raise ValueError(f"official Unitree URDF has no collision geometry: {path}")
    return result


def _quat_from_rotation_matrix_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to MuJoCo's wxyz convention."""
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        quat = np.asarray((0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale,
                           (matrix[0, 2] - matrix[2, 0]) / scale,
                           (matrix[1, 0] - matrix[0, 1]) / scale))
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = 2.0 * math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            quat = np.asarray(((matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                               (matrix[0, 1] + matrix[1, 0]) / scale,
                               (matrix[0, 2] + matrix[2, 0]) / scale))
        elif axis == 1:
            scale = 2.0 * math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            quat = np.asarray(((matrix[0, 2] - matrix[2, 0]) / scale,
                               (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                               (matrix[1, 2] + matrix[2, 1]) / scale))
        else:
            scale = 2.0 * math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            quat = np.asarray(((matrix[1, 0] - matrix[0, 1]) / scale,
                               (matrix[0, 2] + matrix[2, 0]) / scale,
                               (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale))
    quat /= np.linalg.norm(quat)
    return quat


def _world_transform(data: mujoco.MjData, object_id: int, *, geom: bool) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if geom:
        transform[:3, :3] = data.geom_xmat[object_id].reshape(3, 3)
        transform[:3, 3] = data.geom_xpos[object_id]
    else:
        transform[:3, :3] = data.xmat[object_id].reshape(3, 3)
        transform[:3, 3] = data.xpos[object_id]
    return transform


def _transform_from_pos_quat_wxyz(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """Return the rigid transform represented by MuJoCo ``pos``/``quat``."""
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    rotation = np.asarray((
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    ), dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(position, dtype=np.float64)
    return transform


def _mesh_assets_by_name(official_mjcf: Path) -> dict[str, tuple[Path, str | None]]:
    root = ET.parse(official_mjcf).getroot()
    compiler = root.find("compiler")
    meshdir = "" if compiler is None else compiler.get("meshdir", "")
    result: dict[str, tuple[Path, str | None]] = {}
    for mesh in root.findall("./asset/mesh"):
        name, filename = mesh.get("name"), mesh.get("file")
        if not name or not filename:
            continue
        path = (official_mjcf.parent / meshdir / filename).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"official Unitree mesh is missing: {path}")
        result[name] = (path, mesh.get("scale"))
    return result


def _configure_official_unitree_mjcf_collisions(
    root: ET.Element,
    parent_by_child: dict[ET.Element, ET.Element],
    mesh_sources: list[tuple[ET.Element, ET.Element, str]],
    robot_xml: Path,
    official_collision_mjcf: Path | None,
) -> int:
    """Install registered collision geoms from Unitree's matching MJCF.

    The GMR and official models use the same joint tree but do not expose every
    fixed cosmetic link as a named GMR body.  We therefore register every
    official collision geom into the nearest common ancestor body using their
    compiled zero-pose transforms.  This is a robot-asset conversion, not a
    clip-specific scene fit, and it remains invariant under joint motion only
    after the two trees have been verified to share that ancestor frame.
    """
    if official_collision_mjcf is None or not official_collision_mjcf.is_file():
        raise FileNotFoundError(
            "official_unitree_mjcf collision strategy requires an existing "
            "--official-collision-mjcf"
        )
    body_by_name = {
        body.get("name"): body for body in root.findall(".//body") if body.get("name")
    }
    gmr_model = mujoco.MjModel.from_xml_path(str(robot_xml))
    official_model = mujoco.MjModel.from_xml_path(str(official_collision_mjcf))
    gmr_data, official_data = mujoco.MjData(gmr_model), mujoco.MjData(official_model)
    mujoco.mj_resetData(gmr_model, gmr_data)
    mujoco.mj_resetData(official_model, official_data)
    mujoco.mj_forward(gmr_model, gmr_data)
    mujoco.mj_forward(official_model, official_data)
    gmr_pelvis = mujoco.mj_name2id(gmr_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    official_pelvis = mujoco.mj_name2id(official_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if gmr_pelvis < 0 or official_pelvis < 0:
        raise ValueError("both G1 models must expose a pelvis body")
    official_to_gmr = _world_transform(gmr_data, gmr_pelvis, geom=False) @ np.linalg.inv(
        _world_transform(official_data, official_pelvis, geom=False)
    )
    mesh_assets = _mesh_assets_by_name(official_collision_mjcf)
    source_geoms = {geom for geom, _parent, _name in mesh_sources}
    # Disable every pre-existing robot collision source.  Retaining a native
    # primitive beside an official mesh silently turns the same physical link
    # into two different bodies, which was the source of the shoulder/seat
    # false positive in the previous audit.
    for geom in root.findall(".//geom"):
        parent = parent_by_child.get(geom)
        if geom in source_geoms or (parent is not None and parent.tag == "body"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")

    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(0, asset)
    injected = 0
    skipped_bodies: list[str] = []
    for official_geom in range(official_model.ngeom):
        if int(official_model.geom_contype[official_geom]) == 0:
            continue
        official_body = int(official_model.geom_bodyid[official_geom])
        target_name: str | None = None
        ancestor = official_body
        while ancestor > 0:
            candidate = mujoco.mj_id2name(
                official_model, mujoco.mjtObj.mjOBJ_BODY, ancestor
            )
            if candidate in body_by_name:
                target_name = candidate
                break
            ancestor = int(official_model.body_parentid[ancestor])
        if target_name is None:
            skipped_bodies.append(
                mujoco.mj_id2name(official_model, mujoco.mjtObj.mjOBJ_BODY, official_body) or ""
            )
            continue
        body = body_by_name[target_name]
        geom_type = int(official_model.geom_type[official_geom])
        local = np.linalg.inv(_world_transform(gmr_data, mujoco.mj_name2id(
            gmr_model, mujoco.mjtObj.mjOBJ_BODY, target_name), geom=False
        )) @ official_to_gmr @ _world_transform(official_data, official_geom, geom=True)
        # MuJoCo centres every mesh asset at compile time, and records that
        # asset-local correction in ``mesh_pos``/``mesh_quat``.  The compiled
        # geom transform above already includes it.  The new asset will apply
        # it again unless it is removed here; doing so was the source of the
        # 0.43 m false pose error in the first official-model conversion.
        if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh_id = int(official_model.geom_dataid[official_geom])
            asset_transform = _transform_from_pos_quat_wxyz(
                official_model.mesh_pos[mesh_id], official_model.mesh_quat[mesh_id],
            )
            local = local @ np.linalg.inv(asset_transform)
        attributes = {
            "name": f"gmr_official_collision_{official_geom}",
            "pos": _format_vector(local[:3, 3]),
            "quat": _format_vector(_quat_from_rotation_matrix_wxyz(local[:3, :3])),
            "contype": str(ROBOT_SCENE_CONTACT_BIT),
            "conaffinity": "0",
            "density": "0",
            "group": "3",
            "friction": "1 0.005 0.0001",
        }
        if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
            source_mesh = mujoco.mj_id2name(
                official_model, mujoco.mjtObj.mjOBJ_MESH,
                int(official_model.geom_dataid[official_geom]),
            )
            if source_mesh not in mesh_assets:
                raise ValueError(f"official collision mesh lacks an asset file: {source_mesh}")
            mesh_path, mesh_scale = mesh_assets[source_mesh]
            mesh_name = f"gmr_official_collision_mesh_{official_geom}"
            mesh_attributes = {"name": mesh_name, "file": mesh_path.as_posix()}
            if mesh_scale:
                mesh_attributes["scale"] = mesh_scale
            ET.SubElement(asset, "mesh", mesh_attributes)
            attributes.update({"type": "mesh", "mesh": mesh_name})
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            radius = float(official_model.geom_size[official_geom, 0])
            attributes.update({"type": "sphere", "size": f"{radius:.10g}"})
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            radius, half_length = official_model.geom_size[official_geom, :2]
            attributes.update({"type": "cylinder", "size": f"{radius:.10g} {half_length:.10g}"})
        else:
            raise ValueError(f"unsupported compiled official geom type {geom_type}")
        ET.SubElement(body, "geom", attributes)
        injected += 1
    if injected < 20:
        raise ValueError(
            "too few official collision geoms match GMR bodies; "
            f"injected={injected}, skipped={sorted(set(skipped_bodies))}"
        )
    # ``1`` means one shared robot contact category, not one injected geom.
    return 1


def _configure_custom_urdf_cylinders(
    root: ET.Element,
    parent_by_child: dict[ET.Element, ET.Element],
    mesh_sources: list[tuple[ET.Element, ET.Element, str]],
    robot_xml: Path,
    custom_collision_urdf: Path | None,
) -> int:
    """Use the GMR-provided hand-authored G1 cylinders as scene contacts."""
    path = custom_collision_urdf or robot_xml.with_name("g1_custom_collision_29dof.urdf")
    if not path.is_file():
        raise FileNotFoundError(f"custom G1 collision URDF is missing: {path}")
    body_by_name = {
        body.get("name"): body for body in root.findall(".//body") if body.get("name")
    }
    cylinders = _parse_custom_cylinders(path)
    missing_bodies = sorted({body for body, *_ in cylinders} - set(body_by_name))
    if missing_bodies:
        raise ValueError(f"custom collision links are absent from GMR MJCF: {missing_bodies}")

    # The mesh must remain a visual surface only.  Retain existing primitive
    # G1 contacts (notably feet) but give every robot collision geom an
    # exclusive bit so robot parts cannot collide with one another.
    source_geoms = {geom for geom, _parent, _name in mesh_sources}
    for geom in source_geoms:
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    native_geoms = [
        geom for geom in root.findall(".//geom")
        if geom not in source_geoms
        and geom.get("contype") != "0"
        and parent_by_child.get(geom) is not None
        and parent_by_child[geom].tag == "body"
    ]
    contact_geoms = list(native_geoms)
    for index, (body_name, position, quat, radius, length) in enumerate(cylinders):
        geom = ET.SubElement(body_by_name[body_name], "geom", {
            "name": f"gmr_custom_collision_{index}",
            "type": "cylinder",
            "pos": _format_vector(position),
            "quat": _format_vector(quat),
            "size": _format_vector(np.asarray((radius, length / 2.0))),
            "density": "0",
            "group": "3",
        })
        contact_geoms.append(geom)
    if len(contact_geoms) > 30:
        raise ValueError("custom collision bitmask supports at most 30 robot geoms")
    native_geom_set = set(native_geoms)
    for index, geom in enumerate(contact_geoms):
        if geom in native_geom_set:
            # Existing names may be used by visual tooling but not by the
            # physics task.  Prefix the collision copy deterministically so
            # the contact residual can distinguish it from a scene geom.
            geom.set("name", f"gmr_native_collision_{index}")
        geom.set("contype", str(1 << (index + 1)))
        geom.set("conaffinity", "1")
    return len(contact_geoms)


def _add_conservative_robot_collision_proxies(
    tree: ET.ElementTree,
    robot_xml: Path,
    collision_proxy_strategy: str = "legacy_aabb",
    custom_collision_urdf: Path | None = None,
    official_collision_mjcf: Path | None = None,
) -> int:
    """Configure GMR's contact geometry without changing its visual mesh.

    The GMR asset is built for kinematic rendering.  Its mesh geoms report
    signed distances but do not generate MuJoCo contacts against the imported
    chair boxes.  We compile a named temporary copy to recover each mesh AABB,
    disable the source mesh for collision, and add a body-local contact proxy.
    ``legacy_aabb`` reproduces historical boxes; ``hybrid_capsule`` replaces
    only clearly elongated AABBs with aligned capsules.  ``visual_mesh`` keeps
    the compiled visual mesh as the contact mesh, which is required when a
    primitive proxy demonstrably stops the robot before its rendered surface
    reaches the chair.  This temporary physics MJCF never changes the visual
    GMR asset.
    """
    root = tree.getroot()
    parent_by_child = {child: parent for parent in root.iter() for child in parent}
    sources: list[tuple[ET.Element, ET.Element, str]] = []
    for index, geom in enumerate(root.findall(".//geom")):
        if geom.get("type") != "mesh" or geom.get("contype") == "0":
            continue
        parent = parent_by_child.get(geom)
        if parent is None or parent.tag != "body":
            continue
        name = f"gmr_collision_source_{index}"
        geom.set("name", name)
        sources.append((geom, parent, name))
    if not sources:
        raise RuntimeError("GMR MJCF has no mesh collision sources to proxy")

    if collision_proxy_strategy == "custom_urdf_cylinders":
        return _configure_custom_urdf_cylinders(
            root, parent_by_child, sources, robot_xml, custom_collision_urdf,
        )
    if collision_proxy_strategy == "official_unitree_mjcf":
        return _configure_official_unitree_mjcf_collisions(
            root, parent_by_child, sources, robot_xml, official_collision_mjcf,
        )

    if collision_proxy_strategy == "visual_mesh":
        # MuJoCo computes contacts against the mesh convex hull.  Give each
        # robot mesh a unique bit and let only scene geoms accept those bits,
        # exactly as the primitive profile does.  This suppresses robot
        # self-collision while preserving visual-mesh contact fidelity.
        for source_index, (geom, _parent, _name) in enumerate(sources):
            geom.set("contype", str(1 << (source_index + 1)))
            geom.set("conaffinity", "1")
            # G1 bodies have explicit inertial properties.  Do not alter their
            # mass when the visual mesh becomes a collision geom.
            geom.set("density", "0")
        return len(sources)
    if collision_proxy_strategy not in {"legacy_aabb", "hybrid_capsule"}:
        raise ValueError(f"unknown collision proxy strategy: {collision_proxy_strategy}")

    handle = tempfile.NamedTemporaryFile(
        prefix="gmr_collision_proxy_source_",
        suffix=".xml",
        dir=str(robot_xml.parent),
        delete=False,
    )
    handle.close()
    source_path = Path(handle.name)
    try:
        tree.write(source_path, encoding="unicode")
        model = mujoco.MjModel.from_xml_path(str(source_path))
    finally:
        source_path.unlink(missing_ok=True)

    for index, (geom, parent, name) in enumerate(sources):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            raise RuntimeError(f"named mesh source missing after compilation: {name}")
        half_extents = np.asarray(model.geom_size[geom_id], dtype=np.float64)
        if np.any(half_extents <= 1e-8):
            raise RuntimeError(f"invalid mesh AABB for {name}: {half_extents}")
        proxy_type, proxy_size, proxy_quat = _tight_proxy_spec(
            half_extents, np.asarray(model.geom_quat[geom_id], dtype=np.float64), collision_proxy_strategy
        )
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
        ET.SubElement(
            parent,
            "geom",
            {
                "name": f"gmr_physics_proxy_{index}",
                "type": proxy_type,
                "pos": _format_vector(model.geom_pos[geom_id]),
                "quat": _format_vector(proxy_quat),
                "size": _format_vector(proxy_size),
                # Each proxy receives an exclusive contact bit.  Imported
                # scene geoms accept all of those bits; robot proxies accept
                # only the scene bit (1), which excludes robot self-collision.
                "contype": str(1 << (index + 1)),
                "conaffinity": "1",
                "density": "0",
                "friction": "1 0.005 0.0001",
                "group": "3",
            },
        )
    return len(sources)


def _combine_mjcf(
    robot_xml: Path,
    scene_xml: Path,
    collision_proxy_strategy: str = "legacy_aabb",
    custom_collision_urdf: Path | None = None,
    official_collision_mjcf: Path | None = None,
) -> Path:
    tree = ET.parse(robot_xml)
    proxy_count = _add_conservative_robot_collision_proxies(
        tree, robot_xml, collision_proxy_strategy, custom_collision_urdf,
        official_collision_mjcf,
    )
    if proxy_count > 30:
        raise ValueError("collision-proxy bitmask supports at most 30 robot geoms")
    scene_proxy_mask = sum(1 << (index + 1) for index in range(proxy_count))
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"worldbody missing from {robot_xml}")
    scene_root = ET.parse(scene_xml).getroot()
    scene_worldbody = scene_root.find("worldbody")
    if scene_worldbody is None:
        raise ValueError(f"worldbody missing from {scene_xml}")

    asset = root.find("asset")
    scene_asset = scene_root.find("asset")
    if scene_asset is not None:
        if asset is None:
            asset = ET.Element("asset")
            root.insert(0, asset)
        for child in scene_asset:
            imported = copy.deepcopy(child)
            if imported.tag == "mesh" and imported.get("file"):
                mesh_path = Path(imported.attrib["file"])
                if not mesh_path.is_absolute():
                    mesh_path = scene_xml.parent / mesh_path
                imported.set("file", mesh_path.resolve().as_posix())
            asset.append(imported)
    for child in scene_worldbody:
        imported_body = copy.deepcopy(child)
        for geom in imported_body.findall(".//geom"):
            if geom.get("contype") != "0":
                # Required by MuJoCo's two-way contype/conaffinity filter.
                geom.set("contype", "1")
                geom.set("conaffinity", str(scene_proxy_mask))
        worldbody.append(imported_body)

    handle = tempfile.NamedTemporaryFile(
        prefix="gmr_chair_audit_", suffix=".xml", dir=str(robot_xml.parent),
        delete=False,
    )
    handle.close()
    tree.write(handle.name, encoding="unicode")
    return Path(handle.name)


def _load_motion(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    for key in ("root_pos", "root_rot", "dof_pos"):
        if key not in payload:
            raise ValueError(f"{path} is missing {key}")
    return payload


def _assign_qpos(data: mujoco.MjData, motion: dict[str, Any], frame: int) -> None:
    data.qpos[:3] = np.asarray(motion["root_pos"], dtype=np.float64)[frame]
    root_rot = np.asarray(motion["root_rot"], dtype=np.float64)[frame]
    data.qpos[3:7] = root_rot[[3, 0, 1, 2]]
    data.qpos[7:] = np.asarray(motion["dof_pos"], dtype=np.float64)[frame]


def _is_robot_descendant(model: mujoco.MjModel, body_id: int, pelvis_id: int) -> bool:
    body_id = int(body_id)
    while body_id >= 0:
        if body_id == pelvis_id:
            return True
        parent_id = int(model.body_parentid[body_id])
        if parent_id == body_id:
            break
        body_id = parent_id
    return False


def _robot_collision_geoms(model: mujoco.MjModel) -> list[int]:
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id < 0:
        raise ValueError("pelvis body missing from robot MJCF")
    proxies: list[int] = []
    fallback: list[int] = []
    for geom_id in range(model.ngeom):
        if int(model.geom_contype[geom_id]) == 0:
            continue
        if not _is_robot_descendant(model, int(model.geom_bodyid[geom_id]), pelvis_id):
            continue
        fallback.append(geom_id)
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name.startswith("gmr_physics_proxy_"):
            proxies.append(geom_id)
    geoms = proxies or fallback
    if not geoms:
        raise RuntimeError("no collidable robot geoms found under pelvis")
    return geoms


_SEAT_CONTACT_BODIES = {
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
}


def _robot_chair_contact_geoms(model: mujoco.MjModel) -> list[int]:
    """Return lower-body proxy geoms relevant to a seat support contact."""
    geoms = [
        geom_id
        for geom_id in _robot_collision_geoms(model)
        if _body_name(model, geom_id) in _SEAT_CONTACT_BODIES
    ]
    if not geoms:
        raise RuntimeError("no lower-body robot collision proxies found")
    return geoms


def _body_name(model: mujoco.MjModel, geom_id: int) -> str:
    name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])
    )
    return str(name or f"body_{int(model.geom_bodyid[geom_id])}")


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.quantile(values, 0.5)),
        "p05": float(np.quantile(values, 0.05)),
        "minimum": float(np.min(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit kinematic GMR chair contact using MuJoCo signed distances."
    )
    parser.add_argument("--robot-motion", required=True, type=Path)
    parser.add_argument("--scene-mujoco-xml", required=True, type=Path)
    parser.add_argument("--robot-xml", required=True, type=Path)
    parser.add_argument("--contact-anchors", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--arrays", required=True, type=Path)
    parser.add_argument("--seat-geom", default="seat_support_geom")
    parser.add_argument("--clearance-m", type=float, default=0.003)
    parser.add_argument(
        "--collision-proxy-strategy",
        choices=(
            "legacy_aabb", "hybrid_capsule", "visual_mesh",
            "custom_urdf_cylinders", "official_unitree_mjcf",
        ),
        default="legacy_aabb",
    )
    parser.add_argument("--custom-collision-urdf", type=Path, default=None)
    parser.add_argument("--official-collision-mjcf", type=Path, default=None)
    parser.add_argument(
        "--support-bodies",
        default=(
            "pelvis,left_hip_pitch_link,right_hip_pitch_link,"
            "left_hip_roll_link,right_hip_roll_link"
        ),
    )
    args = parser.parse_args()
    if args.clearance_m < 0.0:
        raise ValueError("clearance must be non-negative")

    motion = _load_motion(args.robot_motion)
    root_pos = np.asarray(motion["root_pos"])
    root_rot = np.asarray(motion["root_rot"])
    dof_pos = np.asarray(motion["dof_pos"])
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError("root_pos must be [T,3]")
    if root_rot.shape != (len(root_pos), 4) or dof_pos.shape[0] != len(root_pos):
        raise ValueError("robot motion frame dimensions disagree")
    anchors = np.load(args.contact_anchors)
    if "sit_mask" not in anchors:
        raise ValueError(f"{args.contact_anchors} is missing sit_mask")
    sit_mask = np.asarray(anchors["sit_mask"], dtype=bool)
    if sit_mask.shape != (len(root_pos),):
        raise ValueError("contact anchors do not match robot motion")
    sit_indices = np.flatnonzero(sit_mask)
    if not len(sit_indices):
        raise RuntimeError("no source-SMPL sit frames to audit")

    combined_xml = _combine_mjcf(
        args.robot_xml,
        args.scene_mujoco_xml,
        collision_proxy_strategy=args.collision_proxy_strategy,
        custom_collision_urdf=args.custom_collision_urdf,
        official_collision_mjcf=args.official_collision_mjcf,
    )
    try:
        model = mujoco.MjModel.from_xml_path(str(combined_xml))
        data = mujoco.MjData(model)
        if model.nq != 7 + dof_pos.shape[1]:
            raise ValueError(
                f"motion has {dof_pos.shape[1]} DOFs, MJCF needs {model.nq - 7}"
            )
        seat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, args.seat_geom)
        if seat_id < 0:
            raise ValueError(f"seat geom {args.seat_geom!r} not found")
        robot_geoms = _robot_chair_contact_geoms(model)
        geoms_by_body: dict[str, list[int]] = {}
        for geom_id in robot_geoms:
            geoms_by_body.setdefault(_body_name(model, geom_id), []).append(geom_id)
        support_names = [
            item.strip() for item in args.support_bodies.split(",") if item.strip()
        ]
        support_names = [name for name in support_names if name in geoms_by_body]
        if not support_names:
            raise ValueError("none of --support-bodies have collidable geoms")

        frames = len(root_pos)
        min_distance = np.full(frames, np.nan, dtype=np.float64)
        worst_body = np.full(frames, "", dtype="U64")
        distances_by_body = {
            name: np.full(frames, np.nan, dtype=np.float64) for name in support_names
        }
        for frame in sit_indices:
            _assign_qpos(data, motion, int(frame))
            mujoco.mj_forward(model, data)
            best_distance, best_body = np.inf, ""
            for geom_id in robot_geoms:
                distance = float(
                    mujoco.mj_geomDistance(model, data, geom_id, int(seat_id), 2.0, None)
                )
                body = _body_name(model, geom_id)
                if distance < best_distance:
                    best_distance, best_body = distance, body
                if body in distances_by_body:
                    previous = distances_by_body[body][frame]
                    distances_by_body[body][frame] = (
                        distance if np.isnan(previous) else min(previous, distance)
                    )
            min_distance[frame] = best_distance
            worst_body[frame] = best_body

        candidates: list[dict[str, Any]] = []
        for body, distances in distances_by_body.items():
            values = distances[sit_indices]
            candidates.append(
                {
                    "body": body,
                    **_summary(values),
                    "penetration_frame_ratio": float(np.mean(values < -1e-5)),
                }
            )
        safe_candidates = [
            candidate for candidate in candidates
            if candidate["penetration_frame_ratio"] <= 0.05
        ]
        selected = sorted(
            safe_candidates or candidates,
            key=lambda item: (item["penetration_frame_ratio"], item["median"]),
        )[0]
        support_body = str(selected["body"])
        support_distance = distances_by_body[support_body]
        all_values = min_distance[sit_indices]
        support_values = support_distance[sit_indices]
        penetrates = bool(np.any(all_values < args.clearance_m - 1e-5))
        report = {
            "schema_version": 1,
            "purpose": "gmr_kinematic_chair_contact_audit",
            "status": "unsafe_for_root_z_only" if penetrates else "root_z_candidate_available",
            "robot_motion": str(args.robot_motion),
            "scene_mujoco_xml": str(args.scene_mujoco_xml),
            "collision_proxy_strategy": args.collision_proxy_strategy,
            "seat_geom": args.seat_geom,
            "sit_frames": int(len(sit_indices)),
            "sit_frame_range": [int(sit_indices[0]), int(sit_indices[-1])],
            "target_clearance_m": args.clearance_m,
            "min_robot_to_seat_distance_m": {
                **_summary(all_values),
                "penetration_frame_ratio": float(np.mean(all_values < -1e-5)),
                "worst_body_counts": {
                    body: int(np.sum(worst_body[sit_indices] == body))
                    for body in sorted(set(worst_body[sit_indices]))
                    if body
                },
            },
            "support_candidates": candidates,
            "selected_support_body": support_body,
            "selected_support_distance_m": _summary(support_values),
            "naive_root_lowering_m": float(
                np.median(support_values - args.clearance_m)
            ),
            "decision": (
                "refuse_global_root_z_shift_due_to_existing_seat_penetration"
                if penetrates
                else "root_z_shift_may_be_tested_after_dynamic_validation"
            ),
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.arrays.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        np.savez_compressed(
            args.arrays,
            sit_mask=sit_mask,
            min_robot_to_seat_distance_m=min_distance,
            worst_body=worst_body,
            selected_support_distance_m=support_distance,
            selected_support_body=np.asarray(support_body),
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        combined_xml.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
