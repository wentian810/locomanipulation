#!/usr/bin/env python3
"""Attach the fitted semantic chair to HoloMotion without changing body count.

HoloMotion's frozen policy observes a fixed tensor of G1 body poses.  A chair
must therefore not be appended as an MJCF ``body``: that silently changes
``model.nbody`` and makes the policy input inconsistent.  This bridge flattens
the six fitted primitive chair geoms to world-frame geoms.  They remain static,
collidable and visible, but do not become robot/policy bodies.
"""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


CHAIR_GEOMS = frozenset({
    "seat_support_geom",
    "backrest_geom",
    "leg_front_left_geom",
    "leg_front_right_geom",
    "leg_back_left_geom",
    "leg_back_right_geom",
})


def _parse_vector(value: str | None, width: int, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return default
    parsed = tuple(float(item) for item in value.split())
    if len(parsed) != width:
        raise ValueError(f"expected {width} values, got {value!r}")
    return parsed


def _format_vector(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def _quat_multiply(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _rotate(quaternion: tuple[float, float, float, float], vector: tuple[float, float, float]) -> tuple[float, float, float]:
    pure_vector = (0.0, *vector)
    inverse = (quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3])
    rotated = _quat_multiply(_quat_multiply(quaternion, pure_vector), inverse)
    return rotated[1:]


def _compose_pose(
    parent_pos: tuple[float, float, float],
    parent_quat: tuple[float, float, float, float],
    local_pos: tuple[float, float, float],
    local_quat: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    rotated = _rotate(parent_quat, local_pos)
    return (
        tuple(parent_pos[index] + rotated[index] for index in range(3)),
        _quat_multiply(parent_quat, local_quat),
    )


def _check_supported_pose(element: ET.Element, label: str) -> None:
    unsupported = [
        attribute for attribute in ("euler", "axisangle", "xyaxes", "zaxis", "fromto")
        if element.get(attribute) is not None
    ]
    if unsupported:
        raise ValueError(
            f"{label} uses unsupported pose fields {unsupported}; refusing an ambiguous chair transform"
        )


def _iter_world_geoms(
    element: ET.Element,
    parent_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    parent_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
):
    """Yield (geom, world_pos, world_quat) recursively from an MJCF subtree."""
    for child in list(element):
        if child.tag == "body":
            _check_supported_pose(child, f"body {child.get('name')!r}")
            body_pos = _parse_vector(child.get("pos"), 3, (0.0, 0.0, 0.0))
            body_quat = _parse_vector(child.get("quat"), 4, (1.0, 0.0, 0.0, 0.0))
            world_pos, world_quat = _compose_pose(parent_pos, parent_quat, body_pos, body_quat)
            yield from _iter_world_geoms(child, world_pos, world_quat)
        elif child.tag == "geom":
            _check_supported_pose(child, f"geom {child.get('name')!r}")
            geom_pos = _parse_vector(child.get("pos"), 3, (0.0, 0.0, 0.0))
            geom_quat = _parse_vector(child.get("quat"), 4, (1.0, 0.0, 0.0, 0.0))
            world_pos, world_quat = _compose_pose(parent_pos, parent_quat, geom_pos, geom_quat)
            yield child, world_pos, world_quat


def _absolutise_meshdir(root: ET.Element, source: Path) -> None:
    compiler = root.find("compiler")
    if compiler is not None and compiler.get("meshdir"):
        meshdir = Path(compiler.get("meshdir"))
        if not meshdir.is_absolute():
            compiler.set("meshdir", str((source.parent / meshdir).resolve()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holomotion-robot-xml", required=True, type=Path)
    parser.add_argument("--fitted-chair-scene-xml", required=True, type=Path)
    parser.add_argument("--output-xml", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.output_xml.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite an existing HoloMotion chair scene")

    robot_xml = args.holomotion_robot_xml.resolve()
    original_model = mujoco.MjModel.from_xml_path(str(robot_xml))
    robot_tree = ET.parse(robot_xml)
    robot_root = robot_tree.getroot()
    _absolutise_meshdir(robot_root, robot_xml)
    target_worldbody = robot_root.find("worldbody")
    if target_worldbody is None:
        raise ValueError("HoloMotion robot XML has no worldbody")

    source_xml = args.fitted_chair_scene_xml.resolve()
    source_worldbody = ET.parse(source_xml).getroot().find("worldbody")
    if source_worldbody is None:
        raise ValueError("fitted chair scene has no worldbody")
    imported: list[ET.Element] = []
    for geom, world_pos, world_quat in _iter_world_geoms(source_worldbody):
        if geom.get("name") not in CHAIR_GEOMS:
            continue
        if geom.get("type") == "mesh":
            raise ValueError("HoloMotion chair bridge accepts only semantic primitive chair geoms")
        clone = copy.deepcopy(geom)
        clone.set("pos", _format_vector(world_pos))
        clone.set("quat", _format_vector(world_quat))
        # The physical chair shares its exact primitive geometry with rendering.
        # Explicit category bits ensure it collides with the official G1 geoms.
        clone.set("contype", "1")
        clone.set("conaffinity", "1")
        imported.append(clone)
    found = {geom.get("name") for geom in imported}
    if found != CHAIR_GEOMS:
        raise ValueError(f"fitted scene chair mismatch; found={sorted(found)}")
    for geom in imported:
        target_worldbody.append(geom)

    if robot_root.find(".//geom[@name='floor']") is None:
        ET.SubElement(target_worldbody, "geom", {
            "name": "floor", "type": "plane", "size": "0 0 0.05",
            "rgba": "0.70 0.70 0.70 1", "contype": "1", "conaffinity": "1",
        })

    args.output_xml.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".xml", dir=args.output_xml.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        robot_tree.write(temporary, encoding="unicode")
        combined_model = mujoco.MjModel.from_xml_path(str(temporary))
        if combined_model.nbody != original_model.nbody:
            raise RuntimeError(
                f"chair bridge changed body count {original_model.nbody} -> {combined_model.nbody}"
            )
        geom_ids = {
            name: int(mujoco.mj_name2id(combined_model, mujoco.mjtObj.mjOBJ_GEOM, name))
            for name in sorted(CHAIR_GEOMS)
        }
        if any(identifier < 0 for identifier in geom_ids.values()):
            raise RuntimeError("compiled HoloMotion scene lost a semantic chair geom")
        temporary.replace(args.output_xml)
    finally:
        temporary.unlink(missing_ok=True)

    report = {
        "schema_version": 1,
        "purpose": "holomotion_frozen_tracker_with_static_semantic_chair",
        "status": "ready_for_frozen_policy_mj_step_tracking",
        "inputs": {
            "holomotion_robot_xml": str(robot_xml),
            "fitted_chair_scene_xml": str(source_xml),
        },
        "chair": {
            "geom_names": sorted(CHAIR_GEOMS),
            "geom_ids": geom_ids,
            "body_count_before": int(original_model.nbody),
            "body_count_after": int(combined_model.nbody),
            "insertion": "worldbody geoms only; no chair body added",
        },
        "model": {"nq": int(combined_model.nq), "nv": int(combined_model.nv), "nu": int(combined_model.nu)},
        "output_xml": str(args.output_xml.resolve()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
