#!/usr/bin/env python3
"""Build a SONIC-native G1 scene with the validated GMR chair and contacts.

SONIC's released demonstration XML uses visual-only robot meshes and normally
relies on an elastic support band.  The released policy, however, is tied to
that XML's mass and inertial model.  This tool preserves the SONIC robot XML,
then adds only the official G1 collision geoms already validated in the GMR
task and the six static chair bodies from that task.  It also makes the stock
floor collide with the robot collision category.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco


COLLISION_PREFIX = "gmr_official_collision_"
COLLISION_MESH_PREFIX = "gmr_official_collision_mesh_"
CHAIR_BODY_NAMES = (
    "seat_support",
    "leg_front_left",
    "leg_front_right",
    "leg_back_left",
    "leg_back_right",
    "backrest",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sonic-robot-xml", type=Path, required=True)
    parser.add_argument("--sonic-scene-xml", type=Path, required=True)
    parser.add_argument("--gmr-task-xml", type=Path, required=True)
    parser.add_argument("--output-robot-xml", type=Path, required=True)
    parser.add_argument("--output-scene-xml", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def direct_child(parent: ET.Element, tag: str) -> ET.Element:
    child = parent.find(tag)
    if child is None:
        raise ValueError(f"XML has no direct <{tag}> element")
    return child


def body_map(root: ET.Element) -> dict[str, ET.Element]:
    result: dict[str, ET.Element] = {}
    for body in root.findall(".//body"):
        name = body.get("name")
        if name:
            if name in result:
                raise ValueError(f"duplicate body name {name!r}")
            result[name] = body
    return result


def main() -> None:
    args = parse_args()
    for path, label in (
        (args.sonic_robot_xml, "SONIC robot XML"),
        (args.sonic_scene_xml, "SONIC scene XML"),
        (args.gmr_task_xml, "GMR task XML"),
    ):
        require_file(path, label)
    for path in (args.output_robot_xml, args.output_scene_xml):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to replace {path}; use --overwrite")

    sonic_robot_tree = ET.parse(args.sonic_robot_xml)
    sonic_scene_tree = ET.parse(args.sonic_scene_xml)
    gmr_task_tree = ET.parse(args.gmr_task_xml)
    sonic_robot_root = sonic_robot_tree.getroot()
    sonic_scene_root = sonic_scene_tree.getroot()
    gmr_root = gmr_task_tree.getroot()

    sonic_bodies = body_map(sonic_robot_root)
    gmr_bodies = body_map(gmr_root)
    sonic_assets = direct_child(sonic_robot_root, "asset")
    gmr_assets = direct_child(gmr_root, "asset")
    existing_assets = {asset.get("name") for asset in sonic_assets if asset.get("name")}

    copied_meshes = 0
    for asset in gmr_assets.findall("mesh"):
        name = asset.get("name", "")
        if name.startswith(COLLISION_MESH_PREFIX):
            if name in existing_assets:
                raise ValueError(f"SONIC robot already has collision mesh {name!r}")
            sonic_assets.append(copy.deepcopy(asset))
            existing_assets.add(name)
            copied_meshes += 1

    copied_collision_geoms = 0
    missing_destination_bodies: list[str] = []
    for body_name, gmr_body in gmr_bodies.items():
        collision_geoms = [
            geom for geom in gmr_body.findall("geom")
            if geom.get("name", "").startswith(COLLISION_PREFIX)
        ]
        if not collision_geoms:
            continue
        sonic_body = sonic_bodies.get(body_name)
        if sonic_body is None:
            missing_destination_bodies.append(body_name)
            continue
        for geom in collision_geoms:
            copied = copy.deepcopy(geom)
            # GMR's validated contract: robot collision category 2, static
            # floor/chair affinity 2.  The robot itself need not collide with
            # its own links for this single-contact task.
            copied.set("contype", "2")
            copied.set("conaffinity", "0")
            sonic_body.append(copied)
            copied_collision_geoms += 1
    if missing_destination_bodies:
        raise ValueError(f"SONIC robot misses collision bodies: {missing_destination_bodies}")
    if copied_collision_geoms == 0 or copied_meshes == 0:
        raise ValueError("no official G1 collision geometry was copied")

    args.output_robot_xml.parent.mkdir(parents=True, exist_ok=True)
    sonic_robot_tree.write(args.output_robot_xml, encoding="utf-8", xml_declaration=True)

    includes = sonic_scene_root.findall("include")
    if len(includes) != 1:
        raise ValueError(f"expected one SONIC robot include, found {len(includes)}")
    include_path = includes[0].get("file")
    if include_path != args.sonic_robot_xml.name:
        raise ValueError(
            f"expected scene include {args.sonic_robot_xml.name!r}, got {include_path!r}"
        )
    includes[0].set("file", args.output_robot_xml.name)

    scene_worldbody = direct_child(sonic_scene_root, "worldbody")
    floor = next((geom for geom in scene_worldbody.findall("geom") if geom.get("name") == "floor"), None)
    if floor is None:
        raise ValueError("SONIC scene has no named floor geom")
    floor.set("contype", "1")
    floor.set("conaffinity", "2")

    gmr_worldbody = direct_child(gmr_root, "worldbody")
    static_source = {
        body.get("name"): body for body in gmr_worldbody.findall("body") if body.get("name")
    }
    existing_scene_bodies = {body.get("name") for body in scene_worldbody.findall("body")}
    for name in CHAIR_BODY_NAMES:
        chair_body = static_source.get(name)
        if chair_body is None:
            raise ValueError(f"GMR task has no static chair body {name!r}")
        if name in existing_scene_bodies:
            raise ValueError(f"SONIC scene already has body {name!r}")
        scene_worldbody.append(copy.deepcopy(chair_body))

    args.output_scene_xml.parent.mkdir(parents=True, exist_ok=True)
    sonic_scene_tree.write(args.output_scene_xml, encoding="utf-8", xml_declaration=True)

    model = mujoco.MjModel.from_xml_path(str(args.output_scene_xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0 or model.geom_contype[floor_id] != 1 or model.geom_conaffinity[floor_id] != 2:
        raise RuntimeError("compiled scene did not preserve the floor collision contract")
    for name in CHAIR_BODY_NAMES:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0:
            raise RuntimeError(f"compiled scene misses chair body {name!r}")
    official_collision_count = sum(
        1 for index in range(model.ngeom)
        if model.geom_contype[index] == 2 and model.geom_conaffinity[index] == 0
    )
    if official_collision_count < copied_collision_geoms:
        raise RuntimeError("compiled scene has fewer official collision geoms than requested")

    report = {
        "status": "written",
        "sonic_robot_source": str(args.sonic_robot_xml.resolve()),
        "sonic_scene_source": str(args.sonic_scene_xml.resolve()),
        "gmr_task_source": str(args.gmr_task_xml.resolve()),
        "output_robot": str(args.output_robot_xml.resolve()),
        "output_scene": str(args.output_scene_xml.resolve()),
        "source_sha256": {
            "sonic_robot": sha256(args.sonic_robot_xml),
            "sonic_scene": sha256(args.sonic_scene_xml),
            "gmr_task": sha256(args.gmr_task_xml),
        },
        "copied": {
            "official_collision_meshes": copied_meshes,
            "official_collision_geoms": copied_collision_geoms,
            "static_chair_bodies": list(CHAIR_BODY_NAMES),
        },
        "collision_contract": {
            "robot": {"contype": 2, "conaffinity": 0},
            "floor": {"contype": 1, "conaffinity": 2},
            "chair": {"contype": 1, "conaffinity": 2},
        },
        "mujoco": {"nq": model.nq, "nv": model.nv, "nu": model.nu, "ngeom": model.ngeom},
    }
    report_path = args.output_scene_xml.with_suffix(".physical_scene.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
