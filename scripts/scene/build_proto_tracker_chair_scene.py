#!/usr/bin/env python3
"""Combine ProtoMotions' G1 MJCF with the already-fitted semantic chair only.

The tracker keeps its own G1 collision model and pretrained-policy body order.
This bridge imports exactly the six rendered chair primitives from the fitted
scene, so visual and physical chair geometry remain identical.
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


def _absolutize_meshdir(root: ET.Element, source_xml: Path) -> None:
    compiler = root.find("compiler")
    if compiler is None:
        return
    meshdir = compiler.get("meshdir")
    if meshdir and not Path(meshdir).is_absolute():
        compiler.set("meshdir", str((source_xml.parent / meshdir).resolve()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proto-robot-xml", required=True, type=Path)
    parser.add_argument("--fitted-chair-scene-xml", required=True, type=Path)
    parser.add_argument("--output-xml", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.output_xml.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite an existing tracker scene artifact")

    tree = ET.parse(args.proto_robot_xml)
    root = tree.getroot()
    _absolutize_meshdir(root, args.proto_robot_xml)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("ProtoMotions G1 model has no worldbody")
    chair_root = ET.parse(args.fitted_chair_scene_xml).getroot()
    chair_worldbody = chair_root.find("worldbody")
    if chair_worldbody is None:
        raise ValueError("fitted chair scene has no worldbody")

    imported: list[ET.Element] = []
    found: set[str] = set()
    for child in list(chair_worldbody):
        geoms = [geom for geom in child.findall(".//geom") if geom.get("name") in CHAIR_GEOMS]
        if not geoms:
            continue
        if any(geom.get("type") == "mesh" for geom in geoms):
            raise ValueError("semantic chair bridge supports primitive visual/collision geoms only")
        clone = copy.deepcopy(child)
        for geom in clone.findall(".//geom"):
            name = geom.get("name")
            if name in CHAIR_GEOMS:
                found.add(name)
                # ProtoMotions collision geoms retain MuJoCo's default bit 1.
                # The chair uses that same bit in both directions.
                geom.set("contype", "1")
                geom.set("conaffinity", "1")
        imported.append(clone)
    if found != CHAIR_GEOMS:
        raise ValueError(f"fitted scene chair mismatch; found={sorted(found)}")
    for child in imported:
        worldbody.append(child)

    # The ProtoMotions compatibility MJCF declares explicit foot--floor
    # contact pairs but leaves the plane to its runtime loader.  This bridge
    # validates the combined XML before that loader runs, so retain the same
    # canonical plane here.  It is deliberately independent of the chair:
    # the six chair primitives above remain the only chair geometry.
    if root.find(".//geom[@name='floor']") is None:
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": "floor",
                "type": "plane",
                "size": "0 0 0.05",
                "rgba": "0.7 0.7 0.7 1",
                "contype": "1",
                "conaffinity": "1",
            },
        )

    args.output_xml.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".xml", dir=args.output_xml.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        tree.write(temporary, encoding="unicode")
        model = mujoco.MjModel.from_xml_path(str(temporary))
        chair_geom_ids = {
            name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in sorted(CHAIR_GEOMS)
        }
        if any(identifier < 0 for identifier in chair_geom_ids.values()):
            raise RuntimeError("compiled ProtoMotions scene lost a semantic chair geom")
        temporary.replace(args.output_xml)
    finally:
        temporary.unlink(missing_ok=True)
    report = {
        "schema_version": 1,
        "purpose": "protomotions_tracker_with_fitted_semantic_chair",
        "status": "ready_for_pretrained_tracker_smoke_only",
        "inputs": {
            "proto_robot_xml": str(args.proto_robot_xml),
            "fitted_chair_scene_xml": str(args.fitted_chair_scene_xml),
        },
        "chair": {
            "visual_and_collision_geometry": "same_six_primitives",
            "geom_names": sorted(CHAIR_GEOMS),
            "geom_ids": chair_geom_ids,
        },
        "model": {"nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu), "nbody": int(model.nbody)},
        "output_xml": str(args.output_xml),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
