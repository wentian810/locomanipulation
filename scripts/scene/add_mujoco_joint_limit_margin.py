#!/usr/bin/env python3
"""Create a MuJoCo task XML with an explicit conservative joint-stop margin.

MuJoCo joint-limit constraints are compliant by design.  For physical safety
validation, this tool activates each existing hinge/slide limit slightly before
its declared range so the simulated joint does not numerically overshoot the
robot's published mechanical range during a contact transient.  It neither
changes a range, adds a force, nor changes bodies/geometries/actuators.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


def _model_signature(model: mujoco.MjModel) -> dict[str, int]:
    return {key: int(getattr(model, key)) for key in ("nq", "nv", "nu", "nbody", "njnt", "ngeom")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-xml", required=True, type=Path)
    parser.add_argument("--output-xml", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--margin-rad", required=True, type=float)
    args = parser.parse_args()
    if args.output_xml.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite safety-margin XML/report")
    if not 0.0 < args.margin_rad < 0.1:
        raise ValueError("margin-rad must be in (0, 0.1)")

    input_model = mujoco.MjModel.from_xml_path(str(args.input_xml))
    tree = ET.parse(args.input_xml)
    root = tree.getroot()
    changed: list[str] = []
    for joint in root.iter("joint"):
        joint_type = joint.get("type", "hinge")
        if joint_type not in {"hinge", "slide"} or joint.get("range") is None:
            continue
        joint.set("margin", f"{args.margin_rad:.10g}")
        changed.append(joint.get("name", "<unnamed>"))
    if not changed:
        raise ValueError("no explicitly ranged hinge/slide joints found")
    args.output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.output_xml, encoding="utf-8", xml_declaration=True)
    output_model = mujoco.MjModel.from_xml_path(str(args.output_xml))
    if _model_signature(input_model) != _model_signature(output_model):
        raise RuntimeError("joint margin unexpectedly changed model topology")
    report = {
        "schema_version": 1,
        "status": "ready_for_physical_safety_validation",
        "input_xml": str(args.input_xml.resolve()),
        "output_xml": str(args.output_xml.resolve()),
        "joint_limit_margin_rad": args.margin_rad,
        "changed_joint_count": len(changed),
        "changed_joint_names": changed,
        "topology": _model_signature(output_model),
        "contract": {
            "changes": "early activation of existing MuJoCo joint-limit constraints only",
            "does_not_change": ["joint_range", "qpos", "qvel", "actuators", "bodies", "scene_geometry", "external_forces"],
        },
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
