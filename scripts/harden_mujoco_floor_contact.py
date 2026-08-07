#!/usr/bin/env python3
"""Create a task XML with a single, explicit floor-contact compliance contract.

The default MuJoCo contact time constant (20 ms) permits millimetre-scale
compression under G1's active sole spheres.  This utility stiffens only the
static floor material by giving it a higher MuJoCo contact priority.  Hence it
governs contacts *with the floor* without changing chair contacts, robot
kinematics, reference states, loads, or actuator limits.

It is intended as one shared physical-environment setting for all clips, not
as a per-video height offset.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


def _signature(model: mujoco.MjModel) -> dict[str, int]:
    return {
        key: int(getattr(model, key))
        for key in ("nq", "nv", "nu", "nbody", "njnt", "ngeom", "nsite")
    }


def _as_float_tokens(values: list[float]) -> str:
    return " ".join(f"{value:.10g}" for value in values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-xml", required=True, type=Path)
    parser.add_argument("--output-xml", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--timeconst-s", type=float, default=0.008,
        help="positive solref time constant; must be at least two simulation steps",
    )
    parser.add_argument("--damping-ratio", type=float, default=1.0)
    parser.add_argument(
        "--solimp", type=float, nargs=5,
        default=[0.95, 0.99, 0.0001, 0.5, 2.0],
        metavar=("D0", "D_WIDTH", "WIDTH", "MIDPOINT", "POWER"),
    )
    args = parser.parse_args()
    if args.output_xml.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite hardened floor XML/report")
    if args.timeconst_s <= 0.0 or args.damping_ratio <= 0.0:
        raise ValueError("time constant and damping ratio must be positive")
    if not (0.0 < args.solimp[0] <= args.solimp[1] <= 1.0):
        raise ValueError("solimp D0/D_width must satisfy 0 < D0 <= D_width <= 1")

    source_model = mujoco.MjModel.from_xml_path(str(args.input_xml))
    minimum_timeconst = 2.0 * float(source_model.opt.timestep)
    if args.timeconst_s < minimum_timeconst:
        raise ValueError(
            f"timeconst-s={args.timeconst_s} is below MuJoCo's 2*dt={minimum_timeconst}"
        )

    tree = ET.parse(args.input_xml)
    floor_matches = [geom for geom in tree.getroot().iter("geom") if geom.get("name") == "floor"]
    if len(floor_matches) != 1:
        raise ValueError(f"expected exactly one explicit geom named 'floor', got {len(floor_matches)}")
    floor = floor_matches[0]
    prior = {key: floor.get(key) for key in ("priority", "solref", "solimp")}
    floor.set("priority", "1")
    floor.set("solref", _as_float_tokens([args.timeconst_s, args.damping_ratio]))
    floor.set("solimp", _as_float_tokens(args.solimp))

    args.output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.output_xml, encoding="utf-8", xml_declaration=True)
    output_model = mujoco.MjModel.from_xml_path(str(args.output_xml))
    if _signature(source_model) != _signature(output_model):
        raise RuntimeError("floor compliance patch unexpectedly changed model topology")
    floor_id = mujoco.mj_name2id(output_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    report = {
        "schema_version": 1,
        "status": "ready_for_shared_physical_rollout",
        "input_xml": str(args.input_xml.resolve()),
        "output_xml": str(args.output_xml.resolve()),
        "topology": _signature(output_model),
        "floor_contact": {
            "prior_xml_attributes": prior,
            "priority": int(output_model.geom_priority[floor_id]),
            "solref": output_model.geom_solref[floor_id].tolist(),
            "solimp": output_model.geom_solimp[floor_id].tolist(),
            "minimum_allowed_timeconst_s": minimum_timeconst,
        },
        "contract": {
            "changes": "floor contact compliance and priority only",
            "does_not_change": [
                "qpos", "qvel", "joint_ranges", "actuators", "robot_geometry",
                "chair_geometry", "scene_pose", "external_forces", "motion_reference",
            ],
            "rationale": (
                "the priority applies the floor material only to pairs containing the floor; "
                "chair contact compliance remains the task XML default"
            ),
        },
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
