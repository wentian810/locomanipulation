#!/usr/bin/env python3
"""Create an isolated SONIC MuJoCo config for the GMR physics scene.

SONIC's stock simulator YAML enables an elastic support band for interactive
teleoperation.  That band applies an external wrench to the robot, so it must
never be enabled for the physical GMR validation.  This utility preserves the
stock configuration as input and writes a separate headless config pointing to
one explicit GMR scene XML.

It deliberately does not modify the scene XML, initialise qpos, or generate
controls.  Those operations belong respectively to the validated scene build
and the official SONIC controller.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--scene-xml", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    # The released C++ controller uses the Unitree simulator's default DDS
    # domain unless explicitly rebuilt with a different value.
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument(
        "--interface",
        default="lo",
        help="DDS network interface used by both the simulator and controller",
    )
    parser.add_argument(
        "--num-hand-joints",
        type=int,
        help="Override both declared hand-joint and hand-motor counts for a handless scene",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a top-level YAML mapping")
    return config


def main() -> None:
    args = parse_args()
    if not args.base_config.is_file():
        raise FileNotFoundError(args.base_config)
    if not args.scene_xml.is_file():
        raise FileNotFoundError(args.scene_xml)
    if args.domain_id < 0:
        raise ValueError("--domain-id must be non-negative")
    if not args.interface.strip():
        raise ValueError("--interface must be non-empty")
    if args.num_hand_joints is not None and args.num_hand_joints < 0:
        raise ValueError("--num-hand-joints must be non-negative")
    if args.output_config.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to replace {args.output_config}; use --overwrite")

    config = load_mapping(args.base_config)
    robot_type = str(config.get("ROBOT_TYPE", ""))
    if not robot_type.lower().startswith("g1"):
        raise ValueError(f"expected a G1 config, got ROBOT_TYPE={robot_type!r}")

    # Absolute paths are intentional: SONIC otherwise resolves ROBOT_SCENE
    # relative to its repository root, which would silently select its stock
    # floor-only scene instead of the validated GMR scene.
    scene_xml = args.scene_xml.resolve()
    config["ROBOT_SCENE"] = str(scene_xml)
    config["DOMAIN_ID"] = args.domain_id
    config["INTERFACE"] = args.interface
    config["ENABLE_ELASTIC_BAND"] = False
    config["ENABLE_ONSCREEN"] = False
    config["ENABLE_OFFSCREEN"] = False
    if args.num_hand_joints is not None:
        # GMR's 29-DoF XML has no hand joints.  SONIC's G1 config also
        # supports that model, but its released 43-DoF demo defaults to
        # seven joints per hand, so state this topology explicitly.
        config["NUM_HAND_JOINTS"] = args.num_hand_joints
        config["NUM_HAND_MOTORS"] = args.num_hand_joints
        body_joint_count = int(config["NUM_JOINTS"])
        effort_limits = list(config["motor_effort_limit_list"])
        expected_count = body_joint_count + 2 * args.num_hand_joints
        if len(effort_limits) < expected_count:
            raise ValueError(
                "motor_effort_limit_list is shorter than the declared handless topology"
            )
        config["motor_effort_limit_list"] = effort_limits[:expected_count]

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    with args.output_config.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    report = {
        "status": "written",
        "base_config": str(args.base_config.resolve()),
        "output_config": str(args.output_config.resolve()),
        "robot_scene": str(scene_xml),
        "domain_id": args.domain_id,
        "interface": args.interface,
        "num_hand_joints": config.get("NUM_HAND_JOINTS", 0),
        "motor_effort_limit_count": len(config["motor_effort_limit_list"]),
        "elastic_band": False,
        "root_state_writes": "none",
        "external_wrenches": "none",
    }
    report_path = args.output_config.with_suffix(".physics_contract.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
