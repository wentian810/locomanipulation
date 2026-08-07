#!/usr/bin/env python3
"""Create a SONIC-compatible copy of a validated GMR MuJoCo task scene.

GMR names the floating-base free joint ``pelvis``.  SONIC's released MuJoCo
bridge identifies an otherwise identical free base by the literal name
``floating_base_joint``.  MuJoCo joint names do not affect kinematics,
contacts, mass, or integration; this utility makes only that name change in a
separate XML file, then reloads it to prove its free-base layout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import mujoco


SOURCE_JOINT = "pelvis"
SONIC_JOINT = "floating_base_joint"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-xml", type=Path, required=True)
    parser.add_argument("--output-xml", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    if not args.input_xml.is_file():
        raise FileNotFoundError(args.input_xml)
    if args.output_xml.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to replace {args.output_xml}; use --overwrite")

    source_text = args.input_xml.read_text(encoding="utf-8")
    pattern = re.compile(
        rf'(<freejoint\b[^>]*\bname\s*=\s*["\']){SOURCE_JOINT}(["\'])',
        flags=re.IGNORECASE,
    )
    output_text, replacements = pattern.subn(rf"\g<1>{SONIC_JOINT}\g<2>", source_text)
    if replacements != 1:
        raise ValueError(
            f"expected exactly one <freejoint name={SOURCE_JOINT!r}>, found {replacements}"
        )
    if f'name="{SONIC_JOINT}"' not in output_text and f"name='{SONIC_JOINT}'" not in output_text:
        raise RuntimeError("SONIC floating-base joint rename did not appear in output")

    args.output_xml.parent.mkdir(parents=True, exist_ok=True)
    args.output_xml.write_text(output_text, encoding="utf-8")

    model = mujoco.MjModel.from_xml_path(str(args.output_xml))
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, SONIC_JOINT)
    if joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise RuntimeError("output scene does not contain the required SONIC free joint")
    if model.jnt_qposadr[joint_id] != 0 or model.jnt_dofadr[joint_id] != 0:
        raise RuntimeError("SONIC free joint is not the leading MuJoCo state")

    report = {
        "status": "written",
        "input_xml": str(args.input_xml.resolve()),
        "output_xml": str(args.output_xml.resolve()),
        "input_sha256": sha256(args.input_xml),
        "output_sha256": sha256(args.output_xml),
        "changed": {"freejoint_name": [SOURCE_JOINT, SONIC_JOINT]},
        "mujoco": {"nq": model.nq, "nv": model.nv, "nu": model.nu, "ngeom": model.ngeom},
    }
    report_path = args.output_xml.with_suffix(".compatibility.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
