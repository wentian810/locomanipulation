#!/usr/bin/env python3
"""Retarget a GVHMR MANO sidecar to BrainCo Revo2 hands.

The official Revo2 MJCF describes 11 hinge joints per hand, while the physical
hand exposes six motors.  Five MuJoCo joint equalities added here reproduce the
official URDF mimic rules, so the solved trajectory is both visually useful and
compatible with the real mechanism.
"""

from __future__ import annotations

import argparse
import pathlib
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from sharpa_hand_retarget import load_targets, maybe_trim, solve_side


HERE = pathlib.Path(__file__).resolve().parent
GMR_ROOT = HERE.parent
DEFAULT_BRAINCO_ROOT = (
    GMR_ROOT
    / "third_party"
    / "robot_hands"
    / "brainco_description"
    / "revo2_system"
)
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def find_body(root: ET.Element, name: str) -> ET.Element:
    for body in root.iter("body"):
        if body.attrib.get("name") == name:
            return body
    raise ValueError(f"Body {name!r} is missing from BrainCo MJCF")


def add_site(body: ET.Element, name: str, pos=(0.0, 0.0, 0.0)) -> None:
    ET.SubElement(
        body,
        "site",
        {
            "name": name,
            "pos": " ".join(f"{float(value):.9g}" for value in pos),
            "size": "0.003",
            "type": "sphere",
            "rgba": "1 0 0 0",
        },
    )


def parse_vec(text: str) -> np.ndarray:
    return np.asarray([float(value) for value in text.split()], dtype=np.float64)


def build_ik_xml(source_xml: pathlib.Path, side: str) -> pathlib.Path:
    """Add a six-DoF auxiliary palm, chain sites and hardware mimic rules."""
    tree = ET.parse(source_xml)
    root = tree.getroot()
    hand_body = find_body(root, f"{side}_hand_base_link")

    # Six auxiliary coordinates place the standalone palm in target space.
    # They are removed from the exported hand trajectory.
    auxiliary_joints = (
        (f"{side}_pos_x", "slide", "1 0 0", "-5 5"),
        (f"{side}_pos_y", "slide", "0 1 0", "-5 5"),
        (f"{side}_pos_z", "slide", "0 0 1", "-5 5"),
        (f"{side}_rot_x", "hinge", "1 0 0", "-6.28 6.28"),
        (f"{side}_rot_y", "hinge", "0 1 0", "-6.28 6.28"),
        (f"{side}_rot_z", "hinge", "0 0 1", "-6.28 6.28"),
    )
    insert_at = 1 if hand_body.find("inertial") is not None else 0
    for name, joint_type, axis, joint_range in reversed(auxiliary_joints):
        joint = ET.Element(
            "joint",
            {
                "name": name,
                "type": joint_type,
                "axis": axis,
                "range": joint_range,
                "damping": "0.01",
            },
        )
        hand_body.insert(insert_at, joint)

    add_site(hand_body, f"{side}_palm", (0.0, 0.0, 0.05))

    # Site names intentionally match the morphology-normalized chain solver
    # used by Sharpa. Revo2 has two links on the four long fingers, so the DIP
    # observation site lies halfway along the distal link.
    for finger in FINGERS:
        if finger == "thumb":
            mcp_body = find_body(root, f"{side}_thumb_metacarpal_link")
            pip_body = find_body(root, f"{side}_thumb_proximal_link")
            dip_body = find_body(root, f"{side}_thumb_distal_link")
            tip_body = find_body(root, f"{side}_thumb_tip_link")
            add_site(mcp_body, f"{side}_{finger}_mcp")
            add_site(pip_body, f"{side}_{finger}_pip")
            add_site(dip_body, f"{side}_{finger}_dip")
            add_site(tip_body, f"{side}_{finger}_tip")
            continue

        proximal = find_body(root, f"{side}_{finger}_proximal_link")
        distal = find_body(root, f"{side}_{finger}_distal_link")
        tip = find_body(root, f"{side}_{finger}_tip_link")
        tip_pos = parse_vec(tip.attrib.get("pos", "0 0 0"))
        add_site(proximal, f"{side}_{finger}_mcp")
        add_site(distal, f"{side}_{finger}_pip")
        add_site(distal, f"{side}_{finger}_dip", tip_pos * 0.5)
        add_site(tip, f"{side}_{finger}_tip")

    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    mimic_rules = {"thumb": 1.0, "index": 1.155, "middle": 1.155, "ring": 1.155, "pinky": 1.155}
    for finger, multiplier in mimic_rules.items():
        ET.SubElement(
            equality,
            "joint",
            {
                "name": f"{side}_{finger}_hardware_mimic",
                "joint1": f"{side}_{finger}_distal_joint",
                "joint2": f"{side}_{finger}_proximal_joint",
                "polycoef": f"0 {multiplier:.9g} 0 0 0",
                "solref": "0.002 1",
            },
        )

    tmp = tempfile.NamedTemporaryFile(
        prefix=f"brainco_revo2_{side}_ik_",
        suffix=".xml",
        dir=str(source_xml.parent),
        delete=False,
    )
    tmp.close()
    tree.write(tmp.name, encoding="unicode")
    return pathlib.Path(tmp.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand_npz", required=True, type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--brainco_root", default=DEFAULT_BRAINCO_ROOT, type=pathlib.Path)
    parser.add_argument("--side", choices=["left", "right", "both"], default="both")
    parser.add_argument("--solver", default="daqp")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--wrist_pos_cost", type=float, default=0.3)
    parser.add_argument("--wrist_ori_cost", type=float, default=0.2)
    parser.add_argument("--finger_pos_cost", type=float, default=5.0)
    parser.add_argument("--joint_pos_cost", type=float, default=3.0)
    parser.add_argument("--posture_cost", type=float, default=1e-2)
    parser.add_argument("--temporal_cost", type=float, default=3e-2)
    parser.add_argument("--low_conf_temporal_gain", type=float, default=3.0)
    parser.add_argument("--min_target_confidence_scale", type=float, default=0.10)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--init_steps", type=int, default=50)
    parser.add_argument("--smooth_window", type=int, default=5)
    parser.add_argument("--max_delta", type=float, default=0.12)
    parser.add_argument("--max_accel", type=float, default=0.10)
    parser.add_argument("--low_conf_accel_scale", type=float, default=0.5)
    parser.add_argument("--reproj_good_px", type=float, default=30.0)
    parser.add_argument("--reproj_bad_px", type=float, default=75.0)
    parser.add_argument("--reproj_good_ratio", type=float, default=0.15)
    parser.add_argument("--reproj_bad_ratio", type=float, default=0.45)
    parser.add_argument("--max_pip_bend_deg", type=float, default=125.0)
    parser.add_argument("--max_dip_bend_deg", type=float, default=105.0)
    parser.add_argument("--max_source_bone_delta_deg", type=float, default=45.0)
    parser.add_argument("--max_source_bone_length_ratio", type=float, default=0.25)
    parser.add_argument("--anatomic_repair_max_gap", type=int, default=15)
    parser.add_argument("--max_frames", type=int, default=0)
    args = parser.parse_args()

    source_dir = args.brainco_root / "mjcf"
    for side in ("left", "right"):
        source_xml = source_dir / f"revo2_{side}.xml"
        if not source_xml.is_file():
            raise FileNotFoundError(
                f"BrainCo Revo2 asset is missing: {source_xml}. "
                "Run GMR-master/scripts/setup_robot_hand_assets.sh brainco."
            )

    sides = ("left", "right") if args.side == "both" else (args.side,)
    output = args.output or args.hand_npz.with_name("001_brainco_revo2_hands.npz")
    out: dict[str, object] = {
        "source_hand_npz": str(args.hand_npz),
        "hand_model": "brainco_revo2",
        "brainco_root": str(args.brainco_root),
        "hardware_motor_count_per_hand": 6,
        "scale": float(args.scale),
    }

    for side in sides:
        targets = maybe_trim(
            load_targets(
                args.hand_npz,
                side,
                args.scale,
                args.reproj_good_px,
                args.reproj_bad_px,
                args.reproj_good_ratio,
                args.reproj_bad_ratio,
            ),
            args.max_frames,
        )
        ik_xml = build_ik_xml(source_dir / f"revo2_{side}.xml", side)
        try:
            qpos, names, error_mean, error_p95, source_stats = solve_side(
                ik_xml,
                side,
                targets,
                args.solver,
                args.wrist_pos_cost,
                args.wrist_ori_cost,
                args.finger_pos_cost,
                args.joint_pos_cost,
                args.posture_cost,
                args.temporal_cost,
                args.low_conf_temporal_gain,
                args.min_target_confidence_scale,
                args.steps,
                args.init_steps,
                args.smooth_window,
                args.max_delta,
                args.max_accel,
                args.low_conf_accel_scale,
                args.max_pip_bend_deg,
                args.max_dip_bend_deg,
                args.max_source_bone_delta_deg,
                args.max_source_bone_length_ratio,
                args.anatomic_repair_max_gap,
            )
        finally:
            ik_xml.unlink(missing_ok=True)

        out[f"{side}_qpos"] = qpos
        out[f"{side}_hand_qpos"] = qpos[:, 6:]
        out[f"{side}_qpos_names"] = np.asarray(names)
        out[f"{side}_hand_qpos_names"] = np.asarray(names[6:])
        out[f"{side}_valid"] = targets["valid"]
        out[f"{side}_reliability"] = targets["reliability"]
        out[f"{side}_chain_error_mean"] = float(error_mean)
        out[f"{side}_chain_error_p95"] = float(error_p95)
        for key, value in source_stats.items():
            out[f"{side}_{key}"] = int(value)
        print(
            f"{side}: valid={int(np.sum(targets['valid']))}/{len(targets['valid'])}, "
            f"chain_error_mean={error_mean * 1000.0:.2f}mm, "
            f"p95={error_p95 * 1000.0:.2f}mm"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **out)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
