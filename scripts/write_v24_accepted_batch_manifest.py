#!/usr/bin/env python3
"""Freeze the accepted V24 physical artifacts into an integrity-checked batch manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path("/home/jixingyu/Loco-manipulation-human-only-release")
HOLO = Path("/home/jixingyu/third_party/HoloMotion")
WORK = ROOT / "scene_work/g1_mpc_v24_batch_preflight"
OUTPUT = HOLO / "checkpoints/holomotion_v1.4/mujoco_output_model_14000"


CASES = (
    {
        "clip_id": "pianist",
        "semantic_seat_cue_frame": 128,
        "task_xml": WORK / "pianist_holomotion_static_chair_joint_margin10mrad_v24.xml",
        "physical_reference": WORK / "pianist_holomotion_policy_impedance_049_margin10_v24_reference.npz",
        "controller_reference": WORK / "pianist_holomotion_policy_impedance_049_margin10_v24_reference.npz",
        "actual_rollout": OUTPUT / "pianist_holomotion_policy_impedance_049_margin10_v24_reference_robot.npz",
        "audit": WORK / "pianist_holomotion_v24_batch25_audit.json",
    },
    {
        "clip_id": "dramatic",
        "semantic_seat_cue_frame": 142,
        "task_xml": WORK / "dramatic_holomotion_static_chair_selected_seat_margin10mrad_v24.xml",
        "physical_reference": WORK / "dramatic_holomotion_physical_seat_hold_v24_reference.npz",
        "controller_reference": WORK / "dramatic_holomotion_controller_bias_precap50mm_v24_reference.npz",
        "actual_rollout": OUTPUT / "dramatic_holomotion_controller_bias_precap50mm_v24_reference_robot.npz",
        "audit": WORK / "dramatic_holomotion_v24_batch25_audit.json",
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_accepted_audit(path: Path) -> dict[str, object]:
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("purpose") != "frozen_holomotion_bounded_mj_step_v24_audit":
        raise ValueError(f"{path} is not a V24 HoloMotion audit")
    if audit.get("acceptance", {}).get("physical_smoke_pass") is not True:
        raise ValueError(f"{path} is not physically accepted")
    if audit.get("chair", {}).get("seating", {}).get("pass") is not True:
        raise ValueError(f"{path} lacks accepted load-bearing seat support")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite an accepted batch manifest")

    cases: list[dict[str, object]] = []
    for configured in CASES:
        artifacts = {key: configured[key] for key in (
            "task_xml", "physical_reference", "controller_reference", "actual_rollout", "audit",
        )}
        for path in artifacts.values():
            if not isinstance(path, Path) or not path.is_file():
                raise FileNotFoundError(path)
        audit = _load_accepted_audit(artifacts["audit"])
        seating = audit["chair"]["seating"]
        cases.append({
            "clip_id": configured["clip_id"],
            "semantic_seat_cue_frame": configured["semantic_seat_cue_frame"],
            "artifacts": {
                key: {"path": str(path), "sha256": _sha256(path)}
                for key, path in artifacts.items()
            },
            "metrics": {
                "root_rmse_m": audit["root_tracking"]["rmse_m"],
                "root_p95_position_error_m": audit["root_tracking"]["p95_position_error_m"],
                "max_root_z_frame_jump_m": audit["root_height"]["max_frame_jump_m"],
                "min_chair_distance_m": audit["chair"]["minimum_all_distance_m"],
                "median_seat_weight_fraction": seating["median_weight_fraction"],
                "seat_supported_frame_ratio": seating["supported_frame_ratio"],
            },
            "runtime_control": audit["physical_contract"]["runtime_control"],
            "controller_reference_calibration": audit["physical_contract"]["controller_reference_calibration"],
        })

    manifest = {
        "schema_version": 1,
        "status": "accepted_v24_static_chair_batch",
        "controller": "frozen HoloMotion ONNX model_14000; no training or reinforcement learning",
        "shared_physical_contract": {
            "integration": "one initial qpos/qvel then continuous MuJoCo mj_step",
            "scene": "fixed semantic chair geoms in worldbody; no moving scene geometry",
            "actuation": "bounded internal joint PD torque only; no external pelvis/base force",
            "forbidden": ["per_frame_qpos_reset", "xfrc_applied", "mocap_weld", "moving_scene_geometry"],
            "acceptance_gates": [
                "root tracking and root-Z continuity",
                "published joint limits and actuator torque limits",
                "all robot-chair signed distances",
                "seat support coverage and median load fraction",
            ],
        },
        "cases": cases,
        "selection_note": (
            "Only artifacts whose independent V24 audit already passes are frozen here. "
            "Floor-clearance projection experiments are intentionally excluded because they "
            "introduced a later chair-leg collision under the frozen policy."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
