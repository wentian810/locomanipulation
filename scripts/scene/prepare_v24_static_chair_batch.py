#!/usr/bin/env python3
"""Prepare versioned, contact-ready v24 MPC tasks for static-chair clips.

The input scene must already be an accepted *static* VideoMimic chair fit.
For every case this program validates that the fit and GMR motion have the
same SHA256, derives the stable seated tail from the fit report, constructs a
millimetre-resolution foot-preserving contact reference, and builds a v24
exact-contact MPC task.  It never changes the source motion or scene.

This is deliberately a preparation stage, not a physical pass: an accepted
candidate still needs a bounded-actuator, continuous ``mj_step`` rollout and
the companion trajectory audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MAX_JOINT_DELTA_RAD = 0.40
DEFAULT_POST_IK_SCAN_STEP_M = 0.001


@dataclass(frozen=True)
class Case:
    identifier: str
    robot_motion: Path
    scene_mujoco_xml: Path
    static_chair_fit_report: Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cases(manifest_path: Path) -> list[Case]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = payload.get("cases")
    if not isinstance(items, list) or not items:
        raise ValueError("manifest requires a non-empty 'cases' list")
    cases: list[Case] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("every manifest case must be an object")
        identifier = str(item.get("id", ""))
        if not identifier.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"invalid case id: {identifier!r}")
        if identifier in seen:
            raise ValueError(f"duplicate case id: {identifier}")
        seen.add(identifier)
        try:
            cases.append(Case(
                identifier=identifier,
                robot_motion=Path(item["robot_motion"]),
                scene_mujoco_xml=Path(item["scene_mujoco_xml"]),
                static_chair_fit_report=Path(item["static_chair_fit_report"]),
            ))
        except KeyError as error:
            raise ValueError(f"case {identifier!r} misses {error.args[0]!r}") from error
    return cases


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def read_fit_contract(case: Case) -> tuple[int, int, dict[str, Any]]:
    for path in (case.robot_motion, case.scene_mujoco_xml, case.static_chair_fit_report):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = json.loads(case.static_chair_fit_report.read_text(encoding="utf-8"))
    if report.get("status") != "accepted":
        raise ValueError("static chair fit is not accepted")
    expected_sha = report.get("robot_motion_sha256_after")
    actual_sha = sha256(case.robot_motion)
    if expected_sha != actual_sha:
        raise ValueError(
            "static chair fit motion SHA does not match robot_motion "
            f"({expected_sha!r} != {actual_sha!r})"
        )
    frame_range = report.get("fit_frame_range")
    if (not isinstance(frame_range, list) or len(frame_range) != 2 or
            not all(isinstance(value, int) for value in frame_range)):
        raise ValueError("static chair fit lacks an integer fit_frame_range")
    start, terminal = frame_range
    if not 0 <= start < terminal:
        raise ValueError(f"invalid stable seated tail: {frame_range!r}")
    return start, terminal, report


def build_task(
    python: str,
    builder: Path,
    robot_xml: Path,
    marker_map: Path,
    official_collision_mjcf: Path,
    motion: Path,
    scene: Path,
    output_xml: Path,
    report: Path,
    detect_contact: bool,
    agent_planner: str,
    sampling_trajectories: int,
    actuator_profile: str,
) -> None:
    command = [
        python, str(builder),
        "--robot-motion", str(motion),
        "--robot-xml", str(robot_xml),
        "--scene-mujoco-xml", str(scene),
        "--marker-map", str(marker_map),
        "--output-xml", str(output_xml),
        "--report", str(report),
        "--tracking-profile", "marker_and_posture_chair_contact_v2",
        "--agent-planner", agent_planner,
        "--sampling-trajectories", str(sampling_trajectories),
        "--planner-contact-model", "exact",
        "--actuator-profile", actuator_profile,
        "--collision-proxy-strategy", "official_unitree_mjcf",
        "--official-collision-mjcf", str(official_collision_mjcf),
        "--joint-limit-contract", "official_g1",
        "--reference-joint-limit-tolerance-rad", "0.005",
        "--constraint-profile", "rigid_static_support_v2",
    ]
    if detect_contact:
        command += ["--detect-reference-chair-contact", "--chair-contact-min-consecutive-frames", "2"]
    run(command)


def prepare_case(args: argparse.Namespace, case: Case) -> dict[str, Any]:
    stable_start, stable_terminal, fit = read_fit_contract(case)
    destination = args.output_root / case.identifier
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing case directory: {destination}")
    destination.mkdir(parents=True)
    source_task = destination / "source_v24_task.xml"
    source_task_report = destination / "source_v24_task_report.json"
    candidate_motion = destination / "terminal_static_chair_contact_reference.pkl"
    candidate_report = destination / "terminal_static_chair_contact_reference_report.json"
    terminal_task = destination / "terminal_v24_task.xml"
    terminal_task_report = destination / "terminal_v24_task_report.json"

    build_task(
        args.python, args.task_builder, args.robot_xml, args.marker_map,
        args.official_collision_mjcf, case.robot_motion, case.scene_mujoco_xml,
        source_task, source_task_report, detect_contact=False,
        agent_planner=args.agent_planner,
        sampling_trajectories=args.sampling_trajectories,
        actuator_profile=args.actuator_profile,
    )
    run([
        args.python, str(args.contact_constructor),
        "--source-motion", str(case.robot_motion),
        "--task-xml", str(source_task),
        "--task-build-report", str(source_task_report),
        "--output-motion", str(candidate_motion),
        "--report", str(candidate_report),
        "--sit-start-frame", str(stable_start),
        "--sit-end-frame", str(stable_terminal),
        "--post-ik-scan-step-m", str(args.post_ik_scan_step_m),
        "--max-joint-delta-rad", str(args.max_joint_delta_rad),
    ])
    candidate = json.loads(candidate_report.read_text(encoding="utf-8"))
    if candidate.get("status") != "accepted_kinematic_seat_reference_requires_mj_step":
        raise ValueError(f"contact candidate rejected: {candidate.get('status')!r}")
    build_task(
        args.python, args.task_builder, args.robot_xml, args.marker_map,
        args.official_collision_mjcf, candidate_motion, case.scene_mujoco_xml,
        terminal_task, terminal_task_report, detect_contact=True,
        agent_planner=args.agent_planner,
        sampling_trajectories=args.sampling_trajectories,
        actuator_profile=args.actuator_profile,
    )
    task_report = json.loads(terminal_task_report.read_text(encoding="utf-8"))
    detection = task_report["reference"]["auto_chair_contact_detection"]
    first_contact = detection.get("first_robust_seat_contact_source_frame")
    if not isinstance(first_contact, int):
        raise ValueError("contact-ready task has no robust automatic seat-contact interval")
    return {
        "status": "ready_for_mpc_mj_step_audit",
        "robot_motion_sha256": sha256(case.robot_motion),
        "scene_mujoco_xml": str(case.scene_mujoco_xml),
        "static_chair_fit_report": str(case.static_chair_fit_report),
        "static_fit_frame_range": [stable_start, stable_terminal],
        "candidate_motion": str(candidate_motion),
        "candidate_report": str(candidate_report),
        "task_xml": str(terminal_task),
        "task_report": str(terminal_task_report),
        "first_robust_seat_contact_source_frame": first_contact,
        "candidate_quality": candidate["quality"],
        "static_fit_quality_gate": fit.get("quality_gate"),
    }


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--task-builder", type=Path, default=here / "build_g1_mjpc_task_bundle.py")
    parser.add_argument("--contact-constructor", type=Path, default=here / "construct_gmr_seat_contact_reference.py")
    parser.add_argument("--robot-xml", required=True, type=Path)
    parser.add_argument("--marker-map", required=True, type=Path)
    parser.add_argument("--official-collision-mjcf", required=True, type=Path)
    parser.add_argument(
        "--agent-planner",
        choices=(
            "sampling", "gradient", "ilqg", "ilqs", "robust_sampling",
            "cross_entropy", "sample_gradient",
        ),
        default="ilqg",
        help="forward the audited MJPC solver choice to every case task",
    )
    parser.add_argument(
        "--sampling-trajectories",
        type=int,
        default=32,
        help="rollout count passed to sampling-based solvers for every case",
    )
    parser.add_argument(
        "--actuator-profile",
        choices=(
            "unitree_g1_29dof_position_servo_v1",
            "unitree_g1_29dof_torque_motor_v1",
        ),
        default="unitree_g1_29dof_position_servo_v1",
        help=(
            "physical low-level interface for every case; position-servo "
            "commands remain force-saturated by the official G1 ranges"
        ),
    )
    parser.add_argument("--max-joint-delta-rad", type=float, default=DEFAULT_MAX_JOINT_DELTA_RAD)
    parser.add_argument("--post-ik-scan-step-m", type=float, default=DEFAULT_POST_IK_SCAN_STEP_M)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {args.output_root}")
    if not 0.0 < args.max_joint_delta_rad <= 0.5:
        raise ValueError("max-joint-delta-rad must be in (0, 0.5]")
    if not 0.0 < args.post_ik_scan_step_m <= 0.003:
        raise ValueError("post-ik-scan-step-m must be in (0, 0.003]")
    if args.sampling_trajectories < 2:
        raise ValueError("sampling-trajectories must be at least 2")
    args.output_root.mkdir(parents=True)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "static_chair_contact_reference_v24_batch_preparation",
        "physical_contract": {
            "scene": "one_static_chair_per_case",
            "offline_operation": "bounded_contact_reference_kinematics_only",
            "required_next_stage": "bounded_actuator_continuous_mj_step_then_trajectory_audit",
            "forbidden": ["per_frame_chair_motion", "external_pelvis_force", "mocap_weld", "runtime_root_state_reimposition"],
        },
        "settings": {
            "post_ik_scan_step_m": args.post_ik_scan_step_m,
            "max_joint_delta_rad": args.max_joint_delta_rad,
            "agent_planner": args.agent_planner,
            "sampling_trajectories": args.sampling_trajectories,
            "actuator_profile": args.actuator_profile,
        },
        "cases": {},
    }
    failed = False
    for case in load_cases(args.manifest):
        try:
            summary["cases"][case.identifier] = prepare_case(args, case)
        except Exception as error:  # Keep other independent cases batchable.
            failed = True
            summary["cases"][case.identifier] = {"status": "rejected", "reason": str(error)}
            print(f"{case.identifier}: rejected: {error}", file=sys.stderr, flush=True)
    (args.output_root / "batch_preparation_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
