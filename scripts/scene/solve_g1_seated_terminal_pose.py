#!/usr/bin/env python3
"""Solve one collision-aware G1 seated pose from a frozen GMR reference.

This is an offline *terminal target* optimiser, not a simulator.  It never
steps MuJoCo, applies external forces, or changes the scene.  A later dynamic
pass must initialise once and use only torque control plus ``mj_step``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares, minimize


CHAIR_PREFIXES = ("seat_support_geom", "backrest_geom", "leg_front_", "leg_back_")
FOOT_SITES = ("tracking[ltoe]", "tracking[lheel]", "tracking[rtoe]", "tracking[rheel]")
# The GMR custom collision URDF describes upper legs on the hip-yaw links,
# whereas the native XML has no collision geometry on pelvis/hip-pitch.  Keep
# the ordered list explicit and select only bodies that actually own a
# collision geom in the compiled task.
SUPPORT_BODY_CANDIDATES = (
    "pelvis",
    "left_hip_yaw_link", "right_hip_yaw_link",
    "left_hip_pitch_link", "right_hip_pitch_link",
    "left_knee_link", "right_knee_link",
)


def _name(model: mujoco.MjModel, object_type: mujoco.mjtObj, object_id: int) -> str:
    return mujoco.mj_id2name(model, object_type, object_id) or ""


def _descendants(model: mujoco.MjModel, root_body: int) -> set[int]:
    result = {root_body}
    changed = True
    while changed:
        changed = False
        for body_id in range(1, model.nbody):
            if body_id not in result and int(model.body_parentid[body_id]) in result:
                result.add(body_id)
                changed = True
    return result


def _collision_geoms(model: mujoco.MjModel, bodies: set[int]) -> list[int]:
    return [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in bodies
        and (int(model.geom_contype[geom_id]) != 0 or int(model.geom_conaffinity[geom_id]) != 0)
    ]


def _pair_distances(
    model: mujoco.MjModel, data: mujoco.MjData, robot_geoms: list[int], chair_geoms: list[int]
) -> np.ndarray:
    nearest_points = np.empty(6, dtype=np.float64)
    result = []
    for robot_geom in robot_geoms:
        for chair_geom in chair_geoms:
            result.append(float(mujoco.mj_geomDistance(model, data, robot_geom, chair_geom, 2.0, nearest_points)))
    return np.asarray(result, dtype=np.float64)


def _minimum_distance(
    model: mujoco.MjModel, data: mujoco.MjData, robot_geoms: list[int], chair_geoms: list[int]
) -> float:
    return float(np.min(_pair_distances(model, data, robot_geoms, chair_geoms)))


@dataclass(frozen=True)
class ProblemSpec:
    qpos_reference: np.ndarray
    joint_qpos_indices: np.ndarray
    freeze_root: bool
    foot_site_ids: np.ndarray
    foot_target: np.ndarray
    non_support_geoms: list[int]
    chair_geoms: list[int]
    support_geoms: list[int]
    nonseat_chair_geoms: list[int]
    support_body_names: tuple[str, ...]
    seat_geom: int
    clearance_m: float
    support_gap_m: float


class SeatedPoseProblem:
    def __init__(self, model: mujoco.MjModel, spec: ProblemSpec) -> None:
        self.model = model
        self.spec = spec
        self.data = mujoco.MjData(model)

    def qpos_from_x(self, x: np.ndarray) -> np.ndarray:
        qpos = self.spec.qpos_reference.copy()
        if self.spec.freeze_root:
            qpos[self.spec.joint_qpos_indices] = x
        else:
            qpos[:3] = x[:3]
            qpos[self.spec.joint_qpos_indices] = x[3:]
        return qpos

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = self.qpos_from_x(x)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.data.qpos.copy()

    def residual(self, x: np.ndarray) -> np.ndarray:
        self.forward(x)
        # The physical candidate may only modify joint targets.  The free base
        # is either fixed here for an offline feasibility test or left to the
        # later MuJoCo rollout; it is never a scene-alignment variable.
        if self.spec.freeze_root:
            root_cost = np.empty(0, dtype=np.float64)
            joint_delta = x - self.spec.qpos_reference[self.spec.joint_qpos_indices]
        else:
            root_delta = x[:3] - self.spec.qpos_reference[:3]
            root_cost = root_delta / np.asarray((0.08, 0.08, 0.16))
            joint_delta = x[3:] - self.spec.qpos_reference[self.spec.joint_qpos_indices]
        posture_cost = 0.35 * joint_delta
        foot_cost = 24.0 * (self.data.site_xpos[self.spec.foot_site_ids] - self.spec.foot_target).reshape(-1)

        # Seat support is deliberately excluded from this clearance term: it
        # is handled by the separate support-distance band below.  Treating a
        # permitted hip/pelvis-to-seat pair as a forbidden collision makes
        # the optimisation contradictory and prevents any seated solution.
        non_support_distances = _pair_distances(
            self.model, self.data,
            self.spec.non_support_geoms, self.spec.chair_geoms,
        )
        support_nonseat_distances = _pair_distances(
            self.model, self.data,
            self.spec.support_geoms, self.spec.nonseat_chair_geoms,
        )
        collision_cost = 18.0 * np.maximum(
            0.0,
            self.spec.clearance_m - np.concatenate(
                (non_support_distances, support_nonseat_distances)
            ),
        )
        support_distance = _minimum_distance(
            self.model, self.data, self.spec.support_geoms, [self.spec.seat_geom]
        )
        # This asks the pelvis/upper-thigh support set to finish just above the
        # seat.  It does not add a contact force; the later dynamic pass can
        # only create support through gravity and joint torque.
        support_cost = np.asarray(((support_distance - self.spec.support_gap_m) / 0.004,))
        return np.concatenate((root_cost, posture_cost, foot_cost, collision_cost, support_cost))

    def metrics(self, x: np.ndarray) -> dict[str, float]:
        qpos = self.forward(x)
        non_support_distances = _pair_distances(
            self.model, self.data,
            self.spec.non_support_geoms, self.spec.chair_geoms,
        )
        support_nonseat_distances = _pair_distances(
            self.model, self.data,
            self.spec.support_geoms, self.spec.nonseat_chair_geoms,
        )
        support_distance = _minimum_distance(
            self.model, self.data, self.spec.support_geoms, [self.spec.seat_geom]
        )
        foot_error = self.data.site_xpos[self.spec.foot_site_ids] - self.spec.foot_target
        return {
            "minimum_non_support_chair_distance_m": float(np.min(non_support_distances)),
            "minimum_support_nonseat_distance_m": float(np.min(support_nonseat_distances)),
            "support_distance_to_seat_m": support_distance,
            "maximum_foot_anchor_error_m": float(np.max(np.linalg.norm(foot_error, axis=1))),
            "root_xyz_delta_m": [float(value) for value in qpos[:3] - self.spec.qpos_reference[:3]],
            "joint_l2_delta_rad": float(np.linalg.norm(qpos[self.spec.joint_qpos_indices] - self.spec.qpos_reference[self.spec.joint_qpos_indices])),
        }

    def objective(self, x: np.ndarray) -> float:
        if self.spec.freeze_root:
            joint_delta = x - self.spec.qpos_reference[self.spec.joint_qpos_indices]
            return float(0.15 * np.sum(np.square(joint_delta)))
        root_delta = x[:3] - self.spec.qpos_reference[:3]
        joint_delta = x[3:] - self.spec.qpos_reference[self.spec.joint_qpos_indices]
        return float(
            np.sum(np.square(root_delta / np.asarray((0.08, 0.08, 0.16))))
            + 0.15 * np.sum(np.square(joint_delta))
        )

    def hard_constraints(
        self, x: np.ndarray, foot_anchor_tolerance_m: float, support_max_gap_m: float
    ) -> np.ndarray:
        """Inequality residuals; every component must be non-negative."""
        self.forward(x)
        non_support_distances = _pair_distances(
            self.model, self.data,
            self.spec.non_support_geoms, self.spec.chair_geoms,
        )
        support_nonseat_distances = _pair_distances(
            self.model, self.data,
            self.spec.support_geoms, self.spec.nonseat_chair_geoms,
        )
        foot_delta = self.data.site_xpos[self.spec.foot_site_ids] - self.spec.foot_target
        support_distance = _minimum_distance(
            self.model, self.data, self.spec.support_geoms, [self.spec.seat_geom]
        )
        return np.concatenate((
            non_support_distances - self.spec.clearance_m,
            support_nonseat_distances - self.spec.clearance_m,
            np.square(foot_anchor_tolerance_m) - np.square(foot_delta).reshape(-1),
            np.asarray((
                support_distance - self.spec.support_gap_m,
                support_max_gap_m - support_distance,
            )),
        ))


def _joint_qpos_indices(model: mujoco.MjModel) -> np.ndarray:
    indices = []
    for joint_id in range(model.njnt):
        joint_type = model.jnt_type[joint_id]
        if joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            indices.append(int(model.jnt_qposadr[joint_id]))
    if len(indices) != model.nq - 7:
        raise ValueError("expected a free-base model with one scalar qpos per G1 joint")
    return np.asarray(indices, dtype=np.int32)


def _bounds(
    model: mujoco.MjModel,
    qpos_reference: np.ndarray,
    joint_indices: np.ndarray,
    *,
    freeze_root: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if freeze_root:
        lower = np.empty(len(joint_indices), dtype=np.float64)
        upper = np.empty_like(lower)
    else:
        lower = np.empty(3 + len(joint_indices), dtype=np.float64)
        upper = np.empty_like(lower)
        lower[:3] = qpos_reference[:3] + np.asarray((-0.20, -0.20, -0.20))
        upper[:3] = qpos_reference[:3] + np.asarray((0.20, 0.20, 0.25))
    joint_ranges: dict[int, tuple[float, float]] = {}
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            joint_ranges[int(model.jnt_qposadr[joint_id])] = tuple(float(value) for value in model.jnt_range[joint_id])
    joint_offset = 0 if freeze_root else 3
    for local_index, qpos_index in enumerate(joint_indices, start=joint_offset):
        lower[local_index], upper[local_index] = joint_ranges[int(qpos_index)]
    return lower, upper


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-xml", required=True, type=Path)
    parser.add_argument("--reference-key", type=int, default=0)
    parser.add_argument("--output-pose", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--clearance-m", type=float, default=0.003)
    parser.add_argument("--support-gap-m", type=float, default=0.003)
    parser.add_argument("--support-max-gap-m", type=float, default=0.010)
    parser.add_argument("--foot-anchor-tolerance-m", type=float, default=0.015)
    parser.add_argument("--max-nfev", type=int, default=220)
    parser.add_argument("--slsqp-maxiter", type=int, default=300)
    parser.add_argument(
        "--freeze-root",
        action="store_true",
        help=(
            "solve joint targets at the immutable GMR free-base pose. "
            "Use this for the physical-seat path; it forbids an offline root "
            "translation from being mistaken for contact control."
        ),
    )
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.task_xml))
    if not 0 <= args.reference_key < model.nkey:
        raise ValueError("reference key is outside task keyframes")
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    seat_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "seat_support_geom")
    if pelvis_id < 0 or seat_geom < 0:
        raise ValueError("task lacks G1 pelvis or semantic seat support")
    robot_geoms = _collision_geoms(model, _descendants(model, pelvis_id))
    chair_geoms = [
        geom_id for geom_id in range(model.ngeom)
        if _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id).startswith(CHAIR_PREFIXES)
    ]
    support_geoms: list[int] = []
    support_body_names: list[str] = []
    for body_name in SUPPORT_BODY_CANDIDATES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            continue
        body_geoms = _collision_geoms(model, {body_id})
        if body_geoms:
            support_geoms.extend(body_geoms)
            support_body_names.append(body_name)
    if not support_geoms:
        raise ValueError("task has no active collision geometry on the seated support bodies")
    support_geom_set = set(support_geoms)
    non_support_geoms = [geom_id for geom_id in robot_geoms if geom_id not in support_geom_set]
    nonseat_chair_geoms = [geom_id for geom_id in chair_geoms if geom_id != seat_geom]
    if not non_support_geoms or not nonseat_chair_geoms:
        raise ValueError("task must provide non-support robot and non-seat chair collision geometry")
    foot_site_ids = np.asarray([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in FOOT_SITES
    ], dtype=np.int32)
    if np.any(foot_site_ids < 0):
        raise ValueError("task lacks one or more G1 foot tracking sites")

    qpos_reference = model.key_qpos[args.reference_key].copy()
    initial_data = mujoco.MjData(model)
    initial_data.qpos[:] = qpos_reference
    mujoco.mj_forward(model, initial_data)
    joint_indices = _joint_qpos_indices(model)
    spec = ProblemSpec(
        qpos_reference=qpos_reference,
        joint_qpos_indices=joint_indices,
        freeze_root=bool(args.freeze_root),
        foot_site_ids=foot_site_ids,
        foot_target=initial_data.site_xpos[foot_site_ids].copy(),
        non_support_geoms=non_support_geoms,
        chair_geoms=chair_geoms,
        support_geoms=support_geoms,
        nonseat_chair_geoms=nonseat_chair_geoms,
        support_body_names=tuple(support_body_names),
        seat_geom=seat_geom,
        clearance_m=args.clearance_m,
        support_gap_m=args.support_gap_m,
    )
    problem = SeatedPoseProblem(model, spec)
    lower, upper = _bounds(
        model, qpos_reference, joint_indices, freeze_root=bool(args.freeze_root)
    )
    x_reference = (
        qpos_reference[joint_indices].copy()
        if args.freeze_root
        else np.concatenate((qpos_reference[:3], qpos_reference[joint_indices]))
    )
    seeds = [x_reference.copy()]
    if not args.freeze_root:
        for root_z_offset in (0.05, 0.10, 0.15):
            seed = x_reference.copy()
            seed[2] = np.clip(seed[2] + root_z_offset, lower[2], upper[2])
            seeds.append(seed)
    attempts = []
    best = None
    for seed_index, seed in enumerate(seeds):
        soft_result = least_squares(
            problem.residual, seed, bounds=(lower, upper), loss="soft_l1", f_scale=1.0,
            max_nfev=args.max_nfev, xtol=1e-5, ftol=1e-5, gtol=1e-5,
        )
        hard_result = minimize(
            problem.objective,
            soft_result.x,
            method="SLSQP",
            bounds=list(zip(lower, upper)),
            constraints=({
                "type": "ineq",
                "fun": lambda x: problem.hard_constraints(
                    x, args.foot_anchor_tolerance_m, args.support_max_gap_m
                ),
            },),
            options={"maxiter": args.slsqp_maxiter, "ftol": 1e-7, "disp": False},
        )
        metrics = problem.metrics(hard_result.x)
        constraint_minimum = float(np.min(problem.hard_constraints(
            hard_result.x, args.foot_anchor_tolerance_m, args.support_max_gap_m
        )))
        attempt = {
            "seed_index": seed_index,
            "soft_initialisation": {
                "success": bool(soft_result.success), "status": int(soft_result.status),
                "message": str(soft_result.message), "cost": float(soft_result.cost), "nfev": int(soft_result.nfev),
            },
            "hard_constrained_solve": {
                "success": bool(hard_result.success), "status": int(hard_result.status),
                "message": str(hard_result.message), "objective": float(hard_result.fun),
                "nfev": int(hard_result.nfev), "minimum_constraint_residual": constraint_minimum,
            },
            "metrics": metrics,
        }
        attempts.append(attempt)
        score = (
            constraint_minimum,
            min(
                metrics["minimum_non_support_chair_distance_m"],
                metrics["minimum_support_nonseat_distance_m"],
            ),
            -float(hard_result.fun),
        )
        if best is None or score > best[0]:
            best = (score, hard_result.x.copy(), attempt)
    assert best is not None
    _, best_x, best_attempt = best
    metrics = problem.metrics(best_x)
    accepted = bool(
        metrics["minimum_non_support_chair_distance_m"] >= args.clearance_m
        and metrics["minimum_support_nonseat_distance_m"] >= args.clearance_m
        and metrics["maximum_foot_anchor_error_m"] <= args.foot_anchor_tolerance_m
        and args.support_gap_m <= metrics["support_distance_to_seat_m"] <= args.support_max_gap_m
        and best_attempt["hard_constrained_solve"]["minimum_constraint_residual"] >= -1e-5
    )
    qpos_solution = problem.qpos_from_x(best_x)
    args.output_pose.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_pose,
        qpos=qpos_solution,
        source_task_xml=np.asarray(str(args.task_xml)),
        reference_key=np.asarray(args.reference_key, dtype=np.int32),
    )
    report = {
        "schema_version": 1,
        "purpose": "offline_g1_seated_terminal_pose_feasibility",
        "status": "accepted_static_terminal_pose" if accepted else "rejected_static_terminal_pose",
        "physical_contract": {
            "offline_operation": "target_pose_only_no_mj_step_or_external_force",
            "root_mode": "frozen_gmr_reference" if args.freeze_root else "legacy_free_root_diagnostic",
            "future_runtime_requirement": "initialise_once_then_joint_torque_plus_mj_step",
            "forbidden_runtime_mechanisms": ["root_state_reimposition", "xfrc_applied", "mocap_weld"],
        },
        "inputs": {"task_xml": str(args.task_xml), "task_xml_sha256": _sha256(args.task_xml), "reference_key": args.reference_key},
        "settings": {
            "clearance_m": args.clearance_m, "support_gap_m": args.support_gap_m,
            "support_max_gap_m": args.support_max_gap_m,
            "foot_anchor_tolerance_m": args.foot_anchor_tolerance_m,
            "max_nfev": args.max_nfev, "slsqp_maxiter": args.slsqp_maxiter,
        },
        "result": {"metrics": metrics, "best_attempt": best_attempt, "attempts": attempts},
        "support_collision_bodies": list(support_body_names),
        "output_pose": str(args.output_pose),
        "output_pose_sha256": _sha256(args.output_pose),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
