#!/usr/bin/env python3
"""Temporally couple GMR support-contact repairs without changing world XY.

This creates an auditable static candidate only.  It never makes a backrest
contact an equality constraint and never writes an output when dense geometry or
correction-smoothness checks fail.  A separate visual contract gate and mj_step
replay remain mandatory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize

from evaluate_gmr_chair_contacts import (
    _body_name,
    _combine_mjcf,
    _load_motion,
    _robot_chair_contact_geoms,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _set_qpos(data, root, rot, dof):
    data.qpos[:3] = root
    data.qpos[3:7] = rot[[3, 0, 1, 2]]
    data.qpos[7:] = dof


def _names(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _components(mask):
    ids = np.flatnonzero(mask)
    if not len(ids):
        return []
    return [part for part in np.split(ids, np.flatnonzero(np.diff(ids) > 1) + 1) if len(part)]


def _diff(n, order):
    return np.diff(np.eye(n), n=order, axis=0) if n > order else np.zeros((0, n))


def _smooth(target, mask, transition, fit, boundary, velocity, acceleration):
    """Quadratic episode solve: fit + first/second temporal differences."""
    out = np.zeros_like(target, dtype=np.float64)
    for component in _components(mask):
        lo = max(0, int(component[0]) - transition)
        hi = min(len(mask) - 1, int(component[-1]) + transition)
        frames = np.arange(lo, hi + 1)
        active = mask[frames]
        n = len(frames)
        w = np.zeros(n)
        w[active] = fit
        w[0] = max(w[0], boundary)
        w[-1] = max(w[-1], boundary)
        d1, d2 = _diff(n, 1), _diff(n, 2)
        lhs = np.diag(w) + 1e-9 * np.eye(n)
        if len(d1):
            lhs += velocity * d1.T @ d1
        if len(d2):
            lhs += acceleration * d2.T @ d2
        for c in range(target.shape[1]):
            rhs_target = target[frames, c].copy()
            rhs_target[~active] = 0.0
            rhs_target[0] = rhs_target[-1] = 0.0
            out[frames, c] = np.linalg.solve(lhs, w * rhs_target)
    return out


def _metrics(correction, fps):
    root = correction[:, 0]
    joints = correction[:, 1:]
    def peak(a):
        return float(np.max(np.abs(a))) if a.size else 0.0
    return {
        "max_root_z_correction_m": peak(root),
        "max_joint_correction_rad": peak(joints),
        "max_root_z_speed_mps": peak(np.diff(root) * fps),
        "max_root_z_acceleration_mps2": peak(np.diff(root, n=2) * fps * fps),
        "max_joint_speed_radps": peak(np.diff(joints, axis=0) * fps),
        "max_joint_acceleration_radps2": peak(np.diff(joints, n=2, axis=0) * fps * fps),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robot-motion", required=True, type=Path)
    p.add_argument("--scene-mujoco-xml", required=True, type=Path)
    p.add_argument("--robot-xml", required=True, type=Path)
    p.add_argument("--contact-anchors", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--report", required=True, type=Path)
    p.add_argument("--support-geom", default="seat_support_geom")
    p.add_argument("--support-mask-key", default="sit_mask")
    p.add_argument("--support-body", default="right_hip_pitch_link")
    p.add_argument("--repair-joints", default=(
        "left_hip_pitch_joint,left_hip_roll_joint,left_hip_yaw_joint,left_knee_joint,"
        "right_hip_pitch_joint,right_hip_roll_joint,right_hip_yaw_joint,right_knee_joint"
    ))
    p.add_argument("--clearance-m", type=float, default=0.003)
    p.add_argument("--support-target-m", type=float, default=0.008)
    p.add_argument("--activation-distance-m", type=float, default=0.25)
    p.add_argument("--min-root-dz-m", type=float, default=-0.08)
    p.add_argument("--max-root-dz-m", type=float, default=0.08)
    p.add_argument("--max-joint-correction-rad", type=float, default=0.35)
    p.add_argument("--transition-seconds", type=float, default=0.20)
    p.add_argument("--outer-iterations", type=int, default=3)
    p.add_argument("--local-maxiter", type=int, default=160)
    p.add_argument("--fit-weight", type=float, default=1.0)
    p.add_argument("--boundary-weight", type=float, default=100.0)
    p.add_argument("--velocity-weight", type=float, default=5.0)
    p.add_argument("--acceleration-weight", type=float, default=30.0)
    p.add_argument("--template-weight", type=float, default=10.0)
    p.add_argument("--support-weight", type=float, default=4.0)
    p.add_argument("--max-root-z-speed-mps", type=float, default=0.60)
    p.add_argument("--max-root-z-acceleration-mps2", type=float, default=8.0)
    p.add_argument("--max-joint-speed-radps", type=float, default=4.0)
    p.add_argument("--max-joint-acceleration-radps2", type=float, default=60.0)
    a = p.parse_args()
    if a.clearance_m < 0 or a.support_target_m < a.clearance_m:
        raise ValueError("support target must be at least clearance")
    if a.min_root_dz_m > a.max_root_dz_m or a.max_joint_correction_rad < 0:
        raise ValueError("invalid correction bounds")
    if a.outer_iterations < 1 or a.local_maxiter < 1:
        raise ValueError("iteration counts must be positive")

    report = {
        "schema_version": 1,
        "purpose": "temporally_coupled_gmr_support_contact_candidate",
        "status": "rejected_before_optimization",
        "policy": {
            "root_xy_immutable": True,
            "root_orientation_immutable": True,
            "backrest_objective": "none",
            "source_motion_immutable": True,
            "requires_contract_and_mj_step_gates": True,
        },
        "inputs": {
            "robot_motion": str(a.robot_motion),
            "robot_motion_sha256": _sha256(a.robot_motion),
            "scene_mujoco_xml": str(a.scene_mujoco_xml),
            "anchors": str(a.contact_anchors),
            "support_geom": a.support_geom,
            "support_mask_key": a.support_mask_key,
        },
    }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    combined = None
    try:
        motion = _load_motion(a.robot_motion)
        root = np.asarray(motion["root_pos"], dtype=np.float64)
        rot = np.asarray(motion["root_rot"], dtype=np.float64)
        dof = np.asarray(motion["dof_pos"], dtype=np.float64)
        if root.shape != (len(root), 3) or rot.shape != (len(root), 4) or dof.shape[0] != len(root):
            raise ValueError("incompatible robot-motion array shapes")
        fps = float(np.asarray(motion.get("fps", 30.0)).reshape(-1)[0])
        anchors = np.load(a.contact_anchors)
        if a.support_mask_key not in anchors:
            raise ValueError("missing support mask " + a.support_mask_key)
        mask = np.asarray(anchors[a.support_mask_key], dtype=bool)
        if mask.shape != (len(root),) or not np.any(mask):
            raise ValueError("support mask does not select robot frames")
        transition = round(a.transition_seconds * fps)
        report["support_episode"] = {
            "frame_count": int(mask.sum()),
            "components": [[int(c[0]), int(c[-1]), int(len(c))] for c in _components(mask)],
            "transition_frames": int(transition),
            "fps": fps,
        }

        combined = _combine_mjcf(a.robot_xml, a.scene_mujoco_xml)
        model = mujoco.MjModel.from_xml_path(str(combined))
        data = mujoco.MjData(model)
        if model.nq != 7 + dof.shape[1]:
            raise ValueError("GMR qpos and MJCF differ")
        support_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, a.support_geom)
        if support_id < 0:
            raise ValueError("support geom absent: " + a.support_geom)
        dof_names = [str(x) for x in motion.get("dof_names", [])]
        repair = []
        for name in _names(a.repair_joints):
            if name not in dof_names:
                raise ValueError("repair joint absent: " + name)
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            index = dof_names.index(name)
            qadr = int(model.jnt_qposadr[jid]) if jid >= 0 else -1
            if jid < 0 or qadr != 7 + index:
                raise ValueError("invalid repair joint mapping: " + name)
            repair.append((name, index, jid, qadr))
        all_geoms = _robot_chair_contact_geoms(model)
        support_geoms = [g for g in all_geoms if _body_name(model, g) == a.support_body]
        if not support_geoms:
            raise ValueError("support body lacks collision proxy: " + a.support_body)
        scales = np.array([0.05] + [0.35] * len(repair))

        def configure(frame, x):
            _set_qpos(data, root[frame], rot[frame], dof[frame])
            data.qpos[2] += x[0]
            for delta, (_, _, _, qadr) in zip(x[1:], repair):
                data.qpos[qadr] += delta
            mujoco.mj_forward(model, data)

        def geom_distance(g):
            return float(mujoco.mj_geomDistance(model, data, g, int(support_id), 2.0, None))

        def local_solve(frame, template, previous, fast=False):
            configure(frame, np.zeros(len(scales)))
            active = sorted(set(
                [g for g in all_geoms if geom_distance(g) <= a.activation_distance_m] + support_geoms
            ))
            lower, upper = [a.min_root_dz_m], [a.max_root_dz_m]
            for _, index, jid, _ in repair:
                lower.append(max(-a.max_joint_correction_rad, float(model.jnt_range[jid, 0] - dof[frame, index])))
                upper.append(min(a.max_joint_correction_rad, float(model.jnt_range[jid, 1] - dof[frame, index])))
            lower, upper = np.asarray(lower), np.asarray(upper)

            def constraint(x):
                configure(frame, x)
                return np.array([geom_distance(g) for g in active]) - a.clearance_m

            def cost(x):
                configure(frame, x)
                d = min(geom_distance(g) for g in support_geoms)
                return float(
                    np.dot(x / scales, x / scales)
                    + a.template_weight * np.dot((x - template) / scales, (x - template) / scales)
                    + 0.25 * np.dot((x - previous) / scales, (x - previous) / scales)
                    + a.support_weight * ((d - a.support_target_m) / max(a.support_target_m, 1e-4)) ** 2
                )

            zero = np.zeros(len(scales))
            down, up = zero.copy(), zero.copy()
            down[0], up[0] = max(a.min_root_dz_m, -0.04), min(a.max_root_dz_m, 0.04)
            found = []
            seeds = (template, previous) if fast else (template, previous, zero, down, up)
            for seed in seeds:
                result = minimize(
                    cost, np.clip(seed, lower, upper), method="SLSQP",
                    bounds=list(zip(lower, upper)),
                    constraints={"type": "ineq", "fun": constraint},
                    options={"maxiter": a.local_maxiter, "ftol": 1e-5, "disp": False},
                )
                x = np.asarray(result.x)
                if np.min(constraint(x)) >= -1e-4:
                    found.append((cost(x), x, result))
            if not found:
                configure(frame, np.clip(template, lower, upper))
                raise RuntimeError("no bounded local projection at frame %d (min %.5f m)" % (
                    frame, min(geom_distance(g) for g in active)
                ))
            _, x, r = min(found, key=lambda item: item[0])
            configure(frame, x)
            return x, {
                "frame": int(frame),
                "minimum_distance_m": float(min(geom_distance(g) for g in active)),
                "active_geom_count": len(active),
                "optimizer_success": bool(r.success),
                "root_dz_m": float(x[0]),
                "max_abs_joint_delta_rad": float(np.max(np.abs(x[1:]))) if len(x) > 1 else 0.0,
            }

        ids = np.flatnonzero(mask)
        targets = np.zeros((len(root), len(scales)))
        initial = []
        prev = np.zeros(len(scales))
        for f in ids:
            targets[f], detail = local_solve(int(f), np.zeros(len(scales)), prev)
            initial.append(detail)
            prev = targets[f]
        proposal = _smooth(
            targets, mask, transition, a.fit_weight, a.boundary_weight,
            a.velocity_weight, a.acceleration_weight,
        )
        rounds = []
        for iteration in range(a.outer_iterations):
            projected = np.zeros_like(proposal)
            details, prev = [], np.zeros(len(scales))
            for f in ids:
                projected[f], detail = local_solve(int(f), proposal[f], prev, fast=True)
                details.append(detail)
                prev = projected[f]
            proposal = _smooth(
                projected, mask, transition, a.fit_weight, a.boundary_weight,
                a.velocity_weight, a.acceleration_weight,
            )
            rounds.append({"iteration": iteration + 1, "local_projection": details, "metrics": _metrics(proposal, fps)})

        signed = np.full(len(root), np.nan)
        for f in ids:
            configure(int(f), proposal[f])
            signed[f] = min(geom_distance(g) for g in all_geoms)
        metric = _metrics(proposal, fps)
        temporal_ok = (
            metric["max_root_z_speed_mps"] <= a.max_root_z_speed_mps
            and metric["max_root_z_acceleration_mps2"] <= a.max_root_z_acceleration_mps2
            and metric["max_joint_speed_radps"] <= a.max_joint_speed_radps
            and metric["max_joint_acceleration_radps2"] <= a.max_joint_acceleration_radps2
        )
        clearance_ok = bool(np.all(signed[mask] >= a.clearance_m - 1e-4))
        report["optimization"] = {
            "channels": ["root_z"] + [r[0] for r in repair],
            "initial_local_projections": initial,
            "outer_rounds": rounds,
            "temporal_metrics": metric,
            "temporal_limits": {
                "root_z_speed_mps": a.max_root_z_speed_mps,
                "root_z_acceleration_mps2": a.max_root_z_acceleration_mps2,
                "joint_speed_radps": a.max_joint_speed_radps,
                "joint_acceleration_radps2": a.max_joint_acceleration_radps2,
            },
        }
        report["dense_support_audit"] = {
            "minimum_distance_m": float(np.min(signed[mask])),
            "median_distance_m": float(np.median(signed[mask])),
            "penetration_frame_ratio": float(np.mean(signed[mask] < -1e-5)),
            "clearance_m": a.clearance_m,
        }
        if not clearance_ok:
            report["status"] = "rejected_after_dense_support_audit"
        elif not temporal_ok:
            report["status"] = "rejected_temporal_smoothness_bound"
        else:
            out = dict(motion)
            root_out, dof_out = root.copy(), dof.copy()
            root_out[:, 2] += proposal[:, 0]
            for channel, (_, index, jid, _) in enumerate(repair, 1):
                dof_out[:, index] = np.clip(
                    dof_out[:, index] + proposal[:, channel],
                    model.jnt_range[jid, 0], model.jnt_range[jid, 1],
                )
            if not np.array_equal(root_out[:, :2], root[:, :2]):
                raise RuntimeError("internal root XY invariant failed")
            out["root_pos"] = root_out.astype(np.asarray(motion["root_pos"]).dtype)
            out["dof_pos"] = dof_out.astype(np.asarray(motion["dof_pos"]).dtype)
            out["temporal_support_contact_candidate"] = report
            a.output.parent.mkdir(parents=True, exist_ok=True)
            with a.output.open("wb") as f:
                pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
            report["status"] = "accepted_static_temporal_candidate"
            report["next_gate"] = "Run immutable visual/scene contract audit then mj_step replay."
    except Exception as exc:
        report["error"] = str(exc)
    finally:
        if combined is not None:
            combined.unlink(missing_ok=True)
        a.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

