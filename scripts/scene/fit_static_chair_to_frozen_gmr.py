#!/usr/bin/env python3
"""Fit one static semantic-chair transform to an immutable GMR trajectory.

This is deliberately a scene-side calibration, not a robot retargeting pass.
It never writes the input motion and has only four fitted degrees of freedom:
chair translation (XYZ) and yaw.  A robust contact objective and explicit
quality gates make it safe to use in batch processing: clips without a stable
static fit are rejected instead of being "repaired" frame by frame.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

from evaluate_gmr_chair_contacts import (
    _body_name,
    _combine_mjcf,
    _load_motion,
    _robot_chair_contact_geoms,
    _robot_collision_geoms,
)


_AUTO_SUPPORT_BODIES = (
    # The GMR-provided custom collision asset represents the load-bearing
    # upper thighs on the hip-yaw links.  Keep the historical bodies too so
    # primitive-proxy profiles remain reproducible.
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "pelvis",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_vec(value: str) -> np.ndarray:
    return np.asarray([float(item) for item in value.split()], dtype=np.float64)


def _format(values: np.ndarray) -> str:
    return " ".join(f"{float(item):.9g}" for item in values)


def _quat_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    w, x, y, z = first
    W, X, Y, Z = second
    return np.asarray(
        [
            w * W - x * X - y * Y - z * Z,
            w * X + x * W + y * Z - z * Y,
            w * Y - x * Z + y * W + z * X,
            w * Z + x * Y - y * X + z * W,
        ],
        dtype=np.float64,
    )


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    quat /= np.linalg.norm(quat)
    w, x, y, z = quat
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quat = np.asarray(
            [0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale,
             (matrix[0, 2] - matrix[2, 0]) / scale, (matrix[1, 0] - matrix[0, 1]) / scale]
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            quat = np.asarray([(matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                               (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale])
        elif axis == 1:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            quat = np.asarray([(matrix[0, 2] - matrix[2, 0]) / scale, (matrix[0, 1] + matrix[1, 0]) / scale,
                               0.25 * scale, (matrix[1, 2] + matrix[2, 1]) / scale])
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            quat = np.asarray([(matrix[1, 0] - matrix[0, 1]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale,
                               (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale])
    return quat / np.linalg.norm(quat)


def _yaw_matrix(yaw: float) -> np.ndarray:
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.asarray([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def _huber(value: np.ndarray, delta: float) -> np.ndarray:
    absolute = np.abs(value)
    return np.where(absolute <= delta, 0.5 * (value / delta) ** 2, absolute / delta - 0.5)


def _summary(values: np.ndarray, overlap_limit: float) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "minimum_m": float(np.min(values)),
        "p05_m": float(np.quantile(values, 0.05)),
        "median_m": float(np.median(values)),
        "p95_m": float(np.quantile(values, 0.95)),
        "maximum_m": float(np.max(values)),
        "overlap_ratio": float(np.mean(values < -overlap_limit)),
    }


def _sample(indices: np.ndarray, maximum: int) -> np.ndarray:
    if len(indices) <= maximum:
        return indices
    return np.unique(indices[np.linspace(0, len(indices) - 1, maximum).round().astype(int)])


def _render_mesh_geoms(model: mujoco.MjModel) -> list[int]:
    """Return one visible render mesh per robot body (not its collision duplicate)."""
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id < 0:
        raise ValueError("pelvis body missing from robot MJCF")
    result: list[int] = []
    for geom_id in range(model.ngeom):
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        if int(model.geom_group[geom_id]) != 1:
            continue
        body_id = int(model.geom_bodyid[geom_id])
        while body_id >= 0 and body_id != pelvis_id:
            parent = int(model.body_parentid[body_id])
            if parent == body_id:
                body_id = -1
                break
            body_id = parent
        if body_id == pelvis_id:
            result.append(geom_id)
    if not result:
        raise RuntimeError("no visible robot render meshes found under pelvis")
    return result


def _render_mesh_seat_clearance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mesh_geoms: list[int],
    seat_geom: int,
    footprint_margin_m: float,
) -> float:
    """Directional gap from visible robot mesh to the upward seat plane.

    A GMR link's temporary physics proxy is an AABB.  It is intentionally
    conservative and can extend materially below the rendered mesh after a
    link rotates.  For an immutable visual GMR branch, the scene should be
    fitted to the rendered support surface, while those boxes remain useful
    only as conservative numerical diagnostics.
    """
    seat_rotation = data.geom_xmat[seat_geom].reshape(3, 3)
    normal = seat_rotation[:, 2].copy()
    if normal[2] < 0.0:
        normal *= -1.0
    seat_center = data.geom_xpos[seat_geom]
    seat_top = seat_center + normal * float(model.geom_size[seat_geom, 2])
    axis_x, axis_y = seat_rotation[:, 0], seat_rotation[:, 1]
    half_x, half_y = model.geom_size[seat_geom, :2]
    candidates: list[float] = []
    for geom_id in mesh_geoms:
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            continue
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        local_vertices = model.mesh_vert[start:start + count]
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        vertices = local_vertices @ rotation.T + data.geom_xpos[geom_id]
        relative = vertices - seat_center
        over_seat = (
            (np.abs(relative @ axis_x) <= half_x + footprint_margin_m)
            & (np.abs(relative @ axis_y) <= half_y + footprint_margin_m)
        )
        if np.any(over_seat):
            candidates.append(float(np.min((vertices[over_seat] - seat_top) @ normal)))
    if not candidates:
        # During bounded scene-pose optimization, a trial chair can move its
        # footprint entirely away from the selected support link.  That is an
        # infeasible pose, not a malformed source asset; return an explicit
        # non-finite clearance so the objective can reject the trial.
        return float("inf")
    return min(candidates)


def _load_chair_primitives(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    primitives = payload.get("primitives", [])
    seat = next((item for item in primitives if item.get("name") in {"seat", "seat_support"}), None)
    if seat is None:
        raise ValueError(f"{path} has no seat or seat_support primitive")
    pivot = np.asarray(seat["center"], dtype=np.float64)
    if pivot.shape != (3,):
        raise ValueError("seat center must be a three-vector")
    return payload, pivot


def _inverse_chair_pose(
    root: np.ndarray, rotation_xyzw: np.ndarray, translation: np.ndarray, yaw: float, pivot: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rotation = _yaw_matrix(yaw)
    transformed_root = pivot + rotation.T @ (root - pivot - translation)
    inverse_yaw = np.asarray([np.cos(yaw / 2.0), 0.0, 0.0, -np.sin(yaw / 2.0)])
    transformed_quat = _quat_multiply(inverse_yaw, rotation_xyzw[[3, 0, 1, 2]])
    return transformed_root, transformed_quat / np.linalg.norm(transformed_quat)


def _apply_transform(point: np.ndarray, rotation: np.ndarray, translation: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    return pivot + rotation @ (point - pivot) + translation


def _transform_primitives(path: Path, rotation: np.ndarray, translation: np.ndarray, pivot: np.ndarray) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for primitive in payload.get("primitives", []):
        center = np.asarray(primitive["center"], dtype=np.float64)
        source_rotation = np.asarray(primitive.get("rotation_matrix"), dtype=np.float64)
        primitive["center"] = _apply_transform(center, rotation, translation, pivot).tolist()
        primitive["rotation_matrix"] = (rotation @ source_rotation).tolist()
        if "quat_wxyz" in primitive:
            primitive["quat_wxyz"] = _matrix_to_quat(rotation @ source_rotation).tolist()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _transform_mjcf(path: Path, body_names: set[str], rotation: np.ndarray, translation: np.ndarray, pivot: np.ndarray) -> None:
    tree = ET.parse(path)
    worldbody = tree.getroot().find("worldbody")
    if worldbody is None:
        raise ValueError(f"{path} has no worldbody")
    for body in worldbody.findall("body"):
        if body.get("name") not in body_names:
            continue
        position = _as_vec(body.get("pos", "0 0 0"))
        quat = _as_vec(body.get("quat", "1 0 0 0"))
        body.set("pos", _format(_apply_transform(position, rotation, translation, pivot)))
        body.set("quat", _format(_matrix_to_quat(rotation @ _quat_to_matrix(quat))))
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _transform_obj(path: Path, rotation: np.ndarray, translation: np.ndarray, pivot: np.ndarray) -> None:
    if not path.is_file():
        return
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            vertex = _apply_transform(_as_vec(line[2:]), rotation, translation, pivot)
            lines.append("v " + _format(vertex))
        else:
            lines.append(line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-motion", required=True, type=Path)
    parser.add_argument("--contact-anchors", required=True, type=Path)
    parser.add_argument("--robot-xml", required=True, type=Path)
    parser.add_argument("--source-scene-dir", required=True, type=Path)
    parser.add_argument("--output-scene-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--scene-mujoco-name", default="scene/scene_mujoco_semantic_chair.xml")
    parser.add_argument("--chair-primitives-name", default="scene/semantic_chair_primitives_mujoco.json")
    parser.add_argument("--support-body", default="auto", help="auto or one lower-body support body")
    parser.add_argument(
        "--support-surface", choices=("collision_proxy", "render_mesh"), default="collision_proxy",
        help="fit against conservative physics boxes or the visible mesh surface",
    )
    parser.add_argument(
        "--support-footprint-margin-m", type=float, default=0.01,
        help="seat-footprint margin used only for render-mesh support vertices",
    )
    parser.add_argument("--torso-body", default="torso_link")
    parser.add_argument("--seat-geom", default="seat_support_geom")
    parser.add_argument("--backrest-geom", default="backrest_geom")
    parser.add_argument("--seat-target-m", type=float, default=0.008)
    parser.add_argument("--back-target-m", type=float, default=0.025)
    parser.add_argument("--max-xy-m", type=float, default=0.30)
    parser.add_argument("--max-z-m", type=float, default=0.18)
    parser.add_argument("--max-yaw-deg", type=float, default=20.0)
    parser.add_argument("--max-fit-frames", type=int, default=32)
    parser.add_argument("--settled-tail-frames", type=int, default=20, help="fit the stable tail of the final sit-contact segment, not its descent")
    parser.add_argument("--max-lateral-shift-m", type=float, default=0.04, help="maximum static chair shift along its seat lateral axis")
    parser.add_argument("--seat-weight", type=float, default=4.0, help="relative seat-contact objective weight")
    parser.add_argument("--back-weight", type=float, default=1.0, help="relative backrest-clearance objective weight")
    parser.add_argument(
        "--approach-contact-weight", type=float, default=80.0,
        help=(
            "Penalty weight for robot-chair overlap sampled before the inferred "
            "sit segment. The final gate still checks every approach frame."
        ),
    )
    parser.add_argument(
        "--approach-max-frames", type=int, default=48,
        help="Maximum uniformly sampled pre-sit frames included in the fit objective.",
    )
    parser.add_argument("--maxiter", type=int, default=30)
    parser.add_argument("--max-soft-overlap-m", type=float, default=0.005)
    parser.add_argument("--max-overlap-ratio", type=float, default=0.15)
    parser.add_argument("--max-median-seat-gap-m", type=float, default=0.030)
    parser.add_argument("--full-body-collision-weight", type=float, default=2000.0,
        help="Objective weight for robot-vs-entire-chair penetrations on uniformly sampled full-clip frames.")
    parser.add_argument("--max-full-body-collision-frames", type=int, default=96,
        help="Maximum uniformly sampled full-clip frames used by the collision objective; the final gate remains dense.")
    parser.add_argument("--max-full-body-penetration-m", type=float, default=0.005,
        help="Maximum permitted penetration of any robot collision proxy into any semantic-chair collision geom.")
    parser.add_argument("--max-full-body-penetration-frame-ratio", type=float, default=0.0,
        help="Maximum fraction of all frames permitted to exceed max-full-body-penetration-m.")
    parser.add_argument(
        "--collision-proxy-strategy",
        choices=(
            "legacy_aabb", "hybrid_capsule", "visual_mesh",
            "custom_urdf_cylinders", "official_unitree_mjcf",
        ),
        default="legacy_aabb",
        help=(
            "G1-to-scene collision representation; visual_mesh uses the "
            "compiled visual-mesh convex hull for both fit and collision gate"
        ),
    )
    parser.add_argument(
        "--official-collision-mjcf", type=Path, default=None,
        help="revision-matched Unitree G1 MJCF used with official_unitree_mjcf",
    )
    args = parser.parse_args()
    if args.output_scene_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_scene_dir}")
    if args.max_fit_frames < 4 or args.settled_tail_frames < 4 or args.maxiter < 1:
        raise ValueError("max-fit-frames and settled-tail-frames must be at least 4 and maxiter positive")
    if args.max_lateral_shift_m <= 0.0:
        raise ValueError("max-lateral-shift-m must be positive")
    if args.support_footprint_margin_m < 0.0:
        raise ValueError("support-footprint-margin-m must be non-negative")
    if args.seat_weight <= 0.0 or args.back_weight < 0.0:
        raise ValueError("seat-weight must be positive and back-weight non-negative")
    if args.approach_contact_weight < 0.0 or args.approach_max_frames < 4:
        raise ValueError("approach contact weight must be non-negative and approach max frames at least four")
    if args.full_body_collision_weight < 0.0 or args.max_full_body_collision_frames < 4:
        raise ValueError("full-body collision weight must be non-negative and max frames at least four")
    if args.max_full_body_penetration_m < 0.0:
        raise ValueError("max-full-body-penetration-m must be non-negative")
    if not 0.0 <= args.max_full_body_penetration_frame_ratio <= 1.0:
        raise ValueError("max-full-body-penetration-frame-ratio must be in [0, 1]")
    if args.collision_proxy_strategy == "official_unitree_mjcf":
        if args.official_collision_mjcf is None:
            raise ValueError("official_unitree_mjcf requires --official-collision-mjcf")
        if not args.official_collision_mjcf.is_file():
            raise FileNotFoundError(args.official_collision_mjcf)

    source_xml = args.source_scene_dir / args.scene_mujoco_name
    primitives_path = args.source_scene_dir / args.chair_primitives_name
    motion_digest_before = _sha256(args.robot_motion)
    motion = _load_motion(args.robot_motion)
    root = np.asarray(motion["root_pos"], dtype=np.float64)
    root_rot = np.asarray(motion["root_rot"], dtype=np.float64)
    dof = np.asarray(motion["dof_pos"], dtype=np.float64)
    anchors = np.load(args.contact_anchors)
    sit_mask = np.asarray(anchors["sit_mask"], dtype=bool)
    back_mask = np.asarray(anchors.get("back_contact", np.zeros(len(root), dtype=bool)), dtype=bool) & sit_mask
    if sit_mask.shape != (len(root),) or back_mask.shape != (len(root),):
        raise ValueError("contact-anchor length does not match robot motion")
    sit_frames = np.flatnonzero(sit_mask)
    if len(sit_frames) < 4:
        raise RuntimeError("insufficient sit-contact evidence for static calibration")
    primitive_payload, pivot = _load_chair_primitives(primitives_path)
    seat_primitive = next(item for item in primitive_payload["primitives"] if item.get("name") in {"seat", "seat_support"})
    lateral_axis = np.asarray(seat_primitive["rotation_matrix"], dtype=np.float64)[:, 0]
    lateral_axis /= max(float(np.linalg.norm(lateral_axis)), 1e-12)

    combined = _combine_mjcf(
        args.robot_xml, source_xml,
        collision_proxy_strategy=args.collision_proxy_strategy,
        official_collision_mjcf=args.official_collision_mjcf,
    )
    try:
        model = mujoco.MjModel.from_xml_path(str(combined))
        data = mujoco.MjData(model)
        if model.nq != 7 + dof.shape[1]:
            raise ValueError("robot motion and XML DOF counts differ")
        seat = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, args.seat_geom)
        backrest = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, args.backrest_geom)
        if seat < 0 or backrest < 0:
            raise ValueError("semantic seat or backrest geom missing")
        proxy_groups: dict[str, list[int]] = {}
        for geom in _robot_chair_contact_geoms(model):
            proxy_groups.setdefault(_body_name(model, geom), []).append(geom)
        render_groups: dict[str, list[int]] = {}
        for geom in _render_mesh_geoms(model):
            render_groups.setdefault(_body_name(model, geom), []).append(geom)
        groups = render_groups if args.support_surface == "render_mesh" else proxy_groups
        if args.support_body == "auto":
            candidate_names = [name for name in _AUTO_SUPPORT_BODIES if name in groups]
        else:
            candidate_names = [args.support_body]
        if not candidate_names:
            raise RuntimeError(f"no requested {args.support_surface} support body exists")
        torso_geoms = [geom for geom in _robot_collision_geoms(model) if _body_name(model, geom) == args.torso_body]
        if not torso_geoms:
            raise ValueError(f"torso body {args.torso_body!r} has no collision proxy")
        chair_body_ids = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(item.get("name")))
            for item in primitive_payload.get("primitives", [])
        }
        chair_body_ids.discard(-1)
        chair_geom_ids = {
            geom_id for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) in chair_body_ids
        }
        robot_geom_ids = set(_robot_collision_geoms(model))
        settled_frame_count = min(len(sit_frames), args.settled_tail_frames)
        settled_frames = sit_frames[-settled_frame_count:]
        fit_frames = _sample(settled_frames, args.max_fit_frames)
        # A disabled backrest term must not quietly reject a seat-only fit.
        # When enabled, constrain only frames with independent back-contact
        # evidence rather than every final seated frame.
        fit_back = (
            set(fit_frames[back_mask[fit_frames]].tolist())
            if args.back_weight > 0.0 else set()
        )
        approach_frames = np.flatnonzero(~sit_mask)
        if len(approach_frames) > args.approach_max_frames:
            sample_ids = np.linspace(0, len(approach_frames) - 1, args.approach_max_frames)
            approach_objective_frames = np.unique(
                approach_frames[np.rint(sample_ids).astype(np.int64)]
            )
        else:
            approach_objective_frames = approach_frames
        full_body_objective_frames = _sample(
            np.arange(len(root), dtype=np.int64), args.max_full_body_collision_frames
        )

        def distances(vector: np.ndarray, support_geoms: list[int], frames: np.ndarray, back_frames: set[int]) -> tuple[np.ndarray, np.ndarray]:
            translation = np.asarray(vector[:3], dtype=np.float64)
            yaw = float(vector[3])
            support_values: list[float] = []
            back_values: list[float] = []
            for frame in frames:
                position, quat = _inverse_chair_pose(root[frame], root_rot[frame], translation, yaw, pivot)
                data.qpos[:3] = position
                data.qpos[3:7] = quat
                data.qpos[7:] = dof[frame]
                mujoco.mj_forward(model, data)
                if args.support_surface == "render_mesh":
                    support_values.append(_render_mesh_seat_clearance(
                        model, data, support_geoms, seat, args.support_footprint_margin_m,
                    ))
                else:
                    support_values.append(min(float(mujoco.mj_geomDistance(model, data, geom, seat, 2.0, None)) for geom in support_geoms))
                if int(frame) in back_frames:
                    back_values.append(min(float(mujoco.mj_geomDistance(model, data, geom, backrest, 2.0, None)) for geom in torso_geoms))
            return np.asarray(support_values), np.asarray(back_values)

        def approach_overlap(vector: np.ndarray, frames: np.ndarray) -> tuple[float, float]:
            """Return mean penetration and contact ratio for sampled approach frames."""
            translation = np.asarray(vector[:3], dtype=np.float64)
            yaw = float(vector[3])
            penetrations: list[float] = []
            contact_count = 0
            for frame in frames:
                position, quat = _inverse_chair_pose(root[frame], root_rot[frame], translation, yaw, pivot)
                data.qpos[:3] = position
                data.qpos[3:7] = quat
                data.qpos[7:] = dof[frame]
                mujoco.mj_forward(model, data)
                frame_penetration = 0.0
                for contact_id in range(data.ncon):
                    contact = data.contact[contact_id]
                    pair = {int(contact.geom1), int(contact.geom2)}
                    if pair & robot_geom_ids and pair & chair_geom_ids:
                        frame_penetration = max(frame_penetration, max(0.0, -float(contact.dist)))
                if frame_penetration > 0.0:
                    contact_count += 1
                penetrations.append(frame_penetration)
            if not penetrations:
                return 0.0, 0.0
            return float(np.mean(penetrations)), float(contact_count / len(penetrations))

        def full_body_penetrations(
            vector: np.ndarray, frames: np.ndarray, *, exact_distance: bool = False
        ) -> tuple[np.ndarray, dict[tuple[str, str], dict[str, Any]]]:
            """Measure robot-chair penetration, optionally with narrow-phase distance queries.

            The optimizer keeps the inexpensive generated-contact estimate.
            Promotion uses ``exact_distance``: an embedded geom may not appear
            in ``data.contact`` even when its signed geom distance is negative.
            """
            translation = np.asarray(vector[:3], dtype=np.float64)
            yaw = float(vector[3])
            values: list[float] = []
            details: dict[tuple[str, str], dict[str, Any]] = {}
            for frame in frames:
                position, quat = _inverse_chair_pose(root[frame], root_rot[frame], translation, yaw, pivot)
                data.qpos[:3] = position
                data.qpos[3:7] = quat
                data.qpos[7:] = dof[frame]
                mujoco.mj_forward(model, data)
                frame_penetration = 0.0
                pairs: list[tuple[int, int, float]] = []
                if exact_distance:
                    witness = np.empty(6, dtype=np.float64)
                    for robot_geom in robot_geom_ids:
                        for chair_geom in chair_geom_ids:
                            pairs.append((
                                robot_geom, chair_geom,
                                float(mujoco.mj_geomDistance(
                                    model, data, robot_geom, chair_geom, 2.0, witness
                                )),
                            ))
                else:
                    for contact_id in range(data.ncon):
                        contact = data.contact[contact_id]
                        first, second = int(contact.geom1), int(contact.geom2)
                        if first in robot_geom_ids and second in chair_geom_ids:
                            pairs.append((first, second, float(contact.dist)))
                        elif second in robot_geom_ids and first in chair_geom_ids:
                            pairs.append((second, first, float(contact.dist)))
                for robot_geom, chair_geom, distance in pairs:
                    if distance >= 0.0:
                        continue
                    frame_penetration = max(frame_penetration, -distance)
                    key = (_body_name(model, robot_geom), str(mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, chair_geom)))
                    item = details.setdefault(key, {
                        "minimum_distance_m": distance, "frames": [], "contact_count": 0,
                    })
                    item["minimum_distance_m"] = min(float(item["minimum_distance_m"]), distance)
                    item["contact_count"] = int(item["contact_count"]) + 1
                    if len(item["frames"]) < 32:
                        item["frames"].append(int(frame))
                values.append(frame_penetration)
            return np.asarray(values, dtype=np.float64), details

        bounds = [(-args.max_xy_m, args.max_xy_m), (-args.max_xy_m, args.max_xy_m), (-args.max_z_m, args.max_z_m), (-np.deg2rad(args.max_yaw_deg), np.deg2rad(args.max_yaw_deg))]
        candidates: list[tuple[float, str, np.ndarray]] = []
        for name in candidate_names:
            support_geoms = groups[name]
            def objective(vector: np.ndarray) -> float:
                lateral_shift = float(np.dot(np.asarray(vector[:3], dtype=np.float64), lateral_axis))
                if abs(lateral_shift) > args.max_lateral_shift_m:
                    return 1e4 + 1e4 * (abs(lateral_shift) - args.max_lateral_shift_m)
                support_values, back_values = distances(vector, support_geoms, fit_frames, fit_back)
                if not np.all(np.isfinite(support_values)) or not np.all(np.isfinite(back_values)):
                    return 1e6
                # A zero bound is an intentional locked degree of freedom.  Keep
                # the regularizer well-defined so batch scene fitting can request a
                # pure height correction without introducing numerical NaNs.
                regularizer_scale = np.maximum(
                    np.asarray([
                        args.max_xy_m, args.max_xy_m, args.max_z_m,
                        np.deg2rad(args.max_yaw_deg),
                    ]),
                    1e-6,
                )
                regularizer = 0.02 * np.sum((vector / regularizer_scale) ** 2)
                loss = float(args.seat_weight * np.mean(_huber(support_values - args.seat_target_m, 0.02)) + regularizer)
                if len(back_values):
                    loss += float(args.back_weight * np.mean(_huber(back_values - args.back_target_m, 0.02)))
                if args.approach_contact_weight > 0.0 and len(approach_objective_frames):
                    # The public quality gate rejects *every* pre-sit contact.
                    # A finite soft penalty here previously allowed the solver
                    # to trade a few early collisions for a smaller terminal
                    # seat gap, creating a candidate the final gate was certain
                    # to reject.  Evaluate the complete pre-sit interval and
                    # make any collision infeasible already in the optimizer.
                    penetration, contact_ratio = approach_overlap(vector, approach_frames)
                    if contact_ratio > 0.0:
                        return 1e6 + 1e6 * (penetration + contact_ratio)
                if args.full_body_collision_weight > 0.0:
                    penetrations, _ = full_body_penetrations(vector, full_body_objective_frames)
                    excess = np.maximum(penetrations - args.max_full_body_penetration_m, 0.0)
                    loss += float(args.full_body_collision_weight * (
                        np.mean(_huber(excess, 0.01)) + 0.01 * np.mean(excess > 0.0)
                    ))
                return loss
            result = differential_evolution(objective, bounds, seed=17, popsize=8, maxiter=args.maxiter, tol=2e-3, polish=True, workers=1)
            candidates.append((float(result.fun), name, np.asarray(result.x, dtype=np.float64)))
        _, support_name, solution = min(candidates, key=lambda item: item[0])
        settled_back = set(settled_frames[back_mask[settled_frames]].tolist()) if args.back_weight > 0.0 else set()
        support_values, back_values = distances(solution, groups[support_name], settled_frames, settled_back)
        approach_values, _ = full_body_penetrations(
            solution, approach_frames, exact_distance=True
        )
        approach_contact_frames = [
            int(frame) for frame, penetration in zip(approach_frames, approach_values)
            if penetration > 0.0
        ]
        full_body_values, full_body_pair_details = full_body_penetrations(
            solution, np.arange(len(root), dtype=np.int64), exact_distance=True
        )
    finally:
        combined.unlink(missing_ok=True)

    support_summary = _summary(support_values, args.max_soft_overlap_m)
    back_summary = _summary(back_values, args.max_soft_overlap_m) if len(back_values) else None
    full_body_violation_frames = np.flatnonzero(
        full_body_values > args.max_full_body_penetration_m
    ).astype(int)
    full_body_summary = {
        "evaluated_frames": int(len(full_body_values)),
        "maximum_penetration_m": float(np.max(full_body_values)) if len(full_body_values) else 0.0,
        "p95_penetration_m": float(np.quantile(full_body_values, 0.95)) if len(full_body_values) else 0.0,
        "penetration_frame_count": int(np.count_nonzero(full_body_values > 0.0)),
        "violation_frame_count": int(len(full_body_violation_frames)),
        "violation_frame_ratio": float(len(full_body_violation_frames) / len(full_body_values)) if len(full_body_values) else 0.0,
        "first_violation_frames": full_body_violation_frames[:32].tolist(),
        "pairs_by_max_penetration": [
            {"robot_body": body, "scene_geom": scene_geom, **detail}
            for (body, scene_geom), detail in sorted(
                full_body_pair_details.items(), key=lambda item: item[1]["minimum_distance_m"]
            )
        ],
    }
    accepted = bool(
        support_summary["p05_m"] >= -args.max_soft_overlap_m
        and support_summary["overlap_ratio"] <= args.max_overlap_ratio
        and support_summary["median_m"] <= args.max_median_seat_gap_m
        and (back_summary is None or back_summary["p05_m"] >= -args.max_soft_overlap_m)
        and not approach_contact_frames
        and full_body_summary["maximum_penetration_m"] <= args.max_full_body_penetration_m
        and full_body_summary["violation_frame_ratio"] <= args.max_full_body_penetration_frame_ratio
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "static_semantic_chair_fit_to_frozen_gmr",
        "status": "accepted" if accepted else "rejected_by_static_fit_quality_gate",
        "robot_motion": str(args.robot_motion),
        "robot_motion_sha256_before": motion_digest_before,
        "robot_motion_sha256_after": _sha256(args.robot_motion),
        "robot_motion_modified": False,
        "collision_proxy_strategy": args.collision_proxy_strategy,
        "contact_frames": int(len(sit_frames)),
        "back_contact_frames": int(back_mask.sum()),
        "fit_frame_policy": "stable_tail_of_final_sit_segment",
        "fit_frame_range": [int(settled_frames[0]), int(settled_frames[-1])],
        "back_fit_policy": "torso_to_backrest_over_same_stable_tail",
        "objective_weights": {"seat": args.seat_weight, "backrest": args.back_weight},
        "approach_objective": {
            "weight": args.approach_contact_weight,
            "sampled_frames": int(len(approach_objective_frames)),
            "total_pre_sit_frames": int(len(approach_frames)),
        },
        "approach_contact_gate": {
            "evaluated_frames": int(len(approach_frames)),
            "chair_contact_frames": approach_contact_frames,
            "passed": not bool(approach_contact_frames),
            "policy": "reject any actual MuJoCo robot-chair contact before sit_mask",
        },
        "full_body_collision_objective": {
            "weight": args.full_body_collision_weight,
            "sampled_frames": int(len(full_body_objective_frames)),
            "policy": "all robot collision proxies versus every semantic-chair collision geom",
        },
        "full_body_collision_gate": {
            **full_body_summary,
            "max_penetration_m": args.max_full_body_penetration_m,
            "max_penetration_frame_ratio": args.max_full_body_penetration_frame_ratio,
            "passed": bool(
                full_body_summary["maximum_penetration_m"] <= args.max_full_body_penetration_m
                and full_body_summary["violation_frame_ratio"] <= args.max_full_body_penetration_frame_ratio
            ),
        },
        "support_body": support_name,
        "support_surface": args.support_surface,
        "support_gap_definition": (
            "minimum visible-mesh clearance along the upward semantic-seat normal within its footprint"
            if args.support_surface == "render_mesh" else "MuJoCo signed distance to conservative physics proxy"
        ),
        "support_footprint_margin_m": args.support_footprint_margin_m if args.support_surface == "render_mesh" else None,
        "candidate_support_bodies": candidate_names,
        "transform": {"translation_xyz_m": solution[:3].tolist(), "yaw_deg": float(np.rad2deg(solution[3])), "pivot_xyz_m": pivot.tolist()},
        "seat_distance": support_summary,
        "backrest_distance": back_summary,
        "quality_gate": {"max_soft_overlap_m": args.max_soft_overlap_m, "max_overlap_ratio": args.max_overlap_ratio, "max_median_seat_gap_m": args.max_median_seat_gap_m},
        "collision_policy": (
            "Visible support mesh is fitted, while every collision proxy is an in-fit objective and a dense final promotion gate."
            if args.support_surface == "render_mesh" else
            "The selected load-bearing proxy is fitted, while every collision proxy is an in-fit objective and a dense final promotion gate."
        ),
    }
    if motion_digest_before != report["robot_motion_sha256_after"]:
        raise RuntimeError("input robot motion changed during calibration")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if not accepted:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    rotation = _yaw_matrix(float(solution[3]))
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_scene_dir.name}.stage-",
            dir=args.output_scene_dir.parent,
        )
    )
    try:
        shutil.copytree(args.source_scene_dir, stage, dirs_exist_ok=True)
        names = {str(item.get("name")) for item in primitive_payload.get("primitives", [])}
        for relative in (
            args.chair_primitives_name,
            "scene/primitives_mujoco.json",
            "scene/semantic_chair_primitives_phc.json",
        ):
            path = stage / relative
            if path.is_file():
                _transform_primitives(path, rotation, solution[:3], pivot)
        _transform_mjcf(stage / args.scene_mujoco_name, names, rotation, solution[:3], pivot)
        _transform_obj(stage / "scene/background_mesh_semantic_chair.obj", rotation, solution[:3], pivot)
        visual_sidecar = {
            "schema_version": 1,
            "purpose": "apply_static_chair_transform_to_separate_visual_chair_mesh_only",
            "transform": report["transform"],
            "note": "The package contains no chair-only GLB; do not transform review GLBs because they also contain the human.",
        }
        (stage / "scene/visual_chair_transform.json").write_text(json.dumps(visual_sidecar, indent=2) + "\n", encoding="utf-8")
        (stage / "static_chair_fit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        stage.replace(args.output_scene_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    report["output_scene_dir"] = str(args.output_scene_dir)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
