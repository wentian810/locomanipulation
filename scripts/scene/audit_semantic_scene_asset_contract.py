#!/usr/bin/env python3
"""Prove that Isaac and MuJoCo consume the same semantic chair boxes.

The check compares every world-space corner, rather than comparing Euler
angles or JSON field order.  It intentionally accepts only the compact
semantic representation; raw VideoMimic/NKSR meshes remain outside PHC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vector(value: str | list[float] | None, length: int, label: str) -> np.ndarray:
    values = np.asarray(
        str(value).split() if isinstance(value, str) else value,
        dtype=np.float64,
    ).reshape(-1)
    if values.shape != (length,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must contain {length} finite values")
    return values


def rotation_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        raise ValueError("zero quaternion")
    w, x, y, z = quat / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def corners(center: np.ndarray, rotation: np.ndarray, extents: np.ndarray) -> np.ndarray:
    signs = np.asarray(
        [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
        dtype=np.float64,
    )
    points = center[None, :] + (rotation @ (signs * extents).T).T
    return points[np.lexsort((points[:, 2], points[:, 1], points[:, 0]))]


def checked_box(name: str, center: np.ndarray, rotation: np.ndarray, extents: np.ndarray) -> dict[str, np.ndarray]:
    if not name:
        raise ValueError("semantic box requires a name")
    if np.any(extents <= 0.0):
        raise ValueError(f"{name} has non-positive box extents")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    return {"center": center, "rotation": rotation, "extents": extents, "corners": corners(center, rotation, extents)}


def load_primitives(path: Path) -> dict[str, dict[str, np.ndarray]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("frame") != "mujoco_world_z_up":
        raise RuntimeError(f"{path} is not in mujoco_world_z_up")
    boxes: dict[str, dict[str, np.ndarray]] = {}
    for primitive in payload.get("primitives", []):
        if not isinstance(primitive, dict) or primitive.get("type") != "box":
            raise RuntimeError(f"{path} includes a non-box primitive")
        name = str(primitive.get("name", ""))
        if name in boxes:
            raise RuntimeError(f"{path} repeats primitive name {name!r}")
        boxes[name] = checked_box(
            name,
            vector(primitive.get("center"), 3, f"{name}.center"),
            vector(primitive.get("rotation_matrix"), 9, f"{name}.rotation_matrix").reshape(3, 3),
            vector(primitive.get("extents"), 3, f"{name}.extents"),
        )
    if not boxes:
        raise RuntimeError(f"{path} has no primitives")
    return boxes


def load_mujoco_boxes(path: Path) -> dict[str, dict[str, np.ndarray]]:
    root = ET.parse(path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError(f"{path} has no worldbody")
    boxes: dict[str, dict[str, np.ndarray]] = {}

    def visit(element: ET.Element, position: np.ndarray, rotation: np.ndarray, body_name: str) -> None:
        for child in element:
            if child.tag == "body":
                local_position = vector(child.get("pos", "0 0 0"), 3, "body.pos")
                local_rotation = rotation_from_quat_wxyz(vector(child.get("quat", "1 0 0 0"), 4, "body.quat"))
                visit(
                    child,
                    position + rotation @ local_position,
                    rotation @ local_rotation,
                    child.get("name", body_name),
                )
            elif child.tag == "geom" and child.get("type", "sphere") == "box":
                geom_name = child.get("name", "")
                name = geom_name[:-5] if geom_name.endswith("_geom") else body_name
                if name in boxes:
                    raise RuntimeError(f"{path} repeats MuJoCo box name {name!r}")
                local_position = vector(child.get("pos", "0 0 0"), 3, f"{name}.geom.pos")
                local_rotation = rotation_from_quat_wxyz(vector(child.get("quat", "1 0 0 0"), 4, f"{name}.geom.quat"))
                boxes[name] = checked_box(
                    name,
                    position + rotation @ local_position,
                    rotation @ local_rotation,
                    # MuJoCo's ``size`` for a box is its half extent, whereas
                    # the packaged semantic JSON intentionally stores the full
                    # edge lengths as ``extents``.
                    2.0 * vector(child.get("size"), 3, f"{name}.geom.size"),
                )

    visit(worldbody, np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64), "")
    if not boxes:
        raise RuntimeError(f"{path} has no box geoms")
    return boxes


def compare(reference: dict[str, dict[str, np.ndarray]], candidate: dict[str, dict[str, np.ndarray]]) -> tuple[list[str], float]:
    if set(reference) != set(candidate):
        return sorted(set(reference).symmetric_difference(candidate)), float("inf")
    error = 0.0
    for name in reference:
        error = max(error, float(np.max(np.abs(reference[name]["corners"] - candidate[name]["corners"]))))
    return [], error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mujoco-primitives", required=True, type=Path)
    parser.add_argument("--mujoco-xml", required=True, type=Path)
    parser.add_argument("--isaac-primitives", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-world-vertex-error-m", type=float, default=1e-6)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite asset contract: {args.output}")
    if args.max_world_vertex_error_m < 0.0:
        raise ValueError("max-world-vertex-error-m must be non-negative")

    mujoco_primitives = load_primitives(args.mujoco_primitives)
    mujoco_xml = load_mujoco_boxes(args.mujoco_xml)
    isaac_primitives = load_primitives(args.isaac_primitives)
    xml_name_error, xml_vertex_error = compare(mujoco_primitives, mujoco_xml)
    isaac_name_error, isaac_vertex_error = compare(mujoco_primitives, isaac_primitives)
    max_error = max(xml_vertex_error, isaac_vertex_error)
    accepted = not xml_name_error and not isaac_name_error and max_error <= args.max_world_vertex_error_m
    result = {
        "schema_version": 1,
        "status": "accepted_world_space_equivalent" if accepted else "rejected_world_space_mismatch",
        "frame": "mujoco_world_z_up",
        "collision_representation": "identical_named_semantic_boxes",
        "requires_identical_names": True,
        "mujoco_primitives_sha256": sha256(args.mujoco_primitives),
        "mujoco_xml_sha256": sha256(args.mujoco_xml),
        "isaac_primitives_sha256": sha256(args.isaac_primitives),
        "box_names": sorted(mujoco_primitives),
        "mujoco_xml_name_mismatch": xml_name_error,
        "isaac_name_mismatch": isaac_name_error,
        "mujoco_xml_max_world_vertex_error_m": xml_vertex_error,
        "isaac_max_world_vertex_error_m": isaac_vertex_error,
        "max_world_vertex_error_m": max_error,
        "threshold_m": float(args.max_world_vertex_error_m),
        "inputs": {
            "mujoco_primitives": str(args.mujoco_primitives.resolve()),
            "mujoco_xml": str(args.mujoco_xml.resolve()),
            "isaac_primitives": str(args.isaac_primitives.resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not accepted:
        raise SystemExit("semantic chair asset contract rejected")


if __name__ == "__main__":
    main()
