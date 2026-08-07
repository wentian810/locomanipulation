"""Convert a VideoMimic collision scene into the PHC/GMR MuJoCo frame.

The reconstruction mesh is expressed in VideoMimic's gravity-calibrated
world.  PHC and GMR consume GVHMR motion after their Y-up -> Z-up conversion.
The rigid transform recorded by Phase 2 is therefore applied to *scene*
geometry, while the canonical GVHMR human motion remains untouched.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def _fmt(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.8g}" for value in values)


def _rotation_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [0.25 * scale, (rotation[2, 1] - rotation[1, 2]) / scale,
             (rotation[0, 2] - rotation[2, 0]) / scale,
             (rotation[1, 0] - rotation[0, 1]) / scale],
            dtype=np.float64,
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = math.sqrt(max(0.0, 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])) * 2.0
            quat = np.array(
                [(rotation[2, 1] - rotation[1, 2]) / scale, 0.25 * scale,
                 (rotation[0, 1] + rotation[1, 0]) / scale,
                 (rotation[0, 2] + rotation[2, 0]) / scale],
                dtype=np.float64,
            )
        elif index == 1:
            scale = math.sqrt(max(0.0, 1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])) * 2.0
            quat = np.array(
                [(rotation[0, 2] - rotation[2, 0]) / scale,
                 (rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale,
                 (rotation[1, 2] + rotation[2, 1]) / scale],
                dtype=np.float64,
            )
        else:
            scale = math.sqrt(max(0.0, 1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])) * 2.0
            quat = np.array(
                [(rotation[1, 0] - rotation[0, 1]) / scale,
                 (rotation[0, 2] + rotation[2, 0]) / scale,
                 (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale],
                dtype=np.float64,
            )
    return quat / np.linalg.norm(quat)


def _rotation_to_rpy(rotation: np.ndarray) -> np.ndarray:
    horizontal = math.hypot(float(rotation[0, 0]), float(rotation[1, 0]))
    if horizontal > 1e-8:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        pitch = math.atan2(-float(rotation[2, 0]), horizontal)
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        roll = math.atan2(-float(rotation[1, 2]), float(rotation[1, 1]))
        pitch = math.atan2(-float(rotation[2, 0]), horizontal)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _transform_obj(input_path: Path, output_path: Path, transform: np.ndarray) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vertices = 0
    faces = 0
    with input_path.open("r", encoding="utf-8", errors="strict") as source, output_path.open("w", encoding="utf-8") as target:
        for line in source:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) < 4:
                    raise ValueError(f"invalid OBJ vertex: {line.rstrip()}")
                point = np.array([float(parts[1]), float(parts[2]), float(parts[3]), 1.0], dtype=np.float64)
                transformed = transform @ point
                target.write("v " + _fmt(transformed[:3]) + "\n")
                vertices += 1
            else:
                target.write(line)
                if line.startswith("f "):
                    faces += 1
    if vertices == 0 or faces == 0:
        raise ValueError(f"OBJ has no usable triangle geometry: {input_path}")
    return vertices, faces


def _transform_primitives(source: dict, transform: np.ndarray) -> list[dict]:
    linear = transform[:3, :3]
    scale = float(abs(np.linalg.det(linear)) ** (1.0 / 3.0))
    if not 0.90 <= scale <= 1.10:
        raise ValueError(f"unexpected scene similarity scale for simulation assets: {scale}")
    rotation = linear / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4):
        raise ValueError("scene alignment linear block is not a rigid similarity")
    result: list[dict] = []
    for item in source.get("primitives", []):
        if item.get("type") != "box":
            raise ValueError(f"unsupported collision primitive type: {item.get('type')!r}")
        source_transform = np.asarray(item.get("transform"), dtype=np.float64)
        if source_transform.shape != (4, 4):
            raise ValueError(f"invalid primitive transform for {item.get('name', '<unnamed>')}")
        transformed = transform @ source_transform
        item_rotation = transformed[:3, :3] / scale
        extents = np.asarray(item.get("extents"), dtype=np.float64).reshape(3) * scale
        result.append(
            {
                "name": str(item.get("name", "box")),
                "type": "box",
                "frame": "mujoco_world_z_up",
                "source": str(item.get("source", "videomimic_collision")),
                "center": transformed[:3, 3].tolist(),
                "extents": extents.tolist(),
                "rotation_matrix": item_rotation.tolist(),
                "quat_wxyz": _rotation_to_quat_wxyz(item_rotation).tolist(),
                "rpy": _rotation_to_rpy(item_rotation).tolist(),
                "confidence": item.get("confidence", "warn"),
                "limitations": item.get("limitations", []),
            }
        )
    return result


def _write_mujoco(path: Path, collision_mesh_name: str, primitives: list[dict], visual_mesh_name: str | None = None) -> None:
    root = ET.Element("mujoco", {"model": "contact_aware_scene_mujoco"})
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", {"name": "collision_scene", "file": collision_mesh_name})
    if visual_mesh_name:
        ET.SubElement(asset, "mesh", {"name": "visual_scene", "file": visual_mesh_name})
    worldbody = ET.SubElement(root, "worldbody")
    if visual_mesh_name:
        ET.SubElement(worldbody, "geom", {"name": "visual_scene", "type": "mesh", "mesh": "visual_scene", "contype": "0", "conaffinity": "0", "group": "0", "rgba": "0.72 0.75 0.80 1"})
    ET.SubElement(worldbody, "geom", {"name": "collision_scene", "type": "mesh", "mesh": "collision_scene", "contype": "1", "conaffinity": "1", "group": "3", "rgba": "0.4 0.4 0.4 0"})
    for primitive in primitives:
        body = ET.SubElement(worldbody, "body", {"name": primitive["name"], "pos": _fmt(np.asarray(primitive["center"])), "quat": _fmt(np.asarray(primitive["quat_wxyz"]))})
        ET.SubElement(body, "geom", {"type": "box", "size": _fmt(0.5 * np.asarray(primitive["extents"])), "contype": "1", "conaffinity": "1"})
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="unicode")


def _write_urdf(path: Path, mesh_name: str, primitives: list[dict]) -> None:
    root = ET.Element("robot", {"name": "contact_aware_scene_mujoco"})
    mesh_link = ET.SubElement(root, "link", {"name": "residual_scene"})
    collision = ET.SubElement(mesh_link, "collision")
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "mesh", {"filename": mesh_name, "scale": "1 1 1"})
    for primitive in primitives:
        link = ET.SubElement(root, "link", {"name": primitive["name"]})
        collision = ET.SubElement(link, "collision")
        ET.SubElement(collision, "origin", {"xyz": _fmt(np.asarray(primitive["center"])), "rpy": _fmt(np.asarray(primitive["rpy"]))})
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry, "box", {"size": _fmt(np.asarray(primitive["extents"]))})
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="unicode", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mesh", required=True, type=Path)
    parser.add_argument("--input-primitives", required=True, type=Path)
    parser.add_argument("--input-visual-mesh", type=Path, default=None)
    parser.add_argument("--input-evidence-mesh", type=Path, default=None)
    parser.add_argument("--alignment-npz", required=True, type=Path)
    parser.add_argument("--output-mesh", required=True, type=Path)
    parser.add_argument("--output-visual-mesh", type=Path, default=None)
    parser.add_argument("--output-evidence-mesh", type=Path, default=None)
    parser.add_argument("--output-mujoco", required=True, type=Path)
    parser.add_argument("--output-urdf", required=True, type=Path)
    parser.add_argument("--output-primitives", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    args = parser.parse_args()

    if (args.input_visual_mesh is None) != (args.output_visual_mesh is None):
        raise ValueError("input-visual-mesh and output-visual-mesh must be provided together")
    if (args.input_evidence_mesh is None) != (args.output_evidence_mesh is None):
        raise ValueError("input-evidence-mesh and output-evidence-mesh must be provided together")
    alignment = np.load(args.alignment_npz, allow_pickle=False)
    transform = np.asarray(alignment["T_mujoco_from_videomimic"], dtype=np.float64)
    if transform.shape != (4, 4) or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("alignment T_mujoco_from_videomimic must be a homogeneous 4x4 transform")
    source_primitives = json.loads(args.input_primitives.read_text(encoding="utf-8"))
    primitives = _transform_primitives(source_primitives, transform)
    vertices, faces = _transform_obj(args.input_mesh, args.output_mesh, transform)
    visual_vertices = visual_faces = 0
    evidence_vertices = evidence_faces = 0
    visual_name = None
    if args.input_visual_mesh is not None:
        visual_vertices, visual_faces = _transform_obj(args.input_visual_mesh, args.output_visual_mesh, transform)
        visual_name = args.output_visual_mesh.name
    if args.input_evidence_mesh is not None:
        evidence_vertices, evidence_faces = _transform_obj(args.input_evidence_mesh, args.output_evidence_mesh, transform)
    _write_mujoco(args.output_mujoco, args.output_mesh.name, primitives, visual_name)
    _write_urdf(args.output_urdf, args.output_mesh.name, primitives)
    args.output_primitives.write_text(json.dumps({"schema_version": 1, "frame": "mujoco_world_z_up", "primitives": primitives}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report_json.write_text(json.dumps({"schema_version": 1, "source_frame": "videomimic_gravity_calibrated", "target_frame": "mujoco_world_z_up", "transform": transform.tolist(), "vertex_count": vertices, "face_count": faces, "visual_vertex_count": visual_vertices, "visual_face_count": visual_faces, "evidence_vertex_count": evidence_vertices, "evidence_face_count": evidence_faces, "primitive_count": len(primitives)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[PASS] transformed {vertices} vertices / {faces} faces into MuJoCo frame")


if __name__ == "__main__":
    main()
