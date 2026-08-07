#!/usr/bin/env python3
"""Materialize one static, metric-corrected semantic-chair MuJoCo scene.

This converts semantic chair boxes through a previously accepted SMPL-to-G1
similarity bridge.  It deliberately writes a new directory; no source scene,
motion, camera, or previously rejected diagnostic is overwritten.
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
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fmt(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.10g}" for value in np.asarray(values).reshape(-1))


def rotation_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Numerically stable active-rotation matrix to MuJoCo wxyz quaternion."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.allclose(matrix.T @ matrix, np.eye(3), atol=2e-5):
        raise ValueError("rotation matrix must be orthonormal")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        value = 2.0 * np.sqrt(trace + 1.0)
        quat = np.array([0.25 * value, (matrix[2, 1] - matrix[1, 2]) / value,
                         (matrix[0, 2] - matrix[2, 0]) / value,
                         (matrix[1, 0] - matrix[0, 1]) / value])
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            value = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            quat = np.array([(matrix[2, 1] - matrix[1, 2]) / value, 0.25 * value,
                             (matrix[0, 1] + matrix[1, 0]) / value,
                             (matrix[0, 2] + matrix[2, 0]) / value])
        elif axis == 1:
            value = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            quat = np.array([(matrix[0, 2] - matrix[2, 0]) / value,
                             (matrix[0, 1] + matrix[1, 0]) / value, 0.25 * value,
                             (matrix[1, 2] + matrix[2, 1]) / value])
        else:
            value = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            quat = np.array([(matrix[1, 0] - matrix[0, 1]) / value,
                             (matrix[0, 2] + matrix[2, 0]) / value,
                             (matrix[1, 2] + matrix[2, 1]) / value, 0.25 * value])
    return quat / np.linalg.norm(quat)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-primitives", required=True, type=Path)
    parser.add_argument("--similarity-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--include-transformed-floor", action="store_true",
        help="only for a scene model without a canonical robot floor; normal G1 task bundles already provide one",
    )
    parser.add_argument("--floor-half-size-m", type=float, default=20.0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite a static similarity-scene candidate")
    if args.floor_half_size_m <= 0.0:
        raise ValueError("floor-half-size-m must be positive")
    source = json.loads(args.source_primitives.read_text(encoding="utf-8"))
    bridge = json.loads(args.similarity_report.read_text(encoding="utf-8"))
    if bridge.get("status") != "accepted_similarity_bridge_candidate":
        raise ValueError("similarity report was not accepted")
    if source.get("frame") != "mujoco_world_z_up" or not isinstance(source.get("primitives"), list):
        raise ValueError("source primitives must be in mujoco_world_z_up")
    transform = bridge.get("similarity_smpl_zup_to_g1", {})
    scale = float(transform.get("scale", 0.0))
    rotation = np.asarray(transform.get("rotation_matrix"), dtype=np.float64)
    translation = np.asarray(transform.get("translation_xyz_m"), dtype=np.float64)
    if not 0.0 < scale < 2.0 or rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("invalid accepted similarity transform")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5) or np.linalg.det(rotation) <= 0.0:
        raise ValueError("similarity rotation is not a proper rotation")

    transformed: list[dict[str, object]] = []
    for primitive in source["primitives"]:
        name = primitive.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("every primitive must have a name")
        center = np.asarray(primitive.get("center"), dtype=np.float64)
        local_rotation = np.asarray(primitive.get("rotation_matrix"), dtype=np.float64)
        extents = np.asarray(primitive.get("extents"), dtype=np.float64)
        if center.shape != (3,) or local_rotation.shape != (3, 3) or extents.shape != (3,) or np.any(extents <= 0.0):
            raise ValueError(f"primitive {name!r} is malformed")
        new_rotation = rotation @ local_rotation
        transformed_item = dict(primitive)
        transformed_item["center"] = (scale * (rotation @ center) + translation).tolist()
        transformed_item["rotation_matrix"] = new_rotation.tolist()
        transformed_item["quat_wxyz"] = rotation_to_quat_wxyz(new_rotation).tolist()
        transformed_item["extents"] = (scale * extents).tolist()
        transformed_item["source_similarity_transform"] = "smpl_zup_to_g1_static_sim3_v1"
        transformed.append(transformed_item)

    args.output_dir.mkdir(parents=True)
    primitives_path = args.output_dir / "semantic_chair_primitives_g1_sim3_v1.json"
    primitives_payload = {
        "schema_version": 1, "frame": "mujoco_world_z_up",
        "static_transform": {"scale": scale, "rotation_matrix": rotation.tolist(), "translation_xyz_m": translation.tolist()},
        "primitives": transformed,
    }
    primitives_path.write_text(json.dumps(primitives_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    root = ET.Element("mujoco", {"model": "semantic_chair_g1_similarity_v1"})
    world = ET.SubElement(root, "worldbody")
    if args.include_transformed_floor:
        # The semantic source has no analytic floor geom.  Its packaged world
        # frame defines z=0 as the support plane, so transform that plane once
        # with the accepted Sim(3).  G1 task bundles normally already own the
        # sole canonical floor, in which case adding this would be invalid.
        ET.SubElement(world, "geom", {
            "name": "floor", "type": "plane", "pos": fmt(translation),
            "quat": fmt(rotation_to_quat_wxyz(rotation)),
            "size": fmt(np.array([args.floor_half_size_m, args.floor_half_size_m, 0.1])),
            "rgba": "0.35 0.35 0.35 1", "contype": "1", "conaffinity": "1",
        })
    for primitive in transformed:
        name = str(primitive["name"])
        body = ET.SubElement(world, "body", {
            "name": name, "pos": fmt(np.asarray(primitive["center"])),
            "quat": fmt(np.asarray(primitive["quat_wxyz"])),
        })
        ET.SubElement(body, "geom", {
            "name": f"{name}_geom", "type": "box",
            "size": fmt(0.5 * np.asarray(primitive["extents"])),
            "rgba": "0.92 0.92 0.90 1", "contype": "1", "conaffinity": "1",
        })
    xml_path = args.output_dir / "scene_mujoco_g1_sim3_v1.xml"
    ET.ElementTree(root).write(xml_path, encoding="unicode")
    report = {
        "schema_version": 1,
        "purpose": "static_semantic_chair_scene_from_accepted_smpl_g1_similarity",
        "status": "candidate_requires_projection_and_mj_step_gates",
        "policy": {"chair": "one_static_body_set", "motion_modified": False, "source_overwritten": False},
        "inputs": {
            "source_primitives": str(args.source_primitives), "source_primitives_sha256": sha256(args.source_primitives),
            "similarity_report": str(args.similarity_report), "similarity_report_sha256": sha256(args.similarity_report),
        },
        "outputs": {"primitives": str(primitives_path), "mujoco_xml": str(xml_path)},
        "floor_contract": (
            "transformed source z=0 plane included" if args.include_transformed_floor
            else "no floor emitted: the canonical G1 task's unique z=0 floor is retained"
        ),
        "similarity": primitives_payload["static_transform"],
    }
    (args.output_dir / "scene_candidate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
