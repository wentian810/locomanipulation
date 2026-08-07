#!/usr/bin/env python3
"""Transfer a verified G1 seated qpos between static chairs via seat geometry.

The source pose is an accepted gravity-seat result.  Its robot root is
expressed in the source ``seat_support_geom`` frame and reconstructed in the
target chair's corresponding frame; all 29 robot joint angles remain exactly
unchanged.  This creates only an MJCF keyframe for a subsequent gravity
validation--it never changes the target chair or applies any runtime force.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


def _rotation_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm((w, x, y, z)))
    if norm < 1e-10:
        raise ValueError("zero root quaternion")
    w, x, y, z = (w / norm, x / norm, y / norm, z / norm)
    return np.array((
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    ), dtype=np.float64)


def _wxyz_from_rotation(matrix: np.ndarray) -> np.ndarray:
    # Numerically stable branch form, returned in MuJoCo's WXYZ convention.
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        result = np.array((0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale,
                           (matrix[0, 2] - matrix[2, 0]) / scale,
                           (matrix[1, 0] - matrix[0, 1]) / scale))
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            result = np.array(((matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                               (matrix[0, 1] + matrix[1, 0]) / scale,
                               (matrix[0, 2] + matrix[2, 0]) / scale))
        elif axis == 1:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            result = np.array(((matrix[0, 2] - matrix[2, 0]) / scale,
                               (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                               (matrix[1, 2] + matrix[2, 1]) / scale))
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            result = np.array(((matrix[1, 0] - matrix[0, 1]) / scale,
                               (matrix[0, 2] + matrix[2, 0]) / scale,
                               (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale))
    return result / np.linalg.norm(result)


def _seat_frame(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    seat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "seat_support_geom")
    if seat_id < 0:
        raise ValueError("task lacks seat_support_geom")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return data.geom_xpos[seat_id].copy(), data.geom_xmat[seat_id].reshape(3, 3).copy()


def _free_qpos_address(model: mujoco.MjModel) -> int:
    ids = [index for index in range(model.njnt) if model.jnt_type[index] == mujoco.mjtJoint.mjJNT_FREE]
    if len(ids) != 1:
        raise ValueError("task must contain exactly one free joint")
    return int(model.jnt_qposadr[ids[0]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-task-xml", required=True, type=Path)
    parser.add_argument("--source-landing-report", required=True, type=Path)
    parser.add_argument("--target-task-xml", required=True, type=Path)
    parser.add_argument("--output-task-xml", required=True, type=Path)
    parser.add_argument("--key-name", required=True)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.output_task_xml.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite transferred-seat XML/report")

    source_model = mujoco.MjModel.from_xml_path(str(args.source_task_xml))
    target_model = mujoco.MjModel.from_xml_path(str(args.target_task_xml))
    if source_model.nq != target_model.nq:
        raise ValueError("source/target G1 qpos layouts differ")
    landing = json.loads(args.source_landing_report.read_text(encoding="utf-8"))
    if landing.get("status") != "accepted_gravity_seat_landing":
        raise ValueError("source landing report is not accepted")
    source_qpos = np.asarray(landing["summary"]["settled_final_qpos_wxyz"], dtype=np.float64)
    if source_qpos.shape != (source_model.nq,):
        raise ValueError("source landing qpos does not match source task")
    source_address = _free_qpos_address(source_model)
    target_address = _free_qpos_address(target_model)
    source_seat_pos, source_seat_rotation = _seat_frame(source_model)
    target_seat_pos, target_seat_rotation = _seat_frame(target_model)
    source_root_pos = source_qpos[source_address:source_address + 3]
    source_root_rotation = _rotation_from_wxyz(source_qpos[source_address + 3:source_address + 7])
    relative_root_pos = source_seat_rotation.T @ (source_root_pos - source_seat_pos)
    relative_root_rotation = source_seat_rotation.T @ source_root_rotation
    target_qpos = source_qpos.copy()
    target_qpos[target_address:target_address + 3] = target_seat_pos + target_seat_rotation @ relative_root_pos
    target_qpos[target_address + 3:target_address + 7] = _wxyz_from_rotation(
        target_seat_rotation @ relative_root_rotation
    )

    tree = ET.parse(args.target_task_xml)
    root = tree.getroot()
    keyframe = root.find("keyframe")
    if keyframe is None:
        keyframe = ET.SubElement(root, "keyframe")
    if any(key.get("name") == args.key_name for key in keyframe.findall("key")):
        raise ValueError(f"target XML already has keyframe {args.key_name!r}")
    ET.SubElement(keyframe, "key", {
        "name": args.key_name,
        "qpos": " ".join(f"{value:.17g}" for value in target_qpos),
    })
    args.output_task_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.output_task_xml, encoding="utf-8", xml_declaration=True)
    output_model = mujoco.MjModel.from_xml_path(str(args.output_task_xml))
    key_id = mujoco.mj_name2id(output_model, mujoco.mjtObj.mjOBJ_KEY, args.key_name)
    if key_id < 0 or not np.allclose(output_model.key_qpos[key_id], target_qpos, atol=1e-12, rtol=0.0):
        raise RuntimeError("written keyframe does not round-trip through MuJoCo")
    report = {
        "schema_version": 1,
        "status": "transferred_seat_key_ready_for_gravity_validation",
        "source": {
            "task_xml": str(args.source_task_xml.resolve()),
            "accepted_landing_report": str(args.source_landing_report.resolve()),
        },
        "target": {"task_xml": str(args.target_task_xml.resolve())},
        "output_task_xml": str(args.output_task_xml.resolve()),
        "key_name": args.key_name,
        "key_id": int(key_id),
        "seat_frame_transfer": {
            "source_relative_root_pos_m": relative_root_pos.tolist(),
            "source_relative_root_rotation_wxyz": _wxyz_from_rotation(relative_root_rotation).tolist(),
            "target_root_qpos_wxyz": target_qpos[:7].tolist(),
            "joint_position_delta_l2_rad": float(np.linalg.norm(target_qpos[7:] - source_qpos[7:])),
        },
        "contract": {
            "changes": "one target MJCF keyframe only",
            "does_not_change": ["target_chair_geometry", "target_robot_joints", "actuators", "runtime_state", "external_forces"],
        },
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
