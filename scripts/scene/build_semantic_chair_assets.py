"""Build an explicit, conservative chair from an aligned seat-support box.

The input support primitive is already registered to the canonical human / PHC
frame.  This script intentionally does *not* infer collision from the noisy
monocular mesh: it preserves the validated seat plane and adds only named,
inspectable chair parts (four legs and one backrest).  The resulting JSON is
for Isaac Gym; the MJCF is the matching visual/review scene for MuJoCo/GMR.

For a new scene, the chair's back side is inferred from the stable seated-body
heading, not supplied as a clip-specific switch.  The script rejects ambiguous
heading or seat-axis evidence rather than silently producing a flipped chair.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


BOX_FACES = np.asarray(
    [
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
    ],
    dtype=np.int64,
)

# GVHMR/SMPL uses Y-up.  This is the same documented conversion used by the
# PHC scene-alignment audit: [x, -z, y] -> MuJoCo world Z-up.
GVHMR_Y_UP_TO_MUJOCO_Z_UP = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)


def fmt(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.8g}" for value in values)


def matrix_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Return a normalized MuJoCo-order quaternion without SciPy."""
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quat = np.asarray(
            [0.25 * scale, (rotation[2, 1] - rotation[1, 2]) / scale,
             (rotation[0, 2] - rotation[2, 0]) / scale,
             (rotation[1, 0] - rotation[0, 1]) / scale],
            dtype=np.float64,
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            quat = np.asarray(
                [(rotation[2, 1] - rotation[1, 2]) / scale, 0.25 * scale,
                 (rotation[0, 1] + rotation[1, 0]) / scale,
                 (rotation[0, 2] + rotation[2, 0]) / scale],
                dtype=np.float64,
            )
        elif index == 1:
            scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            quat = np.asarray(
                [(rotation[0, 2] - rotation[2, 0]) / scale,
                 (rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale,
                 (rotation[1, 2] + rotation[2, 1]) / scale],
                dtype=np.float64,
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            quat = np.asarray(
                [(rotation[1, 0] - rotation[0, 1]) / scale,
                 (rotation[0, 2] + rotation[2, 0]) / scale,
                 (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale],
                dtype=np.float64,
            )
    return quat / np.linalg.norm(quat)


def level_rotation_keep_yaw(source_rotation: np.ndarray) -> np.ndarray:
    """Keep the reconstructed chair yaw but make support / legs gravity-aligned."""
    x_axis = np.asarray(source_rotation[:, 0], dtype=np.float64).copy()
    x_axis[2] = 0.0
    length = float(np.linalg.norm(x_axis))
    if length < 1e-6:
        raise ValueError("cannot derive horizontal chair yaw from a vertical source axis")
    x_axis /= length
    z_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Convert one SMPL axis-angle root orientation to a 3x3 matrix."""
    theta = float(np.linalg.norm(axis_angle))
    if theta < 1e-10:
        return np.eye(3, dtype=np.float64)
    axis = np.asarray(axis_angle, dtype=np.float64) / theta
    cross = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + np.sin(theta) * cross + (1.0 - np.cos(theta)) * (cross @ cross)


def infer_layout_from_seated_body(
    *,
    motion_path: Path,
    contact_evidence_path: Path,
    levelled_rotation: np.ndarray,
) -> tuple[bool, int, dict]:
    """Infer the depth axis and backrest side from confident seated frames.

    SMPL's local +Z is the body-forward axis.  A conventional seated subject
    faces away from the chair backrest, so the backrest must lie opposite this
    direction.  This is only used when the heading is temporally coherent and
    closely aligned with one observed seat axis.
    """
    evidence = json.loads(contact_evidence_path.read_text(encoding="utf-8"))
    frames = np.asarray(
        [item["frame"] for item in evidence.get("frames", []) if item.get("seated_candidate")],
        dtype=np.int64,
    )
    if len(frames) < 12:
        raise ValueError("automatic chair layout requires at least 12 seated evidence frames")
    with np.load(motion_path, allow_pickle=False) as motion:
        if "root_orient" not in motion:
            raise ValueError("automatic chair layout requires root_orient in the SMPL motion")
        root_orient = np.asarray(motion["root_orient"], dtype=np.float64)
    if root_orient.ndim != 2 or root_orient.shape[1] != 3:
        raise ValueError("root_orient must have shape [frame, 3]")
    if frames.min() < 0 or frames.max() >= len(root_orient):
        raise ValueError("seated evidence frame index exceeds the supplied SMPL motion")

    # SMPL's local +Z is transformed to the canonical MuJoCo Z-up frame.
    forward = np.asarray(
        [
            GVHMR_Y_UP_TO_MUJOCO_Z_UP @ axis_angle_to_matrix(root_orient[frame]) @ np.asarray([0.0, 0.0, 1.0])
            for frame in frames
        ],
        dtype=np.float64,
    )
    forward[:, 2] = 0.0
    lengths = np.linalg.norm(forward, axis=1)
    if np.any(lengths < 1e-6):
        raise ValueError("seated SMPL body-forward vector is vertical or degenerate")
    forward /= lengths[:, None]
    mean_forward = forward.mean(axis=0)
    heading_concentration = float(np.linalg.norm(mean_forward))
    if heading_concentration < 0.92:
        raise ValueError(
            "automatic chair layout rejected: seated body heading is not stable "
            f"(concentration={heading_concentration:.3f}, required>=0.920)"
        )
    mean_forward /= heading_concentration

    source_x = np.asarray(levelled_rotation[:, 0], dtype=np.float64)
    source_y = np.asarray(levelled_rotation[:, 1], dtype=np.float64)
    source_x[2] = 0.0
    source_y[2] = 0.0
    source_x /= np.linalg.norm(source_x)
    source_y /= np.linalg.norm(source_y)
    alignment_x = np.abs(forward @ source_x)
    alignment_y = np.abs(forward @ source_y)
    median_x = float(np.median(alignment_x))
    median_y = float(np.median(alignment_y))
    if max(median_x, median_y) < 0.85 or abs(median_x - median_y) < 0.10:
        raise ValueError(
            "automatic chair layout rejected: seated heading does not identify one "
            f"seat depth axis (x={median_x:.3f}, y={median_y:.3f})"
        )

    source_depth_axis = "x" if median_x > median_y else "y"
    # The existing asset convention puts local +Y at the nominal backrest.  If
    # source X is depth, swapping makes final +Y equal to -source X.
    swap_horizontal_axes = source_depth_axis == "x"
    final_positive_y = -source_x if swap_horizontal_axes else source_y
    back_direction = -mean_forward
    back_sign = int(np.sign(float(back_direction @ final_positive_y)))
    if back_sign == 0:
        raise ValueError("automatic chair layout rejected: backrest side is ambiguous")
    return swap_horizontal_axes, back_sign, {
        "method": "seated_smpl_body_heading",
        "motion": str(motion_path),
        "contact_evidence": str(contact_evidence_path),
        "seated_frame_count": int(len(frames)),
        "seated_frame_range": [int(frames.min()), int(frames.max())],
        "smpl_body_forward_local_axis": "+z",
        "input_to_mujoco_transform": "[x,-z,y]",
        "heading_concentration": heading_concentration,
        "median_axis_alignment": {"source_x": median_x, "source_y": median_y},
        "inferred_source_depth_axis": source_depth_axis,
        "inferred_back_sign": back_sign,
    }


def box_vertices(center: np.ndarray, rotation: np.ndarray, extents: np.ndarray) -> np.ndarray:
    half = 0.5 * extents
    local = np.asarray(
        [
            [-half[0], -half[1], -half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], half[1], -half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], half[2]],
            [-half[0], half[1], half[2]],
        ],
        dtype=np.float64,
    )
    return center + local @ rotation.T


def write_obj(path: Path, boxes: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for box in boxes:
            vertices = box_vertices(box["center"], box["rotation"], box["extents"])
            for vertex in vertices:
                stream.write("v {:.8g} {:.8g} {:.8g}\n".format(*vertex))
        for index in range(len(boxes)):
            for face in BOX_FACES + 1 + 8 * index:
                stream.write("f {} {} {}\n".format(*face))


def write_mujoco(path: Path, boxes: list[dict]) -> None:
    root = ET.Element("mujoco", {"model": "semantic_chair"})
    worldbody = ET.SubElement(root, "worldbody")
    for box in boxes:
        body = ET.SubElement(
            worldbody,
            "body",
            {
                "name": box["name"],
                "pos": fmt(box["center"]),
                "quat": fmt(box["quat_wxyz"]),
            },
        )
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{box['name']}_geom",
                "type": "box",
                "size": fmt(0.5 * box["extents"]),
                "rgba": "0.92 0.92 0.90 1",
                "contype": "1",
                "conaffinity": "1",
            },
        )
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="unicode", xml_declaration=True)


def write_urdf(path: Path, boxes: list[dict]) -> None:
    root = ET.Element("robot", {"name": "semantic_chair"})
    for box in boxes:
        link = ET.SubElement(root, "link", {"name": box["name"]})
        for tag in ("visual", "collision"):
            element = ET.SubElement(link, tag)
            ET.SubElement(element, "origin", {"xyz": fmt(box["center"]), "rpy": "0 0 0"})
            geometry = ET.SubElement(element, "geometry")
            ET.SubElement(geometry, "box", {"size": fmt(box["extents"])})
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="unicode", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-primitives", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--layout-mode",
        choices=("manual", "auto_seated_body"),
        default="manual",
        help="manual keeps legacy switches; auto_seated_body derives them from seated SMPL evidence",
    )
    parser.add_argument("--motion-npz", type=Path)
    parser.add_argument("--contact-evidence-json", type=Path)
    parser.add_argument("--back-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument(
        "--swap-horizontal-axes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "reinterpret source local x as chair depth and source local y as "
            "chair width; preserves the world-space seat box"
        ),
    )
    parser.add_argument(
        "--level-seat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="retain reconstructed yaw but level the support against Z-up gravity",
    )
    parser.add_argument(
        "--phc-backrest-collision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "include a world-identical backrest proxy in the stable PHC basis; "
            "off by default to preserve the V18 seat-only contract"
        ),
    )
    args = parser.parse_args()

    source = json.loads(args.input_primitives.read_text(encoding="utf-8"))
    supports = source.get("primitives", [])
    if not supports or supports[0].get("type") != "box":
        raise ValueError("the source must provide a first box-shaped seat support")
    support = supports[0]
    center = np.asarray(support["center"], dtype=np.float64)
    rotation = np.asarray(support["rotation_matrix"], dtype=np.float64)
    seat_extents = np.asarray(support["extents"], dtype=np.float64)
    if center.shape != (3,) or rotation.shape != (3, 3) or seat_extents.shape != (3,):
        raise ValueError("invalid aligned support primitive")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4):
        raise ValueError("support rotation is not orthonormal")
    if np.any(seat_extents <= 0):
        raise ValueError("support extents must be positive")

    source_seat_extents = seat_extents.copy()
    source_rotation = rotation.copy()
    if args.level_seat:
        rotation = level_rotation_keep_yaw(source_rotation)
    phc_collision_rotation = rotation.copy()
    if args.layout_mode == "auto_seated_body":
        if args.motion_npz is None or args.contact_evidence_json is None:
            raise ValueError(
                "auto_seated_body requires --motion-npz and --contact-evidence-json"
            )
        if args.swap_horizontal_axes or args.back_sign != 1:
            raise ValueError(
                "auto_seated_body does not accept manual back-sign or axis overrides"
            )
        swap_horizontal_axes, back_sign, layout_inference = infer_layout_from_seated_body(
            motion_path=args.motion_npz,
            contact_evidence_path=args.contact_evidence_json,
            levelled_rotation=rotation,
        )
    else:
        swap_horizontal_axes = bool(args.swap_horizontal_axes)
        back_sign = int(args.back_sign)
        layout_inference = {
            "method": "manual_legacy_override",
            "swap_horizontal_axes": swap_horizontal_axes,
            "back_sign": back_sign,
        }
    if swap_horizontal_axes:
        # New local x is source local y (chair width); new local +y is the
        # negative source local x (chair back). Swapping extents preserves the
        # seat's world-space box exactly.
        rotation = rotation @ np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        seat_extents = seat_extents[[1, 0, 2]]
    quat = matrix_to_quat_wxyz(rotation)
    source_tilt_deg = float(
        np.degrees(np.arccos(np.clip(source_rotation[2, 2], -1.0, 1.0)))
    )

    def make_box(name: str, local_center: tuple[float, float, float], extents: tuple[float, float, float]) -> dict:
        local = np.asarray(local_center, dtype=np.float64)
        return {
            "name": name,
            "type": "box",
            "frame": "mujoco_world_z_up",
            "source": "semantic_chair_from_aligned_seat_support",
            "center": center + rotation @ local,
            "rotation": rotation,
            "rotation_matrix": rotation.tolist(),
            "quat_wxyz": quat,
            "extents": np.asarray(extents, dtype=np.float64),
            "confidence": "semantic_parametric",
        }

    width, depth, thickness = (float(value) for value in seat_extents)
    leg_cross = min(0.055, max(0.040, 0.09 * min(width, depth)))
    leg_height = 0.46
    inset_x = max(0.055, 0.5 * width - 0.8 * leg_cross)
    inset_y = max(0.055, 0.5 * depth - 0.8 * leg_cross)
    leg_z = -0.5 * thickness - 0.5 * leg_height
    # A single broad slatted-back proxy is safer and more stable than trying to
    # promote the wall-shaped connected component from monocular meshing.
    back_thickness = 0.040
    back_height = 0.42
    back_y = back_sign * (0.5 * depth - 0.5 * back_thickness)
    back_z = 0.5 * thickness + 0.5 * back_height
    backrest_extents = (0.82 * width, back_thickness, back_height)

    boxes = [
        make_box("seat_support", (0.0, 0.0, 0.0), tuple(seat_extents)),
        make_box("leg_front_left", (-inset_x, -inset_y, leg_z), (leg_cross, leg_cross, leg_height)),
        make_box("leg_front_right", (inset_x, -inset_y, leg_z), (leg_cross, leg_cross, leg_height)),
        make_box("leg_back_left", (-inset_x, inset_y, leg_z), (leg_cross, leg_cross, leg_height)),
        make_box("leg_back_right", (inset_x, inset_y, leg_z), (leg_cross, leg_cross, leg_height)),
        make_box("backrest", (0.0, back_y, back_z), backrest_extents),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    serializable_boxes = []
    for box in boxes:
        serializable_boxes.append(
            {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in box.items()
                if key != "rotation"
            }
        )
    (args.output_dir / "semantic_chair_primitives_mujoco.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "frame": "mujoco_world_z_up",
                "primitives": serializable_boxes,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    # Isaac Gym triangle-mesh contact is sensitive to the box's local vertex
    # ordering. Keep a dedicated seat-only collision proxy in the source basis,
    # geometrically identical to the semantic V4 seat in world space but already
    # verified by the independent PhysX support probe.
    phc_support = {
        "name": "seat_support",
        "type": "box",
        "frame": "mujoco_world_z_up",
        "source": "semantic_chair_phc_stable_source_basis",
        "center": center.tolist(),
        "rotation_matrix": phc_collision_rotation.tolist(),
        "extents": source_seat_extents.tolist(),
        "confidence": "semantic_parametric_contact_proxy",
    }
    phc_primitives = [phc_support]
    if args.phc_backrest_collision:
        visual_backrest = next(box for box in boxes if box["name"] == "backrest")
        phc_backrest_extents = np.asarray(backrest_extents, dtype=np.float64)
        if args.swap_horizontal_axes:
            phc_backrest_extents = phc_backrest_extents[[1, 0, 2]]
        phc_primitives.append(
            {
                "name": "backrest_support",
                "type": "box",
                "frame": "mujoco_world_z_up",
                "source": "semantic_chair_phc_stable_source_basis",
                "center": visual_backrest["center"].tolist(),
                "rotation_matrix": phc_collision_rotation.tolist(),
                "extents": phc_backrest_extents.tolist(),
                "confidence": "semantic_parametric_contact_proxy",
            }
        )
    (args.output_dir / "semantic_chair_primitives_phc.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "frame": "mujoco_world_z_up",
                "primitives": phc_primitives,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_obj(args.output_dir / "background_mesh_semantic_chair.obj", boxes)
    write_mujoco(args.output_dir / "scene_mujoco_semantic_chair.xml", boxes)
    write_urdf(args.output_dir / "scene_semantic_chair.urdf", boxes)
    report = {
        "schema_version": 1,
        "source_support": support,
        "back_sign": back_sign,
        "horizontal_axis_policy": (
            "source_x_is_depth_source_y_is_width"
            if swap_horizontal_axes
            else "source_x_is_width_source_y_is_depth"
        ),
        "layout_inference": layout_inference,
        "gravity_alignment": {
            "level_seat": bool(args.level_seat),
            "source_seat_tilt_deg": source_tilt_deg,
            "output_seat_tilt_deg": float(
                np.degrees(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0)))
            ),
            "policy": "preserve_yaw_level_support_legs_and_backrest",
        },
        "seat_top_center": (center + rotation @ np.asarray([0.0, 0.0, 0.5 * thickness])).tolist(),
        "parts": [
            {"name": box["name"], "center": box["center"].tolist(), "extents": box["extents"].tolist()}
            for box in boxes
        ],
        "limitations": [
            "parametric chair geometry is used because the monocular mesh fused the chair with the background wall",
            "the source support center and yaw are retained; gravity alignment levels the physical seat by default",
            "automatic mode rejects an unstable seated heading or ambiguous seat axis; manual mode is legacy-only",
            "Isaac/PHC receives a stable-basis seat proxy and, when enabled, a world-identical stable-basis backrest proxy",
        ],
        "physics_collision_proxy": {
            "asset": "semantic_chair_primitives_phc.json",
            "parts": [item["name"] for item in phc_primitives],
            "backrest_collision_enabled": bool(args.phc_backrest_collision),
            "world_space_seat_geometry": "identical_to_semantic_visual_seat",
        },
    }
    (args.output_dir / "semantic_chair_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
