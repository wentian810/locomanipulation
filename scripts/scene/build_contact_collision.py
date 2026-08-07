"""Build conservative collision artifacts from real first-round NKSR geometry.

The raw first-round mesh is retained separately as reconstruction evidence.  The
collision residual removes faces explained by the synchronized swept SMPL body,
then adds a thin contact-inferred support box only when seated pose evidence and
the remaining local elevated surface agree.  It never uses second-round hole
filling, so unobserved space below furniture is not silently made solid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
import smplx


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("cannot normalise a zero-length direction")
    return vector / norm


def _load_human(path: Path, person_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        root = handle["our_pred_humans_smplx_params"][person_id]
        return (
            np.asarray(root["betas"], dtype=np.float32),
            np.asarray(root["body_pose"], dtype=np.float32),
            np.asarray(root["global_orient"], dtype=np.float32),
            np.asarray(root["root_transl"], dtype=np.float32),
        )


def _smpl_geometry(
    model: torch.nn.Module,
    betas: np.ndarray,
    body: np.ndarray,
    orient: np.ndarray,
    transl: np.ndarray,
    frames: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        output = model(
            betas=torch.as_tensor(betas[frames], dtype=torch.float32),
            body_pose=torch.as_tensor(body[frames], dtype=torch.float32),
            global_orient=torch.as_tensor(orient[frames], dtype=torch.float32),
            transl=torch.as_tensor(transl[frames], dtype=torch.float32).reshape(len(frames), 3),
            pose2rot=False,
            return_verts=True,
        )
    return output.vertices.cpu().numpy(), output.joints.cpu().numpy()


def _clusters(points: np.ndarray, radius: float) -> list[np.ndarray]:
    """Return connected horizontal clusters without adding a sklearn dependency."""
    if len(points) == 0:
        return []
    parent = np.arange(len(points))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return int(index)

    for left, right in cKDTree(points[:, :2]).query_pairs(radius):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root
    groups: dict[int, list[int]] = {}
    for index in range(len(points)):
        groups.setdefault(find(index), []).append(index)
    return [points[indices] for indices in groups.values()]



def _sample_indices(frame_count: int, maximum: int) -> np.ndarray:
    """Cover the whole clip without making the swept human cloud unbounded."""
    if frame_count < 1 or maximum < 1:
        raise ValueError("human-mask sampling requires positive frame counts")
    if frame_count <= maximum:
        return np.arange(frame_count, dtype=np.int64)
    return np.unique(np.rint(np.linspace(0, frame_count - 1, maximum)).astype(np.int64))


def _remove_swept_human_surface(
    mesh: trimesh.Trimesh,
    human_points: np.ndarray,
    clearance: float,
) -> tuple[trimesh.Trimesh, dict]:
    """Remove first-round faces explained by the moving foreground body."""
    points = np.asarray(human_points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 32:
        raise ValueError("insufficient swept human points for collision filtering")
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    if len(centers) < 64:
        raise ValueError("first-round mesh has too few faces for collision filtering")
    distances, _ = cKDTree(points).query(centers, workers=-1)
    keep = distances >= clearance
    if int(keep.sum()) < 64:
        raise RuntimeError("foreground subtraction removed nearly all first-round faces")
    residual = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[keep], process=False)
    residual.remove_unreferenced_vertices()
    return residual, {
        "method": "swept_smpl_surface_face_centroid_proximity",
        "clearance_m": float(clearance),
        "human_surface_point_count": int(len(points)),
        "input_faces": int(len(mesh.faces)),
        "removed_faces": int((~keep).sum()),
        "retained_faces": int(keep.sum()),
        "removed_face_ratio": float((~keep).mean()),
        "nearest_swept_human_distance_m": {
            "min": float(np.min(distances)),
            "p01": float(np.quantile(distances, 0.01)),
            "p05": float(np.quantile(distances, 0.05)),
            "p50": float(np.quantile(distances, 0.50)),
            "p95": float(np.quantile(distances, 0.95)),
        },
    }
def _write_urdf(path: Path, residual_name: str, center: np.ndarray, extents: np.ndarray, yaw: float) -> None:
    text = f'''<?xml version="1.0"?>
<robot name="contact_aware_scene">
  <link name="residual_scene">
    <collision><geometry><mesh filename="{residual_name}" scale="1 1 1"/></geometry></collision>
  </link>
  <link name="seat_support">
    <collision>
      <origin xyz="{center[0]:.8f} {center[1]:.8f} {center[2]:.8f}" rpy="0 0 {yaw:.8f}"/>
      <geometry><box size="{extents[0]:.8f} {extents[1]:.8f} {extents[2]:.8f}"/></geometry>
    </collision>
  </link>
</robot>
'''
    path.write_text(text, encoding="utf-8")


def _write_mjcf(path: Path, residual_name: str, center: np.ndarray, extents: np.ndarray, yaw: float) -> None:
    half = extents / 2.0
    text = f'''<mujoco model="contact_aware_scene">
  <asset><mesh name="residual_scene" file="{residual_name}"/></asset>
  <worldbody>
    <geom name="residual_scene" type="mesh" mesh="residual_scene" contype="1" conaffinity="1"/>
    <body name="seat_support" pos="{center[0]:.8f} {center[1]:.8f} {center[2]:.8f}" euler="0 0 {yaw:.8f}">
      <geom name="seat_support" type="box" size="{half[0]:.8f} {half[1]:.8f} {half[2]:.8f}" contype="1" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>
'''
    path.write_text(text, encoding="utf-8")




def _emit_residual_only(
    output: Path,
    residual_mesh: trimesh.Trimesh,
    evidence: dict,
    human_filter: dict,
    reason: str,
) -> None:
    residual_path = output / "residual_mesh.obj"
    residual_mesh.export(residual_path)
    (output / "primitives.json").write_text(
        json.dumps({"schema_version": 1, "primitives": []}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    quality = {
        "schema_version": 1,
        "verdict": "warn",
        "residual_mesh": {
            "path": str(residual_path),
            "role": "first_round_nksr_with_swept_human_surface_removed_no_second_round_hole_fill",
            "vertices": int(len(residual_mesh.vertices)),
            "faces": int(len(residual_mesh.faces)),
            "watertight": bool(residual_mesh.is_watertight),
        },
        "dynamic_human_filter": human_filter,
        "support_evidence": {
            "status": "unavailable",
            "reason": reason,
            "inferred_vertical_axis": evidence.get("inferred_vertical_axis"),
            "seated_frame_count": int(evidence.get("seated_frame_count", 0)),
        },
        "fallback_policy": {
            "crisp": "human-swept first-round residual only; no support primitive was inferred",
            "escalate_to_crisp_if": ["a contact patch has reliable local geometric evidence"],
        },
    }
    (output / "quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[WARN] residual-only collision fallback: {reason}")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrated-h5", required=True, type=Path)
    parser.add_argument("--first-round-mesh", required=True, type=Path)
    parser.add_argument("--contact-evidence", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--person-id", default="1")
    parser.add_argument("--gender", choices=("male", "female", "neutral"), default="neutral")
    parser.add_argument("--seat-thickness", type=float, default=0.04)
    parser.add_argument("--human-mask-clearance-m", type=float, default=0.05)
    parser.add_argument("--human-mask-max-frames", type=int, default=400)
    parser.add_argument("--human-mask-max-points", type=int, default=1_500_000)
    args = parser.parse_args()
    if not 0.02 <= args.human_mask_clearance_m <= 0.15:
        raise ValueError("human-mask-clearance-m must be in [0.02, 0.15] metres")
    if args.human_mask_max_frames < 1 or args.human_mask_max_points < 10_000:
        raise ValueError("human-mask sampling limits are too small")

    if not 0.01 <= args.seat_thickness <= 0.10:
        raise ValueError("seat thickness must be in [0.01, 0.10] metres")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty collision output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    evidence = json.loads(args.contact_evidence.read_text(encoding="utf-8"))
    axis_name = evidence.get("inferred_vertical_axis")
    if axis_name not in {"x", "y", "z"}:
        raise ValueError("contact evidence has no usable gravity-axis record")
    vertical_axis = "xyz".index(axis_name)
    axis_source = evidence.get("gravity_axis_source")
    seated_frames = np.asarray(
        [item["frame"] for item in evidence["frames"] if item.get("seated_candidate")], dtype=np.int64
    )
    mesh = trimesh.load(args.first_round_mesh, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise RuntimeError("first-round residual mesh is invalid")
    betas, body, orient, transl = _load_human(args.calibrated_h5, args.person_id)
    sweep_frames = _sample_indices(len(body), args.human_mask_max_frames)
    sweep_model = smplx.create(
        str(args.model_root), model_type="smpl", gender=args.gender,
        num_betas=10, batch_size=len(sweep_frames),
    )
    sweep_vertices, _ = _smpl_geometry(sweep_model, betas, body, orient, transl, sweep_frames)
    sweep_points = sweep_vertices.reshape(-1, 3)
    point_stride = max(1, int(np.ceil(len(sweep_points) / args.human_mask_max_points)))
    residual_mesh, human_filter = _remove_swept_human_surface(
        mesh, sweep_points[::point_stride], args.human_mask_clearance_m,
    )
    human_filter.update({
        "sampled_frame_count": int(len(sweep_frames)),
        "sampled_frame_stride_max": int(np.diff(sweep_frames).max()) if len(sweep_frames) > 1 else 0,
        "surface_vertex_stride": int(point_stride),
    })
    if axis_source != "videomimic_gravity_calibrated_coordinate_contract":
        _emit_residual_only(output, residual_mesh, evidence, human_filter, "gravity-axis provenance is missing")
        return
    if vertical_axis != 2:
        _emit_residual_only(output, residual_mesh, evidence, human_filter, "contact evidence is not z-up")
        return
    if len(seated_frames) < 12:
        _emit_residual_only(output, residual_mesh, evidence, human_filter, f"only {len(seated_frames)} seated evidence frames")
        return
    model = smplx.create(
        str(args.model_root), model_type="smpl", gender=args.gender, num_betas=10, batch_size=len(seated_frames)
    )
    vertices, joints = _smpl_geometry(model, betas, body, orient, transl, seated_frames)
    hips = joints[:, [1, 2]].mean(axis=1)
    knees = joints[:, [4, 5]].mean(axis=1)
    hip_center = np.median(hips, axis=0)
    forward = np.median(knees - hips, axis=0)
    forward[2] = 0.0
    forward = _unit(forward)
    lateral = _unit(np.cross(np.array([0.0, 0.0, 1.0]), forward))
    hip_height = float(np.median(hips[:, 2]))

    # Search only in a local elevated band: the floor is deliberately excluded
    # and cannot be mistaken for a chair seat.
    local_delta = residual_mesh.vertices - hip_center
    radial = np.linalg.norm(local_delta[:, :2], axis=1)
    local = residual_mesh.vertices[
        (radial <= 1.0)
        & (residual_mesh.vertices[:, 2] <= hip_height - 0.03)
        & (residual_mesh.vertices[:, 2] >= hip_height - 0.35)
    ]
    candidate_clusters = [cluster for cluster in _clusters(local, radius=0.075) if len(cluster) >= 25]
    if not candidate_clusters:
        _emit_residual_only(output, residual_mesh, evidence, human_filter, "no local elevated support-surface cluster")
        return

    def cluster_score(cluster: np.ndarray) -> float:
        horizontal_distance = float(np.linalg.norm(cluster[:, :2].mean(axis=0) - hip_center[:2]))
        return len(cluster) - 80.0 * horizontal_distance

    support = max(candidate_clusters, key=cluster_score)
    support_top = float(np.quantile(support[:, 2], 0.90))
    hip_support_gap = hip_height - support_top
    if not 0.03 <= hip_support_gap <= 0.25:
        _emit_residual_only(output, residual_mesh, evidence, human_filter, f"support height gap {hip_support_gap:.3f}m is outside the chair-seat evidence range")
        return

    thigh_length = float(np.median(np.linalg.norm((knees - hips)[:, :2], axis=1)))
    local_support = support - hip_center
    back_extent = float(np.clip(-np.quantile(local_support @ forward, 0.05), 0.22, 0.45))
    forward_extent = float(np.clip(0.82 * thigh_length, 0.24, 0.42))
    half_width = float(np.clip(np.max(np.abs((hips - hip_center) @ lateral)) + 0.16, 0.24, 0.38))
    length = forward_extent + back_extent
    center = hip_center + forward * ((forward_extent - back_extent) / 2.0)
    center[2] = support_top - args.seat_thickness / 2.0
    extents = np.array([length, 2.0 * half_width, args.seat_thickness], dtype=np.float64)
    yaw = float(np.arctan2(forward[1], forward[0]))

    # Inspect the inferred support volume against seated-body vertices.  These
    # are unsigned diagnostics only; no signed penetration claim is made.
    relative = vertices - center
    forward_coord = relative @ forward
    lateral_coord = relative @ lateral
    in_footprint = (np.abs(forward_coord) <= length / 2.0) & (np.abs(lateral_coord) <= half_width)
    in_support_slab = in_footprint & (vertices[:, :, 2] >= support_top - 0.03) & (vertices[:, :, 2] <= support_top + 0.25)
    below_top = in_footprint & (vertices[:, :, 2] < support_top - 0.01)
    nearest, _ = cKDTree(residual_mesh.vertices).query(vertices.reshape(-1, 3), workers=-1)
    nearest = nearest.reshape(len(seated_frames), -1)

    residual_path = output / "residual_mesh.obj"
    residual_mesh.export(residual_path)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((forward, lateral, np.array([0.0, 0.0, 1.0])))
    transform[:3, 3] = center
    primitive = {
        "name": "seat_support",
        "type": "box",
        "frame": "gravity_calibrated_world_z_up",
        "source": "seated_pose_plus_local_first_round_nksr_cluster",
        "center": center.tolist(),
        "extents": extents.tolist(),
        "top_height_z": support_top,
        "yaw_radians": yaw,
        "transform": transform.tolist(),
        "confidence": "warn",
        "limitations": [
            "dynamic person masking leaves part of the physical seat unobserved",
            "primitive is thin by design and does not fill the volume below the seat",
            "requires signed-distance/visual validation before being promoted to a production contact constraint",
        ],
    }
    (output / "primitives.json").write_text(
        json.dumps({"schema_version": 1, "primitives": [primitive]}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_urdf(output / "scene.urdf", residual_path.name, center, extents, yaw)
    _write_mjcf(output / "scene.xml", residual_path.name, center, extents, yaw)

    quality = {
        "schema_version": 1,
        "verdict": "warn",
        "residual_mesh": {
            "path": str(residual_path),
            "role": "first_round_nksr_with_swept_human_surface_removed_no_second_round_hole_fill",
            "vertices": int(len(residual_mesh.vertices)),
            "faces": int(len(residual_mesh.faces)),
            "watertight": bool(residual_mesh.is_watertight),
        },
        "dynamic_human_filter": human_filter,
        "support_evidence": {
            "seated_frame_count": int(len(seated_frames)),
            "seated_frame_range": [int(seated_frames.min()), int(seated_frames.max())],
            "support_cluster_vertices": int(len(support)),
            "support_top_z": support_top,
            "median_hip_z": hip_height,
            "hip_to_support_gap": hip_support_gap,
            "nearest_vertex_distance_seated": {
                "p01_median": float(np.median(np.quantile(nearest, 0.01, axis=1))),
                "p05_median": float(np.median(np.quantile(nearest, 0.05, axis=1))),
            },
            "human_vertices_in_support_slab_fraction": float(in_support_slab.mean()),
            "human_vertices_below_inferred_seat_top_fraction": float(below_top.mean()),
        },
        "fallback_policy": {
            "crisp": "not invoked: local thin support primitive is available",
            "escalate_to_crisp_if": [
                "signed-distance validation detects material body/seat penetration",
                "visual review rejects support position or chair topology",
                "a semantic watertight object mesh is required rather than a collision proxy",
            ],
        },
    }
    (output / "quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[PASS] residual first-round mesh -> {residual_path}")
    print(f"[PASS] conservative seat primitive -> {output / 'primitives.json'}")
    print(f"[WARN] quality report requires signed/visual validation -> {output / 'quality.json'}")


if __name__ == "__main__":
    main()
