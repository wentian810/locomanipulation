from typing import Dict, Iterable, Mapping, Sequence, Union

import numpy as np
import trimesh
import json

PartIndices = Union[Mapping[str, Iterable[int]], Sequence[Iterable[int]]]


def vertex2face_indices(
    mesh_path: str,
    part_vertex_indices: PartIndices,
    include_partial_faces: bool = False,
    visualize: Union[bool, str] = False,
) -> Dict[str, Dict[str, object]]:
    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[-1] != 3:
        raise ValueError(f"mesh vertices must have shape (N, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError(f"mesh faces must have shape (F, 3), got {faces.shape}")

    parts = _normalize_part_vertex_indices(part_vertex_indices)
    results = {}
    for part_name, indices in parts.items():
        part_indices = _validate_vertex_indices(indices, len(vertices), part_name)
        part_mask = np.zeros(len(vertices), dtype=bool)
        part_mask[part_indices] = True

        face_vertex_mask = part_mask[faces]
        if include_partial_faces:
            selected_face_mask = face_vertex_mask.any(axis=1)
        else:
            selected_face_mask = face_vertex_mask.all(axis=1)

        face_indices = np.flatnonzero(selected_face_mask).astype(np.int64)
        faces_original = faces[face_indices]

        # Use vertices referenced by selected faces so exported meshes contain no
        # isolated vertices. If no face is selected, keep the requested vertices.
        if faces_original.size > 0:
            kept_vertex_indices = np.unique(faces_original.reshape(-1)).astype(np.int64)
        else:
            kept_vertex_indices = part_indices

        results[part_name] = {
            "vertex_indices": kept_vertex_indices,
            "requested_vertex_indices": part_indices,
            "face_indices": face_indices,
            "faces_original": faces_original,
        }
    
    if visualize:
        export_path = visualize if isinstance(visualize, str) else None
        _visualize_part_faces(vertices, faces, results, export_path=export_path)

    return results


def _visualize_part_faces(vertices, faces, part_results, export_path=None):
    import open3d as o3d

    vis_vertices = vertices[faces].reshape(-1, 3)
    vis_faces = np.arange(len(vis_vertices), dtype=np.int64).reshape(-1, 3)

    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vis_vertices),
        triangles=o3d.utility.Vector3iVector(vis_faces),
    )
    mesh.compute_vertex_normals()

    face_colors = np.full((len(faces), 3), 0.72, dtype=np.float64)
    palette = np.array(
        [
            [0.90, 0.12, 0.10],
            [0.10, 0.45, 0.90],
            [0.10, 0.70, 0.25],
            [0.95, 0.70, 0.12],
            [0.55, 0.25, 0.85],
            [0.10, 0.75, 0.75],
            [0.95, 0.35, 0.65],
            [0.45, 0.65, 0.15],
        ],
        dtype=np.float64,
    )

    for color_idx, (_, data) in enumerate(part_results.items()):
        face_indices = np.asarray(data["face_indices"], dtype=np.int64)
        if face_indices.size == 0:
            continue
        face_colors[face_indices] = palette[color_idx % len(palette)]

    vertex_colors = np.repeat(face_colors, 3, axis=0)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    if export_path is not None:
        o3d.io.write_triangle_mesh(export_path, mesh, write_triangle_uvs=False)
        return

    o3d.visualization.draw_geometries([mesh])


def _normalize_part_vertex_indices(part_vertex_indices: PartIndices) -> Dict[str, Iterable[int]]:
    if isinstance(part_vertex_indices, Mapping):
        return dict(part_vertex_indices)

    if isinstance(part_vertex_indices, Sequence):
        return {f"part_{idx:03d}": indices for idx, indices in enumerate(part_vertex_indices)}

    raise TypeError("part_vertex_indices must be a mapping or a sequence of index iterables")


def _validate_vertex_indices(indices: Iterable[int], num_vertices: int, part_name: str) -> np.ndarray:
    indices = np.asarray(list(indices), dtype=np.int64).reshape(-1)
    if indices.size == 0:
        return indices

    if indices.min() < 0 or indices.max() >= num_vertices:
        raise ValueError(
            f"{part_name} vertex indices out of range [0, {num_vertices - 1}], "
            f"got min={indices.min()}, max={indices.max()}"
        )

    return np.unique(indices)

if __name__ == "__main__":
    # Example usage
    mesh_path = "output/vis/vis3d/init/438061_HOURGLASS_WORKOUT_for41/frame_0000.obj"
    with open("assets/smplh-seg/arm_leg_seg.json", "r") as f:
        part_vertex_indices = json.load(f)
    results = vertex2face_indices(
        mesh_path=mesh_path,
        part_vertex_indices=part_vertex_indices,
        include_partial_faces=False,
        visualize=True,
    )

    export_fname_ext = "assets/smplh-seg/arm_leg_seg_faces.json"
    for part_name, data in results.items():
        for key in data.keys():
            data[key] = data[key].tolist()  # Convert numpy arrays to lists for JSON serialization
    with open(export_fname_ext, "w") as f:
        json.dump(results, f, indent=4)
