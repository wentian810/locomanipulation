import os, sys
pwd = os.path.dirname(os.path.abspath(__file__))
code_root = os.path.abspath(os.path.join(pwd, '..'))
if not code_root in sys.path:
    sys.path.insert(0, code_root)

import numpy as np
import trimesh
from scipy.spatial import cKDTree # Cython version of KDTree for faster queries
import json


import torch


class Penetration:
    def __init__(self):
        pass

    @staticmethod
    def _ensure_frame_vertices(mesh_vertices):
        mesh_vertices = np.asarray(mesh_vertices)
        if mesh_vertices.ndim == 2:
            mesh_vertices = mesh_vertices[None, ...]
        if mesh_vertices.ndim != 3 or mesh_vertices.shape[-1] != 3:
            raise ValueError(
                f"mesh_vertices must have shape (T, N, 3) or (N, 3), got {mesh_vertices.shape}"
            )
        return mesh_vertices.astype(np.float32, copy=False)

    @staticmethod
    def _ensure_frame_faces(mesh_faces, num_frames):
        mesh_faces = np.asarray(mesh_faces)
        if mesh_faces.ndim == 2:
            if mesh_faces.shape[-1] != 3:
                raise ValueError(
                    f"mesh_faces must have shape (F, 3) or (T, F, 3), got {mesh_faces.shape}"
                )
            return mesh_faces.astype(np.int64, copy=False)
        if mesh_faces.ndim != 3 or mesh_faces.shape[-1] != 3:
            raise ValueError(
                f"mesh_faces must have shape (F, 3) or (T, F, 3), got {mesh_faces.shape}"
            )
        if mesh_faces.shape[0] != num_frames:
            raise ValueError(
                f"mesh_faces has {mesh_faces.shape[0]} frames, but mesh_vertices has {num_frames} frames"
            )
        return mesh_faces.astype(np.int64, copy=False)

    @staticmethod
    def _ensure_frame_query_mask(query_mask, num_frames, num_vertices):
        query_mask = np.asarray(query_mask)
        if query_mask.ndim == 1:
            query_mask = query_mask[None, None, ...]
        elif query_mask.ndim == 2:
            query_mask = query_mask[None, ...]
        elif query_mask.ndim != 3:
            raise ValueError(
                f"query_mask must have shape (N,), (X, N) or (T, X, N), got {query_mask.shape}"
            )
        if query_mask.shape[-1] != num_vertices:
            raise ValueError(
                f"query_mask last dimension must be {num_vertices}, got {query_mask.shape[-1]}"
            )
        if query_mask.shape[0] not in (1, num_frames):
            raise ValueError(
                f"query_mask frame dimension must be 1 or {num_frames}, got {query_mask.shape[0]}"
            )
        if query_mask.shape[0] == 1 and num_frames > 1:
            query_mask = np.repeat(query_mask, num_frames, axis=0)
        return query_mask.astype(bool, copy=False)

    @staticmethod
    def _query_kdtree(
        tree: cKDTree, 
        points: np.ndarray, 
    ):
        try:
            return tree.query(points, k=1, workers=-1)
        except TypeError:
            return tree.query(points, k=1)

    def check_self_penetration_igl(
        self, 
        mesh_vertices,
        mesh_faces, 
        query_point_mask,
        query_face_mask, 
        depth_threshold=0.1,
        sample_ratio: int = 1, 
    ):
        '''
        Check for self-penetration using libigl.
        Args:
            mesh_vertices: (T, N, 3) array of vertex positions
            mesh_faces: (F, 3) or (T, F, 3) array of face indices
            query_point_mask: list[np.array, shape=(N_i)]
            query_face_mask: list[np.array, shape=(F_i)]
            depth_threshold: float, penetration depth threshold in meters
            sample_ratio: int, sample every N frames for efficiency
        '''

        device = 'cuda'
        mesh_vertices = self._ensure_frame_vertices(mesh_vertices)
        num_frames, num_vertices, _ = mesh_vertices.shape
        mesh_faces    = self._ensure_frame_faces(mesh_faces, num_frames)

        num_batches = len(query_point_mask)
        num_frames_sampled = (num_frames + sample_ratio - 1) // sample_ratio
        penetration_mask     = np.zeros((num_frames_sampled, num_vertices), dtype=bool)
        penetration_depth    = np.zeros((num_frames_sampled, num_vertices), dtype=np.float32)

        wn_upper_bound = 0.6
        wn_lower_bound = 0.4

        total_mask = np.zeros((num_vertices,), dtype=bool)
        for batch_idx in range(num_batches):
            total_mask[query_point_mask[batch_idx]] = True

        for frame_idx in range(0, num_frames, sample_ratio):
            frame_idx_sampled = frame_idx // sample_ratio

            vertices = mesh_vertices[frame_idx] # (N, 3)
            faces = mesh_faces if mesh_faces.ndim == 2 else mesh_faces[frame_idx] # (F, 3)

            # 一次性查询所有点，不要分part batch
            query_points = vertices[total_mask] # (N_i, 3)
            wn = igl.fast_winding_number(vertices, faces, query_points)
            wn_soft = np.clip(
                (wn - wn_lower_bound) / (wn_upper_bound - wn_lower_bound), 
                0.0, 
                1.0, 
            )
            penetrated = wn_soft > 0.9 # (N_i,)
            penetration_mask[frame_idx_sampled][total_mask] = penetrated # (T, N)[frame_i, N] -> (N_i,) = (N_i)

            # ? 算深度这里还有bug
            # ! 也不算bug，主要是手掌那里有些点由于本身手掌曲度，即使不穿模也可能wn比较大。
            continue
            raise NotImplementedError('bug during computing penetration depth with mesh_sdf_cuda')
            for batch_idx in range(num_batches):
                point_mask = query_point_mask[batch_idx] # (N_i,)
                face_mask = query_face_mask[batch_idx] # (F_i,)

                # 算一下穿模距离
                # 当前part的face去掉
                target_faces = faces[face_mask] # (F_i, 3)
                dist, _ = Udf_Query.query(
                    torch.tensor(vertices, dtype=torch.float32, device=device), 
                    torch.tensor(target_faces, dtype=torch.int64, device=device), 
                    torch.tensor(vertices[point_mask], dtype=torch.float32, device=device), 
                )
                penetration_depth[frame_idx_sampled, point_mask] = dist.cpu().numpy()
            penetration_depth *= penetration_mask[frame_idx_sampled]

        return {
            'penetration_mask': penetration_mask,
            'penetration_depth': penetration_depth,
            # TODO
        }

    def check_self_penetration(
        self, 
        mesh_vertices,
        mesh_faces, 
        query_mask,
        depth_threshold=0.1,
    ):
        '''
        Check if points are penetrating the mesh.

        Parameters:
            mesh_vertices: (T, N, 3) np.ndarray of mesh vertices for each frame
            mesh_faces: (F, 3) or (T, F, 3) np.ndarray of mesh face indices, 
            query_mask: (X, N) or (T, X, N) np.ndarray of vertex indices mask to check for penetration
        Returns:
            penetration_info: dict containing penetration details

        给定点云和面片序列，首先初始化每个点的法向量(朝外)，然后将query_mask指定的顶点作为查询点，剩余的作为被查询点。
        对于每一帧，X 表示被查询的批次数量，比如第一批次是专门检查手部顶点是否穿透身体，第二批次是检查脚部顶点是否穿透身体等。
        对于每个查询点，使用KDTree找到其最近的被查询点，并计算其在被查询点法向量方向上的投影距离(有正负区分)。如果该值小组某个阈值(比如-0.01)，则认为发生了穿透。
        '''
        mesh_vertices = self._ensure_frame_vertices(mesh_vertices)
        num_frames, num_vertices, _ = mesh_vertices.shape
        mesh_faces = self._ensure_frame_faces(mesh_faces, num_frames)
        query_mask = self._ensure_frame_query_mask(query_mask, num_frames, num_vertices)

        num_batches = query_mask.shape[1]
        penetration_mask = np.zeros((num_frames, num_batches, num_vertices), dtype=bool)
        penetration_depth = np.zeros((num_frames, num_batches, num_vertices), dtype=np.float32)
        closest_target_index = np.full((num_frames, num_batches, num_vertices), -1, dtype=np.int64)
        closest_target_distance = np.full((num_frames, num_batches, num_vertices), np.inf, dtype=np.float32)

        depth_threshold = float(depth_threshold)

        for frame_idx in range(num_frames):
            vertices = mesh_vertices[frame_idx]
            faces = mesh_faces if mesh_faces.ndim == 2 else mesh_faces[frame_idx]

            if faces.size == 0:
                continue

            # process=False avoids expensive mesh repairs and keeps the loop lightweight.
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
            if vertex_normals.shape != vertices.shape:
                vertex_normals = np.zeros_like(vertices, dtype=np.float32)

            frame_query_mask = query_mask[frame_idx]

            for batch_idx in range(num_batches):
                qmask = frame_query_mask[batch_idx]
                if not np.any(qmask):
                    continue

                target_mask = ~qmask
                if not np.any(target_mask):
                    continue

                query_indices = np.flatnonzero(qmask)
                target_indices = np.flatnonzero(target_mask)
                query_points = vertices[query_indices]
                target_points = vertices[target_indices]

                tree = cKDTree(target_points)
                distances, local_nn = self._query_kdtree(tree, query_points)
                local_nn = np.asarray(local_nn, dtype=np.int64)
                distances = np.asarray(distances, dtype=np.float32)

                nearest_target_indices = target_indices[local_nn]
                nearest_target_points = target_points[local_nn]
                nearest_target_normals = vertex_normals[nearest_target_indices]

                # Signed projection along the target normal.
                signed_projection = np.einsum(
                    'ij,ij->i',
                    query_points - nearest_target_points,
                    nearest_target_normals,
                    optimize=True,
                ).astype(np.float32, copy=False)

                penetrated = signed_projection < -depth_threshold

                closest_target_index[frame_idx, batch_idx, query_indices] = nearest_target_indices
                closest_target_distance[frame_idx, batch_idx, query_indices] = distances

                if not np.any(penetrated):
                    continue

                penetration_mask[frame_idx, batch_idx, query_indices] = penetrated
                penetration_depth[frame_idx, batch_idx, query_indices] = np.where(
                    penetrated,
                    -signed_projection,
                    0.0,
                )

        penetration_info = {
            'penetration_mask': penetration_mask, # (T, X, N) bool array indicating which query points are penetrating
            'penetration_depth': penetration_depth, # (T, X, N) float array of penetration depth for penetrating points, 0 for non-penetrating points
            'closest_target_index': closest_target_index, # (T, X, N) int array of the index of the closest target point for each query point
            'closest_target_distance': closest_target_distance, # (T, X, N) float array of the distance to the closest target point for each query point
            'penetration_count_per_frame_batch': penetration_mask.sum(axis=-1), # (T, X) int array of the count of penetrating points for each frame and batch
            'has_penetration_per_frame_batch': penetration_mask.any(axis=-1), # (T, X) bool array indicating if there is any penetration for each frame and batch
            'depth_threshold': depth_threshold,
        }
        return penetration_info

def main():
    mesh_path = 'output/debug/shapes/init/438061_HOURGLASS_WORKOUT_for41/frame_0018.obj'
    mesh = trimesh.load(mesh_path)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    with open('assets/smplh-seg/coarse_seg.json', 'r') as f:
        seg_data = json.load(f)
    
    selected_indices = []
    for part_name, vertex_indices in seg_data.items():
        # print(f"{part_name}: {len(vertex_indices)} vertices")
        if 'hand' in part_name:
            selected_indices.extend(vertex_indices)

    selected_indices = np.array(selected_indices, dtype=np.int64)
    query_mask = np.zeros((1, vertices.shape[0]), dtype=bool)
    query_mask[0, selected_indices] = True

    penetration_detector = Penetration()
    penetration_info = penetration_detector.check_self_penetration(
        vertices[None, ...], # -> (1, N, 3)
        faces[None, ...], # -> (1, F, 3)
        query_mask[None, ...], # -> (1, 1, N)
        depth_threshold=0.01,
    )
    
    # 可视化
    from utils.vis3d_utils import Visualization3d
    
    # echo penetration_info
    print(f'sum of penetration_mask: {penetration_info["penetration_mask"].sum()} / total query points: {query_mask.sum()}')

    Visualization3d.export_single_mesh(
        vertices=vertices, 
        faces=faces, 
        export_path='output/debug/penetration_result.obj', 
        point_color=penetration_info['penetration_mask'][0, 0] * 1.0, 
    )
    # Visualization3d.export_single_pc(
    #     vertices, 
    #     export_path='output/debug/penetration_result.ply', 
    #     point_color=penetration_info['penetration_mask'][0, 0] * 1.0,
    # )

if __name__ == "__main__":
    main()
