import argparse
import os
import torch
import json
import cv2
import sys
import numpy as np
import multiprocessing as mp
from typing import List
import imageio.v2 as imageio  # Suppress DeprecationWarning
import trimesh
from scipy.spatial.transform import Rotation
from VOT import Cutie, Tracker_2D  
from utils.kalman_filter_6d import KalmanFilter6D



src_path = os.path.join(os.path.dirname(__file__), "..")
foundationpose_path = os.path.join(src_path, "FoundationPose")
if src_path not in sys.path:
    sys.path.append(src_path)
if foundationpose_path not in sys.path:
    sys.path.append(foundationpose_path)


def get_sorted_frame_list(dir: str) -> List:
    files = os.listdir(dir)
    if not files:
        return []
    files = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]
    if not files:
        return []
    if files[0].count('.') == 1:
        files.sort(key=lambda x: int(x.split('.')[0]))
    elif files[0].count('.') == 2:
        files.sort(key=lambda x: int(x.split('.')[0] + x.split('.')[1]))
    return files


def adjust_pose_to_image_point(
        ob_in_cam: torch.Tensor,
        K: torch.Tensor,
        x: float = -1.,
        y: float = -1.,
) -> torch.Tensor:
    """
    Adjusts the 6D pose(s) so that the projection matches the given 2D coordinate (x, y).

    Parameters:
    - ob_in_cam: Original 6D pose(s) as [4,4] or [B,4,4] tensor.
    - K: Camera intrinsic matrix (3x3 tensor).
    - x, y: Desired 2D coordinates on the image plane.

    Returns:
    - ob_in_cam_new: Adjusted pose(s) in same shape as input (tensor).
    """
    device = ob_in_cam.device
    dtype = ob_in_cam.dtype

    is_batched = ob_in_cam.ndim == 3
    if not is_batched:
        ob_in_cam = ob_in_cam.unsqueeze(0)  # [1, 4, 4]

    B = ob_in_cam.shape[0]
    ob_in_cam_new = torch.eye(4, device=device, dtype=dtype).repeat(B, 1, 1)

    for i in range(B):
        R = ob_in_cam[i, :3, :3]
        t = ob_in_cam[i, :3, 3]

        tx, ty = get_pose_xy_from_image_point(ob_in_cam[i], K, x, y)
        t_new = torch.tensor([tx, ty, t[2]], device=device, dtype=dtype)

        ob_in_cam_new[i, :3, :3] = R
        ob_in_cam_new[i, :3, 3] = t_new

    return ob_in_cam_new if is_batched else ob_in_cam_new[0]


def get_pose_xy_from_image_point(
        ob_in_cam: torch.Tensor,
        K: torch.Tensor,
        x: float = -1.,
        y: float = -1.,
        tz_override: float = None,
) -> tuple:
    """
    Computes new (tx, ty) in camera space such that the projection matches image point (x, y).

    Parameters:
    - ob_in_cam: 4x4 pose tensor.
    - K: 3x3 intrinsic matrix tensor.
    - x, y: Desired image coordinates.

    Returns:
    - tx, ty: New x/y in camera coordinate system.
    """

    is_batched = ob_in_cam.ndim == 3
    if is_batched:
        ob_in_cam_new = ob_in_cam[0].cpu()  # [1, 4, 4]
    else:
        ob_in_cam_new = ob_in_cam.cpu()

    if x == -1. or y == -1.:
        return x, y

    t = ob_in_cam_new[:3, 3]

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    tz = tz_override if tz_override is not None else t[2]

    tx = (x - cx) * tz / fx
    ty = (y - cy) * tz / fy

    return tx, ty


# def adjust_pose_to_image_point(
#         ob_in_cam_ori: torch.tensor, 
#         K: np.ndarray, 
#         x: float = -1., 
#         y: float = -1.,
# ) -> np.ndarray:
#     """
#     Adjusts the 6D pose so that its projection matches the given 2D coordinate (x, y).

#     Parameters:
#     - K: Camera intrinsic matrix (3x3).
#     - ob_in_cam: Original 6D pose as a 4x4 transformation matrix.
#     - x, y: Desired 2D coordinates on the image plane.

#     Returns:
#     - ob_in_cam_new: Adjusted 6D pose as a 4x4 transformation matrix.
#     """
#     # Extract rotation (R) and translation (t) from the original pose
#     device = ob_in_cam_ori.device
#     if ob_in_cam_ori.ndim == 3:
#         ob_in_cam = ob_in_cam_ori[0].detach().cpu().numpy()
#     else:
#         ob_in_cam = ob_in_cam_ori.detach().cpu().numpy()
#     R = ob_in_cam[:3, :3]
#     t = ob_in_cam[:3, 3]

#     tx, ty = get_pose_xy_from_image_point(ob_in_cam, K, x, y)

#     # Update the translation vector
#     t_new = np.array([tx, ty, t[2]])

#     # Construct the new transformation matrix with the updated translation
#     ob_in_cam_new = np.eye(4)
#     ob_in_cam_new[:3, :3] = R
#     ob_in_cam_new[:3, 3] = t_new


#     return torch.from_numpy(ob_in_cam_new).to(device)


# def get_pose_xy_from_image_point(
#         ob_in_cam: np.ndarray, 
#         K: np.ndarray, 
#         x: float = -1., 
#         y: float = -1.,
# ) -> np.ndarray:
#     """
#     Adjusts the 6D pose so that its projection matches the given 2D coordinate (x, y).

#     Parameters:
#     - K: Camera intrinsic matrix (3x3).
#     - ob_in_cam: Original 6D pose as a 4x4 transformation matrix.
#     - x, y: Desired 2D coordinates on the image plane.

#     Returns:
#     - ob_in_cam_new: Adjusted 6D pose as a 4x4 transformation matrix.
#     """

#     if x == -1. or y == -1.:
#         return x, y

#     # Extract rotation (R) and translation (t) from the original pose
#     t = ob_in_cam[:3, 3]

#     # Camera intrinsic parameters
#     fx = K[0, 0]
#     fy = K[1, 1]
#     cx = K[0, 2]
#     cy = K[1, 2]

#     # Keep the depth (tz) the same
#     tz = t[2]

#     # Use depth to match the desired 2D point
#     tx = (x - cx) * tz / fx
#     ty = (y - cy) * tz / fy     

#     return tx, ty


def project_3d_to_2d(point_3d_homogeneous, K, ob_in_cam):
    # Transform point to camera frame
    point_cam = ob_in_cam @ point_3d_homogeneous

    # Perspective division to get normalized image coordinates
    x = point_cam[0] / point_cam[2]
    y = point_cam[1] / point_cam[2]

    # Apply camera intrinsics
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]

    return (int(u), int(v))


def project_real_dims_bbox_to_2d(
        real_dims: dict,
        pose: np.ndarray,
        K: np.ndarray,
        img_W: int,
        img_H: int,
) -> tuple:
    """
        将 real_dims 的 3D 包围盒用当前位姿投影到 2D, 得到完整物体的图像范围。
    """
    W = real_dims.get("width", 0)
    H = real_dims.get("height", 0)
    D = real_dims.get("depth", 0)
    if W <= 0 or H <= 0 or D <= 0:
        return None

    # 8 个角点, bbox 中心在原点
    corners = np.array([
        [-W / 2, -H / 2, -D / 2],
        [W / 2, -H / 2, -D / 2],
        [-W / 2, H / 2, -D / 2],
        [-W / 2, -H / 2, D / 2],
        [W / 2, H / 2, -D / 2],
        [W / 2, -H / 2, D / 2],
        [-W / 2, H / 2, D / 2],
        [W / 2, H / 2, D / 2],
    ], dtype=np.float64)

    R = pose[:3, :3]
    t = pose[:3, 3]
    corners_cam = corners @ R.T + t  # (8, 3)

    # 过滤相机后方的点
    in_front = corners_cam[:, 2] > 0.001
    if in_front.sum() < 4:
        return None
    corners_cam = corners_cam[in_front]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = fx * corners_cam[:, 0] / corners_cam[:, 2] + cx
    v = fy * corners_cam[:, 1] / corners_cam[:, 2] + cy

    u_min = max(0, int(np.floor(u.min())))
    u_max = min(img_W - 1, int(np.ceil(u.max())))
    v_min = max(0, int(np.floor(v.min())))
    v_max = min(img_H - 1, int(np.ceil(v.max())))

    if u_max <= u_min or v_max <= v_min:
        return None
    return (u_min, v_min, u_max - u_min, v_max - v_min)


def get_mat_from_6d_pose_arr(pose_arr):
    # 提取位移 (xyz)
    xyz = pose_arr[:3]
    
    # 提取欧拉角
    euler_angles = pose_arr[3:]
    
    # 从欧拉角生成旋转矩阵
    rotation = Rotation.from_euler('xyz', euler_angles, degrees=False)
    rotation_matrix = rotation.as_matrix()
    
    # 创建 4x4 变换矩阵
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = xyz
    
    return transformation_matrix

def get_6d_pose_arr_from_mat(pose):
    if torch.is_tensor(pose):
        is_batched = pose.ndim == 3
        if is_batched:
            pose_np = pose[0].cpu().numpy()
        else:
            pose_np = pose.cpu().numpy()
    else:
        pose_np = pose

    xyz = pose_np[:3, 3]
    rotation_matrix = pose_np[:3, :3]
    euler_angles = Rotation.from_matrix(rotation_matrix).as_euler('xyz', degrees=False)
    return np.r_[xyz, euler_angles]


def pose_track(
        rgb_seq_path: str,
        depth_seq_path: str,
        mesh_path: str,
        init_mask_path: str,
        cam_K: np.ndarray,
        pose_output_path: str,
        mask_visualization_path: str,
        bbox_visualization_path: str,
        pose_visualization_path: str,
        est_refine_iter: int,
        track_refine_iter: int,
        activate_2d_tracker: bool = False,
        activate_kalman_filter: bool = False,
        real_dims: dict = None,
):
    #################################################
    # Read the initial mask
    #################################################
    init_mask_raw = cv2.imread(init_mask_path, cv2.IMREAD_GRAYSCALE)
    if init_mask_raw is None:
        print(f"Failed to read mask file {init_mask_path}.")
        return
    init_mask = init_mask_raw.astype(bool) 

    #################################################
    # Read the frame list
    #################################################
    frame_color_list = get_sorted_frame_list(rgb_seq_path)
    frame_depth_list = get_sorted_frame_list(depth_seq_path)
    if not frame_color_list or not frame_depth_list:
        print(f"No RGB frames found.")
        return

    #################################################
    # Load the initial frame
    #################################################
    init_frame_filename = frame_color_list[0]
    init_frame_path = os.path.join(rgb_seq_path, init_frame_filename)
    init_frame = cv2.imread(init_frame_path)
    if init_frame is None:
        print(f"Failed to read initial frame.")
        return

    #################################################
    # Load the mesh
    #################################################
    from FoundationPose.estimater import trimesh_add_pure_colored_texture
    
    mesh_file = os.path.join(mesh_path)
    if not os.path.exists(mesh_file):
        print(f"Mesh file not found.")
        return
    mesh = trimesh.load(mesh_file)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    # Convert units to meters
    mesh.apply_scale(args.apply_scale)
    if args.force_apply_color:
        mesh = trimesh_add_pure_colored_texture(mesh, color=np.array(args.apply_color), resolution=10)

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    # ── PCA 检测旋转对称性 ──
    sym_axis = None   # 圆柱对称轴
    is_sphere = False  # 球体-所有旋转自由度均不可观测
    if len(mesh.vertices) > 10:
        verts = mesh.vertices  # (N,3)
        centered = verts - verts.mean(axis=0)
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = eigvals.argsort()[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        if eigvals[0] > 1e-6:
            r_ratio = eigvals[0] / max(eigvals[2], 1e-6) 
            r_long = eigvals[1] / eigvals[0]            
            r_cross = eigvals[2] / max(eigvals[1], 1e-6) 

            if r_ratio < 2.0:
                is_sphere = True
                print(f"  [Symmetry] 检测到球体 (PCA λ=[{eigvals[0]:.4f},{eigvals[1]:.4f},{eigvals[2]:.4f}], "
                      f"max/min={r_ratio:.2f}), 将冻结全部旋转")
            elif r_long < 0.5 and r_cross > 0.7:
                sym_axis = eigvecs[:, 0]
                sym_axis = sym_axis / np.linalg.norm(sym_axis)
                print(f"  [Symmetry] 检测到旋转对称轴 "
                      f"(PCA λ=[{eigvals[0]:.4f},{eigvals[1]:.4f},{eigvals[2]:.4f}], "
                      f"axis=[{sym_axis[0]:.3f},{sym_axis[1]:.3f},{sym_axis[2]:.3f}])")
    if sym_axis is None and not is_sphere:
        print(f"  [Symmetry] 未检测到明显旋转对称性")

    #################################################
    # Instantiate the 6D pose estimator
    #################################################

    from FoundationPose.estimater import (
        ScorePredictor,
        PoseRefinePredictor,
        dr,
        FoundationPose,
        logging,
        draw_posed_3d_box,
        draw_xyz_axis,
    )
    import logging

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        glctx=glctx,
    )
    logging.info("Estimator initialization done")

    #################################################
    # Instantiate the 2D tracker
    #################################################

    if activate_2d_tracker:     # Default using Cutie as a 2D tracker
        tracker_2D = Cutie()
    else:
        tracker_2D = Tracker_2D()


    #################################################
    # 6D pose tracking
    #################################################

    if activate_kalman_filter:
        kf = KalmanFilter6D(args.kf_measurement_noise_scale)

    total_frames = len(frame_color_list)
    pose_seq = [None] * total_frames  # Initialize as None
    kf_mean, kf_covariance = None, None
    was_occluded = False  # 追踪是否上一帧被遮挡
    consecutive_bad_frames = 0  # 连续丢失帧计数
    consecutive_unhealthy = 0  # Cutie 连续不正常帧数
    kf_cov_needs_reset = False  # 是否重置 KF 协方差
    prev_healthy_bbox_area = 0.0  # 上一帧正常 bbox 面积
    consecutive_projection_underestimate = 0

    # 高动态场景自适应 KF 过程噪声
    prev_cx_raw, prev_cy_raw = None, None
    motion_level = 0.0
    MOTION_DECAY = 0.4
    HIGH_MOTION_THRESH = 12.0 

    # Forward processing from initial frame
    for i in range(0, total_frames):
        #################################################
        # Read the frame
        #################################################
        frame_color_filename = frame_color_list[i]
        frame_depth_filename = frame_depth_list[i]
        color = imageio.imread(os.path.join(rgb_seq_path, frame_color_filename))[..., :3]
        color = cv2.resize(color, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)

        depth = cv2.imread(os.path.join(depth_seq_path, frame_depth_filename), -1) / 1e3
        depth = cv2.resize(depth, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
        depth[(depth < 0.001) | (depth >= np.inf)] = 0

        if color is None or depth is None:
            print(f"Failed to read color frame {frame_color_filename} or depth map {frame_depth_filename}")
            continue

        #################################################
        # 6D pose tracking
        #################################################

        if i == 0:
            mask = init_mask.astype(np.uint8) * 255
            pose = est.register(K=cam_K, rgb=color, depth=depth, ob_mask=mask, iteration=est_refine_iter)
            if activate_kalman_filter:
                kf_mean, kf_covariance = kf.initiate(get_6d_pose_arr_from_mat(pose))

            
            mask_visualization_color_filename = None
            bbox_visualization_color_filename = None
            if mask_visualization_path is not None:
                os.makedirs(mask_visualization_path, exist_ok=True)
                mask_visualization_color_filename = os.path.join(mask_visualization_path, frame_color_filename)
            if bbox_visualization_path is not None:
                os.makedirs(bbox_visualization_path, exist_ok=True)
                bbox_visualization_color_filename = os.path.join(bbox_visualization_path, frame_color_filename)
            if activate_2d_tracker:
                tracker_2D.initialize(
                    color,
                    init_info={"mask": init_mask_raw}, 
                    mask_visualization_path=mask_visualization_color_filename,
                    bbox_visualization_path=bbox_visualization_color_filename
                )
        else:
            mask_visualization_color_filename = None
            bbox_visualization_color_filename = None
            if mask_visualization_path is not None:
                os.makedirs(mask_visualization_path, exist_ok=True)
                mask_visualization_color_filename = os.path.join(mask_visualization_path, frame_color_filename)
            if bbox_visualization_path is not None:
                os.makedirs(bbox_visualization_path, exist_ok=True)
                bbox_visualization_color_filename = os.path.join(bbox_visualization_path, frame_color_filename)
            if activate_2d_tracker:
                bbox_2d = tracker_2D.track(
                    color,
                    mask_visualization_path=mask_visualization_color_filename,
                    bbox_visualization_path=bbox_visualization_color_filename
                )

            bbox_valid = (isinstance(bbox_2d, (list, tuple)) and len(bbox_2d) >= 4
                          and bbox_2d[2] > 10 and bbox_2d[3] > 10)

            # KF 预测的当前帧 tz 
            kf_tz = None
            if activate_kalman_filter and kf_mean is not None:
                kf_tz = float(kf_mean[2])
            elif est.pose_last is not None:
                tmp = est.pose_last
                if tmp.dim() == 3:
                    tmp = tmp[0]
                kf_tz = float(tmp.cpu().numpy()[2, 3])

            # 用 KF 预测位姿投影 real_dims → 2D 
            if activate_kalman_filter and kf_mean is not None:
                pred_pose_np = get_mat_from_6d_pose_arr(kf_mean[:6])
            elif est.pose_last is not None:
                tmp = est.pose_last
                if tmp.dim() == 3:
                    tmp = tmp[0]
                pred_pose_np = tmp.cpu().numpy()
            else:
                pred_pose_np = None

            projected_bbox = None
            if (real_dims is not None and pred_pose_np is not None
                    and real_dims.get("width", 0) > 0):
                projected_bbox = project_real_dims_bbox_to_2d(
                    real_dims, pred_pose_np, cam_K, color.shape[1], color.shape[0])

            # 确定中心 + 深度采样窗口
            cutie_healthy = True
            projection_underestimate = False 
            if bbox_valid:
                cw, ch = bbox_2d[2], bbox_2d[3]
                cutie_cx = bbox_2d[0] + cw / 2.0
                cutie_cy = bbox_2d[1] + ch / 2.0
                if projected_bbox is not None:
                    pw_, ph_ = projected_bbox[2], projected_bbox[3]
                    proj_cx = projected_bbox[0] + pw_ / 2.0
                    proj_cy = projected_bbox[1] + ph_ / 2.0
                    if pw_ > 0 and ph_ > 0:
                        area_ratio = (cw * ch) / (pw_ * ph_)
                        if area_ratio > 2.0:
                            projection_underestimate = True
                            cutie_healthy = True 
                            cx, cy = cutie_cx, cutie_cy
                            bw = max(cw, pw_)
                            bh = max(ch, ph_)
                        elif area_ratio < 0.4:
                            cutie_healthy = False
                            cx, cy = proj_cx, proj_cy
                            bw, bh = pw_, ph_
                        else:
                            cx, cy = cutie_cx, cutie_cy
                            bw = max(cw, pw_)
                            bh = max(ch, ph_)
                    else:
                        cx, cy = cutie_cx, cutie_cy
                        bw, bh = cw, ch
                else:
                    cx, cy = cutie_cx, cutie_cy
                    bw, bh = cw, ch
            elif projected_bbox is not None:
                cx = projected_bbox[0] + projected_bbox[2] / 2.0
                cy = projected_bbox[1] + projected_bbox[3] / 2.0
                bw, bh = projected_bbox[2], projected_bbox[3]
                cutie_healthy = False
            else:
                cx = cy = 0.0
                bw = bh = 10
                cutie_healthy = False

            # 深度有效性检测
            bbox_has_depth = False
            tz_from_depth = None
            if (not isinstance(cx, float) or not isinstance(cy, float) or
                not (0 <= cx < depth.shape[1] and 0 <= cy < depth.shape[0])):
                pass 
            else:
                hw = max(2, bw * 0.15)
                hh = max(2, bh * 0.15)
                x0 = max(0, int(cx - hw))
                x1 = min(depth.shape[1], int(cx + hw + 1))
                y0 = max(0, int(cy - hh))
                y1 = min(depth.shape[0], int(cy + hh + 1))
                if x1 > x0 and y1 > y0:
                    patch = depth[y0:y1, x0:x1]
                    valid_depths = patch[(patch > 0.001) & np.isfinite(patch)]
                    bbox_has_depth = (len(valid_depths) >= 5)
                    if bbox_has_depth:
                        tz_from_depth = float(np.median(valid_depths))

            is_occluded = bbox_valid and not bbox_has_depth

            # ── 追踪投影低估连续帧数 ──
            if projection_underestimate:
                consecutive_projection_underestimate += 1
            else:
                consecutive_projection_underestimate = 0

            # ── 检测 bbox 面积突变 ──
            bbox_expansion_recovery = False
            cur_bbox_area = (bbox_2d[2] * bbox_2d[3]) if bbox_valid else 0
            if (bbox_valid and bbox_has_depth and cutie_healthy
                    and prev_healthy_bbox_area > 0
                    and consecutive_projection_underestimate >= 2):
                if cur_bbox_area > prev_healthy_bbox_area * 2.0:
                    bbox_expansion_recovery = True
                    print(f"  [BBoxExpansion] 帧 {i}: bbox 面积从 "
                          f"{prev_healthy_bbox_area:.0f} 突增至 {cur_bbox_area:.0f} "
                          f"({cur_bbox_area/prev_healthy_bbox_area:.1f}x), "
                          f"触发 re-register")

            # ── 追踪 Cutie 不健康连续帧数 ──
            just_recovered = (consecutive_unhealthy >= 2) and cutie_healthy
            if not cutie_healthy:
                consecutive_unhealthy += 1
                prev_healthy_bbox_area = 0.0 
            else:
                consecutive_unhealthy = 0
                prev_healthy_bbox_area = cur_bbox_area if cur_bbox_area > 0 else prev_healthy_bbox_area

            cutie_corrupted = False
            if (bbox_valid and bbox_has_depth and projected_bbox is not None
                    and bbox_2d[2] > 10 and bbox_2d[3] > 10
                    and real_dims is not None and real_dims.get("width", 0) > 0):
                pw, ph = projected_bbox[2], projected_bbox[3]
                if pw > 0 and ph > 0:
                    ratio = (bbox_2d[2] * bbox_2d[3]) / (pw * ph)
                    if ratio < 0.5:
                        cutie_corrupted = True

            # 连续丢失帧计数
            RE_REGISTER_THRESHOLD = 2
            need_cutie_reset = False

            if not bbox_valid or is_occluded:
                consecutive_bad_frames += 1
            else:
                if consecutive_bad_frames >= RE_REGISTER_THRESHOLD:
                    need_cutie_reset = True
                consecutive_bad_frames = 0

            # ── Cutie 记忆污染修复与翻转恢复 ──
            if (activate_2d_tracker and bbox_valid
                    and (need_cutie_reset or cutie_corrupted or bbox_expansion_recovery)):
                if bbox_expansion_recovery and not (need_cutie_reset or cutie_corrupted):
                    reason = "翻转恢复"
                    reset_bbox = list(bbox_2d)
                else:
                    reason = "丢失恢复" if need_cutie_reset else "记忆污染"
                    reset_bbox = list(projected_bbox) if projected_bbox is not None else list(bbox_2d)
                print(f"  [CutieReset] 帧 {i}: {reason}, "
                      f"bbox={bbox_2d[2]}x{bbox_2d[3]} "
                      f"→ 重置={reset_bbox[2]}x{reset_bbox[3]}")

                px1, py1, pw_, ph_ = reset_bbox
                px2, py2 = px1 + pw_, py1 + ph_
                rH, rW = depth.shape
                reset_mask = np.zeros((rH, rW), dtype=np.uint8)
                reset_mask[max(0, py1):min(rH, py2), max(0, px1):min(rW, px2)] = 255
                reset_mask = cv2.erode(reset_mask, np.ones((5, 5), np.uint8), iterations=1)
                tracker_2D.initialize(
                    color,
                    init_info={"mask": reset_mask},
                    mask_visualization_path=mask_visualization_color_filename,
                    bbox_visualization_path=bbox_visualization_color_filename
                )
                # 覆盖本帧 bbox/cx/cy
                bbox_2d = reset_bbox
                bbox_valid = True
                is_occluded = False
                was_occluded = True   # 触发 tz 修正
                consecutive_bad_frames = 0
                kf_cov_needs_reset = True  # P 膨胀 → 重置协方差恢复平滑
                # 重算中心 + 深度窗口
                cw, ch = reset_bbox[2], reset_bbox[3]
                cx = reset_bbox[0] + cw / 2.0
                cy = reset_bbox[1] + ch / 2.0
                bw, bh = cw, ch
                # 在新中心重新采样深度 + 更新 tz
                hw2 = max(2, bw * 0.15)
                hh2 = max(2, bh * 0.15)
                x0_2 = max(0, int(cx - hw2))
                x1_2 = min(depth.shape[1], int(cx + hw2 + 1))
                y0_2 = max(0, int(cy - hh2))
                y1_2 = min(depth.shape[0], int(cy + hh2 + 1))
                if x1_2 > x0_2 and y1_2 > y0_2:
                    patch2 = depth[y0_2:y1_2, x0_2:x1_2]
                    valid2 = patch2[(patch2 > 0.001) & np.isfinite(patch2)]
                    if len(valid2) >= 5:
                        tz_from_depth = float(np.median(valid2))
                        bbox_has_depth = True

            # ── 自适应 KF 过程噪声 ──
            if bbox_valid and cx > 0 and cy > 0:
                if prev_cx_raw is not None:
                    disp = np.sqrt((cx - prev_cx_raw)**2 + (cy - prev_cy_raw)**2)
                    motion_level = MOTION_DECAY * motion_level + (1 - MOTION_DECAY) * disp
                prev_cx_raw, prev_cy_raw = cx, cy
            else:
                prev_cx_raw, prev_cy_raw = None, None
                motion_level = 0.0

            is_high_motion = (motion_level > HIGH_MOTION_THRESH)
            if activate_kalman_filter:
                kf.set_process_noise_scale(3.0 if is_high_motion else 1.0)

            # ═══════════════════════════════════════════════════════════════
            # 位姿修正
            # ═══════════════════════════════════════════════════════════════
            if i > 0:
                is_recovering = was_occluded and bbox_valid

                # Step 1: 恢复帧，tz 修正
                if is_recovering and tz_from_depth is not None:
                    pl = est.pose_last.clone()
                    if pl.dim() == 3:
                        pl[0, 2, 3] = tz_from_depth
                    else:
                        pl[2, 3] = tz_from_depth
                    est.pose_last = pl

                # Step 2: 用 bbox 中心修正初始位姿
                if activate_2d_tracker and bbox_valid and cutie_healthy:
                    est.pose_last = adjust_pose_to_image_point(
                        ob_in_cam=est.pose_last, K=cam_K, x=cx, y=cy)

                # Step 3: bbox 丢失且无 KF，跳过本帧
                if not bbox_valid and not activate_kalman_filter:
                    pose_seq[i] = pose_seq[i - 1] if pose_seq[i - 1] is not None else np.eye(4)
                    was_occluded = is_occluded
                    continue

                # Step 4: bbox 丢失但有 KF，提供初始位姿
                if not bbox_valid and activate_kalman_filter:
                    if consecutive_bad_frames > 10:
                        kf_mean[6:12] = 0.0 
                    else:
                        kf_mean, kf_covariance = kf.predict(kf_mean, kf_covariance)
                    est.pose_last = torch.from_numpy(
                        get_mat_from_6d_pose_arr(kf_mean[:6])
                    ).unsqueeze(0).to(est.pose_last.device)

                # Step 5: FoundationPose refine (track_one)
                pose = est.track_one(rgb=color, depth=depth, K=cam_K, iteration=track_refine_iter)

                # Step 6: KF 融合 track_one 结果 + Cutie xy
                if activate_kalman_filter:
                    # ── CutieReset 后 KF 协方差重置 ──
                    if kf_cov_needs_reset:
                        kf_mean, kf_covariance = kf.initiate(
                            get_6d_pose_arr_from_mat(pose))
                        if tz_from_depth is not None:
                            kf_mean[2] = tz_from_depth
                        kf_cov_needs_reset = False
                    elif bbox_valid and bbox_has_depth and cutie_healthy:
                        # 正常帧: 用 track_one 6D + Cutie xy 双源融合
                        kf_mean, kf_covariance = kf.update(
                            kf_mean, kf_covariance,
                            get_6d_pose_arr_from_mat(pose))
                        # 测量2: Cutie 的 xy 中心
                        measurement_xy = np.array(
                            get_pose_xy_from_image_point(
                                ob_in_cam=torch.from_numpy(pose).to(est.pts.device),
                                K=cam_K, x=cx, y=cy,
                                tz_override=float(kf_mean[2])))
                        kf_mean, kf_covariance = kf.update_from_xy(
                            kf_mean, kf_covariance, measurement_xy)
                        # 恢复帧: 用深度修正 tz
                        if is_recovering and tz_from_depth is not None:
                            kf_mean[2] = tz_from_depth
                    elif bbox_valid:
                        # 遮挡中: KF predict, 不 update
                        kf_mean, kf_covariance = kf.predict(kf_mean, kf_covariance)

                    est.pose_last = torch.from_numpy(
                        get_mat_from_6d_pose_arr(kf_mean[:6])
                    ).unsqueeze(0).to(est.pose_last.device)
                else:
                    est.pose_last = torch.from_numpy(
                        pose.reshape(4, 4)
                    ).float().unsqueeze(0).to(est.pts.device)

                # ── 恢复逻辑 ──
                needs_reregister = (just_recovered
                                    or consecutive_projection_underestimate >= 3
                                    or bbox_expansion_recovery)
                if (needs_reregister and bbox_has_depth
                        and activate_2d_tracker):
                    if bbox_expansion_recovery:
                        reason = "bbox面积突变 (翻转恢复)"
                    elif just_recovered:
                        reason = f"Cutie 恢复 (连续 {consecutive_unhealthy} 帧不健康)"
                    else:
                        reason = (f"投影持续低估 (连续 "
                                  f"{consecutive_projection_underestimate} 帧 "
                                  f"area_ratio>2.0), 位姿旋转可能已漂移")
                    print(f"  [ReRegister] 帧 {i}: {reason}, 触发 re-register")
                    bx1 = int(max(0, bbox_2d[0]))
                    by1 = int(max(0, bbox_2d[1]))
                    bx2 = int(min(depth.shape[1], bbox_2d[0] + bbox_2d[2]))
                    by2 = int(min(depth.shape[0], bbox_2d[1] + bbox_2d[3]))
                    if bx2 > bx1 and by2 > by1:
                        reg_mask = np.zeros(depth.shape, dtype=np.uint8)
                        reg_mask[by1:by2, bx1:bx2] = 255
                        reg_mask = cv2.erode(reg_mask, np.ones((3, 3), np.uint8), iterations=1)
                        pose = est.register(K=cam_K, rgb=color, depth=depth,
                                           ob_mask=reg_mask, iteration=est_refine_iter)
                        # 重置 KF
                        if activate_kalman_filter:
                            kf_mean, kf_covariance = kf.initiate(
                                get_6d_pose_arr_from_mat(pose))
                            if tz_from_depth is not None:
                                kf_mean[2] = tz_from_depth
                            est.pose_last = torch.from_numpy(
                                get_mat_from_6d_pose_arr(kf_mean[:6])
                            ).unsqueeze(0).to(est.pose_last.device)
                        else:
                            est.pose_last = torch.from_numpy(
                                pose.reshape(4, 4)
                            ).float().unsqueeze(0).to(est.pts.device)
                        # 重初始化 Cutie
                        tracker_2D.initialize(
                            color,
                            init_info={"mask": reg_mask},
                            mask_visualization_path=mask_visualization_color_filename,
                            bbox_visualization_path=bbox_visualization_color_filename
                        )
                        print(f"  [ReRegister] register 完成, KF + Cutie 已重置")
                        consecutive_projection_underestimate = 0 

                # 更新状态
                was_occluded = is_occluded

                # KF 向前预测一步
                if (activate_kalman_filter and bbox_valid
                        and bbox_has_depth and cutie_healthy):
                    kf_mean, kf_covariance = kf.predict(kf_mean, kf_covariance)
            
            
        # ── 旋转对称约束: 球体冻结全部旋转 / 圆柱移除绕轴旋转漂移 ──
        if i > 0 and pose_seq[i - 1] is not None:
            if is_sphere:
                # 球体: 旋转完全不可观测 → 冻结为初始注册朝向
                pose[:3, :3] = pose_seq[i - 1][:3, :3].copy()
            elif sym_axis is not None:
                # 圆柱: 移除绕对称轴的旋转分量
                R_cur = pose[:3, :3].copy()
                R_prev = pose_seq[i - 1][:3, :3]
                # 帧间旋转增量 (delta)
                R_delta = R_cur @ R_prev.T
                rotvec_delta = Rotation.from_matrix(R_delta).as_rotvec()
                # 对称轴在上帧相机系中的方向
                sym_cam = R_prev @ sym_axis
                sym_cam = sym_cam / (np.linalg.norm(sym_cam) + 1e-10)
                # 投影 → 移除绕轴分量
                proj = float(np.dot(rotvec_delta, sym_cam))
                rotvec_corrected = rotvec_delta - proj * sym_cam
                R_delta_corrected = Rotation.from_rotvec(rotvec_corrected).as_matrix()
                # 用修正后的 delta 重建旋转
                R_new = R_delta_corrected @ R_prev
                pose[:3, :3] = R_new

        pose_seq[i] = pose.reshape(4, 4)

        if pose_visualization_path is not None:
            # depth_normalized = cv2.normalize(np.clip(depth, 0, 900), None, 0, 255, cv2.NORM_MINMAX)
            # depth_8bit = depth_normalized.astype(np.uint8)
            # depth_colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

            center_pose = pose @ np.linalg.inv(to_origin)
            lw_scale = color.shape[1] / 640.0
            vis_color = draw_posed_3d_box(cam_K, img=color, ob_in_cam=center_pose, bbox=bbox,
                                           line_color=(255, 0, 0),
                                           linewidth=max(1, int(1 * lw_scale)))
            vis_color = draw_xyz_axis(
                vis_color,
                ob_in_cam=center_pose,
                scale=0.1,
                K=cam_K,
                thickness=max(2, int(3 * lw_scale)),
                transparency=0,
                is_input_rgb=True,
            )
            # vis_depth = draw_posed_3d_box(cam_K, img=depth_colored, ob_in_cam=center_pose, bbox=bbox)
            # vis_depth = draw_xyz_axis(
            #     vis_depth,
            #     ob_in_cam=center_pose,
            #     scale=0.1,
            #     K=cam_K,
            #     thickness=3,
            #     transparency=0,
            #     is_input_rgb=False,
            # )
            
            if not os.path.exists(pose_visualization_path):
                os.makedirs(pose_visualization_path, exist_ok=True)
            
            pose_visualization_color_filename = os.path.join(pose_visualization_path, frame_color_filename)
            imageio.imwrite(
                pose_visualization_color_filename, vis_color
            )
            # pose_visualization_depth_filename = os.path.join(pose_visualization_path, frame_depth_filename)
            # imageio.imwrite(
            #     pose_visualization_depth_filename, vis_depth
            # )

    #################################################
    # Save pose sequence
    #################################################

    pose_seq_array = np.array(pose_seq)
    np.save(
        pose_output_path, pose_seq_array
    )

    # Clear GPU memory
    torch.cuda.empty_cache()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--rgb_seq_path", type=str, default="/workspace/test_data/color")
    parser.add_argument("--depth_seq_path", type=str, default="/workspace/test_data/depth")
    parser.add_argument("--mesh_path", type=str, default="/workspace/test_data/mesh/mesh.obj")
    parser.add_argument("--init_mask_path", type=str, default="/workspace/test_data/0_mask.png")
    parser.add_argument("--pose_output_path", type=str, default="/workspace/test_data/pose.npy")
    parser.add_argument("--mask_visualization_path", type=str, default="/workspace/test_data/mask_vis")
    parser.add_argument("--bbox_visualization_path", type=str, default="/workspace/test_data/bbox_vis")
    parser.add_argument("--pose_visualization_path", type=str, default="/workspace/test_data/pose_vis")
    parser.add_argument("--cam_K", type=json.loads, default="[[912.7279052734375, 0.0, 667.5955200195312], [0.0, 911.0028076171875, 360.5406799316406], [0.0, 0.0, 1.0]]", help="Camera intrinsic parameters")
    parser.add_argument("--est_refine_iter", type=int, default=10, help="FoundationPose initial refine iterations, see https://github.com/NVlabs/FoundationPose")
    parser.add_argument("--track_refine_iter", type=int, default=5, help="FoundationPose tracking refine iterations, see https://github.com/NVlabs/FoundationPose")
    parser.add_argument("--activate_2d_tracker", action='store_true', help="activate 2d tracker")
    parser.add_argument("--activate_kalman_filter", action='store_true', help="activate kalman_filter")
    parser.add_argument("--kf_measurement_noise_scale", type=float, default=0.05, help="The scale of measurement noise relative to prediction in kalman filter, greater value means more filtering. Only effective if activate_kalman_filter")
    parser.add_argument("--apply_scale", type=float, default=0.01, help="Mesh scale factor in meters (1.0 means no scaling), commonly use 0.01")
    parser.add_argument("--force_apply_color", action='store_true', help="force a color for colorless mesh")
    parser.add_argument("--apply_color", type=json.loads, default="[0, 159, 237]", help="RGB color to apply, in format 'r,g,b'. Only effective if force_apply_color")
    parser.add_argument("--real_dims_path", type=str, default=None,
                        help="real_dims.json 路径, 用于投影 3D bbox 修正手遮挡导致的 Cutie bbox 偏小")
    args = parser.parse_args()

    # 加载 real_dims
    real_dims = None
    if args.real_dims_path and os.path.exists(args.real_dims_path):
        with open(args.real_dims_path) as f:
            real_dims = json.load(f)
        print(f"已加载 real_dims: W={real_dims.get('width', 0)*1000:.0f}mm "
              f"H={real_dims.get('height', 0)*1000:.0f}mm "
              f"D={real_dims.get('depth', 0)*1000:.0f}mm")

    pose_track(
        args.rgb_seq_path,
        args.depth_seq_path,
        args.mesh_path,
        args.init_mask_path,
        np.array(args.cam_K),
        args.pose_output_path,
        args.mask_visualization_path,
        args.bbox_visualization_path,
        args.pose_visualization_path,
        args.est_refine_iter,
        args.track_refine_iter,
        args.activate_2d_tracker,
        args.activate_kalman_filter,
        real_dims=real_dims,
    )

    torch.cuda.empty_cache()
