#!/usr/bin/env python3
"""
从 mask + 深度图 + 相机内参估算物体真实物理尺寸 (W, H, D 米)。

算法:
  1. mask 区域内的有效深度点 → K 反投影到 3D
  2. 深度聚类: 保留 median_z ± 15% 范围内的点 (过滤桌面/背景)
  3. PCA 主成分分析 → 物体自身坐标系 
  4. 在物体自身坐标系中计算 p5-p95 范围 → (width, height, depth)
  5. 多帧累积 (--merge): 保留历史最大尺寸 (解决遮挡导致的尺寸偏小)

用法:
    python scripts/compute_real_dims.py \
      --mask_path test_data/mouse/0_mask.png \
      --depth_path test_data/mouse/depth/0.png \
      --cam_K_path test_data/mouse/cam_K.json \
      --output test_data/mouse/real_dims.json

    合并历史值:
    python scripts/compute_real_dims.py ... --merge
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def compute_real_dims(
    mask: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    median_z_ratio: float = 0.15,
    low_percentile: float = 0.5,
    high_percentile: float = 99.5,
    min_valid_points: int = 100,
    prior_dims: dict = None,
) -> dict:
    mask = mask.astype(bool)
    depth_mask = (depth_m > 0.001) & np.isfinite(depth_m)
    combined = mask & depth_mask

    n_valid = np.count_nonzero(combined)
    if n_valid < min_valid_points:
        print(f"  [RealDims] 有效点数不足: {n_valid} < {min_valid_points}")
        return _fallback_or_prior(prior_dims, n_valid)

    # 1. 提取有效区域的深度值和像素坐标
    rows, cols = np.where(combined)
    z_vals = depth_m[rows, cols]

    # 2. 深度聚类: 保留 median_z ± ratio 范围内的点
    z_median = np.median(z_vals)
    z_low = z_median * (1.0 - median_z_ratio)
    z_high = z_median * (1.0 + median_z_ratio)
    inlier = (z_vals >= z_low) & (z_vals <= z_high)

    z_val = z_vals[inlier]
    rows_inlier = rows[inlier]
    cols_inlier = cols[inlier]
    n_inlier = len(z_val)

    if n_inlier < min_valid_points // 2:
        print(f"  [RealDims] 深度聚类后点数不足: {n_inlier}")
        return _fallback_or_prior(prior_dims, n_inlier)

    # 3. 反投影到 3D 相机坐标系
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    X = (cols_inlier - cx) * z_val / fx
    Y = (rows_inlier - cy) * z_val / fy
    Z = z_val
    pts_3d = np.column_stack([X, Y, Z])  # (N, 3)

    # 4. PCA → 物体自身坐标系
    pts_centered = pts_3d - pts_3d.mean(axis=0)
    try:
        U, S, Vt = np.linalg.svd(pts_centered, full_matrices=False)
        R_obj = Vt  # 行向量 = 物体主轴方向
    except np.linalg.LinAlgError:
        R_obj = np.eye(3)

    pts_obj = pts_centered @ R_obj.T  # (N, 3)

    # 5. p5-p95 百分位在物体自身坐标系中
    x_low, x_high = np.percentile(pts_obj[:, 0], [low_percentile, high_percentile])
    y_low, y_high = np.percentile(pts_obj[:, 1], [low_percentile, high_percentile])
    z_low, z_high = np.percentile(pts_obj[:, 2], [low_percentile, high_percentile])

    width = float(x_high - x_low)
    height = float(y_high - y_low)
    depth = float(z_high - z_low)

    print(f"  [RealDims] median_z={z_median:.3f}m  n_inlier={n_inlier}")
    print(f"  [RealDims] PCA: W={width*1000:.0f}mm H={height*1000:.0f}mm D={depth*1000:.0f}mm")

    # 6. 多帧累积: 取各轴 max (解决遮挡导致尺寸偏小)
    dims = {"width": width, "height": height, "depth": depth}
    if prior_dims and prior_dims.get("valid_points", 0) > 0:
        dims["width"] = max(width, prior_dims.get("width", 0))
        dims["height"] = max(height, prior_dims.get("height", 0))
        dims["depth"] = max(depth, prior_dims.get("depth", 0))
        if dims["width"] > width or dims["height"] > height or dims["depth"] > depth:
            print(f"  [RealDims] 与历史值合并: "
                  f"W={dims['width']*1000:.0f}mm "
                  f"H={dims['height']*1000:.0f}mm "
                  f"D={dims['depth']*1000:.0f}mm")

    dims["width"] = round(dims["width"], 6)
    dims["height"] = round(dims["height"], 6)
    dims["depth"] = round(dims["depth"], 6)
    dims["valid_points"] = n_inlier

    return dims


def _fallback_or_prior(prior_dims: dict, n_valid: int):
    if prior_dims and prior_dims.get("valid_points", 0) > 0:
        print(f"  [RealDims] 回退到历史值: "
              f"W={prior_dims['width']*1000:.0f}mm "
              f"H={prior_dims['height']*1000:.0f}mm "
              f"D={prior_dims['depth']*1000:.0f}mm")
        return {**prior_dims, "valid_points": n_valid}
    return {"width": 0.0, "height": 0.0, "depth": 0.0, "valid_points": n_valid}


def load_inputs(mask_path: str, depth_path: str, cam_K_path: str) -> tuple:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"无法读取 mask: {mask_path}")
    mask_bool = mask > 127

    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"无法读取深度图: {depth_path}")
    if depth.dtype == np.uint16:
        depth_m = depth.astype(np.float32) * 0.001
    else:
        depth_m = depth.astype(np.float32)

    with open(cam_K_path) as f:
        cam_info = json.load(f)
    K = np.array(cam_info["K"], dtype=np.float64)

    return mask_bool, depth_m, K


def main():
    parser = argparse.ArgumentParser(
        description="从 mask + 深度图估算物体真实物理尺寸 (PCA 对齐)"
    )
    parser.add_argument("--mask_path", type=str, required=True)
    parser.add_argument("--depth_path", type=str, required=True)
    parser.add_argument("--cam_K_path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--merge", action="store_true",
                        help="与 output 中已有的尺寸合并 (取各轴 max)")
    args = parser.parse_args()

    try:
        mask, depth_m, K = load_inputs(args.mask_path, args.depth_path, args.cam_K_path)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

    prior_dims = None
    if args.merge and os.path.exists(args.output):
        try:
            with open(args.output) as f:
                prior_dims = json.load(f)
        except Exception:
            prior_dims = None

    dims = compute_real_dims(mask, depth_m, K, prior_dims=prior_dims)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(dims, f, indent=2)
    print(f"[RealDims] 已保存: {args.output}")


if __name__ == "__main__":
    main()
