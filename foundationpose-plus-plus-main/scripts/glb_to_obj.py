#!/usr/bin/env python3
"""
GLB → OBJ 转换器, 支持基于真实物理尺寸的深度缩放。

用法:
    # 基本转换 (不缩放)
    python scripts/glb_to_obj.py \
      --glb_path test_data/mouse/mesh/mesh.glb \
      --output test_data/mouse/mesh/mesh.obj

    # 带深度缩放
    python scripts/glb_to_obj.py \
      --glb_path test_data/mouse/mesh/mesh.glb \
      --output test_data/mouse/mesh/mesh.obj \
      --real_dims_json test_data/mouse/real_dims.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def _uniform_diag_scale(mesh, real_dims: dict, scale_threshold: float = 2.0) -> bool:
    """统一对角线缩放.

    计算 mesh 与 real_dims 的对角线比例, 若超出阈值则统一缩放所有顶点。
    返回 True 表示执行了缩放。
    """
    bbox = mesh.bounds
    mesh_ext = bbox[1] - bbox[0]
    mesh_diam = np.linalg.norm(mesh_ext)

    w = real_dims['width']
    h = real_dims['height']
    d = real_dims['depth']
    real_diam = np.linalg.norm([w, h, d])

    ratio = real_diam / mesh_diam if mesh_diam > 0 else 1.0
    if ratio > scale_threshold or ratio < 1.0 / scale_threshold:
        print(f"  [GLB2OBJ] 统一缩放: mesh_diam={mesh_diam:.4f}m → real_diam={real_diam:.4f}m "
              f"(ratio={ratio:.3f})")
        mesh.vertices *= ratio
        return True
    else:
        print(f"  [GLB2OBJ] 跳过统一缩放: mesh_diam={mesh_diam:.4f}m real_diam={real_diam:.4f}m "
              f"(ratio={ratio:.3f} < {scale_threshold})")
        return False


def _per_axis_obb_scale(mesh, real_dims: dict) -> bool:
    """逐轴 OBB 缩放, 按升序幅值匹配 real_dims 的三维.

    利用 trimesh.oriented_bounds 返回的 extents 始终升序排列的特性,
    将 mesh 最薄轴 → real_dims 最小值, 中等轴 → 中值, 最厚轴 → 最大值。

    返回 True 表示执行了缩放。
    """
    import trimesh

    to_origin, obb_ext = trimesh.bounds.oriented_bounds(mesh)
    # trimesh 保证 obb_ext 升序: [thinnest, middle, thickest]

    real_vals = np.array([real_dims['width'], real_dims['height'], real_dims['depth']])
    real_order = np.argsort(real_vals) 

    # 检查零轴
    if np.any(obb_ext < 1e-10):
        print("  [GLB2OBJ] ⚠ OBB 有接近零的轴, 回退到统一缩放")
        return False

    scales = np.array([real_vals[real_order[i]] / obb_ext[i] for i in range(3)])

    # 极端缩放因子检查
    if np.any(scales > 10.0) or np.any(scales < 0.1):
        print(f"  [GLB2OBJ] ⚠ 逐轴缩放因子异常: X={scales[0]:.3f} Y={scales[1]:.3f} Z={scales[2]:.3f}, "
              f"回退到统一缩放")
        return False

    # 已匹配则跳过
    if np.allclose(scales, 1.0, atol=1e-4):
        print("  [GLB2OBJ] 逐轴缩放: 所有轴已在目标尺寸, 跳过")
        return False

    mesh_diam = np.linalg.norm(obb_ext)
    target_diam = np.linalg.norm(obb_ext * scales)

    # 变换到 OBB 局部坐标 → 逐轴缩放 → 变回世界坐标
    R = to_origin[:3, :3]
    t = to_origin[:3, 3]
    verts_local = mesh.vertices @ R.T + t
    verts_local[:, 0] *= scales[0]
    verts_local[:, 1] *= scales[1]
    verts_local[:, 2] *= scales[2]
    mesh.vertices = (verts_local - t) @ R

    print(f"  [GLB2OBJ] 逐轴缩放: OBB extents (sorted) = "
          f"[{obb_ext[0]*1000:.0f}, {obb_ext[1]*1000:.0f}, {obb_ext[2]*1000:.0f}]mm "
          f"→ [{obb_ext[0]*scales[0]*1000:.0f}, {obb_ext[1]*scales[1]*1000:.0f}, {obb_ext[2]*scales[2]*1000:.0f}]mm")
    print(f"  [GLB2OBJ]            scales = [{scales[0]:.3f}, {scales[1]:.3f}, {scales[2]:.3f}], "
          f"diam {mesh_diam:.4f}→{target_diam:.4f}m")

    return True


def convert_glb_to_obj(
    glb_path: str,
    obj_path: str,
    real_dims: dict = None,
    scale_threshold: float = 2.0,
    per_axis_scale: bool = True,
) -> bool:
    """将 GLB 转换为 OBJ, 可选基于真实尺寸进行缩放.

    Args:
        glb_path: 输入 GLB 路径
        obj_path: 输出 OBJ 路径
        real_dims: 真实物理尺寸 {"width": W, "height": H, "depth": D} (米)
        scale_threshold: 统一缩放模式下, 当 mesh_diam 与 real_diam 之比超过此值时触发缩放
        per_axis_scale: True=逐轴 OBB 缩放, False=统一对角线缩放

    Returns:
        bool: 成功 True, 失败 False
    """
    import trimesh

    if not os.path.exists(glb_path):
        print(f"[GLB2OBJ] 错误: GLB 文件不存在: {glb_path}")
        return False

    # 加载 GLB
    glb_dir = os.path.dirname(glb_path)
    try:
        scene = trimesh.load(glb_path)
    except Exception as e:
        print(f"[GLB2OBJ] 错误: 无法加载 GLB: {e}")
        return False

    # 提取第一个 geometry
    if isinstance(scene, trimesh.Scene):
        # 尝试合并所有 geometry
        try:
            mesh = scene.dump(concatenate=True)
        except Exception:
            # 回退: 取第一个 geometry
            geoms = list(scene.geometry.values())
            if not geoms:
                print("[GLB2OBJ] 错误: GLB 中没有 geometry")
                return False
            mesh = geoms[0]
    else:
        mesh = scene

    if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
        print("[GLB2OBJ] 错误: mesh 中没有顶点")
        return False

    # ── 计算法线 ──
    if not hasattr(mesh, 'face_normals') or mesh.face_normals is None:
        mesh.compute_face_normals()

    # 按顶点平均法线
    # trimesh 在 .export() 时会自动处理法线，但如果需要手动算:
    if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None:
        try:
            mesh.compute_vertex_normals()
        except Exception:
            # 手动计算
            face_normals = mesh.face_normals
            vertex_normals = np.zeros((len(mesh.vertices), 3))
            count = np.zeros(len(mesh.vertices))
            for face_idx, face in enumerate(mesh.faces):
                n = face_normals[face_idx]
                for v_idx in face:
                    vertex_normals[v_idx] += n
                    count[v_idx] += 1
            count[count == 0] = 1
            vertex_normals /= count[:, None]
            # 归一化
            norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
            norms[norms == 0] = 1
            mesh.vertex_normals = vertex_normals / norms

    # ── 深度缩放 ──
    if real_dims and all(real_dims.get(k, 0) > 0 for k in ['width', 'height', 'depth']):
        if per_axis_scale:
            ok = False
            try:
                ok = _per_axis_obb_scale(mesh, real_dims)
            except Exception as e:
                print(f"  [GLB2OBJ] ⚠ OBB 逐轴缩放异常: {e}, 回退到统一缩放")
            if not ok:
                if ok is False:
                    print(f"  [GLB2OBJ] ⚠ OBB 逐轴缩放返回 False, 回退到统一缩放")
                _uniform_diag_scale(mesh, real_dims, scale_threshold)
        else:
            _uniform_diag_scale(mesh, real_dims, scale_threshold)

    # ── 保存 OBJ ──
    os.makedirs(os.path.dirname(obj_path), exist_ok=True)
    mesh.export(obj_path, file_type='obj')

    print(f"[GLB2OBJ] ✓ 完成: {os.path.getsize(obj_path) / 1024:.1f} KB → {obj_path}")

    # 检查纹理/Material 文件是否被导出
    obj_basename = os.path.splitext(os.path.basename(obj_path))[0]
    obj_dir = os.path.dirname(obj_path)
    # GLB 纹理会被导出为 PBR_Material.png (trimesh 默认命名)
    expected_mtl = os.path.join(obj_dir, f"{obj_basename}.mtl")
    if os.path.exists(expected_mtl):
        print(f"  [GLB2OBJ] MTL: {expected_mtl}")
    expected_texture = os.path.join(obj_dir, "PBR_Material.png")
    if os.path.exists(expected_texture):
        print(f"  [GLB2OBJ] 纹理贴图: {expected_texture}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="GLB → OBJ 转换器 (可选深度缩放)"
    )
    parser.add_argument("--glb_path", type=str, required=True,
                        help="输入 GLB 路径")
    parser.add_argument("--output", type=str, required=True,
                        help="输出 OBJ 路径")
    parser.add_argument("--real_dims_json", type=str, default=None,
                        help="真实物理尺寸 JSON (可选, 由 compute_real_dims.py 生成)")
    parser.add_argument("--scale_threshold", type=float, default=2.0,
                        help="统一缩放模式下, 当比率 > threshold 或 < 1/threshold 时触发 (默认 2.0)")
    parser.add_argument("--uniform_scale", action="store_true",
                        help="使用统一对角线缩放 (旧行为), 默认使用逐轴 OBB 缩放")
    args = parser.parse_args()

    real_dims = None
    if args.real_dims_json:
        if os.path.exists(args.real_dims_json):
            with open(args.real_dims_json) as f:
                real_dims = json.load(f)
        else:
            print(f"警告: real_dims_json 不存在: {args.real_dims_json}")

    ok = convert_glb_to_obj(
        glb_path=args.glb_path,
        obj_path=args.output,
        real_dims=real_dims,
        scale_threshold=args.scale_threshold,
        per_axis_scale=not args.uniform_scale,
    )

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
