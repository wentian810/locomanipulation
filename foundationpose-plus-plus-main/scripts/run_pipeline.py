#!/usr/bin/env python3
"""
FoundationPose++ 一键端到端 Pipeline 运行脚本。

用法:
    # 完整 pipeline (手动 mesh 模式)
    python scripts/run_pipeline.py \
      --data_dir test_data \
      --activate_2d_tracker \
      --activate_kalman_filter

    # 集成 Hunyuan3D 自动 mesh 生成
    python scripts/run_pipeline.py \
      --data_dir test_data \
      --use_hunyuan3d \
      --hunyuan3d_url http://localhost:18081 \
      --activate_2d_tracker \
      --activate_kalman_filter

    # 跳过提取，直接推理 (帧已提取)
    python scripts/run_pipeline.py \
      --data_dir test_data \
      --objects mouse \
      --skip_extraction

前置条件:
    1. Docker 容器已启动 (--network host)
    2. 宿主机 SAM-HQ (:9002) API 已启动
    3. 远程 Qwen3-VL-30B 检测服务可访问 (192.168.10.242:12067, 共享服务)
    4. Hunyuan3D API (:18081) 已启动 (仅 --use_hunyuan3d 时需要)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 物体中文描述 (用于 Qwen2-VL 检测)
OBJECT_DESCRIPTIONS = {
    "mouse": "鼠标",
    "notebook": "笔记本",
    "phone": "手机",
}

# 默认 apply_color (仅手动 mesh 模式使用)
DEFAULT_COLORS = {
    "mouse": "[100, 100, 100]",
    "notebook": "[50, 50, 200]",
    "phone": "[0, 0, 0]",
}

# ──────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────

def _fix_permissions(path: Path):
    """递归修复目录权限为 0777 (解决 Docker root 文件对宿主机不可写)."""
    if not path.exists():
        return
    try:
        os.chmod(path, 0o777)
        for root, dirs, files in os.walk(path):
            for d in dirs:
                os.chmod(os.path.join(root, d), 0o777)
            for f in files:
                os.chmod(os.path.join(root, f), 0o666)
    except Exception as e:
        print(f"  [WARN] 权限修复失败: {e} (非致命, sudo chown -R 可解决)")


def _load_cam_K(data_dir: Path) -> dict:
    """读取 cam_K.json 返回完整内容 (含 K 矩阵和图像尺寸)."""
    cam_K_path = data_dir / "cam_K.json"
    with open(cam_K_path) as f:
        return json.load(f)


def _cam_K_changed(old_K: list, new_K: list, threshold: float = 1.0) -> bool:
    """检查相机内参是否发生显著变化.

    比较颜色相机内参矩阵的 Frobenius 范数差异。
    threshold=1.0 意味着 fx/fy 变化 ~1 像素以上才认为相机变了。
    """
    old = np.array(old_K, dtype=np.float64)
    new = np.array(new_K, dtype=np.float64)
    diff = np.linalg.norm(old - new)
    return diff > threshold


def _load_mesh_cache(data_dir: Path) -> dict:
    """加载 mesh 缓存. 不存在则返回 None."""
    cache_path = data_dir / "mesh_cache.json"
    if not cache_path.exists():
        return None
    with open(cache_path) as f:
        return json.load(f)


def _save_mesh_cache(data_dir: Path, K: list, real_dims: dict, mesh_path: str):
    """保存 mesh 缓存."""
    cache = {
        "K": K,
        "real_dims": real_dims,
        "mesh_path": mesh_path,
        "generated": True,
    }
    cache_path = data_dir / "mesh_cache.json"
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)


def _select_mesh(mesh_dir: Path) -> Path:
    """在 mesh 目录中查找可用的 mesh 文件.

    优先级: .obj > .stl > .ply
    """
    if not mesh_dir.exists():
        return None
    for ext in ['.obj', '.stl', '.ply']:
        candidates = sorted(mesh_dir.glob(f"*{ext}"))
        if candidates:
            return candidates[0]
    return None

# ──────────────────────────────────────────────────────────
# Pipeline Steps
# ──────────────────────────────────────────────────────────

def step_extract_frames(args, object_name: str):
    """Step 1: 自动识别数据格式并提取帧（已有结果则跳过）.

    支持两种格式:
      - db3:    realsense ROS2 bag 文件 (realsense_data/<name>.db3)
      - self_demo: stereo 双目光学文件夹 (realsense_data/<name>/images.h5)
    """
    source_dir = Path(args.source)
    output_dir = Path(args.data_dir) / object_name

    # ── 自动检测数据格式 ──
    db3_path = source_dir / f"{object_name}.db3"
    self_demo_path = source_dir / object_name
    input_type = None
    input_path = None

    if db3_path.exists():
        input_type = "db3"
        input_path = db3_path
    elif self_demo_path.is_dir() and (self_demo_path / "images.h5").exists():
        input_type = "self_demo"
        input_path = self_demo_path
    else:
        print(f"  [跳过] 未找到数据源: {db3_path} 或 {self_demo_path}/images.h5")
        return False

    # ── 跳过检查 (输出已存在) ──
    if output_dir.exists():
        color_dir = output_dir / "color"
        depth_dir = output_dir / "depth"
        if color_dir.exists() and depth_dir.exists():
            n_color = len(list(color_dir.glob("*.png")) + list(color_dir.glob("*.jpg")))
            n_depth = len(list(depth_dir.glob("*.png")))
            if n_color > 0 and n_depth > 0:
                print(f"  [跳过] 数据已存在: {output_dir} ({n_color} color + {n_depth} depth)")
                _fix_permissions(output_dir)
                return True

    print(f"\n{'='*60}")
    print(f"[Step 1] 提取帧: {object_name} (格式: {input_type})")
    print(f"{'='*60}")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "extract_realsense_bag.py"),
        "--input_type", input_type,
        "--input_path", str(input_path),
        "--output_dir", str(output_dir),
    ]
    if args.max_frames:
        cmd += ["--max_frames", str(args.max_frames)]

    subprocess.run(cmd, check=True)

    _fix_permissions(output_dir)
    return True


def step_get_bbox(args, object_name: str, description: str):
    """Step 2: Qwen2-VL 获取初始 bbox."""
    data_dir = Path(args.data_dir) / object_name
    color0 = data_dir / "color" / "0.png"
    bbox_out = data_dir / "0_bbox.png"

    # 转为容器内绝对路径 (与 step_get_mask 相同原因)
    color0_abs = PROJECT_ROOT / color0
    bbox_out_abs = PROJECT_ROOT / bbox_out

    if not color0_abs.exists():
        print(f"  [跳过] 首帧不存在: {color0_abs}")
        return None

    print(f"\n{'='*60}")
    print(f"[Step 2] Qwen2-VL BBox 检测: {object_name}")
    print(f"{'='*60}")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "utils" / "obj_bbox.py"),
        "--frame_path", str(color0_abs),
        "--visualize_path", str(bbox_out_abs),
        "--object_name", description,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    bbox_str = result.stdout.strip()

    if bbox_str.startswith("[") and bbox_str.endswith("]"):
        print(f"  BBox: {bbox_str}")
        return bbox_str
    else:
        print(f"  Qwen2-VL 输出: {bbox_str}")
        print(f"  stderr: {result.stderr}")
        return None


def step_get_mask(args, object_name: str, bbox: str):
    """Step 3: SAM-HQ 生成初始 mask + RGBA 抠图."""
    data_dir = Path(args.data_dir) / object_name
    color0 = data_dir / "color" / "0.png"
    mask_out = data_dir / "0_mask.png"

    # 转为容器内绝对路径。obj_mask.py 把路径发给宿主机 SAM-HQ API，
    # API 用 _resolve_host_path() 把 /workspace → 项目根目录。
    # 相对路径无法解析，必须是 /workspace 开头。
    color0_abs = PROJECT_ROOT / color0
    mask_out_abs = PROJECT_ROOT / mask_out

    if not color0_abs.exists():
        print(f"  [跳过] 首帧不存在: {color0_abs}")
        return False

    # 确保数据目录对宿主机 SAM-HQ 可写 (Docker root 创建的目录 owner 是 root)
    _fix_permissions(data_dir)

    print(f"\n{'='*60}")
    print(f"[Step 3] SAM-HQ Mask 生成: {object_name}")
    print(f"{'='*60}")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "utils" / "obj_mask.py"),
        "--frame_path", str(color0_abs),
        "--bbox_xywh", bbox,
        "--output_mask_path", str(mask_out_abs),
    ]

    subprocess.run(cmd, check=True)
    return mask_out.exists()


def step_compute_real_dims(args, object_name: str):
    """Step 4: 从 mask + 深度 + 内参估算物体真实尺寸."""
    data_dir = Path(args.data_dir) / object_name
    mask_path = data_dir / "0_mask.png"
    depth0 = data_dir / "depth" / "0.png"
    cam_K_path = data_dir / "cam_K.json"
    dims_out = data_dir / "real_dims.json"

    if not mask_path.exists():
        print(f"  [跳过] mask 不存在: {mask_path}")
        return None
    if not depth0.exists():
        print(f"  [跳过] 首帧深度不存在: {depth0}")
        return None
    if not cam_K_path.exists():
        print(f"  [跳过] cam_K.json 不存在: {cam_K_path}")
        return None

    print(f"\n{'='*60}")
    print(f"[Step 4] 计算真实尺寸: {object_name}")
    print(f"{'='*60}")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "compute_real_dims.py"),
        "--mask_path", str(mask_path),
        "--depth_path", str(depth0),
        "--cam_K_path", str(cam_K_path),
        "--output", str(dims_out),
        "--merge",
    ]

    subprocess.run(cmd, check=True)

    if dims_out.exists():
        with open(dims_out) as f:
            dims = json.load(f)
        if dims.get("width", 0) > 0:
            return dims

    print(f"  ⚠ 真实尺寸计算失败 (有效点数不足)")
    return None


def step_generate_mesh(args, object_name: str, real_dims: dict = None):
    """Step 5: 通过 Hunyuan3D 生成 mesh (含缓存检查).

    流程:
      1. 检查 cache (mesh_cache.json + K 不变) → 命中则直接返回
      2. 调用 Hunyuan3D API → mesh.glb
      3. GLB → OBJ 转换 (含深度缩放)
      4. 保存 cache
    """
    data_dir = Path(args.data_dir) / object_name
    mesh_dir = data_dir / "mesh"
    rgba_path = data_dir / "0_mask_rgba.png"
    cam_K_path = data_dir / "cam_K.json"

    if not cam_K_path.exists():
        print(f"  [跳过] cam_K.json 不存在")
        return False

    cam_K_data = _load_cam_K(data_dir)
    current_K = cam_K_data["K"]

    # ── 缓存检查 ──
    cache = _load_mesh_cache(data_dir)
    mesh_file = _select_mesh(mesh_dir)

    if cache and cache.get("generated") and mesh_file:
        cached_K = cache.get("K")
        if cached_K and not _cam_K_changed(cached_K, current_K):
            print(f"\n[Step 5] Mesh 缓存命中: {object_name}")
            print(f"  K 未变化, 跳过 mesh 生成")
            if real_dims is None:
                real_dims = cache.get("real_dims", {})
                if real_dims.get("width", 0) > 0:
                    print(f"  real_dims: W={real_dims['width']*1000:.0f}mm "
                          f"H={real_dims['height']*1000:.0f}mm D={real_dims['depth']*1000:.0f}mm")
            return True
        else:
            print(f"  K 已变化, 需重新生成 mesh")
    elif mesh_file and not args.use_hunyuan3d:
        print(f"\n[Step 5] 使用已有 mesh: {mesh_file.name}")
        return True

    print(f"\n{'='*60}")
    print(f"[Step 5] Hunyuan3D Mesh 生成: {object_name}")
    print(f"{'='*60}")

    # ── 1. 检查 RGBA ──
    if not rgba_path.exists():
        print(f"  [跳过] RGBA 抠图不存在: {rgba_path}")
        print(f"        请确保 SAM-HQ API 已更新 (支持 _rgba.png 输出)")
        return False

    # ── 2. Hunyuan3D: RGBA → mesh.glb ──
    mesh_dir.mkdir(parents=True, exist_ok=True)
    glb_path = mesh_dir / "mesh.glb"

    if args.hunyuan3d_backend == "service":
        bridge = PROJECT_ROOT / "scripts" / "hunyuan3d_bridge_v2.1_service.py"
        cmd_hunyuan = [
            sys.executable,
            str(bridge),
            "--rgba_path", str(rgba_path),
            "--output_mesh", str(glb_path),
            "--hunyuan3d_url", args.hunyuan3d_url,
            "--timeout", str(args.hunyuan3d_timeout),
        ]
    else:
        bridge = PROJECT_ROOT / "scripts" / "hunyuan3d_bridge.py"
        cmd_hunyuan = [
            sys.executable,
            str(bridge),
            "--rgba_path", str(rgba_path),
            "--output_mesh", str(glb_path),
            "--face_count", str(args.hunyuan3d_face_count),
            "--poll_timeout", str(args.hunyuan3d_timeout),
        ]
        if args.hunyuan3d_no_pbr:
            cmd_hunyuan.append("--no_pbr")

    result = subprocess.run(cmd_hunyuan, capture_output=False)
    if result.returncode != 0 or not glb_path.exists():
        print(f"  ✗ Hunyuan3D 生成失败 (返回码: {result.returncode})")
        return False

    # ── 3. GLB → OBJ 转换 (含深度缩放) ──
    obj_path = mesh_dir / "mesh.obj"

    cmd_glb2obj = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "glb_to_obj.py"),
        "--glb_path", str(glb_path),
        "--output", str(obj_path),
    ]
    if real_dims and real_dims.get("width", 0) > 0:
        # 传递 real_dims, GLB→OBJ 时会自动做深度缩放
        dims_json = data_dir / "real_dims.json"
        cmd_glb2obj += ["--real_dims_json", str(dims_json)]

    result = subprocess.run(cmd_glb2obj, capture_output=False)
    if result.returncode != 0 or not obj_path.exists():
        print(f"  ✗ GLB→OBJ 转换失败 (返回码: {result.returncode})")
        return False

    # ── 4. 保存缓存 ──
    final_real_dims = real_dims or {}
    _save_mesh_cache(data_dir, current_K, final_real_dims, str(obj_path.relative_to(data_dir)))
    print(f"  缓存已保存: mesh_cache.json")

    return True


def step_run_tracking(args, object_name: str):
    """Step 6: 运行 FoundationPose++ 6D 位姿跟踪."""
    data_dir = Path(args.data_dir) / object_name
    rgb_path = data_dir / "color"
    depth_path = data_dir / "depth"
    mesh_dir = data_dir / "mesh"
    init_mask = data_dir / "0_mask.png"

    if not rgb_path.exists() or not depth_path.exists():
        print(f"  [跳过] 帧数据不完整")
        return False

    if not init_mask.exists():
        print(f"  [跳过] 初始 mask 不存在: {init_mask}")
        print(f"         请先运行 --skip_extraction 后再执行 Step 3 获取 mask")
        return False

    # 查找 mesh 文件
    mesh_file = _select_mesh(mesh_dir)

    if mesh_file is None:
        print(f"  [跳过] mesh 文件不存在: {mesh_dir}")
        print(f"         请将 3D 模型放入 {mesh_dir}/")
        print(f"         或使用 --use_hunyuan3d 自动生成")
        return False

    cam_K_str = json.dumps(_load_cam_K(data_dir)["K"])
    pose_output = data_dir / "pose.npy"
    pose_vis = data_dir / "pose_vis"
    mask_vis = data_dir / "mask_vis"
    bbox_vis = data_dir / "bbox_vis"

    print(f"\n{'='*60}")
    print(f"[Step 6] 6D 位姿跟踪: {object_name}")
    print(f"{'='*60}")
    print(f"  Mesh: {mesh_file.name}")
    print(f"  K: {cam_K_str}")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "obj_pose_track.py"),
        "--rgb_seq_path", str(rgb_path),
        "--depth_seq_path", str(depth_path),
        "--mesh_path", str(mesh_file),
        "--init_mask_path", str(init_mask),
        "--pose_output_path", str(pose_output),
        "--mask_visualization_path", str(mask_vis),
        "--bbox_visualization_path", str(bbox_vis),
        "--pose_visualization_path", str(pose_vis),
        "--cam_K", cam_K_str,
        "--est_refine_iter", str(args.est_refine_iter),
        "--track_refine_iter", str(args.track_refine_iter),
        "--apply_scale", str(args.apply_scale),
    ]

    if args.activate_2d_tracker:
        cmd.append("--activate_2d_tracker")
    if args.activate_kalman_filter:
        cmd.append("--activate_kalman_filter")
        cmd.extend(["--kf_measurement_noise_scale", str(args.kf_measurement_noise_scale)])

    # Hunyuan3D 生成的 mesh 已在米制真实尺寸, 不需要缩放和着色
    cache = _load_mesh_cache(data_dir)
    cam_K_data = _load_cam_K(data_dir) if (data_dir / "cam_K.json").exists() else {}
    data_source = cam_K_data.get("data_source", "")

    if cache and cache.get("generated"):
        if args.force_apply_color:
            print(f"  [Info] Hunyuan3D mesh 已带纹理, 忽略 --force_apply_color")
        # 覆盖 apply_scale: Hunyuan3D mesh 已按 real_dims 缩放到真实尺寸(米)
        cmd.remove("--apply_scale")
        cmd.remove(str(args.apply_scale))
        cmd.extend(["--apply_scale", "1.0"])
        print(f"  [Info] Hunyuan3D mesh 使用 apply_scale=1.0 (已按真实尺寸生成)")
    elif data_source == "self_demo":
        # self-demo 的 scaled_obj 已在真实尺寸(米), 无需缩放
        if args.apply_scale != 1.0:
            cmd.remove("--apply_scale")
            cmd.remove(str(args.apply_scale))
            cmd.extend(["--apply_scale", "1.0"])
            print(f"  [Info] self-demo mesh 已在真实尺寸, 自动设置 apply_scale=1.0")
    elif args.force_apply_color:
        cmd.append("--force_apply_color")
        color = args.apply_color if args.apply_color else DEFAULT_COLORS.get(object_name, "[0, 159, 237]")
        cmd.extend(["--apply_color", color])

    # 传递 real_dims.json 用于投影 3D bbox 修正手遮挡产生的 bbox 偏小
    real_dims_path = data_dir / "real_dims.json"
    if real_dims_path.exists():
        cmd.extend(["--real_dims_path", str(real_dims_path)])

    print(f"  命令: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return True


def step_generate_video(args, object_name: str):
    """Step 7: 生成 mp4 视频."""
    data_dir = Path(args.data_dir) / object_name
    pose_vis = data_dir / "pose_vis"
    output_video = data_dir / f"{object_name}_pose.mp4"

    if not pose_vis.exists() or not list(pose_vis.glob("*.png")):
        print(f"  [跳过] 无可视化帧: {pose_vis}")
        return False

    print(f"\n{'='*60}")
    print(f"[Step 7] 生成视频: {object_name}")
    print(f"{'='*60}")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "generate_video.py"),
        "--frame_dir", str(pose_vis),
        "--output", str(output_video),
        "--fps", str(args.fps),
    ]

    subprocess.run(cmd, check=True)
    return True


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FoundationPose++ 一键端到端 Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 基本参数
    parser.add_argument("--data_dir", type=str, default="test_data",
                        help="输出数据根目录")
    parser.add_argument("--objects", type=str, default=None,
                        help="物体名称列表，逗号分隔。默认自动扫描 realsense_data/ 下所有 .db3")
    parser.add_argument("--source", type=str, default="realsense_data",
                        help=".db3 文件目录")

    # 步骤控制
    parser.add_argument("--skip_extraction", action="store_true",
                        help="跳过 .db3 提取步骤")
    parser.add_argument("--skip_mask", action="store_true",
                        help="跳过 mask 生成 (使用已有的 0_mask.png)")
    parser.add_argument("--skip_mesh", action="store_true",
                        help="跳过 mesh 生成 (使用已有的 mesh/{}.obj)")
    parser.add_argument("--skip_inference", action="store_true",
                        help="跳过推理步骤")
    parser.add_argument("--skip_video", action="store_true",
                        help="跳过视频生成")

    # 提取参数
    parser.add_argument("--max_frames", type=int, default=None,
                        help="最大提取帧数 (调试用)")

    # Hunyuan3D 参数
    parser.add_argument("--use_hunyuan3d", action="store_true",
                        help="使用 Hunyuan3D API 自动生成物体 mesh")
    parser.add_argument(
        "--hunyuan3d_backend",
        choices=["tencent", "service"],
        default=os.environ.get("HUNYUAN3D_BACKEND", "tencent"),
        help="tencent=腾讯云 Hunyuan3D 3.0；service=本地 /generate 服务",
    )
    parser.add_argument("--hunyuan3d_url", type=str,
                        default=os.environ.get("HUNYUAN3D_URL", "http://localhost:18081"),
                        help="service 后端地址 (默认: http://localhost:18081)")
    parser.add_argument("--hunyuan3d_face_count", type=int, default=100000)
    parser.add_argument("--hunyuan3d_timeout", type=int, default=900)
    parser.add_argument("--hunyuan3d_no_pbr", action="store_true")
    parser.add_argument("--force_regenerate_mesh", action="store_true",
                        help="强制重新生成 mesh (即使已有 mesh 文件)")

    # 推理参数
    parser.add_argument("--activate_2d_tracker", action="store_true",
                        help="启用 Cutie 2D 跟踪器")
    parser.add_argument("--activate_kalman_filter", action="store_true",
                        help="启用卡尔曼滤波器")
    parser.add_argument("--kf_measurement_noise_scale", type=float, default=0.05,
                        help="KF 测量噪声比例")
    parser.add_argument("--est_refine_iter", type=int, default=10,
                        help="首帧 refine 迭代次数")
    parser.add_argument("--track_refine_iter", type=int, default=5,
                        help="跟踪 refine 迭代次数")
    parser.add_argument("--apply_scale", type=float, default=0.01,
                        help="Mesh 缩放因子 (米为单位)")
    parser.add_argument("--force_apply_color", action="store_true",
                        help="为无纹理 mesh 强制添加颜色 (Hunyuan3D mesh 忽略此参数)")
    parser.add_argument("--apply_color", type=str, default=None,
                        help="强制应用的颜色, 格式 '[R,G,B]'")

    # 视频参数
    parser.add_argument("--fps", type=int, default=30,
                        help="输出视频帧率")
    parser.add_argument("--keep_pose_frames", action="store_true",
                        help="保留每帧 3D BBox 标注图（默认删除 pose_vis/ 以节省空间）")

    args = parser.parse_args()

    # 自动扫描 realsense_data/ 下的 .db3 文件和 self-demo 文件夹
    if args.objects is None:
        source_dir = Path(args.source)
        if source_dir.exists():
            # db3 文件: mouse.db3 → mouse
            db3_names = [p.stem for p in source_dir.glob("*.db3")]
            # self-demo 文件夹: 包含 images.h5 的子目录
            self_demo_names = [
                p.name for p in source_dir.iterdir()
                if p.is_dir() and (p / "images.h5").exists()
            ]
            args.objects = ",".join(sorted(db3_names + self_demo_names))
            if args.objects:
                n_db3 = len(db3_names)
                n_sd = len(self_demo_names)
                parts = []
                if n_db3: parts.append(f"{n_db3} db3")
                if n_sd: parts.append(f"{n_sd} self-demo")
                print(f"[Auto] 从 {args.source}/ 自动发现物体: {args.objects} ({', '.join(parts)})")
        if not args.objects:
            print("错误: 未指定 --objects 且在 --source 目录下未找到 .db3 文件或 self-demo 文件夹")
            sys.exit(1)

    objects = [o.strip() for o in args.objects.split(",")]

    print("="*60)
    print("FoundationPose++ Pipeline")
    print("="*60)
    print(f"  物体: {objects}")
    print(f"  数据目录: {args.data_dir}")
    print(f"  数据源: {args.source}")
    print(
        "  Hunyuan3D: "
        f"{'启用 (' + args.hunyuan3d_backend + ')' if args.use_hunyuan3d else '禁用 (使用手动 mesh)'}"
    )
    print(f"  2D 跟踪器: {'启用' if args.activate_2d_tracker else '禁用'}")
    print(f"  卡尔曼滤波器: {'启用' if args.activate_kalman_filter else '禁用'}")

    success_count = 0

    for obj in objects:
        print(f"\n{'#'*60}")
        print(f"# 物体: {obj}")
        print(f"{'#'*60}")

        ok = True
        real_dims = None

        # Step 1: 提取帧
        if not args.skip_extraction:
            ok = step_extract_frames(args, obj) and ok

        #
        # Hunyuan3D 模式：先检查 K-based cache, 命中则跳过 mask+mesh
        #
        skip_mask_and_mesh = False
        if args.use_hunyuan3d and not args.skip_mask and not args.skip_mesh:
            data_dir = Path(args.data_dir) / obj
            cam_K_path = data_dir / "cam_K.json"
            if cam_K_path.exists():
                cache = _load_mesh_cache(data_dir)
                mesh_file = _select_mesh(data_dir / "mesh")
                try:
                    cam_K_data = _load_cam_K(data_dir)
                    current_K = cam_K_data["K"]
                except Exception:
                    current_K = None

                if (cache and cache.get("generated") and mesh_file and
                    current_K and cache.get("K") and
                    not _cam_K_changed(cache["K"], current_K)):
                    print(f"\n  [Cache] 物体 '{obj}' 已注册 (K 未变), 跳过 mask + mesh 生成")
                    skip_mask_and_mesh = True

        # Step 2-3: BBox + Mask
        data_dir = Path(args.data_dir) / obj
        mask_exists = (data_dir / "0_mask.png").exists()

        if args.skip_mask or args.skip_inference or skip_mask_and_mesh:
            pass  # 用户明确跳过
        elif mask_exists:
            print(f"\n  [Skip] Mask 已存在: {data_dir / '0_mask.png'}, 跳过 Step 2-3 (Qwen2-VL + SAM-HQ)")
        else:
            obj_desc = OBJECT_DESCRIPTIONS.get(obj, obj)
            bbox = step_get_bbox(args, obj, obj_desc)
            if bbox:
                step_get_mask(args, obj, bbox)
            else:
                print(f"  ⚠ Qwen2-VL 检测失败, 请手动创建 0_mask.png")
                ok = False

        # Step 4: ComputeRealDims (Hunyuan3D 模式)
        if args.use_hunyuan3d and not args.skip_mesh and not args.skip_inference and not skip_mask_and_mesh:
            real_dims = step_compute_real_dims(args, obj)

        # Step 5: Mesh 生成 (Hunyuan3D 模式) / 或使用已有 mesh
        data_dir = Path(args.data_dir) / obj
        mesh_file = _select_mesh(data_dir / "mesh")

        if args.skip_mesh or args.skip_inference:
            pass  # 用户明确跳过
        elif mesh_file and not args.force_regenerate_mesh:
            # mesh 已存在，直接使用 (self_demo 提取时已内置，或之前已生成)
            print(f"\n  [Mesh] 使用已有 mesh: {mesh_file.name} (跳过生成)")
        elif args.use_hunyuan3d:
            ok = step_generate_mesh(args, obj, real_dims) and ok
        else:
            print(f"\n  ⚠ mesh 文件不存在: {data_dir}/mesh/")
            print(f"     请放入 .obj/.stl 文件或使用 --use_hunyuan3d 自动生成")
            ok = False

        # Step 6: 推理（内部已实时编码 mp4）
        if not args.skip_inference:
            ok = step_run_tracking(args, obj) and ok

            # ── 清理逐帧 BBox 标注图（可选） ──
            if ok:
                data_dir = Path(args.data_dir) / obj
                video_output = data_dir / f"{obj}_pose.mp4"
                if video_output.exists() and not args.keep_pose_frames:
                    for vis_dir_name in ["pose_vis", "mask_vis", "bbox_vis"]:
                        vis_dir = data_dir / vis_dir_name
                        if vis_dir.exists():
                            n_files = len(list(vis_dir.glob("*")))
                            shutil.rmtree(vis_dir)
                            print(f"  [清理] 已删除 {vis_dir_name}/ ({n_files} 文件)")
                    print(f"  保留视频: {video_output}")
                    print(f"  (使用 --keep_pose_frames 可保留逐帧标注图)")
                elif args.keep_pose_frames:
                    print(f"  保留逐帧标注图 (--keep_pose_frames)")

        # Step 7: 视频（仅在跳过推理时，pose_vis/ 中已有旧帧需要合成）
        if not args.skip_video:
            step_generate_video(args, obj)

        if ok:
            success_count += 1

    print(f"\n{'='*60}")
    print(f"Pipeline 完成! 成功: {success_count}/{len(objects)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
