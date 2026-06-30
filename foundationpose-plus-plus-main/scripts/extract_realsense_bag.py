#!/usr/bin/env python3
"""
从 RealSense D435 .db3 (ROS2 bag) 或 self-demo Stereo 文件夹中提取数据.

支持两种输入格式:

1. Realsense ROS2 bag (.db3):
   ──────────────────────────────
   从 RealSense D435 录制的 .db3 文件中提取 RGB 和深度帧，
   并完成深度→彩色对齐（模拟 pyrealsense2 的 rs.align(COLOR)）。

   用法:
       python scripts/extract_realsense_bag.py \
         --input_type db3 \
         --input_path realsense_data/mouse.db3 \
         --output_dir test_data/mouse

   原理:
       1. 解析 .db3 内的 sensor_msgs/Image (CDR 序列化)
       2. 彩色图 (640×480, rgb8) + 深度图 (848×480, mono16)
       3. 用 D435 内参 + 出厂外参 T_depth→color 做像素级重映射
       4. 输出彩色图 + 对齐后深度图 (均为 640×480 PNG)
       5. 保存相机内参 cam_K.json

2. self-demo Stereo 文件夹:
   ──────────────────────────────
   从 self-demo 文件夹提取已配准的 RGB 帧、深度帧、物体 mask 和 mesh。

   用法:
       python scripts/extract_realsense_bag.py \
         --input_type self_demo \
         --input_path realsense_data/self-demo1 \
         --output_dir test_data/self-demo1

   文件夹内包含:
       - images.h5:  RGB帧 (frames/rgb/png) + 物体mask (masks/objects/*/png)
       - depth.h5:   深度图 (depth/depth, float32 米, 与 RGB 同分辨率逐像素对齐)
       - meshes.h5:  物体mesh (objects/*/scaled_obj, glb)
       - meta.json:  相机内参

   输出: color/, depth/, mesh/, 0_mask.png, cam_K.json, depth_scale.json

依赖:
    db3 模式:    pip install numpy opencv-python
    self_demo 模式:  pip install numpy opencv-python h5py
"""

import argparse
import json
import os
import sqlite3
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────
# 相机参数 (从 .db3 中动态读取 camera_info，见 read_camera_params)
# ──────────────────────────────────────────────────────────

def _parse_camera_info_string(info_str: str) -> dict:
    """解析 RealSense ROS2 bag 中的 camera_info 字符串格式.

    格式示例: "width=640;height=480;fx=604.151;ppx=322.585;fy=603.716;ppy=256.046;..."
    """
    result = {}
    for item in info_str.rstrip('\x00').split(';'):
        item = item.strip()
        if '=' not in item:
            continue
        key, val = item.split('=', 1)
        try:
            # 尝试转为 int 或 float
            result[key] = float(val) if '.' in val else int(val)
        except ValueError:
            result[key] = val
    return result


def _parse_extrinsic_string(tf_str: str) -> dict:
    """解析 RealSense ROS2 bag 中的 tf/ref_0 字符串格式.

    格式示例: "translation x=0.014... y=-0.000... z=...; rotation..."
    或 JSON 格式。
    """
    # 尝试 JSON
    try:
        return json.loads(tf_str)
    except (json.JSONDecodeError, ValueError):
        pass

    result = {}
    for item in tf_str.rstrip('\x00').split(';'):
        item = item.strip()
        if '=' in item:
            # 子项: "x=0.014840,y=-0.00018341,z=..."
            sub_items = item.split(',')
            vec = []
            for si in sub_items:
                if '=' in si:
                    _, v = si.split('=', 1)
                    try:
                        vec.append(float(v))
                    except ValueError:
                        pass
            if vec:
                result[item.split('=')[0].strip()] = vec
    return result


def read_camera_params(db3_path: str) -> dict:
    """从 .db3 文件中动态读取相机内参和外参.

    Returns:
        dict with keys: K_color, K_depth, T_d2c, color_w, color_h, depth_w, depth_h
    """
    conn = sqlite3.connect(db3_path)
    cursor = conn.cursor()

    # 获取 camera_info topic id
    cursor.execute(
        "SELECT id, name FROM topics "
        "WHERE name LIKE '%Color_0/camera_info' OR name LIKE '%Depth_0/camera_info'"
    )
    info_topics = {name: tid for tid, name in cursor.fetchall()}

    K_color = None
    K_depth = None
    color_w = color_h = depth_w = depth_h = None

    for name, tid in info_topics.items():
        cursor.execute("SELECT data FROM messages WHERE topic_id=? LIMIT 1", (tid,))
        row = cursor.fetchone()
        if not row:
            continue
        # CDR: 4-byte header + 4-byte string length + string
        data = row[0]
        str_len = int.from_bytes(data[4:8], 'little')
        info_str = data[8:8+str_len].decode('utf-8', errors='replace')
        params = _parse_camera_info_string(info_str)

        if 'Color' in name:
            color_w = params.get('width', color_w)
            color_h = params.get('height', color_h)
            K_color = np.array([
                [params.get('fx', 0), 0, params.get('ppx', 0)],
                [0, params.get('fy', 0), params.get('ppy', 0)],
                [0, 0, 1],
            ], dtype=np.float64)
        else:
            depth_w = params.get('width', depth_w)
            depth_h = params.get('height', depth_h)
            K_depth = np.array([
                [params.get('fx', 0), 0, params.get('ppx', 0)],
                [0, params.get('fy', 0), params.get('ppy', 0)],
                [0, 0, 1],
            ], dtype=np.float64)

    # 尝试读取 tf 外参 (depth→color)
    cursor.execute(
        "SELECT id, name FROM topics "
        "WHERE name LIKE '%Depth_0/tf/ref_0' OR name LIKE '%Color_0/tf/ref_0'"
    )
    tf_topics = {name: tid for tid, name in cursor.fetchall()}

    T_d2c = np.eye(4, dtype=np.float64)
    # D435 默认外参: 深度传感器在彩色右方约 15mm
    T_d2c[0, 3] = 0.015

    for name, tid in tf_topics.items():
        cursor.execute("SELECT data FROM messages WHERE topic_id=? LIMIT 1", (tid,))
        row = cursor.fetchone()
        if not row:
            continue
        data = row[0]
        str_len = int.from_bytes(data[4:8], 'little')
        tf_str = data[8:8+str_len].decode('utf-8', errors='replace')
        params = _parse_extrinsic_string(tf_str)

        # 只处理 depth 到 ref_0 的变换
        if 'Depth' in name and 'translation' in params:
            translation = params['translation']
            if len(translation) >= 3:
                T_d2c[0, 3] = translation[0]
                T_d2c[1, 3] = translation[1]
                T_d2c[2, 3] = translation[2]
        if 'Depth' in name and 'rotation' in params:
            pass  # D435 tf ref_0 rotation 通常是单位矩阵, 保持默认

    conn.close()

    # 回退值
    if K_color is None:
        print("[Camera] ⚠ 未找到彩色 camera_info, 使用 D435 典型值")
        K_color = np.array([[604,0,322],[0,604,256],[0,0,1]], dtype=np.float64)
        color_w, color_h = 640, 480
    if K_depth is None:
        print("[Camera] ⚠ 未找到深度 camera_info, 使用 D435 典型值")
        K_depth = np.array([[425,0,422],[0,425,237],[0,0,1]], dtype=np.float64)
        depth_w, depth_h = 848, 480

    return {
        "K_color": K_color,
        "K_depth": K_depth,
        "T_d2c": T_d2c,
        "color_w": color_w,
        "color_h": color_h,
        "depth_w": depth_w,
        "depth_h": depth_h,
    }

# ──────────────────────────────────────────────────────────
# CDR 反序列化 (ROS2 sensor_msgs/msg/Image)
# ──────────────────────────────────────────────────────────

class CDRReader:
    """ROS2 CDR (Common Data Representation) little-endian 字节流解析器.

    CDR 对齐规则:
      - int8/uint8/bool:  1 字节对齐 (可位于任意偏移)
      - int16/uint16:     2 字节对齐
      - int32/uint32/float32: 4 字节对齐
      - int64/uint64/float64: 8 字节对齐
      - string:    长度前缀 (uint32, 4 字节对齐), 后跟字符 (1 字节对齐)
      - sequence:  长度前缀 (uint32, 4 字节对齐), 后跟元素
    """

    def __init__(self, data: bytes):
        self._data = data
        self._offset = 4  # 跳过 CDR header (00 01 00 00)

    def _align(self, alignment: int):
        """将当前偏移对齐到 alignment 边界."""
        self._offset = (self._offset + alignment - 1) & ~(alignment - 1)

    def read_int32(self) -> int:
        self._align(4)
        val = int.from_bytes(self._data[self._offset:self._offset+4], 'little', signed=True)
        self._offset += 4
        return val

    def read_uint32(self) -> int:
        self._align(4)
        val = int.from_bytes(self._data[self._offset:self._offset+4], 'little', signed=False)
        self._offset += 4
        return val

    def read_string(self) -> str:
        """读取 CDR string (长度前缀 + 字符)."""
        length = self.read_uint32()
        s = self._data[self._offset:self._offset+length]
        self._offset += length
        # 去除末尾 \0 (ROS2 字符串通常有 null terminator)
        if s and s[-1] == 0:
            s = s[:-1]
        return s.decode('utf-8', errors='replace')

    def read_uint8(self) -> int:
        # uint8 是 1 字节对齐，无需 align
        val = self._data[self._offset]
        self._offset += 1
        return val

    def read_image_data(self, length: int) -> bytes:
        """读取图像像素数据 (uint8 序列). 序列元素是 1 字节对齐."""
        # 序列长度前缀已在对齐到 4 字节处读取
        # 序列元素 (uint8) 从当前偏移开始
        data = self._data[self._offset:self._offset+length]
        self._offset += length
        return data


def parse_sensor_msgs_image(raw_data: bytes) -> dict:
    """将 CDR 序列化的 sensor_msgs/msg/Image 解析为字典.

    返回字段:
        sec, nanosec: 时间戳
        frame_id:     "Color" 或 "Depth"
        height, width: 图像尺寸
        encoding:     "rgb8" 或 "16UC1" (OpenCV 兼容编码名称)
        is_bigendian: 0 (CDR_LE 始终为 0)
        step:         行步长 (字节)
        data:         像素数据 (bytes)
    """
    reader = CDRReader(raw_data)

    # Header
    sec = reader.read_int32()
    nanosec = reader.read_uint32()
    frame_id = reader.read_string()

    # Image metadata
    height = reader.read_uint32()
    width = reader.read_uint32()
    encoding = reader.read_string()

    is_bigendian = reader.read_uint8()

    # step (uint32, 4字节对齐)
    reader._align(4)
    step = reader.read_uint32()

    # data length (uint32, 4字节对齐)
    reader._align(4)
    data_len = reader.read_uint32()

    # pixel data
    img_data = reader.read_image_data(data_len)

    return {
        'sec': sec,
        'nanosec': nanosec,
        'frame_id': frame_id,
        'height': height,
        'width': width,
        'encoding': encoding,
        'is_bigendian': is_bigendian,
        'step': step,
        'data_len': data_len,
        'data': img_data,
    }


# ──────────────────────────────────────────────────────────
# 深度→彩色对齐
# ──────────────────────────────────────────────────────────

def align_depth_to_color(
    depth_img: np.ndarray,    # uint16 mm, shape (480, 848)
    K_depth: np.ndarray,      # 深度内参 (3×3)
    K_color: np.ndarray,      # 彩色内参 (3×3)
    T_d2c: np.ndarray,        # T_depth→color (4×4)
    color_h: int = 480,
    color_w: int = 640,
) -> np.ndarray:
    """将深度图对齐到彩色坐标系, 输出与彩色图尺寸一致的深度图.

    算法 (模拟 pyrealsense2 的 rs.align(COLOR)):
        对每个深度像素 (u_d, v_d, z):
          1. 反投影到深度相机 3D 坐标
          2. 用 T_depth→color 变换到彩色相机坐标系
          3. 用彩色内参投影到彩色像素平面
          4. 将深度值填入输出 (最近深度去重)

    Args:
        depth_img: 原始深度图 (uint16 mm), shape=(480, 848)
        K_depth: 深度相机内参
        K_color: 彩色相机内参
        T_d2c: T_depth→color 变换矩阵
        color_h, color_w: 彩色图尺寸

    Returns:
        aligned_depth: 对齐后深度图 (uint16 mm), shape=(color_h, color_w)
    """
    depth_h, depth_w = depth_img.shape

    # 初始化输出深度图 (0 = 无效)
    aligned = np.zeros((color_h, color_w), dtype=np.uint16)
    # 记录已填入的深度值 (用于最近深度仲裁)
    depth_buffer = np.full((color_h, color_w), np.inf, dtype=np.float32)

    # 提取 R_dc, t_dc
    R_dc = T_d2c[:3, :3]
    t_dc = T_d2c[:3, 3]

    # 反投影参数
    fx_d, fy_d = K_depth[0, 0], K_depth[1, 1]
    cx_d, cy_d = K_depth[0, 2], K_depth[1, 2]

    # 投影参数
    fx_c, fy_c = K_color[0, 0], K_color[1, 1]
    cx_c, cy_c = K_color[0, 2], K_color[1, 2]

    # 创建深度像素网格
    v_d, u_d = np.mgrid[0:depth_h, 0:depth_w]
    v_d = v_d.astype(np.float32).reshape(-1)
    u_d = u_d.astype(np.float32).reshape(-1)

    # 深度值 (米)
    z = depth_img.ravel().astype(np.float32) * 0.001  # mm → m

    # 过滤无效深度
    valid = z > 0.001  # D435 最小测距约 0.28m, 但保留 1mm 以上
    u_d = u_d[valid]
    v_d = v_d[valid]
    z = z[valid]
    depth_mm = (z * 1000).astype(np.uint16)  # 保留原始 mm 值

    if len(z) == 0:
        return aligned

    # 1. 反投影到深度相机 3D 坐标
    X_d = (u_d - cx_d) * z / fx_d
    Y_d = (v_d - cy_d) * z / fy_d
    Z_d = z

    P_d = np.stack([X_d, Y_d, Z_d], axis=1)  # (N, 3)

    # 2. 变换到彩色相机坐标系
    P_c = (R_dc @ P_d.T + t_dc.reshape(3, 1)).T  # (N, 3)

    # 3. 投影到彩色像素坐标
    u_c = (P_c[:, 0] * fx_c / P_c[:, 2] + cx_c).astype(np.int32)
    v_c = (P_c[:, 1] * fy_c / P_c[:, 2] + cy_c).astype(np.int32)

    # 4. 过滤超出彩色图像范围的像素
    in_bounds = (u_c >= 0) & (u_c < color_w) & (v_c >= 0) & (v_c < color_h)
    u_c = u_c[in_bounds]
    v_c = v_c[in_bounds]
    depth_mm = depth_mm[in_bounds]
    z_valid = z[in_bounds]

    # 5. 最近深度仲裁 (多个深度像素映射到同一彩色像素时, 取最近的)
    for i in range(len(u_c)):
        u, v, d_mm, d_m = u_c[i], v_c[i], int(depth_mm[i]), z_valid[i]
        if d_m < depth_buffer[v, u]:
            depth_buffer[v, u] = d_m
            aligned[v, u] = d_mm

    return aligned


# ──────────────────────────────────────────────────────────
# 帧配对 (Color ↔ Depth 时间戳对齐)
# ──────────────────────────────────────────────────────────

def pair_color_depth(color_msgs: list, depth_msgs: list) -> list:
    """将 color 和 depth 消息按时间戳配对.

    D435 录制时 color 和 depth 是交替采集的 (非硬件同步),
    每个 color 帧后面约 8ms 跟着对应的 depth 帧。
    采取策略: 对每个 color 帧, 找时间戳最接近的 depth 帧。

    Returns:
        list of (color_msg, depth_msg)
    """
    pairs = []
    depth_iter = iter(depth_msgs)
    cur_depth = next(depth_iter, None)

    for color_msg in color_msgs:
        color_ts = color_msg['sec'] * 1e9 + color_msg['nanosec']

        # 向前推进 depth 直到找到最接近的匹配
        best_depth = cur_depth
        while cur_depth is not None:
            depth_ts = cur_depth['sec'] * 1e9 + cur_depth['nanosec']
            if depth_ts <= color_ts:
                # depth 在 color 之前或同时, 继续前进
                best_depth = cur_depth
                cur_depth = next(depth_iter, None)
            else:
                # depth 在 color 之后, 检查哪个更接近
                if best_depth is None:
                    best_depth = cur_depth
                else:
                    best_ts = best_depth['sec'] * 1e9 + best_depth['nanosec']
                    if abs(depth_ts - color_ts) < abs(best_ts - color_ts):
                        best_depth = cur_depth
                break

        if best_depth is not None:
            pairs.append((color_msg, best_depth))

    return pairs


# ──────────────────────────────────────────────────────────
# .db3 数据提取
# ──────────────────────────────────────────────────────────

def extract_from_db3(db3_path: str, output_dir: str, max_frames: int = None):
    """从 .db3 文件提取彩色帧和对齐后深度帧.

    Args:
        db3_path: .db3 文件路径
        output_dir: 输出目录 (将创建 color/ 和 depth/ 子目录)
        max_frames: 最大提取帧数 (None = 全量)
    """
    db3_path = Path(db3_path)
    output_dir = Path(output_dir)
    color_dir = output_dir / "color"
    depth_dir = output_dir / "depth"
    color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Extract] 源文件: {db3_path}")
    print(f"[Extract] 输出: {output_dir}")

    # ── 动态读取相机参数 ──
    cam = read_camera_params(str(db3_path))
    K_color = cam["K_color"]
    K_depth = cam["K_depth"]
    T_d2c = cam["T_d2c"]
    color_w = cam["color_w"]
    color_h = cam["color_h"]

    print(f"[Extract] 彩色内参: fx={K_color[0,0]:.1f} fy={K_color[1,1]:.1f} "
          f"cx={K_color[0,2]:.1f} cy={K_color[1,2]:.1f} ({color_w}x{color_h})")
    print(f"[Extract] 深度内参: fx={K_depth[0,0]:.1f} fy={K_depth[1,1]:.1f} "
          f"cx={K_depth[0,2]:.1f} cy={K_depth[1,2]:.1f} ({cam['depth_w']}x{cam['depth_h']})")
    print(f"[Extract] T_d2c: translation=[{T_d2c[0,3]:.4f}, {T_d2c[1,3]:.4f}, {T_d2c[2,3]:.4f}]")

    # ── 读取 .db3 图像帧 ──
    conn = sqlite3.connect(str(db3_path))
    cursor = conn.cursor()

    # 获取 topic_id 映射
    cursor.execute("SELECT id, name FROM topics WHERE name LIKE '%Color_0/image/data' OR name LIKE '%Depth_0/image/data'")
    topic_map = {name: tid for tid, name in cursor.fetchall()}

    color_tid = None
    depth_tid = None
    for name, tid in topic_map.items():
        if 'Color' in name:
            color_tid = tid
        elif 'Depth' in name:
            depth_tid = tid

    if color_tid is None or depth_tid is None:
        print("[Extract] ✗ 未找到 Color/Depth topic")
        conn.close()
        return

    # 读取所有消息并按类型分组
    cursor.execute(
        "SELECT topic_id, timestamp, data FROM messages "
        "WHERE topic_id IN (?, ?) ORDER BY timestamp",
        (color_tid, depth_tid)
    )

    color_msgs = []
    depth_msgs = []
    for topic_id, ts, data in cursor:
        try:
            parsed = parse_sensor_msgs_image(data)
            if topic_id == color_tid:
                color_msgs.append(parsed)
            else:
                depth_msgs.append(parsed)
        except Exception as e:
            # 跳过解析失败的消息
            continue

    conn.close()

    print(f"[Extract] Color 消息: {len(color_msgs)}, Depth 消息: {len(depth_msgs)}")

    # ── 帧配对 ──
    pairs = pair_color_depth(color_msgs, depth_msgs)
    print(f"[Extract] 配对帧数: {len(pairs)}")

    if max_frames:
        pairs = pairs[:max_frames]

    # ── 逐帧处理 ──
    for idx, (color_msg, depth_msg) in enumerate(pairs):
        # 解码彩色图像 (rgb8 → BGR)
        color_flat = np.frombuffer(color_msg['data'], dtype=np.uint8)
        h, w = color_msg['height'], color_msg['width']
        color_rgb = color_flat.reshape(h, w, 3)
        color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)

        # 解码深度图像 (16UC1 → uint16 mm)
        depth_raw = np.frombuffer(depth_msg['data'], dtype=np.uint16)
        d_h, d_w = depth_msg['height'], depth_msg['width']
        depth_img = depth_raw.reshape(d_h, d_w)

        # 深度→彩色对齐 (使用从 .db3 动态读取的内参和外参)
        depth_aligned = align_depth_to_color(
            depth_img,
            K_depth,
            K_color,
            T_d2c,
            color_h=h,
            color_w=w,
        )

        # 保存
        color_out = color_dir / f"{idx}.png"
        depth_out = depth_dir / f"{idx}.png"
        cv2.imwrite(str(color_out), color_bgr)
        cv2.imwrite(str(depth_out), depth_aligned)

        if (idx + 1) % 50 == 0:
            print(f"[Extract] 已处理 {idx + 1}/{len(pairs)} 帧...")

    print(f"[Extract] ✓ 完成! 共 {len(pairs)} 帧")
    print(f"  Color: {color_dir}/")
    print(f"  Depth: {depth_dir}/")

    # ── 保存相机内参 ──
    cam_K_path = output_dir / "cam_K.json"
    with open(cam_K_path, 'w') as f:
        json.dump({
            "K": K_color.tolist(),
            "width": color_w,
            "height": color_h,
            "depth_scale": 0.001,  # D435 uint16 → 米
            "data_source": "db3",
            "note": "对齐后深度图使用此彩色内参, 像素一一对应",
        }, f, indent=2)
    print(f"  内参: {cam_K_path}")

    # 保存深度 scale
    scale_path = output_dir / "depth_scale.json"
    with open(scale_path, 'w') as f:
        json.dump({"scale": 0.001, "unit": "meter per uint16 count"}, f, indent=2)

    return len(pairs)


# ──────────────────────────────────────────────────────────
# self-demo 文件夹数据提取 (Stereo 双目光学 + FoundationStereo 深度)
# ──────────────────────────────────────────────────────────

def extract_from_self_demo(input_dir: str, output_dir: str, max_frames: int = None):
    """从 self-demo 文件夹提取数据到 test_data 格式.

    self-demo 文件夹内包含:
      - images.h5:  RGB帧 (frames/rgb/png) + 物体mask (masks/objects/*/png)
      - depth.h5:   深度图 (depth/depth, float32 米, shape=(N,1080,1920))
      - meshes.h5:  物体mesh (objects/*/scaled_obj, glb)
      - meta.json:  相机内参 (rectified 字段)

    RGB 和深度已逐像素对齐 (同分辨率 1920×1080), 无需再对齐.
    深度值从 float32 米转为 uint16 毫米保存, 与 db3 输出格式一致.

    Args:
        input_dir:  self-demo 文件夹路径
        output_dir: 输出目录 (将创建 color/ depth/ mesh/ 子目录)
        max_frames: 最大提取帧数 (None = 全量)
    """
    import h5py

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    color_dir = output_dir / "color"
    depth_dir = output_dir / "depth"
    mesh_dir = output_dir / "mesh"

    for d in [color_dir, depth_dir, mesh_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"[SelfDemo] 源文件夹: {input_dir}")
    print(f"[SelfDemo] 输出: {output_dir}")

    # ── 读取 meta.json 获取相机内参 ──
    meta_path = input_dir / "meta.json"
    if not meta_path.exists():
        print(f"[SelfDemo] ✗ 未找到 meta.json: {meta_path}")
        return

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    rectified = meta.get('rectified', {})
    width = rectified.get('width', 1920)
    height = rectified.get('height', 1080)
    fx = rectified.get('fx', 802.9)
    fy = rectified.get('fy', 802.9)
    cx = rectified.get('cx', 991.9)
    cy = rectified.get('cy', 549.0)

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    print(f"[SelfDemo] 相机内参: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} ({width}x{height})")

    # ── 读取物体名称 (从 images.h5 masks 中获取) ──
    images_path = input_dir / "images.h5"
    if not images_path.exists():
        print(f"[SelfDemo] ✗ 未找到 images.h5: {images_path}")
        return

    object_name = None
    try:
        with h5py.File(str(images_path), 'r') as f:
            masks_objects = f.get('masks/objects')
            if masks_objects is not None:
                obj_names = list(masks_objects.keys())
                if obj_names:
                    object_name = obj_names[0]
    except Exception:
        pass
    # Fallback: 从 meta.json qwen_detect 读取
    if object_name is None:
        object_name = meta.get('qwen_detect', {}).get('object', 'object')

    print(f"[SelfDemo] 物体: {object_name}")

    # ── 提取 RGB 帧 ──
    num_frames = 0
    with h5py.File(str(images_path), 'r') as f:
        rgb_pngs = f['frames/rgb/png'][:]
        num_frames = len(rgb_pngs)
        if max_frames:
            num_frames = min(num_frames, max_frames)
            rgb_pngs = rgb_pngs[:num_frames]

        print(f"[SelfDemo] RGB 帧数: {num_frames}")

        for idx in range(num_frames):
            png_bytes = bytes(rgb_pngs[idx])
            img = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)
            cv2.imwrite(str(color_dir / f"{idx}.png"), img)

            if (idx + 1) % 50 == 0:
                print(f"[SelfDemo] RGB 已处理 {idx + 1}/{num_frames} 帧...")

        # ── 提取物体 mask (第一帧) ──
        try:
            masks_objects = f.get('masks/objects')
            if masks_objects is not None:
                obj_names = list(masks_objects.keys())
                if obj_names:
                    mask_png = masks_objects[obj_names[0]]['png'][0]
                    mask_img = cv2.imdecode(
                        np.frombuffer(bytes(mask_png), np.uint8),
                        cv2.IMREAD_GRAYSCALE,
                    )
                    cv2.imwrite(str(output_dir / "0_mask.png"), mask_img)
                    print(f"[SelfDemo] 物体mask已保存: 0_mask.png (object: {obj_names[0]})")
        except Exception as e:
            print(f"[SelfDemo] ⚠ 无法提取mask: {e}")

    # ── 提取深度帧 ──
    depth_path = input_dir / "depth.h5"
    if depth_path.exists():
        with h5py.File(str(depth_path), 'r') as f:
            depths = f['depth/depth'][:]
            if max_frames:
                depths = depths[:max_frames]

            for idx in range(len(depths)):
                depth_m = depths[idx]  # float32, 米
                # 转为 uint16 mm (与 db3 输出格式一致)
                depth_mm = (depth_m * 1000).astype(np.uint16)
                cv2.imwrite(str(depth_dir / f"{idx}.png"), depth_mm)

                if (idx + 1) % 50 == 0:
                    print(f"[SelfDemo] Depth 已处理 {idx + 1}/{len(depths)} 帧...")
    else:
        print(f"[SelfDemo] ⚠ 未找到 depth.h5: {depth_path}")

    # ── 提取 mesh (优先从 meshes.h5, 回退到文件系统) ──
    meshes_h5_path = input_dir / "meshes.h5"
    mesh_extracted = False
    if meshes_h5_path.exists():
        try:
            with h5py.File(str(meshes_h5_path), 'r') as f:
                objects_grp = f.get('objects')
                if objects_grp is not None:
                    obj_names = list(objects_grp.keys())
                    if obj_names:
                        obj_grp = objects_grp[obj_names[0]]

                        if 'scaled_obj' in obj_grp:
                            obj_bytes = bytes(obj_grp['scaled_obj'][0])
                            with open(str(mesh_dir / "mesh.obj"), 'wb') as fout:
                                fout.write(obj_bytes)
                            mesh_extracted = True
                            print(f"[SelfDemo] Mesh已保存: mesh.obj (from meshes.h5)")

                        if 'glb' in obj_grp:
                            glb_bytes = bytes(obj_grp['glb'][0])
                            with open(str(mesh_dir / "mesh.glb"), 'wb') as fout:
                                fout.write(glb_bytes)
        except Exception as e:
            print(f"[SelfDemo] ⚠ 从meshes.h5提取失败: {e}")

    if not mesh_extracted:
        # Fallback: 从文件系统目录复制
        import shutil
        meshes_dir = input_dir / "meshes" / "objects"
        if meshes_dir.exists():
            obj_dirs = list(meshes_dir.iterdir())
            if obj_dirs:
                frame_dirs = [d for d in obj_dirs[0].iterdir() if d.is_dir()]
                if frame_dirs:
                    frame_dir = frame_dirs[0]
                    for fname in ['scaled_mesh.obj', 'mesh.glb', 'mesh.obj']:
                        src = frame_dir / fname
                        if src.exists():
                            dst_name = "mesh.obj" if fname in ('scaled_mesh.obj', 'mesh.obj') else fname
                            shutil.copy2(str(src), str(mesh_dir / dst_name))
                            mesh_extracted = True
                    if mesh_extracted:
                        print(f"[SelfDemo] Mesh文件已从文件系统复制")

    if not mesh_extracted:
        print(f"[SelfDemo] ⚠ 未找到mesh文件")

    # ── 保存相机内参 ──
    cam_K_path = output_dir / "cam_K.json"
    with open(cam_K_path, 'w') as f:
        json.dump({
            "K": K.tolist(),
            "width": width,
            "height": height,
            "depth_scale": 0.001,
            "data_source": "self_demo",
            "note": "self-demo stereo 数据, RGB与深度逐像素对齐 (同分辨率)",
        }, f, indent=2)
    print(f"  内参: {cam_K_path}")

    # 保存深度 scale
    scale_path = output_dir / "depth_scale.json"
    with open(scale_path, 'w') as f:
        json.dump({"scale": 0.001, "unit": "meter per uint16 count"}, f, indent=2)

    print(f"[SelfDemo] ✓ 完成! 共 {num_frames} 帧 (RGB+Depth)")
    print(f"  Color: {color_dir}/")
    print(f"  Depth: {depth_dir}/")
    print(f"  Mesh:  {mesh_dir}/")

    return num_frames


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="从 RealSense D435 .db3 或 self-demo 文件夹中提取帧和数据",
    )
    parser.add_argument(
        "--input_type", type=str, default="db3",
        choices=["db3", "self_demo"],
        help="输入数据类型: db3 (Realsense ROS2 bag) 或 self_demo (Stereo 双目光学文件夹)",
    )
    parser.add_argument(
        "--input_path", type=str, default=None,
        help="输入路径: db3 模式下为 .db3 文件; self_demo 模式下为文件夹路径",
    )
    parser.add_argument(
        "--db3_path", type=str, default=None,
        help="(已废弃, 请用 --input_path) .db3 文件路径",
    )
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录 (默认: test_data/<数据名>/)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="最大提取帧数 (调试用)")
    parser.add_argument("--disable_alignment", action="store_true",
                        help="(仅 db3 模式) 禁用深度对齐 (直接保存原始深度)")
    args = parser.parse_args()

    # 向后兼容: --db3_path → --input_path + --input_type=db3
    if args.input_path is None and args.db3_path is not None:
        args.input_path = args.db3_path
        if args.input_type != "db3":
            print("警告: --db3_path 仅适用于 db3 模式, 已强制设为 db3")
            args.input_type = "db3"

    if args.input_path is None:
        parser.error("必须指定 --input_path (或 --db3_path)")

    # 验证输入路径存在
    if not os.path.exists(args.input_path):
        print(f"错误: 输入路径不存在: {args.input_path}")
        sys.exit(1)

    # 自动解析输出目录名称
    if args.output_dir is None:
        if args.input_type == "db3":
            object_name = Path(args.input_path).stem  # mouse.db3 → mouse
        else:
            object_name = Path(args.input_path).name  # self-demo1 → self-demo1
        args.output_dir = str(Path("test_data") / object_name)

    # ── 分支路由 ──
    if args.input_type == "self_demo":
        extract_from_self_demo(args.input_path, args.output_dir, args.max_frames)
    else:
        extract_from_db3(args.input_path, args.output_dir, args.max_frames)


if __name__ == "__main__":
    main()
