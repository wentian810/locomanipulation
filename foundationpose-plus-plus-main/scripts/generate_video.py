#!/usr/bin/env python3
"""
将 FoundationPose++ 推理产生的逐帧可视化图合成为 mp4 视频。

用法:
    python scripts/generate_video.py \
      --frame_dir test_data/mouse/pose_vis \
      --output test_data/mouse/mouse_pose.mp4 \
      --fps 30

也可以对原始 color 帧直接生成视频 (不带 bbox):
    python scripts/generate_video.py \
      --frame_dir test_data/mouse/color \
      --output test_data/mouse/color_only.mp4 \
      --fps 30
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def generate_video(
    frame_dir: str,
    output_path: str,
    fps: float = 30.0,
    pattern: str = "*.png",
    sort_by: str = "index",
    add_info: bool = False,
    object_name: str = "",
    total_pose_path: str = None,
):
    """将图像序列合成为 mp4 视频.

    Args:
        frame_dir: 帧图像目录
        output_path: 输出 mp4 路径
        fps: 帧率
        pattern: 图像文件匹配模式
        sort_by: 排序方式 ("index" = 按数字索引, "name" = 按文件名)
        add_info: 是否在视频底部添加信息栏
        object_name: 物体名称 (用于信息栏)
        total_pose_path: pose.npy 路径 (用于信息栏)
    """
    frame_dir = Path(frame_dir)
    if not frame_dir.exists():
        print(f"错误: 帧目录不存在: {frame_dir}")
        sys.exit(1)

    # 收集帧文件
    if pattern == "*.png":
        frame_files = sorted(frame_dir.glob("*.png"))
    elif pattern == "*.jpg":
        frame_files = sorted(frame_dir.glob("*.jpg"))
    else:
        frame_files = sorted(frame_dir.glob(pattern))

    if not frame_files:
        print(f"错误: 目录 {frame_dir} 中没有匹配 {pattern} 的图像")
        sys.exit(1)

    # 排序
    if sort_by == "index":
        frame_files.sort(key=lambda p: int(p.stem.split('.')[0]))
    else:
        frame_files.sort(key=lambda p: p.name)

    print(f"[Video] 帧图像: {len(frame_files)} 张")
    print(f"[Video] 输出: {output_path}")

    # 读取第一帧获取尺寸
    first_frame = cv2.imread(str(frame_files[0]))
    if first_frame is None:
        print(f"错误: 无法读取首帧: {frame_files[0]}")
        sys.exit(1)

    h, w = first_frame.shape[:2]

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not out.isOpened():
        print("错误: 无法创建视频文件 (检查编码器和路径)")
        sys.exit(1)

    for i, fpath in enumerate(frame_files):
        frame = cv2.imread(str(fpath))
        if frame is None:
            print(f"警告: 跳过损坏帧 {fpath.name}")
            continue

        if add_info:
            # 添加信息栏
            info_bar = np.zeros((40, w, 3), dtype=np.uint8)
            info_text = f"Frame: {i}/{len(frame_files)}"
            if object_name:
                info_text = f"Object: {object_name} | {info_text}"
            cv2.putText(info_bar, info_text, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            frame = np.vstack([frame, info_bar])

        out.write(frame)

        if (i + 1) % 100 == 0:
            print(f"[Video] 已写入 {i + 1}/{len(frame_files)} 帧...")

    out.release()
    print(f"[Video] ✓ 完成! 输出: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="将图像序列合成为 mp4 视频",
    )
    parser.add_argument("--frame_dir", type=str, required=True,
                        help="帧图像目录")
    parser.add_argument("--output", type=str, required=True,
                        help="输出 mp4 文件路径")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="视频帧率 (默认 30)")
    parser.add_argument("--pattern", type=str, default="*.png",
                        help="文件匹配模式 (默认 *.png)")
    parser.add_argument("--add_info", action="store_true",
                        help="在底部添加帧号等信息栏")
    parser.add_argument("--object_name", type=str, default="",
                        help="物体名称")
    args = parser.parse_args()

    generate_video(
        frame_dir=args.frame_dir,
        output_path=args.output,
        fps=args.fps,
        pattern=args.pattern,
        add_info=args.add_info,
        object_name=args.object_name,
    )


if __name__ == "__main__":
    main()
