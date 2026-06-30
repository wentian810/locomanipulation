#!/usr/bin/env python3
"""
Hunyuan3D Bridge: 调用 Hunyuan3D API 从 RGBA 抠图生成 3D mesh (.glb)。

用法:
    python scripts/hunyuan3d_bridge.py \
      --rgba_path test_data/mouse/0_mask_rgba.png \
      --output_mesh test_data/mouse/mesh/mesh.glb \
      --hunyuan3d_url http://localhost:18081

API 协议:
    POST {hunyuan3d_url}/generate
    Body: JSON {image: base64(png), seed: 42, octree_resolution: 256, ...}
    Response: 原始 GLB 字节流 (model/gltf-binary)

    超时: 900 秒 (15 min), Hunyuan3D 生成 mesh 需要 ~2 min
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import requests


def generate_mesh(
    rgba_path: str,
    output_glb_path: str,
    hunyuan3d_url: str = "http://localhost:18081",
    timeout: int = 900,
) -> bool:
    """调用 Hunyuan3D API 从 RGBA 抠图生成 mesh.

    Args:
        rgba_path: 输入 RGBA PNG 路径 (透明背景抠图)
        output_glb_path: 输出 GLB 文件路径
        hunyuan3d_url: Hunyuan3D API 地址
        timeout: HTTP 超时秒数

    Returns:
        bool: 成功 True, 失败 False
    """
    # 验证输入
    if not os.path.exists(rgba_path):
        print(f"[Hunyuan3D] 错误: RGBA 文件不存在: {rgba_path}")
        return False

    # 读取 RGBA 图片并 base64 编码
    with open(rgba_path, 'rb') as f:
        png_data = f.read()
    image_b64 = base64.b64encode(png_data).decode('utf-8')

    # 构建 API 请求
    api_url = f"{hunyuan3d_url.rstrip('/')}/generate"
    payload = {
        "image": image_b64,
        "remove_background": False,  # 背景已在 SAM-HQ 阶段移除
        "texture": False,            # model_worker 忽略此字段, 实际总是生成纹理
        "seed": 42,
        "octree_resolution": 256,
        "num_inference_steps": 5,
    }

    print(f"[Hunyuan3D] 请求: {api_url}")
    print(f"[Hunyuan3D] 输入: {rgba_path} ({len(png_data) / 1024:.1f} KB)")
    print(f"[Hunyuan3D] 输出: {output_glb_path}")
    print(f"[Hunyuan3D] 等待中 (最长 {timeout}s) ...")

    t_start = time.time()
    try:
        resp = requests.post(api_url, json=payload, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        print(f"[Hunyuan3D] 错误: 请求超时 ({timeout}s)")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"[Hunyuan3D] 错误: 无法连接 API → {api_url}")
        print(f"  {e}")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"[Hunyuan3D] 错误: HTTP {resp.status_code}")
        try:
            print(f"  响应: {resp.text[:500]}")
        except Exception:
            pass
        return False
    except Exception as e:
        print(f"[Hunyuan3D] 错误: {e}")
        return False

    elapsed = time.time() - t_start

    # 保存 GLB
    os.makedirs(os.path.dirname(output_glb_path), exist_ok=True)
    glb_size = len(resp.content)
    with open(output_glb_path, 'wb') as f:
        f.write(resp.content)

    print(f"[Hunyuan3D] ✓ 完成 耗时={elapsed:.1f}s 大小={glb_size / 1024:.1f} KB")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="调用 Hunyuan3D API 从 RGBA 抠图生成 3D mesh (.glb)"
    )
    parser.add_argument("--rgba_path", type=str, required=True,
                        help="RGBA 透明背景抠图路径")
    parser.add_argument("--output_mesh", type=str, required=True,
                        help="输出 GLB 路径")
    parser.add_argument("--hunyuan3d_url", type=str,
                        default=os.environ.get("HUNYUAN3D_URL", "http://localhost:18081"),
                        help="Hunyuan3D API 地址 (默认: http://localhost:18081)")
    parser.add_argument("--timeout", type=int, default=900,
                        help="HTTP 超时秒数 (默认: 900)")
    args = parser.parse_args()

    if not os.path.exists(args.rgba_path):
        print(f"错误: RGBA 文件不存在: {args.rgba_path}")
        sys.exit(1)

    ok = generate_mesh(
        rgba_path=args.rgba_path,
        output_glb_path=args.output_mesh,
        hunyuan3d_url=args.hunyuan3d_url,
        timeout=args.timeout,
    )

    if ok:
        print(f"OK output={args.output_mesh} size={os.path.getsize(args.output_mesh) / 1024:.0f}KB")
        sys.exit(0)
    else:
        print("FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
