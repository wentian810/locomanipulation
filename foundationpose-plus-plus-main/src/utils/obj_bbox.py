import re
import sys
import cv2
import argparse
import base64
import json
import requests
from PIL import Image
from typing import Union, List, Tuple
import numpy as np


def visualize_bbox(
        image: np.ndarray, 
        bbox_xywh: Union[List, Tuple], 
        output_path: str
):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if not isinstance(image, np.ndarray):
        raise ValueError("The image should be a numpy.ndarray.")

    if not (isinstance(bbox_xywh, (list, tuple)) and len(bbox_xywh) == 4):
        raise ValueError("The bbox should be a list or tuple with four elements (x, y, w, h).")

    x1, y1, w, h = bbox_xywh
    x2, y2 = x1 + w, y1 + h

    cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

    label = f"[{x1}, {y1}, {w}, {h}]"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
    text_x = x1
    text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10 + text_size[1]

    cv2.putText(image, label, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)

    cv2.imwrite(output_path, image)


def _parse_bbox_output(
        output_text: str,
        img_H: int,
        img_W: int,
) -> List:
    """从模型输出中解析边界框坐标 (x, y, w, h)。

    支持多种格式:
      - JSON: [{"bbox_2d": [x1, y1, x2, y2], "label": "..."}]  (Qwen3-VL 千分比)
      - 文本: (x1,y1,x2,y2)   Qwen2-VL 千分比归一化格式
      - 文本: [x1, y1, x2, y2]  像素坐标 xyxy
    """
    # 去除 markdown 代码块包裹 (Qwen3-VL 有时输出 ```json ... ```)
    clean = output_text.strip()
    if clean.startswith("```"):
        clean = re.sub(r'^```(?:json)?\s*\n?', '', clean)
        clean = re.sub(r'\n?```\s*$', '', clean)

    # 格式 1: JSON bbox_2d — Qwen3-VL-30B
    try:
        parsed = json.loads(clean)
        if isinstance(parsed, list) and len(parsed) > 0 and "bbox_2d" in parsed[0]:
            x1, y1, x2, y2 = parsed[0]["bbox_2d"]
            label = parsed[0].get("label", "")
            # bbox_2d 是千分比归一化坐标 (0-1000)
            return [
                int(x1 * img_W / 1000),
                int(y1 * img_H / 1000),
                int((x2 - x1) * img_W / 1000),
                int((y2 - y1) * img_H / 1000),
            ]
    except (json.JSONDecodeError, TypeError, IndexError, KeyError):
        pass

    # 格式 2: 纯文本 (数字,数字,数字,数字) — Qwen2-VL 千分比
    pattern1 = r'\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
    matches = re.findall(pattern1, output_text)
    if matches:
        x1, y1, x2, y2 = map(int, matches[0])
        if x2 > 1000 or y2 > 1000:
            return [x1, y1, x2 - x1, y2 - y1]
        return [
            int(x1 * img_W / 1000),
            int(y1 * img_H / 1000),
            int((x2 - x1) * img_W / 1000),
            int((y2 - y1) * img_H / 1000),
        ]

    # 格式 3: 纯文本 [数字, 数字, 数字, 数字]
    pattern2 = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    matches = re.findall(pattern2, output_text)
    if matches:
        a, b, c, d = map(int, matches[0])
        if c < a or d < b:
            return [a, b, c, d]
        return [a, b, c - a, d - b]

    print(f"[WARN] 无法解析 bbox, 模型原始输出:\n{output_text}", file=sys.stderr)
    raise ValueError(f"无法从模型输出中解析 bbox: {output_text[:200]}")


def _encode_image_base64(image_path: str) -> str:
    """将图片编码为 base64 data URL."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _build_openai_vision_request(
    frame_path: str,
    object_name: str,
    reference_img_path: str = None,
    model: str = "qwen3-vl-30b-a3b-instruct",
) -> dict:
    """构建 OpenAI-compatible vision API 请求体.

    使用 Qwen3-VL-30B 远程服务 (OpenAI 兼容接口).
    """
    content = []

    if reference_img_path is not None:
        content.append({
            "type": "text",
            "text": f"第1张图片里是一个{object_name}的照片",
        })
        b64_ref = _encode_image_base64(reference_img_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64_ref}"},
        })

    content.append({
        "type": "text",
        "text": f"你能在图片里找到{object_name}的锚框吗？请用(x,y,x,y)的格式给我锚框",
    })
    b64_frame = _encode_image_base64(frame_path)
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{b64_frame}"},
    })

    return {
        "model": model,
        "messages": [
            {"role": "user", "content": content},
        ],
        "max_tokens": 256,
        "temperature": 0.0,
    }


def request_bbox(
        frame_path: str,
        object_name: str,
        visualize_path: str,
        web_api_url: str,
        reference_img_path: str = None,
):
    """向 Qwen3-VL 远程 API 请求物体边界框 (OpenAI 兼容接口)."""
    frame = np.array(Image.open(frame_path))
    img_H, img_W = frame.shape[0], frame.shape[1]

    body = _build_openai_vision_request(frame_path, object_name, reference_img_path)

    # API key = "EMPTY" for self-hosted vLLM/SGLang
    response = requests.post(
        web_api_url,
        json=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer EMPTY",
        },
        timeout=60,
    )

    if response.status_code == 200:
        result = response.json()
        output_text = result["choices"][0]["message"]["content"]
        bbox_xywh = _parse_bbox_output(output_text, img_H, img_W)
        visualize_bbox(frame, bbox_xywh, visualize_path)
        print(f"{bbox_xywh}")
    else:
        print(f"Error: {response.status_code}, {response.text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_path", type=str, default="/workspace/test_data/color/0.jpg", help="The target frame (.jpg/.png/...) path.")
    parser.add_argument("--object_name", type=str, default="蓝色乐高积木", help="The object description. CHINESE will be better than English, for Qwen models.")
    parser.add_argument("--visualize_path", type=str, default="/workspace/test_data/0_bbox.png", help="The visualization image with bbox overlapped on the target frame.")
    parser.add_argument("--web_api_url", type=str,
                        default="http://192.168.10.242:12067/v1/chat/completions")
    parser.add_argument("--reference_img_path", type=str, default=None, help="One-shot-learning prompt. With this prompt, the Qwen2-VL performance will be marginally better. \
                        NOTICE: only ONE reference image being provided can peak the best performace.")

    args = parser.parse_args()

    request_bbox(
            args.frame_path,
            args.object_name,
            args.visualize_path,
            args.web_api_url,
            args.reference_img_path,
    )