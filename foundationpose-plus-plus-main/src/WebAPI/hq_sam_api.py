import torch
import numpy as np
from typing import List
import os
import cv2
import h5py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, Response
import json
import argparse
import uvicorn
from pydantic import BaseModel
from PIL import Image

from segment_anything import SamPredictor
from segment_anything_hq import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


############################################
# Define Sam Models
############################################
class SAMModel:
    def __init__(self, model_type: str, checkpoint_path: str, device: str):
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)
        self.transform = ResizeLongestSide(self.predictor.model.image_encoder.img_size)
        self.device = device

    def segment(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        if image.shape[2] != 3:
            raise ValueError("The input image should have 3 channels (RGB).")

        original_image_size = image.shape[:2]  # (H, W)
        transformed_image = self.transform.apply_image(image)
        transformed_image = torch.as_tensor(transformed_image, device=self.device)
        transformed_image = transformed_image.permute(2, 0, 1).unsqueeze(0)  # Shape: [1, 3, H, W]

        self.predictor.set_torch_image(transformed_image, original_image_size)

        boxes = np.array([bbox])  # Shape: [1, 4]
        boxes_transformed = self.transform.apply_boxes(boxes, original_image_size)
        boxes_transformed = torch.as_tensor(boxes_transformed, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            masks, scores, logits = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=boxes_transformed,
                multimask_output=False,
                hq_token_only=False,
            )

        mask = masks[0].cpu().numpy()  # Shape: [H, W]
        return mask


############################################
# Release Web API
############################################
app = FastAPI(title="SAM Web API")

# Initialize device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    else "cpu"
)

# ── 路径映射：容器内 /workspace → 宿主机实际项目根目录 ──
_HOST_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _resolve_host_path(container_path: str) -> str:
    """将容器内的 /workspace/xxx 映射为宿主机实际路径。"""
    return container_path.replace("/workspace", _HOST_PROJECT_ROOT, 1)

class Message(BaseModel):
    frame_path: str
    bbox_xywh: List[int]
    output_mask_path: str

@app.post("/hq_sam")
async def segment_image(message: Message):
    try:
        # 路径映射：容器路径 → 宿主机实际路径
        host_frame_path = _resolve_host_path(message.frame_path)
        host_output_mask_path = _resolve_host_path(message.output_mask_path)

        frame = Image.open(host_frame_path)
        frame = np.array(frame)
        bbox_xyxy = [
            message.bbox_xywh[0],
            message.bbox_xywh[1],
            message.bbox_xywh[0] + message.bbox_xywh[2],
            message.bbox_xywh[1] + message.bbox_xywh[3]
        ]

        if frame.shape[2] == 3:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("The input image should have 3 channels.")

        # Perform segmentation
        mask = sam_model.segment(image_rgb, bbox_xyxy)

        # Convert mask to uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        if mask_uint8.ndim == 3 and mask_uint8.shape[0] == 1:
            mask_uint8 = mask_uint8.squeeze(0)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(host_output_mask_path), exist_ok=True)

        # Save binary mask (单通道灰度图, 0=背景 255=物体)
        cv2.imwrite(host_output_mask_path, mask_uint8)

        # Also save masked RGBA (透明背景抠图, 供 Hunyuan3D mesh 生成)
        rgba_path = host_output_mask_path.replace(".png", "_rgba.png")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgba = np.dstack([frame_rgb, mask_uint8])  # RGB + Alpha
        cv2.imwrite(rgba_path, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

        return Response(content="Mask generated successfully", media_type="text/plain", status_code=200)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SAM Web API")

    # 自动检测项目根目录（宿主机和容器内都能正确解析）
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _default_ckpt = os.path.join(_project_root, "sam-hq", "pretrained_checkpoints", "sam_hq_vit_l.pth")

    parser.add_argument("--checkpoint_path", type=str, default=_default_ckpt)
    parser.add_argument("--port", type=int, default=9002)
    args = parser.parse_args()

    # Initialize SAM model with the checkpoint path
    sam_model = SAMModel(
        model_type=os.getenv("HQ_SAM_MODEL_TYPE", "vit_l"),
        checkpoint_path=args.checkpoint_path,
        device=device
    )

    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=args.port)