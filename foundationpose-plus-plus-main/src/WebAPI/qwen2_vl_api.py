import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
import uvicorn
import re
import torch
import argparse
from typing import List, Tuple
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from pydantic import BaseModel


############################################
# Define Qwen2-VL Model
############################################
class Qwen2VLModel:
    def __init__(
            self, 
            weight_path: str, 
            device_map: str = "auto",
            accelerate: bool = False,
    ):
        if accelerate:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=weight_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                device_map=device_map,
            )
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=weight_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                device_map=device_map,
            )
            
        self.processor = AutoProcessor.from_pretrained(
                pretrained_model_name_or_path=weight_path,
        )

    def infer(
            self,
            image_paths: List[str],
            text: str, 
    )->str:
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                    }
                    for img_path in image_paths
                ] + [
                    {
                        "type": "text", 
                        "text": text,
                    },
                ]
            }
        ]

        templated_text = self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(message)
        inputs = self.processor(
            text=[templated_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text
        

############################################
# Release Web API
############################################
# ── 路径映射：容器内 /workspace → 宿主机实际项目根目录 ──
_HOST_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _resolve_host_path(container_path: str) -> str:
    """将容器内的 /workspace/xxx 映射为宿主机实际路径。"""
    return container_path.replace("/workspace", _HOST_PROJECT_ROOT, 1)

app = FastAPI(title="Qwen2-VL Web API")

class Message(BaseModel):
    image_paths: List[str]
    text_input: str

@app.post("/qwen2_vl")
async def generate_bounding_box(message: Message):
    try:
        # 路径映射：容器路径 → 宿主机实际路径
        host_image_paths = [_resolve_host_path(p) for p in message.image_paths]

        output_text = qwen2_vl_model.infer(
            image_paths=host_image_paths,
            text=message.text_input,
        )
        # bbox = _parse_qwen2_vl_output(output_text)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
    return {"output": output_text}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SAM Web API")

    # 自动检测项目根目录（宿主机和容器内都能正确解析）
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _default_weight = os.path.join(_project_root, "Qwen2-VL", "weights")

    parser.add_argument("--weight_path", type=str, default=_default_weight)
    parser.add_argument("--accelerate_or_not", type=bool, default=False, help="Run Qwen with half precise and low CUDA memory. Though save the CUDA memory, it may decrease the correctness of the inference result.")
    parser.add_argument("--port", type=int, default=9003, help="Which port you want deploy your web api on")
    args = parser.parse_args()

    # Initialize Qwen-VL model with the checkpoint path
    qwen2_vl_model = Qwen2VLModel(
        weight_path = args.weight_path,
        accelerate=args.accelerate_or_not,
    )

    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=args.port)