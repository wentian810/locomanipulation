#!/usr/bin/env python3
"""
Hunyuan3D Bridge v2: 调用腾讯云 Hunyuan3D 3.0 API + PBR 材质

流程: 提交任务(SubmitHunyuanTo3DProJob) → 轮询等待 → 下载 zip → 解压 GLB

用法:
    export TENCENTCLOUD_SECRET_ID=...
    export TENCENTCLOUD_SECRET_KEY=...
    python scripts/hunyuan3d_bridge.py \
      --rgba_path test_data/mouse/0_mask_rgba.png \
      --output_mesh test_data/mouse/mesh/mesh.glb
"""

import argparse
import base64
import hashlib
import hmac
import io
import json
import os
import sys
import time
import zipfile
from datetime import datetime, timezone

import requests
from PIL import Image

# ═══════════════════════════════════════════════════════════════
# 生成参数
# ═══════════════════════════════════════════════════════════════
DEFAULT_ENABLE_PBR = True
DEFAULT_FACE_COUNT = 100000
DEFAULT_POLL_TIMEOUT = 900
DEFAULT_POLL_INTERVAL = 5.0
DEFAULT_REGION = "ap-guangzhou"


# ── TC3-HMAC-SHA256 签名 ──

def _sign_tc3(secret_key: str, date: str, service: str, string_to_sign: str) -> str:
    def _hmac(key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

    k_date = _hmac(("TC3" + secret_key).encode("utf-8"), date)
    k_service = _hmac(k_date, service)
    k_signing = _hmac(k_service, "tc3_request")
    return hmac.new(k_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()


def _credentials() -> tuple[str, str]:
    secret_id = os.environ.get("TENCENTCLOUD_SECRET_ID", "").strip()
    secret_key = os.environ.get("TENCENTCLOUD_SECRET_KEY", "").strip()
    if not secret_id or not secret_key:
        raise RuntimeError(
            "缺少腾讯云凭证。请设置 TENCENTCLOUD_SECRET_ID 和 "
            "TENCENTCLOUD_SECRET_KEY；不要把密钥写入源码。"
        )
    return secret_id, secret_key


def _call_tc3(
    action: str,
    payload: dict,
    *,
    region: str,
    request_timeout: int = 30,
) -> dict:
    """调用腾讯云 API (TC3 签名), 返回 Response dict."""
    secret_id, secret_key = _credentials()
    service = "ai3d"
    host = "ai3d.tencentcloudapi.com"
    version = "2025-05-13"
    timestamp = int(time.time())
    date = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")

    payload_str = json.dumps(payload)
    ct = "application/json; charset=utf-8"

    # 1. Canonical Request
    canonical_headers = (f"content-type:{ct}\n"
                         f"host:{host}\n"
                         f"x-tc-action:{action.lower()}\n")
    signed_headers = "content-type;host;x-tc-action"
    hashed_payload = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

    canonical_request = "\n".join([
        "POST", "/", "",
        canonical_headers,
        signed_headers,
        hashed_payload,
    ])

    # 2. String to Sign
    credential_scope = f"{date}/{service}/tc3_request"
    hashed_canonical = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
    string_to_sign = "\n".join([
        "TC3-HMAC-SHA256", str(timestamp), credential_scope, hashed_canonical,
    ])

    # 3. Signature
    signature = _sign_tc3(secret_key, date, service, string_to_sign)

    # 4. Authorization header
    authorization = (
        f"TC3-HMAC-SHA256 Credential={secret_id}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )

    headers = {
        "Authorization": authorization,
        "Content-Type": ct,
        "Host": host,
        "X-TC-Action": action,
        "X-TC-Timestamp": str(timestamp),
        "X-TC-Version": version,
        "X-TC-Region": region,
    }

    resp = requests.post(f"https://{host}", headers=headers, data=payload_str,
                         timeout=request_timeout)
    if resp.status_code != 200:
        print(f"[Hunyuan3D] HTTP {resp.status_code}: {resp.text[:500]}")
        return {"Error": {"Code": str(resp.status_code), "Message": resp.text}}

    result = resp.json()
    if "Response" in result and "Error" in result["Response"]:
        err = result["Response"]["Error"]
        print(f"[Hunyuan3D] API 错误: {err.get('Code')} — {err.get('Message')}")
    return result.get("Response", {})


# ── 核心函数 ──

def submit_job(
    image_base64: str,
    *,
    enable_pbr: bool,
    face_count: int,
    region: str,
) -> str | None:
    """提交 Image-to-3D 任务, 返回 JobId."""
    print(f"[Hunyuan3D] 提交任务 (PBR={enable_pbr}, FaceCount={face_count})")
    resp = _call_tc3(action="SubmitHunyuanTo3DProJob", payload={
        "Model": "3.0",
        "ImageBase64": image_base64,
        "EnablePBR": enable_pbr,
        "FaceCount": face_count,
        "GenerateType": "Normal",
    }, region=region)
    job_id = resp.get("JobId")
    if job_id:
        print(f"[Hunyuan3D] JobId: {job_id}")
    return job_id


def describe_job(job_id: str, *, region: str) -> dict:
    """查询任务状态."""
    return _call_tc3(action="QueryHunyuanTo3DProJob",
                     payload={"JobId": job_id}, region=region)


def wait_for_job(
    job_id: str,
    *,
    region: str,
    poll_timeout: int,
    poll_interval: float,
) -> str | None:
    """轮询直到完成, 返回 ModelUrl."""
    t_start = time.time()
    last_status = ""
    while time.time() - t_start < poll_timeout:
        resp = describe_job(job_id, region=region)
        status = resp.get("Status", "")

        if status == "DONE":
            elapsed = int(time.time() - t_start)
            result_files = resp.get("ResultFile3Ds", [])
            if result_files:
                for f in result_files:
                    if f.get("Type", "").lower() == "glb" and f.get("Url"):
                        print(f"[Hunyuan3D] 任务完成 ({elapsed}s), 找到 GLB")
                        return f["Url"]
                for f in result_files:
                    if f.get("Url"):
                        print(f"[Hunyuan3D] 任务完成 ({elapsed}s), Type={f.get('Type')}")
                        return f["Url"]
            print(f"[Hunyuan3D] DONE 但无 ResultFile3Ds: "
                  f"{json.dumps(resp, ensure_ascii=False)[:300]}")
            return None

        if status == "FAIL":
            print(f"[Hunyuan3D] 任务失败: {json.dumps(resp, ensure_ascii=False)[:300]}")
            return None

        if status != last_status:
            elapsed = int(time.time() - t_start)
            print(f"[Hunyuan3D] [{status}] (已等{elapsed}s)...")
            last_status = status

        time.sleep(poll_interval)

    print(f"[Hunyuan3D] 轮询超时 ({poll_timeout}s)")
    return None


def download_and_save_result(model_url: str, output_glb_path: str) -> bool:
    os.makedirs(os.path.dirname(output_glb_path) or ".", exist_ok=True)
    print(f"[Hunyuan3D] 下载模型...")
    t_start = time.time()

    try:
        resp = requests.get(model_url, timeout=120)
        resp.raise_for_status()
    except Exception as e:
        print(f"[Hunyuan3D] 下载失败: {e}")
        return False

    elapsed = time.time() - t_start
    print(f"[Hunyuan3D] 下载完成 ({elapsed:.1f}s, {len(resp.content) / 1024:.0f} KB)")

    data = resp.content

    # 尝试 zip 解压
    if data[:4] == b'PK\x03\x04':
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            glb_candidates = [n for n in z.namelist() if n.lower().endswith('.glb')]
            if not glb_candidates:
                print(f"[Hunyuan3D] zip 内无 .glb 文件: {z.namelist()}")
                return False
            glb_name = min(glb_candidates, key=lambda n: z.getinfo(n).file_size)
            z.extract(glb_name, os.path.dirname(output_glb_path))
            extracted_path = os.path.join(os.path.dirname(output_glb_path), glb_name)
            if extracted_path != output_glb_path:
                if os.path.exists(output_glb_path):
                    os.remove(output_glb_path)
                os.rename(extracted_path, output_glb_path)
    else:
        # 单个文件, 直接写入
        with open(output_glb_path, 'wb') as f:
            f.write(data)

    print(f"[Hunyuan3D] GLB 已保存: {output_glb_path}")
    return True


def _crop_to_subject(rgba_path: str) -> bytes:
    img = Image.open(rgba_path)
    alpha = img.split()[-1]
    bbox = alpha.getbbox()
    if bbox is None:
        raise ValueError(f"RGBA 图片无有效主体区域: {rgba_path}")

    l, t, r, b = bbox
    w, h = r - l, b - t
    pad_w = int(w * 0.1)
    pad_h = int(h * 0.1)
    l = max(0, l - pad_w)
    t = max(0, t - pad_h)
    r = min(img.width, r + pad_w)
    b = min(img.height, b + pad_h)

    cropped = img.crop((l, t, r, b))
    cw, ch = cropped.width, cropped.height

    if min(cw, ch) < 128:
        target = max(128, max(cw, ch))
        padded = Image.new('RGBA', (max(cw, target), max(ch, target)), (0, 0, 0, 0))
        ox = (padded.width - cw) // 2
        oy = (padded.height - ch) // 2
        padded.paste(cropped, (ox, oy))
        print(f"[Hunyuan3D] 透明 padding: {cw}x{ch} → {padded.width}x{padded.height} (物体未缩放)")
        out_buf = io.BytesIO()
        padded.save(out_buf, format='PNG')
        return out_buf.getvalue()

    print(f"[Hunyuan3D] 裁剪: 原={img.width}x{img.height} → {cw}x{ch}")
    out_buf = io.BytesIO()
    cropped.save(out_buf, format='PNG')
    return out_buf.getvalue()


def generate_mesh(
    rgba_path: str,
    output_glb_path: str,
    *,
    enable_pbr: bool = DEFAULT_ENABLE_PBR,
    face_count: int = DEFAULT_FACE_COUNT,
    region: str = DEFAULT_REGION,
    poll_timeout: int = DEFAULT_POLL_TIMEOUT,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
) -> bool:
    """从 RGBA 抠图生成 mesh (完整流程)."""
    if not os.path.exists(rgba_path):
        print(f"[Hunyuan3D] 错误: RGBA 文件不存在: {rgba_path}")
        return False

    # 裁剪 RGBA 到主体区域 + 编码
    try:
        png_data = _crop_to_subject(rgba_path)
    except ValueError as e:
        print(f"[Hunyuan3D] 错误: {e}")
        return False
    image_b64 = base64.b64encode(png_data).decode('utf-8')
    print(f"[Hunyuan3D] 输入: {rgba_path} ({len(png_data) / 1024:.1f} KB)")

    # 1. 提交任务
    job_id = submit_job(
        image_b64,
        enable_pbr=enable_pbr,
        face_count=face_count,
        region=region,
    )
    if not job_id:
        return False

    # 2. 轮询等待
    model_url = wait_for_job(
        job_id,
        region=region,
        poll_timeout=poll_timeout,
        poll_interval=poll_interval,
    )
    if not model_url:
        return False

    # 3. 下载 & 提取 GLB
    return download_and_save_result(model_url, output_glb_path)


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(
        description="Hunyuan3D Bridge v2: 腾讯云 Hunyuan3D 3.0 API"
    )
    parser.add_argument("--rgba_path", type=str, required=True)
    parser.add_argument("--output_mesh", type=str, required=True)
    parser.add_argument("--face_count", type=int, default=DEFAULT_FACE_COUNT)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--poll_timeout", type=int, default=DEFAULT_POLL_TIMEOUT)
    parser.add_argument("--poll_interval", type=float, default=DEFAULT_POLL_INTERVAL)
    parser.add_argument("--no_pbr", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.rgba_path):
        print(f"错误: RGBA 文件不存在: {args.rgba_path}")
        sys.exit(1)

    if not 3000 <= args.face_count <= 1_500_000:
        parser.error("--face_count 必须在 3000 到 1500000 之间")
    if args.poll_timeout <= 0 or args.poll_interval <= 0:
        parser.error("轮询超时和间隔必须为正数")

    try:
        ok = generate_mesh(
            args.rgba_path,
            args.output_mesh,
            enable_pbr=not args.no_pbr,
            face_count=args.face_count,
            region=args.region,
            poll_timeout=args.poll_timeout,
            poll_interval=args.poll_interval,
        )
    except RuntimeError as exc:
        print(f"[Hunyuan3D] 错误: {exc}")
        sys.exit(2)

    if ok and os.path.exists(args.output_mesh):
        print(f"OK output={args.output_mesh} "
              f"size={os.path.getsize(args.output_mesh) / 1024:.0f}KB")
        sys.exit(0)
    else:
        print("FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
