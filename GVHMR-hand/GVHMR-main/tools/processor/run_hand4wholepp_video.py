#!/usr/bin/env python
"""Run Hand4Whole++ on a video and write GVHMR-compatible MANO tracks.

This script keeps Hand4Whole++ isolated from the main GVHMR process.  The
output is the same ``mano_params.pt`` structure that the existing HaMeR/WiLoR
stage writes, so downstream filters, ``convert_to_npz.py``, and GMR can stay
unchanged.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import os.path as osp
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation
from torch.nn.parallel.data_parallel import DataParallel
from tqdm import tqdm

from hand_bbox_tracking import TrackerConfig, summarize_bbox_track, track_bbox_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", required=True, help="Hand4Whole++ checkout root")
    parser.add_argument("--snapshot", default="", help="Path to snapshot_6.pth")
    parser.add_argument("--bbox_pt", default="", help="Optional GVHMR bbx.pt with bbx_xyxy")
    parser.add_argument("--vitpose_wholebody", default="", help="GVHMR whole-body keypoints used to score 2D hand reprojection")
    parser.add_argument("--batch_size", type=int, default=1, help="GPU inference batch size; automatically reduced after CUDA OOM")
    parser.add_argument("--yolo_model", default="yolo11n.pt", help="Fallback person detector when --bbox_pt is absent")
    parser.add_argument(
        "--joint_source",
        choices=["direct_mano", "fused_smplx"],
        default="direct_mano",
        help=(
            "3D hand joints exported downstream. direct_mano keeps WiLoR pose "
            "and joints from the same MANO prediction; fused_smplx preserves "
            "the legacy whole-body-fused joints."
        ),
    )
    parser.add_argument("--reproj_conf_thr", type=float, default=0.2)
    parser.add_argument("--reproj_min_keypoints", type=int, default=3)
    parser.add_argument(
        "--hand_crop_tracking",
        choices=["off", "flow_kalman"],
        default="off",
        help=(
            "Optional offline DWPose hand-crop tracking. flow_kalman runs one "
            "DWPose prepass, then injects the tracked crop consistently into "
            "HandRoI, WiLoR, and HandControlNet. Default off preserves legacy inference."
        ),
    )
    parser.add_argument(
        "--hand_crop_tracking_max_gap",
        type=int,
        default=8,
        help="Maximum detector-missing frames that optical flow may propagate.",
    )
    parser.add_argument(
        "--hand_crop_tracking_max_prediction_gap",
        type=int,
        default=2,
        help="Maximum consecutive no-flow prediction frames inside a tracking gap.",
    )
    parser.add_argument(
        "--hand_crop_tracking_min_box_size",
        type=float,
        default=8.0,
        help="Original-image minimum hand bbox side used to reject DWPose dummy boxes.",
    )
    parser.add_argument(
        "--hand_crop_tracking_direct_observation_quality",
        type=float,
        default=0.75,
        help=(
            "DWPose hand quality at which the raw observation is passed through "
            "exactly while the tracker state is retained only for short gaps."
        ),
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")


def check_hand4wholepp_assets(root: Path, snapshot: Path) -> None:
    required_files = [
        (snapshot, "Hand4Whole++ snapshot"),
        (root / "common" / "nets" / "mmpose" / "dw-ll_ucoco.pth", "DWPose checkpoint"),
        (
            root
            / "common"
            / "nets"
            / "mmpose"
            / "configs"
            / "wholebody_2d_keypoint"
            / "rtmpose"
            / "ubody"
            / "rtmpose-l_8xb64-270e_coco-ubody-wholebody-256x192.py",
            "DWPose mmpose config",
        ),
        (root / "common" / "nets" / "WiLoR" / "pretrained_models" / "wilor_final.ckpt", "WiLoR checkpoint"),
        (root / "common" / "nets" / "WiLoR" / "pretrained_models" / "detector.pt", "WiLoR detector checkpoint"),
        (root / "common" / "nets" / "WiLoR" / "pretrained_models" / "model_config.yaml", "WiLoR config"),
        (root / "common" / "nets" / "WiLoR" / "mano_data" / "MANO_RIGHT.pkl", "WiLoR MANO_RIGHT.pkl"),
        (root / "common" / "utils" / "human_model_files" / "smpl" / "SMPL_NEUTRAL.pkl", "SMPL neutral model"),
        (root / "common" / "utils" / "human_model_files" / "smplx" / "SMPLX_NEUTRAL.pkl", "SMPL-X neutral model"),
        (root / "common" / "utils" / "human_model_files" / "smplx" / "SMPLX_MALE.pkl", "SMPL-X male model"),
        (root / "common" / "utils" / "human_model_files" / "smplx" / "SMPLX_FEMALE.pkl", "SMPL-X female model"),
        (root / "common" / "utils" / "human_model_files" / "smplx" / "SMPLX_NEUTRAL.npz", "SMPL-X neutral npz model"),
        (root / "common" / "utils" / "human_model_files" / "smplx" / "SMPLX_MALE.npz", "SMPL-X male npz model"),
        (root / "common" / "utils" / "human_model_files" / "smplx" / "SMPLX_FEMALE.npz", "SMPL-X female npz model"),
        (
            root / "common" / "utils" / "human_model_files" / "smplx" / "MANO_SMPLX_vertex_ids.pkl",
            "MANO/SMPL-X vertex ids",
        ),
        (root / "common" / "utils" / "human_model_files" / "smplx" / "SMPLX_to_J14.pkl", "SMPL-X J14 regressor"),
        (root / "common" / "utils" / "human_model_files" / "mano" / "MANO_LEFT.pkl", "MANO left model"),
        (root / "common" / "utils" / "human_model_files" / "mano" / "MANO_RIGHT.pkl", "MANO right model"),
    ]
    required_dirs = [
        (root / "main", "Hand4Whole++ main directory"),
        (root / "common" / "nets" / "WiLoR" / "wilor", "WiLoR source"),
        (root / "common" / "nets" / "mmpose" / "mmpose", "mmpose source"),
    ]

    missing = []
    for path, label in required_files:
        if not path.is_file():
            missing.append(f"{label}: {path}")
    for path, label in required_dirs:
        if not path.is_dir():
            missing.append(f"{label}: {path}")
    if missing:
        raise FileNotFoundError(
            "Hand4Whole++ backend is not ready. Missing:\n  "
            + "\n  ".join(missing)
            + "\nRun: bash scripts/setup_hand4wholepp_assets.sh, then place the official snapshot and mmpose/DWPose assets."
        )


def read_video_frames(video_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return frames


class LazyVideoReader:
    """Read video frames on demand instead of preloading all at once.

    Saves ~frame_count * width * height * 3 bytes of RAM, which for a typical
    1250-frame 1280×960 video amounts to ~4.4 GB.
    """

    def __init__(self, video_path: Path):
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.frame_count <= 0:
            raise RuntimeError(f"No frames in {video_path}")
        self._cache_idx = -1
        self._cache_frame = None

    def __len__(self):
        return self.frame_count

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx == self._cache_idx:
            return self._cache_frame
        if idx != self._cache_idx + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = self.cap.read()
        if not ok:
            raise IndexError(f"Failed to read frame {idx}")
        self._cache_idx = idx
        self._cache_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self._cache_frame

    def close(self):
        self.cap.release()

    def __del__(self):
        self.close()


def load_bbox_tracks(path: str | Path, frame_count: int) -> np.ndarray | None:
    if not path:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    data = torch.load(path, map_location="cpu", weights_only=False)
    if "bbx_xyxy" not in data:
        return None
    bbox = data["bbx_xyxy"].detach().cpu().float().numpy()
    if bbox.ndim != 3 or bbox.shape[-1] != 4:
        raise ValueError(f"Expected bbx_xyxy shape (P,F,4), got {bbox.shape}")
    if bbox.shape[1] < frame_count:
        pad = np.repeat(bbox[:, -1:, :], frame_count - bbox.shape[1], axis=1)
        bbox = np.concatenate([bbox, pad], axis=1)
    return bbox[:, :frame_count]


def load_vitpose_tracks(path: str | Path, frame_count: int) -> np.ndarray | None:
    if not path:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    data = torch.load(path, map_location="cpu", weights_only=True)
    keypoints = data.detach().cpu().float().numpy() if torch.is_tensor(data) else np.asarray(data, dtype=np.float32)
    if keypoints.ndim != 4 or keypoints.shape[-2] < 133 or keypoints.shape[-1] < 3:
        raise ValueError(f"Expected vitpose_wholebody shape (P,F,133,3), got {keypoints.shape}")
    if keypoints.shape[1] < frame_count:
        pad = np.repeat(keypoints[:, -1:, :, :], frame_count - keypoints.shape[1], axis=1)
        keypoints = np.concatenate([keypoints, pad], axis=1)
    return keypoints[:, :frame_count, :133, :3]


def xyxy_to_xywh(box_xyxy: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    box = np.asarray(box_xyxy, dtype=np.float32).reshape(4)
    if (not np.isfinite(box).all()) or box[2] <= box[0] + 1 or box[3] <= box[1] + 1:
        box = np.asarray([0, 0, img_width - 1, img_height - 1], dtype=np.float32)
    box[0::2] = np.clip(box[0::2], 0, img_width - 1)
    box[1::2] = np.clip(box[1::2], 0, img_height - 1)
    return np.asarray([box[0], box[1], box[2] - box[0], box[3] - box[1]], dtype=np.float32)


def detect_one_person(detector, frame_rgb: np.ndarray) -> np.ndarray:
    results = detector(frame_rgb[:, :, ::-1], verbose=False)
    boxes = results[0].boxes
    best_box = None
    best_conf = -1.0
    for box in boxes:
        if int(box.cls) != 0:
            continue
        conf = float(box.conf)
        if conf > best_conf:
            best_conf = conf
            best_box = box.xyxy[0].detach().cpu().numpy().astype(np.float32)
    if best_box is None:
        h, w = frame_rgb.shape[:2]
        return np.asarray([0, 0, w - 1, h - 1], dtype=np.float32)
    return best_box


def map_box_to_original(box_crop: np.ndarray, bb2img_trans: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    box = np.asarray(box_crop, dtype=np.float32).reshape(4)
    corners = np.asarray(
        [[box[0], box[1], 1.0], [box[2], box[1], 1.0], [box[2], box[3], 1.0], [box[0], box[3], 1.0]],
        dtype=np.float32,
    )
    mapped = (bb2img_trans @ corners.T).T
    x1, y1 = np.min(mapped[:, 0]), np.min(mapped[:, 1])
    x2, y2 = np.max(mapped[:, 0]), np.max(mapped[:, 1])
    out = np.asarray([x1, y1, x2, y2], dtype=np.float32)
    out[0::2] = np.clip(out[0::2], 0, img_width - 1)
    out[1::2] = np.clip(out[1::2], 0, img_height - 1)
    return out


def map_box_from_original(
    box_original: np.ndarray,
    img2bb_trans: np.ndarray,
    cfg,
) -> np.ndarray | None:
    """Map an original-video xyxy box back to H4W++ body-image coordinates."""

    box = np.asarray(box_original, dtype=np.float32).reshape(4)
    if (not np.isfinite(box).all()) or box[2] <= box[0] + 1 or box[3] <= box[1] + 1:
        return None
    corners = np.asarray(
        [[box[0], box[1], 1.0], [box[2], box[1], 1.0], [box[2], box[3], 1.0], [box[0], box[3], 1.0]],
        dtype=np.float32,
    )
    mapped = (np.asarray(img2bb_trans, dtype=np.float32) @ corners.T).T
    x1, y1 = np.min(mapped[:, 0]), np.min(mapped[:, 1])
    x2, y2 = np.max(mapped[:, 0]), np.max(mapped[:, 1])
    scale_x = float(cfg.input_body_shape[1]) / float(cfg.input_img_shape[1])
    scale_y = float(cfg.input_body_shape[0]) / float(cfg.input_img_shape[0])
    out = np.asarray([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y], dtype=np.float32)
    out[0::2] = np.clip(out[0::2], 0, float(cfg.input_body_shape[1]))
    out[1::2] = np.clip(out[1::2], 0, float(cfg.input_body_shape[0]))
    if out[2] <= out[0] + 1 or out[3] <= out[1] + 1:
        return None
    return out


def extract_hand_joints(kpt_cam: np.ndarray, smpl_x, mano, prefix: str) -> np.ndarray:
    joints = []
    for name in mano.kpt["name"]:
        smplx_name = f"{prefix}_{name}"
        if smplx_name in smpl_x.kpt["name"]:
            joints.append(kpt_cam[smpl_x.kpt["name"].index(smplx_name)])
        else:
            joints.append(np.zeros(3, dtype=np.float32))
    return np.asarray(joints, dtype=np.float32)


def extract_hand_keypoints(kpt_2d: np.ndarray, smpl_x, mano, prefix: str) -> np.ndarray:
    keypoints = []
    for name in mano.kpt["name"]:
        smplx_name = f"{prefix}_{name}"
        if smplx_name in smpl_x.kpt["name"]:
            keypoints.append(kpt_2d[smpl_x.kpt["name"].index(smplx_name)])
        else:
            keypoints.append(np.zeros(2, dtype=np.float32))
    return np.asarray(keypoints, dtype=np.float32)


def map_points_to_original(
    points_vit: np.ndarray,
    bb2img_trans: np.ndarray,
    cfg,
    img_width: int,
    img_height: int,
) -> np.ndarray:
    points = np.asarray(points_vit, dtype=np.float32).reshape(-1, 2).copy()
    points[:, 0] *= float(cfg.input_img_shape[1]) / float(cfg.vit_output_shape[1])
    points[:, 1] *= float(cfg.input_img_shape[0]) / float(cfg.vit_output_shape[0])
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    mapped = (np.asarray(bb2img_trans, dtype=np.float32) @ homogeneous.T).T
    mapped[:, 0] = np.clip(mapped[:, 0], 0, img_width - 1)
    mapped[:, 1] = np.clip(mapped[:, 1], 0, img_height - 1)
    return mapped.astype(np.float32)


def hand_reprojection_error(
    predicted: np.ndarray,
    target: np.ndarray | None,
    valid: bool,
    conf_thr: float,
    min_keypoints: int,
) -> float:
    if not valid or target is None:
        return 1e6
    predicted = np.asarray(predicted, dtype=np.float32).reshape(-1, 2)
    target = np.asarray(target, dtype=np.float32).reshape(-1, 3)
    count = min(predicted.shape[0], target.shape[0])
    predicted, target = predicted[:count], target[:count]
    usable = (
        np.isfinite(predicted).all(axis=1)
        & np.isfinite(target[:, :2]).all(axis=1)
        & np.isfinite(target[:, 2])
        & (target[:, 2] > float(conf_thr))
    )
    if int(usable.sum()) < int(min_keypoints):
        return 1e6
    weights = np.clip(target[usable, 2], 0.0, 1.0)
    distances = np.linalg.norm(predicted[usable] - target[usable, :2], axis=1)
    return float(np.sum(distances * weights) / np.clip(np.sum(weights), 1e-6, None))


def to_numpy_at(out: dict, key: str, batch_idx: int, fallback_shape: tuple[int, ...]) -> np.ndarray:
    if key not in out:
        return np.zeros(fallback_shape, dtype=np.float32)
    value = out[key]
    if torch.is_tensor(value):
        value = value.detach().cpu().float().numpy()
    value = np.asarray(value, dtype=np.float32)
    if value.ndim > 0:
        value = value[batch_idx]
    return value.reshape(fallback_shape)


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    snapshot = Path(args.snapshot).expanduser().resolve() if args.snapshot else root / "demo" / "snapshot_6.pth"
    output = Path(args.output).expanduser().resolve()

    check_hand4wholepp_assets(root, snapshot)

    main_dir = root / "main"
    sys.path.insert(0, str(main_dir))
    cwd = os.getcwd()
    os.chdir(str(root / "demo"))
    try:
        from config import cfg
        from model import get_model
        from utils.mano import mano
        from utils.preprocessing import get_patch_img, set_aspect_ratio
        from utils.smpl_x import smpl_x

        model = get_model("test")
        model = DataParallel(model).cuda()
        # snapshot_6.pth is about 3 GiB because it also contains optimizer
        # state.  Only the 1.8 GiB network state is needed for inference.
        # mmap keeps those tensors file-backed and avoids retaining another
        # full checkpoint copy beside the CUDA model.
        print(f"[Hand4Whole++] memory-mapped checkpoint: {snapshot}", flush=True)
        ckpt = torch.load(
            snapshot,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        incompatible = model.load_state_dict(ckpt["network"], strict=False)
        external_prefixes = (
            "module.wilor_det.",
            "module.wilor.",
            "module.dwpose.",
        )
        # SMPL-X geometry is constructed from the checked local model assets,
        # rather than stored in snapshot_6.pth.  These are static layer
        # buffers (vertices, regressors, blend-shapes), not missing learned
        # Hand4Whole++ weights.
        runtime_asset_prefixes = ("module.smplx_layer.",)
        external_missing = [
            key
            for key in incompatible.missing_keys
            if key.startswith(external_prefixes)
        ]
        non_external_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(external_prefixes)
        ]
        runtime_asset_missing = [
            key
            for key in non_external_missing
            if key.startswith(runtime_asset_prefixes)
        ]
        core_missing = [
            key
            for key in non_external_missing
            if not key.startswith(runtime_asset_prefixes)
        ]
        if incompatible.unexpected_keys or core_missing:
            verdict = "incompatible_core"
        elif external_missing or runtime_asset_missing:
            verdict = "compatible_runtime_assets"
        else:
            verdict = "compatible_clean"
        external_assets = {}
        for name, path in {
            "wilor": root / "common" / "nets" / "WiLoR" / "pretrained_models" / "wilor_final.ckpt",
            "detector": root / "common" / "nets" / "WiLoR" / "pretrained_models" / "detector.pt",
            "dwpose": root / "common" / "nets" / "mmpose" / "dw-ll_ucoco.pth",
            "smplx_neutral": root / "common" / "utils" / "human_model_files" / "smplx" / "SMPLX_NEUTRAL.pkl",
        }.items():
            external_assets[name] = {
                "path": str(path),
                "exists": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else None,
            }
        prov = {
            "schema_version": 3,
            "checkpoint": str(snapshot),
            "checkpoint_bytes": snapshot.stat().st_size,
            "missing_key_count": len(incompatible.missing_keys),
            "external_module_missing_key_count": len(external_missing),
            "non_external_missing_key_count": len(non_external_missing),
            "runtime_asset_missing_key_count": len(runtime_asset_missing),
            "core_missing_key_count": len(core_missing),
            "unexpected_key_count": len(incompatible.unexpected_keys),
            "missing_keys_sample": incompatible.missing_keys[:20],
            "non_external_missing_keys_sample": non_external_missing[:20],
            "runtime_asset_missing_keys_sample": runtime_asset_missing[:20],
            "core_missing_keys_sample": core_missing[:20],
            "unexpected_keys_sample": incompatible.unexpected_keys[:20],
            "external_assets": external_assets,
            "active_bbox_source": "crop_override_or_dwpose",
            "verdict": verdict,
        }
        prov_path = Path(args.output).expanduser().resolve().parent / "hand4wholepp_checkpoint_provenance.json"
        prov_path.parent.mkdir(parents=True, exist_ok=True)
        with open(prov_path, "w", encoding="utf-8") as fp:
            json.dump(prov, fp, indent=2)

        print(
            "[Hand4Whole++] checkpoint compatibility: "
            f"verdict={verdict}, "
            f"missing={len(incompatible.missing_keys)}, "
            f"external_missing={len(external_missing)}, "
            f"runtime_asset_missing={len(runtime_asset_missing)}, "
            f"core_missing={len(core_missing)}, "
            f"unexpected={len(incompatible.unexpected_keys)}",
            flush=True,
        )
        del ckpt
        gc.collect()
        for module in model.module.trainable_modules + model.module.eval_modules:
            module.eval()

        frames = LazyVideoReader(Path(args.video))
        frame_count = len(frames)
        bbox_tracks = load_bbox_tracks(args.bbox_pt, frame_count)
        vitpose_tracks = load_vitpose_tracks(args.vitpose_wholebody, frame_count)
        if args.vitpose_wholebody and vitpose_tracks is None:
            raise FileNotFoundError(
                f"vitpose_wholebody not found: {args.vitpose_wholebody}"
            )
        person_count = int(bbox_tracks.shape[0]) if bbox_tracks is not None else 1
        if vitpose_tracks is not None and vitpose_tracks.shape[0] < person_count:
            raise ValueError(
                f"VitPose has {vitpose_tracks.shape[0]} people but bbox tracks have {person_count}"
            )
        detector = None
        if bbox_tracks is None:
            from ultralytics import YOLO

            detector = YOLO(args.yolo_model)

        transform = transforms.ToTensor()
        total_samples = frame_count * person_count
        requested_batch_size = max(1, int(args.batch_size))
        bbox_scale = np.asarray(
            [
                cfg.input_img_shape[1] / cfg.input_body_shape[1],
                cfg.input_img_shape[0] / cfg.input_body_shape[0],
                cfg.input_img_shape[1] / cfg.input_body_shape[1],
                cfg.input_img_shape[0] / cfg.input_body_shape[0],
            ],
            dtype=np.float32,
        )
        fallback_body_box = np.asarray(
            [0.0, 0.0, float(cfg.input_body_shape[1]), float(cfg.input_body_shape[0])],
            dtype=np.float32,
        )
        person_bbox_cache = np.full(
            (person_count, frame_count, 4), np.nan, dtype=np.float32
        )

        def build_sample(frame_idx: int, person_idx: int) -> dict:
            frame = frames[frame_idx]
            img_height, img_width = frame.shape[:2]
            if bbox_tracks is not None:
                bbox_xyxy = bbox_tracks[person_idx, frame_idx]
            elif np.isfinite(person_bbox_cache[person_idx, frame_idx]).all():
                bbox_xyxy = person_bbox_cache[person_idx, frame_idx]
            else:
                bbox_xyxy = detect_one_person(detector, frame)
                person_bbox_cache[person_idx, frame_idx] = bbox_xyxy
            bbox_xywh = xyxy_to_xywh(bbox_xyxy, img_width, img_height)
            bbox_xywh = set_aspect_ratio(
                bbox_xywh,
                cfg.input_img_shape[1] / cfg.input_img_shape[0],
            )
            img, img2bb_trans, bb2img_trans = get_patch_img(
                frame,
                bbox_xywh,
                1.0,
                0.0,
                False,
                cfg.input_img_shape,
            )
            return {
                "frame_idx": frame_idx,
                "person_idx": person_idx,
                "img_height": img_height,
                "img_width": img_width,
                "img2bb_trans": img2bb_trans,
                "bb2img_trans": bb2img_trans,
                "img_tensor": transform(img.astype(np.float32)) / 255.0,
            }

        crop_tracking = None
        if args.hand_crop_tracking == "flow_kalman":
            raw_tracks = {
                side: {
                    "xyxy": np.zeros((person_count, frame_count, 4), dtype=np.float32),
                    "observed": np.zeros((person_count, frame_count), dtype=bool),
                    "quality": np.zeros((person_count, frame_count), dtype=np.float32),
                }
                for side in ("left", "right")
            }
            img2bb_tracks = np.zeros(
                (person_count, frame_count, 2, 3), dtype=np.float32
            )
            prepass_idx = 0
            prepass_batch_size = requested_batch_size
            right_hand_indices = list(smpl_x.kpt["part_idx"]["rhand"]) + [
                smpl_x.kpt["name"].index("R_Wrist")
            ]
            left_hand_indices = list(smpl_x.kpt["part_idx"]["lhand"]) + [
                smpl_x.kpt["name"].index("L_Wrist")
            ]

            def hand_keypoint_quality(keypoints: torch.Tensor, indices: list[int]) -> torch.Tensor:
                scores = keypoints[:, indices, 2].clamp(0.0, 1.0)
                visible = scores > 0.3
                visible_count = visible.float().sum(dim=1)
                mean_visible = (scores * visible.float()).sum(dim=1) / visible_count.clamp_min(1.0)
                return mean_visible * (visible_count / float(len(indices)))

            prepass = tqdm(
                total=total_samples,
                desc=f"hand4wholepp crop prepass bs={prepass_batch_size}",
            )
            with torch.no_grad():
                while prepass_idx < total_samples:
                    current_batch_size = min(
                        prepass_batch_size, total_samples - prepass_idx
                    )
                    samples = []
                    for offset in range(current_batch_size):
                        sample_idx = prepass_idx + offset
                        samples.append(
                            build_sample(
                                sample_idx // person_count, sample_idx % person_count
                            )
                        )
                    images = torch.stack(
                        [sample["img_tensor"] for sample in samples]
                    ).cuda()
                    try:
                        body_img = F.interpolate(
                            images, cfg.input_body_shape, mode="bilinear"
                        )
                        dwpose_kpt = model.module.dwpose(body_img)
                        (
                            rhand_bbox,
                            lhand_bbox,
                            rhand_exist,
                            lhand_exist,
                        ) = model.module.dwpose.get_hand_bbox(dwpose_kpt)
                    except torch.cuda.OutOfMemoryError:
                        del images
                        torch.cuda.empty_cache()
                        if current_batch_size <= 1:
                            raise
                        prepass_batch_size = max(1, current_batch_size // 2)
                        prepass.set_description(
                            f"hand4wholepp crop prepass bs={prepass_batch_size} (OOM fallback)"
                        )
                        continue

                    right_bbox_np = rhand_bbox.detach().cpu().float().numpy()
                    left_bbox_np = lhand_bbox.detach().cpu().float().numpy()
                    right_exist_np = rhand_exist.detach().cpu().float().numpy()
                    left_exist_np = lhand_exist.detach().cpu().float().numpy()
                    right_quality_np = hand_keypoint_quality(
                        dwpose_kpt, right_hand_indices
                    ).detach().cpu().float().numpy()
                    left_quality_np = hand_keypoint_quality(
                        dwpose_kpt, left_hand_indices
                    ).detach().cpu().float().numpy()
                    for batch_idx, sample in enumerate(samples):
                        frame_idx = sample["frame_idx"]
                        person_idx = sample["person_idx"]
                        img2bb_tracks[person_idx, frame_idx] = sample["img2bb_trans"]
                        raw_tracks["right"]["xyxy"][person_idx, frame_idx] = map_box_to_original(
                            right_bbox_np[batch_idx] * bbox_scale,
                            sample["bb2img_trans"],
                            sample["img_width"],
                            sample["img_height"],
                        )
                        raw_tracks["left"]["xyxy"][person_idx, frame_idx] = map_box_to_original(
                            left_bbox_np[batch_idx] * bbox_scale,
                            sample["bb2img_trans"],
                            sample["img_width"],
                            sample["img_height"],
                        )
                        raw_tracks["right"]["observed"][person_idx, frame_idx] = (
                            float(right_exist_np[batch_idx]) > 0.5
                        )
                        raw_tracks["left"]["observed"][person_idx, frame_idx] = (
                            float(left_exist_np[batch_idx]) > 0.5
                        )
                        raw_tracks["right"]["quality"][person_idx, frame_idx] = (
                            right_quality_np[batch_idx]
                        )
                        raw_tracks["left"]["quality"][person_idx, frame_idx] = (
                            left_quality_np[batch_idx]
                        )
                    prepass_idx += current_batch_size
                    prepass.update(current_batch_size)
                    del images, body_img, dwpose_kpt, rhand_bbox, lhand_bbox
                    del rhand_exist, lhand_exist
            prepass.close()

            tracker_config = TrackerConfig(
                max_gap=max(0, int(args.hand_crop_tracking_max_gap)),
                max_prediction_gap=max(
                    0, int(args.hand_crop_tracking_max_prediction_gap)
                ),
                min_box_size=max(1.0, float(args.hand_crop_tracking_min_box_size)),
                direct_observation_quality=float(
                    np.clip(args.hand_crop_tracking_direct_observation_quality, 0.0, 1.0)
                ),
            )
            crop_tracking = {
                "mode": args.hand_crop_tracking,
                "config": {
                    "max_gap": tracker_config.max_gap,
                    "max_prediction_gap": tracker_config.max_prediction_gap,
                    "min_box_size": tracker_config.min_box_size,
                    "direct_observation_quality": tracker_config.direct_observation_quality,
                },
                "per_person": [],
            }
            for side in ("left", "right"):
                raw_tracks[side]["tracked_xyxy"] = np.zeros_like(
                    raw_tracks[side]["xyxy"]
                )
                raw_tracks[side]["usable"] = np.zeros(
                    (person_count, frame_count), dtype=bool
                )
                raw_tracks[side]["source"] = np.zeros(
                    (person_count, frame_count), dtype=np.int8
                )
                raw_tracks[side]["flow_points"] = np.zeros(
                    (person_count, frame_count), dtype=np.int16
                )
                raw_tracks[side]["rejected_observation"] = np.zeros(
                    (person_count, frame_count), dtype=bool
                )
                raw_tracks[side]["direct_observation"] = np.zeros(
                    (person_count, frame_count), dtype=bool
                )
                raw_tracks[side]["body_xyxy"] = np.repeat(
                    fallback_body_box[None, None],
                    person_count * frame_count,
                    axis=0,
                ).reshape(person_count, frame_count, 4)

            for person_idx in range(person_count):
                person_summary = {}
                for side in ("left", "right"):
                    result = track_bbox_sequence(
                        frames,
                        raw_tracks[side]["xyxy"][person_idx],
                        raw_tracks[side]["observed"][person_idx],
                        tracker_config,
                        raw_tracks[side]["quality"][person_idx],
                    )
                    for key in (
                        "tracked_xyxy",
                        "usable",
                        "source",
                        "flow_points",
                        "rejected_observation",
                        "direct_observation",
                    ):
                        source_key = "xyxy" if key == "tracked_xyxy" else key
                        raw_tracks[side][key][person_idx] = result[source_key]
                    for frame_idx in range(frame_count):
                        if not raw_tracks[side]["usable"][person_idx, frame_idx]:
                            continue
                        body_box = map_box_from_original(
                            raw_tracks[side]["tracked_xyxy"][person_idx, frame_idx],
                            img2bb_tracks[person_idx, frame_idx],
                            cfg,
                        )
                        if body_box is None:
                            raw_tracks[side]["usable"][person_idx, frame_idx] = False
                            continue
                        raw_tracks[side]["body_xyxy"][person_idx, frame_idx] = body_box
                    person_summary[side] = summarize_bbox_track(
                        raw_tracks[side]["xyxy"][person_idx],
                        raw_tracks[side]["observed"][person_idx],
                        result,
                    )
                crop_tracking["per_person"].append(person_summary)
            crop_tracking["tracks"] = raw_tracks
            print(
                "[Hand4Whole++] hand crop tracking enabled: "
                f"mode={args.hand_crop_tracking}, max_gap={tracker_config.max_gap}",
                flush=True,
            )

        out_data = {
            "left_hand_global_orient": np.zeros((person_count, frame_count, 3, 3), dtype=np.float32),
            "right_hand_global_orient": np.zeros((person_count, frame_count, 3, 3), dtype=np.float32),
            "left_hand_pose": np.zeros((person_count, frame_count, 15, 3, 3), dtype=np.float32),
            "right_hand_pose": np.zeros((person_count, frame_count, 15, 3, 3), dtype=np.float32),
            # Preserve the WiLoR MANO shape prediction so a post-filter stage
            # can re-run the same MANO forward pass and keep pose/joints
            # kinematically consistent after temporal edits.
            "left_hand_betas": np.zeros((person_count, frame_count, 10), dtype=np.float32),
            "right_hand_betas": np.zeros((person_count, frame_count, 10), dtype=np.float32),
            "left_hand_joints_3d": np.zeros((person_count, frame_count, 21, 3), dtype=np.float32),
            "right_hand_joints_3d": np.zeros((person_count, frame_count, 21, 3), dtype=np.float32),
            "left_hand_valid": np.zeros((person_count, frame_count), dtype=bool),
            "right_hand_valid": np.zeros((person_count, frame_count), dtype=bool),
            "left_hand_reproj_error": np.zeros((person_count, frame_count), dtype=np.float32),
            "right_hand_reproj_error": np.zeros((person_count, frame_count), dtype=np.float32),
            "left_hand_bbox_xyxy": np.zeros((person_count, frame_count, 4), dtype=np.float32),
            "right_hand_bbox_xyxy": np.zeros((person_count, frame_count, 4), dtype=np.float32),
        }
        if crop_tracking is not None:
            for side in ("left", "right"):
                track = crop_tracking["tracks"][side]
                prefix = f"{side}_hand_bbox"
                out_data[f"{prefix}_raw_xyxy"] = track["xyxy"].copy()
                out_data[f"{prefix}_track_quality"] = track["quality"].copy()
                out_data[f"{prefix}_track_usable"] = track["usable"].copy()
                out_data[f"{prefix}_track_source"] = track["source"].copy()
                out_data[f"{prefix}_track_flow_points"] = track["flow_points"].copy()
                out_data[f"{prefix}_track_rejected_observation"] = track[
                    "rejected_observation"
                ].copy()
                out_data[f"{prefix}_track_direct_observation"] = track[
                    "direct_observation"
                ].copy()

        effective_batch_size = requested_batch_size
        linear_idx = 0
        progress = tqdm(total=total_samples, desc=f"hand4wholepp bs={effective_batch_size}")
        with torch.no_grad():
            while linear_idx < total_samples:
                current_batch_size = min(effective_batch_size, total_samples - linear_idx)
                samples = []
                for offset in range(current_batch_size):
                    sample_idx = linear_idx + offset
                    samples.append(
                        build_sample(
                            sample_idx // person_count, sample_idx % person_count
                        )
                    )

                inputs = {"img": torch.stack([sample["img_tensor"] for sample in samples]).cuda()}
                if crop_tracking is not None:
                    inputs.update(
                        {
                            "rhand_bbox_override_body_xyxy": torch.from_numpy(
                                crop_tracking["tracks"]["right"]["body_xyxy"][
                                    [sample["person_idx"] for sample in samples],
                                    [sample["frame_idx"] for sample in samples],
                                ]
                            ).cuda(),
                            "lhand_bbox_override_body_xyxy": torch.from_numpy(
                                crop_tracking["tracks"]["left"]["body_xyxy"][
                                    [sample["person_idx"] for sample in samples],
                                    [sample["frame_idx"] for sample in samples],
                                ]
                            ).cuda(),
                            "rhand_exist_override": torch.from_numpy(
                                crop_tracking["tracks"]["right"]["usable"][
                                    [sample["person_idx"] for sample in samples],
                                    [sample["frame_idx"] for sample in samples],
                                ].astype(np.float32)
                            ).cuda(),
                            "lhand_exist_override": torch.from_numpy(
                                crop_tracking["tracks"]["left"]["usable"][
                                    [sample["person_idx"] for sample in samples],
                                    [sample["frame_idx"] for sample in samples],
                                ].astype(np.float32)
                            ).cuda(),
                        }
                    )
                try:
                    out = model(inputs, {}, {}, "test")
                except torch.cuda.OutOfMemoryError:
                    del inputs
                    torch.cuda.empty_cache()
                    if current_batch_size <= 1:
                        raise
                    effective_batch_size = max(1, current_batch_size // 2)
                    progress.set_description(f"hand4wholepp bs={effective_batch_size} (OOM fallback)")
                    continue

                for batch_idx, sample in enumerate(samples):
                    frame_idx = sample["frame_idx"]
                    person_idx = sample["person_idx"]
                    img_height = sample["img_height"]
                    img_width = sample["img_width"]
                    bb2img_trans = sample["bb2img_trans"]

                    left_pose = to_numpy_at(out, "smplx_lhand_pose", batch_idx, (45,))
                    right_pose = to_numpy_at(out, "smplx_rhand_pose", batch_idx, (45,))
                    left_global = to_numpy_at(out, "lhand_root_pose", batch_idx, (3,))
                    right_global = to_numpy_at(out, "rhand_root_pose", batch_idx, (3,))
                    left_betas = to_numpy_at(out, "lhand_shape_param", batch_idx, (10,))
                    right_betas = to_numpy_at(out, "rhand_shape_param", batch_idx, (10,))
                    # Convert axis-angle to rotation matrices (compatible with HaMeR format).
                    left_pose_mat = Rotation.from_rotvec(left_pose.reshape(-1, 3)).as_matrix().reshape(15, 3, 3).astype(np.float32)
                    right_pose_mat = Rotation.from_rotvec(right_pose.reshape(-1, 3)).as_matrix().reshape(15, 3, 3).astype(np.float32)
                    left_global_mat = Rotation.from_rotvec(left_global.reshape(-1, 3)).as_matrix().reshape(3, 3).astype(np.float32)
                    right_global_mat = Rotation.from_rotvec(right_global.reshape(-1, 3)).as_matrix().reshape(3, 3).astype(np.float32)
                    smplx_kpt_cam = to_numpy_at(out, "smplx_kpt_cam", batch_idx, (smpl_x.kpt["num"], 3))
                    if args.joint_source == "direct_mano":
                        mano_regressor = np.asarray(mano.kpt["regressor"], dtype=np.float32)
                        left_vertices = to_numpy_at(
                            out,
                            "lmano_vert_cam",
                            batch_idx,
                            (mano.vertex_num, 3),
                        )
                        right_vertices = to_numpy_at(
                            out,
                            "rmano_vert_cam",
                            batch_idx,
                            (mano.vertex_num, 3),
                        )
                        left_joints = mano_regressor @ left_vertices
                        right_joints = mano_regressor @ right_vertices
                        # WiLoR vertices include the crop-camera translation.
                        # Downstream hand filters and robot retargeting need
                        # wrist-local articulation; body/GVHMR supplies the
                        # actual wrist trajectory independently.
                        left_joints = left_joints - left_joints[:1]
                        right_joints = right_joints - right_joints[:1]
                    else:
                        left_joints = extract_hand_joints(
                            smplx_kpt_cam,
                            smpl_x,
                            mano,
                            "L",
                        )
                        right_joints = extract_hand_joints(
                            smplx_kpt_cam,
                            smpl_x,
                            mano,
                            "R",
                        )
                    left_bbox = map_box_to_original(
                        to_numpy_at(out, "lhand_bbox", batch_idx, (4,)) * bbox_scale,
                        bb2img_trans,
                        img_width,
                        img_height,
                    )
                    right_bbox = map_box_to_original(
                        to_numpy_at(out, "rhand_bbox", batch_idx, (4,)) * bbox_scale,
                        bb2img_trans,
                        img_width,
                        img_height,
                    )
                    left_valid = bool(float(to_numpy_at(out, "lhand_exist", batch_idx, (1,))[0]) > 0.5)
                    right_valid = bool(float(to_numpy_at(out, "rhand_exist", batch_idx, (1,))[0]) > 0.5)

                    smplx_kpt_proj = to_numpy_at(
                        out,
                        "smplx_kpt_proj",
                        batch_idx,
                        (smpl_x.kpt["num"], 2),
                    )
                    left_projected = map_points_to_original(
                        extract_hand_keypoints(smplx_kpt_proj, smpl_x, mano, "L"),
                        bb2img_trans,
                        cfg,
                        img_width,
                        img_height,
                    )
                    right_projected = map_points_to_original(
                        extract_hand_keypoints(smplx_kpt_proj, smpl_x, mano, "R"),
                        bb2img_trans,
                        cfg,
                        img_width,
                        img_height,
                    )
                    left_target = (
                        vitpose_tracks[person_idx, frame_idx, -42:-21]
                        if vitpose_tracks is not None
                        else None
                    )
                    right_target = (
                        vitpose_tracks[person_idx, frame_idx, -21:]
                        if vitpose_tracks is not None
                        else None
                    )
                    left_error = hand_reprojection_error(
                        left_projected,
                        left_target,
                        left_valid,
                        args.reproj_conf_thr,
                        args.reproj_min_keypoints,
                    )
                    right_error = hand_reprojection_error(
                        right_projected,
                        right_target,
                        right_valid,
                        args.reproj_conf_thr,
                        args.reproj_min_keypoints,
                    )

                    out_data["left_hand_pose"][person_idx, frame_idx] = left_pose_mat
                    out_data["right_hand_pose"][person_idx, frame_idx] = right_pose_mat
                    out_data["left_hand_betas"][person_idx, frame_idx] = left_betas
                    out_data["right_hand_betas"][person_idx, frame_idx] = right_betas
                    out_data["left_hand_global_orient"][person_idx, frame_idx] = left_global_mat
                    out_data["right_hand_global_orient"][person_idx, frame_idx] = right_global_mat
                    out_data["left_hand_joints_3d"][person_idx, frame_idx] = left_joints
                    out_data["right_hand_joints_3d"][person_idx, frame_idx] = right_joints
                    out_data["left_hand_valid"][person_idx, frame_idx] = left_valid
                    out_data["right_hand_valid"][person_idx, frame_idx] = right_valid
                    out_data["left_hand_reproj_error"][person_idx, frame_idx] = left_error
                    out_data["right_hand_reproj_error"][person_idx, frame_idx] = right_error
                    out_data["left_hand_bbox_xyxy"][person_idx, frame_idx] = left_bbox
                    out_data["right_hand_bbox_xyxy"][person_idx, frame_idx] = right_bbox

                linear_idx += current_batch_size
                progress.update(current_batch_size)
                del inputs, out
        progress.close()

        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({key: torch.from_numpy(value) for key, value in out_data.items()}, output)
        if crop_tracking is not None:
            crop_tracking_report = {
                "mode": crop_tracking["mode"],
                "config": crop_tracking["config"],
                "coordinate_space": "original_video_xyxy",
                "per_person": crop_tracking["per_person"],
                "note": (
                    "BBox stability is an audit metric, not evidence of hand-pose "
                    "accuracy. Compare reprojection and crop coverage in an A/B run."
                ),
            }
            report_path = output.with_suffix(".crop_tracking.json")
            with report_path.open("w", encoding="utf-8") as f:
                json.dump(crop_tracking_report, f, indent=2, allow_nan=True)
        left_error = out_data["left_hand_reproj_error"]
        right_error = out_data["right_hand_reproj_error"]
        left_finite = left_error[np.isfinite(left_error) & (left_error < 1e5)]
        right_finite = right_error[np.isfinite(right_error) & (right_error < 1e5)]
        left_error_median = float(np.median(left_finite)) if left_finite.size else float("nan")
        right_error_median = float(np.median(right_finite)) if right_finite.size else float("nan")
        print(
            f"-> {output} ({person_count} people, {frame_count} frames, "
            f"left_valid={int(out_data['left_hand_valid'].sum())}, "
            f"right_valid={int(out_data['right_hand_valid'].sum())}, "
            f"joint_source={args.joint_source}, "
            f"reproj_median={left_error_median:.2f}/{right_error_median:.2f}px, "
            f"crop_tracking={args.hand_crop_tracking})"
        )
    finally:
        try:
            frames.close()
        except Exception:
            pass
        os.chdir(cwd)


if __name__ == "__main__":
    main()
