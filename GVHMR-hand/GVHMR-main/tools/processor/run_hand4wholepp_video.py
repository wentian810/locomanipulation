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
import os
import os.path as osp
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation
from torch.nn.parallel.data_parallel import DataParallel
from tqdm import tqdm


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
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(
                "[Hand4Whole++] checkpoint compatibility: "
                f"missing={len(incompatible.missing_keys)}, "
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

        out_data = {
            "left_hand_global_orient": np.zeros((person_count, frame_count, 3, 3), dtype=np.float32),
            "right_hand_global_orient": np.zeros((person_count, frame_count, 3, 3), dtype=np.float32),
            "left_hand_pose": np.zeros((person_count, frame_count, 15, 3, 3), dtype=np.float32),
            "right_hand_pose": np.zeros((person_count, frame_count, 15, 3, 3), dtype=np.float32),
            "left_hand_joints_3d": np.zeros((person_count, frame_count, 21, 3), dtype=np.float32),
            "right_hand_joints_3d": np.zeros((person_count, frame_count, 21, 3), dtype=np.float32),
            "left_hand_valid": np.zeros((person_count, frame_count), dtype=bool),
            "right_hand_valid": np.zeros((person_count, frame_count), dtype=bool),
            "left_hand_reproj_error": np.zeros((person_count, frame_count), dtype=np.float32),
            "right_hand_reproj_error": np.zeros((person_count, frame_count), dtype=np.float32),
            "left_hand_bbox_xyxy": np.zeros((person_count, frame_count, 4), dtype=np.float32),
            "right_hand_bbox_xyxy": np.zeros((person_count, frame_count, 4), dtype=np.float32),
        }

        transform = transforms.ToTensor()
        total_samples = frame_count * person_count
        requested_batch_size = max(1, int(args.batch_size))
        effective_batch_size = requested_batch_size
        linear_idx = 0
        progress = tqdm(total=total_samples, desc=f"hand4wholepp bs={effective_batch_size}")
        with torch.no_grad():
            while linear_idx < total_samples:
                current_batch_size = min(effective_batch_size, total_samples - linear_idx)
                samples = []
                for offset in range(current_batch_size):
                    sample_idx = linear_idx + offset
                    frame_idx = sample_idx // person_count
                    person_idx = sample_idx % person_count
                    frame = frames[frame_idx]
                    img_height, img_width = frame.shape[:2]

                    if bbox_tracks is not None:
                        bbox_xyxy = bbox_tracks[person_idx, frame_idx]
                    else:
                        bbox_xyxy = detect_one_person(detector, frame)
                    bbox_xywh = xyxy_to_xywh(bbox_xyxy, img_width, img_height)
                    bbox_xywh = set_aspect_ratio(
                        bbox_xywh,
                        cfg.input_img_shape[1] / cfg.input_img_shape[0],
                    )
                    img, _img2bb_trans, bb2img_trans = get_patch_img(
                        frame,
                        bbox_xywh,
                        1.0,
                        0.0,
                        False,
                        cfg.input_img_shape,
                    )
                    samples.append(
                        {
                            "frame_idx": frame_idx,
                            "person_idx": person_idx,
                            "img_height": img_height,
                            "img_width": img_width,
                            "bb2img_trans": bb2img_trans,
                            "img_tensor": transform(img.astype(np.float32)) / 255.0,
                        }
                    )

                inputs = {"img": torch.stack([sample["img_tensor"] for sample in samples]).cuda()}
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
                    bbox_scale = np.asarray(
                        [
                            cfg.input_img_shape[1] / cfg.input_body_shape[1],
                            cfg.input_img_shape[0] / cfg.input_body_shape[0],
                            cfg.input_img_shape[1] / cfg.input_body_shape[1],
                            cfg.input_img_shape[0] / cfg.input_body_shape[0],
                        ],
                        dtype=np.float32,
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
            f"reproj_median={left_error_median:.2f}/{right_error_median:.2f}px)"
        )
    finally:
        try:
            frames.close()
        except Exception:
            pass
        os.chdir(cwd)


if __name__ == "__main__":
    main()
