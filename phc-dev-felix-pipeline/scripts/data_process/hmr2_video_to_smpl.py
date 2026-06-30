"""
4D-Humans HMR2 video → SMPL NPZ converter.
Drop-in replacement for GVHMR: takes a monocular video and outputs SMPL .npz files
compatible with convert_zitai_to_phc.py.

Usage:
    python scripts/data_process/hmr2_video_to_smpl.py \
        --video input.mp4 \
        --out_root output/smpl_npz \
        --fps 30

Dependencies: 4D-Humans + detectron2 (ViTDet)
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
FOURDHUMANS_ROOT = REPO_ROOT.parent / "4D-Humans-main"
if str(FOURDHUMANS_ROOT) not in sys.path:
    sys.path.insert(0, str(FOURDHUMANS_ROOT))

from hmr2.configs import CACHE_DIR_4DHUMANS, get_config
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset
from hmr2.utils.renderer import cam_crop_to_full

# Rotation conversion
from pytorch3d.transforms import matrix_to_axis_angle


def parse_args():
    parser = argparse.ArgumentParser(description="4D-Humans HMR2 video → SMPL NPZ")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--out_root", type=str, default="output/hmr2_smpl", help="Output root directory")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (resample)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="HMR2 checkpoint path")
    parser.add_argument("--detector", type=str, default="vitdet", choices=["vitdet", "regnety"])
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for HMR2 inference")
    parser.add_argument("--gender", type=str, default="neutral", choices=["neutral", "male", "female"])
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def get_detector(detector_type: str):
    """Lazy-load detectron2 ViTDet predictor."""
    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy

    if detector_type == "vitdet":
        from detectron2.config import LazyConfig
        import hmr2
        cfg_path = Path(hmr2.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = (
            "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
            "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        )
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        return DefaultPredictor_Lazy(detectron2_cfg)
    elif detector_type == "regnety":
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config(
            "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", trained=True
        )
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        return DefaultPredictor_Lazy(detectron2_cfg)
    else:
        raise ValueError(f"Unknown detector: {detector_type}")


def extract_frames(video_path: Path, target_fps: int) -> tuple:
    """Extract frames from video at target FPS.

    Returns:
        (frames_list, src_fps, frame_indices)
    """
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if src_fps <= 0:
        src_fps = 30.0
    if total_frames <= 0:
        total_frames = 1

    stride = max(1, int(round(src_fps / target_fps)))
    frames = []
    indices = []

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i % stride == 0:
            frames.append(frame)
            indices.append(i)

    cap.release()
    actual_fps = src_fps / stride
    print(f"Extracted {len(frames)} frames at {actual_fps:.1f} FPS (src={src_fps:.1f}, stride={stride})")
    return frames, actual_fps, indices


def rotmat_to_poses(global_orient_rotmat, body_pose_rotmat):
    """Convert HMR2 rotation matrix output → SMPL axis-angle poses (N, 72).

    Args:
        global_orient_rotmat: (N, 1, 3, 3) or (N, 3, 3)
        body_pose_rotmat: (N, 23, 3, 3)
    Returns:
        poses: (N, 72)  axis-angle for 24 joints
    """
    N = body_pose_rotmat.shape[0]
    if global_orient_rotmat.ndim == 4:
        global_orient_rotmat = global_orient_rotmat.squeeze(1)  # (N, 3, 3)

    # Convert to axis-angle
    go_aa = matrix_to_axis_angle(global_orient_rotmat)  # (N, 3)
    bp_aa = matrix_to_axis_angle(body_pose_rotmat.reshape(-1, 3, 3)).reshape(N, -1, 3)  # (N, 23, 3)

    # Merge: global_orient (joint 0) + body_pose (joints 1-23)
    poses = torch.cat([go_aa.unsqueeze(1), bp_aa], dim=1)  # (N, 24, 3)
    return poses.reshape(N, 72)


def process_video(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"Using device: {device}")

    # Download models & load HMR2
    download_models(CACHE_DIR_4DHUMANS)
    model, model_cfg = load_hmr2(args.checkpoint)
    model = model.to(device).eval()

    # Load ViTDet
    detector = get_detector(args.detector)

    # Extract frames
    video_path = Path(args.video)
    frames, fps, indices = extract_frames(video_path, args.fps)
    if not frames:
        raise RuntimeError("No frames extracted from video")

    # Process each frame
    all_poses = []       # (N, 72)
    all_trans = []       # (N, 3)
    all_betas = []       # (N, 10)
    all_joints3d = []    # (N, J, 3)

    img_size = None

    for frame_idx, frame in enumerate(tqdm(frames, desc="HMR2 inference")):
        # Detect person
        det_out = detector(frame)
        det_instances = det_out["instances"]
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

        if len(boxes) == 0:
            print(f"  [WARN] Frame {frame_idx}: no person detected, skipping")
            continue

        # Take the largest detection (main person)
        if len(boxes) > 1:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            boxes = boxes[[areas.argmax()]]

        # Run HMR2
        dataset = ViTDetDataset(model_cfg, frame, boxes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            # Get camera-space translation (root position)
            pred_cam = out["pred_cam"]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size_batch = batch["img_size"].float()
            scaled_focal = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size_batch.max()
            cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size_batch, scaled_focal)

            # Convert rotation matrices → axis-angle poses
            pred_smpl = out["pred_smpl_params"]
            global_orient = pred_smpl["global_orient"].reshape(1, 1, 3, 3)  # (1, 1, 3, 3)
            body_pose = pred_smpl["body_pose"].reshape(1, -1, 3, 3)        # (1, 23, 3, 3)
            betas = pred_smpl["betas"].reshape(1, -1)                       # (1, 10)

            poses_aa = rotmat_to_poses(global_orient, body_pose)  # (1, 72)
            transl = cam_t_full.detach().cpu()                     # (1, 3)

            all_poses.append(poses_aa.cpu().numpy())
            all_trans.append(transl.numpy())
            all_betas.append(betas.cpu().numpy())

            # 3D keypoints for debugging
            pred_j3d = out.get("pred_keypoints_3d")
            if pred_j3d is not None:
                all_joints3d.append(pred_j3d.reshape(1, -1, 3).cpu().numpy())

    if not all_poses:
        raise RuntimeError("No valid frames with person detection")

    # Concatenate
    poses = np.concatenate(all_poses, axis=0).astype(np.float32)  # (N, 72)
    trans = np.concatenate(all_trans, axis=0).astype(np.float32)  # (N, 3)
    betas = np.concatenate(all_betas, axis=0).astype(np.float32)  # (N, 10)

    # Average betas across frames
    betas_mean = betas.mean(axis=0)  # (10,)

    # Pad betas from 10 to 16 (SMPL-X convention)
    betas_16 = np.pad(betas_mean, (0, 6)).astype(np.float32)

    # Build output NPZ
    out_root = Path(args.out_root)
    video_stem = video_path.stem
    clip_dir = out_root / video_stem
    clip_dir.mkdir(parents=True, exist_ok=True)

    npz_path = clip_dir / f"{video_stem}.npz"
    np.savez_compressed(
        npz_path,
        poses=poses,
        root_orient=poses[:, :3].astype(np.float32),
        pose_body=poses[:, 3:66].astype(np.float32),  # joints 1-21 × 3 = 63
        trans=trans,
        betas=betas_16,
        gender=np.array(args.gender),
        mocap_frame_rate=np.array(fps, dtype=np.float32),
        source="4D-Humans-HMR2",
    )
    print(f"Saved SMPL NPZ: {npz_path}")
    print(f"  Frames: {poses.shape[0]}, FPS: {fps:.1f}")
    print(f"  Pose shape: {poses.shape}, Trans shape: {trans.shape}")

    # Also save individual fields as .npy for convert_zitai_to_phc compatibility
    np.save(clip_dir / "poses.npy", poses)
    np.save(clip_dir / "root_orient.npy", poses[:, :3])
    np.save(clip_dir / "pose_body.npy", poses[:, 3:66])
    np.save(clip_dir / "trans.npy", trans)
    np.save(clip_dir / "betas.npy", betas_16)
    np.save(clip_dir / "gender.npy", np.array(args.gender))
    np.save(clip_dir / "mocap_frame_rate.npy", np.array(fps, dtype=np.float32))
    print(f"Also saved individual .npy files to {clip_dir}")

    return npz_path


if __name__ == "__main__":
    args = parse_args()
    process_video(args)
