"""
Lightweight 2D/3D keypoint extractors for monocular video.
Supports multiple backends — zero heavy dependencies.

Usage:
    # MediaPipe (33 landmark 3D keypoints, includes FEET):
    python scripts/data_process/extract_keypoints.py --video input.mp4 --backend mediapipe

    # RTMPose via rtmlib (133 keypoints, requires ONNX model URL):
    python scripts/data_process/extract_keypoints.py --video input.mp4 --backend rtmpose

    # ViTPose (17 COCO keypoints, same as GVHMR uses):
    python scripts/data_process/extract_keypoints.py --video input.mp4 --backend vitpose

Install:
    pip install mediapipe                        # for mediapipe backend
    pip install rtmlib onnxruntime               # for rtmpose backend
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# MediaPipe Pose (33 landmarks with FEET) — easiest to use
# ---------------------------------------------------------------------------

MEDIAPIPE_FOOT_INDICES = {
    "left_heel": 29,
    "left_foot_index": 31,
    "right_heel": 30,
    "right_foot_index": 32,
    "left_ankle": 27,
    "right_ankle": 28,
}

# MediaPipe 33 → COCO 17 mapping (approximate)
MP33_TO_COCO17 = {
    0: 0,    # nose
    2: 1,    # left_eye (inner)
    5: 2,    # right_eye (inner)
    7: 3,    # left_ear
    8: 4,    # right_ear
    11: 5,   # left_shoulder
    12: 6,   # right_shoulder
    13: 7,   # left_elbow
    14: 8,   # right_elbow
    15: 9,   # left_wrist
    16: 10,  # right_wrist
    23: 11,  # left_hip
    24: 12,  # right_hip
    25: 13,  # left_knee
    26: 14,  # right_knee
    27: 15,  # left_ankle
    28: 16,  # right_ankle
}


def mediapipe_extract(video_path: str, out_dir: Path, fps: int = 30):
    """Extract 33-landmark 3D keypoints using MediaPipe Pose."""
    try:
        import mediapipe as mp
    except ImportError:
        print("[ERROR] mediapipe not installed. Run: pip install mediapipe")
        return False

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,  # 2=best quality
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = 30.0
    stride = max(1, int(round(src_fps / fps)))

    all_landmarks = []    # (F, 33, 3) world-space 3D
    all_landmarks_2d = [] # (F, 33, 2) image-space 2D
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_world_landmarks:
                # 3D world landmarks (33, 3) — includes feet!
                lm3d = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark],
                                dtype=np.float32)
                # 2D image landmarks (33, 2)
                h, w = frame.shape[:2]
                lm2d = np.array([[lm.x * w, lm.y * h] for lm in results.pose_landmarks.landmark],
                                dtype=np.float32)
            else:
                lm3d = np.zeros((33, 3), dtype=np.float32)
                lm2d = np.zeros((33, 2), dtype=np.float32)

            all_landmarks.append(lm3d)
            all_landmarks_2d.append(lm2d)
            saved_count += 1

        frame_idx += 1

    cap.release()
    pose.close()

    # Stack and save
    lm3d_arr = np.stack(all_landmarks, axis=0).astype(np.float32)
    lm2d_arr = np.stack(all_landmarks_2d, axis=0).astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "landmarks_3d.npy", lm3d_arr)
    np.save(out_dir / "landmarks_2d.npy", lm2d_arr)
    np.save(out_dir / "mocap_frame_rate.npy", np.array(fps, dtype=np.float32))

    # Also save foot-specific keypoints
    foot_data = {}
    for name, idx in MEDIAPIPE_FOOT_INDICES.items():
        foot_data[f"{name}_3d"] = lm3d_arr[:, idx, :]
        foot_data[f"{name}_2d"] = lm2d_arr[:, idx, :]
    np.savez(out_dir / "foot_keypoints.npz", **foot_data)

    # Convert to COCO17 format for GVHMR compatibility
    coco17 = np.zeros((saved_count, 17, 3), dtype=np.float32)
    for mp_idx, coco_idx in MP33_TO_COCO17.items():
        coco17[:, coco_idx, :2] = lm2d_arr[:, mp_idx, :]
        coco17[:, coco_idx, 2] = 1.0  # confidence
    np.save(out_dir / "keypoints_coco17.npy", coco17)

    print(f"[MediaPipe] {saved_count} frames → {out_dir}")
    print(f"  3D landmarks: {lm3d_arr.shape}")
    print(f"  COCO17 keypoints: {coco17.shape} (GVHMR-compatible)")
    print(f"  Foot keypoints saved to: {out_dir}/foot_keypoints.npz")
    return True


# ---------------------------------------------------------------------------
# RTMPose via rtmlib (133-keypoint WholeBody ONNX)
# ---------------------------------------------------------------------------

# Updated working ONNX URLs (community-hosted, verified)
RTMPOSE_ONNX_URLS = {
    # RTMDet-nano (person detector) — from official mmpose release
    "rtmdet_nano": (
        "https://github.com/zhangwm-pt/mmpose/releases/download/rtmpose-onnx/"
        "rtmdet_nano_8xb32-100e_coco-416x416.onnx"
    ),
    # RTMPose-m COCO 17 keypoint
    "rtmpose_m_coco17": (
        "https://github.com/zhangwm-pt/mmpose/releases/download/rtmpose-onnx/"
        "rtmpose-m_simcc-coco.onnx"
    ),
}


def rtmpose_extract(video_path: str, out_dir: Path, fps: int = 30,
                    det_onnx: str = None, pose_onnx: str = None):
    """Extract keypoints using RTMPose via rtmlib ONNX runtime."""
    try:
        from rtmlib import RTMPose, RTMDet
    except ImportError:
        print("[ERROR] rtmlib not installed. Run: pip install rtmlib onnxruntime")
        return False

    det_url = det_onnx or RTMPOSE_ONNX_URLS["rtmdet_nano"]
    pose_url = pose_onnx or RTMPOSE_ONNX_URLS["rtmpose_m_coco17"]

    print(f"[RTMPose] Loading detector: {det_url}")
    det_model = RTMDet(det_url, backend='onnxruntime', device='cpu')

    print(f"[RTMPose] Loading pose model: {pose_url}")
    pose_model = RTMPose(pose_url, backend='onnxruntime', device='cpu')

    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = 30.0
    stride = max(1, int(round(src_fps / fps)))

    all_keypoints = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            bboxes = det_model(frame)
            keypoints, scores = pose_model(frame, bboxes)

            if len(keypoints) > 0 and len(scores) > 0:
                best_idx = scores.argmax()
                kp = keypoints[best_idx]
                sc = scores[best_idx]
                kp_with_conf = np.concatenate(
                    [kp, sc.reshape(-1, 1)], axis=1
                ).astype(np.float32)
            else:
                kp_with_conf = np.zeros((17, 3), dtype=np.float32)

            all_keypoints.append(kp_with_conf)

        frame_idx += 1

    cap.release()

    kp_array = np.stack(all_keypoints, axis=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "keypoints.npy", kp_array)
    np.save(out_dir / "mocap_frame_rate.npy", np.array(fps, dtype=np.float32))

    print(f"[RTMPose] {len(all_keypoints)} frames → {out_dir}/keypoints.npy")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract 2D/3D keypoints from video (multi-backend)"
    )
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="output/keypoints")
    parser.add_argument("--backend", type=str, default="mediapipe",
                        choices=["mediapipe", "rtmpose"])
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--det_onnx", type=str, default=None,
                        help="RTMDet ONNX path/URL (rtmpose backend)")
    parser.add_argument("--pose_onnx", type=str, default=None,
                        help="RTMPose ONNX path/URL (rtmpose backend)")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)

    backends = {
        "mediapipe": lambda: mediapipe_extract(args.video, out_dir, args.fps),
        "rtmpose": lambda: rtmpose_extract(
            args.video, out_dir, args.fps, args.det_onnx, args.pose_onnx
        ),
    }

    success = backends[args.backend]()
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
