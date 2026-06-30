#!/usr/bin/env python3
"""
Convert a per-frame layout JSON from pose frame to camera frame.

Applies the same transformation as visualize_mesh_3d_video.py:
  1. Reorder translation [tx,ty,tz] -> [tz,tx,ty] and scale by --translation-scale
  2. Convert rotation and translation from pose frame (x-fwd, y-left, z-up)
     to camera frame (x-right, y-down, z-fwd)

Usage:
  python convert_layout_to_camera_frame.py \
    --input /path/to/video_dir/obj_tracking_out/<object>/combined_visualization/layout.json \
    --output /path/to/video_dir/obj_tracking_out/<object>/combined_visualization/layout_camera_frame.json
"""

import argparse
import json
import numpy as np
from scipy.spatial.transform import Rotation as R


# Pose frame -> Camera frame rotation matrix (for row vectors: v_cam = v_pose @ P)
# x_cam = -y_pose, y_cam = -z_pose, z_cam = x_pose
P = np.array([
    [0, 0, 1],
    [-1, 0, 0],
    [0, -1, 0],
], dtype=np.float64)


def convert_translation(tx, ty, tz, translation_scale):
    """Reorder and scale translation, then convert to camera frame."""
    # Reorder as done in visualize_mesh_3d_video.py
    t_pose = np.array([tz * translation_scale, tx * translation_scale, ty * translation_scale])
    # Convert to camera frame: v_cam = v_pose @ P
    t_cam = t_pose @ P
    return t_cam.tolist()


def convert_rotation(quat_wxyz):
    """Convert rotation from pose frame to camera frame."""
    # new_quat is wxyz
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    R_pose = R.from_quat(quat_xyzw).as_matrix()
    # R_cam = P.T @ R_pose  (change of basis)
    R_cam = P.T @ R_pose
    quat_cam_xyzw = R.from_matrix(R_cam).as_quat()  # returns xyzw
    quat_cam_wxyz = [float(quat_cam_xyzw[3]), float(quat_cam_xyzw[0]),
                     float(quat_cam_xyzw[1]), float(quat_cam_xyzw[2])]
    return quat_cam_wxyz


def main():
    parser = argparse.ArgumentParser(
        description='Convert layout JSON poses from pose frame to camera frame.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input', type=str, required=True, help='Input layout JSON')
    parser.add_argument('--output', type=str, required=True, help='Output layout JSON')
    parser.add_argument('--scale', type=float, default=None,
                        help='Mesh scale (written to output for reference, does not affect transform)')
    parser.add_argument('--translation-scale', type=float, default=1.0,
                        help='Scale factor for translation values')
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    for obj in data["objects"]:
        pose = obj["local_to_scene"]
        tx, ty, tz = pose["translation"]
        quat_wxyz = pose["new_quat"]

        t_cam = convert_translation(tx, ty, tz, args.translation_scale)
        q_cam = convert_rotation(quat_wxyz)

        pose["translation_camera_frame"] = t_cam
        pose["quat_wxyz_camera_frame"] = q_cam

    data["frame"] = "camera_frame"
    data["note"] = (
        "Poses converted to camera frame (x-right, y-down, z-fwd). "
        f"translation_scale={args.translation_scale}. "
        "Original pose-frame values preserved."
    )
    if args.scale is not None:
        data["mesh_scale"] = args.scale

    with open(args.output, 'w') as f:
        json.dump(data, f, indent=2)

    num = len([o for o in data["objects"] if "frame_idx" in o or "frame_index" in o])
    print(f"Converted {num} frames to camera frame -> {args.output}")


if __name__ == "__main__":
    main()
