#!/usr/bin/env python3
"""
Interactive video mesh visualization with viser.

Opens a 3D viewer with a frame slider to step through video frames,
updating the object mesh pose and hand meshes per frame.

Usage:
  python visualize_3d.py \
      --frames-dir /path/to/video_dir/all_frames \
      --layout-json /path/to/video_dir/obj_tracking_out/<object>/combined_visualization/layout_camera_frame_optimized.json \
      --mesh /path/to/video_dir/video_segmentation/masks/frame_000081_masks/<object>/<object>.obj \
      --hand-meshes /path/to/video_dir/<video_name>/all_hand_meshes.npz \
      --scale 0.10970 \
      --translation-scale 1.0 \
      --port 8080

Example usage for clip 0641, drink task, blender_lid object:

  python visualize_3d.py \
      --frames-dir /path/to/video_dir/all_frames \
      --layout-json /path/to/video_dir/obj_tracking_out/<object>/combined_visualization/layout_camera_frame_optimized.json \
      --mesh /path/to/video_dir/obj_tracking_out/<object>/frame_000048/<object>/<object>.obj \
      --hand-meshes /path/to/video_dir/<video_name>/all_hand_meshes.npz \
      --scale 0.4638 \
      --translation-scale 1.0 \
      --port 8080

"""

import argparse
import math
import os
import sys
import time
import threading

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

import viser
import trimesh


def parse_args():
    parser = argparse.ArgumentParser(
        description='Interactive video mesh visualization with viser.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--frames-dir', type=str, required=True,
                        help='Directory containing frame images (e.g. 000000.png)')
    parser.add_argument('--layout-json', type=str, required=True,
                        help='Path to layout JSON with per-frame poses')
    parser.add_argument('--mesh', type=str, required=True,
                        help='Path to object mesh (.obj)')
    parser.add_argument('--hand-meshes', type=str, default=None,
                        help='Path to hand meshes NPZ file')
    parser.add_argument('--hands', type=str, default='both',
                        choices=['left', 'right', 'both'],
                        help='Which hand(s) to render')
    parser.add_argument('--scale', type=float, default=0.19408,
                        help='Mesh scale factor')
    parser.add_argument('--translation-scale', type=float, default=1.0,
                        help='Scale factor for translation values from JSON')
    parser.add_argument('--fx', type=float, default=1346.4437866210938)
    parser.add_argument('--fy', type=float, default=1346.4437866210938)
    parser.add_argument('--cx', type=float, default=640.0)
    parser.add_argument('--cy', type=float, default=360.0)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--port', type=int, default=8080,
                        help='Viser server port')
    parser.add_argument('--frustum-scale', type=float, default=0.2,
                        help='Scale of camera frustum visualization')
    parser.add_argument('--fps', type=float, default=30.0,
                        help='Playback FPS for auto-play')
    return parser.parse_args()


def pose_to_camera_frame(verts_pose):
    """Convert vertices from pose frame (x-fwd, y-left, z-up) to camera frame (x-right, y-down, z-fwd)."""
    verts_cam = np.zeros_like(verts_pose)
    verts_cam[:, 0] = -verts_pose[:, 1]
    verts_cam[:, 1] = -verts_pose[:, 2]
    verts_cam[:, 2] = verts_pose[:, 0]
    return verts_cam


def transform_mesh_to_camera_frame(vertices, rot_matrix, translation, scale):
    """Apply scale, rotation, translation in pose frame, then convert to camera frame."""
    verts = vertices * scale
    verts_pose = verts @ rot_matrix.T + translation
    return pose_to_camera_frame(verts_pose)


def load_layout(json_path):
    """Parse layout JSON into dict[frame_idx -> {quat, translation, scale, camera_frame}].

    Auto-detects camera-frame layouts (with translation_camera_frame / quat_wxyz_camera_frame).
    For camera-frame layouts, translation and quat are already in camera frame and need no conversion.
    """
    import json
    with open(json_path) as f:
        data = json.load(f)

    is_camera_frame = data.get("frame") == "camera_frame"

    layout = {}
    for obj in data["objects"]:
        if "frame_index" not in obj and "frame_idx" not in obj:
            continue
        frame_idx = obj.get("frame_index", obj.get("frame_idx"))
        pose = obj["local_to_scene"]

        if is_camera_frame and "translation_camera_frame" in pose and "quat_wxyz_camera_frame" in pose:
            layout[frame_idx] = {
                "quat": pose["quat_wxyz_camera_frame"],
                "translation": pose["translation_camera_frame"],
                "scale": pose.get("scale", None),
                "translation_scale_optimized": pose.get("translation_scale_optimized", None),
                "camera_frame": True,
            }
        else:
            layout[frame_idx] = {
                "quat": pose["new_quat"],
                "translation": pose["translation"],
                "scale": pose.get("scale", None),
                "camera_frame": False,
            }

    if is_camera_frame:
        print("  [info] Detected camera-frame layout, skipping pose-frame conversion")

    return layout


def main():
    args = parse_args()

    # Validate paths
    if not os.path.isdir(args.frames_dir):
        print(f"[error] Frames directory not found: {args.frames_dir}")
        sys.exit(1)
    if not os.path.exists(args.layout_json):
        print(f"[error] Layout JSON not found: {args.layout_json}")
        sys.exit(1)
    if not os.path.exists(args.mesh):
        print(f"[error] Mesh not found: {args.mesh}")
        sys.exit(1)

    # Load layout
    print("Loading layout JSON...")
    layout = load_layout(args.layout_json)
    frame_indices = sorted(layout.keys())
    num_frames = len(frame_indices)
    max_frame = max(frame_indices)
    print(f"  {num_frames} frames, range [{min(frame_indices)}, {max_frame}]")

    # Load mesh (shared topology)
    print("Loading mesh...")
    mesh = trimesh.load_mesh(args.mesh)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    mesh_verts = np.array(mesh.vertices)
    mesh_faces = mesh.faces
    mesh_visual = mesh.visual
    print(f"  {len(mesh_verts)} vertices, {len(mesh_faces)} faces")

    # Load hand meshes
    hand_data = None
    hands_to_render = []
    hand_colors = {'left': (204, 128, 128), 'right': (128, 128, 204)}
    if args.hand_meshes:
        if not os.path.exists(args.hand_meshes):
            print(f"[error] Hand meshes not found: {args.hand_meshes}")
            sys.exit(1)
        print("Loading hand meshes...")
        hand_data = np.load(args.hand_meshes)
        hands_to_render = ['left', 'right'] if args.hands == 'both' else [args.hands]
        for hand in hands_to_render:
            n = hand_data[f'{hand}_vertices'].shape[0]
            print(f"  {hand} hand: {n} frames, {hand_data[f'{hand}_vertices'].shape[1]} vertices")

    # Camera intrinsics
    fov_y = 2.0 * math.atan(args.height / (2.0 * args.fy))
    aspect = args.width / args.height

    # Start viser server
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-y")

    # Even multi-directional lighting so side / back orbit views stay bright.
    # Camera frame: +x right, +y down (so -y is up), +z forward. Directional
    # lights shine from `position` toward the world origin.
    server.scene.add_light_ambient("/lights/ambient", intensity=0.6)
    server.scene.add_light_hemisphere(
        "/lights/hemi",
        sky_color=(255, 255, 255),
        ground_color=(180, 180, 200),
        intensity=0.5,
    )
    server.scene.add_light_directional(
        "/lights/key_front", position=(0.5, -1.5, -1.0), intensity=0.7,
    )
    server.scene.add_light_directional(
        "/lights/fill_back", position=(0.0, -1.0, 2.5), intensity=0.7,
    )
    server.scene.add_light_directional(
        "/lights/fill_left", position=(-2.0, -0.5, 1.0), intensity=0.5,
    )
    server.scene.add_light_directional(
        "/lights/fill_right", position=(2.0, -0.5, 1.0), intensity=0.5,
    )
    server.scene.add_light_directional(
        "/lights/top", position=(0.0, -2.5, 1.0), intensity=0.4,
    )

    # Add coordinate frame
    server.scene.add_frame("/camera/axes", show_axes=True, axes_length=0.1, axes_radius=0.005)

    # GUI controls
    frame_slider = server.gui.add_slider(
        "Frame", min=0, max=max_frame, step=1, initial_value=0,
    )
    play_button = server.gui.add_button("Play")
    fps_slider = server.gui.add_slider(
        "FPS", min=1, max=60, step=1, initial_value=int(args.fps),
    )
    texture_checkbox = server.gui.add_checkbox("Enable Texture", initial_value=False)

    # Playback state
    playing = False
    play_lock = threading.Lock()

    def update_frame(frame_idx):
        """Update the scene for the given frame index."""
        # Load frame image
        img_path = os.path.join(args.frames_dir, f"{frame_idx:06d}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(args.frames_dir, f"{frame_idx:06d}.jpg")
        if not os.path.exists(img_path):
            return

        frame_bgr = cv2.imread(img_path)
        if frame_bgr is None:
            return
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Camera frustum with frame image
        server.scene.add_camera_frustum(
            "/camera",
            fov=fov_y,
            aspect=aspect,
            scale=args.frustum_scale,
            image=frame_rgb,
            format="jpeg",
            jpeg_quality=90,
            color=(30, 30, 30),
        )

        # Get pose from layout
        if frame_idx not in layout:
            return

        pose = layout[frame_idx]
        quat_wxyz = pose["quat"]
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        rot_matrix = R.from_quat(quat_xyzw).as_matrix()

        if pose.get("camera_frame"):
            # Already in camera frame — apply directly
            translation = np.array(pose["translation"])
            verts_cam = (mesh_verts * args.scale) @ rot_matrix.T + translation
        else:
            # Pose frame — reorder translation and convert
            tx, ty, tz = pose["translation"]
            translation = np.array([
                tz * args.translation_scale,
                tx * args.translation_scale,
                ty * args.translation_scale,
            ])
            verts_cam = transform_mesh_to_camera_frame(
                mesh_verts, rot_matrix, translation, args.scale,
            )

        # Compute object frame position and orientation in camera frame
        if pose.get("camera_frame"):
            obj_translation = np.array(pose["translation"])
            obj_rotation = rot_matrix
        else:
            tx, ty, tz = pose["translation"]
            translation_pose = np.array([
                tz * args.translation_scale,
                tx * args.translation_scale,
                ty * args.translation_scale,
            ])
            # Pose-to-camera rotation: maps pose-frame axes to camera-frame axes
            R_pose2cam = np.array([
                [0, -1, 0],
                [0, 0, -1],
                [1, 0, 0],
            ], dtype=float)
            obj_translation = R_pose2cam @ translation_pose
            obj_rotation = R_pose2cam @ rot_matrix

        # Add coordinate frame at the object's origin
        obj_quat_xyzw = R.from_matrix(obj_rotation).as_quat()  # [x,y,z,w]
        obj_quat_wxyz = (obj_quat_xyzw[3], obj_quat_xyzw[0], obj_quat_xyzw[1], obj_quat_xyzw[2])
        server.scene.add_frame(
            "/object_frame",
            show_axes=True,
            axes_length=0.08,
            axes_radius=0.004,
            wxyz=obj_quat_wxyz,
            position=obj_translation,
        )

        # Update object mesh
        if texture_checkbox.value:
            mesh_cam = trimesh.Trimesh(vertices=verts_cam, faces=mesh_faces, visual=mesh_visual)
            server.scene.add_mesh_trimesh("/object_mesh", mesh=mesh_cam)
        else:
            server.scene.add_mesh_simple(
                "/object_mesh",
                vertices=verts_cam.astype(np.float32),
                faces=mesh_faces.astype(np.uint32),
                color=(180, 180, 180),
                side="double",
            )

        # Update hand meshes
        if hand_data is not None:
            for hand in hands_to_render:
                hand_verts = hand_data[f'{hand}_vertices']
                hand_faces = hand_data[f'{hand}_faces']
                # Clamp frame index to available hand frames
                hi = min(frame_idx, hand_verts.shape[0] - 1)
                server.scene.add_mesh_simple(
                    f"/hand_{hand}",
                    vertices=hand_verts[hi].astype(np.float32),
                    faces=hand_faces.astype(np.uint32),
                    color=hand_colors[hand],
                    side="double",
                )

    # Initial frame
    update_frame(0)

    @frame_slider.on_update
    def on_slider_change(_):
        update_frame(int(frame_slider.value))

    @texture_checkbox.on_update
    def on_texture_change(_):
        update_frame(int(frame_slider.value))

    @play_button.on_click
    def on_play_click(_):
        nonlocal playing
        with play_lock:
            playing = not playing
            play_button.name = "Pause" if playing else "Play"

    def playback_loop():
        nonlocal playing
        while True:
            if playing:
                current = int(frame_slider.value)
                next_frame = current + 1
                if next_frame > max_frame:
                    next_frame = 0
                frame_slider.value = next_frame
                update_frame(next_frame)
                time.sleep(1.0 / fps_slider.value)
            else:
                time.sleep(0.05)

    playback_thread = threading.Thread(target=playback_loop, daemon=True)
    playback_thread.start()

    print(f"\nViewer running at http://localhost:{args.port}")
    print(f"Frames: 0-{max_frame}, use slider or Play button to navigate")
    print("Press Ctrl+C to exit")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
