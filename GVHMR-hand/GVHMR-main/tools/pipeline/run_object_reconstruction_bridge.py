#!/usr/bin/env python3
"""One-shot bridge from a completed FoundationPose run to GVHMR/GMR.

This intentionally does not hide the RGB-D prerequisite: ``foundation_dir``
must already contain a metric mesh and ``pose.npy`` produced by FoundationPose.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PIPELINE_ROOT = HERE.parents[3]
GMR_SCRIPTS = PIPELINE_ROOT / "GMR-master" / "scripts"
DEFAULT_PIPELINE_PYTHON = Path(
    os.environ.get(
        "PY_LOCO",
        str(Path.home() / "miniconda3" / "envs" / "locomotion" / "bin" / "python"),
    )
)
if not DEFAULT_PIPELINE_PYTHON.exists():
    DEFAULT_PIPELINE_PYTHON = Path(sys.executable)


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _find_mesh(root: Path) -> Path | None:
    for name in ("mesh.obj", "mesh.ply", "mesh.stl", "mesh.glb"):
        path = root / "mesh" / name
        if path.exists():
            return path
    for pattern in ("*.obj", "*.ply", "*.stl", "*.glb"):
        matches = sorted((root / "mesh").glob(pattern))
        if matches:
            return matches[0]
    return None


def integrate(args) -> None:
    foundation_dir = args.foundation_dir.resolve()
    clip_dir = args.clip_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else clip_dir / "object_reconstruction"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    pose_path = foundation_dir / "pose.npy"
    if not pose_path.exists():
        raise FileNotFoundError(
            f"{pose_path} is missing. Run FoundationPose tracking first; "
            "an RGB video without aligned depth is not sufficient."
        )
    mesh_path = _find_mesh(foundation_dir)
    if mesh_path is None:
        raise FileNotFoundError(
            f"No mesh found under {foundation_dir / 'mesh'}. "
            "Run Hunyuan3D mesh generation first."
        )

    collision_manifest = None
    if not args.skip_collision:
        collision_dir = output_dir / "collision"
        collision_command = [
            args.python,
            str(GMR_SCRIPTS / "build_object_collision_asset.py"),
            "--mesh",
            str(mesh_path),
            "--output_dir",
            str(collision_dir),
            "--method",
            args.collision_method,
            "--density",
            str(args.density),
        ]
        if args.mass is not None:
            collision_command.extend(["--mass", str(args.mass)])
        _run(collision_command)
        collision_manifest = collision_dir / "collision_manifest.json"

    export_command = [
        args.python,
        str(GMR_SCRIPTS / "import_foundationpose_object.py"),
        "--foundation_dir",
        str(foundation_dir),
        "--clip_dir",
        str(clip_dir),
        "--output_dir",
        str(output_dir),
        "--mesh_path",
        str(mesh_path),
        "--frame_offset",
        str(args.frame_offset),
        "--human_yaw_offset_deg",
        str(args.human_yaw_offset_deg),
        "--robot_offset",
        args.robot_offset,
        "--rgba",
        args.rgba,
        "--density",
        str(args.density),
    ]
    if collision_manifest:
        export_command.extend(
            ["--collision_manifest", str(collision_manifest)]
        )
    if args.source_fps is not None:
        export_command.extend(["--source_fps", str(args.source_fps)])
    _run(export_command)

    gvhmr_video = args.gvhmr_video
    if gvhmr_video is None:
        clip_name = clip_dir.name
        gvhmr_video = clip_dir / "gvhmr_out" / clip_name / "1_incam.mp4"
    gvhmr_video = gvhmr_video.resolve()
    overlay_output = gvhmr_video.with_name("1_incam_object.mp4")
    if not args.skip_overlay:
        if not gvhmr_video.exists():
            raise FileNotFoundError(gvhmr_video)
        _run(
            [
                args.render_python or args.python,
                str(HERE / "render_foundationpose_object.py"),
                "--video",
                str(gvhmr_video),
                "--object_motion",
                str(output_dir / "object_motion_gvhmr.npz"),
                "--output",
                str(overlay_output),
                "--alpha",
                str(args.overlay_alpha),
            ]
        )

    summary = {
        "foundation_dir": str(foundation_dir),
        "clip_dir": str(clip_dir),
        "metric_mesh": str(mesh_path.resolve()),
        "foundation_pose": str(pose_path.resolve()),
        "object_motion_gvhmr": str(
            (output_dir / "object_motion_gvhmr.npz").resolve()
        ),
        "object_motion_gmr": str(
            (output_dir / "object_motion_gmr.npz").resolve()
        ),
        "gvhmr_object_video": (
            str(overlay_output.resolve()) if not args.skip_overlay else None
        ),
        "gmr_env": {
            "GMR_OBJECT_MOTION_NAME": (
                f"{output_dir.name}/object_motion_gmr.npz"
                if output_dir.parent == clip_dir
                else str((output_dir / "object_motion_gmr.npz").resolve())
            ),
            "GMR_OVERRIDE": "1",
        },
    }
    summary_path = output_dir / "integration_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved integration summary: {summary_path}")
    print(
        "Next: set GMR_OBJECT_MOTION_NAME="
        f"{summary['gmr_env']['GMR_OBJECT_MOTION_NAME']} and rerun GMR rendering."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Integrate FoundationPose object reconstruction with GVHMR/GMR."
    )
    parser.add_argument("--foundation_dir", type=Path, required=True)
    parser.add_argument("--clip_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--gvhmr_video", type=Path, default=None)
    parser.add_argument("--python", default=str(DEFAULT_PIPELINE_PYTHON))
    parser.add_argument(
        "--render_python",
        default="",
        help="Python containing pyrender/OpenCV; defaults to --python.",
    )
    parser.add_argument("--frame_offset", type=int, default=0)
    parser.add_argument("--source_fps", type=float, default=None)
    parser.add_argument("--human_yaw_offset_deg", type=float, default=180.0)
    parser.add_argument("--robot_offset", default="0,0,0")
    parser.add_argument("--rgba", default="0.95,0.65,0.2,0.9")
    parser.add_argument("--density", type=float, default=600.0)
    parser.add_argument("--mass", type=float, default=None)
    parser.add_argument(
        "--collision_method",
        choices=["auto", "coacd", "convex-hull"],
        default="auto",
    )
    parser.add_argument("--skip_collision", action="store_true")
    parser.add_argument("--skip_overlay", action="store_true")
    parser.add_argument("--overlay_alpha", type=float, default=0.82)
    return parser.parse_args()


if __name__ == "__main__":
    integrate(parse_args())
