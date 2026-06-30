import argparse
import pathlib
import subprocess
import sys


HERE = pathlib.Path(__file__).parent


def output_path_for(src_file, src_root, tgt_root):
    rel = src_file.relative_to(src_root)
    return (tgt_root / rel).with_suffix(".pkl")


def video_path_for(pkl_file, tgt_root, video_root, render_mode):
    rel = pkl_file.relative_to(tgt_root)
    out = video_root / rel.parent / f"{pkl_file.stem}_gmr_{render_mode}.mp4"
    return out


def run_command(command):
    print("+ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True)


def run_retarget(args):
    command = [
        sys.executable,
        str(HERE / "smpl_npz_to_robot_headless.py"),
        "--src_root",
        str(args.src_root),
        "--tgt_root",
        str(args.tgt_root),
        "--pattern",
        args.pattern,
        "--robot",
        args.robot,
        "--model_type",
        args.model_type,
        "--body_model_path",
        str(args.body_model_path),
        "--target_fps",
        str(args.target_fps),
        "--device",
        args.device,
        "--solver",
        args.solver,
        "--coord_transform",
        args.coord_transform,
        "--height_adjust_mode",
        args.height_adjust_mode,
        "--ground_offset",
        str(args.ground_offset),
        "--smooth_window",
        str(args.smooth_window),
        "--smooth_polyorder",
        str(args.smooth_polyorder),
    ]
    if args.override:
        command.append("--override")
    if args.verbose:
        command.append("--verbose")
    run_command(command)


def run_render(args, pkl_file, video_file):
    command = [
        sys.executable,
        str(HERE / "render_robot_motion_headless.py"),
        "--mode",
        args.render_mode,
        "--robot",
        args.robot,
        "--robot_motion_path",
        str(pkl_file),
        "--video_path",
        str(video_file),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--skip",
        str(args.skip),
        "--max_frames",
        str(args.max_frames),
        "--fps",
        str(args.render_fps),
        "--radius",
        str(args.radius),
        "--camera_mode",
        args.camera_mode,
    ]
    if args.render_mode == "mujoco" and args.mujoco_gl:
        command.extend(["--mujoco_gl", args.mujoco_gl])
    run_command(command)


def main():
    parser = argparse.ArgumentParser(
        description="Batch retarget GVHMR/Locomotion NPZ files to robot PKL and render MP4 videos."
    )
    parser.add_argument("--src_root", required=True, type=pathlib.Path)
    parser.add_argument("--tgt_root", required=True, type=pathlib.Path)
    parser.add_argument("--video_root", default=None, type=pathlib.Path)
    parser.add_argument("--pattern", default="*/selected/001/001_selected.npz")

    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--body_model_path", required=True, type=pathlib.Path)
    parser.add_argument("--model_type", choices=["smpl", "smplh", "smplx"], default="smplh")
    parser.add_argument("--target_fps", default=30, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--solver", default="daqp")
    parser.add_argument("--coord_transform", choices=["gvhmr", "none"], default="gvhmr")
    parser.add_argument("--height_adjust_mode", choices=["global", "per_frame", "none"], default="global")
    parser.add_argument("--ground_offset", default=0.0, type=float)
    parser.add_argument("--smooth_window", default=9, type=int)
    parser.add_argument("--smooth_polyorder", default=2, type=int)
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--render_mode", choices=["mujoco", "skeleton", "none"], default="mujoco")
    parser.add_argument("--mujoco_gl", default="egl")
    parser.add_argument("--camera_mode", choices=["custom", "threequarter", "front", "side", "back", "top"], default="side")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--skip", type=int, default=2)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--render_fps", type=int, default=0)
    parser.add_argument("--radius", type=float, default=2.8)
    parser.add_argument("--render_override", action="store_true")
    args = parser.parse_args()

    args.src_root = args.src_root.resolve()
    args.tgt_root = args.tgt_root.resolve()
    video_root = (args.video_root or args.tgt_root).resolve()

    src_files = sorted(args.src_root.glob(args.pattern))
    print(f"Found {len(src_files)} source files")
    if not src_files:
        return

    run_retarget(args)

    if args.render_mode == "none":
        print(f"Retargeting finished. PKL files are under {args.tgt_root}")
        return

    rendered = 0
    skipped = 0
    failed = 0
    for src_file in src_files:
        pkl_file = output_path_for(src_file.resolve(), args.src_root, args.tgt_root)
        if not pkl_file.is_file():
            print(f"Skip missing PKL: {pkl_file}")
            failed += 1
            continue
        video_file = video_path_for(pkl_file, args.tgt_root, video_root, args.render_mode)
        if video_file.exists() and not args.render_override:
            print(f"Skip existing video: {video_file}")
            skipped += 1
            continue
        video_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            run_render(args, pkl_file, video_file)
            rendered += 1
        except subprocess.CalledProcessError as exc:
            print(f"Render failed for {pkl_file}: {exc}")
            failed += 1

    print(
        f"Done. rendered={rendered}, skipped={skipped}, failed={failed}, "
        f"pkl_root={args.tgt_root}, video_root={video_root}"
    )


if __name__ == "__main__":
    main()
