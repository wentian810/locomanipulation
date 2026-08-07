"""Independent VideoMimic/NKSR sidecar runner for one existing clip.

The runner keeps all artifacts in ``scene_work``.  It neither reruns the
project's human front end nor alters PHC/GMR outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


STAGES = ("export", "bstro", "validate", "megasam", "megahunter", "postprocess", "all")


def _run(command: list[str], *, cwd: Path, log: Path, dry_run: bool) -> None:
    rendered = " ".join(command)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(rendered + "\n")
    print(f"[RUN] {rendered}")
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def _conda(env_name: str, command: list[str], *runtime_env: str) -> list[str]:
    return [
        "conda", "run", "--no-capture-output", "-n", env_name, "env",
        "PYTHONNOUSERSITE=1", *runtime_env, *command,
    ]


def _latest_h5(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise RuntimeError(f"missing {label} matching {pattern} in {directory}")
    return matches[-1]


def _status(path: Path, stage: str, state: str, **extra: object) -> None:
    payload = {"schema_version": 1, "stage": stage, "status": state, **extra}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--clip-dir", required=True, type=Path)
    parser.add_argument("--work-video", required=True, type=Path)
    parser.add_argument("--videomimic-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--motion-npz", type=Path, default=None)
    parser.add_argument("--camera-npz", type=Path, default=None)
    parser.add_argument("--gender", choices=("auto", "male", "female", "neutral"), default="auto")
    parser.add_argument("--recon-env", default="vm1recon")
    parser.add_argument("--human-env", default="vm1rs")
    parser.add_argument("--human-mask-mode", choices=("sam2", "bbox"), default="sam2")
    parser.add_argument("--sam2-env", default="vm1rs",
                        help="Environment containing VideoMimic Grounded-SAM2 dependencies.")
    parser.add_argument(
        "--sam2-extra-dilation-radius-px",
        type=int,
        default=0,
        help="Extra raw-image mask radius after SAM2; preserves a backup before changing masks.",
    )
    parser.add_argument(
        "--background-filter-mode",
        choices=("spatiotemporal_ring", "static_outside_human"),
        default="spatiotemporal_ring",
        help="Use the legacy local ring or the full static background outside the human mask.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Optional physical GPU index or CUDA_VISIBLE_DEVICES expression for every GPU stage.",
    )
    parser.add_argument("--overwrite-adapter", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.sam2_extra_dilation_radius_px < 0:
        parser.error("--sam2-extra-dilation-radius-px must be non-negative")
    project = args.project_root.resolve()
    vm = args.videomimic_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # The adapter deliberately refuses a non-empty output directory to avoid
    # mixing a partial reconstruction with a new export.  Keep orchestration
    # metadata next to, rather than inside, that adapter-owned directory so a
    # newly created ``--output`` remains genuinely empty for the first stage.
    runner_dir = output.parent / f"{output.name}_runner"
    runner_dir.mkdir(parents=True, exist_ok=True)
    log = runner_dir / "command_log.txt"
    status = runner_dir / "status.json"
    scene_scripts = project / "scripts" / "scene"
    gpu_env = (
        (f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}", "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128")
        if args.cuda_visible_devices
        else ()
    )
    human_gpu_env = (*gpu_env, "XLA_PYTHON_CLIENT_PREALLOCATE=false")
    clip_id = args.clip_dir.resolve().name
    selected = set(STAGES[:-1] if args.stage == "all" else (args.stage,))
    try:
        if "export" in selected:
            command = [
                "python", str(scene_scripts / "export_videomimic_smpl_inputs.py"),
                "--clip-dir", str(args.clip_dir), "--work-video", str(args.work_video),
                "--output", str(output), "--clip-id", clip_id,
                "--mask-mode", args.human_mask_mode,
            ]
            if args.max_frames > 0:
                command += ["--max-frames", str(args.max_frames)]
            if args.frame_stride != 1:
                command += ["--frame-stride", str(args.frame_stride)]
            if args.motion_npz is not None:
                command += ["--motion-npz", str(args.motion_npz)]
            if args.camera_npz is not None:
                command += ["--camera-npz", str(args.camera_npz)]
            if args.overwrite_adapter:
                command.append("--overwrite")
            _run(_conda(args.recon_env, command, *gpu_env), cwd=project, log=log, dry_run=args.dry_run)
            if args.human_mask_mode == "sam2":
                sam2_output = output / "videomimic" / "input_masks" / clip_id / "cam01"
                _run(_conda(args.sam2_env, [
                    "python", "stage0_preprocessing/sam2_segmentation.py",
                    "--text", "person.", "--video-dir", str(output / "videomimic" / "input_images" / clip_id / "cam01"),
                    "--output-dir", str(sam2_output),
                ], *gpu_env), cwd=vm, log=log, dry_run=args.dry_run)
                _run(_conda(args.recon_env, [
                    "python", str(scene_scripts / "repair_terminal_sam2_mask.py"),
                    "--output", str(output),
                ], *gpu_env), cwd=project, log=log, dry_run=args.dry_run)
                if args.sam2_extra_dilation_radius_px:
                    _run(_conda(args.recon_env, [
                        "python", str(scene_scripts / "apply_videomimic_mask_dilation.py"),
                        "--mask-dir", str(sam2_output / "mask_data"),
                        "--extra-radius-px", str(args.sam2_extra_dilation_radius_px),
                        "--report-json", str(output / "sam2_mask_dilation_report.json"),
                    ], *gpu_env), cwd=project, log=log, dry_run=args.dry_run)
        if "bstro" in selected:
            image_dir_bstro = output / "videomimic" / "input_images" / clip_id / "cam01"
            bbox_dir_bstro = output / "videomimic" / "input_masks" / clip_id / "cam01" / "json_data"
            contact_dir_bstro = output / "videomimic" / "input_contacts" / clip_id / "cam01"
            for required_path in (image_dir_bstro, bbox_dir_bstro):
                if not required_path.is_dir() and not args.dry_run:
                    raise RuntimeError(f"BSTRO requires adapter/SAM2 inputs: {required_path}")
            _run(_conda(args.human_env, [
                "python", "stage0_preprocessing/bstro_contact_detection.py",
                "--video-dir", str(image_dir_bstro), "--bbox-dir", str(bbox_dir_bstro),
                "--output-dir", str(contact_dir_bstro),
                "--feet-contact-ratio-thr", "0.2", "--contact-thr", "0.95",
            ], *human_gpu_env), cwd=vm, log=log, dry_run=args.dry_run)
        if "validate" in selected:
            _run(_conda(args.recon_env, ["python", str(scene_scripts / "validate_videomimic_smpl_adapter.py"), "--output", str(output)], *gpu_env), cwd=project, log=log, dry_run=args.dry_run)
        report_path = output / "adapter_report.json"
        if not report_path.is_file() and not args.dry_run:
            raise RuntimeError(f"missing adapter report: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {"frame_count": args.max_frames, "gender": "neutral"}
        frame_count = int(report["frame_count"])
        source_gender = str(report.get("gender", "neutral")).lower()
        gender = source_gender if args.gender == "auto" and source_gender in {"male", "female", "neutral"} else args.gender
        if gender == "auto":
            gender = "neutral"
        image_dir = output / "videomimic" / "input_images" / clip_id / "cam01"
        mask_dir = output / "videomimic" / "input_masks" / clip_id / "cam01" / "json_data"
        pose_dir = output / "videomimic" / "input_2d_poses" / clip_id / "cam01"
        smpl_dir = output / "videomimic" / "input_3d_meshes" / clip_id / "cam01"
        megasam_dir = output / "videomimic" / "input_megasam"
        if "megasam" in selected:
            end = frame_count
            _run(_conda(args.recon_env, [
                "python", "stage1_reconstruction/megasam_reconstruction.py", "--video-dir", str(image_dir),
                "--out-dir", str(megasam_dir), "--start-frame", "0", "--end-frame", str(end),
                "--stride", "1", "--gsam2",
            ], *gpu_env), cwd=vm, log=log, dry_run=args.dry_run)
        megasam = _latest_h5(megasam_dir, f"megasam_reconstruction_results_{clip_id}_cam01_*.h5", "MegaSAM result") if not args.dry_run else megasam_dir / "<megasam>.h5"
        hunter_dir = output / "videomimic" / "output_smpl_and_points"
        if "megahunter" in selected:
            _run(_conda(args.human_env, [
                "python", str(scene_scripts / "run_megahunter_external.py"), "--videomimic-root", str(vm),
                "--world-env-path", str(megasam), "--bbox-dir", str(mask_dir), "--pose2d-dir", str(pose_dir),
                "--smpl-dir", str(smpl_dir), "--out-dir", str(hunter_dir), "--gender", gender,
                "--preserve-external-root-translation",
            ], *human_gpu_env), cwd=project, log=log, dry_run=args.dry_run)
        hunter = _latest_h5(hunter_dir, "megahunter_*.h5", "MegaHunter result") if not args.dry_run else hunter_dir / "<megahunter>.h5"
        if "postprocess" in selected:
            post_dir = output / "videomimic" / "output_calib_mesh"
            command = [
                "python", "stage3_postprocessing/postprocessing_pipeline.py", "--megahunter-path", str(hunter),
                "--out-dir", str(post_dir), "--gender", gender, "--is-megasam", "--meshification-method", "nksr",
            ]
            if args.background_filter_mode == "static_outside_human":
                command.append("--no-spf")
            _run(_conda(args.recon_env, command, *gpu_env), cwd=vm, log=log, dry_run=args.dry_run)
        _status(
            status,
            args.stage,
            "pass",
            clip_id=clip_id,
            output=str(output),
            background_filter_mode=args.background_filter_mode,
            sam2_extra_dilation_radius_px=args.sam2_extra_dilation_radius_px,
        )
    except Exception as exc:
        _status(status, args.stage, "fail", error=str(exc), clip_id=clip_id)
        raise


if __name__ == "__main__":
    main()
