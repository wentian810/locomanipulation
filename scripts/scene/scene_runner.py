"""Entrypoint for isolated fixed-camera scene sidecars.

``crisp`` remains a deliberately disabled fallback contract.  The production
first path is ``videomimic_nksr``: it consumes completed human-pre results,
keeps its large reconstruction cache under ``scene.work_root``, and publishes
a small validated package under the clip output directory.
"""

from __future__ import annotations

import argparse
import os
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


STAGES = ("preflight", "all")


def _load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("effective config must be a JSON object")
    scene = value.get("scene")
    if not isinstance(scene, dict) or scene.get("backend") not in {"crisp", "videomimic_nksr"}:
        raise ValueError("effective config must contain a supported scene.backend")
    if scene.get("camera", {}).get("mode") != "fixed":
        raise ValueError("only scene.camera.mode='fixed' is supported")
    canonicalization = scene.get("camera", {}).get(
        "canonicalization", "static_optimized"
    )
    if canonicalization not in {"static_exact_reprojection", "static_optimized"}:
        raise ValueError("scene.camera.canonicalization is unsupported")
    return value


def _project_path(project: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project / path).resolve()


def _motion_path(human_dir: Path, source: str) -> Path:
    names = {"converted": "001_converted.npz", "smoothed": "001_smoothed.npz", "final": "001_final.npz"}
    try:
        return human_dir / names[source]
    except KeyError as exc:
        raise ValueError(f"unsupported scene.source_motion: {source}") from exc


def _run(command: list[str], *, cwd: Path) -> None:
    print("[SCENE][RUN] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)



def _conda_env_python(env_name: str) -> Path:
    prefix_text = os.environ.get("CONDA_PREFIX")
    if not prefix_text:
        raise RuntimeError("Stage-4 requires CONDA_PREFIX to locate its isolated environment")
    prefix = Path(prefix_text).resolve()
    candidate = prefix / "bin" / "python" if prefix.name == env_name else prefix.parent / env_name / "bin" / "python"
    if not candidate.is_file():
        raise FileNotFoundError(f"Stage-4 Python for environment {env_name!r} is missing: {candidate}")
    return candidate


def _run_videomimic_stage4(*, config: dict[str, Any], project: Path, clip: str, work: Path, package: Path, root: Path) -> None:
    video = config["scene"]["videomimic"]
    stage4 = video.get("stage4", {})
    if not isinstance(stage4, dict) or not stage4.get("enabled", False):
        return
    stage_python = _conda_env_python(str(stage4.get("env", video.get("human_env", "vm1rs"))))
    stage_version = subprocess.check_output([str(stage_python), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"], text=True).strip()
    cudnn = stage_python.parent.parent / "lib" / f"python{stage_version}" / "site-packages" / "nvidia" / "cudnn" / "lib"
    if not cudnn.is_dir():
        raise FileNotFoundError(f"Stage-4 cuDNN library for Python {stage_version} is missing: {cudnn}")
    output = work / "stage4_visual_clean" / "retarget_poses_g1.h5"
    stage_script = root / "stage4_retargeting" / "robot_motion_retargeting.py"
    contact_dir = work / "videomimic" / "input_contacts" / clip / "cam01"
    required = {
        "visual mesh": package / "scene" / "background_mesh_visual_mujoco.obj",
        "motion": package / "human" / "human_motion_static_hard.npz",
        "frame H5": work / "videomimic" / "output_calib_mesh" / "gravity_calibrated_megahunter.h5",
        "BSTRO contacts": contact_dir,
        "Stage-4 script": stage_script,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Stage-4 sidecar input missing:\n  " + "\n  ".join(missing))
    command = [
        sys.executable, str(project / "scripts" / "scene" / "run_videomimic_stage4_retarget.py"),
        "--smpl-npz", str(required["motion"]),
        "--body-model-root", str(project / "GMR-master" / "assets" / "body_models"),
        "--prepare-evidence-script", str(project / "scripts" / "scene" / "prepare_videomimic_gvhmr_evidence.py"),
        "--evidence-python", sys.executable,
        "--raw-mesh", str(required["visual mesh"]), "--residual-mesh", str(required["visual mesh"]),
        "--megahunter-h5", str(required["frame H5"]), "--contact-dir", str(contact_dir),
        "--stage4-script", str(stage_script), "--stage4-python", str(stage_python),
        "--stage4-cudnn-lib", str(cudnn),
        "--minimum-contact-coverage", str(stage4.get("minimum_contact_coverage", 0.75)),
        "--output-h5", str(output),
    ]
    _run(command, cwd=project)
    manifest_path = output.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "pass":
        raise RuntimeError(f"Stage-4 sidecar did not pass: {manifest_path}")
    summary = {
        "schema_version": 1,
        "status": "pass",
        "kind": "kinematic_scene_optimized_reference_not_mjstep_accepted",
        "candidate": "videomimic_visual_clean_mesh",
        "output_h5": str(output),
        "manifest": str(manifest_path),
        "contact_surface_recovery": manifest.get("contact_surface_recovery"),
    }
    (work / "stage4_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[SCENE][PASS] VideoMimic Stage-4 reference: {output}", flush=True)

def _run_videomimic(config: dict[str, Any], project: Path, clip: str) -> None:
    scene = config["scene"]
    video = scene["videomimic"]
    human_dir = _project_path(project, str(config["output"]["root"])) / clip
    work = _project_path(project, str(scene.get("work_root", "scene_work/videomimic"))) / clip
    package = human_dir / str(scene.get("output_subdir", "scene_reconstruction"))
    aligned_video = human_dir / "gvhmr_out" / clip / "valid_video.mp4"
    motion = _motion_path(human_dir, str(scene.get("source_motion", "smoothed")))
    camera = human_dir / "gvhmr_camera.npz"
    for path in (aligned_video, motion, camera):
        if not path.is_file():
            raise FileNotFoundError(f"VideoMimic scene input is missing: {path}")
    if work.exists():
        raise FileExistsError(f"refusing to mix an existing VideoMimic work directory: {work}")
    if package.exists():
        raise FileExistsError(f"refusing to overwrite an existing scene package: {package}")
    fixed_inputs = human_dir / "scene_fixed_camera_inputs"
    fixed_motion = fixed_inputs / "human_motion_static.npz"
    fixed_camera = fixed_inputs / "gvhmr_camera_static.npz"
    fixed_report = fixed_inputs / "canonicalization_quality.json"
    if fixed_inputs.exists():
        if not all(path.is_file() for path in (fixed_motion, fixed_camera, fixed_report)):
            raise FileExistsError(f"incomplete fixed-camera input package: {fixed_inputs}")
        fixed_quality = json.loads(fixed_report.read_text(encoding="utf-8"))
        expected = {
            "source_motion": motion.resolve(),
            "source_camera": camera.resolve(),
            "output_motion": fixed_motion.resolve(),
            "output_camera": fixed_camera.resolve(),
        }
        if any(Path(str(fixed_quality.get(key, ""))).resolve() != value for key, value in expected.items()):
            raise RuntimeError(f"fixed-camera input provenance does not match this scene run: {fixed_inputs}")
        print(f"[SCENE] reusing accepted fixed-camera inputs: {fixed_inputs}", flush=True)
    else:
        _run(
            [
                sys.executable, str(project / "scripts" / "scene" / "canonicalize_gvhmr_static_camera.py"),
                "--motion-npz", str(motion), "--camera-npz", str(camera),
                "--output-motion-npz", str(fixed_motion),
                "--output-camera-npz", str(fixed_camera),
                "--report-json", str(fixed_report),
                "--mode", str(scene["camera"].get("canonicalization", "static_optimized")),
                "--freeze-method", str(scene["camera"].get("freeze_method", "robust_median")),
            ],
            cwd=project,
        )
        fixed_quality = json.loads(fixed_report.read_text(encoding="utf-8"))
    if fixed_quality.get("status") != "accepted":
        raise RuntimeError(f"fixed-camera inputs were not accepted: {fixed_report}")
    root = _project_path(project, str(video["root"]))
    model_root = root / "assets" / "body_models"
    runtime = scene["runtime"]
    command = [
        sys.executable, str(project / "scripts" / "scene" / "run_videomimic_scene.py"),
        "--project-root", str(project), "--clip-dir", str(human_dir),
        "--work-video", str(aligned_video), "--videomimic-root", str(root),
        "--output", str(work), "--motion-npz", str(fixed_motion), "--camera-npz", str(fixed_camera),
        "--gender", str(video.get("gender", "auto")),
        "--recon-env", str(video.get("recon_env", "vm1recon")),
        "--human-env", str(video.get("human_env", "vm1rs")),
        "--human-mask-mode", str(video.get("human_mask_mode", "sam2")),
        "--sam2-env", str(video.get("sam2_env", "vm1rs")),
        "--sam2-extra-dilation-radius-px", str(video.get("sam2_extra_dilation_radius_px", 0)),
        "--background-filter-mode", str(video.get("background_filter_mode", "spatiotemporal_ring")),
        "--cuda-visible-devices", str(runtime["gpu"]),
    ]
    max_frames = int(video.get("max_frames", 0))
    if max_frames:
        command.extend(("--max-frames", str(max_frames)))
    frame_stride = int(video.get("frame_stride", 1))
    if frame_stride != 1:
        command.extend(("--frame-stride", str(frame_stride)))
    _run(command, cwd=project)
    report = json.loads((work / "adapter_report.json").read_text(encoding="utf-8"))
    gender = str(video.get("gender", "auto"))
    if gender == "auto":
        gender = str(report.get("gender", "neutral")).lower()
    if gender not in {"male", "female", "neutral"}:
        gender = "neutral"
    _run(
        [
            sys.executable, str(project / "scripts" / "scene" / "package_videomimic_scene.py"),
            "--project-root", str(project), "--human-dir", str(human_dir),
            "--work-dir", str(work), "--output-root", str(package),
            "--model-root", str(model_root), "--motion-npz", str(fixed_motion),
            "--camera-npz", str(fixed_camera), "--canonicalization-report", str(fixed_report), "--clip-id", clip,
            "--video-path", str(aligned_video), "--gender", gender,
            "--camera-mode", str(scene["camera"].get("canonicalization", "static_optimized")),
            "--freeze-method", str(scene["camera"].get("freeze_method", "robust_median")),
        ],
        cwd=project,
    )
    _run(
        [
            sys.executable,
            str(project / "scripts" / "scene" / "validate_scene_package.py"),
            "--scene-root",
            str(package),
            "--require-input-provenance",
        ],
        cwd=project,
    )
    _run_videomimic_stage4(config=config, project=project, clip=clip, work=work, package=package, root=root)
    print(f"[SCENE][PASS] VideoMimic/NKSR package: {package}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--stage", choices=STAGES, default="all")
    args = parser.parse_args()
    project = args.project_root.resolve()
    config = _load_config(args.config)
    scene = config["scene"]
    print(f"[SCENE] contract accepted clip={args.clip} backend={scene['backend']} camera=fixed", flush=True)
    if args.stage == "preflight":
        return
    if scene["backend"] == "videomimic_nksr":
        _run_videomimic(config, project, args.clip)
        return
    raise RuntimeError(
        "CRISP is configured only as a fallback contract. Use backend='videomimic_nksr' "
        "for the supported fixed-camera scene path."
    )


if __name__ == "__main__":
    main()
