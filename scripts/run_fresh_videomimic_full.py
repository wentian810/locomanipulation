"""Fresh per-clip VideoMimic -> PHC -> GMR launcher with separate coordinate roots."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(command: list[str], project: Path) -> None:
    print("[FRESH_FULL][RUN] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=project, check=True)


def _under_project(project: Path, value: Path) -> str:
    value = value.resolve()
    try:
        return str(value.relative_to(project))
    except ValueError:
        return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one new clip through GVHMR, VideoMimic scene reconstruction, "
            "static-frame PHC, and GMR. Invoke once per clip because PHC's "
            "static-scene override is process-global."
        )
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--clip-filter", required=True)
    parser.add_argument("--source-output-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--source-scene-subdir",
        default="scene_reconstruction_videomimic_base",
    )
    parser.add_argument(
        "--final-scene-subdir",
        default="scene_reconstruction_semantic_chair_v5",
    )
    args = parser.parse_args()

    project = args.project_root.resolve()
    pipeline = project / "scripts" / "run_pipeline_from_config.py"
    config = args.config.resolve()
    source_root = args.source_output_root.resolve()
    output_root = args.output_root.resolve()
    if source_root == output_root:
        raise ValueError("source-output-root and output-root must be different")
    if not pipeline.is_file() or not config.is_file():
        raise FileNotFoundError("project pipeline or requested config is missing")

    common = [
        sys.executable,
        str(pipeline),
        "--project-root",
        str(project),
        "--config",
        str(config),
        "--clip-filter",
        args.clip_filter,
    ]
    source_text = _under_project(project, source_root)
    output_text = _under_project(project, output_root)

    # Phase A creates source evidence only; no historic bridge may enter it.
    phase_a = [
        "scene.robot_bridge.enabled=false",
        "scene.isaac.enabled=false",
        "scene.semantic_chair.enabled=false",
        f"scene.output_subdir={args.source_scene_subdir}",
    ]
    for stage in ("human_pre", "scene"):
        command = common + ["--output-root", source_text, "--stage", stage]
        for item in phase_a:
            command.extend(("--set", item))
        _run(command, project)

    # Phase B derives a static-coordinate PHC/GMR root from this new scene.
    phase_b = [
        "scene.robot_bridge.enabled=true",
        f"scene.robot_bridge.source_output_root={source_text}",
        f"scene.robot_bridge.source_scene_subdir={args.source_scene_subdir}",
        f"scene.semantic_chair.source_package_subdir={args.source_scene_subdir}",
        f"scene.output_subdir={args.final_scene_subdir}",
    ]
    command = common + [
        "--output-root",
        output_text,
        "--stage",
        "human_post",
        "--enable-phc",
    ]
    for item in phase_b:
        command.extend(("--set", item))
    _run(command, project)
    print("[FRESH_FULL][PASS] fresh VideoMimic + PHC + GMR pipeline completed", flush=True)


if __name__ == "__main__":
    main()
