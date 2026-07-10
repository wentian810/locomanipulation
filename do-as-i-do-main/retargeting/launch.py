"""Retargeting pipeline: preprocess reconstruction output, then physics-optimize onto a robot hand.

Usage: python launch.py --task whisking --raw-dir ../reconstruction/whisking
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import loguru
import tyro

from retargeting.config import Config, filter_config_fields, load_config_yaml
from retargeting.pipeline.decompose_mesh import main as decompose_mesh
from retargeting.pipeline.generate_scene import main as generate_scene
from retargeting.pipeline.optimize_physics import main as optimize_physics
from retargeting.pipeline.process_dataset import main as process_dataset
from retargeting.pipeline.resolve_pedestal import resolve_scene_pedestal
from retargeting.pipeline.solve_ik import main as solve_ik

CONFIG_DIR = Path(__file__).parent / "config"

# Scene flags for the do_as_i_do dataset: the object rests on an auto-placed
# pedestal (with a welded support) rather than directly on the floor.
OBJECT_FLOOR_COLLISION = False
HAND_FLOOR_COLLISION = False
USE_PEDESTAL = True
USE_SUPPORT = True


@dataclass
class PipelineConfig:
    """raw_dir is the reconstruction output dir; task is the video name (e.g. whisking)."""

    raw_dir: str = ""
    task: str = ""
    hand_type: str = "auto"
    data_id: int = 0
    dataset_name: str = "do_as_i_do"
    robot_type: str = "sharpa"
    seed: int = 0
    wait_on_finish: bool = True
    max_sim_steps: int = 0
    force: bool = True
    smoothing: bool = True
    show_viewer: bool = True
    output_root_dir: str = "outputs"
    add_ur3_arm: bool = True


def load_mjwp_config(**overrides) -> Config:
    """Build Config from YAML defaults + dataset override + caller overrides."""
    cfg_dict = load_config_yaml(str(CONFIG_DIR / "default.yaml"))

    override_path = CONFIG_DIR / "override" / "do_as_i_do.yaml"
    if override_path.exists():
        cfg_dict.update(load_config_yaml(str(override_path)))

    cfg_dict.update(overrides)

    filtered = filter_config_fields(cfg_dict)
    if "pair_margin_range" in filtered:
        filtered["pair_margin_range"] = tuple(filtered["pair_margin_range"])
    if "xy_offset_range" in filtered:
        filtered["xy_offset_range"] = tuple(filtered["xy_offset_range"])
    filtered.pop("noise_scale", None)

    return Config(**filtered)


def run_pipeline(cfg: PipelineConfig) -> None:
    if not cfg.raw_dir:
        raise ValueError(
            "--raw-dir is required (the reconstruction pipeline's output "
            "directory, e.g. ../reconstruction/whisking)"
        )
    if not cfg.task:
        raise ValueError("--task is required (the video name, e.g. whisking)")

    # Stage 1: dataset processing
    pipeline_task = process_dataset(
        raw_dir=cfg.raw_dir,
        output_root_dir=cfg.output_root_dir,
        task=cfg.task,
        data_id=cfg.data_id,
        embodiment_type=cfg.hand_type,
        dataset_name=cfg.dataset_name,
        force=cfg.force,
    )
    if pipeline_task is None:
        loguru.logger.error(f"{cfg.dataset_name} processing failed (no task_name returned)")
        sys.exit(1)

    # Stage 2: convex decomposition
    decompose_mesh(
        task=pipeline_task,
        dataset_name=cfg.dataset_name,
        data_id=cfg.data_id,
        embodiment_type=cfg.hand_type,
        thicken=0.002,
        dilate=0.002,
        force=cfg.force,
    )

    # Stage 3: XML generation
    generate_scene(
        task=pipeline_task,
        dataset_name=cfg.dataset_name,
        data_id=cfg.data_id,
        embodiment_type=cfg.hand_type,
        robot_type=cfg.robot_type,
        show_viewer=False,
        friction_scale=1.5,
        object_floor_collision=OBJECT_FLOOR_COLLISION,
        hand_floor_collision=HAND_FLOOR_COLLISION,
        use_pedestal=USE_PEDESTAL,
        use_support=USE_SUPPORT,
        force=cfg.force,
        add_ur3_arm=cfg.add_ur3_arm,
    )

    # Stage 4: inverse kinematics (runs against the pedestal-free scene_ik.xml)
    solve_ik(
        task=pipeline_task,
        dataset_name=cfg.dataset_name,
        data_id=cfg.data_id,
        embodiment_type=cfg.hand_type,
        robot_type=cfg.robot_type,
        show_viewer=False,
        force=cfg.force,
        smoothing=cfg.smoothing,
    )

    # Load the MJWP Config (YAML defaults + dataset override) before Stage 4.5:
    # it is the single source of truth for physics/threshold params, and the
    # pedestal step (Stage 4.5) needs hand_object_distance_thresh from it —
    # otherwise it falls back to in_hand.DEFAULT_DISTANCE_THRESH.
    config = load_mjwp_config(
        dataset_name=cfg.dataset_name,
        task=pipeline_task,
        data_id=cfg.data_id,
        robot_type=cfg.robot_type,
        embodiment_type=cfg.hand_type,
        seed=cfg.seed,
        wait_on_finish=cfg.wait_on_finish,
        max_sim_steps=cfg.max_sim_steps,
        force=cfg.force,
        show_viewer=cfg.show_viewer,
    )

    # Stage 4.5: resolve scene_ik.xml -> scene.xml (+ scene_eq.xml).
    resolve_scene_pedestal(
        output_root_dir=cfg.output_root_dir,
        dataset_name=cfg.dataset_name,
        robot_type=cfg.robot_type,
        embodiment_type=cfg.hand_type,
        task=pipeline_task,
        data_id=cfg.data_id,
        use_pedestal=USE_PEDESTAL,
        use_support=USE_SUPPORT,
        hand_object_distance_thresh=config.hand_object_distance_thresh,
        force=cfg.force,
    )

    # Stage 5: physics optimization (MuJoCo Warp)
    optimize_physics(config)


if __name__ == "__main__":
    cfg = tyro.cli(PipelineConfig)
    run_pipeline(cfg)
