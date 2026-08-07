#!/usr/bin/env python3
"""Configuration-driven launcher for the human + hand + object pipeline."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
VIDEO_ALIAS_KINDS = ("gvhmr", "phc", "gmr", "2x2")
VIDEO_SLUG_STOPWORDS = {
    "a",
    "an",
    "and",
    "demonstration",
    "master",
    "of",
    "the",
    "while",
}
BACKEND_WRAPPERS = {
    "hamer": "run_batch_dataset6_hamer.sh",
    "hand4wholepp": "run_batch_dataset6_hand4wholepp.sh",
    "wilor": "run_batch_dataset6_wilor.sh",
}
FORBIDDEN_CONFIG_KEYS = {
    "secret",
    "secret_id",
    "secret_key",
    "tencent_secret_id",
    "tencent_secret_key",
    "tencentcloud_secret_id",
    "tencentcloud_secret_key",
}
ENV_CONFIG_MAP = {
    "CONDA_BASE": "runtime.conda_base",
    "DATASET": "input.dataset_dir",
    "OUTPUT_BASE": "output.root",
    "CLIP_FILTER": "input.clip_filter",
    "USE_WORK_VIDEO": "input.work_video.enabled",
    "WORK_DATASET": "input.work_video.directory",
    "WORK_WIDTH": "input.work_video.width",
    "WORK_HEIGHT": "input.work_video.height",
    "WORK_CRF": "input.work_video.crf",
    "WORK_FPS": "input.work_video.fps",
    "FORCE_WORK_VIDEO": "input.work_video.force",
    "SKIP_EXISTING": "resume.skip_existing",
    "GVHMR_FORCE_HAND_PREPROCESS": "resume.force_hand_preprocess",
    "FORCE_LOCO": "resume.force_locomotion",
    "FORCE_SMOOTH": "resume.force_smoothing",
    "FORCE_PHC": "resume.force_phc",
    "GMR_OVERRIDE": "resume.force_gmr",
    "GVHMR_HAND_BACKEND": "human.backend",
    "GVHMR_HAND_CONSTRAINT_PROFILE": "human.constraint_profile",
    "GVHMR_BATCH_SIZE": "human.gvhmr_batch_size",
    "GVHMR_VITPOSE_IMG_DS": "human.vitpose_image_scale",
    "GVHMR_LOW_MEMORY": "human.low_memory",
    "GVHMR_ISOLATE_HAND_PREPROCESS": "human.isolate_hand_process",
    "GVHMR_FILTER_MANO_WRIST": "human.filters.wrist",
    "GVHMR_FILTER_MANO_TEMPORAL": "human.filters.temporal",
    "GVHMR_FILTER_MANO_FINGERS": "human.filters.fingers",
    "GVHMR_TEMPORAL_FILTER_GLOBAL_ORIENT_FILL_MODE": (
        "human.filters.global_orient_fill_mode"
    ),
    "GVHMR_FINGER_FILTER_WRIST_MODE": "human.filters.wrist_mode",
    "GVHMR_DIAGNOSE_HAND": "human.diagnostics",
    "GVHMR_HAND4WHOLEPP_CROP_TRACKING": "human.hand_crop_tracking.mode",
    "GVHMR_HAND4WHOLEPP_CROP_TRACKING_MAX_GAP": (
        "human.hand_crop_tracking.max_gap"
    ),
    "GVHMR_HAND4WHOLEPP_CROP_TRACKING_MAX_PREDICTION_GAP": (
        "human.hand_crop_tracking.max_prediction_gap"
    ),
    "GVHMR_HAND4WHOLEPP_CROP_TRACKING_DIRECT_OBSERVATION_QUALITY": (
        "human.hand_crop_tracking.direct_observation_quality"
    ),
    "GVHMR_VISIBLE_HAND_REFINE": "human.visible_hand_refine.enabled",
    "GVHMR_VISIBLE_HAND_REFINE_DEVICE": "human.visible_hand_refine.device",
    "GVHMR_VISIBLE_HAND_REFINE_BATCH_SIZE": "human.visible_hand_refine.batch_size",
    "GVHMR_VISIBLE_HAND_REFINE_STEPS": "human.visible_hand_refine.steps",
    "GVHMR_VISIBLE_HAND_REFINE_LR": "human.visible_hand_refine.lr",
    "GVHMR_VISIBLE_HAND_REFINE_PRIOR_WEIGHT": "human.visible_hand_refine.prior_weight",
    "GVHMR_VISIBLE_HAND_REFINE_FIT_CONFIDENCE": "human.visible_hand_refine.fit_confidence",
    "GVHMR_VISIBLE_HAND_REFINE_FIT_MIN_KEYPOINTS": "human.visible_hand_refine.fit_min_keypoints",
    "GVHMR_VISIBLE_HAND_REFINE_FIT_PARTITION": "human.visible_hand_refine.fit_partition",
    "GVHMR_VISIBLE_HAND_REFINE_HOLDOUT_MIN_KEYPOINTS": "human.visible_hand_refine.holdout_min_keypoints",
    "GVHMR_VISIBLE_HAND_REFINE_HOLDOUT_MAX_RELATIVE_REGRESSION_PX": (
        "human.visible_hand_refine.holdout_max_relative_regression_px"
    ),
    "GVHMR_VISIBLE_HAND_REFINE_MAX_DELTA_DEGREES": "human.visible_hand_refine.max_delta_degrees",
    "GVHMR_VISIBLE_HAND_REFINE_MIN_RELATIVE_IMPROVEMENT": "human.visible_hand_refine.min_relative_improvement",
    "GVHMR_VISIBLE_HAND_REFINE_MIN_RELATIVE_IMPROVEMENT_PX": "human.visible_hand_refine.min_relative_improvement_px",
    "GVHMR_VISIBLE_HAND_REFINE_MAX_ABSOLUTE_REGRESSION_PX": "human.visible_hand_refine.max_absolute_regression_px",
    "GVHMR_VISIBLE_HAND_REFINE_MAX_ANCHOR_ERROR_PX": "human.visible_hand_refine.max_anchor_error_px",
    "GVHMR_VISIBLE_HAND_REFINE_MAX_ANCHOR_ERROR_BBOX_RATIO": "human.visible_hand_refine.max_anchor_error_bbox_ratio",
    "GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_HAND_CONFIDENCE": "human.visible_hand_refine.evidence_hand_confidence",
    "GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_MIN_KEYPOINTS": "human.visible_hand_refine.evidence_min_keypoints",
    "GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_MEAN_CONFIDENCE": "human.visible_hand_refine.evidence_mean_confidence",
    "GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_WRIST_CONFIDENCE": "human.visible_hand_refine.evidence_wrist_confidence",
    "GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_MIN_BBOX_DIAGONAL_PX": "human.visible_hand_refine.evidence_min_bbox_diagonal_px",
    "GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_MIN_RUN": "human.visible_hand_refine.evidence_min_run",
    "RUN_GMR": "gmr.enabled",
    "GMR_HAND_MODEL": "gmr.hand_model",
    "GMR_SOURCE": "gmr.source",
    "GMR_TARGET_FPS": "gmr.target_fps",
    "GMR_HUMAN_YAW_OFFSET_DEG": "gmr.human_yaw_offset_deg",
    "GMR_CAMERA_SOURCE": "gmr.camera_source",
    "GMR_CAMERA_SUBJECT_ALIGN_XY": "gmr.camera_subject_align_xy",
    "GMR_RENDER": "gmr.render",
    "GMR_COMPOSITE": "gmr.composite_2x2",
    "GMR_MUJOCO_GL": "gmr.mujoco_gl",
    "GMR_RENDER_WIDTH": "gmr.render_width",
    "GMR_RENDER_HEIGHT": "gmr.render_height",
    "OBJECT_ENABLED": "object.enabled",
    "OBJECT_MODE": "object.mode",
    "HUNYUAN3D_FACE_COUNT": "object.monocular.hunyuan_face_count",
    "HUNYUAN3D_TIMEOUT": "object.monocular.hunyuan_timeout",
    "HUNYUAN3D_REGION": "object.monocular.hunyuan_region",
    "MOGE_RESOLUTION_LEVEL": "object.monocular.moge_resolution_level",
    "MEGAPOSE_HYPOTHESES": "object.monocular.megapose.pose_hypotheses",
    "MEGAPOSE_COARSE_GRID_STRIDE": (
        "object.monocular.megapose.coarse_grid_stride"
    ),
    "MEGAPOSE_REFINER_ITERATIONS": (
        "object.monocular.megapose.refiner_iterations"
    ),
    "MEGAPOSE_TEMPORAL_ITERATIONS": (
        "object.monocular.megapose.temporal_refiner_iterations"
    ),
    "MEGAPOSE_REINIT_INTERVAL": (
        "object.monocular.megapose.reinit_interval"
    ),
    "MEGAPOSE_MIN_IOU": "object.monocular.megapose.min_iou",
    "MEGAPOSE_REINIT_IOU": "object.monocular.megapose.reinit_iou",
    "MEGAPOSE_MAX_INTERPOLATION_GAP": (
        "object.monocular.megapose.max_interpolation_gap"
    ),
    "OBJECT_NAME": "object.runtime_clip_overrides.object_name",
    "OBJECT_REFERENCE_FRAME": "object.runtime_clip_overrides.reference_frame",
    "OBJECT_DYNAMIC_RELEASE_FRAME": (
        "object.runtime_clip_overrides.dynamic_release_frame"
    ),
    "PIPELINE_PYTHON": "runtime.python",
    "CUDA_VISIBLE_DEVICES": "runtime.cuda_visible_devices",
    "SEGMENTATION_DISPLAY": "runtime.segmentation_display",
    "ENV_SAMHQ": "runtime.conda_envs.samhq",
    "ENV_MOGE2": "runtime.conda_envs.moge2",
    "ENV_MEGAPOSE": "runtime.conda_envs.megapose",
    "ENV_CRISP": "runtime.conda_envs.crisp",
    "SCENE_ENABLED": "scene.enabled",
    "SCENE_REQUIRED": "scene.required",
    "SCENE_BACKEND": "scene.backend",
    "SCENE_WORK_ROOT": "scene.work_root",
    "SCENE_OUTPUT_SUBDIR": "scene.output_subdir",
    "CRISP_ROOT": "scene.crisp.root",
    "CRISP_STATIC_CAMERA": "scene.crisp.static_camera",
    "CRISP_RUN_CONTACT_PASS": "scene.crisp.run_contact_pass",
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return value


def _get_path(config: dict, dotted_path: str, default: Any = None) -> Any:
    value: Any = config
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _set_path(config: dict, dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    target = config
    for part in parts[:-1]:
        current = target.get(part)
        if current is None:
            current = {}
            target[part] = current
        if not isinstance(current, dict):
            raise ValueError(
                f"cannot set {dotted_path}: {part} is not a mapping"
            )
        target = current
    target[parts[-1]] = value


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected boolean environment value, got {value!r}")


def _coerce_environment_value(raw: str, current: Any) -> Any:
    if isinstance(current, bool):
        return _parse_bool(raw)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        parsed = yaml.safe_load(raw)
        return parsed if isinstance(parsed, list) else raw.split(",")
    return raw


def _parse_cli_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid --set value {raw!r}: {exc}") from exc


def _reject_embedded_secrets(value: Any, path: str = "") -> None:
    if not isinstance(value, dict):
        if isinstance(value, list):
            for index, item in enumerate(value):
                _reject_embedded_secrets(item, f"{path}[{index}]")
        return
    for key, item in value.items():
        key_lower = str(key).lower()
        if key_lower in FORBIDDEN_CONFIG_KEYS or (
            "secret" in key_lower and key_lower not in {"secret_env"}
        ):
            raise ValueError(
                f"cloud credential field is forbidden in config: {path}{key}. "
                "Use process environment variables instead."
            )
        _reject_embedded_secrets(item, f"{path}{key}.")


def _bool_env(value: Any) -> str:
    return "1" if bool(value) else "0"


def _validate_visible_hand_refine_config(value: Any) -> dict[str, Any]:
    """Fail closed for the high-evidence wrist-refinement configuration.

    This stage changes final rendered wrist rotations, so a quoted YAML boolean
    such as ``enabled: \"false\"`` must not silently become truthy via Python's
    normal ``bool(str)`` conversion.  Keep validation here, next to the config
    runner, rather than duplicating it in the shell wrapper.
    """
    if not isinstance(value, dict):
        raise ValueError("human.visible_hand_refine must be a mapping")

    enabled = value.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("human.visible_hand_refine.enabled must be a boolean")

    device = value.get("device", "auto")
    if not isinstance(device, str) or device.lower() not in {"auto", "cuda", "cpu"}:
        raise ValueError(
            "human.visible_hand_refine.device must be 'auto', 'cuda', or 'cpu'"
        )
    fit_partition = value.get("fit_partition", "all")
    if not isinstance(fit_partition, str) or fit_partition.lower() not in {
        "all",
        "non_tip",
    }:
        raise ValueError(
            "human.visible_hand_refine.fit_partition must be 'all' or 'non_tip'"
        )

    def finite_number(
        key: str,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
        strict_minimum: bool = False,
    ) -> None:
        if key not in value:
            return
        number = value[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise ValueError(f"human.visible_hand_refine.{key} must be a number")
        numeric = float(number)
        if not math.isfinite(numeric):
            raise ValueError(f"human.visible_hand_refine.{key} must be finite")
        if minimum is not None and (
            numeric < minimum or (strict_minimum and numeric <= minimum)
        ):
            comparator = ">" if strict_minimum else ">="
            raise ValueError(
                f"human.visible_hand_refine.{key} must be {comparator} {minimum}"
            )
        if maximum is not None and numeric > maximum:
            raise ValueError(
                f"human.visible_hand_refine.{key} must be <= {maximum}"
            )

    def bounded_integer(
        key: str,
        *,
        minimum: int,
        maximum: int | None = None,
    ) -> None:
        if key not in value:
            return
        number = value[key]
        if isinstance(number, bool) or not isinstance(number, int):
            raise ValueError(f"human.visible_hand_refine.{key} must be an integer")
        if number < minimum or (maximum is not None and number > maximum):
            upper = f" and <= {maximum}" if maximum is not None else ""
            raise ValueError(
                f"human.visible_hand_refine.{key} must be >= {minimum}{upper}"
            )

    bounded_integer("batch_size", minimum=1)
    bounded_integer("steps", minimum=0)
    bounded_integer("fit_min_keypoints", minimum=1, maximum=21)
    bounded_integer("holdout_min_keypoints", minimum=1, maximum=5)
    bounded_integer("evidence_min_keypoints", minimum=1, maximum=21)
    bounded_integer("evidence_min_run", minimum=1)
    if (
        fit_partition.lower() == "non_tip"
        and int(value.get("fit_min_keypoints", 12)) > 16
    ):
        raise ValueError(
            "human.visible_hand_refine.fit_min_keypoints must be <= 16 "
            "when fit_partition='non_tip'"
        )

    finite_number("lr", minimum=0.0, strict_minimum=True)
    finite_number("prior_weight", minimum=0.0)
    finite_number("fit_confidence", minimum=0.0, maximum=1.0)
    finite_number("holdout_max_relative_regression_px", minimum=0.0)
    finite_number(
        "max_delta_degrees", minimum=0.0, maximum=180.0, strict_minimum=True
    )
    finite_number("min_relative_improvement", minimum=0.0, maximum=1.0)
    finite_number("min_relative_improvement_px", minimum=0.0)
    finite_number("max_absolute_regression_px", minimum=0.0)
    finite_number("max_anchor_error_px", minimum=0.0, strict_minimum=True)
    finite_number(
        "max_anchor_error_bbox_ratio", minimum=0.0, strict_minimum=True
    )
    finite_number("evidence_hand_confidence", minimum=0.0, maximum=1.0)
    finite_number("evidence_mean_confidence", minimum=0.0, maximum=1.0)
    finite_number("evidence_wrist_confidence", minimum=0.0, maximum=1.0)
    finite_number(
        "evidence_min_bbox_diagonal_px", minimum=0.0, strict_minimum=True
    )
    return value


def _validate_scene_config(value: Any) -> dict[str, Any]:
    """Validate the stable fixed-camera scene-sidecar interface.

    Scene reconstruction affects future physics and product contracts, so this
    validator fails closed on accidental YAML coercions and unsupported modes.
    It deliberately validates configuration only; repository/layout checks are
    performed in :meth:`PipelineRunner._check_scene_preflight`.
    """
    if not isinstance(value, dict):
        raise ValueError("scene must be a mapping")

    def required_bool(path: str, container: dict[str, Any], default: bool) -> bool:
        item = container.get(path, default)
        if not isinstance(item, bool):
            raise ValueError(f"scene.{path} must be a boolean")
        return item

    def required_mapping(path: str) -> dict[str, Any]:
        item = value.get(path, {})
        if not isinstance(item, dict):
            raise ValueError(f"scene.{path} must be a mapping")
        return item

    def nonempty_string(label: str, item: Any) -> str:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"scene.{label} must be a non-empty string")
        return item

    def finite_number(
        label: str,
        item: Any,
        *,
        minimum: float = 0.0,
        strict_minimum: bool = False,
    ) -> float:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"scene.{label} must be a number")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"scene.{label} must be finite")
        if number < minimum or (strict_minimum and number <= minimum):
            comparator = ">" if strict_minimum else ">="
            raise ValueError(f"scene.{label} must be {comparator} {minimum}")
        return number

    enabled = required_bool("enabled", value, False)
    required = required_bool("required", value, False)
    if required and not enabled:
        raise ValueError("scene.required cannot be true when scene.enabled is false")
    backend = value.get("backend", "crisp")
    if backend not in {"crisp", "videomimic_nksr"}:
        raise ValueError("scene.backend must be 'crisp' or 'videomimic_nksr'")
    if value.get("source_motion", "smoothed") not in {
        "converted",
        "smoothed",
        "final",
    }:
        raise ValueError("scene.source_motion must be converted, smoothed, or final")
    if value.get("physics_motion", "final") not in {"smoothed", "final"}:
        raise ValueError("scene.physics_motion must be smoothed or final")
    for label, default in (
        ("output_subdir", "scene_reconstruction"),
        ("work_root", "scene_work/crisp" if backend == "crisp" else "scene_work/videomimic"),
    ):
        path_text = nonempty_string(label, value.get(label, default))
        path = Path(path_text)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"scene.{label} must be a project-relative safe path")

    camera = required_mapping("camera")
    if camera.get("mode", "fixed") != "fixed":
        raise ValueError("scene.camera.mode must be 'fixed' for the first release")
    if camera.get("source", "gvhmr") != "gvhmr":
        raise ValueError("scene.camera.source must be 'gvhmr'")
    for name, default in (
        ("validate_static", True),
        ("retain_raw_trajectory", True),
    ):
        item = camera.get(name, default)
        if not isinstance(item, bool):
            raise ValueError(f"scene.camera.{name} must be a boolean")
    if camera.get("freeze_method", "robust_median") not in {
        "reference_frame",
        "robust_median",
    }:
        raise ValueError(
            "scene.camera.freeze_method must be reference_frame or robust_median"
        )
    if camera.get("canonicalization", "static_optimized") not in {
        "static_exact_reprojection",
        "static_optimized",
    }:
        raise ValueError(
            "scene.camera.canonicalization must be static_exact_reprojection "
            "or static_optimized"
        )

    external_human = required_mapping("external_human")
    if external_human.get("model_type", "smpl") != "smpl":
        raise ValueError("scene.external_human.model_type must be 'smpl'")
    if external_human.get("units", "meter") != "meter":
        raise ValueError("scene.external_human.units must be 'meter'")
    for name in ("include_vertices", "include_joints", "include_faces"):
        item = external_human.get(name, True)
        if not isinstance(item, bool):
            raise ValueError(f"scene.external_human.{name} must be a boolean")
        if not item:
            raise ValueError(f"scene.external_human.{name} must be true")

    if backend == "crisp":
        crisp = required_mapping("crisp")
        nonempty_string("crisp.root", crisp.get("root", "external/CRISP-Real2Sim"))
        nonempty_string("crisp.conda_env", crisp.get("conda_env", "crisp"))
        if crisp.get("hmr_type", "external") != "external":
            raise ValueError("scene.crisp.hmr_type must be 'external'")
        for name, default in (
            ("run_mask", True),
            ("run_mogesam", True),
            ("run_ufm", True),
            ("run_contact_pass", False),
            ("run_nksr", False),
            ("static_camera", True),
        ):
            item = crisp.get(name, default)
            if not isinstance(item, bool):
                raise ValueError(f"scene.crisp.{name} must be a boolean")
        if crisp.get("run_contact_pass", False):
            raise ValueError(
                "scene.crisp.run_contact_pass is unavailable until contact-guided "
                "scene reconstruction is deployed"
            )
        if not crisp.get("static_camera", True):
            raise ValueError("scene.crisp.static_camera must be true")
    else:
        videomimic = required_mapping("videomimic")
        nonempty_string("videomimic.root", videomimic.get("root", "external/VideoMimic/real2sim"))
        for name, default in (("recon_env", "vm1recon"), ("human_env", "vm1rs"), ("sam2_env", "vm1rs")):
            nonempty_string(f"videomimic.{name}", videomimic.get(name, default))
        if videomimic.get("gender", "auto") not in {"auto", "male", "female", "neutral"}:
            raise ValueError("scene.videomimic.gender must be auto, male, female or neutral")
        if videomimic.get("human_mask_mode", "sam2") not in {"sam2", "bbox"}:
            raise ValueError("scene.videomimic.human_mask_mode must be 'sam2' or 'bbox'")
        max_frames = videomimic.get("max_frames", 0)
        if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 0:
            raise ValueError("scene.videomimic.max_frames must be a non-negative integer")
        frame_stride = videomimic.get("frame_stride", 1)
        if isinstance(frame_stride, bool) or not isinstance(frame_stride, int) or frame_stride <= 0:
            raise ValueError("scene.videomimic.frame_stride must be a positive integer")
        if videomimic.get("meshification_method", "nksr") != "nksr":
            raise ValueError("scene.videomimic.meshification_method must be 'nksr'")
        stage4 = videomimic.get("stage4", {})
        if not isinstance(stage4, dict):
            raise ValueError("scene.videomimic.stage4 must be a mapping")
        stage4_enabled = stage4.get("enabled", False)
        if not isinstance(stage4_enabled, bool):
            raise ValueError("scene.videomimic.stage4.enabled must be a boolean")
        if stage4_enabled:
            nonempty_string("videomimic.stage4.env", stage4.get("env", videomimic.get("human_env", "vm1rs")))
            coverage = finite_number("videomimic.stage4.minimum_contact_coverage", stage4.get("minimum_contact_coverage", 0.75), minimum=0.0, strict_minimum=True)
            if coverage > 1.0:
                raise ValueError("scene.videomimic.stage4.minimum_contact_coverage must be <= 1")

    physics = required_mapping("physics")
    physics_enabled = physics.get("enabled", False)
    if not isinstance(physics_enabled, bool):
        raise ValueError("scene.physics.enabled must be a boolean")
    if physics_enabled:
        if backend != "videomimic_nksr":
            raise ValueError("scene.physics.enabled requires scene.backend='videomimic_nksr'")
        if physics.get("collision_source") != "first_round_nksr_plus_measured_support_patch":
            raise ValueError("scene.physics.collision_source must use observed first-round NKSR evidence")
        if physics.get("visual_source") != "cleaned_filled_nksr":
            raise ValueError("scene.physics.visual_source must use the cleaned filled NKSR mesh")
        if physics.get("root_trajectory_adjustment", "none") != "none":
            raise ValueError("scene.physics.root_trajectory_adjustment must be 'none'")
        if not isinstance(physics.get("require_mjstep_pass", True), bool):
            raise ValueError("scene.physics.require_mjstep_pass must be a boolean")
        for label, default, minimum, strict in (("render_width", 960, 1.0, True), ("render_height", 720, 1.0, True), ("mj_step_substeps", 8, 1.0, True), ("max_negative_contact_distance_m", 0.012, 0.0, False)):
            number = finite_number(label, physics.get(label, default), minimum=minimum, strict_minimum=strict)
            if label != "max_negative_contact_distance_m" and not number.is_integer():
                raise ValueError(f"scene.physics.{label} must be an integer")
        if float(physics.get("max_negative_contact_distance_m", 0.012)) > 0.03:
            raise ValueError("scene.physics.max_negative_contact_distance_m must be <= 0.03")

    robot_bridge = required_mapping("robot_bridge")
    bridge_enabled = robot_bridge.get("enabled", False)
    if not isinstance(bridge_enabled, bool):
        raise ValueError("scene.robot_bridge.enabled must be a boolean")
    if bridge_enabled:
        if backend != "videomimic_nksr":
            raise ValueError(
                "scene.robot_bridge is available only for videomimic_nksr"
            )
        source_root = nonempty_string(
            "robot_bridge.source_output_root",
            robot_bridge.get("source_output_root", ""),
        )
        source_path = Path(source_root)
        if source_path.is_absolute() or ".." in source_path.parts:
            raise ValueError(
                "scene.robot_bridge.source_output_root must be a project-relative safe path"
            )
        if robot_bridge.get("motion_name", "human_motion_static_hard.npz") != (
            "human_motion_static_hard.npz"
        ):
            raise ValueError(
                "scene.robot_bridge.motion_name must be human_motion_static_hard.npz"
            )
        if robot_bridge.get("camera_name", "gvhmr_camera_static_hard.npz") != (
            "gvhmr_camera_static_hard.npz"
        ):
            raise ValueError(
                "scene.robot_bridge.camera_name must be gvhmr_camera_static_hard.npz"
            )

    isaac = required_mapping("isaac")
    isaac_enabled = isaac.get("enabled", False)
    if not isinstance(isaac_enabled, bool):
        raise ValueError("scene.isaac.enabled must be a boolean")
    if isaac_enabled:
        collision_mode = isaac.get("collision_mode", "raw_mesh_and_support")
        if collision_mode not in {
            "raw_mesh_and_support",
            "seat_support_only",
            "semantic_primitives_only",
            "semantic_seat_support_only",
            "semantic_seat_backrest_support",
            "semantic_mesh_and_support",
        }:
            raise ValueError(
                "scene.isaac.collision_mode must be one of "
                "raw_mesh_and_support, seat_support_only, "
                "semantic_primitives_only, semantic_seat_support_only, semantic_seat_backrest_support, "
                "semantic_mesh_and_support"
            )
        if not bridge_enabled:
            raise ValueError(
                "scene.isaac.enabled requires scene.robot_bridge.enabled so "
                "the PHC motion and scene share the static-camera frame"
            )
        if backend != "videomimic_nksr":
            raise ValueError(
                "scene.isaac.enabled is currently available only for videomimic_nksr"
            )
        semantic_preflight = isaac.get("semantic_preflight_report", "")
        if not isinstance(semantic_preflight, str):
            raise ValueError("scene.isaac.semantic_preflight_report must be a string")
        if semantic_preflight:
            preflight_path = Path(semantic_preflight)
            if preflight_path.is_absolute() or ".." in preflight_path.parts:
                raise ValueError(
                    "scene.isaac.semantic_preflight_report must be a project-relative safe path"
                )
        asset_contract = isaac.get("scene_asset_contract_report", "")
        if not isinstance(asset_contract, str):
            raise ValueError("scene.isaac.scene_asset_contract_report must be a string")
        if asset_contract:
            asset_contract_path = Path(asset_contract)
            if asset_contract_path.is_absolute() or ".." in asset_contract_path.parts:
                raise ValueError(
                    "scene.isaac.scene_asset_contract_report must be a project-relative safe path"
                )

    semantic_chair = required_mapping("semantic_chair")
    semantic_chair_enabled = semantic_chair.get("enabled", False)
    if not isinstance(semantic_chair_enabled, bool):
        raise ValueError("scene.semantic_chair.enabled must be a boolean")
    if semantic_chair_enabled:
        layout_mode = semantic_chair.get("layout_mode", "auto_seated_body")
        if layout_mode not in {"auto_seated_body", "manual"}:
            raise ValueError(
                "scene.semantic_chair.layout_mode must be 'auto_seated_body' or 'manual'"
            )
        if layout_mode == "manual":
            back_sign = semantic_chair.get("back_sign", 1)
            if isinstance(back_sign, bool) or back_sign not in {-1, 1}:
                raise ValueError("scene.semantic_chair.back_sign must be -1 or 1")
            if not isinstance(semantic_chair.get("swap_horizontal_axes", False), bool):
                raise ValueError("scene.semantic_chair.swap_horizontal_axes must be a boolean")
        if not isinstance(semantic_chair.get("level_seat", True), bool):
            raise ValueError("scene.semantic_chair.level_seat must be a boolean")
        if not isinstance(semantic_chair.get("phc_backrest_collision", False), bool):
            raise ValueError("scene.semantic_chair.phc_backrest_collision must be a boolean")

    collision = required_mapping("collision")
    for name, default in (
        ("export_obj", True),
        ("export_urdf", True),
        ("export_mujoco", True),
        ("prefer_box_primitives", True),
        ("allow_mesh_fallback", True),
    ):
        item = collision.get(name, default)
        if not isinstance(item, bool):
            raise ValueError(f"scene.collision.{name} must be a boolean")

    quality = required_mapping("quality")
    for name, default in (("enabled", True), ("required_for_product", False)):
        item = quality.get(name, default)
        if not isinstance(item, bool):
            raise ValueError(f"scene.quality.{name} must be a boolean")
    warn = finite_number(
        "quality.max_primitives_warn", quality.get("max_primitives_warn", 80)
    )
    fail = finite_number(
        "quality.max_primitives_fail", quality.get("max_primitives_fail", 150)
    )
    if not warn.is_integer() or not fail.is_integer() or warn > fail:
        raise ValueError(
            "scene.quality primitive thresholds must be integer-valued and warn <= fail"
        )
    finite_number(
        "quality.contact_distance_warn_m",
        quality.get("contact_distance_warn_m", 0.05),
        strict_minimum=True,
    )
    finite_number(
        "quality.penetration_p95_fail_m",
        quality.get("penetration_p95_fail_m", 0.03),
        strict_minimum=True,
    )

    runtime = required_mapping("runtime")
    gpu = runtime.get("gpu", 0)
    if gpu != "auto" and (isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0):
        raise ValueError("scene.runtime.gpu must be 'auto' or a non-negative integer")
    for name, default in (("max_parallel_clips", 1), ("timeout_seconds", 3600)):
        item = runtime.get(name, default)
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"scene.runtime.{name} must be a positive integer")
    if int(runtime.get("max_parallel_clips", 1)) != 1:
        raise ValueError("scene.runtime.max_parallel_clips must be 1 in the first release")
    return value


def _env_value(value: Any) -> str:
    if isinstance(value, bool):
        return _bool_env(value)
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def _clip_slug(name: str, max_len: int = 64) -> str:
    text = name.strip().lower()
    text = re.sub(r"clip(\d+)", r"c\1", text)
    text = re.sub(r"form(\d+)", r"f\1", text)
    tokens = re.split(r"[^a-z0-9]+", text)
    tokens = [
        token for token in tokens
        if token and token not in VIDEO_SLUG_STOPWORDS
    ]
    slug = "_".join(tokens) or "clip"
    if len(slug) > max_len:
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
        slug = f"{slug[: max_len - 9].rstrip('_')}_{digest}"
    return slug


def _first_existing(paths: list[Path | None]) -> Path | None:
    for path in paths:
        if path is not None and path.is_file():
            return path
    return None


def _first_matching(directory: Path, pattern: str) -> Path | None:
    matches = sorted(path for path in directory.glob(pattern) if path.is_file())
    return matches[0] if matches else None


def _latest_mp4(directory: Path) -> Path | None:
    if not directory.is_dir():
        return None
    videos = [path for path in directory.glob("*.mp4") if path.is_file()]
    if not videos:
        return None
    return max(videos, key=lambda path: path.stat().st_mtime)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and source.resolve() == target.resolve():
        return
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


class PipelineRunner:
    def __init__(
        self,
        default_config_path: Path,
        config_path: Path,
        cli_overrides: list[tuple[str, Any]] | None = None,
        dry_run: bool = False,
    ):
        self.default_config_path = default_config_path.resolve()
        self.config_path = config_path.resolve()
        default_config = _load_yaml(self.default_config_path)
        pipeline_config = _load_yaml(self.config_path)
        config = _deep_merge(default_config, pipeline_config)

        if "PIPELINE_ROOT" in os.environ:
            root = Path(os.environ["PIPELINE_ROOT"]).expanduser().resolve()
        else:
            if "project_root" in pipeline_config:
                project_base = self.config_path.parent
                project_value = pipeline_config["project_root"]
            else:
                project_base = self.default_config_path.parent
                project_value = default_config.get("project_root", "..")
            root_path = Path(os.path.expandvars(str(project_value))).expanduser()
            root = (
                root_path.resolve()
                if root_path.is_absolute()
                else (project_base / root_path).resolve()
            )
        config["project_root"] = str(root)

        self._apply_environment_config_overrides(config)
        for path, value in cli_overrides or []:
            _set_path(config, path, value)

        self.config = _expand(config)
        _reject_embedded_secrets(self.config)
        if int(self.config.get("schema_version", 0)) != 1:
            raise ValueError("only schema_version: 1 is supported")
        self.root = Path(self.config["project_root"]).resolve()
        self.dry_run = dry_run
        self.runtime_python = str(
            self._path(
                self.config.get("runtime", {}).get("python", sys.executable)
            )
        )
        self.environment = self._build_environment()
        # None means no admission decision has been made yet.  An empty set is
        # a valid result: all selected clips were excluded before expensive
        # human-motion inference.
        self._eligible_clips: set[str] | None = None
        self._video_duration_cache: dict[Path, float | None] = {}
        self._human_stage_failures: list[dict[str, Any]] = []
        self._run_id = f"run-{os.urandom(12).hex()}"
        self._validate_cross_stage_contracts()

    def _validate_cross_stage_contracts(self) -> None:
        """Reject combinations that cannot produce one coherent run.

        These checks intentionally do not inspect output files.  They must be
        reachable from ``--check`` before PHC, GMR, or a bridge has written a
        partial artifact tree.
        """
        scene = self._scene_config()
        gmr = self.config.get("gmr", {})
        input_cfg = self.config.get("input", {})
        if not isinstance(gmr, dict):
            raise ValueError("gmr must be a mapping")
        if bool(gmr.get("enabled", True)) and gmr.get("source", "final") != "final":
            raise ValueError(
                "production GMR must use gmr.source='final'; PHC routing belongs "
                "to 001_final.npz, not a renderer-side source choice"
            )

        physics_enabled = bool(scene.get("physics", {}).get("enabled", False))
        bridge_enabled = bool(scene.get("robot_bridge", {}).get("enabled", False))
        isaac_enabled = bool(scene.get("isaac", {}).get("enabled", False))
        if physics_enabled and bridge_enabled:
            raise ValueError(
                "scene.physics.enabled and scene.robot_bridge.enabled cannot be "
                "combined; use separate source/PHC and frozen-result audit configs"
            )
        if isaac_enabled:
            max_videos = input_cfg.get("max_videos")
            clip_filter = str(input_cfg.get("clip_filter", "")).strip()
            configured_clips = [item for item in clip_filter.split(",") if item.strip()]
            if max_videos != 1 or len(configured_clips) != 1:
                raise ValueError(
                    "scene.isaac.enabled requires input.max_videos=1 and exactly "
                    "one input.clip_filter entry"
                )

        scene_xml_name = str(gmr.get("scene_mujoco_name", "")).strip()
        scene_render_uses_xml = bool(gmr.get("enabled", True)) and bool(
            gmr.get("render", True)
        ) and (bool(scene.get("enabled", False)) or bool(scene_xml_name))
        if scene_render_uses_xml and not scene_xml_name:
            raise ValueError(
                "scene-enabled GMR rendering requires an explicit "
                "gmr.scene_mujoco_name; refusing to silently omit the "
                "reconstructed scene"
            )
        if scene_render_uses_xml and bool(gmr.get("camera_subject_align_xy", False)):
            raise ValueError(
                "GMR camera_subject_align_xy must be false when a scene XML is "
                "rendered; camera alignment must not rewrite robot root XY"
            )

    def _git_run_metadata(self) -> dict[str, Any]:
        """Capture the repository state once for an auditable output run."""
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=self.root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {
            "git_commit": commit,
            "git_dirty": bool(status.strip()),
            "git_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        }

    def _ensure_run_manifests(
        self, videos: list[Path], *, allow_existing_clip: bool = False
    ) -> None:
        """Create one immutable provenance record per fresh output clip.

        A legacy non-empty directory without a manifest is never silently
        adopted.  The only exception is the immediately preceding, whitelist
        bridge materialization, which is constructed by this runner and then
        recorded before PHC or GMR starts.
        """
        if self.dry_run:
            return
        config_text = json.dumps(
            self.config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        config_sha256 = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
        git = self._git_run_metadata()
        output_root = self._path(self.config["output"]["root"])
        scene = self._scene_config()
        robot_xml = (
            self.root / "GMR-master" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
        )
        for video in videos:
            clip_root = output_root / video.stem
            manifest_path = clip_root / "run_manifest.json"
            source_video_sha256 = self._sha256_file(video)
            if manifest_path.is_file():
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                expected = {
                    "config_sha256": config_sha256,
                    "source_video_sha256": source_video_sha256,
                    "git_commit": git["git_commit"],
                    "git_status_sha256": git["git_status_sha256"],
                }
                mismatched = [
                    key for key, value in expected.items() if existing.get(key) != value
                ]
                if mismatched:
                    raise RuntimeError(
                        f"{video.stem}: existing run manifest is not this run "
                        f"({', '.join(mismatched)}); use a fresh output.root"
                    )
                continue
            if clip_root.exists() and any(clip_root.iterdir()) and not allow_existing_clip:
                raise RuntimeError(
                    f"{video.stem}: refusing to adopt legacy output without "
                    f"run_manifest.json: {clip_root}; use a fresh output.root"
                )
            clip_root.mkdir(parents=True, exist_ok=True)
            scene_manifest = (
                clip_root
                / str(scene.get("output_subdir", "scene_reconstruction"))
                / "scene_manifest.json"
            )
            values = {
                "schema_version": 1,
                "run_id": self._run_id,
                "config_sha256": config_sha256,
                **git,
                "source_video": str(video.resolve()),
                "source_video_sha256": source_video_sha256,
                "motion_input_sha256": self._sha256_file(clip_root / "001_smoothed.npz")
                if (clip_root / "001_smoothed.npz").is_file()
                else None,
                "camera_input_sha256": self._sha256_file(clip_root / "gvhmr_camera.npz")
                if (clip_root / "gvhmr_camera.npz").is_file()
                else None,
                "scene_manifest_sha256": self._sha256_file(scene_manifest)
                if scene_manifest.is_file()
                else None,
                "robot_xml_sha256": self._sha256_file(robot_xml)
                if robot_xml.is_file()
                else None,
                "robot_motion_sha256": self._sha256_file(clip_root / "robot_motion.pkl")
                if (clip_root / "robot_motion.pkl").is_file()
                else None,
                "parent_run_id": None,
            }
            temporary = manifest_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(values, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(manifest_path)

    @staticmethod
    def _apply_environment_config_overrides(config: dict) -> None:
        # Declared advanced environment values also participate in the same
        # priority chain. The real process environment wins over YAML.
        advanced = config.setdefault("environment", {})
        for key in list(advanced):
            if key in os.environ:
                advanced[key] = os.environ[key]

        for env_name, config_path in ENV_CONFIG_MAP.items():
            if env_name not in os.environ:
                continue
            current = _get_path(config, config_path)
            _set_path(
                config,
                config_path,
                _coerce_environment_value(os.environ[env_name], current),
            )
        backend = str(_get_path(config, "human.backend", "hand4wholepp"))
        backend_batch_env = (
            "GVHMR_HAND4WHOLEPP_BATCH_SIZE"
            if backend == "hand4wholepp"
            else "GVHMR_HAMER_BATCH_SIZE"
        )
        batch_env = (
            "PIPELINE_HAND_BATCH_SIZE"
            if "PIPELINE_HAND_BATCH_SIZE" in os.environ
            else backend_batch_env
        )
        if batch_env in os.environ:
            current = _get_path(config, "human.hand_batch_size")
            _set_path(
                config,
                "human.hand_batch_size",
                _coerce_environment_value(os.environ[batch_env], current),
            )
        if "SKIP_PHC" in os.environ:
            _set_path(
                config,
                "phc.enabled",
                not _parse_bool(os.environ["SKIP_PHC"]),
            )

    def _path(self, value: str | Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.root / path
        return path.resolve()

    def _build_environment(self) -> dict[str, str]:
        cfg = self.config
        input_cfg = cfg.get("input", {})
        work_cfg = input_cfg.get("work_video", {})
        output_cfg = cfg.get("output", {})
        resume = cfg.get("resume", {})
        human = cfg.get("human", {})
        filters = human.get("filters", {})
        if not isinstance(filters, dict):
            raise ValueError("human.filters must be a mapping")
        global_orient_fill_mode = str(
            filters.get("global_orient_fill_mode", "interpolate")
        ).lower()
        if global_orient_fill_mode not in {"interpolate", "preserve"}:
            raise ValueError(
                "human.filters.global_orient_fill_mode must be "
                "'interpolate' or 'preserve'"
            )
        wrist_mode = str(filters.get("wrist_mode", "smooth")).lower()
        if wrist_mode not in {"smooth", "preserve"}:
            raise ValueError(
                "human.filters.wrist_mode must be 'smooth' or 'preserve'"
            )
        hand_crop_tracking = human.get("hand_crop_tracking", {})
        if not isinstance(hand_crop_tracking, dict):
            raise ValueError("human.hand_crop_tracking must be a mapping")
        hand_crop_tracking_mode = hand_crop_tracking.get("mode", "off")
        # PyYAML 1.1 treats an unquoted `off` as False. Keep old hand-written
        # YAML files safe while requiring every other value to be explicit.
        if hand_crop_tracking_mode is False:
            hand_crop_tracking_mode = "off"
        visible_hand_refine = _validate_visible_hand_refine_config(
            human.get("visible_hand_refine", {})
        )
        visible_refine_device = str(visible_hand_refine.get("device", "auto")).lower()
        locomotion = cfg.get("locomotion", {})
        phc = cfg.get("phc", {})
        gmr = cfg.get("gmr", {})
        object_cfg = cfg.get("object", {})
        monocular = object_cfg.get("monocular", {})
        scene = _validate_scene_config(cfg.get("scene", {}))
        scene_crisp = scene.get("crisp", {})
        runtime = cfg.get("runtime", {})

        try:
            work_fps = float(work_cfg.get("fps", 30))
        except (TypeError, ValueError) as exc:
            raise ValueError("input.work_video.fps must be a positive number") from exc
        if work_fps <= 0:
            raise ValueError("input.work_video.fps must be a positive number")
        if abs(work_fps - 30.0) > 1e-6:
            raise ValueError(
                "input.work_video.fps must be 30 until PHC, GMR, and motion "
                "export support a configurable shared frame rate"
            )

        env = os.environ.copy()
        # Free-form compatibility values have already passed through the same
        # YAML -> process environment -> CLI merge. Managed values are applied
        # afterwards so stale inherited shell variables cannot undo a CLI
        # option such as --robot or --backend.
        for key, value in cfg.get("environment", {}).items():
            env[str(key)] = _env_value(value)
        env.update(
            {
                # The generic batch shell is deliberately an internal backend.
                # Only the config runner and backend wrappers may enter it.
                "PIPELINE_WRAPPER_CONFIGURED": "1",
                # Its inline evaluator is obsolete; formal quality reporting is
                # invoked below by this runner through evaluate_clip_quality.py.
                "EVALUATE": "0",
                "PIPELINE_ROOT": str(self.root),
                # User-site packages can shadow the Conda-pinned PyTorch/PyTorch3D ABI.
                # All managed pipeline subprocesses must use the declared environment only.
                "PYTHONNOUSERSITE": "1",
                "CONDA_BASE": str(
                    self._path(
                        runtime.get(
                            "conda_base",
                            str(Path.home() / "miniconda3"),
                        )
                    )
                ),
                "CUDA_VISIBLE_DEVICES": str(
                    runtime.get("cuda_visible_devices", "0")
                ),
                "SEGMENTATION_DISPLAY": str(runtime.get("segmentation_display", ":1")),
                "ENV_SAMHQ": str(
                    runtime.get("conda_envs", {}).get("samhq", "moge2")
                ),
                "ENV_MOGE2": str(
                    runtime.get("conda_envs", {}).get("moge2", "moge2")
                ),
                "ENV_MEGAPOSE": str(
                    runtime.get("conda_envs", {}).get("megapose", "megapose")
                ),
                "ENV_CRISP": str(
                    runtime.get("conda_envs", {}).get(
                        "crisp", scene_crisp.get("conda_env", "crisp")
                    )
                ),
                "SCENE_ENABLED": _bool_env(scene.get("enabled", False)),
                "SCENE_REQUIRED": _bool_env(scene.get("required", False)),
                "SCENE_BACKEND": str(scene.get("backend", "crisp")),
                "SCENE_WORK_ROOT": str(
                    self._path(scene.get("work_root", "scene_work/crisp"))
                ),
                "SCENE_OUTPUT_SUBDIR": str(
                    scene.get("output_subdir", "scene_reconstruction")
                ),
                "CRISP_ROOT": str(
                    self._path(
                        scene_crisp.get("root", "external/CRISP-Real2Sim")
                    )
                ),
                "CRISP_STATIC_CAMERA": _bool_env(
                    scene_crisp.get("static_camera", True)
                ),
                "CRISP_RUN_CONTACT_PASS": _bool_env(
                    scene_crisp.get("run_contact_pass", False)
                ),
                "DATASET": str(self._path(input_cfg.get("dataset_dir", "dataset_new6"))),
                "OUTPUT_BASE": str(
                    self._path(output_cfg.get("root", "output_dir/config_run"))
                ),
                "CLIP_FILTER": str(input_cfg.get("clip_filter", "")),
                "USE_WORK_VIDEO": _bool_env(work_cfg.get("enabled", True)),
                "WORK_DATASET": str(
                    self._path(
                        work_cfg.get("directory", "dataset_new6_work_1280")
                    )
                ),
                "WORK_WIDTH": str(work_cfg.get("width", 1280)),
                "WORK_HEIGHT": str(work_cfg.get("height", 960)),
                "WORK_CRF": str(work_cfg.get("crf", 18)),
                "WORK_FPS": f"{work_fps:g}",
                "FORCE_WORK_VIDEO": _bool_env(work_cfg.get("force", False)),
                "SKIP_EXISTING": _bool_env(resume.get("skip_existing", True)),
                "GVHMR_FORCE_HAND_PREPROCESS": _bool_env(
                    resume.get("force_hand_preprocess", False)
                ),
                "FORCE_LOCO": _bool_env(
                    resume.get("force_locomotion", False)
                ),
                "FORCE_SMOOTH": _bool_env(
                    resume.get("force_smoothing", False)
                ),
                "FORCE_PHC": _bool_env(resume.get("force_phc", False)),
                "GMR_OVERRIDE": _bool_env(resume.get("force_gmr", True)),
                "GVHMR_HAND_CONSTRAINT_PROFILE": str(
                    human.get("constraint_profile", "conservative")
                ),
                "GVHMR_HAND_BACKEND": str(
                    human.get("backend", "hand4wholepp")
                ),
                "GVHMR_BATCH_SIZE": str(human.get("gvhmr_batch_size", 4)),
                "GVHMR_HAND4WHOLEPP_BATCH_SIZE": str(
                    human.get("hand_batch_size", 1)
                ),
                "GVHMR_HAND4WHOLEPP_CROP_TRACKING": str(
                    hand_crop_tracking_mode
                ),
                "GVHMR_HAND4WHOLEPP_CROP_TRACKING_MAX_GAP": str(
                    hand_crop_tracking.get("max_gap", 8)
                ),
                "GVHMR_HAND4WHOLEPP_CROP_TRACKING_MAX_PREDICTION_GAP": str(
                    hand_crop_tracking.get("max_prediction_gap", 2)
                ),
                "GVHMR_HAND4WHOLEPP_CROP_TRACKING_DIRECT_OBSERVATION_QUALITY": str(
                    hand_crop_tracking.get("direct_observation_quality", 0.75)
                ),
                "GVHMR_VISIBLE_HAND_REFINE": _bool_env(
                    visible_hand_refine.get("enabled", False)
                ),
                "GVHMR_VISIBLE_HAND_REFINE_DEVICE": visible_refine_device,
                "GVHMR_VISIBLE_HAND_REFINE_BATCH_SIZE": str(
                    visible_hand_refine.get("batch_size", 16)
                ),
                "GVHMR_VISIBLE_HAND_REFINE_STEPS": str(
                    visible_hand_refine.get("steps", 8)
                ),
                "GVHMR_VISIBLE_HAND_REFINE_LR": str(
                    visible_hand_refine.get("lr", 0.02)
                ),
                "GVHMR_VISIBLE_HAND_REFINE_PRIOR_WEIGHT": str(
                    visible_hand_refine.get("prior_weight", 0.02)
                ),
                "GVHMR_VISIBLE_HAND_REFINE_FIT_CONFIDENCE": str(
                    visible_hand_refine.get("fit_confidence", 0.60)
                ),
                "GVHMR_VISIBLE_HAND_REFINE_FIT_MIN_KEYPOINTS": str(
                    visible_hand_refine.get("fit_min_keypoints", 12)
                ),
                "GVHMR_VISIBLE_HAND_REFINE_FIT_PARTITION": str(
                    visible_hand_refine.get("fit_partition", "all")
                ).lower(),
                "GVHMR_VISIBLE_HAND_REFINE_HOLDOUT_MIN_KEYPOINTS": str(
                    visible_hand_refine.get("holdout_min_keypoints", 3)
                ),
                "GVHMR_VISIBLE_HAND_REFINE_HOLDOUT_MAX_RELATIVE_REGRESSION_PX": str(
                    visible_hand_refine.get(
                        "holdout_max_relative_regression_px", 1.0
                    )
                ),
                "GVHMR_VISIBLE_HAND_REFINE_MAX_DELTA_DEGREES": str(
                    visible_hand_refine.get("max_delta_degrees", 25.0)
                ),
                "GVHMR_VISIBLE_HAND_REFINE_MIN_RELATIVE_IMPROVEMENT": str(
                    visible_hand_refine.get("min_relative_improvement", 0.25)
                ),
                "GVHMR_VISIBLE_HAND_REFINE_MIN_RELATIVE_IMPROVEMENT_PX": str(
                    visible_hand_refine.get("min_relative_improvement_px", 2.0)
                ),
                "GVHMR_VISIBLE_HAND_REFINE_MAX_ABSOLUTE_REGRESSION_PX": str(
                    visible_hand_refine.get("max_absolute_regression_px", 2.0)
                ),
                "GVHMR_VISIBLE_HAND_REFINE_MAX_ANCHOR_ERROR_PX": str(
                    visible_hand_refine.get("max_anchor_error_px", 20.0)
                ),
                "GVHMR_VISIBLE_HAND_REFINE_MAX_ANCHOR_ERROR_BBOX_RATIO": str(
                    visible_hand_refine.get("max_anchor_error_bbox_ratio", 0.25)
                ),
                "GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_HAND_CONFIDENCE": str(
                    visible_hand_refine.get("evidence_hand_confidence", 0.45)
                ),
                "GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_MIN_KEYPOINTS": str(
                    visible_hand_refine.get("evidence_min_keypoints", 8)
                ),
                "GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_MEAN_CONFIDENCE": str(
                    visible_hand_refine.get("evidence_mean_confidence", 0.50)
                ),
                "GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_WRIST_CONFIDENCE": str(
                    visible_hand_refine.get("evidence_wrist_confidence", 0.45)
                ),
                "GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_MIN_BBOX_DIAGONAL_PX": str(
                    visible_hand_refine.get("evidence_min_bbox_diagonal_px", 96.0)
                ),
                "GVHMR_VISIBLE_HAND_REFINE_EVIDENCE_MIN_RUN": str(
                    visible_hand_refine.get("evidence_min_run", 3)
                ),
                "GVHMR_HAMER_BATCH_SIZE": str(
                    human.get("hand_batch_size", 1)
                ),
                "GVHMR_VITPOSE_IMG_DS": str(
                    human.get("vitpose_image_scale", 1.0)
                ),
                "GVHMR_LOW_MEMORY": _bool_env(
                    human.get("low_memory", True)
                ),
                "GVHMR_ISOLATE_HAND_PREPROCESS": _bool_env(
                    human.get("isolate_hand_process", True)
                ),
                "GVHMR_FILTER_MANO_WRIST": _bool_env(
                    filters.get("wrist", False)
                ),
                "GVHMR_FILTER_MANO_TEMPORAL": _bool_env(
                    filters.get("temporal", True)
                ),
                "GVHMR_FILTER_MANO_FINGERS": _bool_env(
                    filters.get("fingers", True)
                ),
                "GVHMR_TEMPORAL_FILTER_GLOBAL_ORIENT_FILL_MODE": (
                    global_orient_fill_mode
                ),
                "GVHMR_FINGER_FILTER_WRIST_MODE": wrist_mode,
                "GVHMR_DIAGNOSE_HAND": _bool_env(
                    human.get("diagnostics", True)
                ),
                "FORCE_LOCO": _bool_env(
                    resume.get("force_locomotion", False)
                ),
                "GMR_SMOOTH_WINDOW": str(
                    locomotion.get("smooth_window", 9)
                ),
                "SKIP_PHC": _bool_env(not phc.get("enabled", True)),
                "RUN_GMR": _bool_env(gmr.get("enabled", True)),
                "GMR_HAND_MODEL": str(gmr.get("hand_model", "sharpa")),
                "GMR_SOURCE": str(gmr.get("source", "smoothed")),
                "GMR_TARGET_FPS": str(gmr.get("target_fps", 30)),
                "GMR_HEIGHT_ADJUST_MODE": str(
                    gmr.get("height_adjust_mode", "global_foot_geom")
                ),
                "GMR_SUPPORT_CONTACT_HEIGHT": str(
                    gmr.get("support_contact_height", 0.08)
                ),
                "GMR_SUPPORT_EXIT_CONTACT_HEIGHT": str(
                    gmr.get("support_exit_contact_height", 0.10)
                ),
                "GMR_SUPPORT_MAX_VERTICAL_SPEED": str(
                    gmr.get("support_max_vertical_speed", 1.20)
                ),
                "GMR_SUPPORT_MAX_HORIZONTAL_SPEED": str(
                    gmr.get("support_max_horizontal_speed", 0.15)
                ),
                "GMR_SUPPORT_MIN_CONTACT_RUN": str(
                    gmr.get("support_min_contact_run", 3)
                ),
                "GMR_SUPPORT_MAX_CONTACT_GAP": str(
                    gmr.get("support_max_contact_gap", 1)
                ),
                "GMR_SUPPORT_ROOT_STEP_LIMIT": str(
                    gmr.get("support_root_step_limit", 0.03)
                ),
                "GMR_SUPPORT_APPLY_ROOT_Z": _bool_env(
                    gmr.get("support_apply_root_z", False)
                ),
                "GMR_HUMAN_YAW_OFFSET_DEG": str(
                    gmr.get("human_yaw_offset_deg", 0.0)
                ),
                "GMR_CAMERA_SOURCE": str(
                    gmr.get("camera_source", "gvhmr")
                ),
                "GMR_CAMERA_SUBJECT_ALIGN_XY": _bool_env(
                    gmr.get("camera_subject_align_xy", True)
                ),
                "GMR_SCENE_MUJOCO_NAME": str(
                    gmr.get("scene_mujoco_name", "")
                ),
                "GMR_RENDER": _bool_env(gmr.get("render", True)),
                "GMR_COMPOSITE": _bool_env(
                    gmr.get("composite_2x2", True)
                ),
                "GMR_MUJOCO_GL": str(gmr.get("mujoco_gl", "egl")),
                "GMR_RENDER_WIDTH": str(gmr.get("render_width", 960)),
                "GMR_RENDER_HEIGHT": str(gmr.get("render_height", 720)),
                "HUNYUAN3D_FACE_COUNT": str(
                    monocular.get("hunyuan_face_count", 100000)
                ),
                "HUNYUAN3D_TIMEOUT": str(
                    monocular.get("hunyuan_timeout", 900)
                ),
                "HUNYUAN3D_REGION": str(
                    monocular.get("hunyuan_region", "ap-guangzhou")
                ),
                "MOGE_RESOLUTION_LEVEL": str(
                    monocular.get("moge_resolution_level", 7)
                ),
                "MEGAPOSE_HYPOTHESES": str(
                    monocular.get("megapose", {}).get(
                        "pose_hypotheses", 5
                    )
                ),
                "MEGAPOSE_COARSE_GRID_STRIDE": str(
                    monocular.get("megapose", {}).get(
                        "coarse_grid_stride", 2
                    )
                ),
                "MEGAPOSE_REFINER_ITERATIONS": str(
                    monocular.get("megapose", {}).get(
                        "refiner_iterations", 5
                    )
                ),
                "MEGAPOSE_TEMPORAL_ITERATIONS": str(
                    monocular.get("megapose", {}).get(
                        "temporal_refiner_iterations", 2
                    )
                ),
                "MEGAPOSE_REINIT_INTERVAL": str(
                    monocular.get("megapose", {}).get(
                        "reinit_interval", 120
                    )
                ),
                "MEGAPOSE_MIN_IOU": str(
                    monocular.get("megapose", {}).get("min_iou", 0.05)
                ),
                "MEGAPOSE_REINIT_IOU": str(
                    monocular.get("megapose", {}).get(
                        "reinit_iou", 0.18
                    )
                ),
                "MEGAPOSE_MAX_INTERPOLATION_GAP": str(
                    monocular.get("megapose", {}).get(
                        "max_interpolation_gap", 8
                    )
                ),
            }
        )
        self._apply_hand_model(env, env["GMR_HAND_MODEL"])
        return env

    @staticmethod
    def _apply_hand_model(env: dict[str, str], model: str) -> None:
        if model in {"sharpa", "sharpa_g1"}:
            env.update(
                {
                    "GMR_ROBOT": "unitree_g1",
                    "GMR_HAND_RETARGET_MODE": "off",
                    "GMR_SHARPA_HANDS": "1",
                    "GMR_SHARPA_AUTO_RETARGET": "1",
                    "GMR_BRAINCO_HANDS": "0",
                    "GMR_BRAINCO_AUTO_RETARGET": "0",
                    "GMR_HEIGHT_ADJUST_MODE": env.get(
                        "GMR_HEIGHT_ADJUST_MODE",
                        "global_foot_geom",
                    ),
                }
            )
        elif model == "sharpa_h1":
            env.update(
                {
                    "GMR_ROBOT": "unitree_h1_with_hand",
                    "GMR_HAND_RETARGET_MODE": "off",
                    "GMR_SHARPA_HANDS": "1",
                    "GMR_SHARPA_AUTO_RETARGET": "1",
                    "GMR_BRAINCO_HANDS": "0",
                    "GMR_BRAINCO_AUTO_RETARGET": "0",
                }
            )
        elif model == "g1":
            env.update(
                {
                    "GMR_ROBOT": "unitree_g1_with_hands",
                    "GMR_SHARPA_HANDS": "0",
                    "GMR_BRAINCO_HANDS": "0",
                }
            )
        elif model == "brainco":
            env.update(
                {
                    "GMR_ROBOT": "unitree_g1",
                    "GMR_HAND_RETARGET_MODE": "off",
                    "GMR_SHARPA_HANDS": "0",
                    "GMR_BRAINCO_HANDS": "1",
                    "GMR_BRAINCO_AUTO_RETARGET": "1",
                }
            )
        else:
            raise ValueError(f"unsupported gmr.hand_model: {model}")

    def _run(self, command: list[str], env: dict[str, str] | None = None) -> None:
        print("+ " + shlex.join(command), flush=True)
        if self.dry_run:
            return
        subprocess.run(
            command,
            cwd=self.root,
            env=env or self.environment,
            check=True,
        )

    def _source_videos(self) -> list[Path]:
        input_cfg = self.config.get("input", {})
        dataset = self._path(input_cfg.get("dataset_dir", "dataset_new6"))
        extensions = {
            str(item).lower()
            for item in input_cfg.get("extensions", VIDEO_EXTENSIONS)
        }
        unknown = extensions - VIDEO_EXTENSIONS
        if unknown:
            raise ValueError(f"unsupported input extensions: {sorted(unknown)}")
        if not dataset.is_dir():
            raise FileNotFoundError(dataset)
        clip_filter = str(input_cfg.get("clip_filter", ""))
        clip_filters = [
            item.strip() for item in clip_filter.split(",") if item.strip()
        ]
        videos = [
            path
            for path in sorted(dataset.iterdir())
            if path.is_file()
            and path.suffix.lower() in extensions
            and (
                not clip_filters
                or any(item in path.name for item in clip_filters)
            )
        ]
        raw_min_duration = input_cfg.get("min_duration_seconds")
        if raw_min_duration is not None:
            try:
                min_duration = float(raw_min_duration)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "input.min_duration_seconds must be a non-negative number"
                ) from exc
            if min_duration < 0.0:
                raise ValueError(
                    "input.min_duration_seconds must be a non-negative number"
                )
            before_duration = len(videos)
            unreadable = 0
            accepted: list[Path] = []
            for video in videos:
                duration = self._video_duration_seconds(video)
                if duration is None:
                    unreadable += 1
                    continue
                if duration + 1e-6 >= min_duration:
                    accepted.append(video)
            videos = accepted
            print(
                "[INPUT] duration gate "
                f">={min_duration:.2f}s: admitted {len(videos)}/{before_duration} "
                f"(short={before_duration - len(videos) - unreadable}, "
                f"unreadable={unreadable})",
                flush=True,
            )

        raw_limit = input_cfg.get("max_videos")
        if raw_limit is not None:
            try:
                max_videos = int(raw_limit)
            except (TypeError, ValueError) as exc:
                raise ValueError("input.max_videos must be a positive integer") from exc
            if max_videos <= 0:
                raise ValueError("input.max_videos must be a positive integer")
            videos = videos[:max_videos]
        if not videos:
            raise ValueError(f"no input videos found in {dataset}")
        return videos

    def _video_duration_seconds(self, video: Path) -> float | None:
        """Read only container metadata and memoize the result for this run."""
        if video in self._video_duration_cache:
            return self._video_duration_cache[video]
        capture = cv2.VideoCapture(str(video))
        try:
            if not capture.isOpened():
                duration: float | None = None
            else:
                frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                duration = frames / fps if frames > 0.0 and fps > 0.0 else None
        finally:
            capture.release()
        self._video_duration_cache[video] = duration
        return duration

    def _videos(self) -> list[Path]:
        videos = self._source_videos()
        if self._eligible_clips is not None:
            videos = [
                video for video in videos if video.stem in self._eligible_clips
            ]
        return videos

    def _fullbody_preflight_config(self) -> dict[str, Any]:
        value = self.config.get("input", {}).get("fullbody_preflight", {})
        if not isinstance(value, dict):
            raise ValueError("input.fullbody_preflight must be a mapping")
        return value

    def _run_fullbody_preflight(self) -> list[Path]:
        """Run the cheap pose gate once, before invoking the GVHMR wrapper."""
        preflight = self._fullbody_preflight_config()
        candidates = self._source_videos()
        if not bool(preflight.get("enabled", True)):
            self._eligible_clips = {video.stem for video in candidates}
            print(
                f"[FULLBODY] disabled; admitting {len(candidates)} selected clips",
                flush=True,
            )
            return candidates

        mode = str(preflight.get("mode", "gate")).lower()
        if mode not in {"gate", "report"}:
            raise ValueError(
                "input.fullbody_preflight.mode must be gate or report"
            )
        output_root = self._path(self.config["output"]["root"])
        report_path = output_root / "fullbody_preflight.jsonl"
        csv_path = output_root / "fullbody_preflight.csv"
        model_path = self._path(
            preflight.get("model", "models/yolo11n-pose.pt")
        )
        script = self.root / "scripts" / "preflight_fullbody_gate.py"
        if not script.is_file():
            raise FileNotFoundError(script)
        if not model_path.is_file():
            raise FileNotFoundError(
                "input.fullbody_preflight.model is missing: " f"{model_path}"
            )
        if self.dry_run:
            self._eligible_clips = {video.stem for video in candidates}
            print(
                f"[DRY-RUN][FULLBODY] would evaluate {len(candidates)} clips "
                f"with {model_path.name}",
                flush=True,
            )
            return candidates

        output_root.mkdir(parents=True, exist_ok=True)
        list_path = output_root / ".fullbody_preflight_inputs.txt"
        list_path.write_text(
            "".join(f"{video.resolve()}\n" for video in candidates),
            encoding="utf-8",
        )
        option_map = {
            "samples": "samples",
            "image_size": "imgsz",
            "person_confidence": "person-confidence",
            "keypoint_confidence": "keypoint-confidence",
            "min_person_ratio": "min-person-ratio",
            "min_top_ratio": "min-top-ratio",
            "min_left_wrist_ratio": "min-left-wrist-ratio",
            "min_right_wrist_ratio": "min-right-wrist-ratio",
            "min_left_ankle_ratio": "min-left-ankle-ratio",
            "min_right_ankle_ratio": "min-right-ankle-ratio",
            "min_full_body_ratio": "min-full-body-ratio",
            "min_person_height_ratio": "min-person-height-ratio",
            "min_usable_scale_ratio": "min-usable-scale-ratio",
            "max_competing_person_ratio": "max-competing-person-ratio",
            "min_competing_person_scale_ratio": (
                "min-competing-person-scale-ratio"
            ),
            "support_posture_mode": "support-posture-mode",
            "support_posture_ratio": "support-posture-ratio",
            "support_posture_min_run_seconds": (
                "support-posture-min-run-seconds"
            ),
            "support_posture_thigh_horizontal_cos": (
                "support-posture-thigh-horizontal-cos"
            ),
            "support_posture_shin_vertical_cos": (
                "support-posture-shin-vertical-cos"
            ),
            "support_posture_torso_vertical_cos": (
                "support-posture-torso-vertical-cos"
            ),
            "support_posture_max_hip_knee_height_ratio": (
                "support-posture-max-hip-knee-height-ratio"
            ),
            "support_posture_min_ankle_below_hip_ratio": (
                "support-posture-min-ankle-below-hip-ratio"
            ),
            "support_posture_max_ankle_height_difference_ratio": (
                "support-posture-max-ankle-height-difference-ratio"
            ),
            "support_posture_object_model": "support-posture-object-model",
            "support_posture_object_confidence": (
                "support-posture-object-confidence"
            ),
            "support_posture_object_ratio": "support-posture-object-ratio",
            "support_posture_weak_pose_ratio": "support-posture-weak-pose-ratio",
        }
        command = [
            self.runtime_python,
            str(script),
            "--video-list",
            str(list_path),
            "--report",
            str(report_path),
            "--csv",
            str(csv_path),
            "--model",
            str(model_path),
            "--device",
            str(preflight.get("device", "auto")),
        ]
        for config_name, option in option_map.items():
            if config_name in preflight:
                value = preflight[config_name]
                # ``support_posture_mode`` used to be represented as a boolean
                # in pipeline YAML files.  The preflight CLI now deliberately
                # exposes the more precise off/report/gate modes.  Normalise
                # the legacy values here so an omitted/false setting cannot be
                # passed to argparse as the invalid string "False".
                if config_name == "support_posture_mode":
                    if value is None or value is False:
                        value = "off"
                    elif value is True:
                        value = "report"
                    else:
                        value = str(value).strip().lower()
                command.extend([f"--{option}", str(value)])
        if not bool(preflight.get("cache", True)):
            command.append("--no-cache")
        try:
            self._run(command)
        finally:
            list_path.unlink(missing_ok=True)

        records: dict[str, dict[str, Any]] = {}
        for line in report_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict) and isinstance(value.get("source"), str):
                records[value["source"]] = value
        missing = [
            video for video in candidates if str(video.resolve()) not in records
        ]
        if missing:
            raise RuntimeError(
                "full-body preflight did not return records for: "
                + ", ".join(video.name for video in missing)
            )
        admitted = [
            video
            for video in candidates
            if bool(records[str(video.resolve())].get("eligible", False))
        ]
        excluded = [video for video in candidates if video not in admitted]
        for video in excluded:
            record = records[str(video.resolve())]
            print(
                f"[FULLBODY] exclude {video.name}: "
                + " | ".join(record.get("reasons", ["unknown_reason"])),
                flush=True,
            )
        selected = candidates if mode == "report" else admitted
        self._eligible_clips = {video.stem for video in selected}
        print(
            f"[FULLBODY] admission mode={mode}: "
            f"{len(admitted)}/{len(candidates)} eligible; "
            f"running={len(selected)}",
            flush=True,
        )
        return selected

    def _video_alias_sources(
        self,
        clip: str,
        output_root: Path,
    ) -> dict[str, Path | None]:
        clip_dir = output_root / clip
        slug = _clip_slug(clip)
        gvhmr_dir = clip_dir / "gvhmr_out" / clip
        return {
            "gvhmr": _first_existing(
                [
                    gvhmr_dir / "1_incam_object.mp4",
                    gvhmr_dir / "1_incam.mp4",
                    gvhmr_dir / f"{clip}_3_incam_global_horiz.mp4",
                    gvhmr_dir / "2_global.mp4",
                ]
            ),
            "phc": _latest_mp4(clip_dir / "phc_renderings"),
            "gmr": _first_existing(
                [
                    clip_dir / f"{slug}__gmr.mp4",
                    _first_matching(clip_dir, "*__gmr.mp4"),
                    clip_dir / "unitree_g1_sharpa_gvhmr.mp4",
                    clip_dir / "unitree_h1_with_hand_sharpa_gvhmr.mp4",
                    clip_dir / "unitree_h1_with_hand_retarget_gvhmr.mp4",
                ]
            ),
            "2x2": _first_existing(
                [
                    clip_dir / f"{slug}__2x2.mp4",
                    _first_matching(clip_dir, "*__2x2.mp4"),
                    clip_dir / "composite_2x2.mp4",
                ]
            ),
        }

    def write_video_aliases(self, clips: list[str] | None = None) -> None:
        """Create stable per-clip video aliases for GVHMR/PHC/GMR/2x2."""
        if self.dry_run:
            print("[DRY-RUN] Skipping video alias write.", flush=True)
            return
        output_root = self._path(self.config["output"]["root"])
        if clips is None:
            clips = [video.stem for video in self._videos()]
        for clip in clips:
            clip_dir = output_root / clip
            if not clip_dir.is_dir():
                continue
            slug = _clip_slug(clip)
            sources = self._video_alias_sources(clip, output_root)
            alias_dir = clip_dir / "videos"
            for kind in VIDEO_ALIAS_KINDS:
                source = sources.get(kind)
                if source is None:
                    continue
                _link_or_copy(source, alias_dir / f"{slug}__{kind}.mp4")
            composite = sources.get("2x2")
            legacy_composite = clip_dir / "composite_2x2.mp4"
            if composite is not None and not legacy_composite.is_file():
                _link_or_copy(composite, legacy_composite)

    def _product_videos(self) -> list[Path]:
        videos = self._videos()
        product = self.config.get("product", {})
        obj = self.config.get("object", {})
        if (
            obj.get("enabled", False)
            and obj.get("only_configured_clips", True)
            and product.get("object_policy", "if_valid") == "require_valid"
        ):
            configured = set(obj.get("clips", {}))
            videos = [video for video in videos if video.stem in configured]
        if not videos:
            raise ValueError(
                "no videos match the product/object clip configuration"
            )
        return videos

    def _completed_output_clips(self) -> list[str]:
        """Return clips that actually completed the human/GMR stage."""
        output_root = self._path(self.config["output"]["root"])
        if not output_root.is_dir():
            return []
        return sorted(
            child.name
            for child in output_root.iterdir()
            if child.is_dir() and (child / "robot_motion.pkl").is_file()
        )

    def _completed_product_clips(self) -> list[str]:
        """Apply product object policy to completed, not candidate, clips."""
        clips = self._completed_output_clips()
        product = self.config.get("product", {})
        obj = self.config.get("object", {})
        if (
            obj.get("enabled", False)
            and obj.get("only_configured_clips", True)
            and product.get("object_policy", "if_valid") == "require_valid"
        ):
            configured = set(obj.get("clips", {}))
            clips = [clip for clip in clips if clip in configured]
        return clips

    def _quality_videos(self) -> list[Path]:
        videos = self._videos()
        obj = self.config.get("object", {})
        if obj.get("enabled", False) and obj.get(
            "only_configured_clips",
            True,
        ):
            configured = set(obj.get("clips", {}))
            videos = [video for video in videos if video.stem in configured]
        if not videos:
            raise ValueError(
                "no videos match the quality/object clip configuration"
            )
        return videos

    def _scene_config(self) -> dict[str, Any]:
        return _validate_scene_config(self.config.get("scene", {}))

    def _scene_output_root(self, clip: str) -> Path:
        scene = self._scene_config()
        return (
            self._path(self.config["output"]["root"])
            / clip
            / str(scene.get("output_subdir", "scene_reconstruction"))
        )

    def _resolve_scene_gpu(self, scene: dict[str, Any]) -> str:
        """Resolve an explicit GPU or select the largest-free-memory device."""
        gpu = scene.get("runtime", {}).get("gpu", 0)
        if isinstance(gpu, int):
            return str(gpu)
        if gpu != "auto":
            raise ValueError("scene.runtime.gpu must be 'auto' or an integer")
        try:
            probe = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "scene.runtime.gpu=auto requires nvidia-smi; set an explicit GPU instead"
            ) from exc
        if probe.returncode != 0:
            raise RuntimeError(
                "scene.runtime.gpu=auto could not query nvidia-smi: "
                + probe.stderr.strip()
            )
        candidates: list[tuple[int, int, int]] = []
        for line in probe.stdout.splitlines():
            try:
                index_text, used_text, total_text = (item.strip() for item in line.split(","))
                index, used, total = int(index_text), int(used_text), int(total_text)
            except ValueError:
                continue
            if total > 0 and used >= 0:
                candidates.append((total - used, index, total))
        if not candidates:
            raise RuntimeError("scene.runtime.gpu=auto found no usable NVIDIA GPU")
        free_mib, index, total_mib = max(candidates, key=lambda item: (item[0], -item[1]))
        print(
            f"[GPU] auto selected GPU {index}: {free_mib}/{total_mib} MiB free",
            flush=True,
        )
        return str(index)

    @staticmethod
    def _scene_motion_filename(kind: str) -> str:
        filenames = {
            "converted": "001_converted.npz",
            "smoothed": "001_smoothed.npz",
            "final": "001_final.npz",
        }
        return filenames[kind]

    def _scene_input_paths(self, clip: str) -> list[Path]:
        scene = self._scene_config()
        clip_root = self._path(self.config["output"]["root"]) / clip
        paths = [
            clip_root / self._scene_motion_filename(scene["source_motion"]),
            clip_root / "001_smplx_hands.npz",
            clip_root / "gvhmr_camera.npz",
        ]
        if scene.get("backend") == "videomimic_nksr":
            paths.append(clip_root / "gvhmr_out" / clip / "valid_video.mp4")
        return paths

    def _scene_conda_executable(self) -> Path:
        runtime = self.config.get("runtime", {})
        return self._path(
            Path(str(runtime.get("conda_base", "miniconda3"))) / "bin" / "conda"
        )

    def _check_scene_preflight(
        self,
        videos: list[Path],
        *,
        require_existing_inputs: bool,
    ) -> None:
        scene = self._scene_config()
        if not scene.get("enabled", False):
            raise ValueError("scene.enabled is false")
        if self.dry_run:
            print(
                "[DRY-RUN][SCENE] configuration validated; skipping scene "
                "repository, environment, and output-input checks.",
                flush=True,
            )
            return

        if require_existing_inputs:
            bridge = scene.get("robot_bridge", {})
            if bridge.get("enabled", False):
                source_root = self._path(str(bridge["source_output_root"]))
                scene_subdir = str(scene.get("output_subdir", "scene_reconstruction"))
                source_scene_subdir = str(
                    bridge.get("source_scene_subdir", scene_subdir)
                )
                motion_name = str(
                    bridge.get("motion_name", "human_motion_static_hard.npz")
                )
                camera_name = str(
                    bridge.get("camera_name", "gvhmr_camera_static_hard.npz")
                )
                required_paths = [
                    path
                    for video in videos
                    for path in (
                        source_root / video.stem / "001_smoothed.npz",
                        source_root / video.stem / "001_smplx_hands.npz",
                        source_root
                        / video.stem
                        / "gvhmr_out"
                        / video.stem
                        / "valid_video.mp4",
                        source_root
                        / video.stem
                        / source_scene_subdir
                        / "scene_manifest.json",
                        source_root
                        / video.stem
                        / source_scene_subdir
                        / "scene"
                        / "scene_mujoco.xml",
                        source_root
                        / video.stem
                        / source_scene_subdir
                        / "scene"
                        / "background_mesh_collision.obj",
                        source_root
                        / video.stem
                        / source_scene_subdir
                        / "human"
                        / motion_name,
                        source_root
                        / video.stem
                        / source_scene_subdir
                        / "human"
                        / camera_name,
                    )
                ]
                missing = [path for path in required_paths if not path.is_file()]
                error_prefix = "scene robot_bridge source contract is incomplete:\n  "
            else:
                missing = [
                    path
                    for video in videos
                    for path in self._scene_input_paths(video.stem)
                    if not path.is_file()
                ]
                error_prefix = "scene requires completed human-pre outputs:\n  "
            if missing:
                raise FileNotFoundError(error_prefix + "\n  ".join(map(str, missing)))

        conda = self._scene_conda_executable()
        if not conda.is_file():
            raise FileNotFoundError(
                "runtime.conda_base does not provide conda for the scene "
                f"environment: {conda}"
            )
        if scene["backend"] == "crisp":
            crisp = scene["crisp"]
            crisp_root = self._path(crisp["root"])
            expected = [
                crisp_root,
                crisp_root / "run_crisp_video.sh",
                crisp_root / "prep",
                crisp_root / "vis_scripts",
            ]
            missing_crisp = [path for path in expected if not path.exists()]
            if missing_crisp:
                raise FileNotFoundError(
                    "CRISP repository is not ready. Expected:\n  "
                    + "\n  ".join(str(path) for path in missing_crisp)
                    + "\nInstall the pinned checkout under scene.crisp.root before "
                    "running a non-dry scene stage."
                )
            probe_command = [
                str(conda), "run", "-n", str(crisp["conda_env"]), "python", "-c",
                "import cv2, numpy, open3d, torch, trimesh; print('CRISP_ENV_OK')",
            ]
            label = str(crisp["conda_env"])
        else:
            videomimic = scene["videomimic"]
            root = self._path(videomimic["root"])
            expected = [
                root,
                root / "stage1_reconstruction" / "megasam_reconstruction.py",
                root / "stage2_optimization" / "megahunter_optimization.py",
                root / "stage3_postprocessing" / "postprocessing_pipeline.py",
                root / "assets" / "body_models",
            ]
            missing = [path for path in expected if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    "VideoMimic repository is not ready. Expected:\n  " + "\n  ".join(str(path) for path in missing)
                )
            probe_command = [
                str(conda), "run", "-n", str(videomimic["recon_env"]), "python", "-c",
                "import cv2, h5py, numpy, smplx, torch, trimesh; import nksr; from geocalib import GeoCalib; print('VIDEOMIMIC_RECON_ENV_OK')",
            ]
            label = str(videomimic["recon_env"])
        probe = subprocess.run(
            probe_command,
            cwd=self.root,
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=90,
        )
        if probe.returncode:
            detail = probe.stderr.strip() or probe.stdout.strip()
            raise RuntimeError(
                "scene environment validation failed for "
                f"{label}: {detail.splitlines()[-1] if detail else probe.returncode}"
            )

    def _write_scene_stage_status(
        self,
        clip: str,
        *,
        status: str,
        reason: str | None = None,
    ) -> None:
        if self.dry_run:
            return
        # A failed sidecar must never create the package directory itself:
        # scene_runner correctly treats any such directory as immutable and a
        # status marker used to make every retry fail before it could start.
        clip_root = self._path(self.config["output"]["root"]) / clip
        package_name = str(self._scene_config().get("output_subdir", "scene_reconstruction"))
        status_name = "." + package_name.replace("/", "_") + ".scene_stage_status.json"
        path = clip_root / status_name
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "status": status,
            "clip": clip,
            "backend": self._scene_config()["backend"],
        }
        if reason:
            payload["reason"] = reason
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def _scene_command(self, clip: str, config_path: Path) -> list[str]:
        scene = self._scene_config()
        environment = (
            scene["crisp"]["conda_env"]
            if scene["backend"] == "crisp"
            else scene["videomimic"]["recon_env"]
        )
        return [
            str(self._scene_conda_executable()),
            "run",
            "-n",
            str(environment),
            "python",
            str(self.root / "scripts" / "scene" / "scene_runner.py"),
            "--project-root",
            str(self.root),
            "--config",
            str(config_path),
            "--clip",
            clip,
            "--stage",
            "all",
        ]

    def run_scene(self) -> None:
        scene = self._scene_config()
        videos = self._videos()
        if not videos:
            print("[SCENE] no admitted clips; skipping sidecar.", flush=True)
            return
        # In bridge mode, materialize the static-frame inputs before scene
        # execution; later human_post reuses this immutable derived clip.
        bridge_enabled = bool(scene.get("robot_bridge", {}).get("enabled", False))
        if not bridge_enabled:
            self._ensure_run_manifests(videos)
        self._prepare_videomimic_robot_bridge(videos)
        if bridge_enabled:
            self._ensure_run_manifests(videos, allow_existing_clip=True)
        self._check_scene_preflight(videos, require_existing_inputs=True)
        if scene.get("robot_bridge", {}).get("enabled", False):
            print(
                "[SCENE] robot_bridge published the validated static scene "
                "package; skipping duplicate VideoMimic reconstruction.",
                flush=True,
            )
            return
        work_root = self._path(scene.get("work_root", "scene_work/videomimic"))
        effective_config_path = work_root / "_runner" / "effective_config.json"
        if not self.dry_run:
            effective_config_path.parent.mkdir(parents=True, exist_ok=True)
            effective_config_path.write_text(
                json.dumps(self.config, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        environment = self.environment.copy()
        environment["CUDA_VISIBLE_DEVICES"] = self._resolve_scene_gpu(scene)
        for video in videos:
            clip = video.stem
            try:
                self._run(
                    self._scene_command(clip, effective_config_path),
                    env=environment,
                )
            except subprocess.CalledProcessError as exc:
                self._write_scene_stage_status(
                    clip,
                    status="failed",
                    reason=f"scene_runner_exit_{exc.returncode}",
                )
                if scene.get("required", False):
                    raise
                print(
                    f"[SCENE] optional sidecar failed for {clip}; continuing "
                    "with the human-only pipeline.",
                    flush=True,
                )

    def check(self, stage: str) -> None:
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        runtime_python = Path(self.runtime_python)
        if not runtime_python.is_file():
            raise FileNotFoundError(
                f"runtime.python is not a file: {runtime_python}"
            )
        videos = (
            self._videos()
            if stage in {"human", "human_pre", "scene", "human_post", "object", "all"}
            else []
        )
        human = self.config.get("human", {})
        hand_crop_tracking = human.get("hand_crop_tracking", {})
        if not isinstance(hand_crop_tracking, dict):
            raise ValueError("human.hand_crop_tracking must be a mapping")
        raw_tracking_mode = hand_crop_tracking.get("mode", "off")
        tracking_mode = (
            "off" if raw_tracking_mode is False else str(raw_tracking_mode)
        )
        if tracking_mode not in {"off", "flow_kalman"}:
            raise ValueError(
                "human.hand_crop_tracking.mode must be off or flow_kalman"
            )
        try:
            max_gap = int(hand_crop_tracking.get("max_gap", 8))
            max_prediction_gap = int(
                hand_crop_tracking.get("max_prediction_gap", 2)
            )
            direct_observation_quality = float(
                hand_crop_tracking.get("direct_observation_quality", 0.75)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "human.hand_crop_tracking values must be numeric"
            ) from exc
        if max_gap < 0 or max_prediction_gap < 0:
            raise ValueError(
                "human.hand_crop_tracking gap limits must be non-negative"
            )
        if max_prediction_gap > max_gap:
            raise ValueError(
                "human.hand_crop_tracking.max_prediction_gap cannot exceed max_gap"
            )
        if not 0.0 <= direct_observation_quality <= 1.0:
            raise ValueError(
                "human.hand_crop_tracking.direct_observation_quality must be in [0, 1]"
            )
        product = self.config.get("product", {})
        product_will_run = stage == "product" or (
            product.get("enabled", False)
            and stage in set(product.get("export_after_stages", []))
        )
        if product_will_run:
            if not product.get("enabled", False):
                raise ValueError(
                    "product.enabled is false; use a product/batch pipeline YAML"
                )
            policy = product.get("object_policy", "if_valid")
            if policy not in {"exclude", "if_valid", "require_valid"}:
                raise ValueError(
                    "product.object_policy must be exclude, if_valid or require_valid"
                )
            output_root = self._path(self.config["output"]["root"])
            product_root = self._path(
                product.get("root", "assets/pipeline_products")
            )
            if output_root == product_root:
                raise ValueError(
                    "product.root must differ from output.root"
                )
        if stage == "quality":
            quality = self.config.get("quality_evaluation", {})
            if not quality.get("enabled", False):
                raise ValueError("quality_evaluation.enabled is false")
        if stage in {"human", "human_pre", "all"}:
            preflight = self._fullbody_preflight_config()
            if bool(preflight.get("enabled", True)):
                mode = str(preflight.get("mode", "gate")).lower()
                if mode not in {"gate", "report"}:
                    raise ValueError(
                        "input.fullbody_preflight.mode must be gate or report"
                    )
                model_path = self._path(
                    preflight.get("model", "models/yolo11n-pose.pt")
                )
                if not model_path.is_file():
                    raise FileNotFoundError(
                        "input.fullbody_preflight.model is missing: "
                        f"{model_path}"
                    )
                for name, value in preflight.items():
                    if (
                        name.startswith("min_")
                        or name.startswith("max_")
                        or name.endswith("confidence")
                    ):
                        try:
                            numeric = float(value)
                        except (TypeError, ValueError) as exc:
                            raise ValueError(
                                f"input.fullbody_preflight.{name} must be numeric"
                            ) from exc
                        if not 0.0 <= numeric <= 1.0:
                            raise ValueError(
                                f"input.fullbody_preflight.{name} must be in [0, 1]"
                            )
                script = self.root / "scripts" / "preflight_fullbody_gate.py"
                if not script.is_file():
                    raise FileNotFoundError(script)
            backend = self.config.get("human", {}).get(
                "backend",
                "hand4wholepp",
            )
            if backend not in BACKEND_WRAPPERS:
                raise ValueError(f"unsupported human.backend: {backend}")
            wrapper = (
                self.root
                / "GVHMR-hand"
                / "GVHMR-main"
                / "tools"
                / "pipeline"
                / BACKEND_WRAPPERS[backend]
            )
            if not wrapper.exists():
                raise FileNotFoundError(wrapper)
        if stage == "human_post":
            backend = self.config.get("human", {}).get(
                "backend",
                "hand4wholepp",
            )
            if backend not in BACKEND_WRAPPERS:
                raise ValueError(f"unsupported human.backend: {backend}")
            wrapper = (
                self.root
                / "GVHMR-hand"
                / "GVHMR-main"
                / "tools"
                / "pipeline"
                / BACKEND_WRAPPERS[backend]
            )
            if not wrapper.exists():
                raise FileNotFoundError(wrapper)
            if not self.dry_run:
                scene = self._scene_config()
                bridge = scene.get("robot_bridge", {})
                if bridge.get("enabled", False):
                    source_root = self._path(str(bridge["source_output_root"]))
                    scene_subdir = str(scene.get("output_subdir", "scene_reconstruction"))
                    source_scene_subdir = str(
                        bridge.get("source_scene_subdir", scene_subdir)
                    )
                    motion_name = str(
                        bridge.get("motion_name", "human_motion_static_hard.npz")
                    )
                    camera_name = str(
                        bridge.get("camera_name", "gvhmr_camera_static_hard.npz")
                    )
                    missing = [
                        path
                        for video in videos
                        for path in (
                            source_root / video.stem / "001_smoothed.npz",
                            source_root / video.stem / "001_smplx_hands.npz",
                            source_root / video.stem / source_scene_subdir / "scene_manifest.json",
                            source_root / video.stem / source_scene_subdir / "scene" / "primitives_mujoco.json",
                            source_root / video.stem / source_scene_subdir / "human" / motion_name,
                            source_root / video.stem / source_scene_subdir / "human" / camera_name,
                        )
                        if not path.is_file()
                    ]
                else:
                    missing = [
                        path
                        for video in videos
                        for path in self._scene_input_paths(video.stem)
                        if not path.is_file()
                    ]
                if missing:
                    raise FileNotFoundError(
                        "human_post requires completed human_pre or scene-bridge inputs:\n  "
                        + "\n  ".join(str(path) for path in missing)
                    )
        scene_enabled = bool(self._scene_config().get("enabled", False))
        if stage == "scene":
            self._check_scene_preflight(videos, require_existing_inputs=True)
        elif stage == "all" and scene_enabled:
            # `all` creates the human-pre files later in this invocation, so
            # validate only CRISP configuration/environment at this point.
            self._check_scene_preflight(videos, require_existing_inputs=False)
        if stage in {"object", "all"}:
            obj = self.config.get("object", {})
            if not obj.get("enabled", False):
                if stage == "object":
                    raise ValueError("object.enabled is false")
                return
            archive_readme = str(obj.get("archive_readme", "")).strip()
            reconstruction_root = self.root / "do-as-i-do-main" / "reconstruction"
            foundationpose_root = (
                self.root / "foundationpose-plus-plus-cutie-realtime-mask"
            )
            if (
                archive_readme
                and self._path(archive_readme).is_file()
                and (
                    not reconstruction_root.is_dir()
                    or not foundationpose_root.is_dir()
                )
            ):
                raise RuntimeError(
                    "Object reconstruction is archived. Restore it with: "
                    f"bash {archive_readme.replace('README.md', 'restore_object_reconstruction.sh')}"
                )
            mode = obj.get("mode", "monocular")
            if mode not in {"monocular", "foundationpose"}:
                raise ValueError(f"unsupported object.mode: {mode}")
            if stage == "object" and not self.dry_run:
                output_root = self._path(self.config["output"]["root"])
                configured = set(obj.get("clips", {}))
                only_configured = obj.get("only_configured_clips", True)
                missing_human_outputs = [
                    output_root / video.stem / "robot_motion.pkl"
                    for video in videos
                    if (not only_configured or video.stem in configured)
                    and not (output_root / video.stem / "robot_motion.pkl").is_file()
                ]
                if missing_human_outputs:
                    missing = "\n  ".join(
                        str(path) for path in missing_human_outputs
                    )
                    raise FileNotFoundError(
                        "--stage object only appends to an existing human/GMR "
                        "run, but these prerequisites are missing:\n  "
                        f"{missing}\nUse --stage all for a new full run, or "
                        "--output-root to select an existing human/GMR output."
                    )
            if mode == "monocular":
                mono = obj.get("monocular", {})
                prompt_mode = mono.get(
                    "segmentation_prompt_mode",
                    "auto",
                )
                if prompt_mode not in {"auto", "click"}:
                    raise ValueError(
                        "object.monocular.segmentation_prompt_mode must be "
                        f"auto or click, got {prompt_mode!r}"
                    )
                if prompt_mode == "auto":
                    detector_model = self._path(
                        mono.get("auto_detector", {}).get(
                            "model",
                            "models/yolo11n.pt",
                        )
                    )
                    if not detector_model.is_file():
                        raise FileNotFoundError(
                            f"automatic object detector model: {detector_model}"
                        )
                    detector_import = subprocess.run(
                        [
                            self.runtime_python,
                            "-c",
                            "from ultralytics import YOLO",
                        ],
                        cwd=self.root,
                        env=self.environment,
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    if detector_import.returncode:
                        detail = (
                            detector_import.stderr.strip().splitlines()[-1]
                            if detector_import.stderr.strip()
                            else f"exit={detector_import.returncode}"
                        )
                        raise RuntimeError(
                            "automatic object detector environment check "
                            f"failed: {detail}"
                        )
                reconstruction_script = self._path(
                    mono.get(
                        "reconstruction_script",
                        "do-as-i-do-main/reconstruction/run_pipeline.sh",
                    )
                )
                if not reconstruction_script.exists():
                    raise FileNotFoundError(reconstruction_script)
                if mono.get("run_reconstruction", True):
                    missing_credentials = [
                        name
                        for name in (
                            "TENCENTCLOUD_SECRET_ID",
                            "TENCENTCLOUD_SECRET_KEY",
                        )
                        if not os.environ.get(name)
                    ]
                    if missing_credentials:
                        raise RuntimeError(
                            "Hunyuan3D requires credentials in the process "
                            "environment: "
                            + ", ".join(missing_credentials)
                            + ". Source configs/secrets.env first."
                        )

                    reconstruction_root = (
                        self.root / "do-as-i-do-main" / "reconstruction"
                    )
                    required_modules = [
                        reconstruction_root / "modules" / "MoGe",
                        reconstruction_root / "modules" / "megapose6d",
                    ]
                    missing_modules = [
                        str(path.relative_to(reconstruction_root))
                        for path in required_modules
                        if not path.is_dir() or not any(path.iterdir())
                    ]
                    if missing_modules:
                        raise RuntimeError(
                            "missing RGB-only reconstruction modules: "
                            + ", ".join(missing_modules)
                        )

                    runtime = self.config.get("runtime", {})
                    conda_base = self._path(
                        runtime.get(
                            "conda_base",
                            str(Path.home() / "miniconda3"),
                        )
                    )
                    missing_envs = []
                    env_python_paths = {}
                    envs = runtime.get("conda_envs", {})
                    for key in ("samhq", "moge2", "megapose"):
                        env_name = envs.get(key, key)
                        env_path = Path(str(env_name)).expanduser()
                        if not env_path.is_absolute():
                            env_path = conda_base / "envs" / env_path
                        if not (env_path / "bin" / "python").is_file():
                            missing_envs.append(str(env_name))
                        else:
                            env_python_paths[key] = env_path / "bin" / "python"
                    if missing_envs:
                        raise RuntimeError(
                            "missing RGB-only object Conda environments: "
                            + ", ".join(sorted(set(missing_envs)))
                        )

                    import_checks = {
                        "samhq": "import cv2, torch, numpy, omegaconf",
                        "moge2": "import torch; from moge.model.v2 import MoGeModel",
                        "megapose": (
                            "import bokeh, png, torch; "
                            "from megapose.utils.load_model import load_named_model"
                        ),
                    }
                    check_environment = self.environment.copy()
                    check_environment["MEGAPOSE_DATA_DIR"] = str(
                        reconstruction_root / "weights" / "megapose"
                    )
                    broken_envs = []
                    for key, statement in import_checks.items():
                        result = subprocess.run(
                            [
                                str(env_python_paths[key]),
                                "-c",
                                statement,
                            ],
                            cwd=self.root,
                            env=check_environment,
                            capture_output=True,
                            text=True,
                            timeout=60,
                        )
                        if result.returncode:
                            detail = (
                                result.stderr.strip().splitlines()[-1]
                                if result.stderr.strip()
                                else f"exit={result.returncode}"
                            )
                            broken_envs.append(f"{key}: {detail}")
                    if broken_envs:
                        raise RuntimeError(
                            "object environment import checks failed: "
                            + "; ".join(broken_envs)
                        )

                    required_weights = [
                        reconstruction_root
                        / "weights"
                        / "moge-2-vitl-normal"
                        / "model.pt",
                        reconstruction_root
                        / "weights"
                        / "megapose"
                        / "megapose-models"
                        / "coarse-rgb-906902141"
                        / "checkpoint.pth.tar",
                        reconstruction_root
                        / "weights"
                        / "megapose"
                        / "megapose-models"
                        / "refiner-rgb-653307694"
                        / "checkpoint.pth.tar",
                    ]
                    missing_weights = [
                        str(path.relative_to(reconstruction_root))
                        for path in required_weights
                        if not path.is_file()
                    ]
                    if missing_weights:
                        raise RuntimeError(
                            "missing MoGe-2/MegaPose weights: "
                            + ", ".join(missing_weights)
                        )

                    samhq_checkpoint = mono.get("samhq_checkpoint")
                    cutie_checkpoint = mono.get("cutie_checkpoint")
                    require_seg = mono.get(
                        "require_segmentation_checkpoint", True
                    )
                    fpp_root = self.root / "foundationpose-plus-plus-main"
                    local_samhq = (
                        fpp_root / "sam-hq" / "pretrained_checkpoints"
                        / "sam_hq_vit_l.pth"
                    )
                    local_cutie = (
                        fpp_root / "Cutie" / "weights" / "cutie-base-mega.pth"
                    )
                    if samhq_checkpoint:
                        samhq_path = self._path(samhq_checkpoint)
                        if not samhq_path.is_file():
                            raise FileNotFoundError(
                                f"object.monocular.samhq_checkpoint: {samhq_path}"
                            )
                    elif require_seg:
                        if not local_samhq.is_file():
                            raise RuntimeError(
                                "SAM-HQ checkpoint not found at "
                                f"{local_samhq}. Download sam_hq_vit_l.pth to "
                                "foundationpose-plus-plus-main/sam-hq/"
                                "pretrained_checkpoints/ or set "
                                "object.monocular.samhq_checkpoint."
                            )
                    if cutie_checkpoint:
                        cutie_path = self._path(cutie_checkpoint)
                        if not cutie_path.is_file():
                            raise FileNotFoundError(
                                f"object.monocular.cutie_checkpoint: {cutie_path}"
                            )
                    elif require_seg:
                        if not local_cutie.is_file():
                            raise RuntimeError(
                                "Cutie checkpoint not found at "
                                f"{local_cutie}. Download cutie-base-mega.pth to "
                                "foundationpose-plus-plus-main/Cutie/weights/ or set "
                                "object.monocular.cutie_checkpoint."
                            )

    def run_human(self) -> None:
        admitted = self._run_fullbody_preflight()
        if not admitted:
            print(
                "[FULLBODY] no selected clips meet the full-body admission rule; "
                "skipping human/GMR inference.",
                flush=True,
            )
            return
        self._ensure_run_manifests(admitted)
        backend = self.config.get("human", {}).get(
            "backend",
            "hand4wholepp",
        )
        wrapper = (
            self.root
            / "GVHMR-hand"
            / "GVHMR-main"
            / "tools"
            / "pipeline"
            / BACKEND_WRAPPERS[backend]
        )
        environment = self.environment.copy()
        # The backend wrapper uses substring filters.  Supplying the exact
        # admitted stem list prevents rejected clips from re-entering through
        # a broad original input.clip_filter.
        environment["CLIP_FILTER"] = ",".join(video.stem for video in admitted)
        try:
            self._run(["bash", str(wrapper)], env=environment)
        except subprocess.CalledProcessError as exc:
            # The batch wrapper completes independent clips before returning a
            # non-zero status for failed ones.  Preserve those completed
            # results and let quality/product stages process only their known
            # good outputs; otherwise one bad video discards a whole batch.
            output_root = self._path(self.config["output"]["root"])
            completed = [
                video
                for video in admitted
                if (output_root / video.stem / "robot_motion.pkl").is_file()
            ]
            failed = [video for video in admitted if video not in completed]
            if not completed:
                raise
            self._eligible_clips = {video.stem for video in completed}
            failure_record = {
                "schema_version": 1,
                "wrapper": str(wrapper),
                "returncode": int(exc.returncode),
                "admitted_count": len(admitted),
                "completed_count": len(completed),
                "failed_count": len(failed),
                "completed_clips": [video.stem for video in completed],
                "failed_clips": [video.stem for video in failed],
            }
            self._human_stage_failures.append(failure_record)
            (output_root / "human_stage_failures.json").write_text(
                json.dumps(failure_record, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                "[HUMAN] batch wrapper returned "
                f"{exc.returncode}; continuing with {len(completed)}/"
                f"{len(admitted)} completed clips. Failed clips are recorded in "
                f"{output_root / 'human_stage_failures.json'}",
                flush=True,
            )
            admitted = completed
        self.write_video_aliases([video.stem for video in admitted])

    def run_human_pre(self) -> None:
        """Run through GVHMR/Locomotion/smoothing, stopping before PHC/GMR."""
        admitted = self._run_fullbody_preflight()
        if not admitted:
            print(
                "[FULLBODY] no selected clips meet the full-body admission rule; "
                "skipping human-pre inference.",
                flush=True,
            )
            return
        self._ensure_run_manifests(admitted)
        backend = self.config.get("human", {}).get("backend", "hand4wholepp")
        wrapper = (
            self.root
            / "GVHMR-hand"
            / "GVHMR-main"
            / "tools"
            / "pipeline"
            / BACKEND_WRAPPERS[backend]
        )
        environment = self.environment.copy()
        environment.update(
            {
                "CLIP_FILTER": ",".join(video.stem for video in admitted),
                "SKIP_PHC": "1",
                "RUN_GMR": "0",
            }
        )
        wrapper_error: subprocess.CalledProcessError | None = None
        try:
            self._run(["bash", str(wrapper)], env=environment)
        except subprocess.CalledProcessError as exc:
            wrapper_error = exc
        completed = [
            video
            for video in admitted
            if all(path.is_file() for path in self._scene_input_paths(video.stem))
        ]
        failed = [video for video in admitted if video not in completed]
        if failed:
            detail = ", ".join(video.stem for video in failed)
            if not completed and wrapper_error is not None:
                raise wrapper_error
            raise RuntimeError(
                "human_pre did not produce the CRISP input contract for: " + detail
            )
        if wrapper_error is not None:
            # A backend wrapper may report a non-zero batch result even when
            # every selected clip produced the complete human-pre contract.
            print(
                f"[HUMAN_PRE] wrapper returned {wrapper_error.returncode} after "
                "all required sidecar inputs were written; retaining outputs.",
                flush=True,
            )
        self._eligible_clips = {video.stem for video in completed}
        self.write_video_aliases([video.stem for video in completed])

    def _apply_smpl_2d_contact_refinement(self, videos: list[Path]) -> None:
        """Publish an accepted upstream SMPL candidate into an isolated bridge root.

        This is intentionally before the post-only GMR wrapper.  It therefore
        changes neither the canonical VideoMimic package nor an existing GMR
        output, and a failed source-motion refinement cannot be consumed by
        retargeting as though it were a successful repair.
        """
        human = self.config.get("human", {})
        refinement = human.get("smpl_2d_contact_refinement", {})
        if refinement in (None, False):
            return
        if not isinstance(refinement, dict):
            raise ValueError("human.smpl_2d_contact_refinement must be a mapping")
        if not bool(refinement.get("enabled", False)):
            return
        scene = self._scene_config()
        if not bool(scene.get("robot_bridge", {}).get("enabled", False)):
            raise ValueError(
                "SMPL 2-D contact refinement requires scene.robot_bridge.enabled=true "
                "so its candidate cannot overwrite a GVHMR baseline"
            )
        script = self.root / "scripts" / "scene" / "refine_gvhmr_2d_contact_motion.py"
        if not script.is_file():
            raise FileNotFoundError(script)
        body_model_root = self._path(
            refinement.get("body_model_root", "GMR-master/assets/body_models")
        )
        evidence_root = self._path(
            refinement.get("evidence_root", "scene_work/gvhmr_2d_contact_evidence_v1")
        )
        gmr_evidence_root = self._path(
            refinement.get(
                "gmr_evidence_root",
                str(scene.get("robot_bridge", {}).get("source_output_root", "")),
            )
        )
        gmr_evidence_motion_name = str(
            refinement.get("gmr_evidence_motion_name", "robot_motion.pkl")
        )
        robot_xml = self._path(
            refinement.get("robot_xml", "GMR-master/assets/unitree_g1/g1_mocap_29dof.xml")
        )
        source_name = str(refinement.get("source_motion_name", "001_smoothed.npz"))
        preserved_name = str(
            refinement.get("preserved_input_name", "001_smoothed_before_2d_contact_refinement.npz")
        )
        candidate_name = str(
            refinement.get("candidate_motion_name", "001_smoothed_2d_contact_refined.npz")
        )
        report_name = str(
            refinement.get("report_name", "smpl_2d_contact_refinement_report.json")
        )
        selection_name = str(
            refinement.get("selection_name", "smpl_2d_contact_refinement_selection.json")
        )
        names = (source_name, preserved_name, candidate_name, report_name, selection_name)
        if any(Path(name).name != name for name in names):
            raise ValueError("SMPL 2-D refinement artifact names must be plain filenames")
        if len({source_name, preserved_name, candidate_name}) != 3:
            raise ValueError("SMPL 2-D refinement motion artifact names must differ")
        if (
            not body_model_root.is_dir()
            or not evidence_root.is_dir()
            or not gmr_evidence_root.is_dir()
            or not robot_xml.is_file()
            or Path(gmr_evidence_motion_name).name != gmr_evidence_motion_name
        ):
            raise FileNotFoundError(
                "SMPL 2-D refinement prerequisites missing: "
                + ", ".join(
                    str(path)
                    for path in (body_model_root, evidence_root, gmr_evidence_root, robot_xml)
                )
            )
        output_root = self._path(self.config["output"]["root"])
        options = {
            "device": "device",
            "minimum_episode_frames": "minimum-episode-frames",
            "window_padding_frames": "window-padding-frames",
            "min_keypoint_confidence": "min-keypoint-confidence",
            "gmr_contact_height_m": "gmr-contact-height-m",
            "iterations": "iterations",
            "learning_rate": "learning-rate",
            "temporal_weight": "temporal-weight",
            "second_difference_weight": "second-difference-weight",
            "max_root_translation_delta_m": "max-root-translation-delta-m",
            "max_leg_joint_delta_rad": "max-leg-joint-delta-rad",
            "max_anchor_error_m": "max-anchor-error-m",
            "max_source_anchor_drift_m": "max-source-anchor-drift-m",
            "max_holdout_p95_regression_px": "max-holdout-p95-regression-px",
            "max_all_p95_regression_px": "max-all-p95-regression-px",
            "max_root_delta_speed_m_s": "max-root-delta-speed-m-s",
            "max_root_delta_acceleration_m_s2": "max-root-delta-acceleration-m-s2",
            "max_leg_delta_speed_rad_s": "max-leg-delta-speed-rad-s",
            "max_leg_delta_acceleration_rad_s2": "max-leg-delta-acceleration-rad-s2",
        }
        rejected: list[str] = []
        for video in videos:
            clip = video.stem
            clip_root = output_root / clip
            source = clip_root / source_name
            preserved = clip_root / preserved_name
            candidate = clip_root / candidate_name
            report = clip_root / report_name
            selection = clip_root / selection_name
            vitpose = clip_root / "gvhmr_out" / clip / "vitpose_wholebody.pt"
            camera = clip_root / "gvhmr_camera.npz"
            evidence = evidence_root / f"{clip}.evidence.npz"
            gmr_motion = gmr_evidence_root / clip / gmr_evidence_motion_name
            required = (source, vitpose, camera, evidence, gmr_motion)
            missing = [path for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"{clip}: SMPL 2-D refinement inputs missing:\n  "
                    + "\n  ".join(str(path) for path in missing)
                )
            # A complete accepted candidate is resumable.  Any partial or
            # rejected prior attempt is deliberately not overwritten: callers
            # must use a distinct batch output directory for another trial.
            if any(path.exists() for path in (preserved, candidate, report, selection)):
                if all(path.is_file() for path in (preserved, candidate, report, selection)):
                    prior = json.loads(report.read_text(encoding="utf-8"))
                    marker = json.loads(selection.read_text(encoding="utf-8"))
                    if (
                        prior.get("accepted") is True
                        and marker.get("selected_motion_sha256") == self._sha256_file(candidate)
                        and self._sha256_file(source) == self._sha256_file(candidate)
                    ):
                        print(f"[SMPL_2D_CONTACT] reusing accepted candidate: {clip}", flush=True)
                        continue
                raise FileExistsError(
                    f"{clip}: refusing to overwrite an existing SMPL 2-D refinement attempt in {clip_root}"
                )
            self._atomic_copy(source, preserved)
            command = [
                self.runtime_python,
                str(script),
                "--input-motion", str(preserved),
                "--camera", str(camera),
                "--vitpose", str(vitpose),
                "--evidence", str(evidence),
                "--gmr-motion", str(gmr_motion),
                "--robot-xml", str(robot_xml),
                "--body-model-root", str(body_model_root),
                "--output-motion", str(candidate),
                "--report", str(report),
            ]
            for key, flag in options.items():
                if key in refinement:
                    command.extend([f"--{flag}", str(refinement[key])])
            environment = self.environment.copy()
            # The Locomotion environment ships the SMPL-X build whose LBS
            # return contract matches the project.  A user-site package with
            # the same import name exposes an incompatible newer helper.
            environment["PYTHONNOUSERSITE"] = "1"
            scene_runtime = scene.get("runtime", {})
            if isinstance(scene_runtime, dict) and "gpu" in scene_runtime:
                environment["CUDA_VISIBLE_DEVICES"] = self._resolve_scene_gpu(scene)
            self._run(command, env=environment)
            if not candidate.is_file() or not report.is_file():
                raise RuntimeError(f"{clip}: SMPL 2-D refinement wrote no candidate/report")
            result = json.loads(report.read_text(encoding="utf-8"))
            if result.get("accepted") is not True:
                rejected.append(f"{clip}: {result.get('failures', ['unknown_rejection'])}")
                continue
            self._atomic_copy(candidate, source)
            marker = {
                "schema_version": 1,
                "kind": "accepted_smpl_2d_contact_refinement_selection",
                "baseline_motion": preserved.name,
                "baseline_motion_sha256": self._sha256_file(preserved),
                "selected_motion": source.name,
                "selected_motion_sha256": self._sha256_file(source),
                "candidate_motion": candidate.name,
                "candidate_motion_sha256": self._sha256_file(candidate),
                "report": report.name,
                "weights_modified": False,
            }
            temporary = selection.with_name(f".{selection.name}.tmp-{os.getpid()}")
            temporary.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
            temporary.replace(selection)
            print(f"[SMPL_2D_CONTACT] accepted upstream motion candidate: {clip}", flush=True)
        if rejected:
            raise RuntimeError(
                "SMPL 2-D contact refinement rejected; GMR will not consume these candidates:\n  "
                + "\n  ".join(rejected)
            )

    def _prepare_videomimic_robot_bridge(self, videos: list[Path]) -> None:
        """Materialize a separate PHC/GMR input root in the scene's coordinates.

        A VideoMimic scene is aligned to the validated fixed-camera GVHMR
        trajectory.  The target starts empty and receives an explicit whitelist
        only; copying the historical clip tree would silently inherit a stale
        PHC choice, robot motion, report, or preview.  The source human-pre
        result and its published scene package remain read-only.
        """
        scene = self._scene_config()
        bridge = scene.get("robot_bridge", {})
        if not bridge.get("enabled", False):
            return
        if self.dry_run:
            print(
                "[DRY-RUN][SCENE_BRIDGE] configuration accepted; "
                "skipping derived clip materialization.",
                flush=True,
            )
            return

        source_root = self._path(str(bridge["source_output_root"]))
        output_root = self._path(self.config["output"]["root"])
        if source_root.resolve() == output_root.resolve():
            raise RuntimeError(
                "scene.robot_bridge.source_output_root must differ from output.root"
            )
        scene_subdir = str(scene.get("output_subdir", "scene_reconstruction"))
        source_scene_subdir = str(
            bridge.get("source_scene_subdir", scene_subdir)
        )
        motion_name = str(bridge.get("motion_name", "human_motion_static_hard.npz"))
        camera_name = str(bridge.get("camera_name", "gvhmr_camera_static_hard.npz"))

        for video in videos:
            clip = video.stem
            source_clip = source_root / clip
            target_clip = output_root / clip
            source_scene = source_clip / source_scene_subdir
            motion_source = source_scene / "human" / motion_name
            camera_source = source_scene / "human" / camera_name
            required_paths = (
                source_clip / "001_smoothed.npz",
                source_clip / "001_smplx_hands.npz",
                source_clip / "gvhmr_camera.npz",
                source_clip / "gvhmr_out" / clip / "valid_video.mp4",
                source_clip / "gvhmr_out" / clip / "1_incam.mp4",
                source_scene / "scene_manifest.json",
                source_scene / "scene" / "scene_mujoco.xml",
                source_scene / "scene" / "background_mesh_collision.obj",
                motion_source,
                camera_source,
            )
            missing = [path for path in required_paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "scene.robot_bridge source contract is incomplete for "
                    f"{clip}:\n  " + "\n  ".join(str(path) for path in missing)
                )
            if target_clip.exists() or target_clip.is_symlink():
                # A prior scene stage has already materialized this derived
                # static-frame clip.  It is immutable by design and must be
                # reused by human_post rather than copied over again.
                expected_target = (
                    target_clip / "scene_robot_bridge.json",
                    target_clip / "001_smoothed.npz",
                    target_clip / "001_smplx_hands.npz",
                    target_clip / "gvhmr_camera.npz",
                    target_clip / "gvhmr_out" / clip / "1_incam.mp4",
                    target_clip / scene_subdir / "scene_manifest.json",
                )
                missing_target = [path for path in expected_target if not path.is_file()]
                if not missing_target:
                    print(
                        "[SCENE_BRIDGE] reusing canonical PHC/GMR input: "
                        f"{target_clip}",
                        flush=True,
                    )
                    continue
                allowed_initial_entries = {"run_manifest.json"}
                existing_entries = {path.name for path in target_clip.iterdir()}
                if existing_entries <= allowed_initial_entries:
                    # ``run_manifest.json`` is created before an expensive
                    # bridge run so a crash cannot leave an untracked target.
                    # It contains no inherited result and is safe to populate.
                    pass
                else:
                    raise RuntimeError(
                        "scene.robot_bridge refuses to overwrite an incomplete "
                        f"derived clip directory: {target_clip}\n  "
                        + "\n  ".join(str(path) for path in missing_target)
                    )

            target_clip.parent.mkdir(parents=True, exist_ok=True)
            target_clip.mkdir(exist_ok=True)
            for source, target in (
                (source_clip / "001_smoothed.npz", target_clip / "001_smoothed.npz"),
                (source_clip / "001_smplx_hands.npz", target_clip / "001_smplx_hands.npz"),
                (source_clip / "gvhmr_camera.npz", target_clip / "gvhmr_camera.npz"),
            ):
                shutil.copy2(source, target)
            source_video = source_clip / "gvhmr_out" / clip / "valid_video.mp4"
            target_video = target_clip / "gvhmr_out" / clip / "valid_video.mp4"
            target_video.parent.mkdir(parents=True, exist_ok=True)
            target_video.symlink_to(source_video)
            for render_name in ("1_incam.mp4", "1_incam_object.mp4"):
                source_render = source_clip / "gvhmr_out" / clip / render_name
                if source_render.is_file():
                    (target_video.parent / render_name).symlink_to(source_render)
            semantic_chair = scene.get("semantic_chair", {})
            copied_scene_subdir = (
                source_scene_subdir
                if semantic_chair.get("enabled", False)
                else scene_subdir
            )
            shutil.copytree(source_scene, target_clip / copied_scene_subdir, symlinks=True)
            for report_name in (
                "motion_continuity_report.json",
                "camera_motion_canonicalization_report.json",
                "fullbody_preflight_report.json",
            ):
                source_report = source_clip / report_name
                if source_report.is_file():
                    shutil.copy2(source_report, target_clip / report_name)
            shutil.copy2(motion_source, target_clip / "001_smoothed.npz")
            shutil.copy2(camera_source, target_clip / "gvhmr_camera.npz")

            # Any existing final selector must point at the canonical pre-PHC
            # motion until the resumed wrapper publishes the PHC result.
            final_motion = target_clip / "001_final.npz"
            if final_motion.exists() or final_motion.is_symlink():
                final_motion.unlink()
            final_motion.symlink_to("001_smoothed.npz")

            if semantic_chair.get("enabled", False):
                layout_mode = str(
                    semantic_chair.get("layout_mode", "auto_seated_body")
                )
                semantic_source_subdir = str(
                    semantic_chair.get(
                        "source_package_subdir", source_scene_subdir
                    )
                )
                semantic_source = target_clip / semantic_source_subdir
                source_primitives = (
                    semantic_source / "scene" / "primitives_mujoco.json"
                )
                source_motion = semantic_source / "human" / motion_name
                source_contact_evidence = (
                    semantic_source / "quality" / "contact_evidence.json"
                )
                semantic_target = target_clip / scene_subdir
                temporary_assets = target_clip / (
                    "." + scene_subdir.replace("/", "_") + ".semantic_assets"
                )
                required_semantic_source = (
                    semantic_source / "scene_manifest.json",
                    source_primitives,
                )
                if layout_mode == "auto_seated_body":
                    required_semantic_source += (
                        source_motion,
                        source_contact_evidence,
                    )
                missing_semantic_source = [
                    path for path in required_semantic_source if not path.is_file()
                ]
                if missing_semantic_source:
                    raise FileNotFoundError(
                        "semantic-chair source package is incomplete:\n  "
                        + "\n  ".join(map(str, missing_semantic_source))
                    )
                if semantic_source.resolve() == semantic_target.resolve():
                    raise RuntimeError(
                        "scene.semantic_chair requires output_subdir to differ "
                        "from its source_package_subdir"
                    )
                if semantic_target.exists() or semantic_target.is_symlink():
                    raise RuntimeError(
                        "semantic-chair target package already exists: "
                        f"{semantic_target}"
                    )
                if temporary_assets.exists() or temporary_assets.is_symlink():
                    raise RuntimeError(
                        "semantic-chair temporary asset directory already exists: "
                        f"{temporary_assets}"
                    )
                build_command = [
                    sys.executable,
                    str(self.root / "scripts" / "scene" / "build_semantic_chair_assets.py"),
                    "--input-primitives",
                    str(source_primitives),
                    "--output-dir",
                    str(temporary_assets),
                    "--layout-mode",
                    layout_mode,
                ]
                if layout_mode == "auto_seated_body":
                    build_command.extend(
                        [
                            "--motion-npz",
                            str(source_motion),
                            "--contact-evidence-json",
                            str(source_contact_evidence),
                        ]
                    )
                else:
                    build_command.extend(
                        [
                            "--back-sign",
                            str(int(semantic_chair.get("back_sign", 1))),
                        ]
                    )
                    if semantic_chair.get("swap_horizontal_axes", False):
                        build_command.append("--swap-horizontal-axes")
                if not semantic_chair.get("level_seat", True):
                    build_command.append("--no-level-seat")
                if semantic_chair.get("phc_backrest_collision", False):
                    build_command.append("--phc-backrest-collision")
                self._run(build_command)
                self._run(
                    [
                        sys.executable,
                        str(self.root / "scripts" / "scene" / "create_semantic_scene_variant.py"),
                        "--source-package",
                        str(semantic_source),
                        "--semantic-assets-dir",
                        str(temporary_assets),
                        "--output-package",
                        str(semantic_target),
                    ],
                )
                shutil.rmtree(temporary_assets)
                self._run(
                    [
                        sys.executable,
                        str(self.root / "scripts" / "scene" / "validate_scene_package.py"),
                        "--scene-root",
                        str(semantic_target),
                    ],
                )
                print(
                    "[SCENE_BRIDGE] published semantic chair variant: "
                    f"{semantic_target}",
                    flush=True,
                )

            source_manifest = json.loads(
                (source_scene / "scene_manifest.json").read_text(encoding="utf-8")
            )
            canonicalization = source_manifest.get("input_provenance", {}).get(
                "canonicalization", {}
            )
            provenance = {
                "schema_version": 1,
                "run_id": self._run_id,
                "coordinate_contract": str(
                    canonicalization.get("mode", "unknown_fixed_camera")
                ),
                "source_output_root": str(bridge["source_output_root"]),
                "source_clip": clip,
                "scene_subdir": scene_subdir,
                "source_scene_subdir": source_scene_subdir,
                "motion_source": f"{source_scene_subdir}/human/{motion_name}",
                "camera_source": f"{source_scene_subdir}/human/{camera_name}",
                "motion_sha256": _sha256_file(motion_source),
                "camera_sha256": _sha256_file(camera_source),
            }
            (target_clip / "scene_robot_bridge.json").write_text(
                json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                "[SCENE_BRIDGE] prepared canonical PHC/GMR input: "
                f"{target_clip}",
                flush=True,
            )

    def _apply_phc_reference_calibration(self, videos: list[Path]) -> None:
        """Optionally compensate an empirically measured PHC tracking bias.

        This is deliberately opt-in and edits only the derived bridge input,
        never the published VideoMimic motion or scene package.  Offsets are
        specified in the shared PHC/MuJoCo Z-up frame, while the bridged SMPL
        file is neg-Y-up.  A smooth ramp prevents injecting a discontinuity
        into the approach-to-sit motion.
        """
        calibration = self._scene_config().get("phc_reference_calibration", {})
        if not calibration.get("enabled", False):
            return

        offset_z_up = np.asarray(
            calibration.get("offset_z_up_m", []), dtype=np.float32
        )
        if offset_z_up.shape != (3,) or not np.all(np.isfinite(offset_z_up)):
            raise ValueError(
                "scene.phc_reference_calibration.offset_z_up_m must contain "
                "three finite values"
            )
        start_frame = int(calibration.get("ramp_start_frame", 0))
        full_frame = int(calibration.get("full_strength_frame", start_frame))
        if start_frame < 0 or full_frame < start_frame:
            raise ValueError(
                "scene.phc_reference_calibration requires "
                "0 <= ramp_start_frame <= full_strength_frame"
            )

        # inverse(Rx(+90)): [x, y, z]_zup -> [x, z, -y]_neg_y
        offset_neg_y = np.asarray(
            [offset_z_up[0], offset_z_up[2], -offset_z_up[1]], dtype=np.float32
        )
        for video in videos:
            motion_path = (
                self._path(self.config["output"]["root"])
                / video.stem
                / "001_smoothed.npz"
            )
            if not motion_path.is_file():
                raise FileNotFoundError(
                    "PHC reference calibration requires bridged motion: "
                    f"{motion_path}"
                )
            with np.load(motion_path, allow_pickle=True) as source:
                payload = {key: source[key] for key in source.files}
            trans = np.asarray(payload.get("trans"), dtype=np.float32)
            if trans.ndim != 2 or trans.shape[1] != 3:
                raise ValueError(f"Unexpected motion translation shape: {trans.shape}")
            if full_frame >= len(trans):
                raise ValueError(
                    "scene.phc_reference_calibration.full_strength_frame must be "
                    f"less than the motion frame count ({len(trans)})"
                )
            weights = np.zeros(len(trans), dtype=np.float32)
            if full_frame == start_frame:
                weights[start_frame:] = 1.0
            else:
                ramp = np.linspace(0.0, np.pi, full_frame - start_frame + 1)
                weights[start_frame : full_frame + 1] = (
                    0.5 - 0.5 * np.cos(ramp)
                ).astype(np.float32)
                weights[full_frame + 1 :] = 1.0
            payload["trans"] = trans + weights[:, None] * offset_neg_y[None, :]
            if "trans_original" in payload:
                original = np.asarray(payload["trans_original"], dtype=np.float32)
                if original.shape == trans.shape:
                    payload["trans_original"] = (
                        original + weights[:, None] * offset_neg_y[None, :]
                    )
            np.savez_compressed(motion_path, **payload)

            calibration_report = {
                "schema_version": 1,
                "purpose": "experimental_phc_tracking_bias_compensation",
                "coordinate_contract": "z_up_offset_to_neg_y_bridged_smpl",
                "offset_z_up_m": offset_z_up.astype(float).tolist(),
                "offset_neg_y_m": offset_neg_y.astype(float).tolist(),
                "ramp_start_frame": start_frame,
                "full_strength_frame": full_frame,
                "motion_sha256_after": _sha256_file(motion_path),
            }
            report_path = motion_path.parent / "phc_reference_calibration.json"
            report_path.write_text(
                json.dumps(calibration_report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                "[SCENE_BRIDGE] applied experimental PHC reference calibration "
                f"to {motion_path.name}: z-up {offset_z_up.tolist()}",
                flush=True,
            )

    def _apply_gmr_scene_contact_alignment(self, videos: list[Path]) -> str | None:
        """Derive a GMR-only contact-aligned motion without moving the scene.

        This is the deliberately small CRISP-style addition: source SMPL contact
        anchors constrain a semantic chair's local support planes, then the G1
        torso proxy is aligned only over evidence-backed contact episodes.
        """
        gmr_cfg = self.config.get("gmr", {})
        alignment = gmr_cfg.get("scene_contact_alignment", {})
        if not isinstance(alignment, dict) or not alignment.get("enabled", False):
            return None
        # A temporally local root translation can hide a chair-contact residual
        # while breaking the GVHMR trajectory contract (the visible "slide to
        # the backrest" failure).  It is never a default batch repair.  We
        # still derive immutable SMPL support anchors for auditing; a legacy
        # correction must be explicitly requested and is only a side candidate.
        mode = str(alignment.get("mode", "audit_only"))
        if mode not in {"audit_only", "legacy_temporal_root_correction"}:
            raise ValueError(
                "gmr.scene_contact_alignment.mode must be audit_only or "
                "legacy_temporal_root_correction"
            )

        scene = self._scene_config()
        if not scene.get("enabled", False):
            raise ValueError("gmr.scene_contact_alignment requires scene.enabled=true")
        output_name = str(
            alignment.get("output_name", "robot_motion_scene_contact_aligned.pkl")
        )
        if Path(output_name).name != output_name or output_name == "robot_motion.pkl":
            raise ValueError(
                "gmr.scene_contact_alignment.output_name must be a non-default "
                "filename in the clip root"
            )

        runtime = self.config.get("runtime", {})
        conda_base = self._path(
            runtime.get("conda_base", str(Path.home() / "miniconda3"))
        )
        conda_envs = runtime.get("conda_envs", {})
        anchor_python = (
            conda_base
            / "envs"
            / str(alignment.get("anchor_conda_env", conda_envs.get("vm1rs", "vm1rs")))
            / "bin"
            / "python"
        )
        gmr_python = (
            conda_base
            / "envs"
            / str(
                alignment.get(
                    "gmr_conda_env", conda_envs.get("locomotion", "locomotion")
                )
            )
            / "bin"
            / "python"
        )
        for interpreter in (anchor_python, gmr_python):
            if not interpreter.is_file():
                raise FileNotFoundError(
                    f"scene-contact alignment interpreter not found: {interpreter}"
                )

        output_root = self._path(self.config["output"]["root"])
        scene_subdir = str(scene.get("output_subdir", "scene_reconstruction"))
        primitive_name = str(
            alignment.get(
                "chair_primitives_name", "semantic_chair_primitives_mujoco.json"
            )
        )
        source_name = str(alignment.get("source_motion_name", "001_phc_smoothed.npz"))
        anchor_name = str(alignment.get("anchor_name", "smpl_support_contacts.npz"))
        anchor_report_name = str(
            alignment.get("anchor_report_name", "smpl_support_contacts_report.json")
        )
        report_name = str(
            alignment.get(
                "report_name", "gmr_scene_contact_alignment_report.json"
            )
        )
        body_model_root = self._path(
            alignment.get("body_model_root", "GMR-master/assets/body_models")
        )
        robot_xml = self._path(
            alignment.get(
                "robot_xml", "GMR-master/assets/unitree_g1/g1_mocap_29dof.xml"
            )
        )
        contact_script = self.root / "scripts" / "scene" / "extract_smpl_support_contacts.py"
        align_script = self.root / "scripts" / "scene" / "align_gmr_robot_to_scene_contact.py"
        required_scripts = (contact_script, align_script, body_model_root, robot_xml)
        missing_scripts = [str(path) for path in required_scripts if not path.exists()]
        if missing_scripts:
            raise FileNotFoundError(
                "scene-contact alignment prerequisites missing:\n  "
                + "\n  ".join(missing_scripts)
            )

        for video in videos:
            clip_dir = output_root / video.stem
            scene_dir = clip_dir / scene_subdir / "scene"
            human_motion = clip_dir / source_name
            chair_primitives = scene_dir / primitive_name
            raw_robot_motion = clip_dir / "robot_motion.pkl"
            anchors = clip_dir / anchor_name
            anchors_report = clip_dir / anchor_report_name
            aligned_motion = clip_dir / output_name
            report = clip_dir / report_name
            missing = [
                path
                for path in (human_motion, chair_primitives, raw_robot_motion)
                if not path.is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"{video.stem}: scene-contact alignment inputs missing:\n  "
                    + "\n  ".join(str(path) for path in missing)
                )

            print(
                "[SCENE_BRIDGE] deriving CRISP-style SMPL support anchors "
                f"for {video.stem}",
                flush=True,
            )
            self._run(
                [
                    str(anchor_python),
                    str(contact_script),
                    "--human-motion",
                    str(human_motion),
                    "--chair-primitives",
                    str(chair_primitives),
                    "--body-model-root",
                    str(body_model_root),
                    "--anchors",
                    str(anchors),
                    "--report",
                    str(anchors_report),
                ]
            )
            if mode == "audit_only":
                audit_only_report = {
                    "schema_version": 2,
                    "status": "audited_no_root_trajectory_modification",
                    "mode": mode,
                    "source_robot_motion": str(raw_robot_motion),
                    "support_anchors": str(anchors),
                    "policy": (
                        "SMPL support anchors are temporal/semantic evidence only; "
                        "they cannot induce a local GMR root translation without "
                        "joint visual, scene-geometry, and mj_step acceptance."
                    ),
                }
                report.write_text(
                    json.dumps(audit_only_report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(
                    "[SCENE_BRIDGE] retained immutable visual GMR motion; "
                    "wrote support-anchor audit only for " + video.stem,
                    flush=True,
                )
                continue

            print(
                "[SCENE_BRIDGE] deriving explicit legacy contact-aligned GMR candidate "
                f"for {video.stem}",
                flush=True,
            )
            command = [
                str(gmr_python),
                str(align_script),
                "--robot-motion",
                str(raw_robot_motion),
                "--chair-primitives",
                str(chair_primitives),
                "--contact-anchors",
                str(anchors),
                "--robot-xml",
                str(robot_xml),
                "--output",
                str(aligned_motion),
                "--report",
                str(report),
                "--clearance-m",
                str(alignment.get("clearance_m", 0.025)),
                "--max-shift-m",
                str(alignment.get("max_shift_m", 0.30)),
                "--min-back-contact-coverage",
                str(alignment.get("min_back_contact_coverage", 0.30)),
                "--ramp-frames",
                str(alignment.get("ramp_frames", 6)),
                "--bridge-max-gap",
                str(alignment.get("bridge_max_gap", 2)),
            ]
            torso_body = alignment.get("torso_body")
            if torso_body:
                command.extend(["--torso-body", str(torso_body)])
            self._run(command)
            if not aligned_motion.is_file() or not report.is_file():
                raise RuntimeError(
                    f"{video.stem}: scene-contact alignment did not produce "
                    f"{output_name} and its report"
                )
        # Audit-only mode writes semantic evidence but deliberately creates no
        # replacement robot motion. Downstream mj_step validation must retain
        # the immutable visual GMR track in that case.
        return None if mode == "audit_only" else output_name

    def _run_gmr_scene_physics_validation(
        self,
        videos: list[Path],
        reference_motion_name: str,
    ) -> None:
        """Audit a rendered GMR track and separately validate MuJoCo contact physics.

        The original visual track is immutable.  The optional seat-projected
        candidate and the ``mj_step`` result are quality-labelled side outputs.
        """
        gmr_cfg = self.config.get("gmr", {})
        physics = gmr_cfg.get("scene_physics_validation", {})
        if not isinstance(physics, dict) or not physics.get("enabled", False):
            return
        scene = self._scene_config()
        if not scene.get("enabled", False):
            raise ValueError("gmr.scene_physics_validation requires scene.enabled=true")

        runtime = self.config.get("runtime", {})
        conda_base = self._path(runtime.get("conda_base", str(Path.home() / "miniconda3")))
        conda_envs = runtime.get("conda_envs", {})
        gmr_python = (
            conda_base / "envs"
            / str(physics.get("gmr_conda_env", conda_envs.get("locomotion", "locomotion")))
            / "bin" / "python"
        )
        robot_xml = self._path(
            physics.get("robot_xml", "GMR-master/assets/unitree_g1/g1_mocap_29dof.xml")
        )
        scripts = {
            "audit": self.root / "scripts" / "scene" / "evaluate_gmr_chair_contacts.py",
            "projection": self.root / "scripts" / "scene" / "project_gmr_seat_contact.py",
            "ramp": self.root / "scripts" / "scene" / "ramp_gmr_root_contact_translation.py",
            "dynamic": self.root / "scripts" / "scene" / "simulate_gmr_scene_contacts.py",
            "visible_ground": self.root / "scripts" / "scene" / "calibrate_visible_ground_contacts.py",
            "static_fit": self.root / "scripts" / "scene" / "fit_static_chair_to_frozen_gmr.py",
            "static_image_contract": self.root / "scripts" / "scene" / "audit_static_support_image_contract.py",
        }
        missing = [str(path) for path in (gmr_python, robot_xml, scripts["audit"]) if not path.is_file()]
        if missing:
            raise FileNotFoundError("GMR scene-physics prerequisites missing:\n  " + "\n  ".join(missing))

        output_root = self._path(self.config["output"]["root"])
        scene_mujoco_name = str(physics.get("scene_mujoco_name", gmr_cfg.get("scene_mujoco_name", "")))
        if not scene_mujoco_name:
            raise ValueError("gmr.scene_physics_validation.scene_mujoco_name is required")
        anchor_name = str(physics.get("anchor_name", "smpl_support_contacts.npz"))
        seat_geom = str(physics.get("seat_geom", "seat_support_geom"))
        clearance_m = float(physics.get("clearance_m", 0.003))

        def read_json(path: Path) -> dict[str, Any]:
            if not path.is_file():
                return {"status": "missing", "path": str(path)}
            return json.loads(path.read_text(encoding="utf-8"))

        for video in videos:
            clip_dir = output_root / video.stem
            reference_motion = clip_dir / reference_motion_name
            scene_mujoco_xml = clip_dir / scene_mujoco_name
            anchors = clip_dir / anchor_name
            audit_report = clip_dir / str(physics.get("audit_report_name", "gmr_scene_contact_audit.json"))
            audit_arrays = clip_dir / str(physics.get("audit_arrays_name", "gmr_scene_contact_audit.npz"))
            missing_inputs = [path for path in (reference_motion, scene_mujoco_xml, anchors) if not path.is_file()]
            if missing_inputs:
                raise FileNotFoundError(
                    f"{video.stem}: scene-physics inputs missing:\n  " + "\n  ".join(str(path) for path in missing_inputs)
                )

            input_reference_motion = reference_motion
            visible_ground_state: dict[str, Any] = {"status": "disabled"}
            visible_ground_cfg = physics.get("visible_ground_calibration", {})
            if isinstance(visible_ground_cfg, dict) and visible_ground_cfg.get("enabled", False):
                if not scripts["visible_ground"].is_file():
                    raise FileNotFoundError(scripts["visible_ground"])
                calibrated_motion = clip_dir / str(
                    visible_ground_cfg.get("output_name", "robot_motion_visible_ground_contact.pkl")
                )
                calibrated_report = clip_dir / str(
                    visible_ground_cfg.get("report_name", "gmr_visible_ground_contact.json")
                )
                if calibrated_motion == reference_motion:
                    raise ValueError("visible_ground_calibration.output_name must not overwrite its input")
                command = [
                    str(gmr_python), str(scripts["visible_ground"]),
                    "--input-motion", str(reference_motion),
                    "--robot-xml", str(robot_xml),
                    "--contact-anchors", str(anchors),
                    "--output-motion", str(calibrated_motion),
                    "--report", str(calibrated_report),
                ]
                option_map = {
                    "left_foot_body": "left-foot-body",
                    "right_foot_body": "right-foot-body",
                    "target_clearance_m": "target-clearance-m",
                    "max_global_offset_m": "max-global-offset-m",
                    "max_support_spread_m": "max-support-spread-m",
                    "min_support_coverage": "min-support-coverage",
                    "sit_fade_frames": "sit-fade-frames",
                    "repair_trigger_m": "repair-trigger-m",
                    "max_stance_clearance_m": "max-stance-clearance-m",
                    "min_stance_repair_frames": "min-stance-repair-frames",
                    "temporal_padding_frames": "temporal-padding-frames",
                    "max_local_joint_delta_rad": "max-local-joint-delta-rad",
                    "max_local_root_z_m": "max-local-root-z-m",
                    "root_z_regularization": "root-z-regularization",
                    "root_temporal_weight": "root-temporal-weight",
                    "stance_contact_weight": "stance-contact-weight",
                    "contact_safety_margin_m": "contact-safety-margin-m",
                }
                for key, flag in option_map.items():
                    if key in visible_ground_cfg:
                        command.extend([f"--{flag}", str(visible_ground_cfg[key])])
                self._run(command)
                visible_ground_state = read_json(calibrated_report)
                if visible_ground_state.get("accepted") and calibrated_motion.is_file():
                    reference_motion = calibrated_motion
                    visible_ground_state["selected_for_static_and_mj_step"] = True
                else:
                    visible_ground_state["selected_for_static_and_mj_step"] = False
            # GVHMR is the visual-coordinate authority.  If seated support is
            # inconsistent, first calibrate the *static chair* from a stable
            # semantic support episode; never introduce a per-frame person/root
            # translation.  The candidate scene is admitted only after an image
            # edge contract, so this remains batch-safe under different chairs.
            selected_scene_mujoco_xml = scene_mujoco_xml
            static_fit_state: dict[str, Any] = {"status": "disabled"}
            static_fit_cfg = physics.get("static_scene_support_fit", {})
            if isinstance(static_fit_cfg, dict) and static_fit_cfg.get("enabled", False):
                if not scripts["static_fit"].is_file():
                    raise FileNotFoundError(scripts["static_fit"])
                source_scene_dir = scene_mujoco_xml.parent.parent
                output_scene_dir = clip_dir / str(
                    static_fit_cfg.get("output_scene_dir_name", "scene_reconstruction_semantic_support")
                )
                static_report = clip_dir / str(
                    static_fit_cfg.get("report_name", "gmr_static_scene_support_fit.json")
                )
                static_command = [
                    str(gmr_python), str(scripts["static_fit"]),
                    "--robot-motion", str(reference_motion),
                    "--contact-anchors", str(anchors),
                    "--robot-xml", str(robot_xml),
                    "--source-scene-dir", str(source_scene_dir),
                    "--output-scene-dir", str(output_scene_dir),
                    "--report", str(static_report),
                    "--seat-geom", seat_geom,
                ]
                fit_options = {
                    "support_body": "support-body",
                    "support_surface": "support-surface",
                    "support_footprint_margin_m": "support-footprint-margin-m",
                    "torso_body": "torso-body",
                    "backrest_geom": "backrest-geom",
                    "seat_target_m": "seat-target-m",
                    "back_target_m": "back-target-m",
                    "max_xy_m": "max-xy-m",
                    "max_z_m": "max-z-m",
                    "max_yaw_deg": "max-yaw-deg",
                    "max_fit_frames": "max-fit-frames",
                    "settled_tail_frames": "settled-tail-frames",
                    "max_lateral_shift_m": "max-lateral-shift-m",
                    "seat_weight": "seat-weight",
                    "back_weight": "back-weight",
                    "approach_contact_weight": "approach-contact-weight",
                    "approach_max_frames": "approach-max-frames",
                    "maxiter": "maxiter",
                    "max_soft_overlap_m": "max-soft-overlap-m",
                    "max_overlap_ratio": "max-overlap-ratio",
                    "max_median_seat_gap_m": "max-median-seat-gap-m",
                    "full_body_collision_weight": "full-body-collision-weight",
                    "max_full_body_collision_frames": "max-full-body-collision-frames",
                    "max_full_body_penetration_m": "max-full-body-penetration-m",
                    "max_full_body_penetration_frame_ratio": "max-full-body-penetration-frame-ratio",
                }
                for key, flag in fit_options.items():
                    if key in static_fit_cfg:
                        static_command.extend([f"--{flag}", str(static_fit_cfg[key])])
                try:
                    self._run(static_command)
                    static_fit_state = {"fit": read_json(static_report)}
                    fitted_scene_mujoco = output_scene_dir / "scene" / scene_mujoco_xml.name
                    fit_accepted = (
                        static_fit_state["fit"].get("status") == "accepted"
                        and fitted_scene_mujoco.is_file()
                    )
                    image_cfg = static_fit_cfg.get("image_contract", {})
                    image_accepted = False
                    if fit_accepted and isinstance(image_cfg, dict) and image_cfg.get("enabled", True):
                        if not scripts["static_image_contract"].is_file():
                            raise FileNotFoundError(scripts["static_image_contract"])
                        camera_name = str(image_cfg.get(
                            "camera_path_name",
                            f"{source_scene_dir.name}/human/gvhmr_camera_static_hard.npz",
                        ))
                        keypoint_name = str(image_cfg.get(
                            "keypoints_name_template",
                            "gvhmr_out/{clip}/vitpose_wholebody.pt",
                        )).format(clip=video.stem)
                        image_report = clip_dir / str(image_cfg.get(
                            "report_name", "gmr_static_scene_support_image_contract.json"
                        ))
                        image_command = [
                            str(gmr_python), str(scripts["static_image_contract"]),
                            "--video", str(video),
                            "--camera-path", str(clip_dir / camera_name),
                            "--keypoints", str(clip_dir / keypoint_name),
                            "--baseline-primitives", str(source_scene_dir / "scene" / "semantic_chair_primitives_mujoco.json"),
                            "--candidate-primitives", str(output_scene_dir / "scene" / "semantic_chair_primitives_mujoco.json"),
                            "--report", str(image_report),
                        ]
                        image_options = {
                            "max_frames": "max-frames",
                            "edge_step_m": "edge-step-m",
                            "human_margin_px": "human-margin-px",
                            "min_points_per_frame": "min-points-per-frame",
                            "min_total_points": "min-total-points",
                            "max_p50_regression_px": "max-p50-regression-px",
                            "max_p90_regression_px": "max-p90-regression-px",
                        }
                        for key, flag in image_options.items():
                            if key in image_cfg:
                                image_command.extend([f"--{flag}", str(image_cfg[key])])
                        self._run(image_command)
                        static_fit_state["image_contract"] = read_json(image_report)
                        image_accepted = (
                            static_fit_state["image_contract"].get("status")
                            == "accepted_image_contract"
                        )
                    if fit_accepted and image_accepted:
                        selected_scene_mujoco_xml = fitted_scene_mujoco
                        static_fit_state["status"] = "accepted_scene_side_support_calibration"
                    elif fit_accepted:
                        static_fit_state["status"] = "rejected_by_missing_or_failed_image_contract"
                    else:
                        static_fit_state["status"] = "rejected_by_static_fit"
                except subprocess.CalledProcessError as exc:
                    static_fit_state = {
                        "status": "failed_to_construct",
                        "returncode": exc.returncode,
                        "note": "Immutable visual GMR and the original scene remain available.",
                    }

            self._run([
                str(gmr_python), str(scripts["audit"]),
                "--robot-motion", str(reference_motion),
                "--scene-mujoco-xml", str(selected_scene_mujoco_xml),
                "--robot-xml", str(robot_xml),
                "--contact-anchors", str(anchors),
                "--report", str(audit_report), "--arrays", str(audit_arrays),
                "--seat-geom", seat_geom, "--clearance-m", str(clearance_m),
            ])

            backrest_cfg = physics.get("backrest_projection", {})
            robot_projection_allowed = bool(
                physics.get("allow_experimental_robot_motion_projection", False)
            )
            projection_input = reference_motion
            backrest_state: dict[str, Any] = {"status": "disabled"}
            if isinstance(backrest_cfg, dict) and backrest_cfg.get("enabled", False) and not robot_projection_allowed:
                backrest_state = {"status": "skipped_immutable_motion_policy"}
                backrest_cfg = {}
            if isinstance(backrest_cfg, dict) and backrest_cfg.get("enabled", False):
                if not scripts["projection"].is_file() or not scripts["ramp"].is_file():
                    raise FileNotFoundError(
                        "backrest projection requires "
                        f"{scripts['projection']} and {scripts['ramp']}"
                    )
                raw_output = clip_dir / str(
                    backrest_cfg.get("raw_output_name", "robot_motion_backrest_raw.pkl")
                )
                raw_report = clip_dir / str(
                    backrest_cfg.get("raw_report_name", "gmr_backrest_raw_projection.json")
                )
                ramped_output = clip_dir / str(
                    backrest_cfg.get("ramped_output_name", "robot_motion_backrest_ramped.pkl")
                )
                ramped_report = clip_dir / str(
                    backrest_cfg.get("ramped_report_name", "gmr_backrest_temporal_ramp.json")
                )
                raw_command = [
                    str(gmr_python), str(scripts["projection"]),
                    "--robot-motion", str(projection_input),
                    "--scene-mujoco-xml", str(scene_mujoco_xml),
                    "--robot-xml", str(robot_xml), "--contact-anchors", str(anchors),
                    "--output", str(raw_output), "--report", str(raw_report),
                    "--seat-geom", seat_geom,
                    "--backrest-geom", str(backrest_cfg.get("backrest_geom", "backrest_geom")),
                    "--support-body", str(backrest_cfg.get("support_body", "right_hip_pitch_link")),
                    "--back-support-body", str(backrest_cfg.get("back_support_body", "torso_link")),
                    "--clearance-m", str(backrest_cfg.get("clearance_m", clearance_m)),
                    "--support-target-m", str(backrest_cfg.get("support_target_m", 0.008)),
                    "--activation-distance-m", str(backrest_cfg.get("activation_distance_m", 0.25)),
                    "--back-contact-target-m", str(backrest_cfg.get("back_contact_target_m", 0.04)),
                    "--max-back-shift-m", str(backrest_cfg.get("max_back_shift_m", 0.20)),
                    "--back-objective-weight", str(backrest_cfg.get("back_objective_weight", 40.0)),
                    # The next stage performs the temporal ramp, then seat projection
                    # re-establishes clearance around that ramped XY trajectory.
                    "--back-ramp-frames", "0",
                    "--min-root-dz-m", str(backrest_cfg.get("min_root_dz_m", -0.10)),
                    "--max-root-dz-m", str(backrest_cfg.get("max_root_dz_m", 0.10)),
                    "--sample-stride", str(backrest_cfg.get("sample_stride", 1)),
                    "--maxiter", str(backrest_cfg.get("maxiter", 350)),
                ]
                try:
                    self._run(raw_command)
                    self._run([
                        str(gmr_python), str(scripts["ramp"]),
                        "--reference-motion", str(reference_motion),
                        "--contact-motion", str(raw_output),
                        "--contact-anchors", str(anchors),
                        "--output", str(ramped_output), "--report", str(ramped_report),
                        "--contact-key", str(backrest_cfg.get("contact_key", "back_contact")),
                        "--ramp-frames", str(backrest_cfg.get("ramp_frames", 10)),
                    ])
                    if not raw_output.is_file() or not ramped_output.is_file():
                        raise RuntimeError(f"{video.stem}: backrest projection did not produce its motion outputs")
                    projection_input = ramped_output
                    backrest_state = {
                        "status": "accepted",
                        "raw_projection": read_json(raw_report),
                        "temporal_ramp": read_json(ramped_report),
                    }
                except subprocess.CalledProcessError as exc:
                    backrest_state = {
                        "status": "failed_to_construct", "returncode": exc.returncode,
                        "note": "Visual GMR remains available; backrest correction is a quality side output.",
                    }
                    if backrest_cfg.get("required", False):
                        raise

            projection_cfg = physics.get("seat_projection", {})
            projection_output: Path | None = None
            projection_state: dict[str, Any] = {"status": "disabled"}
            if isinstance(projection_cfg, dict) and projection_cfg.get("enabled", False) and robot_projection_allowed:
                if not scripts["projection"].is_file():
                    raise FileNotFoundError(scripts["projection"])
                projection_output = clip_dir / str(projection_cfg.get("output_name", "robot_motion_scene_seat_projected.pkl"))
                projection_report = clip_dir / str(projection_cfg.get("report_name", "gmr_seat_projection_report.json"))
                command = [
                    str(gmr_python), str(scripts["projection"]),
                    "--robot-motion", str(reference_motion),
                    "--scene-mujoco-xml", str(scene_mujoco_xml),
                    "--robot-xml", str(robot_xml), "--contact-anchors", str(anchors),
                    "--output", str(projection_output), "--report", str(projection_report),
                    "--seat-geom", seat_geom,
                    "--clearance-m", str(projection_cfg.get("clearance_m", clearance_m)),
                    "--support-target-m", str(projection_cfg.get("support_target_m", 0.008)),
                    "--activation-distance-m", str(projection_cfg.get("activation_distance_m", 0.25)),
                    "--min-root-dz-m", str(projection_cfg.get("min_root_dz_m", -0.10)),
                    "--max-root-dz-m", str(projection_cfg.get("max_root_dz_m", 0.10)),
                    "--sample-stride", str(projection_cfg.get("sample_stride", 1)),
                    "--maxiter", str(projection_cfg.get("maxiter", 350)),
                ]
                try:
                    self._run(command)
                    projection_state = read_json(projection_report)
                    if not projection_output.is_file():
                        raise RuntimeError(f"{video.stem}: seat projection did not produce {projection_output}")
                except subprocess.CalledProcessError as exc:
                    projection_output = None
                    projection_state = {
                        "status": "failed_to_construct", "returncode": exc.returncode,
                        "note": "Visual GMR remains available; this is a quality outcome, not a pipeline failure.",
                    }
                    if projection_cfg.get("required", False):
                        raise

            dynamic_cfg = physics.get("dynamic_validation", {})
            dynamic_state: dict[str, Any] = {"status": "disabled"}
            driven_visual_replay = bool(
                isinstance(dynamic_cfg, dict)
                and dynamic_cfg.get("driven_visual_replay", False)
            )
            if (
                isinstance(dynamic_cfg, dict)
                and dynamic_cfg.get("enabled", False)
                and (
                    bool(physics.get("allow_experimental_external_root_tracking_harness", False))
                    or driven_visual_replay
                )
            ):
                if not scripts["dynamic"].is_file():
                    raise FileNotFoundError(scripts["dynamic"])
                source_kind = str(dynamic_cfg.get("source", "seat_projected"))
                if source_kind == "seat_projected" and projection_output is None:
                    dynamic_state = {"status": "skipped_no_feasible_projection"}
                else:
                    dynamic_source = projection_output if source_kind == "seat_projected" else reference_motion
                    dynamic_output = clip_dir / str(dynamic_cfg.get("output_name", "robot_motion_dynamic_validation.pkl"))
                    dynamic_report = clip_dir / str(dynamic_cfg.get("report_name", "gmr_dynamic_validation.json"))
                    command = [
                        str(gmr_python), str(scripts["dynamic"]),
                        "--robot-motion", str(dynamic_source),
                        "--scene-mujoco-xml", str(selected_scene_mujoco_xml),
                        "--robot-xml", str(robot_xml), "--contact-anchors", str(anchors),
                        "--output-motion", str(dynamic_output), "--report", str(dynamic_report),
                        "--seat-geom", seat_geom,
                    ]
                    option_map = {
                        "substeps": "substeps", "joint_kp": "joint-kp", "joint_kd": "joint-kd",
                        "joint_torque_limit": "joint-torque-limit", "root_pos_kp": "root-pos-kp",
                        "root_pos_kd": "root-pos-kd", "root_force_limit": "root-force-limit",
                        "root_rot_kp": "root-rot-kp", "root_rot_kd": "root-rot-kd",
                        "root_torque_limit": "root-torque-limit", "max_penetration_m": "max-penetration-m",
                        "max_penetration_frame_ratio": "max-penetration-frame-ratio",
                        "max_root_rmse_m": "max-root-rmse-m", "max_joint_median_rad": "max-joint-median-rad",
                        "root_tracking_mode": "root-tracking-mode", "joint_tracking_mode": "joint-tracking-mode",
                        "enable_ground_contact": "enable-ground-contact",
                        "ground_contact_surface": "ground-contact-surface",
                        "ground_contact_bodies": "ground-contact-bodies",
                        "seat_contact_bodies": "seat-contact-bodies",
                        "seat_contact_surface": "seat-contact-surface",
                        "support_footprint_margin_m": "support-footprint-margin-m",
                        "min_seat_support_frame_ratio": "min-seat-support-frame-ratio",
                        "settled_tail_frames": "settled-tail-frames",
                    }
                    static_support_body = None
                    if str(dynamic_cfg.get("seat_contact_bodies", "")) == "auto_static_support":
                        static_support_body = static_fit_state.get("fit", {}).get("support_body")
                        if (
                            static_fit_state.get("status") != "accepted_scene_side_support_calibration"
                            or not isinstance(static_support_body, str)
                            or not static_support_body
                        ):
                            dynamic_state = {"status": "skipped_no_accepted_static_support"}
                    if dynamic_state.get("status") != "skipped_no_accepted_static_support":
                        for key, flag in option_map.items():
                            if key not in dynamic_cfg:
                                continue
                            value = dynamic_cfg[key]
                            if key == "seat_contact_bodies" and value == "auto_static_support":
                                value = static_support_body
                            if key == "enable_ground_contact":
                                command.append("--enable-ground-contact" if bool(value) else "--no-enable-ground-contact")
                                continue
                            command.extend([f"--{flag}", str(value)])
                        self._run(command)
                        dynamic_state = read_json(dynamic_report)

            dynamic_contact_replay_pass = dynamic_state.get("status") == "pass"
            dynamic_autonomous_pass = (
                dynamic_contact_replay_pass
                and bool(dynamic_state.get("autonomous_free_base", False))
            )
            projection_accepted = projection_state.get("status") == "accepted"
            final_status = (
                "complete_with_mujoco_dynamic_pass" if dynamic_autonomous_pass else
                "complete_with_mujoco_contact_constrained_replay" if dynamic_contact_replay_pass else
                "complete_with_visual_fidelity_and_kinematic_physics_candidate" if projection_accepted else
                "complete_with_visual_fidelity_only"
            )
            quality_report = {
                "schema_version": 1,
                "purpose": "gmr_scene_physics_quality_routing",
                "visual_motion_reference": input_reference_motion.name,
                "physics_motion_reference": reference_motion.name,
                "visible_ground_calibration": visible_ground_state,
                "visual_motion_is_not_replaced": True,
                "kinematic_contact_audit": read_json(audit_report),
                "static_scene_support_fit": static_fit_state,
                "selected_scene_mujoco_xml": str(selected_scene_mujoco_xml),
                "backrest_projection": backrest_state,
                "seat_projection": projection_state,
                "dynamic_validation": dynamic_state,
                "mujoco_contact_constrained_replay_pass": dynamic_contact_replay_pass,
                "mujoco_dynamic_pass": dynamic_autonomous_pass,
                "final_status": final_status,
            }
            quality_path = clip_dir / str(physics.get("quality_report_name", "gmr_scene_physics_quality_report.json"))
            quality_path.write_text(json.dumps(quality_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(
                f"[SCENE_BRIDGE] {video.stem}: {final_status}; "
                f"contact_replay_pass={dynamic_contact_replay_pass}; "
                f"autonomous_pass={dynamic_autonomous_pass}",
                flush=True,
            )

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _safe_scene_package_path(root: Path, value: Any, *, label: str) -> Path:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} must be a non-empty relative path")
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{label} must be a safe relative path")
        resolved_root = root.resolve()
        resolved = (root / relative).resolve()
        try:
            resolved.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(f"{label} escapes its package root") from exc
        return resolved

    def _verify_scene_file_record(
        self,
        root: Path,
        record: Any,
        *,
        label: str,
        expected_relative: str | None = None,
        explicit_path: Path | None = None,
    ) -> Path:
        if not isinstance(record, dict):
            raise ValueError(f"{label} provenance must be an object")
        if expected_relative is not None and record.get("relative_path") != expected_relative:
            raise ValueError(f"{label} provenance path must be {expected_relative!r}")
        expected_digest = record.get("sha256")
        if not isinstance(expected_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
            raise ValueError(f"{label} provenance has no valid SHA-256")
        expected_bytes = record.get("bytes")
        if not isinstance(expected_bytes, int) or expected_bytes <= 0:
            raise ValueError(f"{label} provenance has no valid byte count")
        path = explicit_path or self._safe_scene_package_path(root, record.get("relative_path"), label=label)
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise RuntimeError(f"{label} is missing or differs in size: {path}")
        if self._sha256_file(path) != expected_digest:
            raise RuntimeError(f"{label} SHA-256 does not match its provenance: {path}")
        return path

    def _verify_scene_npz_record(
        self,
        path: Path,
        record: Any,
        *,
        label: str,
        root: Path,
        expected_relative: str | None = None,
    ) -> Path:
        verified = self._verify_scene_file_record(
            root,
            record,
            label=label,
            expected_relative=expected_relative,
            explicit_path=path,
        )
        frame_count = record.get("frame_count") if isinstance(record, dict) else None
        shapes = record.get("required_shapes") if isinstance(record, dict) else None
        if not isinstance(frame_count, int) or frame_count <= 0 or not isinstance(shapes, dict) or not shapes:
            raise ValueError(f"{label} provenance lacks temporal shape metadata")
        with np.load(verified, allow_pickle=False) as archive:
            for name, expected_shape in shapes.items():
                if (
                    not isinstance(name, str)
                    or not isinstance(expected_shape, list)
                    or not expected_shape
                    or name not in archive.files
                ):
                    raise RuntimeError(f"{label} temporal schema is invalid for {name!r}")
                actual_shape = [int(size) for size in np.asarray(archive[name]).shape]
                if actual_shape != expected_shape or actual_shape[0] != frame_count:
                    raise RuntimeError(f"{label}.{name} shape does not match its provenance")
        return verified

    def _read_scene_checksums(self, scene_root: Path) -> dict[str, str]:
        checksum_path = scene_root / "checksums.sha256"
        if not checksum_path.is_file():
            raise FileNotFoundError(f"scene package lacks checksums: {checksum_path}")
        checksums: dict[str, str] = {}
        for line_number, raw in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw:
                continue
            parts = raw.split(maxsplit=1)
            if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
                raise ValueError(f"invalid scene checksum at {checksum_path}:{line_number}")
            relative = Path(parts[1])
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError(f"unsafe scene checksum path at {checksum_path}:{line_number}")
            name = relative.as_posix()
            if name in checksums:
                raise ValueError(f"duplicate scene checksum path: {name}")
            checksums[name] = parts[0]
        return checksums

    @staticmethod
    def _atomic_copy(source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        shutil.copy2(source, temporary)
        temporary.replace(target)

    def _prepare_scene_aware_post_inputs(self, video: Path) -> dict[str, Path] | None:
        """Validate a VideoMimic package and stage an isolated PHC/GMR candidate.

        Static-camera canonicalization is an input coordinate transform, not a
        license to overwrite the original human/GMR result.  Every candidate
        gets its own output root so PHC caches and robot_motion.pkl cannot leak
        across clips or attempts.
        """
        scene = self._scene_config()
        physics = scene.get("physics", {})
        if not isinstance(physics, dict) or not bool(physics.get("enabled", False)):
            return None
        if scene.get("backend") != "videomimic_nksr":
            raise ValueError("scene.physics.enabled requires scene.backend='videomimic_nksr'")
        if scene.get("source_motion", "smoothed") != "smoothed":
            raise ValueError("scene.physics.enabled requires scene.source_motion='smoothed'")
        clip_root = self._path(self.config["output"]["root"]) / video.stem
        scene_root = clip_root / str(scene["output_subdir"])
        required = {
            "manifest": scene_root / "scene_manifest.json",
            "checksums": scene_root / "checksums.sha256",
            "quality": scene_root / "scene_quality.json",
            "canonical_quality": scene_root / "quality" / "canonicalization_quality.json",
            "collision_mesh": scene_root / "scene" / "background_mesh_mujoco.obj",
            "primitives": scene_root / "scene" / "primitives_mujoco.json",
            "scene_xml": scene_root / "scene" / "scene_mujoco.xml",
            "motion": scene_root / "human" / "human_motion_static_hard.npz",
            "camera": scene_root / "human" / "gvhmr_camera_static_hard.npz",
        }
        missing = [path for path in required.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("scene-aware PHC/GMR requires a complete package:\n  " + "\n  ".join(map(str, missing)))
        manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "pass"
            or manifest.get("backend") != "videomimic_nksr"
            or manifest.get("camera_mode") != "static_hard"
        ):
            raise RuntimeError("scene package is not a passing VideoMimic static-camera package")
        quality = json.loads(required["quality"].read_text(encoding="utf-8"))
        if quality.get("verdict") != "pass":
            raise RuntimeError(
                "scene package is diagnostic-only: scene_quality.verdict must be "
                "'pass' before it can affect PHC or GMR collision"
            )
        provenance = manifest.get("input_provenance")
        if not isinstance(provenance, dict) or provenance.get("schema_version") != 1:
            raise RuntimeError(
                "scene.physics requires an input-provenance package; rebuild the VideoMimic package instead of reusing a legacy sidecar"
            )
        if provenance.get("clip_id") != video.stem:
            raise RuntimeError("scene package clip identity does not match the requested video")
        checksums = self._read_scene_checksums(scene_root)
        for label, path in required.items():
            if label == "checksums":
                continue
            relative = path.relative_to(scene_root).as_posix()
            expected_digest = checksums.get(relative)
            if expected_digest is None or self._sha256_file(path) != expected_digest:
                raise RuntimeError(f"scene package checksum mismatch: {relative}")

        source_motion = provenance.get("source_motion")
        source_camera = provenance.get("source_camera")
        canonical_motion = provenance.get("canonical_motion")
        canonical_camera = provenance.get("canonical_camera")
        if not all(isinstance(record, dict) for record in (source_motion, source_camera, canonical_motion, canonical_camera)):
            raise RuntimeError("scene package lacks motion/camera provenance")
        source_video = self._verify_scene_file_record(
            clip_root,
            provenance.get("source_video"),
            label="scene source video",
        )
        del source_video
        canonical_motion_path = self._verify_scene_npz_record(
            required["motion"],
            canonical_motion,
            label="canonical scene motion",
            root=scene_root,
            expected_relative="human/human_motion_static_hard.npz",
        )
        canonical_camera_path = self._verify_scene_npz_record(
            required["camera"],
            canonical_camera,
            label="canonical scene camera",
            root=scene_root,
            expected_relative="human/gvhmr_camera_static_hard.npz",
        )
        if canonical_motion.get("frame_count") != canonical_camera.get("frame_count"):
            raise RuntimeError("canonical scene motion/camera frame counts differ")
        canonicalization = provenance.get("canonicalization")
        if not isinstance(canonicalization, dict) or canonicalization.get("mode") not in {
            "static_exact_reprojection",
            "static_optimized",
        }:
            raise RuntimeError("scene package canonicalization mode is unsupported")
        quality_path = self._safe_scene_package_path(
            scene_root,
            canonicalization.get("quality_path"),
            label="canonicalization quality path",
        )
        if quality_path != required["canonical_quality"] or self._sha256_file(quality_path) != canonicalization.get("quality_sha256"):
            raise RuntimeError("canonicalization quality report does not match its provenance")
        canonical_quality = json.loads(quality_path.read_text(encoding="utf-8"))
        if canonical_quality.get("status") != "accepted":
            raise RuntimeError("scene package canonicalization quality was not accepted")
        assets = provenance.get("collision_assets")
        if not isinstance(assets, dict):
            raise RuntimeError("scene package lacks collision-asset provenance")
        self._verify_scene_file_record(
            scene_root,
            assets.get("phc_mesh"),
            label="PHC collision mesh",
            expected_relative="scene/background_mesh_mujoco.obj",
            explicit_path=required["collision_mesh"],
        )
        self._verify_scene_file_record(
            scene_root,
            assets.get("phc_primitives"),
            label="PHC primitives",
            expected_relative="scene/primitives_mujoco.json",
            explicit_path=required["primitives"],
        )
        self._verify_scene_file_record(
            scene_root,
            assets.get("mujoco_xml"),
            label="MuJoCo scene XML",
            expected_relative="scene/scene_mujoco.xml",
            explicit_path=required["scene_xml"],
        )

        if source_motion.get("relative_path") != "001_smoothed.npz" or source_camera.get("relative_path") != "gvhmr_camera.npz":
            raise RuntimeError("scene.physics only accepts packages built from the active smoothed motion and GVHMR camera")
        active_motion = clip_root / "001_smoothed.npz"
        active_camera = clip_root / "gvhmr_camera.npz"
        native_motion = clip_root / "001_smoothed_native_before_static_hard.npz"
        native_camera = clip_root / "gvhmr_camera_native_before_static_hard.npz"
        if not active_motion.is_file() or not active_camera.is_file():
            raise FileNotFoundError("active smoothed motion/camera required before scene candidate staging")
        active_state = []
        for active, raw_record, canonical_record, backup, label in (
            (active_motion, source_motion, canonical_motion, native_motion, "motion"),
            (active_camera, source_camera, canonical_camera, native_camera, "camera"),
        ):
            if self._sha256_file(active) == raw_record.get("sha256"):
                self._verify_scene_npz_record(active, raw_record, label=f"active raw {label}", root=clip_root)
                active_state.append("raw")
            elif self._sha256_file(active) == canonical_record.get("sha256"):
                self._verify_scene_npz_record(backup, raw_record, label=f"native backup {label}", root=clip_root)
                active_state.append("canonical")
            else:
                raise RuntimeError(
                    f"active {label} matches neither the package source nor canonical artifact; refusing to overwrite it"
                )
        if len(set(active_state)) != 1:
            raise RuntimeError("active motion and camera have inconsistent static-camera state")
        if active_state[0] == "canonical":
            self._atomic_copy(native_motion, active_motion)
            self._atomic_copy(native_camera, active_camera)
            self._verify_scene_npz_record(active_motion, source_motion, label="restored raw motion", root=clip_root)
            self._verify_scene_npz_record(active_camera, source_camera, label="restored raw camera", root=clip_root)

        hand_sidecar = clip_root / "001_smplx_hands.npz"
        if not hand_sidecar.is_file() or hand_sidecar.stat().st_size == 0:
            raise FileNotFoundError(f"scene post requires the preserved hand sidecar: {hand_sidecar}")
        run_seed = json.dumps(
            {
                "manifest_sha256": self._sha256_file(required["manifest"]),
                "motion_sha256": canonical_motion["sha256"],
                "camera_sha256": canonical_camera["sha256"],
                "scene_xml_sha256": assets["mujoco_xml"]["sha256"],
                "nonce": os.urandom(12).hex(),
            },
            sort_keys=True,
        ).encode("utf-8")
        candidate_id = hashlib.sha256(run_seed).hexdigest()[:20]
        candidate_root = clip_root / "scene_physics_candidates" / candidate_id
        candidate_clip = candidate_root / video.stem
        if candidate_root.exists():
            raise FileExistsError(f"refusing to reuse scene-physics candidate directory: {candidate_root}")
        candidate_clip.mkdir(parents=True)
        self._atomic_copy(canonical_motion_path, candidate_clip / "001_smoothed.npz")
        self._atomic_copy(canonical_camera_path, candidate_clip / "gvhmr_camera.npz")
        self._atomic_copy(hand_sidecar, candidate_clip / "001_smplx_hands.npz")
        contract = {
            "schema_version": 2,
            "kind": "videomimic_scene_physics_candidate_input",
            "candidate_id": candidate_id,
            "promotion_status": "candidate_only_not_a_physics_acceptance",
            "source_active_inputs_remain_unmodified": True,
            "source_clip": str(clip_root),
            "scene_package": str(scene_root),
            "scene_manifest_sha256": self._sha256_file(required["manifest"]),
            "input_provenance": provenance,
            "candidate_inputs": {
                "motion_sha256": self._sha256_file(candidate_clip / "001_smoothed.npz"),
                "camera_sha256": self._sha256_file(candidate_clip / "gvhmr_camera.npz"),
                "hand_sidecar_sha256": self._sha256_file(candidate_clip / "001_smplx_hands.npz"),
            },
            "scene_assets": {
                "collision_mesh": str(required["collision_mesh"]),
                "primitives": str(required["primitives"]),
                "scene_mujoco_xml": str(required["scene_xml"]),
            },
        }
        contract_path = candidate_clip / "scene_post_input_contract.json"
        contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {
            "collision_mesh": required["collision_mesh"],
            "primitives": required["primitives"],
            "scene_xml": required["scene_xml"],
            "candidate_root": candidate_root,
            "candidate_clip": candidate_clip,
            "contract": contract_path,
        }

    def _write_scene_candidate_result(
        self,
        *,
        video: Path,
        candidate_clip: Path,
        contract: Path,
        gmr_enabled: bool,
    ) -> None:
        selection_path = candidate_clip / "001_final_selection.json"
        if not selection_path.is_file():
            raise RuntimeError(f"scene candidate lacks a final-motion selection: {selection_path}")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selection_reason = str(selection.get("selection_reason", ""))
        phc_accepted = selection_reason == "phc_repaired"
        robot_motion = candidate_clip / "robot_motion.pkl"
        if gmr_enabled and (not robot_motion.is_file() or robot_motion.stat().st_size == 0):
            raise RuntimeError(f"scene candidate did not produce a fresh robot motion: {robot_motion}")
        result = {
            "schema_version": 1,
            "kind": "scene_physics_candidate_result",
            "clip": video.stem,
            "status": "candidate_gmr_ready" if gmr_enabled else "phc_only_candidate",
            "promotion_status": "candidate_only_not_a_physics_acceptance",
            "input_contract": str(contract),
            "input_contract_sha256": self._sha256_file(contract),
            "candidate_clip": str(candidate_clip),
            "phc_final_selection": selection,
            "phc_output_accepted_by_tracking_gate": phc_accepted,
            "robot_motion": str(robot_motion) if gmr_enabled else None,
            "robot_motion_sha256": self._sha256_file(robot_motion) if gmr_enabled else None,
        }
        candidate_result = candidate_clip / "scene_post_result.json"
        candidate_result.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        clip_root = self._path(self.config["output"]["root"]) / video.stem
        latest = clip_root / "scene_post_candidate_latest.json"
        temporary = latest.with_suffix(latest.suffix + ".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(latest)

    def _run_general_scene_physics_post(self, videos: list[Path], wrapper: Path) -> None:
        """Run immutable VideoMimic scene candidates without touching baseline outputs."""
        scene = self._scene_config()
        if self.dry_run:
            print(
                "[DRY-RUN][SCENE_PHYSICS] would validate/package isolated PHC/GMR candidates; "
                "no existing scene assets are inspected.",
                flush=True,
            )
            return
        gmr_enabled = bool(self.config.get("gmr", {}).get("enabled", True))
        candidate_clips: dict[str, Path] = {}
        for video in videos:
            assets = self._prepare_scene_aware_post_inputs(video)
            assert assets is not None
            environment = self.environment.copy()
            environment["CLIP_FILTER"] = video.stem
            if "runtime" in scene and "gpu" in scene["runtime"]:
                environment["CUDA_VISIBLE_DEVICES"] = self._resolve_scene_gpu(scene)
            environment["OUTPUT_BASE"] = str(assets["candidate_root"])
            environment["GMR_OUT_ROOT"] = str(assets["candidate_root"])
            environment["PIPELINE_POST_ONLY"] = "1"
            environment["SKIP_EXISTING"] = "0"
            environment["FORCE_PHC"] = "1"
            environment["GMR_OVERRIDE"] = "1"
            environment["RUN_GMR"] = "1" if gmr_enabled else "0"
            environment["PHC_EXTRA_OVERRIDES"] = " ".join((
                f"+env.static_scene_mesh={assets['collision_mesh']}",
                f"+env.static_scene_primitives={assets['primitives']}",
                "env.num_envs=1",
            ))
            environment["PHC_GROUND_FIX"] = "0"
            print(
                f"[SCENE_PHYSICS] {video.stem}: isolated candidate uses VideoMimic/NKSR collision evidence; baseline output is read-only.",
                flush=True,
            )
            self._run(["bash", str(wrapper)], env=environment)
            candidate_clip = assets["candidate_clip"]
            self._write_scene_candidate_result(
                video=video,
                candidate_clip=candidate_clip,
                contract=assets["contract"],
                gmr_enabled=gmr_enabled,
            )
            candidate_clips[video.stem] = candidate_clip
        if not gmr_enabled:
            print(
                "[SCENE_PHYSICS] PHC-only candidates are recorded as non-promotable; GMR/mj_step is deferred.",
                flush=True,
            )
            return
        self._run_gmr_scene_mjstep_audit(videos, candidate_clips=candidate_clips)

    def _verify_semantic_phc_preflight(
        self,
        *,
        scene: dict[str, Any],
        collision_mode: str,
        primitives: Path,
        mujoco_scene_xml: Path | None,
    ) -> None:
        """Allow only a hash-bound semantic chair candidate into isolated PHC.

        A VideoMimic ``warn`` package is never a general collision admission.
        An explicit PHC certificate from stable SMPL seated-contact evidence is
        required.  The default scope is a seat-only proxy; the sole permitted
        full-chair scope additionally requires an independent world-space
        equivalence report binding the exact same named boxes in Isaac and
        MuJoCo.  Raw NKSR geometry is never admitted.
        """
        supported_modes = {
            "semantic_seat_support_only",
            "semantic_primitives_only",
        }
        if collision_mode not in supported_modes:
            raise RuntimeError(
                "a diagnostic scene package may enter PHC only through "
                "semantic_seat_support_only or an attested "
                "semantic_primitives_only full chair"
            )
        isaac = scene.get("isaac", {})
        report_value = str(isaac.get("semantic_preflight_report", "")).strip()
        if not report_value:
            raise RuntimeError(
                "scene_quality=warn requires scene.isaac.semantic_preflight_report "
                "before PHC can consume a semantic seat proxy"
            )
        report_path = self._path(report_value)
        if not report_path.is_file():
            raise FileNotFoundError(
                "semantic PHC preflight report is missing: " + str(report_path)
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("schema_version") != 1
            or report.get("status") != "accepted_for_isolated_phc_candidate"
            or report.get("collision_scope") != collision_mode
            or report.get("raw_nksr_mesh_permitted") is not False
        ):
            raise RuntimeError(
                "semantic PHC preflight report does not grant the required "
                "isolated-candidate collision contract"
            )
        inputs = report.get("inputs")
        if not isinstance(inputs, dict) or not isinstance(inputs.get("phc_primitives"), dict):
            raise RuntimeError("semantic PHC preflight report lacks primitive provenance")
        expected_hash = inputs["phc_primitives"].get("sha256")
        actual_hash = self._sha256_file(primitives)
        if expected_hash != actual_hash:
            raise RuntimeError(
                "semantic PHC preflight primitive hash does not match the asset "
                f"selected for Isaac: {primitives}"
            )
        stable = report.get("stable_smpl_seat_evidence")
        if not isinstance(stable, dict) or int(stable.get("frame_count", 0)) < 12:
            raise RuntimeError("semantic PHC preflight lacks stable SMPL seat evidence")
        if float(stable.get("contact_band_coverage", 0.0)) < 1.0:
            raise RuntimeError("semantic PHC preflight has incomplete stable-seat coverage")
        if collision_mode == "semantic_primitives_only":
            asset_contract_value = str(isaac.get("scene_asset_contract_report", "")).strip()
            if not asset_contract_value:
                raise RuntimeError(
                    "full semantic PHC chair requires "
                    "scene.isaac.scene_asset_contract_report"
                )
            asset_contract_path = self._path(asset_contract_value)
            if not asset_contract_path.is_file():
                raise FileNotFoundError(
                    "semantic scene asset contract is missing: " + str(asset_contract_path)
                )
            asset_contract = json.loads(asset_contract_path.read_text(encoding="utf-8"))
            if (
                asset_contract.get("schema_version") != 1
                or asset_contract.get("status") != "accepted_world_space_equivalent"
                or asset_contract.get("collision_representation") != "identical_named_semantic_boxes"
                or asset_contract.get("requires_identical_names") is not True
                or asset_contract.get("mujoco_primitives_sha256") != actual_hash
                or asset_contract.get("isaac_primitives_sha256") != actual_hash
                or float(asset_contract.get("max_world_vertex_error_m", float("inf"))) > 1e-6
            ):
                raise RuntimeError(
                    "semantic scene asset contract does not prove that Isaac and "
                    "MuJoCo consume the exact same full chair boxes"
                )
            if mujoco_scene_xml is not None:
                if not mujoco_scene_xml.is_file():
                    raise FileNotFoundError(
                        "configured GMR scene XML is missing: " + str(mujoco_scene_xml)
                    )
                if asset_contract.get("mujoco_xml_sha256") != self._sha256_file(mujoco_scene_xml):
                    raise RuntimeError(
                        "full-chair asset contract XML hash does not match the "
                        "MuJoCo XML selected for GMR"
                    )
            certificate_contract = inputs.get("scene_asset_contract_report")
            if (
                not isinstance(certificate_contract, dict)
                or certificate_contract.get("sha256") != self._sha256_file(asset_contract_path)
            ):
                raise RuntimeError(
                    "full-chair PHC certificate is not bound to the selected asset contract"
                )
        print(
            "[SCENE_BRIDGE] admitted hash-bound semantic PHC candidate "
            f"({collision_mode}): "
            + str(report_path),
            flush=True,
        )

    def run_human_post(self) -> None:
        """Resume PHC/GMR, optionally from a validated static scene coordinate frame."""
        videos = self._videos()
        if not videos:
            print("[HUMAN_POST] no admitted clips; skipping PHC/GMR.", flush=True)
            return
        self._prepare_videomimic_robot_bridge(videos)
        self._ensure_run_manifests(
            videos,
            allow_existing_clip=bool(
                self._scene_config().get("robot_bridge", {}).get("enabled", False)
            ),
        )
        self._apply_smpl_2d_contact_refinement(videos)
        self._apply_phc_reference_calibration(videos)
        backend = self.config.get("human", {}).get("backend", "hand4wholepp")
        wrapper = (
            self.root
            / "GVHMR-hand"
            / "GVHMR-main"
            / "tools"
            / "pipeline"
            / BACKEND_WRAPPERS[backend]
        )
        scene = self._scene_config()
        physics = scene.get("physics", {})
        if isinstance(physics, dict) and bool(physics.get("enabled", False)):
            if scene.get("robot_bridge", {}).get("enabled", False):
                raise ValueError("scene.physics.enabled and scene.robot_bridge.enabled cannot be combined")
            self._run_general_scene_physics_post(videos, wrapper)
            return
        environment = self.environment.copy()
        environment["CLIP_FILTER"] = ",".join(video.stem for video in videos)
        scene = self._scene_config()
        if "runtime" in scene and "gpu" in scene["runtime"]:
            environment["CUDA_VISIBLE_DEVICES"] = self._resolve_scene_gpu(scene)
        if scene.get("robot_bridge", {}).get("enabled", False):
            environment["PIPELINE_POST_ONLY"] = "1"
        if scene.get("isaac", {}).get("enabled", False):
            if len(videos) != 1:
                raise RuntimeError(
                    "scene.isaac.enabled currently supports exactly one clip per "
                    "run because PHC's static-mesh override is process-global"
                )
            clip_root = self._path(self.config["output"]["root"]) / videos[0].stem
            scene_root = clip_root / str(scene["output_subdir"])
            scene_dir = scene_root / "scene"
            scene_quality_path = scene_root / "scene_quality.json"
            if not scene_quality_path.is_file():
                raise FileNotFoundError(
                    "Isaac scene bridge requires scene_quality.json before PHC "
                    f"can consume collision assets: {scene_quality_path}"
                )
            scene_quality = json.loads(scene_quality_path.read_text(encoding="utf-8"))
            collision_mode = scene["isaac"].get(
                "collision_mode", "raw_mesh_and_support"
            )
            assets_by_mode: dict[str, tuple[str | None, str]] = {
                "raw_mesh_and_support": (
                    "background_mesh_mujoco.obj",
                    "primitives_mujoco.json",
                ),
                # This is the controlled diagnostic for a reconstructed support:
                # it excludes all raw mesh/background collision triangles.
                "seat_support_only": (None, "primitives_mujoco.json"),
                # The semantic chair builder emits named primitive-only assets.
                "semantic_primitives_only": (
                    None,
                    "semantic_chair_primitives_mujoco.json",
                ),
                "semantic_seat_support_only": (
                    None,
                    "semantic_chair_primitives_phc.json",
                ),
                "semantic_seat_backrest_support": (
                    None,
                    "semantic_chair_primitives_phc.json",
                ),
                "semantic_mesh_and_support": (
                    "background_mesh_semantic_chair.obj",
                    "semantic_chair_primitives_mujoco.json",
                ),
            }
            mesh_name, primitives_name = assets_by_mode[collision_mode]
            mesh = scene_dir / mesh_name if mesh_name is not None else None
            primitives = scene_dir / primitives_name
            required_assets = [primitives]
            if mesh is not None:
                required_assets.append(mesh)
            missing = [path for path in required_assets if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Isaac scene bridge requires packaged MuJoCo-frame assets:\n  "
                    + "\n  ".join(str(path) for path in missing)
                )
            if scene_quality.get("verdict") != "pass":
                gmr_cfg = self.config.get("gmr", {})
                mujoco_scene_xml: Path | None = None
                if bool(gmr_cfg.get("enabled", True)) and bool(gmr_cfg.get("render", True)):
                    scene_xml_name = str(gmr_cfg.get("scene_mujoco_name", "")).strip()
                    if not scene_xml_name:
                        raise RuntimeError(
                            "full/shared PHC chair requires an explicit "
                            "gmr.scene_mujoco_name when GMR rendering is enabled"
                        )
                    mujoco_scene_xml = clip_root / scene_xml_name
                self._verify_semantic_phc_preflight(
                    scene=scene,
                    collision_mode=collision_mode,
                    primitives=primitives,
                    mujoco_scene_xml=mujoco_scene_xml,
                )
            # Paths are validated under the project root and contain no shell
            # whitespace; the pipeline shell forwards these as Hydra overrides.
            overrides = [
                f"+env.static_scene_primitives={primitives}",
                # Packaged scene coordinates are global; PHC rejects static
                # geometry in a replicated multi-environment simulation.
                "env.num_envs=1",
            ]
            if mesh is not None:
                overrides.insert(0, f"+env.static_scene_mesh={mesh}")
            environment["PHC_EXTRA_OVERRIDES"] = " ".join(overrides)
            # A global "lift the whole SMPL mesh above z=0" post-process is
            # valid for an unstructured floor-only motion, but it invalidates
            # the measured relation to a raised support such as a chair.  A
            # scene-aware contact solver may opt in explicitly; otherwise the
            # raw PHC state remains the sole physical result to audit.
            if not scene["isaac"].get("allow_global_floor_fix", False):
                environment["PHC_GROUND_FIX"] = "0"
            print(
                "[SCENE_BRIDGE] PHC/Isaac static collision enabled "
                f"({collision_mode}): "
                + (f"{mesh.name} + " if mesh is not None else "")
                + primitives.name,
                flush=True,
            )
            if environment.get("PHC_GROUND_FIX") == "0":
                print(
                    "[SCENE_BRIDGE] disabled global floor lift for scene-aware "
                    "PHC replay; use the raw state alignment audit instead.",
                    flush=True,
                )
        # SKIP_EXISTING is managed by YAML and defaults to true.  The scene
        # bridge additionally sets PIPELINE_POST_ONLY, which makes the wrapper
        # consume the canonical motion directly rather than merely hoping that
        # its GVHMR configuration fingerprint still matches the source run.
        self._run(["bash", str(wrapper)], env=environment)
        # A physics-only diagnostic intentionally omits GMR rendering and
        # therefore has no robot_motion.pkl.  Preserve the strict artifact
        # check for all normal PHC+GMR production runs.
        if self.config.get("gmr", {}).get("enabled", True):
            output_root = self._path(self.config["output"]["root"])
            missing = [
                output_root / video.stem / "robot_motion.pkl"
                for video in videos
                if not (output_root / video.stem / "robot_motion.pkl").is_file()
            ]
            if missing:
                raise RuntimeError(
                    "human_post did not produce robot_motion.pkl:\n  "
                    + "\n  ".join(str(path) for path in missing)
                )
            aligned_motion_name = self._apply_gmr_scene_contact_alignment(videos)
            if aligned_motion_name is not None:
                self._rerender_gmr(
                    [video.stem for video in videos],
                    motion_name=aligned_motion_name,
                    reuse_existing_motion=True,
                )
            self._run_gmr_scene_physics_validation(
                videos,
                aligned_motion_name or "robot_motion.pkl",
            )

    def _run_gmr_scene_mjstep_audit(
        self,
        videos: list[Path],
        *,
        candidate_clips: dict[str, Path] | None = None,
    ) -> None:
        """Run the continuous post-GMR reference controller and enforce its gate.

        The renderer initializes once, then uses bounded joint targets and
        continuous ``mj_step``. A physical fall or chair collision is rejected;
        the input GMR root is never restored to hide the response.
        """
        scene = self._scene_config()
        physics = scene.get("physics", {})
        if not isinstance(physics, dict) or not bool(physics.get("enabled", False)):
            return
        if self.dry_run:
            print("[DRY-RUN][SCENE_PHYSICS] would run the MuJoCo mj_step audit; existing artifacts are not inspected.", flush=True)
            return
        gmr = self.config.get("gmr", {})
        robot = str(gmr.get("robot", self.environment.get("GMR_ROBOT", "unitree_g1")))
        robot_xmls = {
            "unitree_g1": self.root / "GMR-master" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml",
        }
        if robot not in robot_xmls:
            raise ValueError(f"scene mj_step audit has no robot XML contract for {robot!r}")
        if not robot_xmls[robot].is_file():
            raise FileNotFoundError(robot_xmls[robot])
        output_root = self._path(self.config["output"]["root"])
        width = int(physics.get("render_width", 960))
        height = int(physics.get("render_height", 720))
        substeps = int(physics.get("mj_step_substeps", 8))
        threshold = float(physics.get("max_negative_contact_distance_m", 0.012))
        visible_threshold = float(physics.get(
            "max_negative_visible_mesh_distance_m", threshold
        ))
        if width <= 0 or height <= 0 or substeps < 1 or min(threshold, visible_threshold) < 0.0:
            raise ValueError("scene.physics render dimensions, substeps, and penetration thresholds are invalid")
        require_pass = bool(physics.get("require_mjstep_pass", True))
        report_name = str(physics.get("mjstep_report_name", "gmr_scene_mjstep_report.json"))
        video_name = str(physics.get("mjstep_video_name", "gmr_scene_mjstep.mp4"))
        for name in (report_name, video_name):
            if Path(name).name != name:
                raise ValueError("scene.physics mj_step artifact names must be filenames")
        for video in videos:
            clip_root = output_root / video.stem
            candidate_clip = (
                candidate_clips.get(video.stem, clip_root)
                if candidate_clips is not None
                else clip_root
            )
            scene_root = clip_root / str(scene["output_subdir"])
            scene_xml = scene_root / "scene" / "scene_mujoco.xml"
            if not scene_xml.is_file():
                raise FileNotFoundError(scene_xml)
            robot_motion = candidate_clip / "robot_motion.pkl"
            if not robot_motion.is_file() or not robot_motion.stat().st_size:
                raise FileNotFoundError(robot_motion)
            report = candidate_clip / report_name
            video_path = candidate_clip / video_name
            if report.exists() or video_path.exists():
                raise FileExistsError("refusing to overwrite a mj_step audit artifact: " + str(report if report.exists() else video_path))
            environment = self.environment.copy()
            environment["MUJOCO_GL"] = str(gmr.get("mujoco_gl", "egl"))
            environment.setdefault("PYOPENGL_PLATFORM", environment["MUJOCO_GL"])
            command = [
                self.runtime_python,
                str(self.root / "scripts" / "scene" / "render_gmr_scene_mjstep.py"),
                "--robot-motion", str(robot_motion),
                "--robot-xml", str(robot_xmls[robot]),
                "--scene-mujoco-xml", str(scene_xml),
                "--video", str(video_path),
                "--report", str(report),
                "--mujoco-gl", environment["MUJOCO_GL"],
                "--width", str(width), "--height", str(height),
                "--substeps", str(substeps),
                "--max-negative-contact-distance-m", str(threshold),
                "--max-negative-visible-mesh-distance-m", str(visible_threshold),
            ]
            self._run(command, env=environment)
            result = json.loads(report.read_text(encoding="utf-8"))
            if result.get("uses_mujoco_mj_step") is not True:
                raise RuntimeError("mj_step audit did not satisfy its simulator contract")
            if int(result.get("initial_mj_forward_count", 0)) > 1:
                raise RuntimeError("mj_step audit used more than one initialization forward pass")
            if require_pass and result.get("verdict") != "pass":
                raise RuntimeError(f"GMR scene mj_step penetration gate failed for {video.stem}: {result.get('acceptance')}")

    def _reject_unpromoted_scene_candidates(self, consumer: str) -> None:
        """Keep diagnostic scene candidates out of product/object/main-quality paths."""
        scene = self._scene_config()
        physics = scene.get("physics", {})
        if not isinstance(physics, dict) or not bool(physics.get("enabled", False)):
            return
        output_root = self._path(self.config["output"]["root"])
        blocked: list[str] = []
        for video in self._videos():
            marker = output_root / video.stem / "scene_post_candidate_latest.json"
            if not marker.is_file():
                continue
            try:
                candidate = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"invalid scene candidate marker for {video.stem}: {exc}") from exc
            if candidate.get("promotion_status") != "accepted_autonomous_controller":
                blocked.append(str(marker))
        if blocked:
            raise RuntimeError(
                f"{consumer} cannot consume scene PHC/GMR candidates before an autonomous-controller acceptance; "
                "the immutable baseline remains the only promotable output:\n  "
                + "\n  ".join(blocked)
            )

    def run_product(self) -> None:
        """Export safe NPZ digital-asset bundles from trusted work outputs."""
        from export_dataset_product import export_from_config

        if self._eligible_clips == set():
            print("[PRODUCT] no full-body-admitted clips; skipping export.")
            return
        self._reject_unpromoted_scene_candidates("product export")
        clips = self._completed_product_clips()
        if not clips:
            print("[PRODUCT] no completed human/GMR clips; skipping export.")
            return
        export_from_config(
            self.root,
            self.config,
            clips,
            dry_run=self.dry_run,
        )

    def run_quality(self, *, human_only: bool = False) -> None:
        """Evaluate completed clips and write pass/warn/fail reports."""
        from evaluate_clip_quality import evaluate_clips

        if self._eligible_clips == set():
            print("[QUALITY] no full-body-admitted clips; skipping evaluation.")
            return
        if not human_only:
            self._reject_unpromoted_scene_candidates("full quality evaluation")
        quality_config = self.config
        clips = self._completed_output_clips()
        if human_only and self.config.get("object", {}).get("enabled", False):
            quality_config = copy.deepcopy(self.config)
            quality_config.setdefault("object", {})["enabled"] = False
        if not clips:
            print("[QUALITY] no completed human/GMR clips; skipping evaluation.")
            return
        evaluate_clips(
            self.root,
            quality_config,
            clips,
            dry_run=self.dry_run,
        )

    def _clip_object_config(self, clip: str) -> dict:
        obj = self.config.get("object", {})
        mono = obj.get("monocular", {})
        defaults = {
            "object_name": mono.get("default_object_name", "object"),
            "reference_frame": mono.get("default_reference_frame", 0),
            "frame_offset": 0,
            "robot_offset": "0,0,0",
            "dynamic_release_frame": 0,
            "object_prompt": mono.get("segmentation_prompt", ""),
            "detector_labels": None,
            "prompt_mode": mono.get("segmentation_prompt_mode", "click"),
            "allow_no_object": True,
            "expected_size_m": None,
            "metric_max_extent_m": None,
        }
        configured = _deep_merge(
            defaults,
            obj.get("clips", {}).get(clip, {}),
        )
        return _deep_merge(
            configured,
            obj.get("runtime_clip_overrides", {}),
        )

    def _prepare_monocular_video(
        self,
        source: Path,
        clip: str,
        work_root: Path,
    ) -> tuple[Path, Path]:
        raw_dir = work_root / clip
        destination = raw_dir / source.name
        if self.dry_run:
            return raw_dir, destination
        raw_dir.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.stat().st_size != source.stat().st_size:
                raise RuntimeError(
                    f"existing monocular work video differs: {destination}"
                )
            return raw_dir, destination
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        return raw_dir, destination

    def _select_monocular_video(
        self,
        source: Path,
        clip: str,
        clip_dir: Path,
    ) -> Path:
        """Use the exact 1280x960 RGB stream associated with GVHMR intrinsics."""
        work_cfg = self.config.get("input", {}).get("work_video", {})
        candidates = []
        if work_cfg.get("enabled", True):
            work_dataset = self._path(
                work_cfg.get("directory", "dataset_new6_work_1280")
            )
            candidates.append(work_dataset / source.name)
        candidates.append(
            clip_dir / "gvhmr_out" / clip / "valid_video.mp4"
        )
        candidates.append(source)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            "No RGB video found for object tracking; checked "
            + ", ".join(str(path) for path in candidates)
        )

    def _run_object_bridge(
        self,
        prepared_dir: Path,
        clip_dir: Path,
        clip_cfg: dict,
    ) -> None:
        obj = self.config["object"]
        collision = obj.get("collision", {})
        overlay = obj.get("overlay", {})
        command = [
            self.runtime_python,
            str(
                self.root
                / "GVHMR-hand"
                / "GVHMR-main"
                / "tools"
                / "pipeline"
                / "run_object_reconstruction_bridge.py"
            ),
            "--foundation_dir",
            str(prepared_dir),
            "--clip_dir",
            str(clip_dir),
            "--collision_method",
            str(collision.get("method", "auto")),
            "--density",
            str(collision.get("density_kg_m3", 600.0)),
            "--frame_offset",
            str(clip_cfg.get("frame_offset", 0)),
            "--robot_offset",
            str(clip_cfg.get("robot_offset", "0,0,0")),
            "--human_yaw_offset_deg",
            self.environment["GMR_HUMAN_YAW_OFFSET_DEG"],
            "--overlay_alpha",
            str(overlay.get("alpha", 0.82)),
        ]
        if collision.get("mass_kg") is not None:
            command.extend(["--mass", str(collision["mass_kg"])])
        if not overlay.get("enabled", True):
            command.append("--skip_overlay")
        self._run(command)

    def _run_monocular_object(
        self,
        video: Path,
        clip: str,
        clip_dir: Path,
        clip_cfg: dict,
    ) -> bool:
        obj = self.config["object"]
        mono = obj.get("monocular", {})
        runtime = obj.get("runtime", {})
        quality = obj.get("quality", {})
        person_roi = mono.get("person_roi", {})
        auto_detector = mono.get("auto_detector", {})
        allow_no_object = clip_cfg.get("allow_no_object", True)
        work_root = self._path(
            mono.get("work_root", "object_work/do_as_i_do")
        )
        recon_dir = clip_dir / "object_reconstruction"
        if not self.dry_run:
            recon_dir.mkdir(parents=True, exist_ok=True)
        stage_status_path = recon_dir / "object_stage_status.json"

        stages: dict[str, dict] = {}
        degraded = False

        def _save_stages(ok: bool):
            if self.dry_run:
                return
            status = {
                "schema_version": 1,
                "clip": clip,
                "ok": ok,
                "stages": stages,
                "degraded_to_no_object": degraded,
            }
            stage_status_path.write_text(
                json.dumps(status, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        try:
            aligned_video = self._select_monocular_video(
                video, clip, clip_dir
            )
            raw_dir, work_video = self._prepare_monocular_video(
                aligned_video,
                clip,
                work_root,
            )
            stages["prepare_video"] = {
                "ok": True,
                "path": str(work_video),
                "source": str(aligned_video),
            }
            _save_stages(True)
        except Exception as exc:
            stages["prepare_video"] = {"ok": False, "reason": str(exc)}
            _save_stages(False)
            self._handle_object_failure(clip, clip_dir, "prepare_video", exc, allow_no_object)
            return False

        quality_only = runtime.get("quality_only", False)
        adapter_dir = recon_dir / "monocular_adapter"

        if mono.get("run_reconstruction", True) and not quality_only:
            try:
                reconstruction_script = self._path(
                    mono.get(
                        "reconstruction_script",
                        "do-as-i-do-main/reconstruction/run_pipeline.sh",
                    )
                )
                self._run(
                    [
                        "bash",
                        str(reconstruction_script),
                        "--video",
                        str(work_video),
                        "--camera",
                        str(clip_dir / "gvhmr_camera.npz"),
                        "--work-dir",
                        str(raw_dir),
                        "--adapter-dir",
                        str(adapter_dir),
                        "--reference-frame",
                        str(clip_cfg["reference_frame"]),
                        "--object-name",
                        str(clip_cfg["object_name"]),
                        "--prompt-mode",
                        str(clip_cfg.get("prompt_mode", "click")),
                    ]
                    + (
                        [
                            "--object-prompt",
                            str(clip_cfg["object_prompt"]),
                        ]
                        if clip_cfg.get("object_prompt")
                        else []
                    )
                    + (
                        [
                            "--segmentation-checkpoint",
                            str(self._path(mono["samhq_checkpoint"])),
                        ]
                        if mono.get("samhq_checkpoint")
                        else []
                    )
                    + (
                        [
                            "--cutie-checkpoint",
                            str(self._path(mono["cutie_checkpoint"])),
                        ]
                        if mono.get("cutie_checkpoint")
                        else []
                    )
                    + (
                        [
                            "--person-bbox",
                            str(
                                clip_dir
                                / "gvhmr_out"
                                / clip
                                / "preprocess"
                                / "bbx.pt"
                            ),
                            "--person-roi-scale",
                            str(person_roi.get("bbox_scale", 2.0)),
                            "--person-roi-min-mask-ratio",
                            str(
                                person_roi.get(
                                    "min_mask_inside_ratio",
                                    0.60,
                                )
                            ),
                        ]
                        if person_roi.get("enabled", True)
                        else []
                    )
                    + (
                        [
                            "--auto-detector-model",
                            str(
                                self._path(
                                    auto_detector.get(
                                        "model",
                                        "models/yolo11n.pt",
                                    )
                                )
                            ),
                            "--auto-detector-confidence",
                            str(auto_detector.get("confidence", 0.05)),
                            "--auto-detector-min-roi",
                            str(
                                auto_detector.get(
                                    "min_box_inside_person_roi",
                                    0.50,
                                )
                            ),
                            "--auto-detector-padding",
                            str(
                                auto_detector.get(
                                    "box_padding_ratio",
                                    0.08,
                                )
                            ),
                        ]
                        + (
                            [
                                "--auto-detector-labels",
                                ",".join(
                                    str(value)
                                    for value in clip_cfg["detector_labels"]
                                ),
                            ]
                            if isinstance(
                                clip_cfg.get("detector_labels"),
                                list,
                            )
                            else []
                        )
                        if clip_cfg.get("prompt_mode") == "auto"
                        else []
                    )
                    + (
                        [
                            "--target-max-extent-m",
                            str(clip_cfg["metric_max_extent_m"]),
                        ]
                        if clip_cfg.get("metric_max_extent_m") is not None
                        else []
                    )
                    + (
                        [
                            "--expected-min-extent-m",
                            str(clip_cfg["expected_size_m"][0]),
                            "--expected-max-extent-m",
                            str(clip_cfg["expected_size_m"][1]),
                        ]
                        if isinstance(clip_cfg.get("expected_size_m"), list)
                        and len(clip_cfg["expected_size_m"]) == 2
                        else []
                    )
                )
                stages["rgb_reconstruction"] = {
                    "ok": True,
                    "raw_dir": str(raw_dir),
                    "adapter": str(adapter_dir),
                }
            except Exception as exc:
                stages["rgb_reconstruction"] = {
                    "ok": False,
                    "reason": str(exc),
                }
                _save_stages(False)
                self._handle_object_failure(
                    clip,
                    clip_dir,
                    "rgb_reconstruction",
                    exc,
                    allow_no_object,
                )
                return False
        else:
            stages["rgb_reconstruction"] = {
                "ok": True,
                "note": "skipped (reconstruction disabled or quality-only)",
            }

        if not runtime.get("skip_adapter", False):
            # MegaPose writes the adapter contract directly; validate it before
            # any coordinate conversion or rendering.
            validation_path = recon_dir / "object_adapter_validation.json"
            try:
                validator_cmd = [
                    self.runtime_python,
                    str(
                        self.root
                        / "GMR-master"
                        / "scripts"
                        / "validate_object_adapter.py"
                    ),
                    "--adapter_dir",
                    str(adapter_dir),
                    "--clip_dir",
                    str(clip_dir),
                    "--output",
                    str(validation_path),
                    "--min_valid_ratio",
                    str(
                        quality.get("min_valid_ratio", 0.35)
                    ),
                    "--min_observed_ratio",
                    str(
                        quality.get("min_observed_ratio", 0.10)
                    ),
                ]
                if work_video.exists():
                    validator_cmd.extend(["--video", str(work_video)])
                expected = clip_cfg.get("expected_size_m")
                if isinstance(expected, list) and len(expected) == 2:
                    validator_cmd.extend(
                        [
                            "--min_extent_m",
                            str(expected[0]),
                            "--max_extent_m",
                            str(expected[1]),
                        ]
                    )
                self._run(validator_cmd)
                stages["adapter_validation"] = {"ok": True, "path": str(validation_path)}
            except Exception as exc:
                stages["adapter_validation"] = {"ok": False, "reason": str(exc)}
                _save_stages(False)
                self._handle_object_failure(
                    clip,
                    clip_dir,
                    "adapter_validation",
                    exc,
                    allow_no_object,
                )
                return False
        else:
            stages["adapter_validation"] = {
                "ok": True,
                "note": "skipped",
            }

        if not runtime.get("skip_bridge", False) and not quality_only:
            try:
                self._run_object_bridge(adapter_dir, clip_dir, clip_cfg)
                stages["bridge"] = {"ok": True, "path": str(recon_dir)}
            except Exception as exc:
                stages["bridge"] = {"ok": False, "reason": str(exc)}
                _save_stages(False)
                self._handle_object_failure(clip, clip_dir, "bridge", exc, allow_no_object)
                return False
        else:
            stages["bridge"] = {"ok": True, "note": "skipped"}

        stages["rerender"] = {"ok": True, "note": "deferred to batch rerender"}
        dynamic = obj.get("dynamic_validation", {})
        stages["dynamic_validation"] = {
            "ok": dynamic.get("enabled", False),
            "reason": "enabled" if dynamic.get("enabled", False) else "disabled",
        }
        if not self.dry_run:
            (recon_dir / "object_unavailable.json").unlink(missing_ok=True)
        _save_stages(True)
        return True

    def _handle_object_failure(
        self,
        clip: str,
        clip_dir: Path,
        stage: str,
        exc: Exception,
        allow_no_object: bool,
    ) -> None:
        """Write object_unavailable.json and either raise or continue."""
        recon_dir = clip_dir / "object_reconstruction"
        recon_dir.mkdir(parents=True, exist_ok=True)
        unavailable = {
            "schema_version": 1,
            "status": "failed",
            "stage": stage,
            "reason": str(exc),
            "degraded_product_grade": "human_only",
        }
        unavailable_path = recon_dir / "object_unavailable.json"
        unavailable_path.write_text(
            json.dumps(unavailable, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"[FAILED] object stage={stage} for {clip}: {exc}",
            flush=True,
        )
        if not allow_no_object:
            raise RuntimeError(
                f"Object reconstruction failed at stage={stage} "
                f"and allow_no_object=false for {clip}: {exc}"
            ) from exc
        print(
            f"[INFO] Degrading {clip} to human_only product grade "
            f"(allow_no_object=true)",
            flush=True,
        )

    def _run_foundationpose_object(
        self,
        clip: str,
        clip_dir: Path,
        clip_cfg: dict,
    ) -> None:
        template = self.config["object"].get("foundationpose", {}).get(
            "result_dir_template",
            "foundationpose-plus-plus-main/test_data/{clip}",
        )
        prepared_dir = self._path(str(template).format(clip=clip))
        self._run_object_bridge(prepared_dir, clip_dir, clip_cfg)

    def _rerender_gmr(
        self,
        clips: list[str],
        *,
        motion_name: str | None = None,
        reuse_existing_motion: bool = False,
    ) -> None:
        env = self.environment.copy()
        output_root = self._path(self.config["output"]["root"])
        env.update(
            {
                "SHOW_ROOT": str(output_root),
                "OUT_ROOT": str(output_root),
                "GMR_INCLUDE_CLIPS": ",".join(clips),
                "GMR_OVERRIDE": "1",
                "GMR_OBJECT_MOTION_NAME": (
                    "object_reconstruction/object_motion_gmr.npz"
                ),
            }
        )
        if motion_name is not None:
            env["GMR_OUTPUT_NAME"] = motion_name
        if reuse_existing_motion:
            env.update(
                {
                    "GMR_REUSE_EXISTING_MOTION": "1",
                    # The hand sidecar is already valid for the same frames and
                    # robot morphology.  Rendering should not overwrite it.
                    "GMR_SHARPA_AUTO_RETARGET": "0",
                    "GMR_BRAINCO_AUTO_RETARGET": "0",
                }
            )
        self._run(
            ["bash", str(self.root / "GMR-master" / "run_show_gmr_batch.sh")],
            env=env,
        )

    def _dynamic_validation(self, clips: list[str]) -> None:
        dynamic = self.config["object"].get("dynamic_validation", {})
        if not dynamic.get("enabled", False):
            return
        output_root = self._path(self.config["output"]["root"])
        hand_model = self.environment["GMR_HAND_MODEL"]
        for clip in clips:
            clip_cfg = self._clip_object_config(clip)
            clip_dir = output_root / clip
            command = [
                self.runtime_python,
                str(
                    self.root
                    / "GMR-master"
                    / "scripts"
                    / "simulate_robot_object_contacts.py"
                ),
                "--robot_motion_path",
                str(clip_dir / "robot_motion.pkl"),
                "--object_motion_path",
                str(
                    clip_dir
                    / "object_reconstruction"
                    / "object_motion_gmr.npz"
                ),
                "--output",
                str(
                    clip_dir
                    / "object_reconstruction"
                    / "object_dynamic_sim.npz"
                ),
                "--robot",
                self.environment["GMR_ROBOT"],
                "--release_frame",
                str(clip_cfg.get("dynamic_release_frame", 0)),
                "--substeps",
                str(dynamic.get("substeps", 16)),
                "--lift_threshold",
                str(dynamic.get("lift_threshold_m", 0.05)),
                "--mujoco_gl",
                self.environment.get("GMR_MUJOCO_GL", "osmesa"),
            ]
            if hand_model in {"sharpa", "sharpa_g1", "sharpa_h1"}:
                command.extend(
                    [
                        "--sharpa_hand_npz",
                        str(clip_dir / "001_sharpa_chain_hands.npz"),
                    ]
                )
            if dynamic.get("object_floor_collision", True):
                command.append("--object_floor_collision")
            if dynamic.get("render_video", True):
                command.extend(
                    [
                        "--video_path",
                        str(
                            clip_dir
                            / "object_reconstruction"
                            / "object_dynamic_sim.mp4"
                        ),
                    ]
                )
            self._run(command)

    def run_objects(self) -> None:
        if self._eligible_clips == set():
            print("[OBJECT] no full-body-admitted clips; skipping object stage.")
            return
        self._reject_unpromoted_scene_candidates("object stage")
        obj = self.config["object"]
        output_root = self._path(self.config["output"]["root"])
        configured = set(obj.get("clips", {}))
        only_configured = obj.get("only_configured_clips", True)
        processed = []
        failed = []
        for video in self._videos():
            clip = video.stem
            if only_configured and clip not in configured:
                continue
            clip_dir = output_root / clip
            if not self.dry_run and not (clip_dir / "robot_motion.pkl").exists():
                raise FileNotFoundError(
                    f"{clip_dir / 'robot_motion.pkl'} is missing; run the human/GMR stage first"
                )
            clip_cfg = self._clip_object_config(clip)
            try:
                if obj.get("mode", "monocular") == "monocular":
                    object_ok = self._run_monocular_object(
                        video,
                        clip,
                        clip_dir,
                        clip_cfg,
                    )
                else:
                    self._run_foundationpose_object(
                        clip,
                        clip_dir,
                        clip_cfg,
                    )
                    object_ok = True
                if object_ok:
                    processed.append(clip)
                else:
                    unavailable_path = (
                        clip_dir
                        / "object_reconstruction"
                        / "object_unavailable.json"
                    )
                    reason = "degraded to human_only"
                    if unavailable_path.exists():
                        try:
                            unavailable = json.loads(
                                unavailable_path.read_text(encoding="utf-8")
                            )
                            reason = str(unavailable.get("reason", reason))
                        except Exception:
                            pass
                    failed.append((clip, reason))
            except Exception as exc:
                failed.append((clip, str(exc)))
                print(
                    f"[ERROR] object processing failed for {clip}: {exc}",
                    flush=True,
                )
        if not processed and not failed:
            if not configured:
                print(
                    "[INFO] No clips configured for object processing. "
                    "Add entries under object.clips in your pipeline YAML.",
                    flush=True,
                )
            else:
                print(
                    "[INFO] No clips matched both the input directory and "
                    "object.clips configuration.",
                    flush=True,
                )
            return
        if failed:
            print(
                f"[WARN] {len(failed)} clip(s) failed object processing: "
                + ", ".join(f"{c}" for c, _ in failed),
                flush=True,
            )
        if not processed:
            print(
                "[INFO] All configured clips failed object processing; "
                "skipping rerender and dynamic validation.",
                flush=True,
            )
            self._write_summary_csv(processed, failed)
            strict_failed = [
                clip
                for clip, _ in failed
                if not self._clip_object_config(clip).get(
                    "allow_no_object",
                    True,
                )
            ]
            if strict_failed:
                raise RuntimeError(
                    "Required object reconstruction failed for: "
                    + ", ".join(strict_failed)
                )
            return
        if obj.get("rerender_gmr_after_import", True):
            self._rerender_gmr(processed)
            self.write_video_aliases(processed)
        self._dynamic_validation(processed)
        self._write_summary_csv(processed, failed)
        strict_failed = [
            clip
            for clip, _ in failed
            if not self._clip_object_config(clip).get(
                "allow_no_object",
                True,
            )
        ]
        if strict_failed:
            raise RuntimeError(
                "Required object reconstruction failed for: "
                + ", ".join(strict_failed)
            )

    def _write_summary_csv(
        self,
        processed: list[str],
        failed: list[tuple[str, str]],
    ) -> None:
        """Write or update batch summary CSV with object-stage fields."""
        if self.dry_run:
            print("[DRY-RUN] Skipping summary CSV write.", flush=True)
            return

        import csv as _csv

        output_root = self._path(self.config["output"]["root"])
        summary_path = output_root / "summary.csv"
        obj = self.config.get("object", {})

        rows = {}
        existing_fieldnames: list[str] = []
        # Read existing summary if available.
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8", newline="") as fh:
                reader = _csv.DictReader(fh)
                existing_fieldnames = list(reader.fieldnames or [])
                for row in reader:
                    rows[row.get("clip", "")] = row

        object_fieldnames = [
            "clip",
            "human_status",
            "hand_backend",
            "hand_quality_left_mean",
            "hand_quality_right_mean",
            "gmr_robot",
            "object_enabled",
            "object_status",
            "object_source",
            "object_valid_ratio",
            "object_observed_ratio",
            "object_mesh_extent_max_m",
            "object_projection_ok",
            "object_contact_ok",
            "dynamic_lift_success",
            "product_grade",
            "failure_reason",
        ]
        fieldnames = existing_fieldnames + [
            field
            for field in object_fieldnames
            if field not in existing_fieldnames
        ]

        for clip in processed:
            clip_dir = output_root / clip
            row = rows.get(clip, {"clip": clip})
            row.setdefault("clip", clip)
            row.setdefault("human_status", "")
            row.setdefault("gmr_robot", self.environment.get("GMR_ROBOT", ""))
            row.setdefault("hand_backend", self.config.get("human", {}).get("backend", ""))

            row["object_enabled"] = "true"
            row["object_status"] = "ok"

            # Read adapter validation if available.
            val_path = clip_dir / "object_reconstruction" / "object_adapter_validation.json"
            if val_path.exists():
                try:
                    val = json.loads(val_path.read_text(encoding="utf-8"))
                    row["object_valid_ratio"] = str(val.get("valid_ratio", ""))
                    row["object_observed_ratio"] = str(val.get("observed_ratio", ""))
                    extents = val.get("mesh_extents_m")
                    if extents:
                        row["object_mesh_extent_max_m"] = str(round(max(extents), 4))
                    if not val.get("ok"):
                        row["object_status"] = "validation_failed"
                except Exception:
                    pass

            # Read manifest for source.
            manifest_path = clip_dir / "object_reconstruction" / "object_manifest.json"
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    row["object_source"] = str(manifest.get("source", ""))
                except Exception:
                    pass

            # Read dynamic sim report.
            sim_json = clip_dir / "object_reconstruction" / "object_dynamic_sim.json"
            if sim_json.exists():
                try:
                    sim = json.loads(sim_json.read_text(encoding="utf-8"))
                    row["dynamic_lift_success"] = str(
                        sim.get("dynamic_lift_success", "")
                    )
                except Exception:
                    pass

            # Read stage status for product grade.
            stage_path = clip_dir / "object_reconstruction" / "object_stage_status.json"
            if stage_path.exists():
                try:
                    stage = json.loads(stage_path.read_text(encoding="utf-8"))
                    if stage.get("degraded_to_no_object"):
                        row["product_grade"] = "human_only"
                        row["object_status"] = "degraded"
                    else:
                        row["product_grade"] = "object_visual"
                except Exception:
                    pass

            # Check unavailable.
            unavail = clip_dir / "object_reconstruction" / "object_unavailable.json"
            if unavail.exists():
                try:
                    u = json.loads(unavail.read_text(encoding="utf-8"))
                    row["object_status"] = "failed"
                    row["failure_reason"] = str(u.get("reason", ""))
                    row["product_grade"] = "human_only"
                except Exception:
                    pass

            rows[clip] = row

        for clip, reason in failed:
            row = rows.get(clip, {"clip": clip})
            row.setdefault("clip", clip)
            row["object_enabled"] = "true"
            row["object_status"] = "failed"
            row["failure_reason"] = reason
            row["product_grade"] = "human_only"
            rows[clip] = row

        with summary_path.open("w", encoding="utf-8", newline="") as fh:
            writer = _csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for clip_name in sorted(rows):
                writer.writerow(rows[clip_name])

        print(f"Wrote summary CSV: {summary_path}")

    def print_environment(self) -> None:
        selected = {
            key: value
            for key, value in sorted(self.environment.items())
            if key.startswith(
                (
                    "GVHMR_",
                    "GMR_",
                    "FORCE_",
                    "SKIP_",
                    "OBJECT_",
                    "HUNYUAN3D_",
                    "ENV_",
                )
            )
            or key
            in {
                "CONDA_BASE",
                "CUDA_VISIBLE_DEVICES",
                "SEGMENTATION_DISPLAY",
                "PIPELINE_ROOT",
                "DATASET",
                "OUTPUT_BASE",
                "RUN_GMR",
            }
        }
        print(json.dumps(selected, indent=2, ensure_ascii=False))

    def print_effective_config(self) -> None:
        print(
            yaml.safe_dump(
                self.config,
                allow_unicode=True,
                sort_keys=False,
            )
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the pipeline with priority: CLI > environment > pipeline YAML "
            "> default YAML."
        )
    )
    parser.add_argument(
        "--config_dir",
        "--config-dir",
        "--config",
        dest="config_dir",
        type=Path,
        default=Path("configs/pipelines/rgb_monocular_sharpa.yaml"),
        help="Pipeline-specific YAML layered on top of --default_config.",
    )
    parser.add_argument(
        "--default_config",
        "--default-config",
        dest="default_config",
        type=Path,
        default=Path("configs/default.yaml"),
    )
    parser.add_argument(
        "--stage",
        choices=[
            "human",
            "human_pre",
            "scene",
            "human_post",
            "object",
            "quality",
            "product",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--print-env", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--clip-filter", default=None)
    parser.add_argument(
        "--backend",
        choices=sorted(BACKEND_WRAPPERS),
        default=None,
    )
    parser.add_argument(
        "--robot",
        "--hand-model",
        dest="hand_model",
        choices=["sharpa", "g1", "brainco"],
        default=None,
    )
    parser.add_argument(
        "--object-mode",
        choices=["monocular", "foundationpose"],
        default=None,
    )
    parser.add_argument(
        "--mesh-backend",
        choices=["hunyuan"],
        default=None,
        help="Compatibility option; RGB-only reconstruction now always uses Hunyuan3D.",
    )
    parser.add_argument("--object-name", default=None)
    parser.add_argument("--reference-frame", type=int, default=None)
    parser.add_argument(
        "--anchor-hand",
        choices=["left", "right"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--dynamic-release-frame", type=int, default=None)
    parser.add_argument(
        "--collision-method",
        choices=["auto", "coacd", "convex-hull"],
        default=None,
    )
    object_group = parser.add_mutually_exclusive_group()
    object_group.add_argument(
        "--enable-object",
        dest="object_enabled",
        action="store_true",
    )
    object_group.add_argument(
        "--disable-object",
        dest="object_enabled",
        action="store_false",
    )
    parser.set_defaults(object_enabled=None)
    parser.add_argument(
        "--object-no-reconstruct",
        action="store_true",
        help="Skip do-as-i-do reconstruction (use existing raw_dir).",
    )
    parser.add_argument(
        "--object-skip-adapter",
        action="store_true",
        help="Skip adapter validation (MegaPose generates the adapter directly).",
    )
    parser.add_argument(
        "--object-skip-bridge",
        action="store_true",
        help="Skip object bridge import.",
    )
    parser.add_argument(
        "--object-quality-only",
        action="store_true",
        help="Run only QA/validation checks, no reconstruction.",
    )
    parser.add_argument(
        "--object-product-grade",
        choices=["auto", "none", "visual", "contact", "policy"],
        default=None,
    )
    phc_group = parser.add_mutually_exclusive_group()
    phc_group.add_argument(
        "--enable-phc",
        dest="phc_enabled",
        action="store_true",
    )
    phc_group.add_argument(
        "--skip-phc",
        dest="phc_enabled",
        action="store_false",
    )
    parser.set_defaults(phc_enabled=None)
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help=(
            "Generic YAML override, repeatable. Example: "
            "--set human.hand_batch_size=2"
        ),
    )
    return parser.parse_args()


def _cli_overrides(args) -> list[tuple[str, Any]]:
    pairs = [
        ("project_root", args.project_root),
        ("input.dataset_dir", args.dataset_dir),
        ("output.root", args.output_root),
        ("input.clip_filter", args.clip_filter),
        ("human.backend", args.backend),
        ("gmr.hand_model", args.hand_model),
        ("object.mode", args.object_mode),
        ("object.monocular.mesh_backend", args.mesh_backend),
        ("object.runtime_clip_overrides.object_name", args.object_name),
        (
            "object.runtime_clip_overrides.reference_frame",
            args.reference_frame,
        ),
        (
            "object.runtime_clip_overrides.dynamic_release_frame",
            args.dynamic_release_frame,
        ),
        ("object.collision.method", args.collision_method),
        ("object.enabled", args.object_enabled),
        ("phc.enabled", args.phc_enabled),
    ]
    if args.object_no_reconstruct:
        pairs.append(("object.monocular.run_reconstruction", False))
    if args.object_skip_adapter:
        pairs.append(("object.runtime.skip_adapter", True))
    if args.object_skip_bridge:
        pairs.append(("object.runtime.skip_bridge", True))
    if args.object_quality_only:
        pairs.append(("object.runtime.quality_only", True))
    if args.object_product_grade:
        pairs.append(("object.runtime.product_grade", args.object_product_grade))
    overrides = [(path, value) for path, value in pairs if value is not None]
    for item in args.set_values:
        if "=" not in item:
            raise ValueError(f"--set expects PATH=VALUE, got {item!r}")
        path, raw_value = item.split("=", 1)
        path = path.strip()
        if not path:
            raise ValueError(f"--set path is empty: {item!r}")
        overrides.append((path, _parse_cli_value(raw_value)))
    return overrides


def main():
    args = parse_args()
    runner = PipelineRunner(
        default_config_path=args.default_config,
        config_path=args.config_dir,
        cli_overrides=_cli_overrides(args),
        dry_run=args.dry_run,
    )
    if args.print_env:
        runner.print_environment()
    if args.print_config:
        runner.print_effective_config()
    runner.check(args.stage)
    print(
        "Configuration OK\n"
        f"  default:  {args.default_config.resolve()}\n"
        f"  pipeline: {args.config_dir.resolve()}\n"
        "  priority: CLI > environment > pipeline YAML > default YAML"
    )
    if args.check:
        return
    if args.stage == "human":
        runner.run_human()
    elif args.stage == "human_pre":
        runner.run_human_pre()
    elif args.stage == "scene":
        runner.run_scene()
    elif args.stage == "human_post":
        runner.run_human_post()
    elif args.stage == "all":
        if runner._scene_config().get("enabled", False):
            runner.run_human_pre()
            runner.run_scene()
            runner.run_human_post()
        else:
            # Preserve the mature human-only behavior byte-for-byte whenever
            # the new sidecar is disabled.
            runner.run_human()
    if args.stage == "object" or (
        args.stage == "all"
        and runner.config.get("object", {}).get("enabled", False)
    ):
        runner.run_objects()
    quality = runner.config.get("quality_evaluation", {})
    quality_required_for_product = bool(
        args.stage == "product"
        and quality.get("enabled", False)
        and quality.get("require_for_product", False)
    )
    if args.stage == "quality" or quality_required_for_product or (
        quality.get("enabled", False)
        and args.stage in set(quality.get("run_after_stages", []))
    ):
        runner.run_quality(human_only=args.stage == "human")
    product = runner.config.get("product", {})
    if args.stage == "product":
        runner.run_product()
    elif product.get("enabled", False) and args.stage in set(
        product.get("export_after_stages", [])
    ):
        runner.run_product()


if __name__ == "__main__":
    main()
