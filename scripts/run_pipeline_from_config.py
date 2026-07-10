#!/usr/bin/env python3
"""Configuration-driven launcher for the human + hand + object pipeline."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

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
    "GVHMR_DIAGNOSE_HAND": "human.diagnostics",
    "RUN_GMR": "gmr.enabled",
    "GMR_HAND_MODEL": "gmr.hand_model",
    "GMR_SOURCE": "gmr.source",
    "GMR_TARGET_FPS": "gmr.target_fps",
    "GMR_HUMAN_YAW_OFFSET_DEG": "gmr.human_yaw_offset_deg",
    "GMR_CAMERA_SOURCE": "gmr.camera_source",
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
        locomotion = cfg.get("locomotion", {})
        phc = cfg.get("phc", {})
        gmr = cfg.get("gmr", {})
        object_cfg = cfg.get("object", {})
        monocular = object_cfg.get("monocular", {})
        runtime = cfg.get("runtime", {})

        env = os.environ.copy()
        # Free-form compatibility values have already passed through the same
        # YAML -> process environment -> CLI merge. Managed values are applied
        # afterwards so stale inherited shell variables cannot undo a CLI
        # option such as --robot or --backend.
        for key, value in cfg.get("environment", {}).items():
            env[str(key)] = _env_value(value)
        env.update(
            {
                "PIPELINE_ROOT": str(self.root),
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
                "GMR_HUMAN_YAW_OFFSET_DEG": str(
                    gmr.get("human_yaw_offset_deg", 0.0)
                ),
                "GMR_CAMERA_SOURCE": str(
                    gmr.get("camera_source", "gvhmr")
                ),
                "GMR_RENDER": _bool_env(gmr.get("render", True)),
                "GMR_COMPOSITE": _bool_env(
                    gmr.get("composite_2x2", True)
                ),
                "GMR_MUJOCO_GL": str(gmr.get("mujoco_gl", "osmesa")),
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
        if model == "sharpa":
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

    def _videos(self) -> list[Path]:
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
        videos = [
            path
            for path in sorted(dataset.iterdir())
            if path.is_file()
            and path.suffix.lower() in extensions
            and (not clip_filter or clip_filter in path.name)
        ]
        if not videos:
            raise ValueError(f"no input videos found in {dataset}")
        return videos

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

    def check(self, stage: str) -> None:
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        runtime_python = Path(self.runtime_python)
        if not runtime_python.is_file():
            raise FileNotFoundError(
                f"runtime.python is not a file: {runtime_python}"
            )
        videos = self._videos()
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
        if stage == "product":
            output_root = self._path(self.config["output"]["root"])
            missing_human_outputs = [
                output_root / video.stem / "robot_motion.pkl"
                for video in self._product_videos()
                if not (output_root / video.stem / "robot_motion.pkl").is_file()
            ]
            if missing_human_outputs and not self.dry_run:
                missing = "\n  ".join(str(path) for path in missing_human_outputs)
                raise FileNotFoundError(
                    "product export requires completed human/GMR outputs:\n  "
                    f"{missing}"
                )
        if stage == "quality":
            quality = self.config.get("quality_evaluation", {})
            if not quality.get("enabled", False):
                raise ValueError("quality_evaluation.enabled is false")
            output_root = self._path(self.config["output"]["root"])
            missing_outputs = [
                output_root / video.stem / "robot_motion.pkl"
                for video in self._quality_videos()
                if not (output_root / video.stem / "robot_motion.pkl").is_file()
            ]
            if missing_outputs and not self.dry_run:
                missing = "\n  ".join(str(path) for path in missing_outputs)
                raise FileNotFoundError(
                    "quality evaluation requires completed GMR outputs:\n  "
                    f"{missing}"
                )
        if stage in {"human", "all"}:
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
        if stage in {"object", "all"}:
            obj = self.config.get("object", {})
            if not obj.get("enabled", False):
                if stage == "object":
                    raise ValueError("object.enabled is false")
                return
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
        self._run(["bash", str(wrapper)])
        self.write_video_aliases()

    def run_product(self) -> None:
        """Export safe NPZ digital-asset bundles from trusted work outputs."""
        from export_dataset_product import export_from_config

        clips = [video.stem for video in self._product_videos()]
        export_from_config(
            self.root,
            self.config,
            clips,
            dry_run=self.dry_run,
        )

    def run_quality(self, *, human_only: bool = False) -> None:
        """Evaluate completed clips and write pass/warn/fail reports."""
        from evaluate_clip_quality import evaluate_clips

        quality_config = self.config
        videos = self._quality_videos()
        if human_only and self.config.get("object", {}).get("enabled", False):
            quality_config = copy.deepcopy(self.config)
            quality_config.setdefault("object", {})["enabled"] = False
            videos = self._videos()
        clips = [video.stem for video in videos]
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

    def _rerender_gmr(self, clips: list[str]) -> None:
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
            if hand_model == "sharpa":
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
        choices=["human", "object", "quality", "product", "all"],
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
    if args.stage in {"human", "all"}:
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
