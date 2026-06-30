#!/usr/bin/env python3
"""Configuration-driven launcher for the human + hand + object pipeline."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
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
    "GMR_CAMERA_SOURCE": "gmr.camera_source",
    "GMR_RENDER": "gmr.render",
    "GMR_COMPOSITE": "gmr.composite_2x2",
    "GMR_MUJOCO_GL": "gmr.mujoco_gl",
    "GMR_RENDER_WIDTH": "gmr.render_width",
    "GMR_RENDER_HEIGHT": "gmr.render_height",
    "OBJECT_ENABLED": "object.enabled",
    "OBJECT_MODE": "object.mode",
    "OBJECT_MESH_BACKEND": "object.monocular.mesh_backend",
    "HUNYUAN3D_FACE_COUNT": "object.monocular.hunyuan_face_count",
    "HUNYUAN3D_TIMEOUT": "object.monocular.hunyuan_timeout",
    "HUNYUAN3D_REGION": "object.monocular.hunyuan_region",
    "OBJECT_NUM_POSE_SAMPLES": "object.monocular.num_pose_samples",
    "OBJECT_EULER_STEPS": "object.monocular.euler_steps",
    "OBJECT_TORCH_COMPILE": "object.monocular.torch_compile",
    "OBJECT_NAME": "object.runtime_clip_overrides.object_name",
    "OBJECT_REFERENCE_FRAME": "object.runtime_clip_overrides.reference_frame",
    "OBJECT_ANCHOR_HAND": "object.runtime_clip_overrides.anchor_hand",
    "OBJECT_DYNAMIC_RELEASE_FRAME": (
        "object.runtime_clip_overrides.dynamic_release_frame"
    ),
    "PIPELINE_PYTHON": "runtime.python",
    "CUDA_VISIBLE_DEVICES": "runtime.cuda_visible_devices",
    "SAM3_DISPLAY": "runtime.sam3_display",
    "ENV_SAM3": "runtime.conda_envs.sam3",
    "ENV_SAM3D": "runtime.conda_envs.sam3d",
    "ENV_HAWOR": "runtime.conda_envs.hawor",
    "ENV_TAPNET": "runtime.conda_envs.tapnet",
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
                "SAM3_DISPLAY": str(runtime.get("sam3_display", ":1")),
                "ENV_SAM3": str(
                    runtime.get("conda_envs", {}).get("sam3", "sam3")
                ),
                "ENV_SAM3D": str(
                    runtime.get("conda_envs", {}).get("sam3d", "sam3d")
                ),
                "ENV_HAWOR": str(
                    runtime.get("conda_envs", {}).get("hawor", "hawor")
                ),
                "ENV_TAPNET": str(
                    runtime.get("conda_envs", {}).get("tapnet", "tapnet")
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
                "OBJECT_MESH_BACKEND": str(
                    monocular.get("mesh_backend", "sam3d")
                ),
                "HUNYUAN3D_FACE_COUNT": str(
                    monocular.get("hunyuan_face_count", 100000)
                ),
                "HUNYUAN3D_TIMEOUT": str(
                    monocular.get("hunyuan_timeout", 900)
                ),
                "HUNYUAN3D_REGION": str(
                    monocular.get("hunyuan_region", "ap-guangzhou")
                ),
                "OBJECT_NUM_POSE_SAMPLES": str(
                    monocular.get("num_pose_samples", 25)
                ),
                "OBJECT_EULER_STEPS": str(
                    monocular.get("euler_steps", 25)
                ),
                "OBJECT_TORCH_COMPILE": _bool_env(
                    monocular.get("torch_compile", True)
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

    def check(self, stage: str) -> None:
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        runtime_python = Path(self.runtime_python)
        if not runtime_python.is_file():
            raise FileNotFoundError(
                f"runtime.python is not a file: {runtime_python}"
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
            if mode == "monocular":
                mono = obj.get("monocular", {})
                mesh_backend = mono.get("mesh_backend", "sam3d")
                if mesh_backend not in {"sam3d", "hunyuan"}:
                    raise ValueError(
                        f"unsupported object.monocular.mesh_backend: {mesh_backend}"
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
                    if mesh_backend == "hunyuan":
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
                                "Hunyuan mesh backend requires rotated credentials "
                                "in process environment: "
                                + ", ".join(missing_credentials)
                                + ". Source configs/secrets.env first."
                            )
                    module_dir = (
                        self.root
                        / "do-as-i-do-main"
                        / "reconstruction"
                        / "modules"
                        / "sam3"
                    )
                    if not module_dir.is_dir() or not any(module_dir.iterdir()):
                        raise RuntimeError(
                            "do-as-i-do reconstruction submodules are empty. Run "
                            "`cd do-as-i-do-main/reconstruction && "
                            "./setup/00_init_submodules.sh`, then prepare its environments."
                        )
                    runtime = self.config.get("runtime", {})
                    conda_base = self._path(
                        runtime.get(
                            "conda_base",
                            str(Path.home() / "miniconda3"),
                        )
                    )
                    missing_envs = []
                    for env_name in runtime.get("conda_envs", {}).values():
                        env_path = Path(str(env_name)).expanduser()
                        if not env_path.is_absolute():
                            env_path = conda_base / "envs" / env_path
                        if not (env_path / "bin" / "python").is_file():
                            missing_envs.append(str(env_name))
                    if missing_envs:
                        raise RuntimeError(
                            "missing do-as-i-do Conda environments: "
                            + ", ".join(sorted(set(missing_envs)))
                            + ". Run `cd do-as-i-do-main/reconstruction && "
                            "./setup/01_create_envs.sh`."
                        )

                    reconstruction_root = (
                        self.root / "do-as-i-do-main" / "reconstruction"
                    )
                    required_weights = [
                        reconstruction_root
                        / "weights"
                        / "tapnet"
                        / "bootstapir_checkpoint_v2.pt",
                        reconstruction_root
                        / "weights"
                        / "sam3d_shared"
                        / "hf"
                        / "slat_generator.ckpt",
                        reconstruction_root
                        / "modules"
                        / "HaWoR"
                        / "weights"
                        / "hawor"
                        / "checkpoints"
                        / "hawor.ckpt",
                        reconstruction_root
                        / "modules"
                        / "HaWoR"
                        / "_DATA"
                        / "data"
                        / "mano"
                        / "MANO_RIGHT.pkl",
                        reconstruction_root
                        / "modules"
                        / "HaWoR"
                        / "_DATA"
                        / "data_left"
                        / "mano_left"
                        / "MANO_LEFT.pkl",
                    ]
                    missing_weights = [
                        str(path.relative_to(reconstruction_root))
                        for path in required_weights
                        if not path.is_file()
                    ]
                    if missing_weights:
                        raise RuntimeError(
                            "missing do-as-i-do weights/assets: "
                            + ", ".join(missing_weights)
                            + ". Run `./setup/02_fetch_weights.sh --download`; "
                            "MANO_LEFT/RIGHT.pkl require a licensed manual download."
                        )
        self._videos()

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

    def _clip_object_config(self, clip: str) -> dict:
        obj = self.config.get("object", {})
        mono = obj.get("monocular", {})
        defaults = {
            "object_name": mono.get("default_object_name", "object"),
            "reference_frame": mono.get("default_reference_frame", 0),
            "anchor_hand": mono.get("default_anchor_hand", "right"),
            "frame_offset": 0,
            "robot_offset": "0,0,0",
            "dynamic_release_frame": 0,
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
    ) -> None:
        obj = self.config["object"]
        mono = obj.get("monocular", {})
        work_root = self._path(
            mono.get("work_root", "object_work/do_as_i_do")
        )
        raw_dir, work_video = self._prepare_monocular_video(
            video,
            clip,
            work_root,
        )
        if mono.get("run_reconstruction", True):
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
                    str(work_video),
                    str(clip_cfg["reference_frame"]),
                    str(clip_cfg["object_name"]),
                    str(clip_cfg["anchor_hand"]),
                ]
            )

        adapter_dir = clip_dir / "object_reconstruction" / "monocular_adapter"
        self._run(
            [
                self.runtime_python,
                str(
                    self.root
                    / "GMR-master"
                    / "scripts"
                    / "adapt_do_as_i_do_object.py"
                ),
                "--raw_dir",
                str(raw_dir),
                "--object_name",
                str(clip_cfg["object_name"]).replace(" ", "_"),
                "--output_dir",
                str(adapter_dir),
            ]
        )
        self._run_object_bridge(adapter_dir, clip_dir, clip_cfg)

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
            if obj.get("mode", "monocular") == "monocular":
                self._run_monocular_object(
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
            processed.append(clip)
        if not processed:
            raise ValueError("no clips selected for object processing")
        if obj.get("rerender_gmr_after_import", True):
            self._rerender_gmr(processed)
        self._dynamic_validation(processed)

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
                "SAM3_DISPLAY",
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
        choices=["human", "object", "all"],
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
        choices=["sam3d", "hunyuan"],
        default=None,
        help="RGB-only reference mesh generator.",
    )
    parser.add_argument("--object-name", default=None)
    parser.add_argument("--reference-frame", type=int, default=None)
    parser.add_argument(
        "--anchor-hand",
        choices=["left", "right"],
        default=None,
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
        ("object.runtime_clip_overrides.anchor_hand", args.anchor_hand),
        (
            "object.runtime_clip_overrides.dynamic_release_frame",
            args.dynamic_release_frame,
        ),
        ("object.collision.method", args.collision_method),
        ("object.enabled", args.object_enabled),
        ("phc.enabled", args.phc_enabled),
    ]
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


if __name__ == "__main__":
    main()
