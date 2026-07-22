"""Fail-closed config checks for visible-hand wrist refinement."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_pipeline_from_config.py"
SPEC = importlib.util.spec_from_file_location("pipeline_config_runner", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_visible_hand_refine_config_accepts_valid_minimal_mapping() -> None:
    config = {"enabled": False, "device": "auto", "steps": 0}
    assert MODULE._validate_visible_hand_refine_config(config) == config


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"enabled": "false"}, "enabled must be a boolean"),
        ({"device": "gpu"}, "device must be 'auto', 'cuda', or 'cpu'"),
        ({"batch_size": 0}, "batch_size must be >= 1"),
        ({"fit_confidence": 1.1}, "fit_confidence must be <= 1.0"),
        ({"fit_partition": "tips"}, "fit_partition must be 'all' or 'non_tip'"),
        ({"holdout_min_keypoints": 6}, "holdout_min_keypoints must be >= 1 and <= 5"),
        (
            {"fit_partition": "non_tip", "fit_min_keypoints": 17},
            "fit_min_keypoints must be <= 16",
        ),
        ({"evidence_min_keypoints": 22}, "evidence_min_keypoints must be >= 1 and <= 21"),
    ],
)
def test_visible_hand_refine_config_rejects_invalid_values(
    config: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        MODULE._validate_visible_hand_refine_config(config)
