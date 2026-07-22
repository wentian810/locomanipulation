"""Unit tests for the no-manual-label fingertip holdout partition."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
GVHMR_ROOT = ROOT / "GVHMR-hand" / "GVHMR-main"
if str(GVHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(GVHMR_ROOT))
SPEC = importlib.util.spec_from_file_location(
    "visible_hand_refiner",
    GVHMR_ROOT / "tools" / "processor" / "refine_visible_hand_orientation.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_all_partition_reproduces_all_high_confidence_points() -> None:
    confidence = torch.tensor(
        [[True] * 21, [True] + [False] * 20], dtype=torch.bool
    )

    fit, holdout = MODULE.partition_fit_points(confidence, "all")

    assert torch.equal(fit, confidence)
    assert not bool(holdout.any())


def test_non_tip_partition_reserves_exactly_five_tips() -> None:
    confidence = torch.ones((1, 21), dtype=torch.bool)

    fit, holdout = MODULE.partition_fit_points(confidence, "non_tip")

    assert int(fit.sum()) == 16
    assert int(holdout.sum()) == 5
    assert not bool(fit[:, list(MODULE.FINGERTIP_INDICES)].any())
    assert bool(holdout[:, list(MODULE.FINGERTIP_INDICES)].all())


def test_partition_rejects_unknown_mode_or_shape() -> None:
    with pytest.raises(ValueError, match="unsupported fit partition"):
        MODULE.partition_fit_points(torch.ones((1, 21), dtype=torch.bool), "bad")
    with pytest.raises(ValueError, match="shape"):
        MODULE.partition_fit_points(torch.ones((1, 20), dtype=torch.bool), "all")
