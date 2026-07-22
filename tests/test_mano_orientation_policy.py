"""Regression tests for the isolated MANO wrist-orientation A/B switches."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TEMPORAL_PATH = ROOT / "GVHMR-hand/GVHMR-main/tools/processor/filter_mano_temporal.py"
FINGER_PATH = ROOT / "GVHMR-hand/GVHMR-main/tools/processor/filter_mano_fingers.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def rotation_z(degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    cosine, sine = np.cos(radians), np.sin(radians)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def test_temporal_preserve_does_not_interpolate_global_orientation():
    temporal = load_module("mano_temporal_policy", TEMPORAL_PATH)
    values = np.stack([rotation_z(0), rotation_z(-35), rotation_z(90)])[:, None]
    fill = np.asarray([False, True, False])
    reliable = ~fill

    preserved = temporal.apply_global_orient_fill_policy(
        values, fill, reliable, "preserve"
    )
    interpolated = temporal.apply_global_orient_fill_policy(
        values, fill, reliable, "interpolate"
    )

    assert np.array_equal(preserved, values)
    assert not np.allclose(interpolated[1], values[1])


def test_finger_preserve_leaves_global_orientation_and_masks_unchanged():
    fingers = load_module("mano_finger_policy", FINGER_PATH)
    global_orient = np.stack([rotation_z(0), rotation_z(45), rotation_z(90)])[:, None]
    args = SimpleNamespace(wrist_mode="preserve")

    out, filled, limited, changed = fingers.apply_wrist_orientation_policy(
        mano={},
        side="left",
        person_idx=0,
        global_orient=global_orient,
        valid=np.asarray([True, True, True]),
        low_evidence=np.asarray([False, True, False]),
        args=args,
    )

    assert np.array_equal(out, global_orient)
    assert not filled.any()
    assert not limited.any()
    assert not changed.any()
