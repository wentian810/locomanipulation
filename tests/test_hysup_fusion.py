"""Safety regression tests for HySUP-inspired temporal-anchor selection."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "GVHMR-hand/GVHMR-main/tools/processor/hysup_fusion.py"


def load_module():
    spec = importlib.util.spec_from_file_location("hysup_fusion_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def rotation_z(degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    cosine, sine = np.cos(radians), np.sin(radians)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def test_repaired_frame_is_not_used_as_hysup_anchor():
    hysup = load_module()
    pose = np.stack([rotation_z(0), rotation_z(-40), rotation_z(80)])[None, :, None]
    pose = np.repeat(pose, 15, axis=2)
    joints = np.zeros((1, 3, 21, 3), dtype=np.float64)
    quality = np.ones((1, 3), dtype=np.float64)
    source_reliable = np.asarray([[True, False, True]])

    pose_ref, _, source = hysup.temporal_reference(
        pose,
        joints,
        quality,
        source_reliable,
        anchor_quality=0.85,
        max_gap=12,
        edge_hold=6,
    )

    assert source[0, 1] == 1
    assert not np.allclose(pose_ref[0, 1], pose[0, 1])
