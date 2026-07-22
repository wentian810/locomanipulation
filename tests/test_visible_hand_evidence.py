import importlib.util
from pathlib import Path
import sys

import numpy as np


MODULE = (
    Path(__file__).resolve().parents[1]
    / "GVHMR-hand"
    / "GVHMR-main"
    / "tools"
    / "processor"
    / "visible_hand_evidence.py"
)
SPEC = importlib.util.spec_from_file_location("visible_hand_evidence", MODULE)
EVIDENCE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EVIDENCE
SPEC.loader.exec_module(EVIDENCE)


def _mano(frames=5):
    shape = (1, frames)
    mano = {
        "left_hand_reliable_mask": np.ones(shape, dtype=bool),
        "left_hand_bbox_xyxy": np.tile(np.array([100, 100, 220, 220], dtype=np.float32), (1, frames, 1)),
    }
    for suffix in EVIDENCE._UNSAFE_MASK_SUFFIXES:
        mano[f"left_{suffix}"] = np.zeros(shape, dtype=bool)
    return mano


def _points(frames=5):
    hand = np.zeros((1, frames, 21, 3), dtype=np.float32)
    hand[..., 0] = 160.0
    hand[..., 1] = 160.0
    hand[..., 2] = 0.8
    body = np.zeros((1, frames, 17, 3), dtype=np.float32)
    body[..., 7, :] = [120.0, 160.0, 0.8]  # left elbow
    body[..., 9, :] = [160.0, 160.0, 0.8]  # left wrist
    return hand, body


def test_strict_gate_accepts_contiguous_high_evidence_frames():
    hand, body = _points()
    result = EVIDENCE.build_visible_evidence(_mano(), "left", hand, body)
    assert result["available"]
    assert result["visible_evidence_mask"].all()
    assert result["refine_eligible_mask"].all()


def test_strict_gate_rejects_short_runs_and_missing_provenance():
    hand, body = _points()
    hand[:, 2, :, 2] = 0.0
    result = EVIDENCE.build_visible_evidence(_mano(), "left", hand, body)
    assert not result["refine_eligible_mask"].any()

    mano = _mano()
    del mano["left_hand_bbox_jump_mask"]
    unavailable = EVIDENCE.build_visible_evidence(mano, "left", hand, body)
    assert not unavailable["available"]
    assert unavailable["reason_code"].max() == EVIDENCE.REASON_SCHEMA_UNAVAILABLE
