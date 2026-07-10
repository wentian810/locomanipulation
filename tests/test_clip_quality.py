#!/usr/bin/env python3
"""Tests for no-ground-truth clip quality tiering."""

from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_clip_quality",
    ROOT / "scripts" / "evaluate_clip_quality.py",
)
QUALITY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(QUALITY)


def _fixture(tmp_path: Path) -> tuple[Path, dict, str]:
    project = tmp_path / "project"
    clip = "sample"
    clip_dir = project / "work" / clip
    clip_dir.mkdir(parents=True)
    frames = 4
    fps = 30.0
    body = {
        "root_orient": np.zeros((frames, 3), np.float32),
        "pose_body": np.zeros((frames, 63), np.float32),
        "trans": np.zeros((frames, 3), np.float32),
        "betas": np.zeros(16, np.float32),
        "gender": np.asarray("neutral"),
        "mocap_frame_rate": np.float32(fps),
    }
    np.savez(clip_dir / "001_smoothed.npz", **body)
    np.savez(clip_dir / "001_phc_smoothed.npz", **body)
    np.savez(
        clip_dir / "001_smplx_hands.npz",
        left_hand_pose=np.zeros((frames, 45), np.float32),
        right_hand_pose=np.zeros((frames, 45), np.float32),
        left_hand_valid=np.ones(frames, bool),
        right_hand_valid=np.ones(frames, bool),
        left_hand_reproj_error_relative=np.full(frames, 0.1, np.float32),
        right_hand_reproj_error_relative=np.full(frames, 0.1, np.float32),
        left_hand_spike_mask=np.zeros(frames, bool),
        right_hand_spike_mask=np.zeros(frames, bool),
        left_hand_source_repaired=np.zeros(frames, bool),
        right_hand_source_repaired=np.zeros(frames, bool),
    )
    robot = {
        "fps": fps,
        "root_pos": np.zeros((frames, 3), np.float32),
        "root_rot": np.tile([0.0, 0.0, 0.0, 1.0], (frames, 1)),
        "dof_pos": np.zeros((frames, 2), np.float32),
    }
    with (clip_dir / "robot_motion.pkl").open("wb") as stream:
        pickle.dump(robot, stream)
    np.savez(
        clip_dir / "001_sharpa_chain_hands.npz",
        left_hand_qpos=np.zeros((frames, 22), np.float32),
        right_hand_qpos=np.zeros((frames, 22), np.float32),
    )
    np.savez(
        clip_dir / "gvhmr_camera.npz",
        T_w2c=np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1)),
    )
    (clip_dir / "composite_2x2.mp4").write_bytes(b"test")

    xml = project / "GMR-master" / "assets" / "unitree_h1" / "h1_with_hand.xml"
    xml.parent.mkdir(parents=True)
    xml.write_text(
        """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="root"><freejoint/>
      <body><joint name="joint_a" range="-1 1"/>
        <body><joint name="joint_b" range="-1 1"/></body>
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip(),
        encoding="utf-8",
    )
    config = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    config["output"]["root"] = "work"
    config["gmr"]["source"] = "smoothed"
    config["gmr"]["hand_model"] = "sharpa"
    config["object"]["enabled"] = False
    return project, config, clip


def _mock_video_metadata(_path: Path) -> dict:
    return {
        "readable": True,
        "frames": 4,
        "fps": 30.0,
        "width": 640,
        "height": 480,
    }


def test_good_human_clip_passes(tmp_path):
    project, config, clip = _fixture(tmp_path)
    original = QUALITY._video_metadata
    QUALITY._video_metadata = _mock_video_metadata
    try:
        report = QUALITY.ClipEvaluator(project, config, clip).run()
    finally:
        QUALITY._video_metadata = original

    assert report["status"] == "pass"
    assert report["stage_status"] == {
        "files": "pass",
        "human": "pass",
        "hands": "pass",
        "gmr": "pass",
        "visualization": "pass",
    }
    assert report["metrics"]["frame_count_match_ratio"]["value"] == 1.0


def test_reprojection_p90_creates_warn_tier(tmp_path):
    project, config, clip = _fixture(tmp_path)
    path = project / "work" / clip / "001_smplx_hands.npz"
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    values["right_hand_reproj_error_relative"] = np.full(
        4,
        0.42,
        np.float32,
    )
    np.savez(path, **values)
    original = QUALITY._video_metadata
    QUALITY._video_metadata = _mock_video_metadata
    try:
        report = QUALITY.ClipEvaluator(project, config, clip).run()
    finally:
        QUALITY._video_metadata = original

    assert report["status"] == "warn"
    assert report["stage_status"]["hands"] == "warn"
    assert any(
        reason.startswith("right_hand_reproj_error_relative_p90=")
        for reason in report["warn_reasons"]
    )


def test_missing_required_robot_motion_is_critical_fail(tmp_path):
    project, config, clip = _fixture(tmp_path)
    (project / "work" / clip / "robot_motion.pkl").unlink()
    original = QUALITY._video_metadata
    QUALITY._video_metadata = _mock_video_metadata
    try:
        report = QUALITY.ClipEvaluator(project, config, clip).run()
    finally:
        QUALITY._video_metadata = original

    assert report["status"] == "fail"
    assert report["stage_status"]["files"] == "fail"
    assert report["critical_fail_reasons"]


def test_object_pose_jump_counter_combines_translation_and_rotation():
    position = np.asarray(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    quaternion = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    result = QUALITY._object_pose_jump_metrics(
        position,
        quaternion,
        np.ones(3, bool),
        position_threshold_m=0.2,
        rotation_threshold_deg=45.0,
    )
    assert result["count"] == 1
    assert result["position_jump_count"] == 1
    assert result["rotation_jump_count"] == 1
