#!/usr/bin/env python3
"""Tests for no-ground-truth clip quality tiering."""

from __future__ import annotations

import importlib.util
import json
import pickle
import csv
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
        "source_motion": "001_smoothed.npz",
    }
    with (clip_dir / "robot_motion.pkl").open("wb") as stream:
        pickle.dump(robot, stream)
    np.savez(
        clip_dir / "001_sharpa_chain_hands.npz",
        left_hand_qpos=np.zeros((frames, 22), np.float32),
        right_hand_qpos=np.zeros((frames, 22), np.float32),
        left_chain_error_mean=np.float32(0.0025),
        left_chain_error_p95=np.float32(0.0050),
        right_chain_error_mean=np.float32(0.0030),
        right_chain_error_p95=np.float32(0.0060),
    )
    np.savez(
        clip_dir / "gvhmr_camera.npz",
        T_w2c=np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1)),
    )
    (clip_dir / "composite_2x2.mp4").write_bytes(b"test")
    input_video = project / "dataset_new6_work_1280" / f"{clip}.mp4"
    input_video.parent.mkdir(parents=True)
    input_video.write_bytes(b"test")

    xml = (
        project
        / "GMR-master"
        / "assets"
        / "unitree_g1"
        / "g1_mocap_29dof.xml"
    )
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
    assert np.isclose(
        report["metrics"]["left_hand_reproj_error_relative_p50"]["value"], 0.1
    )
    assert np.isclose(
        report["metrics"]["right_hand_reproj_error_relative_p95"]["value"], 0.1
    )
    assert np.isclose(
        report["metrics"]["left_hand_chain_error_mean_mm"]["value"], 2.5
    )
    assert np.isclose(
        report["metrics"]["right_hand_chain_error_p95_mm"]["value"], 6.0
    )


def test_macro_quality_exposes_one_compact_pipeline_view(tmp_path):
    project, config, clip = _fixture(tmp_path)
    report = _run_quality(project, config, clip)

    macro = report["macro_quality"]
    assert macro["score"] == report["overall_score"]
    assert macro["grade"] == "A"
    assert macro["verdict"] == "accept"
    assert [stage["key"] for stage in macro["stages"]] == [
        "files",
        "human",
        "hands",
        "gmr",
        "visualization",
    ]
    assert macro["weakest_stage"]["key"] in {
        "files",
        "human",
        "hands",
        "gmr",
        "visualization",
    }


def test_batch_overview_is_compact_and_replaces_legacy_summaries(tmp_path):
    project, config, clip = _fixture(tmp_path)
    report = _run_quality(project, config, clip)
    output_root = project / "work"
    (output_root / clip / "quality_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    (output_root / "quality_summary.csv").write_text("obsolete\n", encoding="utf-8")
    (output_root / "quality_summary.jsonl").write_text("obsolete\n", encoding="utf-8")

    QUALITY._write_batch_summary(output_root)

    with (output_root / "quality_overview.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["clip"] == clip
    assert rows[0]["mode"] == "human_only"
    assert rows[0]["score"] == "100.0"
    assert rows[0]["grade"] == "A"
    assert rows[0]["verdict"] == "accept"
    assert rows[0]["status"] == "pass"
    assert rows[0]["weakest_stage"] == "files"
    assert rows[0]["weakest_score"] == "100.0"
    assert rows[0]["attention"] == ""
    assert "hands=A/100.0(pass)" in rows[0]["stage_overview"]
    assert not (output_root / "quality_summary.csv").exists()
    assert not (output_root / "quality_summary.jsonl").exists()


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


def test_hand_advisory_keeps_temporal_fills_out_of_direct_evidence(tmp_path):
    project, config, clip = _fixture(tmp_path)
    path = project / "work" / clip / "001_smplx_hands.npz"
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    values.update(
        left_hand_source_reliable=np.asarray([True, True, False, False]),
        right_hand_source_reliable=np.ones(4, bool),
        left_hand_source_repaired=np.asarray([False, False, True, True]),
        right_hand_source_repaired=np.zeros(4, bool),
        left_hand_global_orient=np.zeros((4, 3), np.float32),
        right_hand_global_orient=np.zeros((4, 3), np.float32),
        left_hand_palm_normal=np.tile([1.0, 0.0, 0.0], (4, 1)),
        right_hand_palm_normal=np.tile([1.0, 0.0, 0.0], (4, 1)),
        left_hand_wrist_frame_valid=np.ones(4, bool),
        right_hand_wrist_frame_valid=np.ones(4, bool),
    )
    np.savez(path, **values)
    original = QUALITY._video_metadata
    QUALITY._video_metadata = _mock_video_metadata
    try:
        report = QUALITY.ClipEvaluator(project, config, clip).run()
    finally:
        QUALITY._video_metadata = original

    left = report["metrics"]
    assert np.isclose(left["left_hand_source_reliable_ratio"]["value"], 0.5)
    assert np.isclose(
        left["left_hand_observed_reproj_error_relative_p90"]["value"], 0.1
    )
    assert np.isclose(
        left["left_hand_palm_normal_temporal_dot_p05"]["value"], 1.0
    )
    assert report["hand_data_use_advisory"]["human_hand_estimation"]["status"] == "holdout"
    assert report["hand_data_use_advisory"]["robot_hand_retargeting"]["status"] == "retain"


def _visible_refine_stats(value: float, count: int = 2) -> dict[str, float | int]:
    return {
        "count": count,
        "mean": value,
        "p50": value,
        "p90": value,
        "p95": value,
    }


def _visible_refine_side(
    *,
    include_applied_baseline: bool = True,
    non_tip_holdout: bool = False,
) -> dict:
    side = {
        "evidence_available": True,
        "missing_evidence_fields": [],
        "visible_evidence": {"frames": 4, "longest_run": 4},
        "refine_eligible": {"frames": 3, "longest_run": 3},
        "fit_frames": {"frames": 2, "longest_run": 2},
        "applied": {"frames": 2, "longest_run": 2},
        "absolute_reprojection_after_px": _visible_refine_stats(8.0),
        "relative_reprojection_after_px": _visible_refine_stats(5.0),
        "accepted_delta_degrees": _visible_refine_stats(12.0),
        "rejection_counts": {"no_improvement": 1},
    }
    if include_applied_baseline:
        side["absolute_reprojection_before_applied_px"] = _visible_refine_stats(10.0)
        side["relative_reprojection_before_applied_px"] = _visible_refine_stats(7.0)
    if non_tip_holdout:
        side.update(
            fit_partition="non_tip",
            holdout_fit_frames={"frames": 3, "longest_run": 3},
            holdout_relative_reprojection_before_applied_px=_visible_refine_stats(
                9.0
            ),
            holdout_relative_reprojection_after_px=_visible_refine_stats(6.0),
        )
    return side


def _write_visible_refine_audit(
    clip_dir: Path,
    *,
    include_applied_baseline: bool = True,
    matching_direct_input: bool = True,
    non_tip_holdout: bool = False,
) -> None:
    refined = clip_dir / "mano_params_visible_refined.pt"
    audit = {
        "schema_version": 1,
        "purpose": "audit-only acceptance report for visible-hand wrist refinement",
        "output": str(refined),
        "limits": {"palm_back_semantics": "not evaluated"},
        "sides": {
            "left": _visible_refine_side(
                include_applied_baseline=include_applied_baseline,
                non_tip_holdout=non_tip_holdout,
            ),
            "right": _visible_refine_side(
                include_applied_baseline=include_applied_baseline,
                non_tip_holdout=non_tip_holdout,
            ),
        },
    }
    (clip_dir / "mano_params_visible_refined.json").write_text(
        json.dumps(audit), encoding="utf-8"
    )
    direct_input = refined if matching_direct_input else clip_dir / "other_mano.pt"
    (clip_dir / "mano_params_direct_mano_recomputed.json").write_text(
        json.dumps({"input": str(direct_input)}), encoding="utf-8"
    )


def _run_quality(project: Path, config: dict, clip: str) -> dict:
    original = QUALITY._video_metadata
    QUALITY._video_metadata = _mock_video_metadata
    try:
        return QUALITY.ClipEvaluator(project, config, clip).run()
    finally:
        QUALITY._video_metadata = original


def test_visible_refine_audit_is_informational_and_same_population_only(tmp_path):
    project, config, clip = _fixture(tmp_path)
    clip_dir = project / "work" / clip
    _write_visible_refine_audit(clip_dir)

    report = _run_quality(project, config, clip)

    assert report["status"] == "pass"
    assert report["stage_status"] == {
        "files": "pass",
        "human": "pass",
        "hands": "pass",
        "gmr": "pass",
        "visualization": "pass",
    }
    metric = report["metrics"]["visible_hand_refinement"]
    assert metric["status"] == "info"
    assert metric["value"] is True
    left = metric["details"]["audit"]["sides"]["left"]
    assert left["comparison_status"] == "available"
    assert left["refine_eligible"]["ratio"] == 0.75
    assert left["absolute_reprojection_before_applied_px"]["p95"] == 10.0
    assert left["absolute_reprojection_after_px"]["p95"] == 8.0


def test_visible_refine_legacy_summary_never_mixes_populations(tmp_path):
    project, config, clip = _fixture(tmp_path)
    _write_visible_refine_audit(
        project / "work" / clip,
        include_applied_baseline=False,
    )

    report = _run_quality(project, config, clip)

    assert report["status"] == "pass"
    metric = report["metrics"]["visible_hand_refinement"]
    assert metric["status"] == "skipped"
    left = metric["details"]["audit"]["sides"]["left"]
    assert left["comparison_status"] == "skipped"
    assert left["comparison_reason"] == "same_population_baseline_missing_or_invalid"
    assert left["absolute_reprojection_before_applied_px"] is None
    assert left["absolute_reprojection_after_px"] is None


def test_visible_refine_non_tip_holdout_is_reported_separately(tmp_path):
    project, config, clip = _fixture(tmp_path)
    _write_visible_refine_audit(
        project / "work" / clip,
        non_tip_holdout=True,
    )

    report = _run_quality(project, config, clip)

    left = report["metrics"]["visible_hand_refinement"]["details"]["audit"][
        "sides"
    ]["left"]
    assert left["fit_partition"] == "non_tip"
    assert left["holdout_comparison_status"] == "available"
    assert left["holdout_fit_frames"]["ratio"] == 0.75
    assert left["holdout_relative_reprojection_before_applied_px"]["p95"] == 9.0
    assert left["holdout_relative_reprojection_after_px"]["p95"] == 6.0


def test_visible_refine_audit_requires_direct_mano_provenance_match(tmp_path):
    project, config, clip = _fixture(tmp_path)
    _write_visible_refine_audit(
        project / "work" / clip,
        matching_direct_input=False,
    )

    report = _run_quality(project, config, clip)

    assert report["status"] == "pass"
    metric = report["metrics"]["visible_hand_refinement"]
    assert metric["status"] == "skipped"
    assert (
        metric["details"]["reason"]
        == "direct_mano_input_does_not_match_visible_refine_output"
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


def test_input_motion_duration_mismatch_is_critical_fail(tmp_path):
    project, config, clip = _fixture(tmp_path)
    original = QUALITY._video_metadata

    def metadata(path: Path) -> dict:
        value = _mock_video_metadata(path)
        if path.parent.name == "dataset_new6_work_1280":
            value["fps"] = 25.0
        return value

    QUALITY._video_metadata = metadata
    try:
        report = QUALITY.ClipEvaluator(project, config, clip).run()
    finally:
        QUALITY._video_metadata = original

    metric = report["metrics"]["input_motion_duration_relative_error"]
    assert report["status"] == "fail"
    assert report["stage_status"]["files"] == "fail"
    assert metric["status"] == "fail"
    assert metric["value"] > 0.03
    assert report["metrics"]["work_video_fps_relative_error"]["status"] == "fail"


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
