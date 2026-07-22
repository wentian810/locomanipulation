#!/usr/bin/env python3
"""Tests for compact, pickle-free pipeline product bundles."""

from __future__ import annotations

import importlib.util
import json
import pickle
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "export_dataset_product",
    ROOT / "scripts" / "export_dataset_product.py",
)
EXPORTER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EXPORTER)


def _fixture(tmp_path: Path) -> tuple[Path, dict, str]:
    project = tmp_path / "project"
    clip = "sample"
    clip_dir = project / "work" / clip
    clip_dir.mkdir(parents=True)
    frames = 4
    fps = 30.0

    body_values = {
        "root_orient": np.zeros((frames, 3), np.float32),
        "pose_body": np.zeros((frames, 63), np.float32),
        "trans": np.zeros((frames, 3), np.float32),
        "betas": np.zeros(16, np.float32),
        "gender": np.asarray("neutral"),
        "mocap_frame_rate": np.float32(fps),
    }
    np.savez(clip_dir / "001_smoothed.npz", **body_values)
    np.savez(clip_dir / "001_phc_smoothed.npz", **body_values)
    np.savez(
        clip_dir / "001_smplx_hands.npz",
        left_hand_pose=np.zeros((frames, 45), np.float32),
        right_hand_pose=np.zeros((frames, 45), np.float32),
        left_hand_valid=np.ones(frames, bool),
        right_hand_valid=np.ones(frames, bool),
        left_hand_quality=np.ones(frames, np.float32),
        right_hand_quality=np.ones(frames, np.float32),
        left_hand_bbox_xyxy=np.full((frames, 4), np.nan, np.float32),
        right_hand_bbox_xyxy=np.zeros((frames, 4), np.float32),
        hand_backend=np.asarray("test"),
    )
    robot = {
        "fps": fps,
        "root_pos": np.zeros((frames, 3), np.float32),
        "root_rot": np.tile([0.0, 0.0, 0.0, 1.0], (frames, 1)),
        "dof_pos": np.zeros((frames, 2), np.float32),
        "human_yaw_offset_deg": 180.0,
    }
    with (clip_dir / "robot_motion.pkl").open("wb") as stream:
        pickle.dump(robot, stream)
    np.savez(
        clip_dir / "001_sharpa_chain_hands.npz",
        left_hand_qpos=np.zeros((frames, 22), np.float32),
        right_hand_qpos=np.zeros((frames, 22), np.float32),
        left_hand_qpos_names=np.asarray([f"L{i}" for i in range(22)]),
        right_hand_qpos_names=np.asarray([f"R{i}" for i in range(22)]),
        left_valid=np.ones(frames, bool),
        right_valid=np.ones(frames, bool),
    )
    identity = np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1))
    np.savez_compressed(
        clip_dir / "gvhmr_camera.npz",
        camera_pos_world=np.zeros((frames, 3), np.float32),
        camera_target_world=np.zeros((frames, 3), np.float32),
        camera_pos_isaac=np.zeros((frames, 3), np.float32),
        camera_target_isaac=np.zeros((frames, 3), np.float32),
        subject_world=np.zeros((frames, 3), np.float32),
        subject_isaac=np.zeros((frames, 3), np.float32),
        T_w2c=identity,
        K_fullimg=np.tile(np.eye(3, dtype=np.float32), (frames, 1, 1)),
        world_to_isaac=np.eye(3, dtype=np.float32),
        alignment_offset_world=np.zeros(3, np.float32),
        gravity_axis=np.asarray("neg_y"),
    )
    (clip_dir / "composite_2x2.mp4").write_bytes(b"preview")

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
  <worldbody>
    <body name="root">
      <freejoint/>
      <body name="a"><joint name="joint_a"/>
        <body name="b"><joint name="joint_b"/></body>
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip(),
        encoding="utf-8",
    )
    config = {
        "schema_version": 1,
        "output": {"root": "work"},
        "human": {"backend": "test"},
        "locomotion": {"enabled": True},
        "phc": {"enabled": True},
        "gmr": {"source": "smoothed", "hand_model": "sharpa"},
        "object": {"enabled": False},
        "product": {
            "enabled": True,
            "root": "products",
            "include_camera": True,
            "include_preview": True,
            "require_preview": True,
            "object_policy": "exclude",
            "retention": {
                "prune_workspace_after_export": False,
                "prune_object_work_after_export": False,
            },
        },
    }
    return project, config, clip


def _add_object(project: Path, config: dict, clip: str) -> None:
    clip_dir = project / "work" / clip
    recon = clip_dir / "object_reconstruction"
    mesh_dir = recon / "monocular_adapter" / "mesh"
    mesh_dir.mkdir(parents=True)
    mesh = mesh_dir / "mesh.obj"
    mesh.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        encoding="utf-8",
    )
    frames = 4
    pose = np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1))
    np.savez(
        recon / "object_motion_gvhmr.npz",
        coordinate_system=np.asarray("opencv_camera"),
        fps=np.float32(30.0),
        pose_camera_object=pose,
        pose_gvhmr_world_object=pose,
        valid=np.ones(frames, bool),
        K=np.eye(3, dtype=np.float32),
    )
    np.savez(
        recon / "object_motion_gmr.npz",
        coordinate_system=np.asarray("mujoco_robot_world_zup"),
        fps=np.float32(30.0),
        position=np.zeros((frames, 3), np.float32),
        quat_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1)),
        valid=np.ones(frames, bool),
        source_frame_index=np.arange(frames, dtype=np.int32),
        visual_mesh_path=np.asarray(str(mesh.resolve())),
        collision_mesh_paths=np.asarray([], dtype=str),
    )
    (recon / "object_adapter_validation.json").write_text(
        json.dumps(
            {
                "ok": True,
                "errors": [],
                "adapter_dir": str((recon / "monocular_adapter").resolve()),
                "valid_ratio": 1.0,
                "observed_ratio": 0.75,
                "mesh_extents_m": [1.0, 0.5, 0.2],
            }
        ),
        encoding="utf-8",
    )
    config["object"] = {
        "enabled": True,
        "quality": {"min_valid_ratio": 0.5, "min_observed_ratio": 0.5},
        "clips": {clip: {"expected_size_m": [0.1, 1.2]}},
    }
    config["product"]["object_policy"] = "require_valid"


def test_export_is_compact_pickle_free_and_self_describing(tmp_path):
    project, config, clip = _fixture(tmp_path)
    result = EXPORTER.export_clip(project, config, clip)
    bundle = Path(result["bundle"])

    assert (bundle / "motion.npz").is_file()
    assert not (bundle / "human_motion.npz").exists()
    assert not (bundle / "robot_motion.npz").exists()
    assert not list(bundle.rglob("*.pkl"))
    assert not list(bundle.rglob("*.pt"))
    assert EXPORTER.verify_bundle(bundle)["status"] == "passed"

    for path in bundle.rglob("*.npz"):
        with np.load(path, allow_pickle=False) as archive:
            assert all(archive[key].dtype != object for key in archive.files)
            for key in archive.files:
                if np.issubdtype(archive[key].dtype, np.number):
                    assert np.isfinite(archive[key]).all()

    with np.load(bundle / "motion.npz", allow_pickle=False) as motion:
        assert motion["robot__root_quaternion_order"].item() == "xyzw"
        assert motion["robot__coordinate_system"].item() == "mujoco_world_z_up"
        assert motion["robot__dof_names"].tolist() == ["joint_a", "joint_b"]
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert "rights" not in manifest
    assert not (bundle / "rights.json").exists()
    assert manifest["validation"]["status"] == "passed"
    for path in bundle.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".yaml"}:
            assert str(project) not in path.read_text(encoding="utf-8")


def test_prune_requires_verified_external_bundle(tmp_path):
    project, config, clip = _fixture(tmp_path)
    result = EXPORTER.export_clip(project, config, clip)
    bundle = Path(result["bundle"])
    config["product"]["retention"]["prune_workspace_after_export"] = True

    prune = EXPORTER.prune_exported_workspace(project, config, clip, bundle)

    assert not (project / "work" / clip).exists()
    assert bundle.is_dir()
    assert Path(prune["marker"]).is_file()


def test_batch_export_writes_relative_catalog(tmp_path):
    project, config, clip = _fixture(tmp_path)

    results = EXPORTER.export_from_config(project, config, [clip])

    assert len(results) == 1
    catalog = project / "products" / "catalog.jsonl"
    record = json.loads(catalog.read_text(encoding="utf-8"))
    assert record["clip"] == clip
    assert record["bundle"] == clip
    assert record["has_object"] is False
    assert "commercial_ready" not in record


def test_object_product_uses_relative_paths_and_enforces_expected_size(tmp_path):
    project, config, clip = _fixture(tmp_path)
    _add_object(project, config, clip)

    result = EXPORTER.export_clip(project, config, clip)
    bundle = Path(result["bundle"])
    with np.load(bundle / "motion.npz", allow_pickle=False) as motion:
        assert motion["object__visual_mesh_path"].item() == "object/mesh/mesh.obj"
        assert motion["object__robot_coordinate_system"].item() == (
            "mujoco_robot_world_z_up"
        )
        assert motion["object__robot_quat_wxyz"].shape == (4, 4)

    config["object"]["clips"][clip]["expected_size_m"] = [0.1, 0.9]
    try:
        EXPORTER.export_clip(project, config, clip)
    except EXPORTER.ProductExportError as exc:
        assert "outside expected_size_m" in str(exc)
    else:
        raise AssertionError("oversized object mesh should fail product export")


def test_automated_quality_report_gates_product_export(tmp_path):
    project, config, clip = _fixture(tmp_path)
    clip_dir = project / "work" / clip
    config["quality_evaluation"] = {"require_for_product": True}
    config["product"]["minimum_quality_status"] = "warn"
    report_path = clip_dir / "quality_report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "clip": clip,
                "mode": "human_only",
                "status": "fail",
                "overall_score": 72.5,
                "critical_fail_reasons": ["missing_required_file=robot_motion.pkl"],
                "warn_reasons": [],
            }
        ),
        encoding="utf-8",
    )

    try:
        EXPORTER.export_clip(project, config, clip)
    except EXPORTER.ProductExportError as exc:
        assert "quality status 'fail'" in str(exc)
    else:
        raise AssertionError("failed automated quality must reject product export")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["status"] = "warn"
    report["critical_fail_reasons"] = []
    report["warn_reasons"] = ["right_hand_reproj_error_relative_p90=0.4200"]
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = EXPORTER.export_clip(project, config, clip)
    bundle = Path(result["bundle"])
    bundled_report = json.loads(
        (bundle / "quality_report.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert bundled_report["status"] == "warn"
    assert manifest["quality"]["automated"]["status"] == "warn"


def test_batch_quality_preflight_skips_failed_clips(tmp_path):
    project, config, good_clip = _fixture(tmp_path)
    bad_clip = "bad_sample"
    good_dir = project / "work" / good_clip
    bad_dir = project / "work" / bad_clip
    shutil.copytree(good_dir, bad_dir)
    config["quality_evaluation"] = {"require_for_product": True}
    config["product"]["minimum_quality_status"] = "warn"
    (good_dir / "quality_report.json").write_text(
        json.dumps(
            {"schema_version": 3, "clip": good_clip, "status": "warn"}
        ),
        encoding="utf-8",
    )
    (bad_dir / "quality_report.json").write_text(
        json.dumps(
            {"schema_version": 3, "clip": bad_clip, "status": "fail"}
        ),
        encoding="utf-8",
    )

    results = EXPORTER.export_from_config(project, config, [good_clip, bad_clip])

    assert len(results) == 1
    assert results[0]["clip"] == good_clip
    assert Path(results[0]["bundle"]).is_dir()
    assert not (project / "products" / bad_clip).exists()
    assert not list((project / "products").glob(".*.export-*"))
    catalog = project / "products" / "catalog.jsonl"
    record = json.loads(catalog.read_text(encoding="utf-8"))
    assert record["clip"] == good_clip


def test_product_rejects_stale_quality_schema_when_required(tmp_path):
    project, config, clip = _fixture(tmp_path)
    clip_dir = project / "work" / clip
    config["quality_evaluation"] = {"require_for_product": True}
    config["product"]["minimum_quality_schema_version"] = 2
    (clip_dir / "quality_report.json").write_text(
        json.dumps({"schema_version": 1, "clip": clip, "status": "warn"}),
        encoding="utf-8",
    )

    try:
        EXPORTER.export_clip(project, config, clip)
    except EXPORTER.ProductExportError as exc:
        assert "schema_version=1" in str(exc)
        assert "rerun the quality stage" in str(exc)
    else:
        raise AssertionError("stale quality reports must be rejected")
