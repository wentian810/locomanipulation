#!/usr/bin/env python3
"""Metrics that describe hand observation support separately from robot IK.

These helpers intentionally do *not* infer a visually correct palm direction.
For monocular, occluded hands that is not identifiable from temporal smoothness
alone.  They instead report how much of a track is backed by a direct source
observation, plus limited continuity diagnostics that can route a clip to
manual review.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


HandArrays = Mapping[str, np.ndarray]


def _finite_percentile(values: np.ndarray, percentile: float) -> float | None:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if not array.size:
        return None
    return float(np.percentile(array, percentile))


def _frame_vector(
    arrays: HandArrays,
    key: str,
    frame_count: int,
    default: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Load a per-frame vector, falling back safely when old sidecars lack it."""

    if key not in arrays:
        return np.asarray(default).copy(), False
    value = np.asarray(arrays[key]).reshape(-1)
    if value.shape[0] != frame_count:
        return np.asarray(default).copy(), False
    return value, True


def _rotation_matrices_from_axis_angle(rotvec: np.ndarray) -> np.ndarray | None:
    """Convert frame-wise axis-angle values to matrices without SciPy."""

    values = np.asarray(rotvec, dtype=np.float64)
    if values.ndim == 3 and values.shape[-2:] == (3, 3):
        return values
    if values.ndim != 2 or values.shape[1] != 3:
        return None

    theta = np.linalg.norm(values, axis=1)
    axis = np.zeros_like(values)
    nonzero = theta > 1e-12
    axis[nonzero] = values[nonzero] / theta[nonzero, None]
    skew = np.zeros((values.shape[0], 3, 3), dtype=np.float64)
    skew[:, 0, 1] = -axis[:, 2]
    skew[:, 0, 2] = axis[:, 1]
    skew[:, 1, 0] = axis[:, 2]
    skew[:, 1, 2] = -axis[:, 0]
    skew[:, 2, 0] = -axis[:, 1]
    skew[:, 2, 1] = axis[:, 0]
    identity = np.broadcast_to(np.eye(3, dtype=np.float64), skew.shape)
    return (
        identity
        + np.sin(theta)[:, None, None] * skew
        + (1.0 - np.cos(theta))[:, None, None] * (skew @ skew)
    )


def _rotation_step_degrees(rotations: np.ndarray) -> np.ndarray:
    if rotations.shape[0] < 2:
        return np.empty(0, dtype=np.float64)
    relative = rotations[1:] @ np.swapaxes(rotations[:-1], -1, -2)
    trace = np.trace(relative, axis1=-2, axis2=-1)
    cosine = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def hand_observation_metrics(arrays: HandArrays, side: str) -> dict[str, Any]:
    """Return direct-observation coverage and reprojection proxy statistics.

    ``*_hand_valid`` describes whether the final track is usable after temporal
    filling.  ``*_hand_source_reliable`` instead identifies frames supported by
    a reliable source observation.  The distinction is central for not treating
    a long interpolation as visual evidence.
    """

    valid_key = f"{side}_hand_valid"
    if valid_key not in arrays:
        raise KeyError(f"missing required hand validity field: {valid_key}")
    valid = np.asarray(arrays[valid_key], dtype=bool).reshape(-1)
    frame_count = int(valid.shape[0])
    reliable, reliable_available = _frame_vector(
        arrays,
        f"{side}_hand_source_reliable",
        frame_count,
        valid,
    )
    reliable = np.asarray(reliable, dtype=bool)
    repaired, repaired_available = _frame_vector(
        arrays,
        f"{side}_hand_source_repaired",
        frame_count,
        np.zeros(frame_count, dtype=bool),
    )
    repaired = np.asarray(repaired, dtype=bool)

    reproj_key = f"{side}_hand_reproj_error_relative"
    reproj, reproj_available = _frame_vector(
        arrays,
        reproj_key,
        frame_count,
        np.full(frame_count, np.nan, dtype=np.float64),
    )
    reproj = np.asarray(reproj, dtype=np.float64)
    observed_reproj = reproj[reliable & np.isfinite(reproj)]

    return {
        "frame_count": frame_count,
        "source_reliable_ratio": float(np.mean(reliable)) if frame_count else 0.0,
        "source_repaired_ratio": float(np.mean(repaired)) if frame_count else 0.0,
        "source_reliable_frames": int(np.count_nonzero(reliable)),
        "source_repaired_frames": int(np.count_nonzero(repaired)),
        "observed_reproj_frame_count": int(observed_reproj.size),
        "observed_reproj_error_relative_p50": _finite_percentile(observed_reproj, 50),
        "observed_reproj_error_relative_p90": _finite_percentile(observed_reproj, 90),
        "observed_reproj_error_relative_p95": _finite_percentile(observed_reproj, 95),
        "source_reliability_available": reliable_available,
        "source_repair_available": repaired_available,
        "reprojection_available": reproj_available,
    }


def hand_orientation_continuity_metrics(
    arrays: HandArrays,
    side: str,
) -> dict[str, Any]:
    """Return temporal orientation diagnostics, never a palm-facing verdict.

    A high palm-normal dot product only establishes internal continuity.  It
    cannot validate whether the palm/front-back branch agrees with the image,
    especially during occlusion.  That caveat is carried into the caller's
    metric details.
    """

    valid_key = f"{side}_hand_valid"
    if valid_key not in arrays:
        raise KeyError(f"missing required hand validity field: {valid_key}")
    valid = np.asarray(arrays[valid_key], dtype=bool).reshape(-1)
    frame_count = int(valid.shape[0])
    reliable, _ = _frame_vector(
        arrays,
        f"{side}_hand_source_reliable",
        frame_count,
        valid,
    )
    reliable = np.asarray(reliable, dtype=bool)

    normals_key = f"{side}_hand_palm_normal"
    normals_available = normals_key in arrays
    normals = np.asarray(
        arrays[normals_key]
        if normals_available
        else np.full((frame_count, 3), np.nan, dtype=np.float64),
        dtype=np.float64,
    )
    if normals.ndim != 2 or normals.shape != (frame_count, 3):
        normals = np.full((frame_count, 3), np.nan, dtype=np.float64)
        normals_available = False
    frame_valid, frame_valid_available = _frame_vector(
        arrays,
        f"{side}_hand_wrist_frame_valid",
        frame_count,
        np.ones(frame_count, dtype=bool),
    )
    frame_valid = np.asarray(frame_valid, dtype=bool)
    norm = np.linalg.norm(normals, axis=1)
    normal_valid = (
        reliable
        & frame_valid
        & np.isfinite(norm)
        & (norm > 1e-8)
    )
    unit = np.zeros_like(normals)
    unit[normal_valid] = normals[normal_valid] / norm[normal_valid, None]
    adjacent = normal_valid[1:] & normal_valid[:-1]
    dots = np.sum(unit[1:] * unit[:-1], axis=1)[adjacent]

    orientation_key = f"{side}_hand_global_orient"
    orientation_available = orientation_key in arrays
    orientation = np.asarray(
        arrays[orientation_key]
        if orientation_available
        else np.full((frame_count, 3), np.nan, dtype=np.float64),
        dtype=np.float64,
    )
    matrices = _rotation_matrices_from_axis_angle(orientation)
    if matrices is None:
        step = np.empty(0, dtype=np.float64)
        orientation_available = False
    else:
        pair_mask = reliable[1:] & reliable[:-1]
        step = _rotation_step_degrees(matrices)[pair_mask]

    return {
        "palm_normal_temporal_pair_count": int(dots.size),
        "palm_normal_temporal_dot_p05": _finite_percentile(dots, 5),
        "palm_normal_temporal_flip_ratio": (
            float(np.mean(dots < 0.0)) if dots.size else None
        ),
        "wrist_angular_step_deg_p95": _finite_percentile(step, 95),
        "wrist_angular_step_deg_max": _finite_percentile(step, 100),
        "palm_normal_available": normals_available and frame_valid_available,
        "global_orientation_available": orientation_available,
    }


def _json_nonnegative_integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)


def _json_stat(value: Any) -> dict[str, float | int] | None:
    """Validate the scalar summary written by the visible-hand refiner."""
    if not isinstance(value, Mapping):
        return None
    count = _json_nonnegative_integer(value.get("count"))
    if count is None or count <= 0:
        return None
    result: dict[str, float | int] = {"count": count}
    for key in ("mean", "p50", "p90", "p95"):
        raw = value.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None
        number = float(raw)
        if not math.isfinite(number):
            return None
        result[key] = number
    return result


def _contiguous_coverage(value: Any, frame_count: int | None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {
            "frames": None,
            "longest_run": None,
            "ratio": None,
        }
    frames = _json_nonnegative_integer(value.get("frames"))
    longest_run = _json_nonnegative_integer(value.get("longest_run"))
    return {
        "frames": frames,
        "longest_run": longest_run,
        "ratio": (
            float(frames / frame_count)
            if frames is not None and frame_count is not None and frame_count > 0
            else None
        ),
    }


def visible_hand_refinement_metrics(
    summary: Mapping[str, Any],
    frame_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Normalize evidence-gated wrist-refinement diagnostics from JSON only.

    The refiner's accepted set is a strict subset of eligible frames.  A
    before/after reprojection comparison is therefore exposed *only* when the
    summary includes the applied-set baseline emitted by newer refiners.  This
    prevents a legacy report from incorrectly comparing eligible-before with
    applied-after.

    This helper does not read tensor files and does not infer palm/back-facing
    correctness from 2D observations.
    """
    if not isinstance(summary, Mapping):
        raise ValueError("visible refinement summary must be a mapping")
    sides = summary.get("sides")
    if not isinstance(sides, Mapping):
        raise ValueError("visible refinement summary is missing sides")

    result: dict[str, Any] = {
        "schema_version": summary.get("schema_version"),
        "purpose": summary.get("purpose"),
        "palm_back_semantics": (
            summary.get("limits", {}).get("palm_back_semantics")
            if isinstance(summary.get("limits"), Mapping)
            else None
        ),
        "sides": {},
        "same_population_comparison_available": False,
        "same_population_holdout_comparison_available": False,
    }
    for side in ("left", "right"):
        side_summary = sides.get(side)
        if not isinstance(side_summary, Mapping):
            result["sides"][side] = {
                "comparison_status": "skipped",
                "comparison_reason": "side_summary_missing",
            }
            continue

        frame_count = _json_nonnegative_integer(
            frame_counts.get(side, frame_counts.get("hands"))
        )
        visible = _contiguous_coverage(side_summary.get("visible_evidence"), frame_count)
        eligible = _contiguous_coverage(side_summary.get("refine_eligible"), frame_count)
        fit = _contiguous_coverage(side_summary.get("fit_frames"), frame_count)
        holdout_fit = _contiguous_coverage(
            side_summary.get("holdout_fit_frames"), frame_count
        )
        applied = _contiguous_coverage(side_summary.get("applied"), frame_count)
        applied_frames = applied["frames"]
        eligible_frames = eligible["frames"]
        evidence_available = side_summary.get("evidence_available") is True

        absolute_before = _json_stat(
            side_summary.get("absolute_reprojection_before_applied_px")
        )
        absolute_after = _json_stat(
            side_summary.get("absolute_reprojection_after_px")
        )
        relative_before = _json_stat(
            side_summary.get("relative_reprojection_before_applied_px")
        )
        relative_after = _json_stat(
            side_summary.get("relative_reprojection_after_px")
        )
        delta_degrees = _json_stat(side_summary.get("accepted_delta_degrees"))
        fit_partition = side_summary.get("fit_partition")
        if not isinstance(fit_partition, str) or fit_partition not in {
            "all",
            "non_tip",
        }:
            fit_partition = None
        holdout_before = _json_stat(
            side_summary.get("holdout_relative_reprojection_before_applied_px")
        )
        holdout_after = _json_stat(
            side_summary.get("holdout_relative_reprojection_after_px")
        )

        population_matches = (
            evidence_available
            and isinstance(applied_frames, int)
            and applied_frames > 0
            and all(
                stats is not None and stats["count"] == applied_frames
                for stats in (
                    absolute_before,
                    absolute_after,
                    relative_before,
                    relative_after,
                    delta_degrees,
                )
            )
        )
        if population_matches:
            comparison_status = "available"
            comparison_reason = None
            result["same_population_comparison_available"] = True
        elif not evidence_available:
            comparison_status = "skipped"
            comparison_reason = "evidence_unavailable"
        elif not isinstance(applied_frames, int) or applied_frames <= 0:
            comparison_status = "skipped"
            comparison_reason = "no_applied_frames"
        else:
            comparison_status = "skipped"
            comparison_reason = "same_population_baseline_missing_or_invalid"

        holdout_population_matches = (
            fit_partition == "non_tip"
            and isinstance(applied_frames, int)
            and applied_frames > 0
            and holdout_before is not None
            and holdout_after is not None
            and holdout_before["count"] == applied_frames
            and holdout_after["count"] == applied_frames
        )
        if holdout_population_matches:
            holdout_comparison_status = "available"
            holdout_comparison_reason = None
            result["same_population_holdout_comparison_available"] = True
        elif fit_partition == "all":
            holdout_comparison_status = "not_requested"
            holdout_comparison_reason = None
        elif fit_partition != "non_tip":
            holdout_comparison_status = "skipped"
            holdout_comparison_reason = "holdout_partition_unknown_or_legacy"
        elif not isinstance(applied_frames, int) or applied_frames <= 0:
            holdout_comparison_status = "skipped"
            holdout_comparison_reason = "no_applied_frames"
        else:
            holdout_comparison_status = "skipped"
            holdout_comparison_reason = "same_population_holdout_baseline_missing_or_invalid"

        rejection_counts = side_summary.get("rejection_counts")
        if not isinstance(rejection_counts, Mapping):
            rejection_counts = {}
        else:
            rejection_counts = {
                str(key): count
                for key, count in rejection_counts.items()
                if _json_nonnegative_integer(count) is not None
            }
        result["sides"][side] = {
            "frame_count": frame_count,
            "evidence_available": evidence_available,
            "visible_evidence": visible,
            "refine_eligible": eligible,
            "fit_frames": fit,
            "fit_partition": fit_partition,
            "holdout_fit_frames": holdout_fit,
            "applied": applied,
            "applied_given_eligible_ratio": (
                float(applied_frames / eligible_frames)
                if isinstance(applied_frames, int)
                and isinstance(eligible_frames, int)
                and eligible_frames > 0
                else None
            ),
            "comparison_status": comparison_status,
            "comparison_reason": comparison_reason,
            "absolute_reprojection_before_applied_px": (
                absolute_before if population_matches else None
            ),
            "absolute_reprojection_after_px": (
                absolute_after if population_matches else None
            ),
            "relative_reprojection_before_applied_px": (
                relative_before if population_matches else None
            ),
            "relative_reprojection_after_px": (
                relative_after if population_matches else None
            ),
            "holdout_comparison_status": holdout_comparison_status,
            "holdout_comparison_reason": holdout_comparison_reason,
            "holdout_relative_reprojection_before_applied_px": (
                holdout_before if holdout_population_matches else None
            ),
            "holdout_relative_reprojection_after_px": (
                holdout_after if holdout_population_matches else None
            ),
            "accepted_delta_degrees": delta_degrees if population_matches else None,
            "rejection_counts": rejection_counts,
            "missing_evidence_fields": list(
                side_summary.get("missing_evidence_fields", [])
            )
            if isinstance(side_summary.get("missing_evidence_fields"), list)
            else [],
        }
    return result
