#!/usr/bin/env python3
"""Opt-in final-render-space wrist refinement for visibly observed hands.

This is intentionally conservative.  It only changes a small wrist rotation
when final SMPL-X camera-space 2D evidence is both strong and auditable.  It
does not infer palm-facing truth from 2D keypoints, and it never proposes a
180-degree palm flip.

Run this after temporal/finger filters and before direct-MANO joint
recomputation.  The report contains the final-render-space metrics used for
acceptance; legacy H4W scalar reprojection values are not reused.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

try:
    from tools.processor.visible_hand_evidence import VisibleEvidenceConfig, build_visible_evidence
except ImportError:  # Direct execution from this directory.
    from visible_hand_evidence import VisibleEvidenceConfig, build_visible_evidence


# COCO-WholeBody/MANO order: Wrist, Thumb1..4, Index1..4, Middle1..4,
# Ring1..4, Pinky1..4.  SMPL-X fingertips are vertex landmarks.
HAND21 = {
    "left": {
        "joints": [20, 37, 38, 39, None, 25, 26, 27, None, 28, 29, 30, None, 34, 35, 36, None, 31, 32, 33, None],
        "verts": [None, None, None, None, 5361, None, None, None, 4933, None, None, None, 5058, None, None, None, 5169, None, None, None, 5286],
        "parent_chain": [2, 5, 8, 12, 15, 17],
        "wrist_idx": 19,
        "vitpose_slice": slice(-42, -21),
    },
    "right": {
        "joints": [21, 52, 53, 54, None, 40, 41, 42, None, 43, 44, 45, None, 49, 50, 51, None, 46, 47, 48, None],
        "verts": [None, None, None, None, 8079, None, None, None, 7669, None, None, None, 7794, None, None, None, 7905, None, None, None, 8022],
        "parent_chain": [2, 5, 8, 13, 16, 18],
        "wrist_idx": 20,
        "vitpose_slice": slice(-21, None),
    },
}

# COCO-WholeBody/MANO tips: Thumb4, Index4, Middle4, Ring4, Pinky4.
# A non-tip fit can use the remaining joints while treating the fingertips as
# an image-space holdout diagnostic. This is not an independent label source,
# but it avoids evaluating solely on the exact landmarks used for fitting.
FINGERTIP_INDICES = (4, 8, 12, 16, 20)

REJECT_NOT_ELIGIBLE = 1 << 0
REJECT_FIT_KEYPOINTS = 1 << 1
REJECT_BODY_ANCHOR = 1 << 2
REJECT_NO_IMPROVEMENT = 1 << 3
REJECT_ABSOLUTE_REGRESSION = 1 << 4
REJECT_DELTA_LIMIT = 1 << 5
REJECT_NUMERICAL = 1 << 6
REJECT_HOLDOUT_KEYPOINTS = 1 << 7
REJECT_HOLDOUT_REGRESSION = 1 << 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mano_params", type=Path, required=True)
    parser.add_argument("--hmr4d_results", type=Path, required=True)
    parser.add_argument("--vitpose_wholebody", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--prior_weight", type=float, default=0.02)
    parser.add_argument("--fit_confidence", type=float, default=0.60)
    parser.add_argument("--fit_min_keypoints", type=int, default=12)
    parser.add_argument(
        "--fit_partition",
        choices=("all", "non_tip"),
        default="all",
        help=(
            "all: fit/evaluate all high-confidence hand points; non_tip: fit "
            "only wrist and non-tip joints, then hold out fingertips for acceptance"
        ),
    )
    parser.add_argument("--holdout_min_keypoints", type=int, default=3)
    parser.add_argument(
        "--holdout_max_relative_regression_px",
        type=float,
        default=1.0,
        help="non_tip mode: maximum allowed fingertip relative-error regression",
    )
    parser.add_argument("--max_delta_degrees", type=float, default=25.0)
    parser.add_argument("--min_relative_improvement", type=float, default=0.25)
    parser.add_argument("--min_relative_improvement_px", type=float, default=2.0)
    parser.add_argument("--max_absolute_regression_px", type=float, default=2.0)
    parser.add_argument("--max_anchor_error_px", type=float, default=20.0)
    parser.add_argument("--max_anchor_error_bbox_ratio", type=float, default=0.25)
    parser.add_argument("--evidence_hand_confidence", type=float, default=0.45)
    parser.add_argument("--evidence_min_keypoints", type=int, default=8)
    parser.add_argument("--evidence_mean_confidence", type=float, default=0.50)
    parser.add_argument("--evidence_wrist_confidence", type=float, default=0.45)
    parser.add_argument("--evidence_min_bbox_diagonal_px", type=float, default=96.0)
    parser.add_argument("--evidence_min_run", type=int, default=3)
    return parser.parse_args()


def load_torch(path: Path, *, weights_only: bool) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=weights_only)
    except TypeError:  # Older torch versions.
        return torch.load(path, map_location="cpu")


def to_numpy(value: Any, dtype=np.float32) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_track_shape(value: Any, people: int, frames: int, trailing: tuple[int, ...], name: str) -> torch.Tensor:
    tensor = value.detach().cpu() if torch.is_tensor(value) else torch.as_tensor(value)
    expected = (people, frames, *trailing)
    if tuple(tensor.shape) == expected:
        return tensor
    if people == 1 and tuple(tensor.shape) == expected[1:]:
        return tensor.unsqueeze(0)
    raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {expected}")


def smplx_hand21(smplx_output, side: str) -> torch.Tensor:
    spec = HAND21[side]
    return torch.stack(
        [
            smplx_output.joints[:, joint] if joint is not None else smplx_output.vertices[:, vertex]
            for joint, vertex in zip(spec["joints"], spec["verts"])
        ],
        dim=1,
    )


def project_full_image(points_cam: torch.Tensor, camera: torch.Tensor) -> torch.Tensor:
    """Perspective project without clipping, so off-image errors stay visible."""

    uvw = torch.einsum("bij,bnj->bni", camera, points_cam)
    z = uvw[..., 2:3]
    z_safe = torch.where(
        z.abs() < 1e-6,
        torch.where(z >= 0.0, torch.full_like(z, 1e-6), torch.full_like(z, -1e-6)),
        z,
    )
    return uvw[..., :2] / z_safe


def weighted_errors(pred_uv: torch.Tensor, target_xyc: torch.Tensor, fit_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return absolute, wrist-relative, and wrist-anchor errors in pixels."""

    weights = target_xyc[..., 2].clamp(0.0, 1.0) * fit_mask.float()
    absolute_dist = torch.linalg.norm(pred_uv - target_xyc[..., :2], dim=-1)
    absolute = (absolute_dist * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)

    relative_pred = pred_uv[:, 1:] - pred_uv[:, :1]
    relative_target = target_xyc[:, 1:, :2] - target_xyc[:, :1, :2]
    relative_weights = weights[:, 1:]
    relative_dist = torch.linalg.norm(relative_pred - relative_target, dim=-1)
    relative = (relative_dist * relative_weights).sum(dim=-1) / relative_weights.sum(dim=-1).clamp_min(1.0)
    anchor = torch.linalg.norm(pred_uv[:, 0] - target_xyc[:, 0, :2], dim=-1)
    return absolute, relative, anchor


def partition_fit_points(
    high_confidence: torch.Tensor,
    partition: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fit and holdout masks for a (B,21) confidence mask.

    ``all`` reproduces the production behavior exactly. ``non_tip`` leaves the
    five fingertips out of the optimizer and reserves them for a conservative
    non-regression check. The shared detector still makes this an internal
    consistency check, not a ground-truth benchmark.
    """
    if high_confidence.ndim != 2 or high_confidence.shape[1] != 21:
        raise ValueError(
            "high_confidence must have shape (batch, 21), got "
            f"{tuple(high_confidence.shape)}"
        )
    if partition == "all":
        return high_confidence, torch.zeros_like(high_confidence)
    if partition != "non_tip":
        raise ValueError(f"unsupported fit partition: {partition}")
    tip_mask = torch.zeros_like(high_confidence)
    tip_mask[:, list(FINGERTIP_INDICES)] = True
    holdout = high_confidence & tip_mask
    return high_confidence & ~tip_mask, holdout


def active_global_keys(mano: Mapping[str, Any], side: str) -> tuple[str, list[str]]:
    base = f"{side}_hand_global_orient"
    filtered = f"{base}_filtered"
    if base not in mano:
        raise KeyError(f"Missing required MANO global orientation: {base}")
    # Render uses *_filtered when available, while direct-MANO recomputation
    # reads the base key. Update both so those two consumers cannot diverge.
    return (filtered if filtered in mano else base), [base, filtered] if filtered in mano else [base]


def build_subsequence(
    hmr_incam: Mapping[str, Any],
    mano: Mapping[str, Any],
    person_idx: int,
    frame_ids: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    hmr_sub = {}
    for key, value in hmr_incam.items():
        if not torch.is_tensor(value) or value.ndim < 2:
            continue
        hmr_sub[key] = value[person_idx : person_idx + 1, frame_ids].detach().cpu()
    hand_keys = []
    for side in ("left", "right"):
        hand_keys.extend(
            [
                f"{side}_hand_pose",
                f"{side}_hand_betas",
                f"{side}_hand_valid",
                f"{side}_hand_global_orient",
                f"{side}_hand_global_orient_filtered",
            ]
        )
    mano_sub = {}
    for key in hand_keys:
        value = mano.get(key)
        if value is None or not torch.is_tensor(value) or value.ndim < 2:
            continue
        mano_sub[key] = value[person_idx : person_idx + 1, frame_ids].detach().cpu()
    return hmr_sub, mano_sub


def build_render_params(smplx, hmr_sub: Mapping[str, Any], mano_sub: Mapping[str, Any], frames: int, device: torch.device) -> dict[str, torch.Tensor]:
    # Import lazily: this script is a pipeline module but unit tests for the
    # evidence gate do not need the large HMR/SMPL-X dependency graph.
    from tools.processor.generate_smplxs import fetch_smpl_params_with_hands

    params = fetch_smpl_params_with_hands(hmr_sub, 0, mano_sub, frames, smplx)
    result = {}
    for key, value in params.items():
        result[key] = value.to(device) if torch.is_tensor(value) else value
    result["global_orient"] = result["global_orient"].reshape(frames, 3)
    result["body_pose"] = result["body_pose"].reshape(frames, 21, 3)
    return result


def forward_with_wrist_delta(smplx, params0: Mapping[str, torch.Tensor], side: str, delta_axis_angle: torch.Tensor):
    from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

    spec = HAND21[side]
    batch = delta_axis_angle.shape[0]
    body_aa0 = params0["body_pose"].reshape(batch, 21, 3)
    body_rot0 = axis_angle_to_matrix(body_aa0.reshape(-1, 3)).reshape(batch, 21, 3, 3)
    root_rot = axis_angle_to_matrix(params0["global_orient"].reshape(batch, 3))
    parent_rot = root_rot
    for joint_idx in spec["parent_chain"]:
        parent_rot = parent_rot @ body_rot0[:, joint_idx]
    old_global_wrist = parent_rot @ body_rot0[:, spec["wrist_idx"]]
    new_global_wrist = axis_angle_to_matrix(delta_axis_angle) @ old_global_wrist
    body_rot = body_rot0.clone()
    body_rot[:, spec["wrist_idx"]] = parent_rot.transpose(-1, -2) @ new_global_wrist
    body_axis = matrix_to_axis_angle(body_rot.reshape(-1, 3, 3)).reshape(batch, 21, 3)
    call = {key: value for key, value in params0.items() if key != "body_pose"}
    call["body_pose"] = body_axis.reshape(batch, -1)
    return smplx(**call), new_global_wrist


def max_rotation_delta_degrees(delta_axis_angle: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(delta_axis_angle, dim=-1) * (180.0 / math.pi)


def contiguous_stats(mask: np.ndarray) -> dict[str, int]:
    total = int(mask.sum())
    longest = 0
    for sequence in mask:
        current = 0
        for value in sequence:
            current = current + 1 if value else 0
            longest = max(longest, current)
    return {"frames": total, "longest_run": longest}


def tensor_stat(values: np.ndarray, mask: np.ndarray) -> dict[str, float | int | None]:
    selected = values[mask & np.isfinite(values)]
    if not selected.size:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "p95": None}
    return {
        "count": int(selected.size),
        "mean": float(np.mean(selected)),
        "p50": float(np.percentile(selected, 50)),
        "p90": float(np.percentile(selected, 90)),
        "p95": float(np.percentile(selected, 95)),
    }


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def run_side(
    side: str,
    mano: dict[str, Any],
    hmr_incam: Mapping[str, Any],
    camera_all: torch.Tensor,
    vitpose: torch.Tensor,
    smplx,
    device: torch.device,
    args: argparse.Namespace,
    evidence_config: VisibleEvidenceConfig,
) -> dict[str, Any]:
    people, frames = to_numpy(mano[f"{side}_hand_valid"], dtype=np.float32).shape[:2]
    hand_slice = HAND21[side]["vitpose_slice"]
    hand_kpts = ensure_track_shape(vitpose[:, :, hand_slice, :], people, frames, (21, 3), "vitpose hand")
    body_kpts = ensure_track_shape(vitpose[:, :, :17, :], people, frames, (17, 3), "vitpose body")
    evidence = build_visible_evidence(mano, side, hand_kpts, body_kpts, evidence_config)
    visible = np.asarray(evidence["visible_evidence_mask"], dtype=bool)
    eligible = np.asarray(evidence["refine_eligible_mask"], dtype=bool)

    shape = (people, frames)
    applied = np.zeros(shape, dtype=bool)
    fit_mask_track = np.zeros(shape, dtype=bool)
    holdout_fit_mask_track = np.zeros(shape, dtype=bool)
    reject = np.full(shape, REJECT_NOT_ELIGIBLE, dtype=np.int32)
    reject[eligible] = 0
    abs_before = np.full(shape, np.nan, dtype=np.float32)
    abs_after = np.full(shape, np.nan, dtype=np.float32)
    rel_before = np.full(shape, np.nan, dtype=np.float32)
    rel_after = np.full(shape, np.nan, dtype=np.float32)
    fit_rel_before = np.full(shape, np.nan, dtype=np.float32)
    fit_rel_after = np.full(shape, np.nan, dtype=np.float32)
    holdout_rel_before = np.full(shape, np.nan, dtype=np.float32)
    holdout_rel_after = np.full(shape, np.nan, dtype=np.float32)
    anchor_error = np.full(shape, np.nan, dtype=np.float32)
    delta_degrees = np.full(shape, np.nan, dtype=np.float32)

    if not evidence["available"]:
        reject[:] |= REJECT_NOT_ELIGIBLE
        return {
            "side": side,
            "evidence": evidence,
            "applied": applied,
            "fit_mask": fit_mask_track,
            "holdout_fit_mask": holdout_fit_mask_track,
            "reject": reject,
            "abs_before": abs_before,
            "abs_after": abs_after,
            "rel_before": rel_before,
            "rel_after": rel_after,
            "fit_rel_before": fit_rel_before,
            "fit_rel_after": fit_rel_after,
            "holdout_rel_before": holdout_rel_before,
            "holdout_rel_after": holdout_rel_after,
            "anchor_error": anchor_error,
            "delta_degrees": delta_degrees,
            "fit_partition": str(args.fit_partition),
        }

    active_key, write_keys = active_global_keys(mano, side)
    max_delta_rad = float(args.max_delta_degrees) * math.pi / 180.0
    for person_idx in range(people):
        frame_ids_np = np.flatnonzero(eligible[person_idx])
        for offset in range(0, len(frame_ids_np), max(1, int(args.batch_size))):
            frames_np = frame_ids_np[offset : offset + max(1, int(args.batch_size))]
            if not len(frames_np):
                continue
            frame_ids = torch.as_tensor(frames_np, dtype=torch.long)
            hmr_sub, mano_sub = build_subsequence(hmr_incam, mano, person_idx, frame_ids)
            try:
                params0 = build_render_params(smplx, hmr_sub, mano_sub, len(frames_np), device)
                target = hand_kpts[person_idx, frame_ids].to(device=device, dtype=torch.float32)
                camera = camera_all[frame_ids].to(device=device, dtype=torch.float32)
                all_points = target[..., 2] >= float(args.fit_confidence)
                fit_points, holdout_points = partition_fit_points(
                    all_points,
                    str(args.fit_partition),
                )
                fit_frames = fit_points.sum(dim=-1) >= int(args.fit_min_keypoints)
                fit_frames &= target[:, 0, 2] >= float(args.fit_confidence)
                if args.fit_partition == "non_tip":
                    holdout_frames = (
                        holdout_points.sum(dim=-1)
                        >= int(args.holdout_min_keypoints)
                    )
                    holdout_record = holdout_frames
                else:
                    holdout_frames = torch.ones_like(fit_frames)
                    holdout_record = torch.zeros_like(fit_frames)
                if not bool(fit_frames.any()):
                    reject[person_idx, frames_np] |= REJECT_FIT_KEYPOINTS
                    continue
                with torch.no_grad():
                    reference = smplx(
                        **{
                            **params0,
                            "body_pose": params0["body_pose"].reshape(
                                len(frames_np), -1
                            ),
                        }
                    )
                    zero_out, zero_global = forward_with_wrist_delta(
                        smplx, params0, side, torch.zeros((len(frames_np), 3), device=device)
                    )
                    if not torch.allclose(reference.vertices, zero_out.vertices, atol=1e-4, rtol=1e-4):
                        raise RuntimeError("zero-delta wrist path does not reproduce final renderer geometry")
                    base_uv = project_full_image(smplx_hand21(reference, side), camera)
                    base_abs, base_rel, base_anchor = weighted_errors(
                        base_uv, target, all_points
                    )
                    base_fit_abs, base_fit_rel, _ = weighted_errors(
                        base_uv, target, fit_points
                    )
                    if args.fit_partition == "non_tip":
                        _, base_holdout_rel, _ = weighted_errors(
                            base_uv, target, holdout_points
                        )
                    else:
                        base_holdout_rel = torch.full_like(base_rel, torch.nan)
                bbox_diag = torch.as_tensor(
                    evidence["hand_bbox_diagonal_px"][person_idx, frames_np], device=device, dtype=torch.float32
                )
                anchor_limit = torch.maximum(
                    torch.full_like(bbox_diag, float(args.max_anchor_error_px)),
                    bbox_diag * float(args.max_anchor_error_bbox_ratio),
                )
                anchor_ok = base_anchor <= anchor_limit
                optimize = (
                    fit_frames
                    & holdout_frames
                    & anchor_ok
                    & torch.isfinite(base_abs)
                    & torch.isfinite(base_rel)
                    & torch.isfinite(base_fit_abs)
                    & torch.isfinite(base_fit_rel)
                )
                if args.fit_partition == "non_tip":
                    optimize &= torch.isfinite(base_holdout_rel)
                fit_mask_track[person_idx, frames_np] = fit_frames.detach().cpu().numpy()
                holdout_fit_mask_track[person_idx, frames_np] = (
                    holdout_record.detach().cpu().numpy()
                )
                abs_before[person_idx, frames_np] = base_abs.detach().cpu().numpy()
                rel_before[person_idx, frames_np] = base_rel.detach().cpu().numpy()
                fit_rel_before[person_idx, frames_np] = base_fit_rel.detach().cpu().numpy()
                holdout_rel_before[person_idx, frames_np] = (
                    base_holdout_rel.detach().cpu().numpy()
                )
                anchor_error[person_idx, frames_np] = base_anchor.detach().cpu().numpy()
                reject[person_idx, frames_np[~fit_frames.detach().cpu().numpy()]] |= REJECT_FIT_KEYPOINTS
                if args.fit_partition == "non_tip":
                    reject[
                        person_idx,
                        frames_np[~holdout_frames.detach().cpu().numpy()],
                    ] |= REJECT_HOLDOUT_KEYPOINTS
                reject[person_idx, frames_np[~anchor_ok.detach().cpu().numpy()]] |= REJECT_BODY_ANCHOR
                if not bool(optimize.any()):
                    continue

                delta = torch.zeros((len(frames_np), 3), device=device, dtype=torch.float32, requires_grad=True)
                optimizer = torch.optim.Adam([delta], lr=float(args.lr))
                for _ in range(max(0, int(args.steps))):
                    optimizer.zero_grad(set_to_none=True)
                    output, _ = forward_with_wrist_delta(smplx, params0, side, delta)
                    uv = project_full_image(smplx_hand21(output, side), camera)
                    _, relative, _ = weighted_errors(uv, target, fit_points)
                    prior = (delta * delta).sum(dim=-1)
                    loss = (relative[optimize] + float(args.prior_weight) * prior[optimize]).mean()
                    if not torch.isfinite(loss):
                        break
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        norm = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1e-8)
                        delta.mul_((max_delta_rad / norm).clamp(max=1.0))
                with torch.no_grad():
                    output, new_global = forward_with_wrist_delta(smplx, params0, side, delta)
                    uv = project_full_image(smplx_hand21(output, side), camera)
                    new_abs, new_rel, _ = weighted_errors(uv, target, all_points)
                    new_fit_abs, new_fit_rel, _ = weighted_errors(
                        uv, target, fit_points
                    )
                    if args.fit_partition == "non_tip":
                        _, new_holdout_rel, _ = weighted_errors(
                            uv, target, holdout_points
                        )
                        holdout_ok = (
                            torch.isfinite(new_holdout_rel)
                            & (
                                new_holdout_rel
                                <= base_holdout_rel
                                + float(args.holdout_max_relative_regression_px)
                            )
                        )
                    else:
                        new_holdout_rel = torch.full_like(new_rel, torch.nan)
                        holdout_ok = torch.ones_like(optimize)
                    degrees = max_rotation_delta_degrees(delta)
                    improvement = base_fit_rel - new_fit_rel
                    required = torch.maximum(
                        base_fit_rel * float(args.min_relative_improvement),
                        torch.full_like(
                            base_fit_rel,
                            float(args.min_relative_improvement_px),
                        ),
                    )
                    accepted = (
                        optimize
                        & torch.isfinite(new_abs)
                        & torch.isfinite(new_rel)
                        & torch.isfinite(new_fit_abs)
                        & torch.isfinite(new_fit_rel)
                        & (improvement >= required)
                        & (new_abs <= base_abs + float(args.max_absolute_regression_px))
                        & (degrees <= float(args.max_delta_degrees) + 1e-4)
                        & holdout_ok
                    )
                    abs_after[person_idx, frames_np] = new_abs.detach().cpu().numpy()
                    rel_after[person_idx, frames_np] = new_rel.detach().cpu().numpy()
                    fit_rel_after[person_idx, frames_np] = (
                        new_fit_rel.detach().cpu().numpy()
                    )
                    holdout_rel_after[person_idx, frames_np] = (
                        new_holdout_rel.detach().cpu().numpy()
                    )
                    delta_degrees[person_idx, frames_np] = degrees.detach().cpu().numpy()
                    accepted_cpu = accepted.detach().cpu().numpy()
                    no_improvement = optimize & (improvement < required)
                    abs_regressed = optimize & (new_abs > base_abs + float(args.max_absolute_regression_px))
                    delta_limited = optimize & (degrees > float(args.max_delta_degrees) + 1e-4)
                    holdout_regressed = optimize & ~holdout_ok
                    reject[person_idx, frames_np[no_improvement.detach().cpu().numpy()]] |= REJECT_NO_IMPROVEMENT
                    reject[person_idx, frames_np[abs_regressed.detach().cpu().numpy()]] |= REJECT_ABSOLUTE_REGRESSION
                    reject[person_idx, frames_np[delta_limited.detach().cpu().numpy()]] |= REJECT_DELTA_LIMIT
                    if args.fit_partition == "non_tip":
                        reject[
                            person_idx,
                            frames_np[holdout_regressed.detach().cpu().numpy()],
                        ] |= REJECT_HOLDOUT_REGRESSION
                    if accepted_cpu.any():
                        values = new_global.detach().cpu()
                        accepted_index = accepted.detach().cpu()
                        selected_frame_ids = frame_ids[accepted_index]
                        for global_key in write_keys:
                            target_global = ensure_track_shape(mano[global_key], people, frames, (3, 3), global_key)
                            target_global[person_idx, selected_frame_ids] = values[accepted_index]
                            mano[global_key] = target_global
                        applied[person_idx, frames_np[accepted_cpu]] = True
            except torch.cuda.OutOfMemoryError:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    "visible-hand refiner ran out of GPU memory; rerun with a smaller --batch_size"
                )
            except Exception:
                reject[person_idx, frames_np] |= REJECT_NUMERICAL
                raise

    return {
        "side": side,
        "evidence": evidence,
        "applied": applied,
        "fit_mask": fit_mask_track,
        "holdout_fit_mask": holdout_fit_mask_track,
        "reject": reject,
        "abs_before": abs_before,
        "abs_after": abs_after,
        "rel_before": rel_before,
        "rel_after": rel_after,
        "fit_rel_before": fit_rel_before,
        "fit_rel_after": fit_rel_after,
        "holdout_rel_before": holdout_rel_before,
        "holdout_rel_after": holdout_rel_after,
        "anchor_error": anchor_error,
        "delta_degrees": delta_degrees,
        "fit_partition": str(args.fit_partition),
    }


def serialise_side(result: Mapping[str, Any]) -> dict[str, Any]:
    evidence = result["evidence"]
    applied = result["applied"]
    eligible = np.asarray(evidence["refine_eligible_mask"], dtype=bool)
    return {
        "fit_partition": result["fit_partition"],
        "evidence_available": bool(evidence["available"]),
        "missing_evidence_fields": list(evidence["missing_fields"]),
        "visible_evidence": contiguous_stats(
            np.asarray(evidence["visible_evidence_mask"], dtype=bool)
        ),
        "refine_eligible": contiguous_stats(eligible),
        "fit_frames": contiguous_stats(result["fit_mask"]),
        "holdout_fit_frames": contiguous_stats(result["holdout_fit_mask"]),
        "applied": contiguous_stats(applied),
        "absolute_reprojection_before_px": tensor_stat(
            result["abs_before"], eligible
        ),
        "absolute_reprojection_before_applied_px": tensor_stat(
            result["abs_before"], applied
        ),
        "absolute_reprojection_after_px": tensor_stat(result["abs_after"], applied),
        "relative_reprojection_before_px": tensor_stat(
            result["rel_before"], eligible
        ),
        "relative_reprojection_before_applied_px": tensor_stat(
            result["rel_before"], applied
        ),
        "relative_reprojection_after_px": tensor_stat(result["rel_after"], applied),
        "fit_relative_reprojection_before_applied_px": tensor_stat(
            result["fit_rel_before"], applied
        ),
        "fit_relative_reprojection_after_px": tensor_stat(
            result["fit_rel_after"], applied
        ),
        "holdout_relative_reprojection_before_applied_px": tensor_stat(
            result["holdout_rel_before"], applied
        ),
        "holdout_relative_reprojection_after_px": tensor_stat(
            result["holdout_rel_after"], applied
        ),
        "wrist_anchor_error_px": tensor_stat(
            result["anchor_error"], eligible
        ),
        "accepted_delta_degrees": tensor_stat(
            result["delta_degrees"], applied
        ),
        "rejection_counts": {
            "not_eligible": int(
                np.sum((result["reject"] & REJECT_NOT_ELIGIBLE) != 0)
            ),
            "fit_keypoints": int(
                np.sum((result["reject"] & REJECT_FIT_KEYPOINTS) != 0)
            ),
            "body_anchor": int(
                np.sum((result["reject"] & REJECT_BODY_ANCHOR) != 0)
            ),
            "no_improvement": int(
                np.sum((result["reject"] & REJECT_NO_IMPROVEMENT) != 0)
            ),
            "absolute_regression": int(
                np.sum((result["reject"] & REJECT_ABSOLUTE_REGRESSION) != 0)
            ),
            "delta_limit": int(
                np.sum((result["reject"] & REJECT_DELTA_LIMIT) != 0)
            ),
            "numerical": int(
                np.sum((result["reject"] & REJECT_NUMERICAL) != 0)
            ),
            "holdout_keypoints": int(
                np.sum((result["reject"] & REJECT_HOLDOUT_KEYPOINTS) != 0)
            ),
            "holdout_regression": int(
                np.sum((result["reject"] & REJECT_HOLDOUT_REGRESSION) != 0)
            ),
        },
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.steps < 0:
        raise ValueError("--batch_size must be >=1 and --steps must be >=0")
    if args.fit_min_keypoints < 1 or args.fit_min_keypoints > 21:
        raise ValueError("--fit_min_keypoints must be in [1, 21]")
    if args.holdout_min_keypoints < 1 or args.holdout_min_keypoints > 5:
        raise ValueError("--holdout_min_keypoints must be in [1, 5]")
    if not math.isfinite(args.holdout_max_relative_regression_px) or (
        args.holdout_max_relative_regression_px < 0.0
    ):
        raise ValueError("--holdout_max_relative_regression_px must be finite and >=0")
    if args.fit_partition == "non_tip" and args.fit_min_keypoints > 16:
        raise ValueError("non_tip partition supports at most 16 fit keypoints")
    for path in (args.mano_params, args.hmr4d_results, args.vitpose_wholebody):
        if not path.is_file():
            raise FileNotFoundError(path)
    device = choose_device(args.device)
    mano = load_torch(args.mano_params, weights_only=False)
    hmr = load_torch(args.hmr4d_results, weights_only=True)
    vitpose = load_torch(args.vitpose_wholebody, weights_only=True)
    if not isinstance(mano, dict) or not isinstance(hmr, dict):
        raise TypeError("mano_params and hmr4d_results must be dictionaries")
    if "smpl_params_incam" not in hmr or "K_fullimg" not in hmr:
        raise KeyError("hmr4d_results requires smpl_params_incam and K_fullimg")
    if not torch.is_tensor(vitpose):
        vitpose = torch.as_tensor(vitpose)
    if vitpose.ndim != 4 or vitpose.shape[-2:] != (133, 3):
        raise ValueError(f"Expected vitpose_wholebody (P,F,133,3), got {tuple(vitpose.shape)}")
    camera = hmr["K_fullimg"]
    if not torch.is_tensor(camera) or camera.ndim != 3 or camera.shape[-2:] != (3, 3):
        raise ValueError("K_fullimg must have shape (F,3,3)")

    evidence_config = VisibleEvidenceConfig(
        hand_confidence=float(args.evidence_hand_confidence),
        hand_min_keypoints=int(args.evidence_min_keypoints),
        hand_mean_confidence=float(args.evidence_mean_confidence),
        wrist_confidence=float(args.evidence_wrist_confidence),
        min_bbox_diagonal_px=float(args.evidence_min_bbox_diagonal_px),
        min_contiguous_frames=int(args.evidence_min_run),
    )
    # This must be the exact bridge used by render_incam after the 45-D MANO
    # fidelity repair. Do not change the global HMR model default.
    from hmr4d.utils.smplx_utils import make_smplx

    smplx = make_smplx("supermotion", use_pca=False).to(device).eval()
    try:
        results = {
            side: run_side(
                side,
                mano,
                hmr["smpl_params_incam"],
                camera,
                vitpose,
                smplx,
                device,
                args,
                evidence_config,
            )
            for side in ("left", "right")
        }
    finally:
        del smplx
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Persist every gate and acceptance decision in the output track. The
    # direct-MANO recompute stage will retain these fields while refreshing
    # joints from the accepted global orientations.
    for side, result in results.items():
        evidence = result["evidence"]
        for suffix, value in (
            ("visible_evidence_mask", evidence["visible_evidence_mask"]),
            ("visible_refine_eligible_mask", evidence["refine_eligible_mask"]),
            ("visible_refine_applied_mask", result["applied"]),
            ("visible_refine_fit_mask", result["fit_mask"]),
            ("visible_refine_holdout_fit_mask", result["holdout_fit_mask"]),
            ("visible_refine_rejection_code", result["reject"]),
            ("visible_refine_evidence_score", evidence["evidence_score"]),
            ("visible_refine_delta_degrees", result["delta_degrees"]),
            ("visible_refine_abs_reproj_before_px", result["abs_before"]),
            ("visible_refine_abs_reproj_after_px", result["abs_after"]),
            ("visible_refine_rel_reproj_before_px", result["rel_before"]),
            ("visible_refine_rel_reproj_after_px", result["rel_after"]),
            ("visible_refine_fit_rel_reproj_before_px", result["fit_rel_before"]),
            ("visible_refine_fit_rel_reproj_after_px", result["fit_rel_after"]),
            ("visible_refine_holdout_rel_reproj_before_px", result["holdout_rel_before"]),
            ("visible_refine_holdout_rel_reproj_after_px", result["holdout_rel_after"]),
            ("visible_refine_wrist_anchor_error_px", result["anchor_error"]),
        ):
            mano[f"{side}_hand_{suffix}"] = torch.from_numpy(np.asarray(value))
    mano["visible_hand_refine_config"] = {
        "schema_version": 2,
        "purpose": "small final-render-space wrist orientation correction under high 2D evidence",
        "palm_facing": "unknown_from_2d_keypoints",
        "args": jsonable_args(args),
        "evidence": evidence_config.to_dict(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mano, args.output)
    summary = {
        "schema_version": 2,
        "purpose": "audit-only acceptance report for visible-hand wrist refinement",
        "source_mano_params": str(args.mano_params.resolve()),
        "hmr4d_results": str(args.hmr4d_results.resolve()),
        "vitpose_wholebody": str(args.vitpose_wholebody.resolve()),
        "output": str(args.output.resolve()),
        "device": str(device),
        "config": jsonable_args(args),
        "evidence": evidence_config.to_dict(),
        "sides": {side: serialise_side(result) for side, result in results.items()},
        "limits": {
            "palm_back_semantics": "not evaluated; 2D keypoints cannot prove depth-facing orientation",
            "acceptance": "fit-partition relative reprojection improves while full-hand absolute reprojection and body-wrist anchor remain bounded",
            "holdout": (
                "non_tip partition reserves high-confidence fingertips for a "
                "relative-error non-regression check; it is still not an "
                "independent ground-truth label"
            ),
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    for side, stats in summary["sides"].items():
        print(
            f"[visible-refine] {side}: eligible={stats['refine_eligible']['frames']} "
            f"fit={stats['fit_frames']['frames']} "
            f"holdout={stats['holdout_fit_frames']['frames']} "
            f"applied={stats['applied']['frames']}",
            flush=True,
        )
    print(f"[visible-refine] output -> {args.output}")
    print(f"[visible-refine] summary -> {args.summary}")


if __name__ == "__main__":
    main()
