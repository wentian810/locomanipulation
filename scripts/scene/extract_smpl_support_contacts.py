#!/usr/bin/env python3
"""Extract conservative SMPL seat and backrest contact anchors for V20.

The input motion remains in neg-Y-up coordinates. This utility reconstructs the
SMPL-H mesh, converts it to the packaged Z-up chair frame, and stores candidate
posterior torso and lower-body anchors. It is diagnostic: no scene or motion is
modified here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import smplx


def _primitive(items: list[dict], name: str) -> dict:
    for item in items:
        if item.get("name") == name:
            return item
    raise KeyError(f"Missing primitive {name!r}")


def _z_up_from_neg_y(points: np.ndarray) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64)[..., [0, 2, 1]].copy()
    result[..., 1] *= -1.0
    return result


def _summary(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"median_m": float("nan"), "p05_m": float("nan"), "p95_m": float("nan")}
    return {
        "median_m": float(np.median(finite)),
        "p05_m": float(np.percentile(finite, 5)),
        "p95_m": float(np.percentile(finite, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-motion", type=Path, required=True)
    parser.add_argument("--chair-primitives", type=Path, required=True)
    parser.add_argument("--body-model-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--contact-gap-m", type=float, default=0.08)
    args = parser.parse_args()

    motion = np.load(args.human_motion, allow_pickle=False)
    required = {"trans", "root_orient", "pose_body", "betas"}
    missing = required - set(motion.files)
    if missing:
        raise KeyError(f"human motion missing keys: {sorted(missing)}")
    chair = json.loads(args.chair_primitives.read_text(encoding="utf-8"))
    if chair.get("frame") != "mujoco_world_z_up":
        raise ValueError("chair primitives must use mujoco_world_z_up")
    seat = _primitive(chair["primitives"], "seat_support")
    backrest = _primitive(chair["primitives"], "backrest")
    center = np.asarray(seat["center"], dtype=np.float64)
    rotation = np.asarray(seat["rotation_matrix"], dtype=np.float64)
    extents = np.asarray(seat["extents"], dtype=np.float64)
    seat_top = 0.5 * extents[2]
    back_center_local = (np.asarray(backrest["center"], dtype=np.float64) - center) @ rotation
    back_thickness = float(np.asarray(backrest["extents"], dtype=np.float64)[1])
    # The V4 visual chair has its thin backrest axis on local +Y. The front face
    # is the side toward the seat centre.
    back_front_y = float(back_center_local[1] - np.sign(back_center_local[1]) * 0.5 * back_thickness)

    trans = np.asarray(motion["trans"], dtype=np.float32)
    T = len(trans)
    requested_gender = str(motion["gender"].item()) if "gender" in motion.files else "neutral"
    gender = requested_gender.lower()
    expected = args.body_model_root / "smplh" / f"SMPLH_{gender.upper()}.npz"
    if not expected.is_file():
        gender = "neutral"
    model = smplx.create(
        str(args.body_model_root),
        model_type="smplh",
        gender=gender,
        ext="npz",
        num_betas=len(motion["betas"]),
        use_pca=False,
        batch_size=T,
    ).to(args.device)
    betas = np.repeat(np.asarray(motion["betas"], dtype=np.float32)[None], T, axis=0)
    with torch.no_grad():
        output = model(
            betas=torch.as_tensor(betas, device=args.device),
            global_orient=torch.as_tensor(motion["root_orient"], dtype=torch.float32, device=args.device),
            body_pose=torch.as_tensor(motion["pose_body"], dtype=torch.float32, device=args.device),
            transl=torch.as_tensor(trans, dtype=torch.float32, device=args.device),
            return_verts=True,
        )
    vertices = _z_up_from_neg_y(output.vertices.detach().cpu().numpy())
    pelvis = _z_up_from_neg_y(output.joints[:, 0].detach().cpu().numpy())
    pelvis_local = (pelvis - center) @ rotation
    sit_mask = (
        (np.abs(pelvis_local[:, 0]) <= 0.5 * extents[0] + 0.15)
        & (np.abs(pelvis_local[:, 1]) <= 0.5 * extents[1] + 0.15)
        & (pelvis_local[:, 2] >= 0.10)
        & (pelvis_local[:, 2] <= 0.65)
    )

    seat_anchor = np.full((T, 3), np.nan, dtype=np.float32)
    back_anchor = np.full((T, 3), np.nan, dtype=np.float32)
    seat_signed = np.full(T, np.nan, dtype=np.float32)
    back_gap = np.full(T, np.nan, dtype=np.float32)
    for frame in np.flatnonzero(sit_mask):
        local_vertices = (vertices[frame] - center) @ rotation
        relative = vertices[frame] - pelvis[frame]
        relative_height = relative[:, 2]
        # Lower pelvis/thigh region: exclude feet and head while requiring the
        # candidate to project onto the actual seat footprint.
        seat_candidates = (
            (relative_height >= -0.35)
            & (relative_height <= 0.05)
            & (np.abs(local_vertices[:, 0]) <= 0.5 * extents[0] + 0.10)
            & (np.abs(local_vertices[:, 1]) <= 0.5 * extents[1] + 0.10)
        )
        if np.any(seat_candidates):
            candidate_ids = np.flatnonzero(seat_candidates)
            chosen = candidate_ids[np.argmin(np.abs(local_vertices[candidate_ids, 2] - seat_top))]
            seat_anchor[frame] = vertices[frame, chosen]
            seat_signed[frame] = local_vertices[chosen, 2] - seat_top

        toward_back = np.asarray(backrest["center"], dtype=np.float64) - pelvis[frame]
        toward_back[2] = 0.0
        norm = float(np.linalg.norm(toward_back))
        if norm < 1e-6:
            continue
        toward_back /= norm
        side = np.cross(np.asarray([0.0, 0.0, 1.0]), toward_back)
        # Posterior central torso region: a robust per-frame farthest point
        # toward the known backrest, excluding arms/head/legs by height and width.
        back_candidates = (
            (relative_height >= 0.08)
            & (relative_height <= 0.60)
            & ((relative @ toward_back) >= 0.02)
            & (np.abs(relative @ side) <= 0.28)
        )
        if np.any(back_candidates):
            candidate_ids = np.flatnonzero(back_candidates)
            chosen = candidate_ids[np.argmax(relative[candidate_ids] @ toward_back)]
            back_anchor[frame] = vertices[frame, chosen]
            back_gap[frame] = back_front_y - local_vertices[chosen, 1]

    seat_contact = np.isfinite(seat_signed) & (seat_signed >= -0.02) & (seat_signed <= 0.04)
    back_contact = np.isfinite(back_gap) & (back_gap >= -0.03) & (back_gap <= args.contact_gap_m)
    args.anchors.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.anchors,
        frame_index=np.arange(T, dtype=np.int32),
        sit_mask=sit_mask,
        seat_anchor_z_up=seat_anchor,
        back_anchor_z_up=back_anchor,
        seat_signed_distance_m=seat_signed,
        backrest_front_gap_m=back_gap,
        seat_contact=seat_contact,
        back_contact=back_contact,
    )
    report = {
        "schema_version": 1,
        "mode": "diagnostic_no_scene_or_motion_adjustment",
        "human_motion": str(args.human_motion),
        "chair_primitives": str(args.chair_primitives),
        "model_gender_requested": requested_gender,
        "model_gender_used": gender,
        "sit_frames": np.flatnonzero(sit_mask).astype(int).tolist(),
        "seat_contact_coverage": float(np.mean(seat_contact[sit_mask])) if np.any(sit_mask) else 0.0,
        "back_contact_coverage": float(np.mean(back_contact[sit_mask])) if np.any(sit_mask) else 0.0,
        "seat_signed_distance": _summary(seat_signed[sit_mask]),
        "backrest_front_gap": _summary(back_gap[sit_mask]),
        "contact_band_m": {"seat": [-0.02, 0.04], "backrest": [-0.03, args.contact_gap_m]},
        "next_gate": (
            "Use only stable anchors from this report in V20 planar refinement; "
            "do not infer a chair adjustment from root translation alone."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

