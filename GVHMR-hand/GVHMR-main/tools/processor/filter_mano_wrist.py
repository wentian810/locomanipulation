#!/usr/bin/env python
"""Temporal wrist-orientation filter for HaMeR MANO tracks.

The filter keeps HaMeR finger articulation untouched and only chooses a more
stable global wrist orientation.  For each frame it builds a small candidate set
and uses Viterbi dynamic programming to choose a temporally coherent path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp


LEFT_PARENT_CHAIN = np.asarray([3, 6, 9, 13, 16, 18], dtype=np.int64) - 1
RIGHT_PARENT_CHAIN = np.asarray([3, 6, 9, 14, 17, 19], dtype=np.int64) - 1
WRIST_LOCAL_INDEX = {"left": 19, "right": 20}


def as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def to_tensor_like(array, like):
    tensor = torch.from_numpy(np.asarray(array))
    if torch.is_tensor(like):
        return tensor.to(dtype=like.dtype)
    return tensor


def squeeze_hand_rot(rot):
    """Return (rotmats, had_single_axis, was_axis_angle)."""
    arr = as_numpy(rot)
    if arr.shape[-3:] == (1, 3, 3):
        return arr[..., 0, :, :], True, False
    if arr.shape[-2:] == (3, 3):
        return arr, False, False
    # hand4wholepp backend produces axis-angle (..., 3) instead of matrices
    if arr.shape[-1] == 3 and arr.ndim >= 3:
        arr = Rotation.from_rotvec(arr.reshape(-1, 3)).as_matrix().reshape(
            arr.shape[:-1] + (3, 3)
        )
        return arr, False, True
    raise ValueError(f"Expected rotation matrices or axis-angle, got shape {arr.shape}")


def restore_hand_rot(rot, had_single_axis):
    if had_single_axis:
        return rot[..., None, :, :]
    return rot


def rotation_angle_matrix(rot):
    trace = np.trace(rot, axis1=-2, axis2=-1)
    cos = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.arccos(cos)


def rot_about_axis(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    axis = axis / np.clip(norm, 1e-8, None)
    return Rotation.from_rotvec(axis * float(angle)).as_matrix()


def slerp_pair(rot_a, rot_b, weight):
    rots = Rotation.from_matrix(np.stack([rot_a, rot_b], axis=0))
    return Slerp([0.0, 1.0], rots)([float(weight)]).as_matrix()[0]


def body_wrist_global(smpl_params, side):
    global_orient = as_numpy(smpl_params["global_orient"])
    body_pose = as_numpy(smpl_params["body_pose"])
    if global_orient.shape[-3:] == (1, 3, 3):
        global_orient = global_orient[..., 0, :, :]
    if global_orient.ndim != 4 or body_pose.ndim != 5:
        raise ValueError(
            "Expected global_orient=(P,F,1,3,3) and body_pose=(P,F,21,3,3), "
            f"got {global_orient.shape} and {body_pose.shape}"
        )

    chain = LEFT_PARENT_CHAIN if side == "left" else RIGHT_PARENT_CHAIN
    wrist_idx = WRIST_LOCAL_INDEX[side]
    parent = global_orient.copy()
    for idx in chain:
        parent = parent @ body_pose[:, :, idx]
    return parent @ body_pose[:, :, wrist_idx]


def candidate_rotations(hamer, body, mix_weight):
    frame_count = hamer.shape[0]
    cands = np.empty((frame_count, 4, 3, 3), dtype=np.float64)
    cands[:, 0] = hamer

    # Approximate forearm/hand longitudinal axis from the body wrist frame.
    # The exact elbow-wrist vector is not stored in mano_params.pt, but this
    # gives the Viterbi filter an explicit palm/back alternate branch.
    axis = body[:, :, 2]
    cands[:, 1] = rot_about_axis(axis, np.pi) @ hamer
    cands[:, 2] = body
    for t in range(frame_count):
        cands[t, 3] = slerp_pair(body[t], hamer[t], mix_weight)
    return cands


def viterbi_filter(
    cands,
    body,
    valid,
    w_temp,
    w_body,
    w_temp_low,
    w_body_low,
    hamer_penalty,
    flip_penalty,
    body_penalty,
    mix_penalty,
    invalid_hamer_penalty,
    invalid_flip_penalty,
    invalid_mix_penalty,
):
    frame_count, cand_count = cands.shape[:2]
    unary = np.empty((frame_count, cand_count), dtype=np.float64)
    valid_prior = np.asarray(
        [hamer_penalty, flip_penalty, body_penalty, mix_penalty],
        dtype=np.float64,
    )
    invalid_prior = np.asarray(
        [invalid_hamer_penalty, invalid_flip_penalty, 0.0, invalid_mix_penalty],
        dtype=np.float64,
    )
    for t in range(frame_count):
        body_delta = cands[t] @ np.swapaxes(body[t], -1, -2)
        body_cost = rotation_angle_matrix(body_delta) ** 2
        if valid[t]:
            unary[t] = valid_prior + w_body * body_cost
        else:
            unary[t] = invalid_prior + w_body_low * body_cost

    dp = np.empty_like(unary)
    prev = np.zeros((frame_count, cand_count), dtype=np.int64)
    dp[0] = unary[0]
    for t in range(1, frame_count):
        trans = cands[t, :, None] @ np.swapaxes(cands[t - 1, None], -1, -2)
        trans_cost = rotation_angle_matrix(trans) ** 2
        temp_weight = w_temp if valid[t] else w_temp_low
        scores = dp[t - 1][None, :] + temp_weight * trans_cost
        prev[t] = np.argmin(scores, axis=1)
        dp[t] = unary[t] + np.min(scores, axis=1)

    path = np.zeros(frame_count, dtype=np.int64)
    path[-1] = int(np.argmin(dp[-1]))
    for t in range(frame_count - 1, 0, -1):
        path[t - 1] = prev[t, path[t]]
    filtered = cands[np.arange(frame_count), path]
    return filtered, path


def filter_side(mano, smpl_params, side, args):
    key = f"{side}_hand_global_orient"
    valid_key = f"{side}_hand_valid"
    if key not in mano:
        return {}, {}

    raw_tensor = mano[key]
    hamer, had_single_axis, was_axis_angle = squeeze_hand_rot(raw_tensor)
    body_all = body_wrist_global(smpl_params, side)
    valid_all = as_numpy(mano.get(valid_key, np.ones(hamer.shape[:2], dtype=bool))).astype(bool)

    people, frames = hamer.shape[:2]
    filtered = np.empty_like(hamer, dtype=np.float64)
    paths = np.zeros((people, frames), dtype=np.int64)
    for person_idx in range(people):
        cands = candidate_rotations(
            hamer[person_idx].astype(np.float64),
            body_all[person_idx].astype(np.float64),
            args.mix_weight,
        )
        fixed, path = viterbi_filter(
            cands,
            body_all[person_idx].astype(np.float64),
            valid_all[person_idx],
            args.w_temp,
            args.w_body,
            args.w_temp_low_conf,
            args.w_body_low_conf,
            args.hamer_penalty,
            args.flip_penalty,
            args.body_penalty,
            args.mix_penalty,
            args.invalid_hamer_penalty,
            args.invalid_flip_penalty,
            args.invalid_mix_penalty,
        )
        filtered[person_idx] = fixed
        paths[person_idx] = path

    restored = restore_hand_rot(filtered.astype(np.float32), had_single_axis)
    updates = {
        f"{side}_hand_global_orient_unfiltered": raw_tensor.detach().cpu().clone()
        if torch.is_tensor(raw_tensor)
        else np.asarray(raw_tensor).copy(),
        f"{side}_hand_global_orient_filtered": to_tensor_like(restored, raw_tensor),
        key: to_tensor_like(restored, raw_tensor),
        f"{side}_hand_wrist_candidate_path": torch.from_numpy(paths),
    }
    stats = {
        f"{side}_candidate_hamer": int(np.sum(paths == 0)),
        f"{side}_candidate_flip": int(np.sum(paths == 1)),
        f"{side}_candidate_body": int(np.sum(paths == 2)),
        f"{side}_candidate_mix": int(np.sum(paths == 3)),
    }
    return updates, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mano_params", required=True)
    parser.add_argument("--hmr4d_results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--w_temp", type=float, default=2.0)
    parser.add_argument("--w_body", type=float, default=0.5)
    parser.add_argument("--w_temp_low_conf", type=float, default=4.0)
    parser.add_argument("--w_body_low_conf", type=float, default=1.0)
    parser.add_argument("--mix_weight", type=float, default=0.35)
    parser.add_argument("--hamer_penalty", type=float, default=0.0)
    parser.add_argument("--flip_penalty", type=float, default=0.15)
    parser.add_argument("--body_penalty", type=float, default=0.35)
    parser.add_argument("--mix_penalty", type=float, default=0.1)
    parser.add_argument("--invalid_hamer_penalty", type=float, default=1.0)
    parser.add_argument("--invalid_flip_penalty", type=float, default=1.0)
    parser.add_argument("--invalid_mix_penalty", type=float, default=0.2)
    args = parser.parse_args()

    mano_path = Path(args.mano_params)
    hmr_path = Path(args.hmr4d_results)
    output = Path(args.output)
    mano = torch.load(mano_path, map_location="cpu", weights_only=False)
    hmr = torch.load(hmr_path, map_location="cpu", weights_only=False)
    smpl_params = hmr["smpl_params_incam"]

    out = dict(mano)
    stats = {}
    for side in ("left", "right"):
        updates, side_stats = filter_side(out, smpl_params, side, args)
        out.update(updates)
        stats.update(side_stats)

    out["wrist_filter_stats"] = stats
    out["wrist_filter_config"] = {
        "w_temp": float(args.w_temp),
        "w_body": float(args.w_body),
        "w_temp_low_conf": float(args.w_temp_low_conf),
        "w_body_low_conf": float(args.w_body_low_conf),
        "mix_weight": float(args.mix_weight),
        "hamer_penalty": float(args.hamer_penalty),
        "flip_penalty": float(args.flip_penalty),
        "body_penalty": float(args.body_penalty),
        "mix_penalty": float(args.mix_penalty),
        "invalid_hamer_penalty": float(args.invalid_hamer_penalty),
        "invalid_flip_penalty": float(args.invalid_flip_penalty),
        "invalid_mix_penalty": float(args.invalid_mix_penalty),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output)
    print(f"Saved wrist-filtered MANO params to {output}")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
