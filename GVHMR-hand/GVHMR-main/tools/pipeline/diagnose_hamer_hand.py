#!/usr/bin/env python3
"""Visual diagnostics for the GVHMR-hand / HaMeR hand track.

The script overlays VitPose hand keypoints and HaMeR crop boxes on the input
video, then adds compact timelines for the signals that matter most for wrist
flip / grasp debugging:

  * valid mask and reprojection error
  * MANO global-orient speed
  * finger curl proxy from 21 MANO joints
  * disagreement between stored palm normal and palm normal rebuilt from
    21-joint MCP geometry

It is intentionally read-only and can run on cached pipeline outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch


HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
FINGERTIPS = np.asarray([4, 8, 12, 16, 20], dtype=np.int64)
MCP_JOINTS = np.asarray([5, 9, 13, 17], dtype=np.int64)
TIMELINE_HEIGHT = 340


def as_np(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_vitpose(path: Path) -> np.ndarray:
    data = torch.load(path, map_location="cpu", weights_only=True)
    arr = as_np(data)
    if arr.ndim != 4 or arr.shape[-2] < 133:
        raise ValueError(f"Expected vitpose_wholebody shape (P,F,133,3), got {arr.shape}")
    return arr.astype(np.float32)


def choose_person(vitpose: np.ndarray, person_idx: int) -> np.ndarray:
    if person_idx < 0:
        # Pick the track with the most confident body keypoints.
        scores = vitpose[:, :, :17, 2].mean(axis=(1, 2))
        person_idx = int(np.argmax(scores))
    return vitpose[person_idx]


def bbox_from_keypoints(keyp, conf_thr=0.5, low_conf_thr=0.2, hi_min_keypoints=6, min_keypoints=4, min_size=96):
    keyp = np.asarray(keyp, dtype=np.float32)
    finite_xy = np.isfinite(keyp[:, :2]).all(axis=1)
    conf = np.where(finite_xy & np.isfinite(keyp[:, 2]), keyp[:, 2], -np.inf)
    valid_hi = conf > float(conf_thr)
    valid_lo = conf > float(low_conf_thr)
    selected = valid_hi if int(valid_hi.sum()) >= int(hi_min_keypoints) else valid_lo
    if int(selected.sum()) < int(min_keypoints):
        return None, False
    pts = keyp[selected, :2]
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    side = max(float(x2 - x1), float(y2 - y1), float(min_size))
    return np.asarray([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2], dtype=np.float32), True


def normalize(vec, eps=1e-8):
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    out = np.zeros_like(vec, dtype=np.float64)
    valid = norm[..., 0] > eps
    out[valid] = vec[valid] / norm[valid]
    return out, valid


def geom_palm_normal_from_21(joints, side):
    joints = np.asarray(joints, dtype=np.float64)
    z_axis, z_valid = normalize(joints[:, 9] - joints[:, 0])
    if side == "right":
        y_aux, y_valid = normalize(joints[:, 5] - joints[:, 13])
    else:
        y_aux, y_valid = normalize(joints[:, 13] - joints[:, 5])
    normal, n_valid = normalize(np.cross(y_aux, z_axis))
    return normal, z_valid & y_valid & n_valid


def curl_proxy_from_21(joints):
    joints = np.asarray(joints, dtype=np.float64)
    wrist = joints[:, 0]
    tips = joints[:, FINGERTIPS]
    dist = np.linalg.norm(tips - wrist[:, None, :], axis=-1)
    # Per-finger open baseline from the largest observed distances. This is not
    # a physical curl angle; it is just a stable motion indicator.
    open_dist = np.percentile(dist, 95, axis=0)
    open_dist = np.clip(open_dist, 1e-6, None)
    curl = 1.0 - dist / open_dist[None, :]
    return np.clip(curl, -0.5, 1.0)


def hand_2d_open_raw(keypoints, low_conf_thr=0.2, min_keypoints=4):
    keypoints = np.asarray(keypoints, dtype=np.float32)
    frame_count = keypoints.shape[0]
    raw = np.full(frame_count, np.nan, dtype=np.float32)
    evidence = np.zeros(frame_count, dtype=np.float32)
    visible = np.zeros(frame_count, dtype=bool)
    for frame_idx in range(frame_count):
        kp = keypoints[frame_idx]
        finite = np.isfinite(kp[:, :2]).all(axis=1) & np.isfinite(kp[:, 2])
        valid = finite & (kp[:, 2] > float(low_conf_thr))
        if int(valid.sum()) < int(min_keypoints) or not valid[0]:
            continue

        mcp_valid = MCP_JOINTS[valid[MCP_JOINTS]]
        tip_valid = FINGERTIPS[valid[FINGERTIPS]]
        if tip_valid.size < 2:
            continue

        wrist = kp[0, :2]
        if mcp_valid.size >= 2:
            palm = np.median(np.linalg.norm(kp[mcp_valid, :2] - wrist[None], axis=1))
        else:
            pts = kp[valid, :2]
            side = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 1.0)
            palm = side * 0.35
        if not np.isfinite(palm) or palm <= 1.0:
            continue

        tips = kp[tip_valid, :2]
        tip_extent = float(np.mean(np.linalg.norm(tips - wrist[None], axis=1)) / palm)
        if tips.shape[0] >= 2:
            diffs = tips[:, None, :] - tips[None, :, :]
            tip_spread = float(np.max(np.linalg.norm(diffs, axis=-1)) / palm)
        else:
            tip_spread = 0.0
        raw[frame_idx] = 0.7 * tip_extent + 0.3 * tip_spread
        evidence[frame_idx] = float(np.mean(kp[valid, 2]) * np.sqrt(int(valid.sum())))
        visible[frame_idx] = True
    return raw, evidence, visible


def normalize_percentile(values, lo_pct=20, hi_pct=85):
    values = np.asarray(values, dtype=np.float32)
    out = np.full_like(values, np.nan, dtype=np.float32)
    finite = np.isfinite(values)
    if int(finite.sum()) < 10:
        return out
    lo, hi = np.percentile(values[finite], [lo_pct, hi_pct])
    denom = max(float(hi - lo), 1e-6)
    out[finite] = np.clip((values[finite] - float(lo)) / denom, 0.0, 1.0)
    return out


def rotation_speed(rotation):
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.ndim >= 3 and rotation.shape[-2:] == (3, 3):
        rotmat = rotation.reshape(-1, 3, 3)
        speed = np.zeros(rotmat.shape[0], dtype=np.float32)
        if rotmat.shape[0] > 1:
            rel = rotmat[1:] @ np.swapaxes(rotmat[:-1], -1, -2)
            trace = np.einsum("...ii->...", rel)
            cos = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
            speed[1:] = np.arccos(cos).astype(np.float32)
        return speed

    rotvec = rotation.reshape(-1, 3)
    if rotvec.shape[0] <= 1:
        return np.zeros(rotvec.shape[0], dtype=np.float32)
    speed = np.zeros(rotvec.shape[0], dtype=np.float32)
    speed[1:] = np.linalg.norm(np.diff(rotvec, axis=0), axis=1)
    return speed


def load_sidecar(path: Path):
    with np.load(path, allow_pickle=True) as data:
        out = {key: np.asarray(data[key]) for key in data.files}
    return out


def squeeze_mano_track(value):
    arr = as_np(value)
    if arr.ndim >= 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim >= 3 and arr.shape[1] == 1 and arr.shape[-2:] == (3, 3):
        arr = arr[:, 0]
    return np.asarray(arr)


def load_mano_params(path: Path):
    data = torch.load(path, map_location="cpu")
    out = {}
    for side in ("left", "right"):
        for suffix in (
            "hand_global_orient",
            "hand_pose",
            "hand_joints_3d",
            "hand_valid",
            "hand_reproj_error",
            "hand_bbox_xyxy",
            "hand_bad_mask",
            "hand_spike_mask",
            "hand_temporal_fixed_mask",
            "hand_bbox_shrink_mask",
            "hand_bbox_jump_mask",
            "hand_bbox_overlap_mask",
            "hand_bbox_merge_fixed_mask",
            "hand_bbox_size_floor_mask",
            "hand_finger_fixed_mask",
            "hand_curl_spike_mask",
            "hand_finger_low_evidence_mask",
            "hand_open2d_mano_curl_mismatch_mask",
            "hand_open_rescue_mask",
            "hand_open_anchor_mask",
            "hand_finger_rate_limited_mask",
            "hand_joint_rate_limited_mask",
            "hand_size_floor_mask",
        ):
            key = f"{side}_{suffix}"
            if key in data:
                out[key] = squeeze_mano_track(data[key])
        if f"{side}_hand_bad_mask" not in out and f"{side}_hand_valid" in out:
            out[f"{side}_hand_bad_mask"] = ~out[f"{side}_hand_valid"].astype(bool)
        if f"{side}_hand_spike_mask" not in out and f"{side}_hand_valid" in out:
            out[f"{side}_hand_spike_mask"] = np.zeros_like(out[f"{side}_hand_valid"], dtype=bool)
    return out


def load_hand_track(path: Path):
    if path.suffix == ".pt":
        return load_mano_params(path), "mano_params"
    return load_sidecar(path), "sidecar"


def hand_track_frame_count(hand_track):
    for key in ("left_hand_valid", "left_hand_joints_3d", "left_hand_global_orient", "left_hand_pose"):
        if key in hand_track:
            return int(np.asarray(hand_track[key]).shape[0])
    raise ValueError("Could not infer hand track frame count")


def compute_side_metrics(sidecar, frame_count):
    metrics = {}
    for side in ("left", "right"):
        joints = np.asarray(sidecar[f"{side}_hand_joints_3d"], dtype=np.float64)[:frame_count]
        valid = np.asarray(sidecar.get(f"{side}_hand_valid", np.ones(frame_count)), dtype=bool)[:frame_count]
        if f"{side}_hand_wrist_frame_valid" in sidecar:
            valid = valid & np.asarray(sidecar[f"{side}_hand_wrist_frame_valid"], dtype=bool)[:frame_count]
        geom_normal, geom_valid = geom_palm_normal_from_21(joints, side)
        if f"{side}_hand_palm_normal" in sidecar:
            palm = np.asarray(sidecar[f"{side}_hand_palm_normal"], dtype=np.float64)[:frame_count]
            palm_norm, palm_valid = normalize(palm)
            dot = np.einsum("ij,ij->i", palm_norm, geom_normal)
            dot[~(valid & geom_valid & palm_valid)] = np.nan
            dot_mode = "stored_vs_geom"
        else:
            dot = np.full(frame_count, np.nan, dtype=np.float64)
            if frame_count > 1:
                continuity = np.einsum("ij,ij->i", geom_normal[1:], geom_normal[:-1])
                dot[1:] = continuity
                dot[~(valid & geom_valid)] = np.nan
            dot_mode = "temporal_geom"
        curl = curl_proxy_from_21(joints)
        global_orient = np.asarray(sidecar[f"{side}_hand_global_orient"], dtype=np.float64)[:frame_count]
        reproj = np.asarray(sidecar.get(f"{side}_hand_reproj_error", np.full(frame_count, np.nan)), dtype=np.float64)[:frame_count]
        bad = np.asarray(sidecar.get(f"{side}_hand_bad_mask", ~valid), dtype=bool)[:frame_count]
        spike = np.asarray(sidecar.get(f"{side}_hand_spike_mask", np.zeros(frame_count, dtype=bool)), dtype=bool)[:frame_count]
        fixed = np.asarray(sidecar.get(f"{side}_hand_temporal_fixed_mask", np.zeros(frame_count, dtype=bool)), dtype=bool)[:frame_count]
        shrink = np.asarray(sidecar.get(f"{side}_hand_bbox_shrink_mask", np.zeros(frame_count, dtype=bool)), dtype=bool)[:frame_count]
        jump = np.asarray(sidecar.get(f"{side}_hand_bbox_jump_mask", np.zeros(frame_count, dtype=bool)), dtype=bool)[:frame_count]
        overlap = np.asarray(sidecar.get(f"{side}_hand_bbox_overlap_mask", np.zeros(frame_count, dtype=bool)), dtype=bool)[:frame_count]
        merge_fixed = np.asarray(sidecar.get(f"{side}_hand_bbox_merge_fixed_mask", np.zeros(frame_count, dtype=bool)), dtype=bool)[:frame_count]
        bbox_size_floor = np.asarray(sidecar.get(f"{side}_hand_bbox_size_floor_mask", np.zeros(frame_count, dtype=bool)), dtype=bool)[:frame_count]
        finger_fixed = np.asarray(sidecar.get(f"{side}_hand_finger_fixed_mask", np.zeros(frame_count, dtype=bool)), dtype=bool)[:frame_count]
        curl_spike = np.asarray(sidecar.get(f"{side}_hand_curl_spike_mask", np.zeros(frame_count, dtype=bool)), dtype=bool)[:frame_count]
        finger_low_evidence = np.asarray(sidecar.get(f"{side}_hand_finger_low_evidence_mask", np.zeros(frame_count, dtype=bool)), dtype=bool)[:frame_count]
        open_rescue = np.asarray(sidecar.get(f"{side}_hand_open_rescue_mask", np.zeros(frame_count, dtype=bool)), dtype=bool)[:frame_count]
        open_anchor = np.asarray(sidecar.get(f"{side}_hand_open_anchor_mask", np.zeros(frame_count, dtype=bool)), dtype=bool)[:frame_count]
        rate_limited = np.asarray(sidecar.get(f"{side}_hand_finger_rate_limited_mask", np.zeros(frame_count, dtype=bool)), dtype=bool)[:frame_count]
        joint_rate_limited = np.asarray(sidecar.get(f"{side}_hand_joint_rate_limited_mask", np.zeros(frame_count, dtype=bool)), dtype=bool)[:frame_count]
        size_floor = np.asarray(sidecar.get(f"{side}_hand_size_floor_mask", np.zeros(frame_count, dtype=bool)), dtype=bool)[:frame_count]
        bbox = None
        if f"{side}_hand_bbox_xyxy" in sidecar:
            bbox = np.asarray(sidecar[f"{side}_hand_bbox_xyxy"], dtype=np.float32)[:frame_count]
        metrics[side] = {
            "valid": valid,
            "joints": joints.astype(np.float32),
            "dot": dot.astype(np.float32),
            "dot_mode": dot_mode,
            "curl": np.nanmean(curl, axis=1).astype(np.float32),
            "reproj": reproj.astype(np.float32),
            "global_speed": rotation_speed(global_orient),
            "bad": bad,
            "spike": spike,
            "fixed": fixed,
            "shrink": shrink,
            "jump": jump,
            "overlap": overlap,
            "merge_fixed": merge_fixed,
            "bbox_size_floor": bbox_size_floor,
            "finger_fixed": finger_fixed,
            "curl_spike": curl_spike,
            "finger_low_evidence": finger_low_evidence,
            "open_rescue": open_rescue,
            "open_anchor": open_anchor,
            "rate_limited": rate_limited,
            "joint_rate_limited": joint_rate_limited,
            "size_floor": size_floor,
            "bbox": bbox,
        }
    return metrics


def add_vitpose_finger_metrics(
    metrics,
    vitpose,
    frame_count,
    low_conf_thr=0.2,
    min_keypoints=4,
    open_score_thr=0.7,
    curl_percentile=55.0,
    evidence_thr=1.0,
):
    for side, keypoints in (
        ("left", vitpose[:frame_count, -42:-21]),
        ("right", vitpose[:frame_count, -21:]),
    ):
        open_raw, evidence, visible = hand_2d_open_raw(
            keypoints,
            low_conf_thr=low_conf_thr,
            min_keypoints=min_keypoints,
        )
        open_score = normalize_percentile(open_raw)
        curl = np.asarray(metrics[side]["curl"], dtype=np.float32)[:frame_count]
        finite_curl = curl[np.isfinite(curl)]
        curl_thr = float(np.percentile(finite_curl, curl_percentile)) if finite_curl.size else np.inf
        mismatch = (
            visible
            & np.isfinite(open_score)
            & (open_score >= float(open_score_thr))
            & np.isfinite(curl)
            & (curl >= curl_thr)
            & (evidence >= float(evidence_thr))
        )
        metrics[side]["open2d_raw"] = open_raw
        metrics[side]["open2d_score"] = open_score
        metrics[side]["open2d_evidence"] = evidence
        metrics[side]["open2d_visible"] = visible
        metrics[side]["open2d_mano_curl_mismatch"] = mismatch
        metrics[side]["open2d_curl_threshold"] = np.full(frame_count, curl_thr, dtype=np.float32)
    return metrics


def draw_hand_keypoints(frame, keyp, color, conf_thr=0.2):
    pts = np.asarray(keyp, dtype=np.float32)
    for a, b in HAND_EDGES:
        if pts[a, 2] > conf_thr and pts[b, 2] > conf_thr:
            cv2.line(frame, tuple(np.round(pts[a, :2]).astype(int)), tuple(np.round(pts[b, :2]).astype(int)), color, 2, cv2.LINE_AA)
    for idx, point in enumerate(pts):
        if point[2] > conf_thr:
            radius = 4 if idx in FINGERTIPS else 3
            cv2.circle(frame, tuple(np.round(point[:2]).astype(int)), radius, color, -1, cv2.LINE_AA)


def draw_bbox(frame, bbox, color):
    if bbox is None:
        return
    x1, y1, x2, y2 = np.round(bbox).astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)


def draw_text(frame, text, org, color=(255, 255, 255), scale=0.55):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def frame_flags(metrics, side, frame_idx):
    m = metrics[side]
    empty = np.zeros_like(m["valid"], dtype=bool)
    flags = []
    if bool(m.get("fixed", empty)[frame_idx]):
        flags.append("FIXED")
    if bool(m.get("shrink", empty)[frame_idx]):
        flags.append("SHRINK")
    if bool(m.get("jump", empty)[frame_idx]):
        flags.append("JUMP")
    if bool(m.get("overlap", empty)[frame_idx]):
        flags.append("OVERLAP")
    if bool(m.get("merge_fixed", empty)[frame_idx]):
        flags.append("MERGE_FIX")
    if bool(m.get("bbox_size_floor", empty)[frame_idx]):
        flags.append("BBOX_SIZE")
    if bool(m.get("finger_fixed", empty)[frame_idx]):
        flags.append("FINGER_FIXED")
    if bool(m.get("curl_spike", empty)[frame_idx]):
        flags.append("CURL_SPIKE")
    if bool(m.get("finger_low_evidence", empty)[frame_idx]):
        flags.append("LOW_2D")
    if bool(m.get("open_rescue", empty)[frame_idx]):
        flags.append("OPEN_RESCUE")
    if bool(m.get("rate_limited", empty)[frame_idx]):
        flags.append("RATE_LIMIT")
    if bool(m.get("joint_rate_limited", empty)[frame_idx]):
        flags.append("JOINT_LIMIT")
    if bool(m.get("size_floor", empty)[frame_idx]):
        flags.append("SIZE_FLOOR")
    if bool(m.get("open2d_mano_curl_mismatch", empty)[frame_idx]):
        flags.append("2D_OPEN_MANO_CURL")
    if bool(m["bad"][frame_idx]):
        flags.append("BAD")
    return flags


def bbox_color(metrics, side, frame_idx, base_color):
    flags = set(frame_flags(metrics, side, frame_idx))
    if "BAD" in flags:
        return (40, 40, 230)
    if "OVERLAP" in flags:
        return (230, 80, 230)
    if "MERGE_FIX" in flags:
        return (255, 220, 40)
    if "BBOX_SIZE" in flags:
        return (60, 220, 220)
    if "2D_OPEN_MANO_CURL" in flags:
        return (0, 160, 255)
    if "OPEN_RESCUE" in flags:
        return (0, 220, 255)
    if "CURL_SPIKE" in flags or "FINGER_FIXED" in flags:
        return (0, 200, 180)
    if "SHRINK" in flags or "JUMP" in flags:
        return (40, 180, 255)
    if "FIXED" in flags:
        return (0, 240, 255)
    if not bool(metrics[side]["valid"][frame_idx]):
        return (80, 80, 160)
    return base_color


def draw_fingertip_emphasis(frame, keyp, color, conf_thr=0.2):
    pts = np.asarray(keyp, dtype=np.float32)
    for idx in FINGERTIPS:
        if pts[idx, 2] > conf_thr and np.isfinite(pts[idx, :2]).all():
            cv2.circle(frame, tuple(np.round(pts[idx, :2]).astype(int)), 7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, tuple(np.round(pts[idx, :2]).astype(int)), 6, color, 2, cv2.LINE_AA)


def draw_mano_inset(frame, joints, side, origin, size, color, label):
    joints = np.asarray(joints, dtype=np.float32)
    x0, y0 = origin
    x1, y1 = x0 + size, y0 + size
    h, w = frame.shape[:2]
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return

    roi = frame[y0:y1, x0:x1]
    panel = np.zeros_like(roi)
    panel[:] = (18, 18, 22)
    if joints.shape[0] >= 21 and np.isfinite(joints).all():
        root = joints[0]
        z_axis = joints[9] - root
        z_norm = float(np.linalg.norm(z_axis))
        if z_norm > 1e-8:
            z_axis = z_axis / z_norm
            x_aux = joints[5] - joints[13] if side == "right" else joints[13] - joints[5]
            x_aux_norm = float(np.linalg.norm(x_aux))
            if x_aux_norm > 1e-8:
                x_axis = x_aux / x_aux_norm
                coords = np.stack([(joints - root) @ x_axis, -((joints - root) @ z_axis)], axis=1)
                max_abs = float(np.max(np.abs(coords)))
                if max_abs > 1e-8:
                    pts = coords / max_abs * (size * 0.38) + np.asarray([size * 0.5, size * 0.55])
                    for a, b in HAND_EDGES:
                        pa = tuple(np.round(pts[a]).astype(int))
                        pb = tuple(np.round(pts[b]).astype(int))
                        cv2.line(panel, pa, pb, color, 2, cv2.LINE_AA)
                    for idx, point in enumerate(pts):
                        radius = 4 if idx in FINGERTIPS else 3
                        cv2.circle(panel, tuple(np.round(point).astype(int)), radius, color, -1, cv2.LINE_AA)
    draw_text(panel, label, (6, 18), color=(255, 255, 255), scale=0.42)
    frame[y0:y1, x0:x1] = cv2.addWeighted(roi, 0.35, panel, 0.65, 0)


def draw_series(panel, values, y0, height, color, vmin, vmax, label, current_idx):
    h, w = panel.shape[:2]
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        draw_text(panel, f"{label}: no data", (8, y0 + 18), color=color, scale=0.45)
        return
    x = np.linspace(0, w - 1, values.shape[0])
    clipped = np.clip(values, vmin, vmax)
    y = y0 + height - 1 - (clipped - vmin) / max(vmax - vmin, 1e-6) * (height - 1)
    prev = None
    for idx in range(values.shape[0]):
        if not finite[idx]:
            prev = None
            continue
        point = (int(round(x[idx])), int(round(y[idx])))
        if prev is not None:
            cv2.line(panel, prev, point, color, 1, cv2.LINE_AA)
        prev = point
    cv2.rectangle(panel, (0, y0), (w - 1, y0 + height - 1), (80, 80, 80), 1)
    cv2.line(panel, (int(round(x[current_idx])), y0), (int(round(x[current_idx])), y0 + height - 1), (255, 255, 255), 1)
    cur = values[current_idx] if current_idx < values.shape[0] else np.nan
    draw_text(panel, f"{label}: {cur:.3g}", (8, y0 + 18), color=color, scale=0.45)


def make_timeline_panel(metrics, width, current_idx):
    panel = np.zeros((TIMELINE_HEIGHT, width, 3), dtype=np.uint8)
    panel[:] = (25, 25, 28)
    left = metrics["left"]
    right = metrics["right"]
    left_dot_label = "stored-vs-joint palm dot" if left.get("dot_mode") == "stored_vs_geom" else "temporal palm dot"
    right_dot_label = "stored-vs-joint palm dot" if right.get("dot_mode") == "stored_vs_geom" else "temporal palm dot"
    draw_series(panel, left["dot"], 8, 44, (80, 220, 120), -1, 1, f"L {left_dot_label}", current_idx)
    draw_series(panel, right["dot"], 56, 44, (80, 180, 255), -1, 1, f"R {right_dot_label}", current_idx)
    draw_series(panel, left["curl"], 106, 36, (80, 220, 120), -0.1, 0.7, "L MANO curl", current_idx)
    draw_series(panel, right["curl"], 146, 36, (80, 180, 255), -0.1, 0.7, "R MANO curl", current_idx)
    draw_series(panel, left.get("open2d_score", np.full_like(left["curl"], np.nan)), 190, 36, (0, 210, 255), 0, 1, "L 2D open", current_idx)
    draw_series(panel, right.get("open2d_score", np.full_like(right["curl"], np.nan)), 230, 36, (0, 160, 255), 0, 1, "R 2D open", current_idx)
    mismatch = np.maximum(
        left.get("open2d_mano_curl_mismatch", np.zeros_like(left["curl"], dtype=bool)).astype(np.float32),
        right.get("open2d_mano_curl_mismatch", np.zeros_like(right["curl"], dtype=bool)).astype(np.float32),
    )
    draw_series(panel, mismatch, 270, 24, (0, 120, 255), 0, 1, "2D-open/MANO-curl mismatch", current_idx)
    reproj = np.nanmean(np.stack([left["reproj"], right["reproj"]], axis=0), axis=0)
    draw_series(panel, reproj, 300, 32, (180, 180, 255), 0, 100, "mean reproj px", current_idx)
    return panel


def summarize(metrics):
    out = {}
    for side, m in metrics.items():
        dot = m["dot"]
        reproj = m["reproj"]
        curl = m["curl"]
        finite_dot = dot[np.isfinite(dot)]
        finite_err = reproj[np.isfinite(reproj) & (reproj < 1e5)]
        out[side] = {
            "valid_frames": int(np.sum(m["valid"])),
            "bad_frames": int(np.sum(m["bad"])),
            "spike_frames": int(np.sum(m["spike"])),
            "temporal_fixed_frames": int(np.sum(m.get("fixed", 0))),
            "bbox_shrink_frames": int(np.sum(m.get("shrink", 0))),
            "bbox_jump_frames": int(np.sum(m.get("jump", 0))),
            "bbox_overlap_frames": int(np.sum(m.get("overlap", 0))),
            "bbox_merge_fixed_frames": int(np.sum(m.get("merge_fixed", 0))),
            "bbox_size_floor_frames": int(np.sum(m.get("bbox_size_floor", 0))),
            "open2d_visible_frames": int(np.sum(m.get("open2d_visible", 0))),
            "open2d_mano_curl_mismatch_frames": int(np.sum(m.get("open2d_mano_curl_mismatch", 0))),
            "finger_fixed_frames": int(np.sum(m.get("finger_fixed", 0))),
            "curl_spike_frames": int(np.sum(m.get("curl_spike", 0))),
            "finger_low_evidence_frames": int(np.sum(m.get("finger_low_evidence", 0))),
            "open_rescue_frames": int(np.sum(m.get("open_rescue", 0))),
            "open_anchor_frames": int(np.sum(m.get("open_anchor", 0))),
            "finger_rate_limited_frames": int(np.sum(m.get("rate_limited", 0))),
            "joint_rate_limited_frames": int(np.sum(m.get("joint_rate_limited", 0))),
            "hand_size_floor_frames": int(np.sum(m.get("size_floor", 0))),
            "open2d_score_p50_p75_p90": np.percentile(m["open2d_score"][np.isfinite(m["open2d_score"])], [50, 75, 90]).round(4).tolist()
            if "open2d_score" in m and np.any(np.isfinite(m["open2d_score"]))
            else [],
            "palm_dot_p01_p05_p50_p95_p99": np.percentile(finite_dot, [1, 5, 50, 95, 99]).round(5).tolist() if finite_dot.size else [],
            "palm_dot_negative_frames": int(np.sum(finite_dot < 0.0)) if finite_dot.size else 0,
            "palm_dot_strong_disagree_frames": int(np.sum(finite_dot < -0.5)) if finite_dot.size else 0,
            "reproj_p50_p90_p99": np.percentile(finite_err, [50, 90, 99]).round(4).tolist() if finite_err.size else [],
            "curl_min_max_std": [float(np.nanmin(curl)), float(np.nanmax(curl)), float(np.nanstd(curl))],
            "global_speed_p50_p95_max": np.percentile(m["global_speed"], [50, 95, 100]).round(5).tolist(),
        }
    return out


def default_paths_from_clip(clip_dir: Path):
    clip = clip_dir.name
    gvhmr_dir = clip_dir / "gvhmr_out" / clip
    sidecar = clip_dir / "001_smplx_hands.npz"
    mano_params = gvhmr_dir / "mano_params.pt"
    return {
        "video": gvhmr_dir / "valid_video.mp4",
        "vitpose": gvhmr_dir / "vitpose_wholebody.pt",
        "hand_track": sidecar if sidecar.exists() else mano_params,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_dir", type=Path, default=None, help="Pipeline clip dir containing gvhmr_out and 001_smplx_hands.npz")
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--vitpose_wholebody", type=Path, default=None)
    parser.add_argument("--sidecar", type=Path, default=None)
    parser.add_argument("--mano_params", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--person_idx", type=int, default=-1)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--skip", type=int, default=1)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--hand_kpt_conf_thr", type=float, default=0.5)
    parser.add_argument("--hand_kpt_low_conf_thr", type=float, default=0.2)
    parser.add_argument("--hand_kpt_hi_min_keypoints", type=int, default=6)
    parser.add_argument("--hand_min_keypoints", type=int, default=4)
    parser.add_argument("--hand_bbox_min_size", type=float, default=128.0)
    parser.add_argument("--finger_open_score_thr", type=float, default=0.7)
    parser.add_argument("--finger_curl_percentile", type=float, default=55.0)
    parser.add_argument("--finger_evidence_thr", type=float, default=1.0)
    args = parser.parse_args()

    if args.clip_dir:
        paths = default_paths_from_clip(args.clip_dir)
        video_path = args.video or paths["video"]
        vitpose_path = args.vitpose_wholebody or paths["vitpose"]
        hand_track_path = args.sidecar or args.mano_params or paths["hand_track"]
    else:
        hand_arg = args.sidecar or args.mano_params
        if args.video is None or args.vitpose_wholebody is None or hand_arg is None:
            raise ValueError("Provide --clip_dir or all of --video/--vitpose_wholebody plus --sidecar or --mano_params")
        video_path = args.video
        vitpose_path = args.vitpose_wholebody
        hand_track_path = hand_arg

    vitpose = choose_person(load_vitpose(vitpose_path), args.person_idx)
    hand_track, hand_track_type = load_hand_track(hand_track_path)
    frame_count = min(vitpose.shape[0], hand_track_frame_count(hand_track))
    metrics = compute_side_metrics(hand_track, frame_count)
    metrics = add_vitpose_finger_metrics(
        metrics,
        vitpose,
        frame_count,
        low_conf_thr=args.hand_kpt_low_conf_thr,
        min_keypoints=args.hand_min_keypoints,
        open_score_thr=args.finger_open_score_thr,
        curl_percentile=args.finger_curl_percentile,
        evidence_thr=args.finger_evidence_thr,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps = fps / max(1, args.skip)
    source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = args.width / float(source_w)
    main_h = int(round(source_h * scale))
    output_size = (args.width, main_h + TIMELINE_HEIGHT)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, out_fps, output_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {args.output}")

    total = frame_count if args.max_frames <= 0 else min(frame_count, args.max_frames)
    frame_idx = 0
    written = 0
    while frame_idx < total:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % max(1, args.skip) != 0:
            frame_idx += 1
            continue
        frame = cv2.resize(frame, (args.width, main_h), interpolation=cv2.INTER_AREA)
        kpts = vitpose[frame_idx].copy()
        kpts[:, :2] *= scale
        left_kp = kpts[-42:-21]
        right_kp = kpts[-21:]
        left_bbox = right_bbox = None
        left_bbox_valid = right_bbox_valid = False
        if metrics["left"].get("bbox") is not None:
            left_bbox_valid = bool(metrics["left"]["valid"][frame_idx])
            left_bbox = metrics["left"]["bbox"][frame_idx].copy() * scale
        else:
            left_bbox, left_bbox_valid = bbox_from_keypoints(
                left_kp,
                conf_thr=args.hand_kpt_conf_thr,
                low_conf_thr=args.hand_kpt_low_conf_thr,
                hi_min_keypoints=args.hand_kpt_hi_min_keypoints,
                min_keypoints=args.hand_min_keypoints,
                min_size=args.hand_bbox_min_size * scale,
            )
        if metrics["right"].get("bbox") is not None:
            right_bbox_valid = bool(metrics["right"]["valid"][frame_idx])
            right_bbox = metrics["right"]["bbox"][frame_idx].copy() * scale
        else:
            right_bbox, right_bbox_valid = bbox_from_keypoints(
                right_kp,
                conf_thr=args.hand_kpt_conf_thr,
                low_conf_thr=args.hand_kpt_low_conf_thr,
                hi_min_keypoints=args.hand_kpt_hi_min_keypoints,
                min_keypoints=args.hand_min_keypoints,
                min_size=args.hand_bbox_min_size * scale,
            )
        draw_hand_keypoints(frame, left_kp, (60, 220, 80))
        draw_hand_keypoints(frame, right_kp, (60, 170, 255))
        if bool(metrics["left"]["open2d_mano_curl_mismatch"][frame_idx]):
            draw_fingertip_emphasis(frame, left_kp, (0, 210, 255))
        if bool(metrics["right"]["open2d_mano_curl_mismatch"][frame_idx]):
            draw_fingertip_emphasis(frame, right_kp, (0, 160, 255))
        draw_bbox(frame, left_bbox, bbox_color(metrics, "left", frame_idx, (60, 220, 80)))
        draw_bbox(frame, right_bbox, bbox_color(metrics, "right", frame_idx, (60, 170, 255)))

        inset = min(132, max(96, args.width // 7))
        draw_mano_inset(
            frame,
            metrics["left"]["joints"][frame_idx],
            "left",
            (12, 84),
            inset,
            (80, 220, 120),
            f"L MANO curl {metrics['left']['curl'][frame_idx]:.2f}",
        )
        draw_mano_inset(
            frame,
            metrics["right"]["joints"][frame_idx],
            "right",
            (args.width - inset - 12, 84),
            inset,
            (80, 180, 255),
            f"R MANO curl {metrics['right']['curl'][frame_idx]:.2f}",
        )

        y = 24
        for side, color in (("left", (60, 220, 80)), ("right", (60, 170, 255))):
            m = metrics[side]
            flags = frame_flags(metrics, side, frame_idx)
            text = (
                f"{side[0].upper()} valid={int(m['valid'][frame_idx])} "
                f"err={m['reproj'][frame_idx]:.1f} "
                f"dot={m['dot'][frame_idx]:.2f} "
                f"curl={m['curl'][frame_idx]:.2f} "
                f"open2d={m['open2d_score'][frame_idx]:.2f} "
                f"gspeed={m['global_speed'][frame_idx]:.2f}"
            )
            if flags:
                text += " " + ",".join(flags)
            draw_text(frame, text, (12, y), color=color)
            y += 24
        draw_text(frame, f"frame {frame_idx}/{frame_count - 1}", (12, y), color=(255, 255, 255))

        timeline = make_timeline_panel(metrics, args.width, frame_idx)
        writer.write(np.vstack([frame, timeline]))
        written += 1
        frame_idx += 1

    cap.release()
    writer.release()

    summary = summarize(metrics)
    summary.update(
        {
            "video": str(video_path),
            "vitpose_wholebody": str(vitpose_path),
            "hand_track": str(hand_track_path),
            "hand_track_type": hand_track_type,
            "output": str(args.output),
            "written_frames": int(written),
            "source_frames": int(frame_count),
        }
    )
    summary_path = args.summary or args.output.with_suffix(".json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved diagnostic video: {args.output}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
