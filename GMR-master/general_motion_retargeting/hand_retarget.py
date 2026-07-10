"""Finger-joint retargeting from SMPL-X MANO hand poses to robot revolute joints.

This module directly reads the raw 45-D MANO axis-angle hand pose (as stored in
the sidecar NPZ) and projects each joint's rotation onto the corresponding robot
revolute joint axis.

MANO hand pose ordering (15 joints × 3 DoF = 45-D):
  0: thumb1 (CMC),   1: thumb2 (MCP),   2: thumb3 (IP)
  3: index1 (MCP),   4: index2 (PIP),   5: index3 (DIP)
  6: middle1 (MCP),  7: middle2 (PIP),  8: middle3 (DIP)
  9: ring1 (MCP),   10: ring2 (PIP),   11: ring3 (DIP)
 12: pinky1 (MCP),  13: pinky2 (PIP),  14: pinky3 (DIP)

Robot hand DoF indices are auto-detected from the MuJoCo model.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# ---------------------------------------------------------------------------
# Per-robot hand mapping specification
# ---------------------------------------------------------------------------
# Each entry maps a robot joint name to:
#   mano_idx   – index (0-14) into the 45-D MANO hand pose vector
#   axis       – "x" (0), "y" (1), or "z" (2) — which component to extract
#   sign       – +1 or -1 (flip direction)

G1_HAND_SPEC: Dict[str, Dict] = {
    # Left hand (7 joints)
    "left_hand_thumb_0_joint":  {"mano_idx": 0, "axis": "y", "sign": -1},
    "left_hand_thumb_1_joint":  {"mano_idx": 1, "axis": "z", "sign": -1},
    "left_hand_thumb_2_joint":  {"mano_idx": 2, "axis": "z", "sign": -1},
    "left_hand_index_0_joint":  {"mano_idx": 3, "axis": "z", "sign":  1},
    "left_hand_index_1_joint":  {"mano_idx": 4, "axis": "z", "sign":  1},
    "left_hand_middle_0_joint": {"mano_idx": 6, "axis": "z", "sign":  1},
    "left_hand_middle_1_joint": {"mano_idx": 7, "axis": "z", "sign":  1},
    # Right hand (7 joints)
    "right_hand_thumb_0_joint":  {"mano_idx": 0, "axis": "y", "sign": -1},
    "right_hand_thumb_1_joint":  {"mano_idx": 1, "axis": "z", "sign": -1},
    "right_hand_thumb_2_joint":  {"mano_idx": 2, "axis": "z", "sign": -1},
    "right_hand_index_0_joint":  {"mano_idx": 3, "axis": "z", "sign":  1},
    "right_hand_index_1_joint":  {"mano_idx": 4, "axis": "z", "sign":  1},
    "right_hand_middle_0_joint": {"mano_idx": 6, "axis": "z", "sign":  1},
    "right_hand_middle_1_joint": {"mano_idx": 7, "axis": "z", "sign":  1},
}

H1_WITH_HAND_SPEC: Dict[str, Dict] = {
    # Left hand: H1 finger joints use positive qpos for curl, while MANO left
    # index/middle/ring curl is negative in the raw axis-angle z component.
    "L_thumb_proximal_yaw_joint":   {"mano_idx": 0,  "axis": "y", "sign":  1},
    "L_thumb_proximal_pitch_joint": {"mano_idx": 1,  "axis": "z", "sign": -1},
    "L_thumb_intermediate_joint":   {"mano_idx": 1,  "axis": "z", "sign": -1},
    "L_thumb_distal_joint":         {"mano_idx": 2,  "axis": "z", "sign": -1},
    "L_index_proximal_joint":       {"mano_idx": 3,  "axis": "z", "sign": -1},
    "L_index_intermediate_joint":   {"mano_idx": 4,  "axis": "z", "sign": -1},
    "L_middle_proximal_joint":      {"mano_idx": 6,  "axis": "z", "sign": -1},
    "L_middle_intermediate_joint":  {"mano_idx": 7,  "axis": "z", "sign": -1},
    "L_ring_proximal_joint":        {"mano_idx": 9,  "axis": "z", "sign": -1},
    "L_ring_intermediate_joint":    {"mano_idx": 10, "axis": "z", "sign": -1},
    "L_pinky_proximal_joint":       {"mano_idx": 12, "axis": "z", "sign":  1},
    "L_pinky_intermediate_joint":   {"mano_idx": 13, "axis": "z", "sign": -1},
    # Right hand.
    "R_thumb_proximal_yaw_joint":   {"mano_idx": 0,  "axis": "y", "sign": -1},
    "R_thumb_proximal_pitch_joint": {"mano_idx": 1,  "axis": "z", "sign":  1},
    "R_thumb_intermediate_joint":   {"mano_idx": 1,  "axis": "z", "sign":  1},
    "R_thumb_distal_joint":         {"mano_idx": 2,  "axis": "z", "sign":  1},
    "R_index_proximal_joint":       {"mano_idx": 3,  "axis": "z", "sign":  1},
    "R_index_intermediate_joint":   {"mano_idx": 4,  "axis": "z", "sign":  1},
    "R_middle_proximal_joint":      {"mano_idx": 6,  "axis": "z", "sign":  1},
    "R_middle_intermediate_joint":  {"mano_idx": 7,  "axis": "z", "sign":  1},
    "R_ring_proximal_joint":        {"mano_idx": 9,  "axis": "z", "sign":  1},
    "R_ring_intermediate_joint":    {"mano_idx": 10, "axis": "z", "sign":  1},
    "R_pinky_proximal_joint":       {"mano_idx": 12, "axis": "z", "sign": -1},
    "R_pinky_intermediate_joint":   {"mano_idx": 13, "axis": "z", "sign":  1},
}

ROBOT_HAND_SPEC = {
    "unitree_g1": G1_HAND_SPEC,
    "unitree_g1_with_hands": G1_HAND_SPEC,
    "unitree_h1_with_hand": H1_WITH_HAND_SPEC,
    "unitree_h1_with_hand_wrist": H1_WITH_HAND_SPEC,
}

AXIS_IDX = {"x": 0, "y": 1, "z": 2}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_hand_dof_indices(model, joint_names=None) -> Dict[str, int]:
    """Return {joint_name: dof_index_in_dof_pos} for hand joints.

    ``dof_index`` is the column index in the ``dof_pos`` array returned by
    ``smpl_npz_to_robot_headless`` (i.e. qpos[7:]).
    """
    import mujoco as mj

    indices: Dict[str, int] = {}
    requested = set(joint_names or [])
    qpos_to_dof = {}
    for dof_idx in range(model.nv):
        jnt_id = model.dof_jntid[dof_idx]
        qpos_adr = model.jnt_qposadr[jnt_id]
        if qpos_adr >= 7:  # skip the free joint (qpos 0–6)
            qpos_to_dof[qpos_adr] = qpos_adr - 7

    for jnt_id in range(model.njnt):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jnt_id)
        if name and ("hand" in name.lower() or name in requested):
            qpos_adr = model.jnt_qposadr[jnt_id]
            if qpos_adr in qpos_to_dof:
                indices[name] = qpos_to_dof[qpos_adr]
    return indices


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _as_hand_pose(name: str, pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.ndim == 2 and pose.shape[1] == 45:
        return pose.reshape(-1, 15, 3)
    if pose.ndim == 3 and pose.shape[1:] == (15, 3):
        return pose
    raise ValueError(f"{name} must have shape (T,45) or (T,15,3), got {pose.shape}")


def _normalize_valid(valid: Optional[np.ndarray], frame_count: int) -> np.ndarray:
    if valid is None:
        return np.ones(frame_count, dtype=bool)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    if valid.shape[0] == frame_count:
        return valid
    if valid.shape[0] > frame_count:
        return valid[:frame_count]
    padded = np.zeros(frame_count, dtype=bool)
    padded[: valid.shape[0]] = valid
    return padded


def _resample_pose_and_valid(
    pose: np.ndarray,
    valid: Optional[np.ndarray],
    frame_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    pose3 = _as_hand_pose("hand_pose", pose)
    src_frames = pose3.shape[0]
    valid_arr = _normalize_valid(valid, src_frames)

    if frame_count <= 0:
        return np.zeros((0, 45), dtype=np.float32), np.zeros(0, dtype=bool)
    if src_frames == frame_count:
        return pose3.reshape(frame_count, 45).astype(np.float32), valid_arr.astype(bool)
    if src_frames == 0:
        return np.zeros((frame_count, 45), dtype=np.float32), np.zeros(frame_count, dtype=bool)

    source_t = np.arange(src_frames, dtype=np.float32)
    target_t = np.linspace(0, src_frames - 1, frame_count, dtype=np.float32)
    flat = pose3.reshape(src_frames, -1)
    resampled = np.stack(
        [np.interp(target_t, source_t, flat[:, col]) for col in range(flat.shape[1])],
        axis=1,
    ).astype(np.float32)

    nearest = np.rint(target_t).astype(np.int64)
    nearest = np.clip(nearest, 0, src_frames - 1)
    return resampled, valid_arr[nearest].astype(bool)


def _hold_last_valid(pose: np.ndarray, valid: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    valid = _normalize_valid(valid, pose.shape[0])
    out = np.zeros_like(pose)
    last = np.zeros(pose.shape[1], dtype=np.float32)
    for idx in range(pose.shape[0]):
        if valid[idx]:
            last = pose[idx]
        out[idx] = last
    return out


def _interpolate_short_invalid(pose: np.ndarray, valid: np.ndarray, max_gap: int = 15) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    valid = _normalize_valid(valid, pose.shape[0])
    if pose.shape[0] == 0:
        return pose.copy()
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        return np.zeros_like(pose)

    out = _hold_last_valid(pose, valid)
    first = int(valid_idx[0])
    last = int(valid_idx[-1])
    out[:first] = pose[first]
    out[last + 1 :] = pose[last]

    max_gap = int(max_gap)
    for left_idx, right_idx in zip(valid_idx[:-1], valid_idx[1:]):
        gap = int(right_idx - left_idx - 1)
        if gap <= 0:
            continue
        if max_gap > 0 and gap > max_gap:
            continue
        alpha = np.linspace(0.0, 1.0, gap + 2, dtype=np.float32)[1:-1, None]
        out[left_idx + 1 : right_idx] = (1.0 - alpha) * pose[left_idx] + alpha * pose[right_idx]
    return out


# ---------------------------------------------------------------------------
# Velocity-capped spike detection (ported from Do As I Do's process_dataset.py)
# ---------------------------------------------------------------------------
# Iteratively detects per-joint angular-velocity spikes using a sliding-window
# median + MAD threshold, capped by a fixed velocity limit. Gaps ≤ 1 frame are
# merged; bursts > 10 frames are preserved (presumed real motion). The mask is
# OR-combined across all 15 MANO joints so a spike in any finger flags the
# whole hand pose. Tracker-invalid frames seed the mask and are NEVER
# un-flagged by the max_burst heuristic.

_SPIKE_CFG = dict(k_mad=8.0, v_cap=0.40, window=31)  # rad/frame angular velocity
_SPIKE_MAX_BURST = 10
_SPIKE_GAP_MERGE = 1


def _runs(mask: np.ndarray) -> list:
    """Return [(start, end_exclusive), ...] for True runs in a 1-D bool mask."""
    n = len(mask)
    out = []
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((int(i), int(j)))
            i = j
        else:
            i += 1
    return out


def _post_process_spike_mask(
    bad: np.ndarray,
    gap_merge: int = _SPIKE_GAP_MERGE,
    max_burst: int = _SPIKE_MAX_BURST,
) -> np.ndarray:
    """Merge short good-runs between bad-runs, then unmask long bursts."""
    out = bad.copy()
    n = len(out)
    if not out.any():
        return out
    i = 0
    while i < n:
        if not out[i]:
            j = i
            while j < n and not out[j]:
                j += 1
            if (j - i) <= gap_merge:
                out[i:j] = True
            i = j
        else:
            i += 1
    for s, e in _runs(out):
        if e - s > max_burst:
            out[s:e] = False
    return out


def _rolling_median_mad(v: np.ndarray, window: int) -> tuple:
    """Per-edge median and MAD over a centered window, clipped at the ends."""
    n = len(v)
    half = window // 2
    med = np.empty(n)
    mad = np.empty(n)
    for i in range(n):
        w = v[max(0, i - half): min(n, i + half + 1)]
        m = np.median(w)
        med[i] = m
        mad[i] = np.median(np.abs(w - m))
    return med, mad


def _angular_velocity(rotvec: np.ndarray) -> np.ndarray:
    """Angular magnitude between consecutive rotvec frames (N, 3)."""
    R_rel = Rotation.from_rotvec(rotvec[1:]) * Rotation.from_rotvec(rotvec[:-1]).inv()
    return R_rel.magnitude()


def _edge_to_frame_mask(edge_bad: np.ndarray, n: int) -> np.ndarray:
    """Convert edge mask to frame mask: a frame is bad if either edge is bad."""
    out = np.zeros(n, dtype=bool)
    out[:-1] |= edge_bad
    out[1:] |= edge_bad
    return out


def _slerp_interpolate_rotvec(rotvec: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """SLERP-interpolate masked frames in (N, 3) axis-angle array."""
    if not bad.any():
        return rotvec
    out = rotvec.astype(np.float64, copy=True)
    good_idx = np.where(~bad)[0]
    if len(good_idx) == 0:
        return out
    if len(good_idx) == 1:
        out[bad] = rotvec[good_idx[0]]
        return out
    R_good = Rotation.from_rotvec(out[good_idx])
    slerp = Slerp(good_idx, R_good)
    bad_idx = np.where(bad)[0]
    in_range = (bad_idx >= good_idx[0]) & (bad_idx <= good_idx[-1])
    if in_range.any():
        out[bad_idx[in_range]] = slerp(bad_idx[in_range]).as_rotvec()
    for i in bad_idx[~in_range]:
        nearest = good_idx[0] if i < good_idx[0] else good_idx[-1]
        out[i] = rotvec[nearest]
    return out


def _detect_velocity_spikes_rotvec(
    rotvec: np.ndarray,   # (T, 15, 3)
    valid: np.ndarray,    # (T,) bool
    cfg: dict = None,
) -> np.ndarray:
    """Detect velocity-spike frames in a 15-joint rotvec hand pose.

    Returns a boolean mask of shape (T,) where True = spike frame.
    Tracker-invalid frames are NOT included in the return (they are handled
    separately by the caller).
    """
    if cfg is None:
        cfg = _SPIKE_CFG
    T = rotvec.shape[0]
    if T < 3:
        return np.zeros(T, dtype=bool)

    seed = ~valid.astype(bool)
    bad = seed.copy()

    for _ in range(2):
        # Interpolate currently-bad frames before computing velocities
        work = rotvec.copy()
        if bad.any():
            for j in range(rotvec.shape[1]):  # per-joint
                work[:, j, :] = _slerp_interpolate_rotvec(work[:, j, :], bad)

        # Compute angular velocity magnitude per joint, then max across joints
        vel_max = np.zeros(T - 1)
        for j in range(rotvec.shape[1]):
            v = _angular_velocity(work[:, j, :])
            vel_max = np.maximum(vel_max, v)

        med, mad = _rolling_median_mad(vel_max, cfg["window"])
        thresh = np.minimum(
            med + cfg["k_mad"] * np.maximum(mad, 1e-10),
            cfg["v_cap"],
        )
        vel_bad = _edge_to_frame_mask(vel_max > thresh, T)
        new_bad = bad | vel_bad
        if np.array_equal(new_bad, bad):
            break
        bad = new_bad

    # Return only velocity-spike frames (exclude tracker-invalid seed)
    return bad & ~seed


def clean_hand_pose_spikes(
    hand_pose: np.ndarray,   # (T, 45) or (T, 15, 3)
    valid: np.ndarray,       # (T,) bool from tracker
    cfg: dict = None,
    gap_merge: int = _SPIKE_GAP_MERGE,
    max_burst: int = _SPIKE_MAX_BURST,
) -> np.ndarray:
    """Velocity-capped spike cleaning for a single hand's MANO pose.

    Returns cleaned (T, 45) hand pose with spike frames SLERP-interpolated.
    Tracker-invalid frames (``valid=False``) are also interpolated — the
    caller should apply ``prepare_hand_poses`` afterward for hold/interp/zero
    handling.
    """
    pose3 = _as_hand_pose("hand_pose", hand_pose)  # (T, 15, 3)
    T = pose3.shape[0]
    valid_bool = _normalize_valid(valid, T)

    # 1. Detect velocity spikes (excludes tracker-invalid from return)
    spike_bad = _detect_velocity_spikes_rotvec(pose3, valid_bool, cfg)

    # 2. Post-process: merge short gaps, preserve long bursts
    spike_bad = _post_process_spike_mask(spike_bad, gap_merge, max_burst)

    # 3. Combine with tracker-invalid (these are always interpolated)
    shared_bad = spike_bad | ~valid_bool

    # 4. SLERP-interpolate all bad frames per joint
    if shared_bad.any():
        for j in range(15):
            pose3[:, j, :] = _slerp_interpolate_rotvec(pose3[:, j, :], shared_bad)

    return pose3.reshape(T, 45).astype(np.float32)


def prepare_hand_poses(
    left_hand_pose: np.ndarray,
    right_hand_pose: np.ndarray,
    left_valid: Optional[np.ndarray],
    right_valid: Optional[np.ndarray],
    frame_count: int,
    invalid_mode: str = "hold",
    interp_max_gap: int = 15,
    clean_spikes: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample raw sidecar hand poses to body frames and apply valid masks.

    When ``clean_spikes=True`` (default), velocity-capped spike detection
    (ported from Do As I Do) is run on the raw MANO poses before the
    invalid-mode logic.  This removes isolated angular-velocity glitches
    while preserving genuine fast motions (bursts > 10 frames).

    ``invalid_mode="hold"`` keeps the last valid hand pose and uses zero pose
    before the first valid frame. This avoids sudden all-zero finger snaps when
    HaMeR marks a hand frame invalid.
    """
    left, left_valid_out = _resample_pose_and_valid(left_hand_pose, left_valid, frame_count)
    right, right_valid_out = _resample_pose_and_valid(right_hand_pose, right_valid, frame_count)

    if clean_spikes:
        left = clean_hand_pose_spikes(left, left_valid_out)
        right = clean_hand_pose_spikes(right, right_valid_out)

    if invalid_mode == "hold":
        left = _hold_last_valid(left, left_valid_out)
        right = _hold_last_valid(right, right_valid_out)
    elif invalid_mode == "interp":
        left = _interpolate_short_invalid(left, left_valid_out, interp_max_gap)
        right = _interpolate_short_invalid(right, right_valid_out, interp_max_gap)
    elif invalid_mode == "zero":
        left = left.copy()
        right = right.copy()
        left[~left_valid_out] = 0.0
        right[~right_valid_out] = 0.0
    elif invalid_mode != "raw":
        raise ValueError(f"Unsupported invalid_mode={invalid_mode!r}")

    return left.astype(np.float32), right.astype(np.float32), left_valid_out, right_valid_out

def amplify_curl_power(
    angles: np.ndarray,
    power: float = 0.5,
) -> np.ndarray:
    """Power-law amplification that boosts small angles more than large ones.

    ``y = sign(x) * |x|^power``

    With ``power=0.5`` (square root)::

        0.04 rad → 0.20 rad   (5× boost)
        0.09 rad → 0.30 rad   (3.3×)
        0.50 rad → 0.71 rad   (1.4×)
        1.00 rad → 1.00 rad   (unchanged)

    When *power* is 1.0 the function is a no-op.  Values in (0, 1) give
    stronger amplification for tiny curl signals, which is exactly what
    conservative HaMeR hand predictions need.
    """
    power = float(power)
    if abs(power - 1.0) < 1e-8:
        return np.asarray(angles, dtype=np.float32).copy()
    return np.sign(angles) * np.power(np.maximum(np.abs(angles), 0.0), power)


def compute_finger_angles_from_raw(
    left_hand_pose: np.ndarray,   # (T, 45)
    right_hand_pose: np.ndarray,  # (T, 45)
    hand_spec: Dict[str, Dict],
    curl_power: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Compute per-joint finger angles directly from raw MANO hand poses.

    Parameters
    ----------
    left_hand_pose : (T, 45) raw left-hand axis-angle.
    right_hand_pose : (T, 45) raw right-hand axis-angle.
    hand_spec : per-robot hand mapping (e.g. ``G1_HAND_SPEC``).
    curl_power : power-law exponent for small-angle amplification.
       1.0 = linear (no amplification), 0.5 = sqrt boost.

    Returns
    -------
    angles : dict  {robot_joint_name: ndarray (T,) in radians}
    """
    left = _as_hand_pose("left_hand_pose", left_hand_pose)
    right = _as_hand_pose("right_hand_pose", right_hand_pose)

    result: Dict[str, np.ndarray] = {}
    for joint_name, spec in hand_spec.items():
        mano_idx = spec["mano_idx"]
        axis_col = AXIS_IDX[spec["axis"]]
        sign = float(spec.get("sign", 1.0))

        # Determine which hand this joint belongs to. G1 uses right_* while
        # H1's five-finger hand uses R_* / L_* joint names.
        is_right = joint_name.startswith("right_") or joint_name.startswith("R_")
        hand_data = right if is_right else left

        # Extract the axis component directly from the raw axis-angle
        angles = sign * hand_data[:, mano_idx, axis_col].copy()
        if abs(float(curl_power) - 1.0) > 1e-8:
            angles = amplify_curl_power(angles, power=curl_power)
        result[joint_name] = angles.astype(np.float32)

    return result


def clamp_angles(
    angles: Dict[str, np.ndarray],
    model,
) -> Dict[str, np.ndarray]:
    """Clamp per-joint angles to the MuJoCo model's joint limits (in-place)."""
    import mujoco as mj

    for jnt_id in range(model.njnt):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jnt_id)
        if name not in angles:
            continue
        jnt_range = model.jnt_range[jnt_id]
        if jnt_range is not None:
            lo, hi = float(jnt_range[0]), float(jnt_range[1])
            np.clip(angles[name], lo, hi, out=angles[name])
    return angles


def apply_finger_angles_to_dof_pos(
    dof_pos: np.ndarray,                # (T, D)
    angles: Dict[str, np.ndarray],      # {joint_name: (T,)}
    model,
) -> np.ndarray:
    """Overwrite hand-joint columns of *dof_pos* with computed finger angles.

    Returns a **copy** of ``dof_pos`` (the input is not mutated).
    """
    dof_pos = np.asarray(dof_pos, dtype=np.float32).copy()
    hand_indices = _get_hand_dof_indices(model, angles.keys())

    for joint_name, ang in angles.items():
        if joint_name not in hand_indices:
            continue
        col = hand_indices[joint_name]
        ang_flat = np.asarray(ang, dtype=np.float32).reshape(-1)
        n = min(dof_pos.shape[0], ang_flat.shape[0])
        dof_pos[:n, col] = ang_flat[:n]
    return dof_pos


def get_hand_dof_names(model) -> List[str]:
    """Return ordered list of hand joint names (for debugging / reporting)."""
    indices = _get_hand_dof_indices(model)
    return sorted(indices, key=indices.get)


def has_hand_joints(model, joint_names=None) -> bool:
    """Return True if the MuJoCo model contains any hand-like joints."""
    import mujoco as mj

    requested = set(joint_names or [])
    for jnt_id in range(model.njnt):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jnt_id)
        if name and ("hand" in name.lower() or name in requested):
            return True
    return False


def get_hand_spec(robot_name: str) -> Optional[Dict[str, Dict]]:
    """Return the per-robot hand mapping spec, or None if unsupported."""
    return ROBOT_HAND_SPEC.get(robot_name)
