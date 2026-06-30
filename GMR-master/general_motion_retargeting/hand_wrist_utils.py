"""Geometric wrist orientation computation from hand keypoint positions.

Ported from the "Do As I Do" retargeting pipeline
(``process_dataset.py``), which uses an MCP-based geometric frame
rather than trusting the MANO ``global_orient`` directly.

The geometric frame is:
  z  = wrist -> middle_MCP
  y  = ring_MCP -> index_MCP (right hand; reversed for left)
  x  = cross(y_aux, z),  y = cross(z, x)

A fixed rotation offset  R_offset = R_mano⁻¹ @ R_geom  is averaged
across all valid frames via quaternion eigendecomposition and then
applied to every frame, denoising the per-frame jitter.

All ``*_from_joints`` functions accept a ``joint_names`` list so that
indices are resolved correctly for SMPL-H (52 joints) and SMPL-X (127
joints) alike.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

# Fingertip landmark indices in the 21-joint OpenPose / HaMeR convention.
FINGERTIP_JOINT_IDX_21 = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_index(name: str, joint_names: List[str]) -> int:
    """Return the int index of *name* in *joint_names*.

    Raises ``ValueError`` if the name is not found (e.g. SMPL-H model
    which lacks the extra face/body landmarks of SMPL-X).
    """
    try:
        return joint_names.index(name)
    except ValueError:
        raise ValueError(
            f"Joint '{name}' not found in joint_names ({len(joint_names)} entries); "
            f"are you using the right model_type?"
        ) from None


def _norm(v: np.ndarray) -> np.ndarray:
    """Normalise the last axis of *v* in-place-safe."""
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Geometric wrist frame (ported from Do As I Do's ``compute_wrist_rotation``)
# ---------------------------------------------------------------------------

def compute_wrist_rotation(
    wrist: np.ndarray,
    middle_mcp: np.ndarray,
    index_mcp: np.ndarray,
    ring_mcp: np.ndarray,
) -> Rotation:
    """Build a right-handed coordinate frame from hand-joint positions.

    Parameters
    ----------
    wrist : (..., 3)
    middle_mcp : (..., 3)
    index_mcp : (..., 3)
    ring_mcp : (..., 3)

    Returns
    -------
    Rotation
        Orthonormal frame with columns (x, y, z), where:
      z  = wrist → middle_mcp
      y  = ring_mcp → index_mcp (right hand; reverse for left)
      x  = cross(y_aux, z),  y = cross(z, x)
    """
    z = _norm(middle_mcp - wrist)                  # wrist → middle MCP
    y_aux = _norm(index_mcp - ring_mcp)             # ring MCP → index MCP
    x = _norm(np.cross(y_aux, z))
    y = _norm(np.cross(z, x))
    return Rotation.from_matrix(np.stack([x, y, z], axis=-1))


def compute_wrist_rotation_from_joints(
    joints: np.ndarray,          # (T, J, 3)
    joint_names: List[str],
    is_right: bool = True,
) -> Rotation:
    """Extract hand MCP positions from a SMPL joint array and compute the
    geometric wrist frame.

    Works for SMPL-H (52 joints) and SMPL-X (127 joints) — indices are
    resolved from *joint_names* at runtime.
    """
    side = "right" if is_right else "left"
    w_idx = _resolve_index(f"{side}_wrist", joint_names)
    m_idx = _resolve_index(f"{side}_middle1", joint_names)
    i_idx = _resolve_index(f"{side}_index1", joint_names)
    r_idx = _resolve_index(f"{side}_ring1", joint_names)

    wrist = joints[..., w_idx, :]
    middle_mcp = joints[..., m_idx, :]
    index_mcp = joints[..., i_idx, :]
    ring_mcp = joints[..., r_idx, :]

    # For left hand, swap index ↔ ring so y points the correct way
    if not is_right:
        return compute_wrist_rotation(
            wrist=wrist,
            middle_mcp=middle_mcp,
            index_mcp=ring_mcp,
            ring_mcp=index_mcp,
        )
    return compute_wrist_rotation(
        wrist=wrist,
        middle_mcp=middle_mcp,
        index_mcp=index_mcp,
        ring_mcp=ring_mcp,
    )


def compute_wrist_rotation_from_21(
    joints_21: np.ndarray,        # (..., 21, 3) – OpenPose/HaMeR hand keypoints
    is_right: bool = True,
) -> Rotation:
    """Variant of :func:`compute_wrist_rotation` using 21-joint format.

    Joint indices (0-based):
      0 = wrist,  5 = index MCP,  9 = middle MCP,  13 = ring MCP
    """
    wrist = joints_21[..., 0, :]
    middle_mcp = joints_21[..., 9, :]
    index_mcp = joints_21[..., 5, :]
    ring_mcp = joints_21[..., 13, :]
    if not is_right:
        return compute_wrist_rotation(
            wrist=wrist,
            middle_mcp=middle_mcp,
            index_mcp=ring_mcp,
            ring_mcp=index_mcp,
        )
    return compute_wrist_rotation(
        wrist=wrist,
        middle_mcp=middle_mcp,
        index_mcp=index_mcp,
        ring_mcp=ring_mcp,
    )


# ---------------------------------------------------------------------------
# Frame offset (R_offset) – invariant under finger articulation
# ---------------------------------------------------------------------------

def _quaternion_mean(quat_xyzw: np.ndarray) -> Rotation:
    """Mean quaternion via eigendecomposition of the outer-product matrix.

    *quat_xyzw*: (N, 4) with columns (x, y, z, w).
    """
    q = quat_xyzw.copy()
    q = q * np.sign(q @ q[0])[:, None]   # align signs
    _, eigvecs = np.linalg.eigh(q.T @ q)
    return Rotation.from_quat(eigvecs[:, -1])


def compute_wrist_frame_offset(
    joints: np.ndarray,              # (T, J, 3) or (T, 21, 3)
    mano_global_orient: np.ndarray,  # (T, 3) – MANO global_orient rotvec
    is_right: bool = True,
    joint_names: Optional[List[str]] = None,
    joint_format: str = "smpl",
) -> Rotation:
    """Compute a fixed :math:`R_{offset} = R_{mano}^{-1} @ R_{geom}`.

    MCP joints are fixed in the MANO wrist frame (roots of finger chains,
    unaffected by ``hand_pose``), so ``R_offset`` is frame-invariant by
    construction.  We average across all valid frames via quaternion mean
    to denoise joint-position noise.

    Parameters
    ----------
    joints : (T, J, 3) or (T, 21, 3)
    mano_global_orient : (T, 3)
    is_right : bool
    joint_names : list of str, required when ``joint_format="smpl"``
    joint_format : ``"smpl"`` (SMPL-H/X with *joint_names*) or ``"21"``
    """
    if joint_format == "smpl":
        if joint_names is None:
            raise ValueError("joint_names is required for joint_format='smpl'")
        R_geom = compute_wrist_rotation_from_joints(joints, joint_names, is_right=is_right)
    elif joint_format == "21":
        R_geom = compute_wrist_rotation_from_21(joints, is_right=is_right)
    else:
        raise ValueError(f"Unknown joint_format={joint_format!r}")

    R_mano = Rotation.from_rotvec(mano_global_orient)
    q_offset = (R_mano.inv() * R_geom).as_quat()  # (N, 4) xyzw
    return _quaternion_mean(q_offset)


# ---------------------------------------------------------------------------
# Fingertip extraction
# ---------------------------------------------------------------------------

def extract_fingertips_from_joints(
    joints: np.ndarray,          # (T, J, 3)
    joint_names: List[str],
    is_right: bool = True,
) -> np.ndarray:
    """Return (T, 5, 3) fingertip positions from a SMPL joint array."""
    side = "right" if is_right else "left"
    tip_names = ["thumb", "index", "middle", "ring", "pinky"]
    indices = [_resolve_index(f"{side}_{t}", joint_names) for t in tip_names]
    return joints[..., indices, :]


def extract_fingertips_from_21(
    joints_21: np.ndarray,       # (T, 21, 3)
) -> np.ndarray:
    """Return (T, 5, 3) fingertip positions from 21-joint hand keypoints."""
    return joints_21[..., FINGERTIP_JOINT_IDX_21, :]


# ---------------------------------------------------------------------------
# Convenience: axis-angle ↔ quaternion
# ---------------------------------------------------------------------------

def rotvec_to_quat_wxyz(rotvec: np.ndarray) -> np.ndarray:
    """Convert axis-angle rotvec (..., 3) to wxyz quaternion (..., 4)."""
    q = Rotation.from_rotvec(rotvec).as_quat()   # xyzw
    return q[..., [3, 0, 1, 2]]                   # wxyz
