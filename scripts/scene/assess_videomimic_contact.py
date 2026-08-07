"""Phase-2 evidence report for a VideoMimic scene and its external SMPL body.

This intentionally measures proximity against the *first-round* NKSR mesh.
The second-round mesh is useful for visualization, but its hole filling can
incorrectly seal the volume underneath chairs and tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
import smplx


def _normalise(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-8)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(values.min()),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
    }


def _load_human(path: Path, person_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        root = handle["our_pred_humans_smplx_params"][person_id]
        betas = np.asarray(root["betas"], dtype=np.float32)
        body = np.asarray(root["body_pose"], dtype=np.float32)
        orient = np.asarray(root["global_orient"], dtype=np.float32)
        transl = np.asarray(root["root_transl"], dtype=np.float32)
    return betas, body, orient, transl


def _human_geometry(
    model: torch.nn.Module,
    betas: np.ndarray,
    body: np.ndarray,
    orient: np.ndarray,
    transl: np.ndarray,
    frames: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        result = model(
            betas=torch.as_tensor(betas[frames], dtype=torch.float32),
            body_pose=torch.as_tensor(body[frames], dtype=torch.float32),
            global_orient=torch.as_tensor(orient[frames], dtype=torch.float32),
            transl=torch.as_tensor(transl[frames], dtype=torch.float32).reshape(len(frames), 3),
            pose2rot=False,
            return_verts=True,
        )
    return result.vertices.cpu().numpy(), result.joints.cpu().numpy()


def _persistent(mask: np.ndarray, minimum: int) -> tuple[np.ndarray, list[list[int]]]:
    """Keep only temporally coherent posture detections."""
    retained = np.zeros_like(mask, dtype=bool)
    runs: list[list[int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if bool(value) and start is None:
            start = index
        if start is not None and (not bool(value) or index == len(mask) - 1):
            end = index if bool(value) and index == len(mask) - 1 else index - 1
            if end - start + 1 >= minimum:
                retained[start : end + 1] = True
                runs.append([int(start), int(end)])
            start = None
    return retained, runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrated-h5", required=True, type=Path)
    parser.add_argument("--first-round-mesh", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--person-id", default="1")
    parser.add_argument("--gender", choices=("male", "female", "neutral"), default="neutral")
    parser.add_argument(
        "--gravity-axis", choices=("x", "y", "z"), default="z",
        help="Declared up axis of the gravity-calibrated reconstruction.",
    )
    parser.add_argument("--min-seated-run", type=int, default=4)
    args = parser.parse_args()
    if args.min_seated_run < 1:
        raise ValueError("min-seated-run must be positive")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(args.first_round_mesh, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise RuntimeError(f"invalid first-round mesh: {args.first_round_mesh}")

    betas, body, orient, transl = _load_human(args.calibrated_h5, args.person_id)
    frame_count = int(body.shape[0])
    if not (betas.shape[0] == orient.shape[0] == transl.shape[0] == frame_count):
        raise RuntimeError("calibrated human tensors do not have matching frame counts")

    model = smplx.create(
        str(args.model_root), model_type="smpl", gender=args.gender, num_betas=10, batch_size=frame_count
    )
    all_frames = np.arange(frame_count, dtype=np.int64)
    vertices, joints = _human_geometry(model, betas, body, orient, transl, all_frames)

    # A scene can be a thin frontal crop or a tall room: its bounding-box shape
    # is not gravity evidence. VideoMimic's calibrated coordinate contract is
    # supplied explicitly instead of guessing from the reconstructed mesh.
    vertical_axis = "xyz".index(args.gravity_axis)
    hips = joints[:, [1, 2]].mean(axis=1)
    knees = joints[:, [4, 5]].mean(axis=1)
    ankles = joints[:, [7, 8]].mean(axis=1)
    thigh = _normalise(knees - hips)
    shin = _normalise(ankles - knees)
    horizontal = 1.0 - np.abs(thigh[:, vertical_axis])
    vertical = np.abs(shin[:, vertical_axis])
    seated_score = 0.5 * (horizontal + vertical)
    seated_raw = (horizontal >= 0.70) & (vertical >= 0.65)
    seated_mask, seated_runs = _persistent(seated_raw, args.min_seated_run)

    tree = cKDTree(np.asarray(mesh.vertices))
    distances, _ = tree.query(vertices.reshape(-1, 3), k=1, workers=-1)
    distances = distances.reshape(frame_count, -1)
    proximity = [_quantiles(row) for row in distances]

    ranked = np.argsort(seated_score)[::-1]
    selected: list[int] = [0]
    for index in ranked:
        frame = int(index)
        if frame not in selected and all(abs(frame - prior) >= 10 for prior in selected):
            selected.append(frame)
        if len(selected) == 4:
            break
    if frame_count - 1 not in selected:
        selected.append(frame_count - 1)
    selected = sorted(selected)

    colors = np.array(
        [[220, 55, 55, 220], [240, 180, 45, 220], [55, 180, 90, 220], [70, 135, 230, 220], [180, 70, 190, 220]],
        dtype=np.uint8,
    )
    overlay = trimesh.Scene()
    background = mesh.copy()
    background.visual.face_colors = np.tile(np.array([[195, 200, 210, 255]], dtype=np.uint8), (len(background.faces), 1))
    overlay.add_geometry(background, node_name="nksr_first_round_mesh")
    for color, frame in zip(colors, selected):
        human = trimesh.Trimesh(vertices=vertices[frame], faces=model.faces, process=False)
        human.visual.face_colors = np.tile(color[None], (len(human.faces), 1))
        overlay.add_geometry(human, node_name=f"human_frame_{frame:05d}")
    overlay_path = output / "first_round_mesh_plus_human.glb"
    overlay.export(overlay_path)

    report = {
        "schema_version": 1,
        "mesh_role": "first_round_nksr_only",
        "mesh_path": str(args.first_round_mesh.resolve()),
        "calibrated_h5": str(args.calibrated_h5.resolve()),
        "frame_count": frame_count,
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "inferred_vertical_axis": args.gravity_axis,
        "gravity_axis_source": "videomimic_gravity_calibrated_coordinate_contract",
        "mesh_extent_axis_inference": "disabled",
        "warning": "nearest_vertex_distance is a conservative diagnostic, not signed mesh penetration",
        "raw_seated_frame_count": int(seated_raw.sum()),
        "seated_frame_count": int(seated_mask.sum()),
        "seated_evidence_runs": seated_runs,
        "min_seated_run": int(args.min_seated_run),
        "selected_overlay_frames": selected,
        "frames": [
            {
                "frame": int(index),
                "hip_height_axis": float(hips[index, vertical_axis]),
                "thigh_horizontal_score": float(horizontal[index]),
                "shin_vertical_score": float(vertical[index]),
                "seated_score": float(seated_score[index]),
                "seated_candidate_raw": bool(seated_raw[index]),
                "seated_candidate": bool(seated_mask[index]),
                "nearest_vertex_distance": proximity[index],
            }
            for index in range(frame_count)
        ],
        "artifacts": {
            "human_overlay_glb": str(overlay_path),
        },
    }
    report_path = output / "contact_evidence.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[PASS] contact evidence: {frame_count} frames, {int(seated_mask.sum())} seated candidates -> {report_path}")
    print(f"[PASS] first-round mesh + human overlay -> {overlay_path}")


if __name__ == "__main__":
    main()
