import argparse
import concurrent.futures
import gc
import hashlib
import json
import os
import os.path as osp
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data_process.convert_zitai_to_phc import ROBOT_CFG, _find_clip_dirs, convert_clip
from scripts.data_process.export_repaired_smpl_npz import (
    _resample_linear,
    convert_sequence,
    evaluate_speed_check,
    match_reference_timing,
)
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot


def hydra_str(value):
    return json.dumps(str(value), ensure_ascii=False)


def sanitize_name(name, max_len=80):
    sanitized = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")
    if not sanitized:
        sanitized = "repair"
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]
    suffix = f"_{digest}"
    budget = max_len - len(suffix)
    if budget <= 0:
        return digest[:max_len]
    trimmed = sanitized[:budget].strip("_") or "repair"
    return f"{trimmed}{suffix}"


def _state_candidates(states_dir, exp_name=None, state_rel_path=None):
    states_dir = Path(states_dir)
    targeted = []

    if state_rel_path is not None:
        rel_path = Path(state_rel_path)
        direct = states_dir / rel_path
        if direct.exists():
            targeted.append(direct)

    if exp_name:
        targeted.extend(states_dir.glob(f"{exp_name}-*.pkl"))

    if targeted:
        return sorted({path.resolve() for path in targeted if path.exists()}, key=lambda p: p.stat().st_mtime)

    # Do not fall back to arbitrary state files when a specific exp_name or rel_path
    # was requested, otherwise one job can accidentally consume another job's output.
    if exp_name is not None or state_rel_path is not None:
        return []

    fallback = states_dir.rglob("*.pkl")
    return sorted({path.resolve() for path in fallback if path.exists()}, key=lambda p: p.stat().st_mtime)


def find_latest_state(states_dir, exp_name, state_rel_path=None):
    matches = _state_candidates(states_dir, exp_name=exp_name, state_rel_path=state_rel_path)
    if not matches:
        raise FileNotFoundError(
            f"No state file found for exp_name={exp_name}, state_rel_path={state_rel_path} in {states_dir}"
        )
    return matches[-1]


def clear_old_state_files(states_dir, exp_name, state_rel_path=None):
    states_dir = Path(states_dir)
    for path in states_dir.glob(f"{exp_name}-*.pkl"):
        path.unlink(missing_ok=True)
    if state_rel_path is not None:
        (states_dir / state_rel_path).unlink(missing_ok=True)


def wait_for_state_file(states_dir, exp_name, state_rel_path=None, timeout_s=180, proc=None):
    start = time.time()
    last_path = None
    last_size = -1
    stable_count = 0
    proc_exit_code = None
    proc_exit_at = None
    post_exit_grace_s = 8

    while time.time() - start < timeout_s:
        matches = _state_candidates(states_dir, exp_name=exp_name, state_rel_path=state_rel_path)
        if matches:
            curr = matches[-1]
            curr_size = curr.stat().st_size
            if last_path == curr and curr_size == last_size and curr_size > 0:
                stable_count += 1
            else:
                stable_count = 0
                last_path = curr
                last_size = curr_size

            if stable_count >= 3:
                return curr
        if proc is not None:
            exit_code = proc.poll()
            if exit_code is not None:
                if proc_exit_at is None:
                    proc_exit_code = exit_code
                    proc_exit_at = time.time()
                elif time.time() - proc_exit_at >= post_exit_grace_s:
                    raise RuntimeError(
                        f"Repair subprocess exited before writing state file "
                        f"(exit_code={proc_exit_code}) for exp_name={exp_name}, state_rel_path={state_rel_path}"
                    )
        time.sleep(1)

    raise TimeoutError(
        f"Timed out waiting for state file for exp_name={exp_name}, state_rel_path={state_rel_path}"
    )


def wait_for_marker_file(marker_path, proc=None, timeout_s=None):
    marker_path = Path(marker_path)
    start = time.time()
    proc_exit_code = None
    proc_exit_at = None
    post_exit_grace_s = 8

    while timeout_s is None or time.time() - start < timeout_s:
        if marker_path.is_file():
            return marker_path
        if proc is not None:
            exit_code = proc.poll()
            if exit_code is not None:
                if proc_exit_at is None:
                    proc_exit_code = exit_code
                    proc_exit_at = time.time()
                elif time.time() - proc_exit_at >= post_exit_grace_s:
                    raise RuntimeError(
                        f"Repair subprocess exited before writing marker file "
                        f"(exit_code={proc_exit_code}): {marker_path}"
                    )
        time.sleep(0.5)

    raise TimeoutError(f"Timed out waiting for marker file: {marker_path}")


class SpeedCheckFailedError(RuntimeError):
    def __init__(self, speed_check):
        self.speed_check = speed_check
        super().__init__(
            "post speed check failed: "
            f"{speed_check['outlier_count']} outliers, "
            f"max_acceleration={speed_check['max_acceleration']:.4f}, "
            f"threshold={speed_check['threshold']:.4f}"
        )


class TrackingQualityFailedError(RuntimeError):
    """A completed PHC rollout that did not track its reference."""

    def __init__(self, audit):
        self.audit = audit
        super().__init__(
            "PHC tracking quality gate failed: "
            f"root p95={audit['root_tracking_error_m']['p95']:.3f}m, "
            f"early max={audit['early_root_tracking_error_m']['max']:.3f}m"
        )


def audit_tracking_state(
    state_file, report_file, *, early_frames=30,
    max_root_p95_m=0.20, max_early_root_error_m=0.25,
):
    """Verify tracking in Isaac Z-up without modifying reference timing or pose.

    PHC's actor and target pelvis may have a fixed origin offset.  Removing only
    that first-frame constant exposes actual drift/falls and prevents a finished
    simulator process from being mislabelled as a usable physical repair.
    """
    payload = joblib.load(state_file)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"invalid PHC state payload: {state_file}")
    key, sequence = max(
        payload.items(), key=lambda item: np.asarray(item[1].get("trans", [])).shape[0]
    )
    actual = np.asarray(sequence.get("trans", []), dtype=np.float64)
    target = np.asarray(sequence.get("ref_body_pos_full", []), dtype=np.float64)
    if (
        actual.ndim != 2 or actual.shape[1] != 3
        or target.ndim != 3 or target.shape[1:] != (24, 3)
        or target.shape[0] != actual.shape[0] or actual.shape[0] < 2
    ):
        raise ValueError("PHC state has no frame-aligned actor/target pelvis trace")
    raw_offset = actual - target[:, 0, :]
    error = np.linalg.norm(raw_offset - raw_offset[:1], axis=1)
    early = error[: min(int(early_frames), len(error))]
    report = {
        "schema_version": 1,
        "purpose": "phc_reference_tracking_quality_gate",
        "coordinate_contract": "Isaac Z-up; remove first-frame actor-to-target pelvis offset once, without changing motion",
        "state_file": str(state_file),
        "sequence_key": str(key),
        "frames": int(len(error)),
        "initial_actor_to_target_offset_m": raw_offset[0].tolist(),
        "root_tracking_error_m": {
            "median": float(np.median(error)),
            "p95": float(np.quantile(error, 0.95)),
            "max": float(np.max(error)),
        },
        "early_root_tracking_error_m": {
            "frames": int(len(early)), "max": float(np.max(early)),
        },
        "thresholds_m": {
            "max_root_p95": float(max_root_p95_m),
            "max_early_root_error": float(max_early_root_error_m),
        },
        "first_frame_over_0p05m": (
            int(np.flatnonzero(error > 0.05)[0]) if np.any(error > 0.05) else None
        ),
    }
    report["status"] = (
        "passed"
        if report["root_tracking_error_m"]["p95"] <= max_root_p95_m
        and report["early_root_tracking_error_m"]["max"] <= max_early_root_error_m
        else "rejected"
    )
    report_file = Path(report_file)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def sequence_frame_count(sequence):
    for key in ("poses", "root_orient", "pose_body", "trans", "trans_original"):
        value = sequence.get(key)
        if isinstance(value, np.ndarray) and value.ndim >= 1:
            return int(value.shape[0])
    raise ValueError("Cannot infer repaired sequence frame count")


def slice_sequence(sequence, start_index, end_index):
    num_frames = sequence_frame_count(sequence)
    start = max(0, min(int(start_index), num_frames))
    end = max(0, min(int(end_index), num_frames))
    if end <= start:
        raise ValueError(f"Invalid repaired slice [{start}, {end}) for sequence length {num_frames}")

    sliced = {}
    for key, value in sequence.items():
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == num_frames:
            sliced[key] = value[start:end]
        else:
            sliced[key] = value
    return sliced


def valid_speed_intervals(num_frames, outlier_indices, min_length):
    breakpoints = sorted({int(idx) for idx in outlier_indices if 0 <= int(idx) < num_frames})
    intervals = []
    start = 0
    for breakpoint in breakpoints:
        end = breakpoint
        if end - start >= min_length:
            intervals.append((start, end))
        start = breakpoint + 1
    if num_frames - start >= min_length:
        intervals.append((start, num_frames))
    return intervals


def infer_period_name(output_file, repair_period=None):
    if repair_period:
        return str(repair_period)
    stem = Path(output_file).stem
    if stem.endswith("_repaired"):
        stem = stem[: -len("_repaired")]
    if "-" in stem:
        return stem.split("-", 1)[0]
    return "period_0"


def infer_repair_start_index(output_file, repair_start_index=None):
    if repair_start_index is not None:
        return int(repair_start_index)
    stem = Path(output_file).stem
    if stem.endswith("_repaired"):
        stem = stem[: -len("_repaired")]
    match = re.search(r"-(\d+)_(\d+)$", stem)
    if match:
        return int(match.group(1))
    return 0


def save_stage1_style_repair_outputs(
    repaired,
    output_file,
    *,
    speed_check,
    repair_start_index=None,
    repair_period=None,
    min_length=90,
):
    num_frames = sequence_frame_count(repaired)
    min_length = max(1, int(min_length))
    intervals = valid_speed_intervals(num_frames, speed_check["outlier_indices"], min_length)
    if not intervals:
        enriched = dict(speed_check)
        enriched["min_length"] = min_length
        enriched["valid_intervals"] = []
        raise SpeedCheckFailedError(enriched)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    global_offset = infer_repair_start_index(output_file, repair_start_index)
    period_name = infer_period_name(output_file, repair_period)
    output_file.unlink(missing_ok=True)
    for stale_path in output_file.parent.glob(f"{period_name}-*_validated.npz"):
        stale_path.unlink()
    saved = []
    for local_start, local_end in intervals:
        global_start = global_offset + local_start
        global_end = global_offset + local_end
        save_path = output_file.parent / f"{period_name}-{global_start}_{global_end}_validated.npz"
        np.savez_compressed(save_path, **slice_sequence(repaired, local_start, local_end))
        saved.append(
            {
                "path": save_path.as_posix(),
                "local_start_index": int(local_start),
                "local_end_index": int(local_end),
                "start_index": int(global_start),
                "end_index": int(global_end),
                "num_frames": int(local_end - local_start),
            }
        )

    return {
        "speed_check": {
            **speed_check,
            "min_length": min_length,
            "valid_intervals": saved,
        },
        "repaired_npzs": [item["path"] for item in saved],
        "primary_repaired_npz": saved[0]["path"],
    }


def anchor_translation_to_reference(repaired, state_sequence, reference_npz, gravity_axis):
    """Restore the source SMPL translation convention after PHC state export.

    PHC stores state_sequence['trans'] as the actor pelvis in Isaac's Z-up
    frame. It is not the same origin as the SMPL translation used to construct
    the PHC reference skeleton. Anchor the output to the reference NPZ and add
    only the measured pelvis tracking displacement.
    """
    actual_pelvis = np.asarray(state_sequence.get("trans", []), dtype=np.float32)
    target_positions = np.asarray(
        state_sequence.get("ref_body_pos_full", []), dtype=np.float32
    )
    if (
        actual_pelvis.ndim != 2
        or actual_pelvis.shape[1] != 3
        or target_positions.ndim != 3
        or target_positions.shape[0] != actual_pelvis.shape[0]
        or target_positions.shape[1] < 1
        or target_positions.shape[2] != 3
    ):
        # Older PHC state files do not contain the target body trace.
        return repaired

    with np.load(reference_npz, allow_pickle=True) as reference:
        reference_trans = np.asarray(reference["trans"], dtype=np.float32)
    target_len = int(np.asarray(repaired["poses"]).shape[0])
    if reference_trans.shape != (target_len, 3):
        raise ValueError(
            "Reference translation must match the repaired sequence length; "
            f"got {reference_trans.shape} for {target_len} frames"
        )

    pelvis_delta_z_up = actual_pelvis - target_positions[:, 0, :]
    pelvis_delta_z_up = _resample_linear(pelvis_delta_z_up, target_len)
    if gravity_axis == "neg_y":
        # inverse of Rx(+90): [x, y, z]_z_up -> [x, z, -y]_neg_y
        pelvis_delta_source = np.stack(
            (
                pelvis_delta_z_up[:, 0],
                pelvis_delta_z_up[:, 2],
                -pelvis_delta_z_up[:, 1],
            ),
            axis=1,
        )
    elif gravity_axis == "neg_z":
        pelvis_delta_source = pelvis_delta_z_up
    else:
        raise ValueError(f"Unsupported gravity axis: {gravity_axis!r}")

    anchored = dict(repaired)
    anchored["trans"] = (reference_trans + pelvis_delta_source).astype(np.float32)
    anchored["trans_original"] = anchored["trans"].copy()
    return anchored

def export_longest_state_to_npz(
    state_file,
    output_file,
    reference_npz=None,
    force_fps=None,
    gravity_axis="neg_z",
    post_check_speed=False,
    speed_acc_threshold=14.7,
    post_check_min_length=90,
    repair_start_index=None,
    repair_period=None,
    return_metadata=False,
):
    data = joblib.load(state_file)
    key, seq = max(data.items(), key=lambda item: np.asarray(item[1]["trans"]).shape[0])
    repaired = convert_sequence(seq, gravity_axis=gravity_axis)
    if reference_npz is not None:
        repaired = match_reference_timing(repaired, reference_npz)
        repaired = anchor_translation_to_reference(
            repaired, seq, reference_npz, gravity_axis
        )
    if force_fps is not None:
        repaired["mocap_frame_rate"] = np.array(float(force_fps))
    metadata = {
        "longest_key": key,
        "repaired_npz": Path(output_file).as_posix(),
        "repaired_npzs": [Path(output_file).as_posix()],
        "post_check_speed": bool(post_check_speed),
    }
    if post_check_speed:
        speed_check = evaluate_speed_check(repaired, acceleration_threshold=speed_acc_threshold, smooth=True)
        repair_outputs = save_stage1_style_repair_outputs(
            repaired,
            output_file,
            speed_check=speed_check,
            repair_start_index=repair_start_index,
            repair_period=repair_period,
            min_length=post_check_min_length,
        )
        metadata.update(
            {
                "repaired_npz": repair_outputs["primary_repaired_npz"],
                "repaired_npzs": repair_outputs["repaired_npzs"],
                "speed_check": repair_outputs["speed_check"],
            }
        )
        return metadata if return_metadata else key

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, **repaired)
    return metadata if return_metadata else key


def build_subprocess_env(python_exec):
    env = os.environ.copy()
    python_path = Path(python_exec).resolve()
    env_lib_dir = python_path.parent.parent / "lib"
    ld_library_path = env.get("LD_LIBRARY_PATH", "")
    if env_lib_dir.is_dir():
        env["LD_LIBRARY_PATH"] = f"{env_lib_dir.as_posix()}:{ld_library_path}" if ld_library_path else env_lib_dir.as_posix()
    phc_paths = [REPO_ROOT.as_posix(), (REPO_ROOT / "phc").as_posix(), (REPO_ROOT / "poselib").as_posix()]
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join(phc_paths + ([pythonpath] if pythonpath else []))
    return env


def append_hydra_output_overrides(cmd, exp_name, keep_hydra_outputs):
    if keep_hydra_outputs:
        return
    hydra_run_dir = Path(tempfile.gettempdir()) / "phc_hydra_runs" / exp_name
    cmd.extend(
        [
            f"hydra.run.dir={hydra_str(hydra_run_dir.as_posix())}",
            "hydra.output_subdir=null",
        ]
    )


def append_rendering_output_overrides(cmd, exp_name, keep_renderings, rendering_output_root=None):
    if rendering_output_root:
        rendering_root = Path(rendering_output_root)
        rendering_root.mkdir(parents=True, exist_ok=True)
        cmd.append(f"+rendering_output_root={hydra_str(rendering_root.as_posix())}")
        return
    if keep_renderings:
        return
    rendering_root = Path(tempfile.gettempdir()) / "phc_renderings" / exp_name
    rendering_root.mkdir(parents=True, exist_ok=True)
    cmd.append(f"+rendering_output_root={hydra_str(rendering_root.as_posix())}")


def run_subprocess_until_state(
    cmd,
    states_root,
    exp_name,
    state_rel_path,
    python_exec,
    wait_for_process_exit=False,
    process_timeout_s=600,
    wait_marker_path=None,
    post_marker_exit_timeout_s=8,
):
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT.as_posix(), env=build_subprocess_env(python_exec))
    try:
        state_file = wait_for_state_file(states_root, exp_name, state_rel_path=state_rel_path, proc=proc)
        if wait_marker_path is not None:
            wait_for_marker_file(wait_marker_path, proc=proc, timeout_s=None)
            if proc.poll() is None:
                try:
                    proc.wait(timeout=post_marker_exit_timeout_s)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=10)
        elif wait_for_process_exit:
            try:
                proc.wait(timeout=process_timeout_s)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(
                    f"Timed out waiting for repair subprocess to finish after state export "
                    f"for exp_name={exp_name}, state_rel_path={state_rel_path}"
                ) from exc
            if proc.returncode not in (0, None):
                raise RuntimeError(
                    f"Repair subprocess exited with code {proc.returncode} after writing state file "
                    f"for exp_name={exp_name}, state_rel_path={state_rel_path}"
                )
        time.sleep(2.0)  # Allow OS to fully release GPU resources
        return state_file
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        else:
            proc.wait(timeout=1)


def run_single_clip(
    clip_file,
    input_root,
    phc_motion_root,
    repaired_root,
    states_root,
    python_exec,
    gravity_axis,
    keep_shape,
    target_fps,
    primitive_model_path,
    composer_checkpoint_path,
    episode_length,
    zero_out_far,
    enable_early_termination,
    render_o3d,
    gym_viewer,
    no_virtual_display,
    auto_quit_after_record,
    wait_for_o3d_close_after_record,
    keep_hydra_outputs,
    keep_renderings,
    rendering_output_root=None,
    hydra_overrides=None,
    post_check_speed=False,
    speed_acc_threshold=14.7,
    post_check_min_length=30,
    o3d_auto_rotate=False,
    o3d_rotation_speed=0.0,
    o3d_num_rotations=2.5,
    record_isaac_gym=False,
    isaac_record_width=960,
    isaac_record_height=720,
    isaac_record_fps=30,
    isaac_record_fov=70.0,
    isaac_camera_distance=5.0,
    isaac_camera_side=1.2,
    isaac_camera_height=1.2,
    isaac_camera_target_height=0.30,
    isaac_camera_smoothing=0.15,
    isaac_camera_mode="follow",
    isaac_camera_path=None,
):
    clip_file = Path(clip_file)
    rel_parent = clip_file.parent.relative_to(input_root)
    base_name = clip_file.stem
    state_rel_path = rel_parent / f"{base_name}.pkl"

    motion_pkl = Path(phc_motion_root) / rel_parent / f"{base_name}.pkl"
    repaired_npz = Path(repaired_root) / rel_parent / f"{base_name}_repaired.npz"
    exp_name = sanitize_name(f"repair_{rel_parent.as_posix()}_{base_name}".replace("/", "__"))

    smpl_local_robot = LocalRobot(ROBOT_CFG, data_dir="data/smpl")
    motion_pkl.parent.mkdir(parents=True, exist_ok=True)
    motion_dict = {
        motion_pkl.as_posix(): convert_clip(
            clip_dir=clip_file,
            smpl_local_robot=smpl_local_robot,
            force_neutral=not keep_shape,
            target_fps=target_fps,
            gravity_axis=gravity_axis,
        )
    }
    joblib.dump(motion_dict, motion_pkl)

    use_headless = not gym_viewer  # Isaac Gym camera recording works headlessly via camera sensors.
    auto_record = bool(render_o3d or record_isaac_gym)

    cmd = [
        python_exec,
        "phc/run_hydra.py",
        "learning=im_mcp_big",
        f"exp_name={exp_name}",
        f"output_path={hydra_str(Path(composer_checkpoint_path).parent.as_posix())}",
        "env=env_im_getup_mcp",
        "robot=smpl_humanoid",
        f"env.zero_out_far={'True' if zero_out_far else 'False'}",
        "robot.real_weight_porpotion_boxes=False",
        "env.num_prim=3",
        f"env.motion_file={hydra_str(motion_pkl.as_posix())}",
        f"env.models=[{hydra_str(primitive_model_path)}]",
        "env.num_envs=1",
        # Reference tracking is an evaluation, not a random get-up episode.
        # Callers can still intentionally override these values via --hydra_overrides.
        "env.stateInit=Start",
        "env.hybridInitProb=0.0",
        "env.fallInitProb=0.0",
        "env.recoveryEpisodeProb=0.0",
        f"headless={'True' if use_headless else 'False'}",
        "epoch=-1",
        "test=True",
        "im_eval=True",
        f"render_o3d={'True' if render_o3d else 'False'}",
        f"no_virtual_display={'True' if no_virtual_display else 'False'}",
        f"auto_record_motion={'True' if auto_record else 'False'}",
        f"auto_quit_after_record={'True' if auto_quit_after_record else 'False'}",
        f"+record_isaac_gym={'True' if record_isaac_gym else 'False'}",
        f"+isaac_record_width={int(isaac_record_width)}",
        f"+isaac_record_height={int(isaac_record_height)}",
        f"+isaac_record_fps={int(isaac_record_fps)}",
        f"+isaac_record_fov={float(isaac_record_fov)}",
        f"+isaac_record_camera_distance={float(isaac_camera_distance)}",
        f"+isaac_record_camera_side={float(isaac_camera_side)}",
        f"+isaac_record_camera_height={float(isaac_camera_height)}",
        f"+isaac_record_camera_target_height={float(isaac_camera_target_height)}",
        f"+isaac_record_camera_smoothing={float(isaac_camera_smoothing)}",
        f"+isaac_record_camera_mode={hydra_str(isaac_camera_mode)}",
        f"+wait_for_o3d_close_after_record={'True' if wait_for_o3d_close_after_record else 'False'}",
        f"+o3d_auto_rotate={'True' if o3d_auto_rotate else 'False'}",
        f"+o3d_rotation_speed={o3d_rotation_speed}",
        f"+o3d_num_rotations={o3d_num_rotations}",
        f"record_states_only={'False' if record_isaac_gym else 'True'}",
        f"seq_export_states={'False' if render_o3d else 'True'}",
        f"repair_reference_root={hydra_str(Path(input_root).as_posix())}",
        f"repair_motion_root={hydra_str(Path(phc_motion_root).as_posix())}",
        f"repair_output_root={hydra_str(Path(repaired_root).as_posix())}",
        f"repair_state_root={hydra_str(Path(states_root).as_posix())}",
        f"repair_output_gravity_axis={gravity_axis}",
        f"env.episode_length={episode_length}",
        f"env.enableEarlyTermination={'True' if enable_early_termination else 'False'}",
    ]
    if isaac_camera_path:
        cmd.append(f"+isaac_record_camera_path={hydra_str(Path(isaac_camera_path).as_posix())}")
    append_hydra_output_overrides(cmd, exp_name, keep_hydra_outputs)
    append_rendering_output_overrides(cmd, exp_name, keep_renderings, rendering_output_root)
    if hydra_overrides:
        cmd.extend(hydra_overrides)
    clear_old_state_files(states_root, exp_name, state_rel_path=state_rel_path)
    wait_marker_path = None
    if render_o3d and wait_for_o3d_close_after_record and rendering_output_root:
        wait_marker_path = Path(rendering_output_root) / f"{exp_name}-o3d-closed.marker"
        wait_marker_path.unlink(missing_ok=True)
    wait_for_process_exit = bool(auto_record and auto_quit_after_record and wait_marker_path is None)
    process_timeout_s = max(300, int(episode_length / max(target_fps, 1)) + 240)
    state_file = run_subprocess_until_state(
        cmd,
        states_root,
        exp_name,
        state_rel_path,
        python_exec,
        wait_for_process_exit=wait_for_process_exit,
        process_timeout_s=process_timeout_s,
        wait_marker_path=wait_marker_path,
    )
    tracking_audit = audit_tracking_state(
        state_file,
        Path(states_root) / rel_parent / f"{base_name}_tracking_audit.json",
    )
    if tracking_audit["status"] != "passed":
        raise TrackingQualityFailedError(tracking_audit)

    export_result = export_longest_state_to_npz(
        state_file,
        repaired_npz,
        reference_npz=clip_file,
        gravity_axis=gravity_axis,
        post_check_speed=post_check_speed,
        speed_acc_threshold=speed_acc_threshold,
        post_check_min_length=post_check_min_length,
        return_metadata=True,
    )
    return {
        "clip_file": clip_file.as_posix(),
        "motion_pkl": motion_pkl.as_posix(),
        "state_file": state_file.as_posix(),
        "repaired_npz": export_result.get("repaired_npz", repaired_npz.as_posix()),
        "repaired_npzs": export_result.get("repaired_npzs", [export_result.get("repaired_npz", repaired_npz.as_posix())]),
        "longest_key": export_result.get("longest_key"),
        "speed_check": export_result.get("speed_check"),
    }


def build_motion_pkls(
    clip_files,
    input_root,
    phc_motion_root,
    gravity_axis,
    keep_shape,
    target_fps,
):
    smpl_local_robot = LocalRobot(ROBOT_CFG, data_dir="data/smpl")
    built = []
    for clip_file in tqdm(clip_files, desc="Converting motions"):
        clip_file = Path(clip_file)
        rel_parent = clip_file.parent.relative_to(input_root)
        base_name = clip_file.stem
        motion_pkl = Path(phc_motion_root) / rel_parent / f"{base_name}.pkl"
        motion_pkl.parent.mkdir(parents=True, exist_ok=True)
        motion_dict = {
            motion_pkl.as_posix(): convert_clip(
                clip_dir=clip_file,
                smpl_local_robot=smpl_local_robot,
                force_neutral=not keep_shape,
                target_fps=target_fps,
                gravity_axis=gravity_axis,
            )
        }
        joblib.dump(motion_dict, motion_pkl)
        built.append(motion_pkl)
    return built


def run_reuse_gym_viewer(
    clip_files,
    input_root,
    phc_motion_root,
    repaired_root,
    states_root,
    python_exec,
    gravity_axis,
    keep_shape,
    target_fps,
    primitive_model_path,
    composer_checkpoint_path,
    episode_length,
    zero_out_far,
    enable_early_termination,
    render_o3d,
    gym_viewer,
    no_virtual_display,
    auto_quit_after_record,
    wait_for_o3d_close_after_record,
    keep_hydra_outputs,
    keep_renderings,
    rendering_output_root=None,
    hydra_overrides=None,
    o3d_auto_rotate=False,
    o3d_rotation_speed=0.0,
    o3d_num_rotations=2.5,
):
    phc_motion_root = Path(phc_motion_root)
    if phc_motion_root.exists():
        shutil.rmtree(phc_motion_root)
    phc_motion_root.mkdir(parents=True, exist_ok=True)

    build_motion_pkls(
        clip_files=clip_files,
        input_root=input_root,
        phc_motion_root=phc_motion_root,
        gravity_axis=gravity_axis,
        keep_shape=keep_shape,
        target_fps=target_fps,
    )

    exp_name = sanitize_name(f"repair_seq_{Path(input_root).name}", max_len=60)
    clear_old_state_files(states_root, exp_name)

    use_headless = not gym_viewer  # Only create gym viewer if explicitly requested; O3D uses its own window

    cmd = [
        python_exec,
        "phc/run_hydra.py",
        "learning=im_mcp_big",
        f"exp_name={exp_name}",
        f"output_path={hydra_str(Path(composer_checkpoint_path).parent.as_posix())}",
        "env=env_im_getup_mcp",
        "robot=smpl_humanoid",
        f"env.zero_out_far={'True' if zero_out_far else 'False'}",
        "robot.real_weight_porpotion_boxes=False",
        "env.num_prim=3",
        f"env.motion_file={hydra_str(phc_motion_root.as_posix())}",
        "env.seq_motions=True",
        f"env.models=[{hydra_str(primitive_model_path)}]",
        "env.num_envs=1",
        # Reference tracking is an evaluation, not a random get-up episode.
        # Callers can still intentionally override these values via --hydra_overrides.
        "env.stateInit=Start",
        "env.hybridInitProb=0.0",
        "env.fallInitProb=0.0",
        "env.recoveryEpisodeProb=0.0",
        f"headless={'True' if use_headless else 'False'}",
        "epoch=-1",
        "test=True",
        "im_eval=True",
        f"render_o3d={'True' if render_o3d else 'False'}",
        f"no_virtual_display={'True' if no_virtual_display else 'False'}",
        f"auto_record_motion={'True' if render_o3d else 'False'}",
        f"auto_quit_after_record={'True' if auto_quit_after_record else 'False'}",
        f"+wait_for_o3d_close_after_record={'True' if wait_for_o3d_close_after_record else 'False'}",
        f"+o3d_auto_rotate={'True' if o3d_auto_rotate else 'False'}",
        f"+o3d_rotation_speed={o3d_rotation_speed}",
        f"+o3d_num_rotations={o3d_num_rotations}",
        "record_states_only=True",
        "seq_export_states=True",
        f"repair_reference_root={hydra_str(Path(input_root).as_posix())}",
        f"repair_motion_root={hydra_str(Path(phc_motion_root).as_posix())}",
        f"repair_output_root={hydra_str(Path(repaired_root).as_posix())}",
        f"repair_state_root={hydra_str(Path(states_root).as_posix())}",
        f"repair_output_gravity_axis={gravity_axis}",
        f"env.episode_length={episode_length}",
        f"env.enableEarlyTermination={'True' if enable_early_termination else 'False'}",
    ]
    append_hydra_output_overrides(cmd, exp_name, keep_hydra_outputs)
    append_rendering_output_overrides(cmd, exp_name, keep_renderings, rendering_output_root)
    if hydra_overrides:
        cmd.extend(hydra_overrides)
    subprocess.run(cmd, cwd=REPO_ROOT.as_posix(), check=True, env=build_subprocess_env(python_exec))

    success = []
    failures = []
    for clip_file in clip_files:
        clip_file = Path(clip_file)
        rel_parent = clip_file.parent.relative_to(input_root)
        repaired_npz = Path(repaired_root) / rel_parent / f"{clip_file.stem}_repaired.npz"
        item = {
            "clip_file": clip_file.as_posix(),
            "repaired_npz": repaired_npz.as_posix(),
        }
        if repaired_npz.exists():
            success.append(item)
        else:
            failures.append({**item, "error": "missing repaired npz after sequential run"})
    return success, failures


def run_parallel_single_clip_jobs(clip_files, args, input_root):
    summary = []
    failures = []

    def _run_job(clip_file):
        return run_single_clip(
            clip_file=clip_file,
            input_root=input_root,
            phc_motion_root=args.phc_motion_root,
            repaired_root=args.repaired_root,
            states_root=args.states_root,
            python_exec=args.python_exec,
            gravity_axis=args.gravity_axis,
            keep_shape=args.keep_shape,
            target_fps=args.target_fps,
            primitive_model_path=args.primitive_model_path,
            composer_checkpoint_path=args.composer_checkpoint_path,
            episode_length=args.episode_length,
            zero_out_far=args.zero_out_far,
            enable_early_termination=args.enable_early_termination,
            render_o3d=args.render_o3d,
            gym_viewer=args.gym_viewer,
            no_virtual_display=args.no_virtual_display,
            auto_quit_after_record=args.auto_quit_after_record,
            wait_for_o3d_close_after_record=args.wait_for_o3d_close_after_record,
            keep_hydra_outputs=args.keep_hydra_outputs,
            keep_renderings=args.keep_renderings,
            rendering_output_root=args.rendering_output_root,
            hydra_overrides=args.hydra_overrides,
            post_check_speed=args.post_check_speed,
            speed_acc_threshold=args.speed_acc_threshold,
            post_check_min_length=args.post_check_min_length,
            o3d_auto_rotate=args.o3d_auto_rotate,
            o3d_rotation_speed=args.o3d_rotation_speed,
            o3d_num_rotations=args.o3d_num_rotations,
            record_isaac_gym=args.record_isaac_gym,
            isaac_record_width=args.isaac_record_width,
            isaac_record_height=args.isaac_record_height,
            isaac_record_fps=args.isaac_record_fps,
            isaac_record_fov=args.isaac_record_fov,
            isaac_camera_distance=args.isaac_camera_distance,
            isaac_camera_side=args.isaac_camera_side,
            isaac_camera_height=args.isaac_camera_height,
            isaac_camera_target_height=args.isaac_camera_target_height,
            isaac_camera_smoothing=args.isaac_camera_smoothing,
            isaac_camera_mode=args.isaac_camera_mode,
            isaac_camera_path=args.isaac_camera_path,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel_workers) as executor:
        future_to_clip = {executor.submit(_run_job, clip_file): clip_file for clip_file in clip_files}
        for future in tqdm(concurrent.futures.as_completed(future_to_clip), total=len(future_to_clip)):
            clip_file = future_to_clip[future]
            try:
                result = future.result()
                summary.append(result)
                print(f"[OK] {clip_file} -> {result['repaired_npz']}")
            except Exception as exc:
                failures.append({"clip_file": clip_file.as_posix(), "error": str(exc)})
                print(f"[FAIL] {clip_file}: {exc}")

    return summary, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, required=True, help="Root directory containing many zitai subfolders with one optimized npz each.")
    parser.add_argument("--phc_motion_root", type=str, default="data/custom_motions/batch_zitai_phc")
    parser.add_argument("--repaired_root", type=str, default="output/repaired_smpl_batch")
    parser.add_argument("--states_root", type=str, default="output/states")
    parser.add_argument("--python_exec", type=str, default=sys.executable)
    parser.add_argument("--gravity_axis", type=str, default="neg_y", choices=["neg_y", "neg_z"])
    parser.add_argument("--keep_shape", action="store_true")
    parser.add_argument("--target_fps", type=int, default=30)
    parser.add_argument("--zero_out_far", action="store_true", help="Use PHC's zero-out-far recovery behavior during repair.")
    parser.add_argument("--enable_early_termination", action="store_true", help="Allow PHC to terminate failed rollouts instead of recording through them.")
    parser.add_argument("--primitive_model_path", type=str, default="output/HumanoidIm/phc_3/Humanoid.pth")
    parser.add_argument("--composer_checkpoint_path", type=str, default="output/HumanoidIm/phc_comp_3/Humanoid.pth")
    parser.add_argument("--episode_length", type=int, default=1000)
    parser.add_argument("--render_o3d", action="store_true")
    parser.add_argument("--record_isaac_gym", action="store_true", help="Record Isaac Gym camera-sensor MP4 during PHC repair without using Open3D.")
    parser.add_argument("--isaac_record_width", type=int, default=960)
    parser.add_argument("--isaac_record_height", type=int, default=720)
    parser.add_argument("--isaac_record_fps", type=int, default=30)
    parser.add_argument("--isaac_record_fov", type=float, default=70.0)
    parser.add_argument("--isaac_camera_distance", type=float, default=5.0)
    parser.add_argument("--isaac_camera_side", type=float, default=1.2)
    parser.add_argument("--isaac_camera_height", type=float, default=1.2)
    parser.add_argument("--isaac_camera_target_height", type=float, default=0.30)
    parser.add_argument("--isaac_camera_smoothing", type=float, default=0.15)
    parser.add_argument("--isaac_camera_mode", type=str, default="follow", choices=["follow", "global", "static", "path", "gvhmr", "path_static", "gvhmr_static"])
    parser.add_argument("--isaac_camera_path", type=str, default=None, help="NPZ camera path exported by tools/pipeline/export_gvhmr_camera.py.")
    parser.add_argument("--gym_viewer", action="store_true", help="Open Isaac Gym viewer for each clip. Disabled by default for batch runs.")
    parser.add_argument("--reuse_gym_viewer", action="store_true", help="Use one PHC process and reuse the same Gym viewer across clips.")
    parser.add_argument("--no_virtual_display", action="store_true")
    parser.add_argument("--auto_quit_after_record", dest="auto_quit_after_record", action="store_true", default=False)
    parser.add_argument("--no_auto_quit_after_record", dest="auto_quit_after_record", action="store_false")
    parser.add_argument(
        "--wait_for_o3d_close_after_record",
        action="store_true",
        help="After auto O3D recording finishes, keep the O3D window open until the user closes it.",
    )
    parser.add_argument(
        "--o3d_auto_rotate",
        action="store_true",
        help="Automatically rotate the O3D camera around the subject during recording.",
    )
    parser.add_argument(
        "--o3d_rotation_speed",
        type=float,
        default=0.0,
        help="Degrees per frame for O3D auto-rotation (0 = auto-calculate from --o3d_num_rotations).",
    )
    parser.add_argument(
        "--o3d_num_rotations",
        type=float,
        default=2.5,
        help="Target number of full 360-degree rotations during O3D recording (default: 2.5). Only used when --o3d_rotation_speed is 0.",
    )
    parser.add_argument(
        "--keep_hydra_outputs",
        action="store_true",
        help="Keep PHC Hydra run directories under the default output tree instead of redirecting them to /tmp.",
    )
    parser.add_argument(
        "--keep_renderings",
        action="store_true",
        help="Keep PHC rendering outputs under the default output tree instead of redirecting them to /tmp.",
    )
    parser.add_argument(
        "--rendering_output_root",
        type=str,
        default=None,
        help="Optional explicit root for PHC rendering outputs such as O3D mp4s.",
    )
    parser.add_argument("--parallel_workers", type=int, default=1, help="Number of clip repair jobs to run in parallel. Each job launches its own PHC process.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N clips.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip clips whose repaired npz already exists under repaired_root.")
    parser.add_argument("--post_check_speed", action="store_true", help="Drop repaired segments whose root acceleration exceeds --speed_acc_threshold.")
    parser.add_argument("--speed_acc_threshold", type=float, default=14.7, help="Linear acceleration threshold used by --post_check_speed.")
    parser.add_argument("--post_check_min_length", type=int, default=30, help="Minimum validated segment length kept by --post_check_speed.")
    parser.add_argument("--hydra_overrides", nargs="*", default=[], help="Extra Hydra overrides appended to phc/run_hydra.py.")
    args = parser.parse_args()
    input_root = Path(args.input_root)
    clip_files = sorted(input_root.rglob("*_optimized.npz"))
    if args.limit is not None:
        clip_files = clip_files[:args.limit]

    if args.skip_existing:
        filtered = []
        skipped = 0
        repaired_root = Path(args.repaired_root)
        for clip_file in clip_files:
            rel_parent = clip_file.parent.relative_to(input_root)
            repaired_npz = repaired_root / rel_parent / f"{clip_file.stem}_repaired.npz"
            if repaired_npz.exists():
                skipped += 1
            else:
                filtered.append(clip_file)
        clip_files = filtered
        print(f"Skipping {skipped} clips with existing repaired outputs")

    if not clip_files:
        raise FileNotFoundError(f"No optimized npz files to process under {args.input_root}")

    if args.gym_viewer and args.render_o3d:
        print(
            "======================================================================\n"
            "WARNING: Both --gym_viewer and --render_o3d are enabled.\n"
            "This opens two GPU-rendered windows simultaneously and may cause\n"
            "instability or freezes. Consider using only --render_o3d for\n"
            "visualization, which produces an MP4 recording as well.\n"
            "======================================================================\n"
        )

    if args.parallel_workers < 1:
        raise ValueError("--parallel_workers must be >= 1")

    if args.parallel_workers > 1 and args.reuse_gym_viewer:
        raise ValueError("--parallel_workers > 1 cannot be combined with --reuse_gym_viewer")

    if args.parallel_workers > 1 and (args.gym_viewer or args.render_o3d or args.record_isaac_gym):
        raise ValueError("--parallel_workers > 1 requires plain headless repair. Do not combine it with --gym_viewer, --render_o3d, or --record_isaac_gym")

    if args.reuse_gym_viewer and args.record_isaac_gym:
        raise ValueError("--record_isaac_gym is only supported for single-clip PHC processes, not --reuse_gym_viewer")

    if args.reuse_gym_viewer and args.post_check_speed:
        raise ValueError("--post_check_speed is only supported for single-clip repair jobs, not --reuse_gym_viewer")

    if args.reuse_gym_viewer:
        summary, failures = run_reuse_gym_viewer(
            clip_files=clip_files,
            input_root=input_root,
            phc_motion_root=args.phc_motion_root,
            repaired_root=args.repaired_root,
            states_root=args.states_root,
            python_exec=args.python_exec,
            gravity_axis=args.gravity_axis,
            keep_shape=args.keep_shape,
            target_fps=args.target_fps,
            primitive_model_path=args.primitive_model_path,
            composer_checkpoint_path=args.composer_checkpoint_path,
            episode_length=args.episode_length,
            zero_out_far=args.zero_out_far,
            enable_early_termination=args.enable_early_termination,
            render_o3d=args.render_o3d,
            gym_viewer=args.gym_viewer,
            no_virtual_display=args.no_virtual_display,
            auto_quit_after_record=args.auto_quit_after_record,
            wait_for_o3d_close_after_record=args.wait_for_o3d_close_after_record,
            keep_hydra_outputs=args.keep_hydra_outputs,
            keep_renderings=args.keep_renderings,
            rendering_output_root=args.rendering_output_root,
            hydra_overrides=args.hydra_overrides,
            o3d_auto_rotate=args.o3d_auto_rotate,
            o3d_rotation_speed=args.o3d_rotation_speed,
            o3d_num_rotations=args.o3d_num_rotations,
        )
        for item in summary:
            print(f"[OK] {item['clip_file']} -> {item['repaired_npz']}")
        for item in failures:
            print(f"[FAIL] {item['clip_file']}: {item['error']}")
    elif args.parallel_workers > 1:
        summary, failures = run_parallel_single_clip_jobs(
            clip_files=clip_files,
            args=args,
            input_root=input_root,
        )
    else:
        summary = []
        failures = []
        for clip_file in tqdm(clip_files):
            try:
                result = run_single_clip(
                    clip_file=clip_file,
                    input_root=input_root,
                    phc_motion_root=args.phc_motion_root,
                    repaired_root=args.repaired_root,
                    states_root=args.states_root,
                    python_exec=args.python_exec,
                    gravity_axis=args.gravity_axis,
                    keep_shape=args.keep_shape,
                    target_fps=args.target_fps,
                    primitive_model_path=args.primitive_model_path,
                    composer_checkpoint_path=args.composer_checkpoint_path,
                    episode_length=args.episode_length,
                    zero_out_far=args.zero_out_far,
                    enable_early_termination=args.enable_early_termination,
                    render_o3d=args.render_o3d,
                    gym_viewer=args.gym_viewer,
                    no_virtual_display=args.no_virtual_display,
                    auto_quit_after_record=args.auto_quit_after_record,
                    wait_for_o3d_close_after_record=args.wait_for_o3d_close_after_record,
                    keep_hydra_outputs=args.keep_hydra_outputs,
                    keep_renderings=args.keep_renderings,
                    rendering_output_root=args.rendering_output_root,
                    hydra_overrides=args.hydra_overrides,
                    post_check_speed=args.post_check_speed,
                    speed_acc_threshold=args.speed_acc_threshold,
                    post_check_min_length=args.post_check_min_length,
                    o3d_auto_rotate=args.o3d_auto_rotate,
                    o3d_rotation_speed=args.o3d_rotation_speed,
                    o3d_num_rotations=args.o3d_num_rotations,
                    record_isaac_gym=args.record_isaac_gym,
                    isaac_record_width=args.isaac_record_width,
                    isaac_record_height=args.isaac_record_height,
                    isaac_record_fps=args.isaac_record_fps,
                    isaac_record_fov=args.isaac_record_fov,
                    isaac_camera_distance=args.isaac_camera_distance,
                    isaac_camera_side=args.isaac_camera_side,
                    isaac_camera_height=args.isaac_camera_height,
                    isaac_camera_target_height=args.isaac_camera_target_height,
                    isaac_camera_smoothing=args.isaac_camera_smoothing,
                    isaac_camera_mode=args.isaac_camera_mode,
                    isaac_camera_path=args.isaac_camera_path,
                )
                summary.append(result)
                print(f"[OK] {clip_file} -> {result['repaired_npz']}")
            except Exception as exc:
                failures.append({"clip_file": clip_file.as_posix(), "error": str(exc)})
                print(f"[FAIL] {clip_file}: {exc}")
            # Release GPU resources between clips
            gc.collect()
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:
                pass
            time.sleep(1.5)

    summary_path = Path(args.repaired_root) / "batch_summary.pkl"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"success": summary, "failures": failures}, summary_path)
    print(f"Summary written to {summary_path}")
    print(f"Success: {len(summary)}  Failures: {len(failures)}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
