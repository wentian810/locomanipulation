import argparse
import contextlib
import concurrent.futures
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.append(os.getcwd())
from scripts.data_process.perfetto_trace import PerfettoTrace

from scripts.data_process.batch_repair_zitai import (
    SpeedCheckFailedError,
    build_subprocess_env,
    clear_old_state_files,
    export_longest_state_to_npz,
    wait_for_state_file,
)
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
from scripts.data_process.convert_zitai_to_phc import _build_key


CFG_PATH_DEFAULT = "data/cfg/data_read_cfg.yaml"
SUMMARY_SUCCESS_NAME = "repair_summary.yaml"
SUMMARY_FAILURE_NAME = "repair_failures.json"
REPAIR_FAILURE_STATUSES = {"failed_repair", "failed_post_check_speed"}


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def dump_yaml(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_structured(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        return load_json(path)
    return load_yaml(path)


def dump_structured(path, data):
    path = Path(path)
    if path.suffix.lower() == ".json":
        dump_json(path, data)
        return
    dump_yaml(path, data)


def load_existing_summary(path, key):
    path = Path(path)
    if not path.exists():
        return []
    payload = load_structured(path)
    value = payload.get(key, [])
    return value if isinstance(value, list) else []


def load_trace_targets(path):
    if not path:
        return set()
    path = Path(path)
    if not path.exists():
        return set()
    payload = load_structured(path)
    if not isinstance(payload, dict):
        return set()
    record_ids = payload.get("record_ids", [])
    if not isinstance(record_ids, list):
        return set()
    return {str(item) for item in record_ids if item}


def infer_output_dir_name(source_npz, entry):
    if entry.get("output_dir_name"):
        return str(entry["output_dir_name"])
    source_npz = Path(source_npz)
    return source_npz.parent.name


def infer_output_basename(source_npz, entry):
    if entry.get("output_basename"):
        return str(entry["output_basename"])
    source_npz = Path(source_npz)
    return source_npz.stem


def normalize_entries(data):
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        raise ValueError(f"Unsupported manifest payload type: {type(data)}")

    for key in ("items", "entries", "records", "data", "samples"):
        value = data.get(key)
        if isinstance(value, list):
            return value

    if all(isinstance(v, dict) for v in data.values()):
        entries = []
        for key, value in data.items():
            item = dict(value)
            item.setdefault("key", key)
            entries.append(item)
        return entries

    raise ValueError("Cannot find repair entries in manifest. Expected a list or one of items/entries/records/data/samples.")


def seq_len_from_npz(npz_obj):
    for key in ("poses", "root_orient", "pose_body", "trans", "trans_original"):
        if key in npz_obj:
            value = np.asarray(npz_obj[key])
            if value.ndim >= 1:
                return int(value.shape[0])
    raise ValueError("Cannot infer sequence length from npz")


def normalize_slice_bounds(length, start_index, end_index):
    start = 0 if start_index is None else int(start_index)
    end = length if end_index is None else int(end_index)

    if start < 0:
        start += length
    if end < 0:
        end += length

    start = max(0, min(start, length))
    end = max(0, min(end, length))
    if end <= start:
        raise ValueError(f"Invalid slice range [{start}, {end}) for sequence length {length}")
    return start, end


def slice_npz(source_npz, output_npz, start_index=None, end_index=None):
    with np.load(source_npz, allow_pickle=True) as data:
        seq_len = seq_len_from_npz(data)
        start, end = normalize_slice_bounds(seq_len, start_index, end_index)
        sliced = {}
        for key in data.files:
            value = data[key]
            if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == seq_len:
                sliced[key] = value[start:end]
            else:
                sliced[key] = value

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **sliced)
    return {
        "output_npz": output_npz.as_posix(),
        "start_index": start,
        "end_index": end,
        "num_frames": end - start,
        "source_num_frames": seq_len,
    }


def manifest_name(manifest_path, payload):
    if isinstance(payload, dict) and payload.get("name"):
        return str(payload["name"])
    return Path(manifest_path).stem


def collect_manifest_files(fail_filter_root):
    fail_filter_root = Path(fail_filter_root)
    if not fail_filter_root.exists():
        raise FileNotFoundError(f"fail_filter_pth does not exist: {fail_filter_root}")
    return sorted(p for p in fail_filter_root.rglob("*.yml")) + sorted(p for p in fail_filter_root.rglob("*.yaml"))


def build_source_index(source_roots):
    source_index = defaultdict(list)
    source_roots = [Path(root).resolve() for root in source_roots]

    for root in source_roots:
        if not root.exists():
            continue
        for npz_path in root.rglob("*.npz"):
            npz_path = npz_path.resolve()
            rel = npz_path.relative_to(root)
            stem = npz_path.stem
            filename = npz_path.name
            rel_no_suffix = rel.with_suffix("").as_posix()
            rel_key = rel_no_suffix.replace("/", "__")
            parent_key = _build_key(root, npz_path)
            for key in {stem, filename, rel_no_suffix, rel_key, parent_key, npz_path.as_posix()}:
                source_index[key].append(npz_path)
    return source_index


def resolve_source_npz(entry, manifest_path, source_index, source_roots):
    candidate_keys = []
    explicit_path = entry.get("npz_path") or entry.get("path") or entry.get("file") or entry.get("source_npz")
    if explicit_path:
        explicit = Path(explicit_path)
        if explicit.exists():
            return explicit.resolve()
        manifest_relative = (Path(manifest_path).parent / explicit).resolve()
        if manifest_relative.exists():
            return manifest_relative
        for root in source_roots:
            joined = (Path(root) / explicit_path).resolve()
            if joined.exists():
                return joined
        candidate_keys.append(str(explicit_path))
        candidate_keys.append(explicit.name)
        candidate_keys.append(explicit.stem)

    for key_name in ("key", "motion_key", "clip_key", "name", "id"):
        value = entry.get(key_name)
        if value:
            candidate_keys.append(str(value))

    checked = []
    for candidate in candidate_keys:
        checked.append(candidate)
        matches = source_index.get(candidate, [])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous source npz for key '{candidate}': {[m.as_posix() for m in matches]}")

    raise FileNotFoundError(f"Cannot resolve source npz. checked={checked}")


def make_prepared_path(prepared_root, manifest_label, entry, source_npz):
    output_dir = infer_output_dir_name(source_npz, entry)
    output_basename = infer_output_basename(source_npz, entry)
    start = entry.get("start_index")
    end = entry.get("end_index")
    has_slice = start is not None or end is not None
    suffix = f"__{start if start is not None else 'None'}_{end if end is not None else 'None'}" if has_slice else ""
    return Path(prepared_root) / output_dir / f"{output_basename}{suffix}.npz"


def build_entry_uid(source_npz, output_dir_name, output_basename, start_index, end_index):
    return "||".join(
        [
            Path(source_npz).resolve().as_posix(),
            str(output_dir_name),
            str(output_basename),
            str(start_index),
            str(end_index),
        ]
    )


def build_failure_uid(manifest_path, entry_index, key, status):
    return "||".join(
        [
            Path(manifest_path).resolve().as_posix(),
            str(entry_index),
            str(key),
            str(status),
        ]
    )


def initialize_runtime_state(paths, args, source_roots):
    success_records = load_existing_summary(Path(paths["results_repair_pth"]) / args.summary_name, "success")
    failure_records = load_existing_summary(Path(paths["results_fail_pth"]) / args.failure_name, "failures")

    completed_uids = set()
    failure_uids = set()
    repaired_paths = set()

    for item in success_records:
        if not isinstance(item, dict):
            continue
        uid = item.get("uid")
        if uid:
            completed_uids.add(uid)
        repaired_npz = item.get("repaired_npz")
        if repaired_npz:
            repaired_paths.add(Path(repaired_npz).resolve().as_posix())
        for repaired_npz in item.get("repaired_npzs") or []:
            repaired_paths.add(Path(repaired_npz).resolve().as_posix())

    for item in failure_records:
        if not isinstance(item, dict):
            continue
        uid = item.get("uid")
        status = item.get("status")
        # Keep permanent prepare-time errors deduplicated, but allow prior
        # execution/runtime failures to be retried on later runs after fixes.
        if uid and status == "failed_prepare":
            failure_uids.add(uid)

    return {
        "manifest_cache": {},
        "success_records": success_records,
        "failure_records": failure_records,
        "completed_uids": completed_uids,
        "failure_uids": failure_uids,
        "repaired_paths": repaired_paths,
        "summary_dirty": True,
        "config": {
            "cfg": Path(args.cfg).resolve().as_posix(),
            "fail_filter_pth": paths["fail_filter_pth"].as_posix(),
            "results_repair_pth": paths["results_repair_pth"].as_posix(),
            "results_fail_pth": paths["results_fail_pth"].as_posix(),
            "source_roots": [str(Path(p).resolve()) for p in source_roots],
            "prepared_root": paths["prepared_root"].as_posix(),
            "phc_motion_root": Path(args.phc_motion_root).resolve().as_posix(),
            "states_root": Path(args.states_root).resolve().as_posix(),
        },
    }


def get_manifest_cache_entry(manifest_path, paths, source_roots, source_index, runtime_state):
    manifest_path = Path(manifest_path).resolve()
    mtime_ns = manifest_path.stat().st_mtime_ns
    cached = runtime_state["manifest_cache"].get(manifest_path.as_posix())
    if cached and cached.get("mtime_ns") == mtime_ns:
        return cached

    payload = load_yaml(manifest_path)
    label = manifest_name(manifest_path, payload)
    resolved_entries = []
    parse_failures = []

    try:
        entries = normalize_entries(payload)
    except Exception as exc:
        parse_failures.append({
            "manifest": manifest_path.as_posix(),
            "manifest_name": label,
            "status": "failed_prepare",
            "error": str(exc),
            "uid": build_failure_uid(manifest_path, -1, label, "failed_prepare"),
        })
        cache_entry = {
            "mtime_ns": mtime_ns,
            "label": label,
            "resolved_entries": resolved_entries,
            "parse_failures": parse_failures,
        }
        runtime_state["manifest_cache"][manifest_path.as_posix()] = cache_entry
        return cache_entry

    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            parse_failures.append({
                "manifest": manifest_path.as_posix(),
                "manifest_name": label,
                "entry_index": idx,
                "status": "failed_prepare",
                "error": f"Entry is not a dict: {entry}",
                "uid": build_failure_uid(manifest_path, idx, "invalid_entry", "failed_prepare"),
            })
            continue

        key_name = entry.get("key") or entry.get("motion_key") or entry.get("name")
        try:
            source_npz = resolve_source_npz(entry, manifest_path, source_index, source_roots)
            output_dir_name = infer_output_dir_name(source_npz, entry)
            output_basename = infer_output_basename(source_npz, entry)
            prepared_npz = make_prepared_path(paths["prepared_root"], label, entry, source_npz)
            start_index = entry.get("start_index")
            end_index = entry.get("end_index")
            repaired_npz = Path(paths["results_repair_pth"]) / output_dir_name / f"{output_basename}_repaired.npz"
            uid = build_entry_uid(source_npz, output_dir_name, output_basename, start_index, end_index)
            resolved_entries.append({
                "manifest": manifest_path.as_posix(),
                "manifest_name": label,
                "entry_index": idx,
                "entry": entry,
                "key": key_name or source_npz.stem,
                "source_npz": source_npz,
                "output_dir_name": output_dir_name,
                "output_basename": output_basename,
                "prepared_npz": prepared_npz,
                "repaired_npz": repaired_npz,
                "uid": uid,
            })
        except Exception as exc:
            parse_failures.append({
                "manifest": manifest_path.as_posix(),
                "manifest_name": label,
                "entry_index": idx,
                "key": key_name,
                "status": "failed_prepare",
                "error": str(exc),
                "uid": build_failure_uid(manifest_path, idx, key_name or "unknown", "failed_prepare"),
            })

    cache_entry = {
        "mtime_ns": mtime_ns,
        "label": label,
        "resolved_entries": resolved_entries,
        "parse_failures": parse_failures,
    }
    runtime_state["manifest_cache"][manifest_path.as_posix()] = cache_entry
    return cache_entry


def build_job(record, args, paths):
    prepared_npz = Path(record["prepared_npz"])
    return dict(
        record=record,
        clip_file=prepared_npz,
        input_root=Path(paths["prepared_root"]),
        phc_motion_root=Path(paths["phc_motion_root"]),
        repaired_root=Path(paths["results_repair_pth"]),
        states_root=Path(paths["states_root"]),
        python_exec=args.python_exec,
        gravity_axis=args.gravity_axis,
        keep_shape=args.keep_shape,
        primitive_model_path=args.primitive_model_path,
        composer_checkpoint_path=args.composer_checkpoint_path,
        episode_length=args.episode_length,
        render_o3d=args.render_o3d,
        gym_viewer=args.gym_viewer,
        no_virtual_display=args.no_virtual_display,
        gpu_id=record.get("gpu_id"),
        keep_hydra_outputs=args.keep_hydra_outputs,
        keep_renderings=args.keep_renderings,
        post_check_speed=args.post_check_speed,
        speed_acc_threshold=args.speed_acc_threshold,
        post_check_min_length=args.post_check_min_length,
        trace_events_path=args.perfetto_trace_events,
        trace_json_path=args.perfetto_trace_json,
        trace_record_ids=args.trace_record_ids,
    )


def append_hydra_output_overrides(cmd, exp_name, keep_hydra_outputs):
    if keep_hydra_outputs:
        return
    hydra_run_dir = Path(tempfile.gettempdir()) / "phc_hydra_runs" / exp_name
    cmd.extend(
        [
            f"hydra.run.dir={hydra_run_dir.as_posix()}",
            "hydra.output_subdir=null",
        ]
    )


def append_rendering_output_overrides(cmd, exp_name, keep_renderings):
    if keep_renderings:
        return
    rendering_root = Path(tempfile.gettempdir()) / "phc_renderings" / exp_name
    cmd.append(f"+rendering_output_root={rendering_root.as_posix()}")


def run_single_clip_local(
    record,
    clip_file,
    input_root,
    phc_motion_root,
    repaired_root,
    states_root,
    python_exec,
    gravity_axis,
    keep_shape,
    primitive_model_path,
    composer_checkpoint_path,
    episode_length,
    render_o3d,
    gym_viewer,
    no_virtual_display,
    gpu_id,
    keep_hydra_outputs,
    keep_renderings,
    post_check_speed,
    speed_acc_threshold,
    post_check_min_length,
    trace_events_path,
    trace_json_path,
    trace_record_ids,
):
    from scripts.data_process.batch_repair_zitai import sanitize_name
    from scripts.data_process.convert_zitai_to_phc import ROBOT_CFG, _build_key, convert_clip
    from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot

    clip_file = Path(clip_file)
    rel_parent = clip_file.parent.relative_to(input_root)
    base_name = clip_file.stem
    state_rel_path = rel_parent / f"{base_name}.pkl"

    motion_pkl = Path(phc_motion_root) / rel_parent / f"{base_name}.pkl"
    repaired_npz = Path(repaired_root) / rel_parent / f"{base_name}_repaired.npz"
    exp_name = sanitize_name(f"repair_{rel_parent.as_posix()}_{base_name}".replace("/", "__"))
    hydra_log = Path(states_root).parent / "hydra_logs" / rel_parent / f"{base_name}.log"
    trace = PerfettoTrace(trace_events_path, trace_json_path, process_name="ext_phc_repair")
    trace_record_id = str(record.get("key") or record.get("name") or "")
    trace_enabled = trace.enabled and trace_record_id in trace_record_ids
    if trace_enabled:
        trace.instant(
            "repair_item_selected",
            "repair",
            {
                "record_id": trace_record_id,
                "clip_file": clip_file.as_posix(),
            },
        )

    with trace.span(
        f"repair_item:{trace_record_id or base_name}",
        "repair",
        {
            "record_id": trace_record_id,
            "clip_file": clip_file.as_posix(),
            "gpu_id": gpu_id,
        },
    ) if trace_enabled else contextlib.nullcontext():
        with trace.span("convert_clip_to_motion", "repair") if trace_enabled else contextlib.nullcontext():
            smpl_local_robot = LocalRobot(ROBOT_CFG, data_dir="data/smpl")
            motion_dict = {
                _build_key(clip_file.parent, clip_file): convert_clip(
                    clip_dir=clip_file,
                    smpl_local_robot=smpl_local_robot,
                    force_neutral=not keep_shape,
                    target_fps=30,
                    gravity_axis=gravity_axis,
                )
            }
            motion_pkl.parent.mkdir(parents=True, exist_ok=True)
            import joblib
            joblib.dump(motion_dict, motion_pkl)

        use_headless = not (gym_viewer or render_o3d)
        cmd = [
            python_exec,
            "phc/run_hydra.py",
        "learning=im_mcp_big",
        f"exp_name={exp_name}",
        f"output_path={Path(composer_checkpoint_path).parent.as_posix()}",
        "env=env_im_getup_mcp",
        "robot=smpl_humanoid",
        "env.zero_out_far=False",
        "robot.real_weight_porpotion_boxes=False",
        "env.num_prim=3",
        f"env.motion_file={motion_pkl.as_posix()}",
        f"env.models=['{primitive_model_path}']",
        "env.num_envs=1",
        f"headless={'True' if use_headless else 'False'}",
        "epoch=-1",
        "test=True",
        f"render_o3d={'True' if render_o3d else 'False'}",
        f"no_virtual_display={'True' if no_virtual_display else 'False'}",
        "auto_record_motion=True",
        "auto_quit_after_record=True",
        f"repair_state_root={Path(states_root).as_posix()}",
        f"env.episode_length={episode_length}",
        "env.enableEarlyTermination=False",
    ]
        append_hydra_output_overrides(cmd, exp_name, keep_hydra_outputs)
        append_rendering_output_overrides(cmd, exp_name, keep_renderings)
        env = build_subprocess_env(python_exec)
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            cmd.extend([
                "device_id=0",
                "rl_device=cuda:0",
            ])
        clear_old_state_files(states_root, exp_name, state_rel_path=state_rel_path)
        hydra_log.parent.mkdir(parents=True, exist_ok=True)
        with hydra_log.open("a", encoding="utf-8") as hydra_log_f:
            hydra_log_f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} :: {exp_name} =====\n")
            hydra_log_f.write("CMD: " + " ".join(cmd) + "\n")
            hydra_log_f.flush()
            with trace.span("run_hydra_wait_state", "repair") if trace_enabled else contextlib.nullcontext():
                proc = subprocess.Popen(
                    cmd,
                    cwd=REPO_ROOT.as_posix(),
                    env=env,
                    stdout=hydra_log_f,
                    stderr=subprocess.STDOUT,
                )
                try:
                    state_file = wait_for_state_file(states_root, exp_name, state_rel_path=state_rel_path, proc=proc)
                except Exception as exc:
                    raise RuntimeError(f"{exc}. See hydra log: {hydra_log}") from exc
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

        with trace.span("export_repaired_npz", "repair") if trace_enabled else contextlib.nullcontext():
            export_result = export_longest_state_to_npz(
                state_file,
                repaired_npz,
                reference_npz=clip_file,
                gravity_axis=gravity_axis,
                post_check_speed=post_check_speed,
                speed_acc_threshold=speed_acc_threshold,
                post_check_min_length=post_check_min_length,
                repair_start_index=record.get("start_index"),
                repair_period=record.get("period"),
                return_metadata=True,
            )
    return {
        "clip_file": clip_file.as_posix(),
        "motion_pkl": motion_pkl.as_posix(),
        "state_file": state_file.as_posix(),
        "repaired_npz": export_result.get("repaired_npz", repaired_npz.as_posix()),
        "repaired_npzs": export_result.get("repaired_npzs", [export_result.get("repaired_npz", repaired_npz.as_posix())]),
        "hydra_log": hydra_log.as_posix(),
        "longest_key": export_result.get("longest_key"),
        "speed_check": export_result.get("speed_check"),
    }


def run_jobs(records, args, paths):
    success = []
    failures = []

    def _run(record):
        job = build_job(record, args, paths)
        result = run_single_clip_local(**job)
        return record, result

    if args.parallel_workers == 1:
        for record in records:
            try:
                item, result = _run(record)
                success.append({**item, **result, "status": "success"})
            except SpeedCheckFailedError as exc:
                failures.append(
                    {
                        **record,
                        "status": "failed_post_check_speed",
                        "error": str(exc),
                        "speed_check": exc.speed_check,
                    }
                )
            except Exception as exc:
                failures.append({**record, "status": "failed_repair", "error": str(exc)})
        return success, failures

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel_workers) as executor:
        future_map = {executor.submit(_run, record): record for record in records}
        for future in concurrent.futures.as_completed(future_map):
            record = future_map[future]
            try:
                item, result = future.result()
                success.append({**item, **result, "status": "success"})
            except SpeedCheckFailedError as exc:
                failures.append(
                    {
                        **record,
                        "status": "failed_post_check_speed",
                        "error": str(exc),
                        "speed_check": exc.speed_check,
                    }
                )
            except Exception as exc:
                failures.append({**record, "status": "failed_repair", "error": str(exc)})

    return success, failures


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default=CFG_PATH_DEFAULT)
    parser.add_argument("--source_root", action="append", default=None, help="Root containing original npz files. Repeatable.")
    parser.add_argument("--prepared_root", type=str, default=None, help="Temporary sliced npz root. Default: <results_repair_pth>/_prepared_inputs")
    parser.add_argument("--phc_motion_root", type=str, default="output/repaired_motion_inputs")
    parser.add_argument("--states_root", type=str, default="output/repaired_states")
    parser.add_argument("--python_exec", type=str, default=sys.executable)
    parser.add_argument("--gravity_axis", type=str, default="neg_y", choices=["neg_y", "neg_z"])
    parser.add_argument("--keep_shape", action="store_true")
    parser.add_argument("--primitive_model_path", type=str, default="output/HumanoidIm/phc_3/Humanoid.pth")
    parser.add_argument("--composer_checkpoint_path", type=str, default="output/HumanoidIm/phc_comp_3/Humanoid.pth")
    parser.add_argument("--episode_length", type=int, default=1000)
    parser.add_argument("--render_o3d", action="store_true")
    parser.add_argument("--gym_viewer", action="store_true")
    parser.add_argument("--no_virtual_display", action="store_true")
    parser.add_argument("--parallel_workers", type=int, default=1)
    parser.add_argument("--gpu_ids", type=str, default="0", help="Comma-separated GPU ids for parallel PHC jobs, e.g. 0,1,2,3")
    parser.add_argument("--summary_name", type=str, default=SUMMARY_SUCCESS_NAME)
    parser.add_argument("--failure_name", type=str, default=SUMMARY_FAILURE_NAME)
    parser.add_argument("--watch", action="store_true", help="Keep watching fail_filter for new work instead of exiting after one batch.")
    parser.add_argument("--watch_interval", type=int, default=10, help="Seconds to sleep between watch scans when no new work is found.")
    parser.add_argument(
        "--post_check_speed",
        action="store_true",
        help="Run a linear-acceleration speed check on repaired SMPL results before writing npz.",
    )
    parser.add_argument(
        "--speed_acc_threshold",
        type=float,
        default=14.7,
        help="Linear-acceleration threshold used by --post_check_speed.",
    )
    parser.add_argument(
        "--post_check_min_length",
        type=int,
        default=90,
        help="Minimum frame count for repaired slices kept after --post_check_speed.",
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
    parser.add_argument("--perfetto_trace_events", type=str, default=None)
    parser.add_argument("--perfetto_trace_json", type=str, default=None)
    parser.add_argument("--perfetto_trace_targets_path", type=str, default=None)
    return parser.parse_args()


def build_counts(runtime_state):
    failure_records = runtime_state["failure_records"]
    return {
        "prepared": len(runtime_state["success_records"]) + sum(
            1 for item in failure_records if item.get("status") in REPAIR_FAILURE_STATUSES
        ),
        "repair_success": len(runtime_state["success_records"]),
        "repair_failed": sum(1 for item in failure_records if item.get("status") == "failed_repair"),
        "post_check_speed_failed": sum(
            1 for item in failure_records if item.get("status") == "failed_post_check_speed"
        ),
        "prepare_failed": sum(1 for item in failure_records if item.get("status") == "failed_prepare"),
    }


def repair_outputs_exist(results_repair_pth, output_dir_name, output_basename):
    repair_dir = Path(results_repair_pth) / str(output_dir_name)
    legacy_path = repair_dir / f"{output_basename}_repaired.npz"
    if legacy_path.exists():
        return True
    period_name = str(output_basename).split("-", 1)[0] if output_basename else "period_0"
    return any(repair_dir.glob(f"{period_name}-*_validated.npz"))


def collect_batch(args, paths, source_roots, gpu_ids, source_index, runtime_state):
    fail_filter_pth = paths["fail_filter_pth"]

    manifest_files = collect_manifest_files(fail_filter_pth)
    if not manifest_files:
        return [], []

    prepared_records = []
    prepare_failures = []
    for manifest_path in manifest_files:
        cache_entry = get_manifest_cache_entry(manifest_path, paths, source_roots, source_index, runtime_state)
        for failure in cache_entry["parse_failures"]:
            if failure["uid"] not in runtime_state["failure_uids"]:
                prepare_failures.append(failure)

        for item in cache_entry["resolved_entries"]:
            uid = item["uid"]
            if uid in runtime_state["completed_uids"] or uid in runtime_state["failure_uids"]:
                continue

            repaired_npz = item["repaired_npz"]
            repaired_key = repaired_npz.resolve().as_posix()
            if (
                repaired_key in runtime_state["repaired_paths"]
                or repaired_npz.exists()
                or repair_outputs_exist(paths["results_repair_pth"], item["output_dir_name"], item["output_basename"])
            ):
                runtime_state["repaired_paths"].add(repaired_key)
                runtime_state["completed_uids"].add(uid)
                continue

            entry = item["entry"]
            try:
                slice_info = slice_npz(
                    source_npz=item["source_npz"],
                    output_npz=item["prepared_npz"],
                    start_index=entry.get("start_index"),
                    end_index=entry.get("end_index"),
                )
                prepared_records.append({
                    "manifest": item["manifest"],
                    "manifest_name": item["manifest_name"],
                    "entry_index": item["entry_index"],
                    "key": item["key"],
                    "source_npz": item["source_npz"].as_posix(),
                    "output_dir_name": item["output_dir_name"],
                    "output_basename": item["output_basename"],
                    "prepared_npz": item["prepared_npz"].as_posix(),
                    "gpu_id": gpu_ids[len(prepared_records) % len(gpu_ids)],
                    "uid": uid,
                    **slice_info,
                })
            except Exception as exc:
                prepare_failures.append({
                    "manifest": item["manifest"],
                    "manifest_name": item["manifest_name"],
                    "entry_index": item["entry_index"],
                    "key": item["key"],
                    "status": "failed_prepare",
                    "error": str(exc),
                    "uid": build_failure_uid(item["manifest"], item["entry_index"], item["key"], "failed_prepare"),
                })

    return prepared_records, prepare_failures


def update_runtime_state(runtime_state, prepare_failures, repair_success, repair_failures):
    changed = False

    for item in repair_success:
        uid = item.get("uid")
        if uid and uid not in runtime_state["completed_uids"]:
            runtime_state["completed_uids"].add(uid)
            runtime_state["success_records"].append(item)
            repaired_npz = item.get("repaired_npz")
            if repaired_npz:
                runtime_state["repaired_paths"].add(Path(repaired_npz).resolve().as_posix())
            for repaired_npz in item.get("repaired_npzs") or []:
                runtime_state["repaired_paths"].add(Path(repaired_npz).resolve().as_posix())
            changed = True

    for item in prepare_failures + repair_failures:
        uid = item.get("uid")
        if uid and uid not in runtime_state["failure_uids"]:
            runtime_state["failure_uids"].add(uid)
            runtime_state["failure_records"].append(item)
            changed = True

    runtime_state["summary_dirty"] = runtime_state["summary_dirty"] or changed
    return changed


def write_summaries(args, paths, runtime_state):
    results_repair_pth = paths["results_repair_pth"]
    results_fail_pth = paths["results_fail_pth"]
    summary = {
        "config": runtime_state["config"],
        "counts": build_counts(runtime_state),
        "success": runtime_state["success_records"],
    }
    failures_doc = {
        "counts": summary["counts"],
        "failures": runtime_state["failure_records"],
    }
    dump_yaml(results_repair_pth / args.summary_name, summary)
    failure_summary_path = results_fail_pth / args.failure_name
    dump_structured(failure_summary_path, failures_doc)
    if failure_summary_path.suffix.lower() == ".json":
        failure_summary_path.with_suffix(".yaml").unlink(missing_ok=True)
    runtime_state["summary_dirty"] = False
    return summary


def main():
    args = parse_args()
    args.trace_record_ids = load_trace_targets(args.perfetto_trace_targets_path)
    cfg = load_yaml(args.cfg)

    fail_filter_pth = Path(cfg["fail_filter_pth"]).resolve()
    results_repair_pth = Path(cfg["results_repair_pth"]).resolve()
    results_fail_pth = Path(cfg["results_fail_pth"]).resolve()
    source_roots = list(args.source_root or cfg.get("source_roots", []))

    if not source_roots:
        raise ValueError("No source roots configured. Add source_roots to data_read_cfg.yaml or pass --source_root")

    gpu_ids = [token.strip() for token in str(args.gpu_ids).split(",") if token.strip()]
    if not gpu_ids:
        gpu_ids = ["0"]

    prepared_root = Path(args.prepared_root).resolve() if args.prepared_root else (fail_filter_pth.parent / "_prepared_inputs")
    paths = {
        "fail_filter_pth": fail_filter_pth,
        "results_repair_pth": results_repair_pth,
        "results_fail_pth": results_fail_pth,
        "prepared_root": prepared_root,
        "phc_motion_root": Path(args.phc_motion_root).resolve(),
        "states_root": Path(args.states_root).resolve(),
    }

    source_index = build_source_index(source_roots)
    runtime_state = initialize_runtime_state(paths, args, source_roots)

    while True:
        prepared_records, prepare_failures = collect_batch(args, paths, source_roots, gpu_ids, source_index, runtime_state)
        repair_success, repair_failures = run_jobs(prepared_records, args, paths) if prepared_records else ([], [])
        changed = update_runtime_state(runtime_state, prepare_failures, repair_success, repair_failures)
        if changed or runtime_state["summary_dirty"]:
            summary = write_summaries(args, paths, runtime_state)
        else:
            summary = {"counts": build_counts(runtime_state)}

        print(f"Prepared: {summary['counts']['prepared']}")
        print(f"Repair success: {summary['counts']['repair_success']}")
        print(f"Repair failures: {summary['counts']['repair_failed']}")
        print(f"Post-check speed failures: {summary['counts']['post_check_speed_failed']}")
        print(f"Prepare failures: {summary['counts']['prepare_failed']}")
        print(f"Success summary: {(results_repair_pth / args.summary_name).as_posix()}")
        print(f"Failure summary: {(results_fail_pth / args.failure_name).as_posix()}")

        if not args.watch:
            break

        if summary['counts']['prepared'] == 0 and summary['counts']['prepare_failed'] == 0:
            time.sleep(args.watch_interval)
            continue

        time.sleep(args.watch_interval)


if __name__ == "__main__":
    main()
