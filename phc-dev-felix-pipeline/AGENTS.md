# ext-phc Agent Guide

## What This Repository Does

`ext-phc` is the PHC / Isaac Gym backend used by the `smpl_work` pipeline for stage2 repair.

In the production flow under `/data/smpl_work`:

1. `smplpipeline` stage1 writes filter outputs under a run directory such as `.../motionx/filter/`.
2. `smplpipeline` dispatcher calls `code/scripts/run_two_stage_repair.py`.
3. That dispatcher drives `ext-phc/scripts/data_process/repair_from_fail_filter.py`.
4. `repair_from_fail_filter.py` prepares sliced `.npz` clips, launches PHC jobs through `phc/run_hydra.py`, and writes repaired outputs plus YAML summaries.

Use this repo as the stage2 repair engine first. Treat direct training or demo commands as secondary unless the task explicitly asks for them.

## Repository Map

- `phc/`: core runtime, envs, configs, and the low-level Hydra entrypoint.
- `phc/run_hydra.py`: underlying PHC execution entrypoint used by repair jobs.
- `phc/data/cfg/`: Hydra config tree.
  - `sim/default_sim.yaml`
  - `env/env_im_getup_mcp.yaml`
  - `control/default_control.yaml`
- `scripts/data_process/repair_from_fail_filter.py`: production stage2 repair watcher / dispatcher target.
- `scripts/data_process/batch_repair_zitai.py`: direct batch repair tool for one input tree.
- `scripts/data_process/export_repaired_smpl_npz.py`: exports repaired state dumps back to SMPL `.npz`.
- `docker-compose.yml`: container build and bind mounts for code, data, output, and runs.
- `checkpoints/`, `data/`, `sample_data/`: required runtime assets; do not assume a fresh clone contains the full payload.
- `output/`, `runs/`: generated artifacts; do not commit routine outputs.
- `isaacgym/`: vendored tree in this repo. Note that current `docker-compose.yml` still references `../isaacgym` as an additional build context.

## Current Build And Runtime Expectations

Run from the repo root.

Typical build variables in `/data/smpl_work`:

```bash
PHC_CODE_PATH=/data/smpl_work/ext-phc \
PHC_DATA_PATH=/data/smpl_work/ext-phc/data \
PHC_OUTPUT_PATH=/data/smpl_work/ext-phc/output \
PHC_RUNS_PATH=/data/smpl_work/ext-phc/runs \
docker compose build phc
```

Important caveat:

- `docker-compose.yml` currently sets `additional_contexts.isaacgym_src: ../isaacgym`.
- In this workspace the repo also contains `ext-phc/isaacgym`, but the compose file does not point at it.
- On a fresh machine, either provide a sibling `../isaacgym` path that matches the compose file, or update the compose file before rebuilding.

Standalone compose uses container name `phc`. When `smplpipeline` manages the same runtime, you may instead see a wrapper container such as `ext_phc_server`.

## Production Entry Points

### `repair_from_fail_filter.py`

Use this for real pipeline work.

It reads:

- `fail_filter_pth`
- `results_repair_pth`
- `results_fail_pth`

from `data/cfg/data_read_cfg.yaml` or explicit CLI arguments, prepares sliced inputs, runs PHC jobs, and writes:

- repaired clips under `results_repair_pth/<seq_dir>/*_repaired.npz`
- success summary under `results_repair_pth/repair_summary.yaml`
- failure summary under `results_fail_pth/repair_failures.yaml`
- temporary sliced clips under `_prepared_inputs/`

Important arguments:

- `--source_root`
- `--prepared_root`
- `--phc_motion_root`
- `--states_root`
- `--parallel_workers`
- `--gpu_ids`
- `--watch`
- `--watch_interval`

### `batch_repair_zitai.py`

Use this for isolated debugging outside the full `smplpipeline` stage2 loop.

Important arguments:

- `--input_root`
- `--repaired_root`
- `--states_root`
- `--parallel_workers`
- `--limit`
- `--skip_existing`
- `--primitive_model_path`
- `--composer_checkpoint_path`

Behavioral constraints enforced by code:

- `--parallel_workers >= 1`
- `--parallel_workers > 1` cannot be combined with `--reuse_gym_viewer`
- `--parallel_workers > 1` cannot be combined with `--gym_viewer`
- `--parallel_workers > 1` cannot be combined with `--render_o3d`

## Config Notes That Matter For Debugging

Checked-in defaults in the current branch:

- `phc/data/cfg/sim/default_sim.yaml`: `physx.step_dt: 1/60`
- `phc/data/cfg/env/env_im_getup_mcp.yaml`: `controlFrequencyInv: 2`
- `phc/data/cfg/control/default_control.yaml`: `decimation: 4`

Changing these affects repair behavior, control frequency, and exported FPS. Do not treat them as harmless speed knobs.

Also note:

- `data/cfg/data_read_cfg.yaml` still contains machine-local example paths and is not a trustworthy production default on a new machine.
- Prefer explicit paths from the caller when debugging cross-machine issues.

## Working Rules For Agents

- Stay at repo root when running commands so relative config and checkpoint paths resolve.
- Prefer targeted smoke tests over large reruns.
- For pipeline issues, debug the handoff in this order:
  1. input manifests and source `.npz` resolution
  2. prepared clip generation
  3. `phc/run_hydra.py` launch success
  4. repaired `.npz` export
  5. summary YAML updates
- When repair appears stuck, inspect:
  - active `run_hydra.py` processes
  - `repair_summary.yaml`
  - `repair_failures.yaml`
  - prepared input directories
  - output timestamps under `results_repair_pth`
- Do not commit generated artifacts, downloaded models, secrets, or large archives unless the user explicitly asks for that.
