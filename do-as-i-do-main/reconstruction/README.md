# Do as I Do · Reconstruction

Hand and object **reconstruction + 6-DoF pose tracking** from a single hand-object demo video. Given a
video, a reference frame, an object name, and the anchor hand, the pipeline segments the object
and hand, reconstructs a 3D mesh for the object, estimates per-frame object pose, and reconstructs the hand (HaWoR).

## Layout

```
reconstruction/
├── run_pipeline.sh          # the driver 
├── config/paths.sh          # ← the ONLY file you edit to relocate things
├── scripts/                 # 7 standalone first-party stage scripts
├── modules/             # submodules with our changes
├── weights/                 # model weights 
├── env/                     # conda env docs (see env/README.md)
└── setup/                   # 00 submodules · 01 envs · 02 weights
```

## Requirements
- An NVIDIA GPU with ≥ 32 GB VRAM.
- HuggingFace auth with access to the repos `facebook/sam-3d-objects` and `facebook/sam3`; Plus a MANO download
  (https://mano.is.tue.mpg.de) for HaWoR.
- SAM 3 segmentation is implemented with a click-based GUI needing an X display.
  The launcher uses `SAM3_DISPLAY`, then `DISPLAY`, then `:0`; on a headless host,
  use X forwarding or replace the interactive prompt with explicit points/text.

## Setup (one time)

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/malik-group/do-as-i-do.git
cd do-as-i-do/reconstruction
./setup/00_init_submodules.sh                 # only needed if you didn't clone with recursive submodules
./setup/01_create_envs.sh                     # FALLBACK build of all 4 envs (sam3, sam3d, hawor, tapnet) — prefer each fork's own setup, see "Setting up the conda envs" below
./setup/02_fetch_weights.sh --download        # fetch weights (needs hf auth)
```
**Setting up the conda envs.** The recommended route is to build each env by following
its fork's own setup instructions (the repos vendored under `modules/`):

- `sam3`  → [malik-group/sam3](https://github.com/malik-group/sam3) (`modules/sam3`)
- `sam3d` → [malik-group/sam-3d-objects](https://github.com/malik-group/sam-3d-objects) (`modules/sam-3d-objects`, see its `doc/setup.md`) 
- `hawor` → [malik-group/HaWoR](https://github.com/malik-group/HaWoR) (`modules/HaWoR`)
- `tapnet` → [malik-group/tapnet](https://github.com/malik-group/tapnet) (`modules/tapnet`)

See [`env/README.md`](env/README.md) for the per-env cu128 recipes (or `./setup/01_create_envs.sh` to script them).

After the `sam3d` env is built, two manual Stage-2 steps are needed: un-shadow the repo's
`notebook/` package (`pip uninstall -y notebook`) and build the Mip-Splatting
`diff_gaussian_rasterization` for the renderer's `inria` backend. Commands in
[`env/README.md`](env/README.md).

Review `config/paths.sh`.


## Run

```bash
./run_pipeline.sh VIDEO_PATH [FRAME_N] [OBJECT] [ANCHOR_HAND]
# e.g.
./run_pipeline.sh whisking/whisking.mp4 125 whisk right
```

### Details on Pipeline Stages
| # | stage | script | env |
|---|---|---|---|
| 0 | extract frames | ffmpeg | sam3 |
| 1 | SAM3 segmentation (object click + hand text) | `scripts/run_sam3_video.py` | sam3 |
| 2 | masks → 3D mesh | `modules/sam-3d-objects/generate_mesh_sam3d.py` | sam3d |
| 2 | MoGe pointmaps (reference frame + all frames) | `scripts/get_pointmap_dir.py` | sam3d |
| 2 | hand reconstruction | `modules/HaWoR/demo.py` | hawor |
| 2 | gravity estimation (GeoCalib) | `scripts/predict_video_gravity.py` | sam3d |
| 2.5 | velocity tracking | `scripts/tapir_velocity_tracking.py` | tapnet |
| 3 | guided pose prediction | `modules/Fast-SAM3D/track_object.py` | sam3d |
| 3 | project mesh / to camera frame | `scripts/run_project_mesh_combined.py`, `scripts/convert_layout_to_camera_frame.py` | sam3d |
| 4 | optimize translation/scale | `scripts/optimize_translation_scale.py` | sam3d |
| 4 | (optional) 3D viser viz | `scripts/visualize_3d.py` | sam3d |


### Outputs (under the video's directory)
`video_segmentation/masks/…`, per-object `.obj` meshes, `*_pointmap.npy` / `*_intrinsics.txt`,
HaWoR `all_hand_meshes.npz`, `gravity.json` (camera-frame up direction from GeoCalib),
and `obj_tracking_out/<object>/combined_visualization/`
with `layout.json → layout_camera_frame.json → layout_camera_frame_optimized.json` and projected
frames. This directory is the input to the [`retargeting/`](../retargeting/README.md) pipeline.


## Credits & licenses
The `modules/` contain external references with our changes to the sources, and each of them retain their upstream `LICENSE`. We gratefully
acknowledge the original authors.

| fork | upstream @ pinned commit | fork commit | license |
|---|---|---|---|
| `malik-group/sam-3d-objects` | facebookresearch/sam-3d-objects @ `81a8237` | `875b010` | SAM License (Meta) |
| `malik-group/Fast-SAM3D`     | wlfeng0509/Fast-SAM3D @ `c0f99e8`           | `823d478` | MIT (+ embedded SAM-3D under SAM License) |
| `malik-group/HaWoR`          | ThunderVVV/HaWoR @ `de90272`                | `2c3fa0c` | CC BY-NC-ND 4.0 |
| `malik-group/tapnet`         | google-deepmind/tapnet @ `96d3f84`          | `f2f8888` | Apache-2.0 |
| `malik-group/sam3`           | facebookresearch/sam3 @ `757bbb0`           | `b8e18f5` | SAM License (Meta) |
