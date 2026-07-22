# Human / G1 + Sharpa artifact policy

This policy is for large-scale dataset production. The execution workspace is
disposable; a verified product bundle is the only retained numerical result.

## During one clip run

The pipeline may create converted, locomotion, PHC, hand-filter, camera and
robot cache files. They are stage-local inputs needed for retries and must not
be treated as dataset assets. Keep them below the configured scratch output.

## Retained per clip

`assets/<dataset>/<clip>/` retains:

- `motion.npz`: the only NPZ payload. Fields are prefixed by component,
  e.g. `human__*`, `robot__*`, `sharpa__*`, and `camera__*`. A
  `human_phc__*` namespace is added only when PHC is not already the
  canonical human source.
- `preview_2x2.mp4` when previews are enabled.
- `manifest.json`, `checksums.sha256`, `quality_report.json`, and
  `pipeline_config.yaml`.
- Object mesh files only if the object line is enabled in a future export.

For the default PHC-grounded G1 pipeline, `human__*` is already the final PHC
body trajectory, so it is not duplicated. Robot motion and external Sharpa
motion are namespaces in the same file.

## Quality recording

The root-level batch record is `quality_overview.csv` (and the equivalent
`quality_overview.json`). It contains one compact pipeline-quality indicator
per clip: 0--100 PQI score, A--E grade, accept/review/reject verdict, weakest
stage, and a stage overview. Low-level evidence remains only in the clip's
`quality_report.json` and is not duplicated in the batch summary.

## Batch execution

Use `configs/pipelines/human_sharpa_batch.yaml`. Its output root is scratch and
`prune_workspace_after_export: true` removes each clip workspace only after
the exported bundle has passed checksum verification. Never point `product.root`
inside the scratch output root.

## Motion contract

The default embodiment is `sharpa`, defined as Unitree G1 body plus external
Sharpa Wave hands at native scale. `001_final.npz` is a temporary zero-copy
selection link: it targets PHC-smoothed/global-grounded motion when PHC
succeeds, and targets `001_smoothed.npz` when the current PHC attempt fails.
`final_motion_selection.json` records which case occurred. G1 receives one
global foot-geometry height calibration; do not apply a per-frame root-height
projection after PHC.
