# Scene-added release

This branch starts from the human-reconstruction handover commit
`de65768739b747706106218c9a161a788d927b41`
(`docs/human-reconstruction-handover-20260806`).  It retains the original
GVHMR-hand → Locomotion/PHC → GMR pipeline, its handover documents, and the
original acceptance contract.

It adds the current scene-aware layer:

- `scripts/scene/`: VideoMimic evidence preparation, local static-scene
  reconstruction, coordinate/asset contracts, GMR chair-contact audits,
  MuJoCo `mj_step` scene playback, Sonic/HoloMotion physical-seat utilities,
  and batch-validation helpers.
- `scripts/run_pipeline_from_config.py`: scene sidecar and isolated
  PHC/GMR-scene orchestration.  Other root `scripts/` contain the v24 rollout,
  floor/contact audits, accepted-manifest, and synchronized 2×2 rendering
  tools.
- `configs/`: the scene/VideoMimic configurations used by the batch pipeline,
  in addition to the original human-only configurations.
- `phc-dev-felix-pipeline/`: source/config/documentation snapshot for the PHC
  path used by the orchestrator.  It intentionally does **not** include Isaac
  Gym, model/data downloads, sample pickles, cached results, or videos; see
  `SOURCE_ONLY_NOTICE.md` there.
- `patches/`: exact patches for the two external repositories that must be
  applied after checking out their pinned upstream revisions.

## Reproducing the external source revisions

```bash
git clone https://github.com/hongsukchoi/VideoMimic.git external/VideoMimic
git -C external/VideoMimic checkout cfded6d85dd55893074c844f74a49297e318980c
git -C external/VideoMimic apply ../../patches/videomimic_scene_added.patch

git clone https://github.com/HorizonRobotics/HoloMotion.git third_party/HoloMotion
git -C third_party/HoloMotion checkout 71ab7e976de23aa9bb351030dd61b05426d3443c
git -C third_party/HoloMotion apply ../../patches/holomotion_v24_evaluator.patch
```

The orchestrator supports explicit dependency paths in its configuration; do
not assume that a local checkout must use the server's historical absolute
paths.  Consult `patches/UPSTREAM_VERSIONS.md` and the handover documents
before running a batch.

## Intentional exclusions

This is a source release, not an artifact dump.  It adds no `scene_work/`,
VideoMimic data/assets/caches, HoloMotion checkpoints and meshes, PHC outputs
and Isaac Gym installation, videos/2×2 results, model weights, Python caches,
and all `*.bak*`/`*.orig` backup files.  The release therefore stays reviewable
and reproducible without publishing tens of gigabytes of generated data.

The inherited handover baseline retains three historical
`GMR-master/output/**/summary.csv` records that were already tracked in
`de65768`; they are metadata from that handover, not new scene-run products.
