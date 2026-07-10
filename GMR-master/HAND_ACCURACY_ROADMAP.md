# Hand accuracy roadmap (June 2026)

## What was wrong locally

Two pipeline mismatches were independent of the hand estimator:

1. The GVHMR panel rendered the temporal/finger-filtered MANO track, then
   `convert_to_npz.py` cleaned and interpolated it a second time for GMR.
   GMR therefore consumed a different gesture from the one shown by GVHMR.
   Backend wrappers now export the already-filtered track with
   `GVHMR_HAND_REFINE_MODE=raw`; reprojection confidence still controls IK
   weights, but no second hidden pose rewrite occurs.
2. G1 previously forced wrist pitch/yaw to zero and applied only palm roll.
   G1/Dex3 and G1/BrainCo now preserve the complete GMR wrist orientation.
   H1/Sharpa keeps its one-axis wrist-specific treatment.

WiLoR still uses all shared temporal and anatomical cleaning:

- 2D confidence and bbox continuity checks;
- temporal invalid-frame repair;
- quaternion wrist and MANO finger smoothing;
- joint-angle/joint-position step limits;
- open-hand rescue;
- normalized reprojection confidence;
- target-space robot IK confidence weighting, joint limits, velocity and
  acceleration limits.

The same source and target constraints are enabled for HaMeR and
Hand4Whole++; only the backend estimator and a small reliable-frame smoothing
weight differ.

## Constraint profiles

All three estimator wrappers accept one coordinated profile:

```bash
GVHMR_HAND_CONSTRAINT_PROFILE=balanced \
  bash GVHMR-hand/GVHMR-main/tools/pipeline/run_batch_dataset6_wilor.sh
```

- `responsive`: clear, large hands and fast gestures; shortest windows.
- `balanced`: default 30 FPS setting.
- `conservative`: distant, noisy or frequently occluded hands.

The source MANO stage handles observation errors and temporal repair. The
robot stage only handles morphology, joint limits and executable
velocity/acceleration. In particular, G1's old 25-frame second smoothing
window has been reduced to 3/5/9 frames by profile to avoid flattening a track
that was already filtered.

Sidecars now carry a single `left/right_hand_quality` score. Direct reliable
frames retain image-driven IK; weak observations are scaled to 0.35, and
temporally repaired/held frames to at most 0.20. This is a soft IK weight, not
another pose replacement or invalid-frame filter.

## Most relevant recent work

| Work | Useful idea for this pipeline | Integration judgment |
|---|---|---|
| Hand4Whole++ (2026) | Body-hand feature fusion and body-consistent wrist orientation | Already represented by the independent Hand4Whole++ backend. |
| DanceHMR (2026) | Temporal whole-body model with residual body/hand fusion and close-up-aware training | Very relevant, but no official runnable code was found at review time. Track rather than replace the current pipeline now. |
| GEM-X (2025/2026 release) | Video-native 77-joint whole-body estimator including hands, dynamic camera recovery and G1 retargeting | Best independent end-to-end baseline to evaluate next; SOMA output needs a converter rather than being mixed into MANO silently. |
| Dyn-HaMR (CVPR 2025) | Hierarchical initialization, handedness/hallucination prevention, generative two-hand prior and test-time multi-objective optimization | The tracker safeguards and optimization losses are directly reusable. The full system is heavier and oriented toward interacting/egocentric hands. |
| HaWoR (CVPR 2025) | Motion infilling through long occlusion and world/camera-motion decoupling | Useful for missing hands; not a direct drop-in for third-person full-body clips. |
| Pose-Guided Temporal Enhancement (CVPR 2025) | Temporal joint features fused back into dense visual features for low-resolution hands | Directly addresses small hand crops; requires a learned temporal model rather than a numeric filter. |
| UniHand (ICLR 2026) | Conditional diffusion prior combining images, MANO and 2D skeletons under occlusion | Strong candidate for optional offline gap infilling; do not apply it to reliable visible frames. |
| Hand3R (2026) | Persistent scene memory and metric hand/scene reconstruction | Promising for hand-object interaction, but not yet a practical replacement for the current full-body pipeline. |

Primary sources:

- https://arxiv.org/abs/2603.14726
- https://arxiv.org/abs/2605.18102
- https://github.com/NVlabs/GEM-X
- https://github.com/ZhengdiYu/Dyn-HaMR
- https://github.com/ThunderVVV/HaWoR
- https://openaccess.thecvf.com/content/CVPR2025/html/Fan_Pose-Guided_Temporal_Enhancement_for_Robust_Low-Resolution_Hand_Reconstruction_CVPR_2025_paper.html
- https://openreview.net/forum?id=upUl6hMYwy
- https://arxiv.org/abs/2602.03200

## Recommended next implementation order

1. Add a per-frame audit video that overlays the exact 2D observations,
   projected MANO joints and projected robot fingertips. This separates
   estimator error, wrist-frame error and morphology/IK error numerically.
2. Add high-confidence visible-frame test-time optimization using robust 2D
   keypoint loss, silhouette/hand-mask loss, MANO pose prior and temporal
   acceleration. Visible frames should get stronger image evidence and less
   smoothing, not more generic filtering.
3. Add sequence-level crop tracking: optical-flow propagation plus detector
   observations, crop-scale hysteresis and multi-frame feature fusion. This
   targets the small-hand/low-resolution failure described by the CVPR 2025
   temporal enhancement work.
4. Add handedness persistence and hallucination rejection modeled after the
   current Dyn-HaMR tracker update.
5. Use a learned motion prior only for low-confidence gaps or occlusion
   intervals. UniHand/HaWoR-style infilling should never overwrite a clear,
   well-observed hand merely to make motion smoother.
6. For object manipulation clips, add fingertip/object non-penetration and
   contact persistence after an object track is available.
7. Evaluate GEM-X as a fourth, fully independent whole-body baseline. Compare
   2D reprojection, hand acceleration, wrist continuity and robot fingertip
   error instead of judging only a montage.
