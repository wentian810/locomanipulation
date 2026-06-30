# Robot hand assets and integration status

The source hand estimator (HaMeR, WiLoR or Hand4Whole++) and the target robot
hand are independent. Estimators produce MANO pose/joints; each robot hand
needs its own kinematic retargeting configuration, joint limits and wrist mount.

| Target hand | Local status | Official source | Integration note |
|---|---|---|---|
| Unitree G1 Dex3-1 | Integrated | `GMR-master/assets/unitree_g1/g1_mocap_29dof_with_hands.xml` | Select with `GMR_HAND_MODEL=g1`. Full G1 wrist orientation is preserved; fingers currently use the local MANO mapping. |
| Unitree standalone Dex3-1 | Downloaded | `unitreerobotics/xr_teleoperate` | Official left/right URDF and DexPilot/vector mappings are retained for the next target-space IK upgrade. |
| 因时 Inspire | Optional download | `unitreerobotics/xr_teleoperate` | Official integration URDF/config; do not confuse this with 逐际动力. |
| 逐际动力 LimX HU_D04 hand | Optional download | `limxdynamics/humanoid-description` | Five-finger hand is embedded in the HU_D04 full-body URDF; it must be extracted or mounted carefully. |
| 强脑 BrainCo Revo2 | Downloaded and integrated | `BrainCoTech/brainco-description` + `unitreerobotics/xr_teleoperate` | Select with `GMR_HAND_MODEL=brainco`. Uses Revo2 because BrainCo's official G1 adaptation targets Revo2. The IK respects six active motors per hand and the five official mimic rules. |
| 强脑 BrainCo Revo3 | Not selected | `BrainCoTech/brainco-description` | Revo3 is not downloaded by default because the official G1 integration path currently targets Revo2. |

Audit local availability:

```bash
bash GMR-master/scripts/setup_robot_hand_assets.sh status
```

Download shallow, sparse official checkouts:

```bash
bash GMR-master/scripts/setup_robot_hand_assets.sh brainco
bash GMR-master/scripts/setup_robot_hand_assets.sh download
```

Downloaded vendor trees are placed under
`GMR-master/third_party/robot_hands/`. Vendor trees remain unmodified and are
referenced in place.

## Choosing the GMR hand

The estimator and robot hand are independent. The same command can use any of
the three GMR targets:

```bash
# H1 + Sharpa Wave
GMR_HAND_MODEL=sharpa bash GVHMR-hand/GVHMR-main/tools/pipeline/run_batch_dataset6_wilor.sh

# G1 + native Dex3-1
GMR_HAND_MODEL=g1 bash GVHMR-hand/GVHMR-main/tools/pipeline/run_batch_dataset6_wilor.sh

# G1 + BrainCo Revo2
GMR_HAND_MODEL=brainco bash GVHMR-hand/GVHMR-main/tools/pipeline/run_batch_dataset6_wilor.sh
```

The same `GMR_HAND_MODEL` switch works with the HaMeR and Hand4Whole++ wrapper
scripts. Their default output roots include the estimator and selected target
hand, so the three results do not overwrite each other.

BrainCo's public description repository currently states that license
information is still pending. Keep the vendor checkout local and review the
upstream license before redistribution.
