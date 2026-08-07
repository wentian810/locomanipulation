"""Call MegaHunter without letting it rewrite the existing local human pose."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videomimic-root", required=True, type=Path)
    parser.add_argument("--world-env-path", required=True, type=Path)
    parser.add_argument("--bbox-dir", required=True, type=Path)
    parser.add_argument("--pose2d-dir", required=True, type=Path)
    parser.add_argument("--smpl-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--gender", choices=("male", "female", "neutral"), default="neutral")
    parser.add_argument("--gradient-thr", type=float, default=0.15)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument(
        "--optimize-root-rotation",
        action="store_true",
        help="Allow per-frame root rotation changes. Disabled by default to preserve a global external-human coordinate bridge.",
    )
    parser.add_argument(
        "--preserve-external-root-translation",
        action="store_true",
        help="Keep adapter-supplied root translations rather than replacing them with per-frame 2-D fitting results.",
    )
    parser.add_argument("--optimize-local-pose", action="store_true")
    parser.add_argument("--vis", action="store_true")
    args = parser.parse_args()
    root = args.videomimic_root.resolve()
    # MegaHunter's upstream visualization helper resolves
    # ``./assets/body_models`` relative to the current working directory.
    # This sidecar is launched from the host project, so make that implicit
    # path unambiguous before importing/running the upstream module.
    os.chdir(root)
    sys.path.insert(0, str(root))
    from stage2_optimization.megahunter_optimization import run_jax_alignment

    model = root / "assets" / "body_models" / "smpl" / f"SMPL_{args.gender.upper()}.pkl"
    if not model.is_file():
        parser.error(f"SMPL model unavailable: {model}")
    run_jax_alignment(
        world_env_path=str(args.world_env_path),
        bbox_dir=str(args.bbox_dir),
        pose2d_dir=str(args.pose2d_dir),
        smpl_dir=str(args.smpl_dir),
        out_dir=str(args.out_dir),
        num_iterations=args.iterations,
        gradient_thr=args.gradient_thr,
        optimize_root_rotation=args.optimize_root_rotation,
        optimize_local_pose=args.optimize_local_pose,
        post_temporal_smoothing=True,
        preserve_external_root_translation=args.preserve_external_root_translation,
        smpl_model_path=str(model),
        vis=args.vis,
    )


if __name__ == "__main__":
    main()
