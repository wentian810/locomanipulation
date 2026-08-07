"""CLI for exporting existing human results into VideoMimic input directories."""

from __future__ import annotations

import argparse
from pathlib import Path

from videomimic_adapter import AdapterError, export_videomimic_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-dir", required=True, type=Path)
    parser.add_argument("--work-video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--motion-npz", type=Path, default=None)
    parser.add_argument("--camera-npz", type=Path, default=None)
    parser.add_argument("--human-output-dir", type=Path, default=None)
    parser.add_argument("--clip-id", default=None)
    parser.add_argument("--person-id", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0, help="0 exports the complete clip")
    parser.add_argument("--frame-stride", type=int, default=1, help="Uniform scene-evidence sampling stride")
    parser.add_argument("--mask-mode", choices=("bbox", "sam2"), default="bbox")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--static-camera-warn-threshold", type=float, default=0.05)
    args = parser.parse_args()
    try:
        report = export_videomimic_inputs(
            clip_dir=args.clip_dir,
            work_video=args.work_video,
            output_root=args.output,
            motion_path=args.motion_npz,
            camera_path=args.camera_npz,
            human_output_dir=args.human_output_dir,
            clip_id=args.clip_id,
            person_id=args.person_id,
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
            mask_mode=args.mask_mode,
            overwrite=args.overwrite,
            static_camera_warn_threshold=args.static_camera_warn_threshold,
        )
    except AdapterError as exc:
        parser.error(str(exc))
    print(f"[PASS] VideoMimic inputs exported: {report['frame_count']} frames -> {args.output}")
    for warning in report["warnings"]:
        print(f"[WARN] {warning}")


if __name__ == "__main__":
    main()
