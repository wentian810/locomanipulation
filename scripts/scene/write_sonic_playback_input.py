#!/usr/bin/env python3
"""Create a deterministic control-input replay for SONIC offline motions.

The released controller intentionally loads offline CSV motions paused at
frame zero.  Its documented ``--playback-input-file`` format allows a
headless run to issue the same start/play/stop state changes without a TTY or
a gamepad.  It does not alter the motion reference or any robot state.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--input-hz", type=float, default=100.0)
    parser.add_argument("--settle-s", type=float, default=0.5,
                        help="Time to keep play=false before the final stop command.")
    parser.add_argument("--motion-index", type=int, default=0)
    return parser.parse_args()


def row(
    motion_index: int,
    play: int,
    start: int,
    stop: int,
) -> str:
    # Fields follow G1Deploy::Input(): motion, frame, play, start, stop,
    # planner_enabled, planner_initialized, locomotion_mode, direction xyz,
    # facing xyz, speed, height.
    values = [motion_index, 0, play, start, stop, 0, 0, 0,
              0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    return ",".join(str(value) for value in values)


def main() -> None:
    args = parse_args()
    if args.duration_s <= 0.0 or args.input_hz <= 0.0 or args.settle_s < 0.0:
        raise ValueError("--duration-s and --input-hz must be positive; --settle-s non-negative")
    if args.motion_index < 0:
        raise ValueError("--motion-index must be non-negative")

    active_rows = max(1, int(round(args.duration_s * args.input_hz)))
    # Input is polled at 100 Hz while SONIC's state machine is sampled at
    # 50 Hz.  Keep the start request asserted during active replay so it
    # cannot be missed between those clocks.  Once CONTROL is entered the
    # flag has no further state-transition effect.
    rows = [row(args.motion_index, play=1, start=1, stop=0)
            for tick in range(active_rows)]
    settle_rows = int(round(args.settle_s * args.input_hz))
    rows.extend(row(args.motion_index, play=0, start=0, stop=0) for _ in range(settle_rows))
    rows.append(row(args.motion_index, play=0, start=0, stop=1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} replay rows to {args.output}")


if __name__ == "__main__":
    main()
