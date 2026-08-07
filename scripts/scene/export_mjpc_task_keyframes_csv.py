#!/usr/bin/env python3
"""Export ordered MuJoCo qpos keyframes from a generated MJPC task XML.

The output is a plain ``[T, nq]`` CSV for an external tracker smoke test.  It
copies only immutable reference keyframes and never reads a simulated rollout.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-xml", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-nq", type=int, default=36)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    root = ET.parse(args.task_xml).getroot()
    keyframe = root.find("keyframe")
    if keyframe is None:
        raise ValueError("task XML has no keyframe section")
    rows: list[np.ndarray] = []
    for index, key in enumerate(keyframe.findall("key")):
        values = np.fromstring(key.get("qpos", ""), sep=" ", dtype=np.float64)
        if values.shape != (args.expected_nq,) or not np.isfinite(values).all():
            raise ValueError(f"keyframe {index} has invalid qpos shape {values.shape}")
        rows.append(values)
    if len(rows) < 2:
        raise ValueError("at least two qpos keyframes are required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, np.stack(rows), delimiter=",", fmt="%.10g")
    print({"status": "exported", "frames": len(rows), "nq": args.expected_nq, "output": str(args.output)})


if __name__ == "__main__":
    main()
