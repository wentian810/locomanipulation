#!/usr/bin/env python
"""Compare observed hand quality and Sharpa chain-fit metrics for an A/B run.

The sidecar reprojection error is the WiLoR/Hand4Whole++ observation residual
recorded before post-processing.  It is useful for stratifying an ablation, but
it is not a recomputed error of a temporal-fused pose.  This report makes that
limitation explicit instead of falsely reporting a post-filter improvement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def distribution(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values) & (values < 1e5)]
    if values.size == 0:
        return {"count": 0}
    q = np.percentile(values, [0, 25, 50, 75, 95, 100])
    return {
        "count": int(values.size), "mean": float(values.mean()), "min": float(q[0]),
        "p25": float(q[1]), "p50": float(q[2]), "p75": float(q[3]),
        "p95": float(q[4]), "max": float(q[5]),
    }


def sidecar_metrics(path):
    if not path:
        return None
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        result = {"path": str(path), "sides": {}}
        for side in ("left", "right"):
            error_key = f"{side}_hand_reproj_error"
            quality_key = f"{side}_hand_quality"
            alpha_key = f"{side}_hand_hysup_alpha"
            result["sides"][side] = {
                "observed_reproj_error_px": distribution(data[error_key]) if error_key in data else None,
                "source_quality": distribution(data[quality_key]) if quality_key in data else None,
                "hysup_alpha": distribution(data[alpha_key]) if alpha_key in data else None,
            }
    return result


def sharpa_metrics(path):
    if not path:
        return None
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        result = {"path": str(path), "sides": {}}
        for side in ("left", "right"):
            result["sides"][side] = {
                key: float(np.asarray(data[key]).reshape(-1)[0])
                for key in (f"{side}_chain_error_mean", f"{side}_chain_error_p95")
                if key in data
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before_hand_npz", default="")
    parser.add_argument("--after_hand_npz", default="")
    parser.add_argument("--before_sharpa_npz", default="")
    parser.add_argument("--after_sharpa_npz", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = {
        "reprojection_caveat": (
            "hand_reproj_error is an upstream detector/model residual. It must be recomputed from a fused pose "
            "and camera model before it can support a claim of post-fusion reprojection improvement."
        ),
        "before": {
            "hand": sidecar_metrics(args.before_hand_npz),
            "sharpa": sharpa_metrics(args.before_sharpa_npz),
        },
        "after": {
            "hand": sidecar_metrics(args.after_hand_npz),
            "sharpa": sharpa_metrics(args.after_sharpa_npz),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
