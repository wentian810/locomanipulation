#!/usr/bin/env python3
"""Combine a reference RGB frame and binary object mask into an RGBA PNG."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(args.image)
    if mask is None:
        raise FileNotFoundError(args.mask)
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(
            mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    alpha = np.where(mask > 127, 255, 0).astype(np.uint8)
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    rgba[..., 3] = alpha
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), rgba):
        raise RuntimeError(f"failed to write {args.output}")
    print(f"Saved Hunyuan RGBA input: {args.output}")


if __name__ == "__main__":
    main()
