#!/usr/bin/env python3
"""Collect GVHMR/PHC/GMR/2x2 videos into a S3-ready upload directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHOW_ROOT = (
    ROOT
    / "output_dir"
    / "dataset_new6_hand4wholepp_directmano_gmr_sharpa_aligned"
)
DEFAULT_GMR_ROOT = (
    ROOT
    / "GMR-master"
    / "output"
    / "show_gmr"
    / "unitree_h1_with_hand_full_auto_per_frame_foot_geom"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "output_dir"
    / "s3_upload"
    / "dataset_new6_hand4wholepp_directmano_gmr_sharpa_aligned"
)
KINDS = ("gvhmr", "phc", "gmr", "2x2")
STOPWORDS = {
    "a",
    "an",
    "and",
    "demonstration",
    "master",
    "of",
    "the",
    "while",
}


@dataclass
class Row:
    clip: str
    slug: str
    kind: str
    status: str
    source: str
    target: str
    bytes: int


def clip_slug(name: str, max_len: int = 64) -> str:
    text = name.strip().lower()
    text = re.sub(r"clip(\d+)", r"c\1", text)
    text = re.sub(r"form(\d+)", r"f\1", text)
    tokens = re.split(r"[^a-z0-9]+", text)
    tokens = [token for token in tokens if token and token not in STOPWORDS]
    slug = "_".join(tokens) or "clip"
    if len(slug) > max_len:
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
        slug = f"{slug[: max_len - 9].rstrip('_')}_{digest}"
    return slug


def latest_mp4(directory: Path) -> Path | None:
    files = [path for path in directory.glob("*.mp4") if path.is_file()]
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def find_gvhmr(show_clip: Path, clip: str) -> Path | None:
    slug = clip_slug(clip)
    gvhmr_dir = show_clip / "gvhmr_out" / clip
    return first_existing(
        [
            show_clip / "videos" / f"{slug}__gvhmr.mp4",
            gvhmr_dir / "1_incam_object.mp4",
            gvhmr_dir / "1_incam.mp4",
            gvhmr_dir / f"{clip}_3_incam_global_horiz.mp4",
            gvhmr_dir / "2_global.mp4",
        ]
    )


def find_phc(show_clip: Path) -> Path | None:
    slug = clip_slug(show_clip.name)
    alias = show_clip / "videos" / f"{slug}__phc.mp4"
    if alias.is_file():
        return alias
    phc_dir = show_clip / "phc_renderings"
    if not phc_dir.is_dir():
        return None
    return latest_mp4(phc_dir)


def find_gmr(gmr_clip: Path, slug: str) -> Path | None:
    return first_existing(
        [
            gmr_clip / "videos" / f"{slug}__gmr.mp4",
            gmr_clip / f"{slug}__gmr.mp4",
            gmr_clip / "unitree_h1_with_hand_sharpa_gvhmr.mp4",
            gmr_clip / "unitree_h1_with_hand_retarget_gvhmr.mp4",
        ]
    ) or first_existing(
        sorted(
            [
                path
                for path in gmr_clip.glob("*.mp4")
                if "composite" not in path.name and "__2x2" not in path.name
            ]
        )
    )


def find_composite(gmr_clip: Path, slug: str) -> Path | None:
    return first_existing(
        [
            gmr_clip / "videos" / f"{slug}__2x2.mp4",
            gmr_clip / f"{slug}__2x2.mp4",
            gmr_clip / "composite_2x2.mp4",
        ]
    ) or first_existing(sorted(gmr_clip.glob("*2x2*.mp4")))


def materialize(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if mode == "symlink":
        target.symlink_to(source.resolve())
        return
    if mode == "copy":
        shutil.copy2(source, target)
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def collect(args: argparse.Namespace) -> list[Row]:
    rows: list[Row] = []
    args.output_root.mkdir(parents=True, exist_ok=True)
    clip_dirs = [
        path
        for path in sorted(args.show_root.iterdir())
        if path.is_dir() and not path.name.startswith(".")
    ]
    for show_clip in clip_dirs:
        clip = show_clip.name
        slug = clip_slug(clip, max_len=args.max_slug_len)
        gmr_clip = args.gmr_root / clip
        sources = {
            "gvhmr": find_gvhmr(show_clip, clip),
            "phc": find_phc(show_clip),
            "gmr": find_gmr(gmr_clip, slug) if gmr_clip.is_dir() else None,
            "2x2": find_composite(gmr_clip, slug) if gmr_clip.is_dir() else None,
        }
        for kind in KINDS:
            source = sources[kind]
            target = args.output_root / slug / f"{slug}__{kind}.mp4"
            if source is None:
                rows.append(Row(clip, slug, kind, "missing", "", str(target), 0))
                continue
            if not args.dry_run:
                materialize(source, target, args.link_mode)
            rows.append(
                Row(
                    clip=clip,
                    slug=slug,
                    kind=kind,
                    status="ok",
                    source=str(source),
                    target=str(target),
                    bytes=source.stat().st_size,
                )
            )
    return rows


def write_manifest(rows: list[Row], output_root: Path) -> None:
    csv_path = output_root / "manifest.csv"
    md_path = output_root / "manifest.md"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(Row.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)

    by_clip: dict[tuple[str, str], list[Row]] = {}
    for row in rows:
        by_clip.setdefault((row.clip, row.slug), []).append(row)
    with md_path.open("w", encoding="utf-8") as stream:
        stream.write("# S3 video upload manifest\n\n")
        for (clip, slug), clip_rows in by_clip.items():
            stream.write(f"## {slug}\n\n")
            stream.write(f"Original clip: `{clip}`\n\n")
            stream.write("| kind | status | file |\n")
            stream.write("| --- | --- | --- |\n")
            for row in clip_rows:
                rel = Path(row.target).relative_to(output_root)
                stream.write(f"| {row.kind} | {row.status} | `{rel}` |\n")
            stream.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-root", type=Path, default=DEFAULT_SHOW_ROOT)
    parser.add_argument("--gmr-root", type=Path, default=DEFAULT_GMR_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--link-mode",
        choices=["hardlink", "copy", "symlink"],
        default="hardlink",
        help="hardlink avoids duplicating video bytes; falls back to copy across filesystems.",
    )
    parser.add_argument("--max-slug-len", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect(args)
    if not args.dry_run:
        write_manifest(rows, args.output_root)
    ok = sum(1 for row in rows if row.status == "ok")
    missing = sum(1 for row in rows if row.status != "ok")
    print(f"clips={len({row.clip for row in rows})} ok_files={ok} missing={missing}")
    print(f"output={args.output_root}")
    if missing:
        print("Missing entries:")
        for row in rows:
            if row.status != "ok":
                print(f"  {row.clip}: {row.kind}")


if __name__ == "__main__":
    main()
