"""Write a small, atomic, non-product status record for an isolated scene run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .scene_schema import SCENE_SCHEMA_VERSION
except ImportError:  # direct execution
    from scene_schema import SCENE_SCHEMA_VERSION


def write_scene_status(
    path: Path,
    *,
    status: str,
    clip: str,
    backend: str = "crisp",
    reason: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Atomically update sidecar status without touching human/GMR artefacts."""
    payload: dict[str, Any] = {
        "schema_version": SCENE_SCHEMA_VERSION,
        "status": status,
        "clip": clip,
        "backend": backend,
    }
    if reason:
        payload["reason"] = reason
    if extra:
        payload["extra"] = extra
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--status", required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--backend", default="crisp")
    parser.add_argument("--reason", default=None)
    args = parser.parse_args()
    write_scene_status(
        args.output,
        status=args.status,
        clip=args.clip,
        backend=args.backend,
        reason=args.reason,
    )


if __name__ == "__main__":
    main()
