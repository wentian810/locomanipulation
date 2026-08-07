"""Content-addressed cache fingerprint for a CRISP scene package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from .scene_schema import SCENE_SCHEMA_VERSION
except ImportError:  # direct `python scripts/scene/scene_fingerprint.py`
    from scene_schema import SCENE_SCHEMA_VERSION


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_scene_fingerprint(
    *,
    inputs: dict[str, Path],
    config: dict[str, Any],
    crisp_commit: str,
    adapter_code_hash: str,
    checkpoint_identifiers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serialisable fingerprint from all result-affecting inputs."""
    missing = [name for name, path in inputs.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("fingerprint input missing: " + ", ".join(missing))
    payload: dict[str, Any] = {
        "schema_version": SCENE_SCHEMA_VERSION,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sorted(inputs.items())
        },
        "config": config,
        "crisp_commit": crisp_commit,
        "adapter_code_hash": adapter_code_hash,
        "checkpoint_identifiers": dict(sorted((checkpoint_identifiers or {}).items())),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["fingerprint_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config-json", required=True, type=Path)
    parser.add_argument("--crisp-commit", required=True)
    parser.add_argument("--adapter-code-hash", required=True)
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Repeat for each input that affects reconstruction.",
    )
    args = parser.parse_args()
    inputs: dict[str, Path] = {}
    for item in args.input:
        if "=" not in item:
            raise ValueError(f"--input expects NAME=PATH, got {item!r}")
        name, raw_path = item.split("=", 1)
        if not name or name in inputs:
            raise ValueError(f"invalid or duplicate fingerprint input: {item!r}")
        inputs[name] = Path(raw_path)
    config = json.loads(args.config_json.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("--config-json must contain an object")
    result = build_scene_fingerprint(
        inputs=inputs,
        config=config,
        crisp_commit=args.crisp_commit,
        adapter_code_hash=args.adapter_code_hash,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
