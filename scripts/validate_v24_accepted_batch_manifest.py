#!/usr/bin/env python3
"""Re-check artifact integrity and acceptance labels in a V24 batch manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite a manifest-validation report")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "accepted_v24_static_chair_batch":
        raise ValueError("manifest is not an accepted V24 static-chair batch")
    results = []
    for case in manifest.get("cases", []):
        artifacts = case.get("artifacts", {})
        artifact_results = {}
        for name, descriptor in artifacts.items():
            path = Path(descriptor["path"])
            actual = _sha256(path) if path.is_file() else None
            artifact_results[name] = {
                "path": str(path),
                "exists": path.is_file(),
                "sha256_matches": actual == descriptor["sha256"],
            }
        audit_descriptor = artifacts.get("audit")
        audit_pass = False
        if audit_descriptor is not None and Path(audit_descriptor["path"]).is_file():
            audit = json.loads(Path(audit_descriptor["path"]).read_text(encoding="utf-8"))
            audit_pass = bool(audit.get("acceptance", {}).get("physical_smoke_pass"))
        passed = bool(audit_pass and all(
            item["exists"] and item["sha256_matches"] for item in artifact_results.values()
        ))
        results.append({"clip_id": case.get("clip_id"), "artifacts": artifact_results, "pass": passed})
    report = {
        "schema_version": 1,
        "manifest": str(args.manifest.resolve()),
        "cases": results,
        "pass": bool(results) and all(case["pass"] for case in results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
