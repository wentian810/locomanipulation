"""Dependency-light validation of the stable, simulation-ready scene package."""

from __future__ import annotations

import argparse
import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from .scene_schema import (
        FINAL_REQUIRED_FILES,
        ROOT_REQUIRED_FILES,
        load_json_object,
        scene_package_paths,
        validate_manifest_basics,
    )
except ImportError:  # direct execution
    from scene_schema import (
        FINAL_REQUIRED_FILES,
        ROOT_REQUIRED_FILES,
        load_json_object,
        scene_package_paths,
        validate_manifest_basics,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_checksums(scene_root: Path, required_paths: dict[str, Path]) -> list[str]:
    checksum_path = scene_root / "checksums.sha256"
    if not checksum_path.is_file():
        return [f"missing: {checksum_path}"]
    entries: dict[str, str] = {}
    errors: list[str] = []
    for line_number, raw in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw:
            continue
        parts = raw.split(maxsplit=1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            errors.append(f"invalid checksum record at line {line_number}")
            continue
        relative = Path(parts[1])
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            errors.append(f"unsafe checksum path at line {line_number}")
            continue
        name = relative.as_posix()
        if name in entries:
            errors.append(f"duplicate checksum path: {name}")
            continue
        entries[name] = parts[0]
    for path in required_paths.values():
        if path == checksum_path:
            continue
        relative = path.relative_to(scene_root).as_posix()
        if relative not in entries:
            errors.append(f"missing checksum: {relative}")
            continue
        if path.is_file() and _sha256(path) != entries[relative]:
            errors.append(f"checksum mismatch: {relative}")
    return errors


def validate_scene_package(
    scene_root: Path,
    *,
    require_input_provenance: bool = False,
) -> list[str]:
    """Return all cheap structural failures; an empty list means pass.

    CRISP/MuJoCo/Isaac-specific numeric and simulator smoke tests are added in
    the production sidecar phase.  This early validator is intentionally safe
    to execute from the project environment before CRISP installation.
    """
    try:
        manifest = load_json_object(scene_root / "scene_manifest.json")
    except (OSError, ValueError) as exc:
        return [f"invalid scene_manifest.json: {exc}"]
    try:
        paths = scene_package_paths(scene_root, str(manifest.get("backend", "")))
    except ValueError as exc:
        return [str(exc)]
    errors = [f"missing: {path}" for path in paths.values() if not path.is_file()]
    if errors:
        return errors
    errors.extend(
        validate_manifest_basics(
            manifest,
            require_input_provenance=require_input_provenance,
        )
    )
    try:
        quality = load_json_object(scene_root / "scene_quality.json")
    except (OSError, ValueError) as exc:
        errors.append(f"invalid scene_quality.json: {exc}")
    else:
        if quality.get("schema_version") != manifest.get("schema_version"):
            errors.append("scene_quality.schema_version does not match manifest")
        if quality.get("verdict") not in {"pass", "warn"}:
            errors.append("scene_quality.verdict must be pass or warn")
    for name in paths:
        if not paths[name].stat().st_size:
            errors.append(f"empty: {paths[name]}")
    if require_input_provenance:
        errors.extend(_validate_checksums(scene_root, paths))
    mujoco_path = paths.get("mujoco")
    if mujoco_path is not None:
        try:
            mujoco_root = ET.parse(mujoco_path).getroot()
        except ET.ParseError as exc:
            errors.append(f"invalid MuJoCo XML: {exc}")
        else:
            mesh_names = set()
            for mesh in mujoco_root.findall("./asset/mesh"):
                name = mesh.get("name", "")
                if name:
                    mesh_names.add(name)
                mesh_file = mesh.get("file", "")
                if not mesh_file:
                    errors.append(f"MuJoCo mesh without file: {name or '<unnamed>'}")
                    continue
                mesh_path = Path(mesh_file)
                if mesh_path.is_absolute() or ".." in mesh_path.parts:
                    errors.append(f"MuJoCo mesh path is not portable: {mesh_file}")
                elif not (mujoco_path.parent / mesh_path).is_file():
                    errors.append(f"missing MuJoCo mesh: {mujoco_path.parent / mesh_path}")
            for geom in mujoco_root.findall(".//geom[@type='mesh']"):
                mesh_name = geom.get("mesh", "")
                if mesh_name not in mesh_names:
                    errors.append(f"MuJoCo geom references undeclared mesh: {mesh_name}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, default=None)
    parser.add_argument("--mesh", type=Path, default=None, help="compatibility check")
    parser.add_argument("--urdf", type=Path, default=None, help="compatibility check")
    parser.add_argument("--params", type=Path, default=None, help="compatibility check")
    parser.add_argument("--require-input-provenance", action="store_true")
    args = parser.parse_args()
    if args.scene_root is None:
        provided = [path for path in (args.mesh, args.urdf, args.params) if path]
        if len(provided) != 3:
            parser.error("provide --scene-root or all of --mesh, --urdf, --params")
        errors = [f"missing: {path}" for path in provided if not path.is_file() or not path.stat().st_size]
    else:
        errors = validate_scene_package(
            args.scene_root,
            require_input_provenance=args.require_input_provenance,
        )
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        raise SystemExit(1)
    print("[PASS] scene package validation")


if __name__ == "__main__":
    main()
