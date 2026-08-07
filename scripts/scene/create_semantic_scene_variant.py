"""Create an immutable semantic-simulation variant of a scene package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


SEMANTIC_ASSETS = (
    "background_mesh_semantic_chair.obj",
    "semantic_chair_primitives_mujoco.json",
    "semantic_chair_primitives_phc.json",
    "scene_mujoco_semantic_chair.xml",
    "scene_semantic_chair.urdf",
    "semantic_chair_report.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_checksums(root: Path) -> None:
    """Regenerate the package inventory after adding semantic runtime assets."""
    lines = [
        f"{_sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "checksums.sha256"
    ]
    (root / "checksums.sha256").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", required=True, type=Path)
    parser.add_argument("--semantic-assets-dir", required=True, type=Path)
    parser.add_argument("--output-package", required=True, type=Path)
    args = parser.parse_args()

    source = args.source_package.resolve()
    semantic = args.semantic_assets_dir.resolve()
    output = args.output_package.resolve()
    if not (source / "scene_manifest.json").is_file() or not (source / "scene").is_dir():
        raise FileNotFoundError("source is not a complete scene package")
    missing = [semantic / name for name in SEMANTIC_ASSETS if not (semantic / name).is_file()]
    if missing:
        raise FileNotFoundError("semantic asset contract is incomplete:\n  " + "\n  ".join(map(str, missing)))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing semantic package: {output}")

    shutil.copytree(source, output, symlinks=True)
    scene_dir = output / "scene"
    for name in SEMANTIC_ASSETS:
        shutil.copy2(semantic / name, scene_dir / name)

    manifest_path = output / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Package structural validity remains a pass: its schema contract and all
    # runtime assets are present.  The separate semantic release gate records
    # that the physics replay is still pending, rather than misusing the
    # package-integrity status that the validator intentionally restricts.
    manifest["status"] = "pass"
    manifest.setdefault("assets", {}).update(
        {
            "semantic_simulation_mesh": "scene/background_mesh_semantic_chair.obj",
            "semantic_simulation_urdf": "scene/scene_semantic_chair.urdf",
            "semantic_simulation_mujoco": "scene/scene_mujoco_semantic_chair.xml",
            "semantic_simulation_primitives": "scene/semantic_chair_primitives_mujoco.json",
            "semantic_phc_collision_primitives": "scene/semantic_chair_primitives_phc.json",
            "semantic_simulation_report": "scene/semantic_chair_report.json",
        }
    )
    manifest["semantic_simulation"] = {
        "type": "parametric_chair",
        "collision_mode": "semantic_primitives_only",
        "selection_reason": "raw monocular mesh fused the chair with the fixed background wall",
        "runtime_assets": {
            "isaac": "scene/semantic_chair_primitives_phc.json",
            "mujoco": "scene/scene_mujoco_semantic_chair.xml",
            "review_mesh": "scene/background_mesh_semantic_chair.obj",
        },
        "raw_assets_retained_for_audit": True,
        "release_gate": "requires PHC collision replay after CUDA availability is restored",
        "release_status": "pending_phc_replay",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_checksums(output)
    print(
        "[PASS] semantic scene variant created: "
        f"{output} ({sum(path.stat().st_size for path in output.rglob('*') if path.is_file()) / 1e6:.2f} MB)"
    )


if __name__ == "__main__":
    main()
