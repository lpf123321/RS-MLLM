#!/usr/bin/env python3
"""Publish four trained experts under the canonical evaluation/inference names.

The command validates every source before changing ``models/`` and creates
relative directory symlinks, so multi-GB model weights are not copied. Existing
real directories are never replaced; a conflicting symlink requires the
explicit ``--replace-links`` option.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


EXPERTS = {
    "expert_general": ("expert-root", "expert_general_full"),
    "expert_ground": ("expert-root", "expert_ground_full"),
    "expert_change": ("five-stage-root", "a2b_change"),
    "expert_caption": ("five-stage-root", "caption"),
}


def validate_model(path: Path) -> dict[str, object]:
    """Require a reloadable Transformers model and all indexed weight shards."""
    if not path.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {path}")
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"model has no config.json: {path}")

    weight_files: list[Path]
    index = path / "model.safetensors.index.json"
    if index.is_file():
        try:
            weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            names = sorted(set(weight_map.values()))
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid weight index: {index}") from exc
        weight_files = [path / name for name in names]
        missing = [item.name for item in weight_files if not item.is_file()]
        if missing:
            raise FileNotFoundError(
                f"model is missing indexed weight shard(s): {path}: {missing[:5]}"
            )
    elif (path / "model.safetensors").is_file():
        weight_files = [path / "model.safetensors"]
    else:
        weight_files = sorted(path.glob("*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"model has no safetensors weights: {path}")

    return {
        "source": str(path),
        "weight_files": [item.name for item in weight_files],
        "weight_bytes": sum(item.stat().st_size for item in weight_files),
    }


def relative_target(source: Path, destination: Path) -> str:
    return os.path.relpath(source, start=destination.parent.resolve())


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--five-stage-root",
        type=Path,
        default=repo / "models/training35/runs/five-stage/merged",
        help="directory containing a2b_change/ and caption/",
    )
    parser.add_argument(
        "--expert-root",
        type=Path,
        default=repo / "models/training35/runs/training35-full",
        help="directory containing expert_general_full/ and expert_ground_full/",
    )
    parser.add_argument("--models-root", type=Path, default=repo / "models")
    parser.add_argument("--general", type=Path, help="override General source")
    parser.add_argument("--ground", type=Path, help="override Grounding source")
    parser.add_argument("--change", type=Path, help="override Change source")
    parser.add_argument("--caption", type=Path, help="override Caption source")
    parser.add_argument(
        "--replace-links",
        action="store_true",
        help="replace conflicting expert_* symlinks; real directories are never replaced",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    five_stage_root = args.five_stage_root.expanduser().resolve()
    expert_root = args.expert_root.expanduser().resolve()
    models_root = args.models_root.expanduser().resolve()
    overrides = {
        "expert_general": args.general,
        "expert_ground": args.ground,
        "expert_change": args.change,
        "expert_caption": args.caption,
    }

    sources: dict[str, Path] = {}
    validation: dict[str, dict[str, object]] = {}
    for name, (root_name, child) in EXPERTS.items():
        override = overrides[name]
        root = expert_root if root_name == "expert-root" else five_stage_root
        source = (override if override is not None else root / child).expanduser().resolve()
        sources[name] = source
        validation[name] = validate_model(source)

    # Resolve every conflict before creating the first link.
    actions: dict[str, str] = {}
    for name, source in sources.items():
        destination = models_root / name
        if destination.is_symlink():
            if destination.resolve(strict=False) == source:
                actions[name] = "reuse"
            elif args.replace_links:
                actions[name] = "replace-link"
            else:
                raise FileExistsError(
                    f"conflicting symlink: {destination} -> "
                    f"{destination.resolve(strict=False)}; use --replace-links"
                )
        elif destination.exists():
            raise FileExistsError(
                f"refusing to replace real directory/file: {destination}"
            )
        else:
            actions[name] = "link"

    report = {
        "models_root": str(models_root),
        "experts": {
            name: {
                **validation[name],
                "destination": str(models_root / name),
                "action": actions[name],
            }
            for name in EXPERTS
        },
    }
    if args.dry_run:
        print(json.dumps({**report, "dry_run": True}, indent=2))
        return

    models_root.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        destination = models_root / name
        if actions[name] == "reuse":
            continue
        temporary = models_root / f".{name}.publish-{os.getpid()}"
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
        temporary.symlink_to(relative_target(source, destination), target_is_directory=True)
        os.replace(temporary, destination)

    manifest = models_root / "TRAINED_EXPERTS_MANIFEST.json"
    temporary_manifest = models_root / f".{manifest.name}.publish-{os.getpid()}"
    temporary_manifest.write_text(
        json.dumps({**report, "dry_run": False}, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest)
    print(json.dumps({**report, "manifest": str(manifest)}, indent=2))


if __name__ == "__main__":
    main()
