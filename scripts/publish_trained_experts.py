#!/usr/bin/env python3
"""Publish four trained experts under the canonical evaluation/inference names.

The command validates every source before changing ``models/`` and creates
relative directory symlinks, so multi-GB model weights are not copied. Existing
real directories are never replaced; a conflicting symlink requires the
explicit ``--replace-links`` option.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


EXPERTS = {
    "expert_general": ("expert-root", "expert_general_full"),
    "expert_ground": ("expert-root", "expert_ground_full"),
    "expert_change": ("five-stage-root", "a2b_change"),
    "expert_caption": ("five-stage-root", "caption"),
}

WEIGHT_SUFFIXES = (
    ".safetensors",
    ".safetensors.index.json",
    ".bin",
    ".bin.index.json",
    ".pt",
    ".pth",
    ".ckpt",
    ".gguf",
)
RUNTIME_FILES = frozenset(
    {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "processor_config.json",
        "video_preprocessor_config.json",
        "image_processor_config.json",
        "audio_processor_config.json",
        "composition_manifest.json",
    }
)
BASE_PROFILE = {
    "key": "qwen35_4b",
    "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "license": {
        "id": "Apache-2.0",
        "url": "https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/LICENSE",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pinned_file(path: Path) -> dict[str, int | str]:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def build_evaluation_profile(name: str, source: Path) -> dict[str, object]:
    """Build the evaluator's fail-closed profile for one locally trained model."""
    composition = source / "composition_manifest.json"
    if not composition.is_file():
        raise FileNotFoundError(
            f"trained model has no composition_manifest.json: {source}"
        )
    try:
        composition_data = json.loads(composition.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid composition manifest: {composition}") from exc
    if "peft_lora" not in str(composition_data.get("operation") or ""):
        raise ValueError(f"composition did not merge a PEFT LoRA: {composition}")

    files = [entry for entry in source.iterdir() if entry.is_file()]
    weights = {
        entry.name: pinned_file(entry)
        for entry in sorted(files)
        if entry.name.endswith(WEIGHT_SUFFIXES)
    }
    runtime_files = {
        entry.name: pinned_file(entry)
        for entry in sorted(files)
        if entry.name in RUNTIME_FILES
    }
    if not weights or "config.json" not in runtime_files:
        raise ValueError(f"cannot profile incomplete model: {source}")
    fingerprint = hashlib.sha256(
        "".join(str(value["sha256"]) for value in weights.values()).encode()
    ).hexdigest()[:16]
    return {
        "schema_version": 1,
        "kind": "derived_candidate_profile",
        "key": f"{name}_{fingerprint}",
        "model_path": str(source.resolve()),
        "base_profile": BASE_PROFILE["key"],
        "base_revision": BASE_PROFILE["revision"],
        "license": BASE_PROFILE["license"],
        "evidence": {
            "kind": "rs_mllm_full_training_pipeline",
            "provenance": {"status": "complete", "scope": "full", "rows_failed": 0},
            "composition_manifest": {
                "path": str(composition),
                "sha256": sha256(composition),
            },
        },
        "weights": weights,
        "runtime_files": runtime_files,
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

    # Hash and validate all four profiles before modifying any source or link.
    built_profiles = {
        name: build_evaluation_profile(name, source)
        for name, source in sources.items()
    }
    profiles: dict[str, dict[str, object]] = {}
    for name, source in sources.items():
        profile = built_profiles[name]
        profile_path = source / "evaluation_profile.json"
        temporary_profile = source / f".{profile_path.name}.publish-{os.getpid()}"
        temporary_profile.write_text(
            json.dumps(profile, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary_profile, profile_path)
        profiles[name] = {
            "path": str(profile_path),
            "key": profile["key"],
        }

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

    report["evaluation_profiles"] = profiles
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
