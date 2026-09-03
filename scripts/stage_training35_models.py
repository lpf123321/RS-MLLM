#!/usr/bin/env python3
"""Prepare the portable README 3.6 ``models/`` layout.

Existing local directories are linked without copying. With ``--download``,
missing entries are fetched from the ModelScope ids in ``rsmllm.config``.
No network request is made by ``--dry-run``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

MODEL_FILES = ("config.json",)
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")

MODEL_SPECS = {
    "Qwen3.5-4B": ("base", False, "base"),
    "expert_general": ("expert_general", False, "general"),
    "expert_ground": ("expert_ground", False, "ground"),
    "expert_change": ("expert_change", False, "change"),
    "expert_caption": ("expert_caption", False, "caption"),
    "expert_general_lora": ("expert_general_lora", True, "general_lora"),
    "expert_ground_lora": ("expert_ground_lora", True, "ground_lora"),
    "expert_general_full": ("expert_general_full", False, "general_full"),
    "expert_ground_full": ("expert_ground_full", False, "ground_full"),
}
PROFILES = {
    # The five-stage pipeline starts from the raw base and produces all four
    # experts itself.  Continued expert-LoRA experiments are a separate path.
    "five-stage": ("Qwen3.5-4B",),
    "expert-lora": ("expert_general", "expert_ground"),
    # Backward-compatible aggregate used by delta/reproduction workflows.
    "training": tuple(list(MODEL_SPECS)[:5]),
    "all": tuple(MODEL_SPECS),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(path: Path, adapter: bool = False) -> dict[str, object]:
    required = ADAPTER_FILES if adapter else MODEL_FILES
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{path}: missing {', '.join(missing)}")
    marker = path / required[-1]
    weights = sorted(path.glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"{path}: no safetensors weight file")
    return {
        "target": str(path.resolve()),
        "marker": marker.name,
        "sha256": sha256(marker),
        "weight_files": {
            item.name: {"bytes": item.stat().st_size, "sha256": sha256(item)}
            for item in weights
        },
    }


def ensure_link(link: Path, target: Path) -> None:
    target = target.expanduser().resolve()
    if link.is_symlink():
        if link.resolve() != target:
            raise FileExistsError(
                f"conflicting link: {link} -> {link.resolve()} (wanted {target})"
            )
        return
    if link.exists():
        if link.resolve() == target:
            return
        raise FileExistsError(f"refusing to replace existing path: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, default=repo / "models")
    parser.add_argument(
        "--cache-root", type=Path, default=repo / ".models" / "training35"
    )
    parser.add_argument("--profile", choices=PROFILES, default="five-stage")
    parser.add_argument(
        "--download",
        action="store_true",
        help="download missing entries from the registered ModelScope repositories",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the local/ModelScope resolution plan without writing or downloading",
    )
    parser.add_argument("--base", type=Path)
    parser.add_argument("--general-delta", type=Path)
    parser.add_argument("--ground-delta", type=Path)
    parser.add_argument("--change", type=Path)
    parser.add_argument("--caption", type=Path)
    parser.add_argument("--general-lora", type=Path)
    parser.add_argument("--ground-lora", type=Path)
    parser.add_argument("--general-full", type=Path)
    parser.add_argument("--ground-full", type=Path)
    parser.add_argument(
        "--build-general", action="store_true", help="compose base+general delta"
    )
    parser.add_argument(
        "--build-ground", action="store_true", help="compose base+ground delta"
    )
    parser.add_argument(
        "--build-experts", action="store_true", help="compose both base+delta experts"
    )
    args = parser.parse_args()

    # Delayed import keeps --dry-run CPU-only and independent of ModelScope.
    sys.path.insert(0, str(repo))
    from rsmllm.config import MODEL_REGISTRY

    explicit = {
        "base": args.base,
        "change": args.change,
        "caption": args.caption,
        "general_lora": args.general_lora,
        "ground_lora": args.ground_lora,
        "general_full": args.general_full,
        "ground_full": args.ground_full,
    }
    selected = PROFILES[args.profile]
    plan: dict[str, dict[str, str]] = {}

    for name in selected:
        alias, _, option_name = MODEL_SPECS[name]
        canonical = args.models_root / name
        supplied = explicit.get(option_name)
        marker = "adapter_config.json" if name.endswith("_lora") else "config.json"
        if (canonical / marker).is_file():
            plan[name] = {"action": "reuse", "source": str(canonical.resolve())}
        elif supplied is not None:
            plan[name] = {
                "action": "link",
                "source": str(supplied.expanduser().resolve()),
            }
        elif name == "expert_general" and (args.build_general or args.build_experts):
            plan[name] = {
                "action": "compose",
                "source": str((args.cache_root / name).resolve()),
            }
        elif name == "expert_ground" and (args.build_ground or args.build_experts):
            plan[name] = {
                "action": "compose",
                "source": str((args.cache_root / name).resolve()),
            }
        else:
            plan[name] = {
                "action": "download" if args.download else "missing",
                "source": MODEL_REGISTRY[alias],
            }

    if args.dry_run:
        print(json.dumps({"profile": args.profile, "models": plan}, indent=2))
        return

    missing = [name for name, item in plan.items() if item["action"] == "missing"]
    if missing:
        parser.error(
            "missing model source(s): "
            + ", ".join(missing)
            + "; pass local --base/--change/... paths or use --download"
        )

    def resolve_base() -> Path:
        canonical = args.models_root / "Qwen3.5-4B"
        if (canonical / "config.json").is_file():
            return canonical.resolve()
        if args.base is not None:
            return args.base.expanduser().resolve()
        if not args.download:
            raise FileNotFoundError("base model is required to compose an expert")
        from rsmllm.models import get_model

        return Path(get_model("base", cache_dir=str(args.cache_root / "modelscope")))

    def materialize(name: str, delta: Path | None) -> Path:
        if delta is None:
            option = "general" if name == "expert_general" else "ground"
            raise FileNotFoundError(
                f"--{option}-delta is required when composing {name}"
            )
        base = resolve_base()
        delta = delta.expanduser().resolve()
        output = args.cache_root / name
        manifest_path = output / "composition_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = {
                "base_config_sha256": sha256(base / "config.json"),
                "delta_sha256": sha256(delta),
            }
            conflicts = {
                key: {"found": manifest.get(key), "expected": value}
                for key, value in expected.items()
                if manifest.get(key) != value
            }
            if conflicts:
                raise FileExistsError(
                    f"refusing to reuse conflicting composed expert {output}: "
                    f"{json.dumps(conflicts, sort_keys=True)}"
                )
            validate(output)
            return output
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"refusing to overwrite incomplete expert cache: {output}")
        command = [
            sys.executable,
            "-m",
            "expert_lora.compose",
            "--base",
            str(base),
            "--delta",
            str(delta),
            "--output",
            str(output),
        ]
        env = os.environ.copy()
        package = repo / "training" / "distillation" / "expert_lora" / "src"
        env["PYTHONPATH"] = str(package) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        subprocess.run(command, check=True, env=env)
        return output

    targets: dict[str, tuple[Path, bool]] = {}
    for name in selected:
        alias, adapter, option_name = MODEL_SPECS[name]
        action = plan[name]["action"]
        if action == "reuse":
            target = (args.models_root / name).resolve()
        elif action == "link":
            supplied = explicit[option_name]
            assert supplied is not None
            target = supplied.expanduser().resolve()
        elif action == "compose":
            delta = args.general_delta if name == "expert_general" else args.ground_delta
            target = materialize(name, delta)
        else:
            from rsmllm.models import get_model

            target = Path(
                get_model(alias, cache_dir=str(args.cache_root / "modelscope"))
            )
        targets[name] = (target, adapter)

    manifest = {
        name: validate(path.expanduser().resolve(), adapter)
        for name, (path, adapter) in targets.items()
    }
    for name, (target, _) in targets.items():
        ensure_link(args.models_root / name, target)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    (args.cache_root / "staging_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "profile": args.profile,
                "models_root": str(args.models_root.resolve()),
                "models": manifest,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
