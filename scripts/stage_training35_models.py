#!/usr/bin/env python3
"""Create the README 3.5 models/ layout using validated symbolic links."""
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
            raise FileExistsError(f"conflicting link: {link} -> {link.resolve()} (wanted {target})")
        return
    if link.exists():
        raise FileExistsError(f"refusing to replace existing path: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, default=repo / "models")
    parser.add_argument("--cache-root", type=Path, default=repo / ".models" / "training35")
    parser.add_argument("--base", type=Path, default=Path("/users/u2024311136/shared/shared_models/lora_expert/base_model"))
    parser.add_argument("--general-delta", type=Path, default=Path("/users/u2024311136/shared/shared_models/lora_expert/lora/general/delta_model.pt"))
    parser.add_argument("--ground-delta", type=Path, default=Path("/users/u2024311136/shared/shared_models/lora_expert/lora/grounding/delta_model.pt"))
    parser.add_argument("--change", type=Path, default=Path("/users/u2024311136/shared/shared_models/lora_expert/_merge/change_merged"))
    parser.add_argument("--caption", type=Path, default=Path("/users/u2024311136/shared/shared_models/lora_expert/_merge/caption_merged"))
    parser.add_argument("--general-lora", type=Path, default=Path("/users/u2024311164/shared/exp7_old_mme_plus_old_xlrs_plus_vrsbench_vqa_weighted_lora"))
    parser.add_argument("--ground-lora", type=Path, default=Path("/users/u2024311164/shared/ground_expert_update_lora"))
    parser.add_argument("--general-full", type=Path, default=Path("/users/u2024311136/shared/shared_models/lora_expert/_merge/general_exp7_merged"))
    parser.add_argument("--ground-full", type=Path, default=Path("/users/u2024311136/shared/shared_models/lora_expert/_merge/ground_expert_update_merged"))
    parser.add_argument("--build-general", action="store_true", help="materialize base+general delta when absent")
    parser.add_argument("--build-ground", action="store_true", help="materialize base+ground delta when absent")
    parser.add_argument("--build-experts", action="store_true", help="materialize both plain experts when absent")
    args = parser.parse_args()

    def materialize(name: str, delta: Path, enabled: bool) -> Path:
        output = args.cache_root / name
        manifest_path = output / "composition_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = {
                "base_config_sha256": sha256(args.base / "config.json"),
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
        if not enabled:
            raise FileNotFoundError(f"plain expert is absent at {output}; rerun with --build-experts")
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"refusing to overwrite incomplete expert cache: {output}")
        command = [sys.executable, "-m", "expert_lora.compose", "--base", str(args.base), "--delta", str(delta), "--output", str(output)]
        env = os.environ.copy()
        package = repo / "training" / "distillation" / "expert_lora" / "src"
        env["PYTHONPATH"] = str(package) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        subprocess.run(command, check=True, env=env)
        return output

    general = materialize("expert_general", args.general_delta, args.build_general or args.build_experts)
    ground = materialize("expert_ground", args.ground_delta, args.build_ground or args.build_experts)

    targets = {
        "Qwen3.5-4B": (args.base, False), "expert_general": (general, False),
        "expert_ground": (ground, False), "expert_change": (args.change, False),
        "expert_caption": (args.caption, False), "expert_general_lora": (args.general_lora, True),
        "expert_ground_lora": (args.ground_lora, True), "expert_general_full": (args.general_full, False),
        "expert_ground_full": (args.ground_full, False),
    }
    manifest = {name: validate(path.expanduser().resolve(), adapter) for name, (path, adapter) in targets.items()}
    for name, (target, _) in targets.items():
        ensure_link(args.models_root / name, target)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    (args.cache_root / "staging_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"models_root": str(args.models_root.resolve()), "models": manifest}, indent=2))


if __name__ == "__main__":
    main()
