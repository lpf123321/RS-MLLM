"""Resolve frozen assets and launch one retained Expert LoRA experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .data import load_records, validate_records


REQUIRED_CONFIG_KEYS = {
    "schema_version",
    "name",
    "expert",
    "expert_delta",
    "data",
    "data_schema",
    "expected_records",
    "world_size",
    "effective_batch_size",
    "epochs",
    "learning_rate",
    "rank",
    "alpha",
    "dropout",
    "workers",
    "seed",
    "max_pixels",
}


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"config is missing required keys: {missing}")
    if config["schema_version"] != 1:
        raise ValueError(f"unsupported config schema: {config['schema_version']}")
    if config["expert"] not in {"general", "grounding"}:
        raise ValueError("expert must be general or grounding")
    return config


def resolve_init_adapter(config: dict, asset_root: Path, output_root: Path) -> Path | None:
    name = config.get("init_adapter")
    if not name:
        return None
    candidates = [
        output_root / "adapters" / name,
        asset_root / "expert_models" / "adapters" / name,
    ]
    for candidate in candidates:
        if (candidate / "adapter_config.json").is_file():
            return candidate
    raise FileNotFoundError(
        f"missing initial adapter {name!r}; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--gpus", help="comma-separated visible GPU IDs")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    asset_root = args.asset_root.resolve()
    output_root = (args.output_root or asset_root / "outputs").resolve()
    base = asset_root / "base" / "Qwen3.5-4B"
    delta = asset_root / "expert_models" / config["expert_delta"]
    data = asset_root / "datasets" / config["data"]
    image_root = asset_root / "datasets"
    runtime_model = output_root / "runtime_models" / f"{config['expert']}_expert"
    adapter_output = output_root / "adapters" / config["name"]

    if not (base / "config.json").is_file():
        raise FileNotFoundError(f"missing Qwen3.5-4B base: {base}")
    if not delta.is_file():
        raise FileNotFoundError(f"missing {config['expert']} expert delta: {delta}")
    if not data.is_file():
        raise FileNotFoundError(f"missing training data: {data}")

    records = load_records(data)
    report = validate_records(
        records,
        schema=config["data_schema"],
        image_root=image_root,
        check_images=True,
        allow_absolute=False,
    )
    if len(records) != int(config["expected_records"]):
        report["passed"] = False
        report["errors"].append(
            f"expected {config['expected_records']} records, found {len(records)}"
        )
    if not report["passed"]:
        raise ValueError(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"dataset_validation": report}, ensure_ascii=False, indent=2))

    if not (runtime_model / "config.json").is_file():
        compose_command = [
            sys.executable,
            "-m",
            "expert_lora.compose",
            "--base",
            str(base),
            "--delta",
            str(delta),
            "--output",
            str(runtime_model),
        ]
        print(json.dumps({"compose_command": compose_command}, indent=2))
        if not args.dry_run:
            runtime_model.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(compose_command, check=True)

    if args.prepare_only:
        return

    world_size = int(config["world_size"])
    gpu_ids = [item.strip() for item in (args.gpus or ",".join(map(str, range(world_size)))).split(",") if item.strip()]
    if len(gpu_ids) != world_size:
        raise ValueError(
            f"{config['name']} retained run requires {world_size} GPUs; got {gpu_ids}"
        )
    effective_batch = int(config["effective_batch_size"])
    if effective_batch % world_size:
        raise ValueError("effective_batch_size must be divisible by world_size")
    grad_accum = effective_batch // world_size
    init_adapter = resolve_init_adapter(config, asset_root, output_root)
    if adapter_output.exists() and any(adapter_output.iterdir()):
        raise FileExistsError(f"refusing to overwrite adapter: {adapter_output}")

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={world_size}",
        "-m",
        "expert_lora.train",
        "--model",
        str(runtime_model),
        "--data",
        str(data),
        "--image-root",
        str(image_root),
        "--output",
        str(adapter_output),
        "--expected-records",
        str(config["expected_records"]),
        "--epochs",
        str(config["epochs"]),
        "--grad-accum",
        str(grad_accum),
        "--lr",
        str(config["learning_rate"]),
        "--rank",
        str(config["rank"]),
        "--alpha",
        str(config["alpha"]),
        "--dropout",
        str(config["dropout"]),
        "--workers",
        str(config["workers"]),
        "--seed",
        str(config["seed"]),
        "--max-pixels",
        str(config["max_pixels"]),
    ]
    if init_adapter:
        command.extend(["--init-lora", str(init_adapter)])

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(gpu_ids),
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "4"),
        }
    )
    print(
        json.dumps(
            {
                "experiment": config["name"],
                "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
                "command": command,
            },
            indent=2,
        )
    )
    if not args.dry_run:
        adapter_output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
