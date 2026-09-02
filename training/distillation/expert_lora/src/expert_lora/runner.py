"""Validate frozen assets and launch one retained Expert LoRA experiment."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .data import load_records, validate_records

REQUIRED_CONFIG_KEYS = {
    "schema_version", "name", "expert", "model_alias", "data", "data_schema",
    "expected_records", "world_size", "effective_batch_size", "epochs",
    "learning_rate", "rank", "alpha", "dropout", "workers", "seed", "max_pixels",
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


def resolve_init_adapter(config: dict, override: Path | None, output_root: Path) -> Path | None:
    if override:
        candidate = override.expanduser().resolve()
        if not (candidate / "adapter_config.json").is_file():
            raise FileNotFoundError(f"invalid --init-lora adapter: {candidate}")
        return candidate
    name = config.get("init_adapter")
    if not name:
        return None
    candidate = output_root / name
    if (candidate / "adapter_config.json").is_file():
        return candidate
    raise FileNotFoundError(
        f"missing initial adapter {name!r} under {output_root}; pass --init-lora explicitly"
    )


def gpu_ids(value: str | None, retained_world_size: int) -> list[str]:
    if value is None:
        return [str(index) for index in range(retained_world_size)]
    if "," in value:
        result = [item.strip() for item in value.split(",") if item.strip()]
    else:
        count = int(value)
        if count < 1:
            raise ValueError("--gpus must be a positive GPU count")
        result = [str(index) for index in range(count)]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--models-root", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--gpus", help="GPU count, e.g. 1 or 4; use --gpu-ids for explicit IDs")
    parser.add_argument("--gpu-ids", help="comma-separated visible GPU IDs")
    parser.add_argument("--init-lora", type=Path)
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.max_updates < 0:
        raise ValueError("max_updates must be non-negative")
    if args.gpus and args.gpu_ids:
        raise ValueError("use only one of --gpus and --gpu-ids")

    config = load_config(args.config)
    dataset_root = args.dataset_root.expanduser().resolve()
    models_root = args.models_root.expanduser().resolve()
    run_id = os.environ.get("RS_MLLM_RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S")
    default_output = models_root / "training35" / "runs" / run_id
    output_root = (args.output_root or default_output).expanduser().resolve()
    model = models_root / config["model_alias"]
    data = dataset_root / config["data"]
    adapter_output = output_root / config["name"]

    if not (model / "config.json").is_file():
        raise FileNotFoundError(f"missing training start model: {model}")
    if not data.is_file():
        raise FileNotFoundError(f"missing training data: {data}")

    records = load_records(data)
    report = validate_records(records, schema=config["data_schema"], image_root=dataset_root, check_images=True, allow_absolute=False)
    if len(records) != int(config["expected_records"]):
        report["passed"] = False
        report["errors"].append(f"expected {config['expected_records']} records, found {len(records)}")
    if not report["passed"]:
        raise ValueError(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"dataset_validation": report}, ensure_ascii=False, indent=2))
    if args.prepare_only:
        return

    retained_world_size = int(config["world_size"])
    ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()] if args.gpu_ids else gpu_ids(args.gpus, retained_world_size)
    world_size = len(ids)
    effective_batch = int(config["effective_batch_size"])
    if effective_batch % world_size:
        raise ValueError(f"effective_batch_size {effective_batch} is not divisible by {world_size} GPUs")
    grad_accum = effective_batch // world_size
    init_adapter = resolve_init_adapter(config, args.init_lora, output_root)
    if adapter_output.exists() and any(adapter_output.iterdir()):
        raise FileExistsError(f"refusing to overwrite adapter: {adapter_output}")

    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        f"--nproc_per_node={world_size}", "-m", "expert_lora.train",
        "--model", str(model), "--data", str(data), "--image-root", str(dataset_root),
        "--output", str(adapter_output), "--expected-records", str(config["expected_records"]),
        "--epochs", str(config["epochs"]), "--grad-accum", str(grad_accum),
        "--lr", str(config["learning_rate"]), "--rank", str(config["rank"]),
        "--alpha", str(config["alpha"]), "--dropout", str(config["dropout"]),
        "--workers", str(config["workers"]), "--seed", str(config["seed"]),
        "--max-pixels", str(config["max_pixels"]),
    ]
    if init_adapter:
        command.extend(["--init-lora", str(init_adapter)])
    if args.max_updates:
        command.extend(["--max-updates", str(args.max_updates)])

    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": ",".join(ids), "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "4")})
    print(json.dumps({"experiment": config["name"], "retained_world_size": retained_world_size, "actual_world_size": world_size, "effective_batch_size": effective_batch, "gradient_accumulation": grad_accum, "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"], "output": str(adapter_output), "command": command}, indent=2))
    if not args.dry_run:
        adapter_output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
