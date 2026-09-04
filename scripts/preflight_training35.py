#!/usr/bin/env python3
"""CPU-only preflight for every README 3.6 training entry point."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TRAINING_MODELS = {
    "Qwen3.5-4B": "config.json",
    "expert_general": "config.json",
    "expert_ground": "config.json",
}
OUTPUT_MODELS = {
    "expert_general_lora": "adapter_config.json",
    "expert_ground_lora": "adapter_config.json",
    "expert_general_full": "config.json",
    "expert_ground_full": "config.json",
}
FIVE_STAGE = {
    "stage1_clean.json": 165395,
    "ga2_general.json": 28830,
    "a1_grounding.json": 31871,
    "a2b_change.json": 37180,
    "caption.jsonl": 10064,
}
EXPERT_CONFIGS = (
    "expert_general_lora",
    "expert_ground_lora_bootstrap",
    "expert_ground_lora",
)


def load(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"expected JSON array: {path}")
    return value


def check_images(rows: list[dict], root: Path) -> tuple[int, int]:
    paths: set[Path] = set()
    for row in rows:
        images = row.get("image", [])
        if isinstance(images, str):
            images = [images]
        for value in images:
            path = Path(value)
            paths.add(path if path.is_absolute() else root / path)
    return len(paths), sum(not path.is_file() for path in paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    parser.add_argument(
        "--training-data",
        "--expert-data",
        dest="training_data",
        type=Path,
        help="canonical adapted-data directory (default: datasets/training35)",
    )
    parser.add_argument(
        "--require-outputs",
        action="store_true",
        help="also require the LoRA/full models produced after training",
    )
    parser.add_argument(
        "--include-expert",
        action="store_true",
        help="also validate optional General/Grounding continuation inputs",
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    training_data = (
        args.training_data or repo / "datasets/training35"
    ).expanduser().resolve()
    failures: list[str] = []
    report: dict[str, object] = {
        "models": {},
        "configs": {},
        "five_stage": {},
        "expert_data": {},
    }

    required_models = {"Qwen3.5-4B": "config.json"}
    if args.include_expert:
        required_models.update(TRAINING_MODELS)
    if args.require_outputs:
        required_models.update(OUTPUT_MODELS)
    for name, marker in required_models.items():
        path = repo / "models" / name
        passed = (path / marker).is_file()
        report["models"][name] = {
            "path": str(path.resolve()) if path.exists() else str(path),
            "passed": passed,
        }
        if not passed:
            failures.append(f"missing model {name}/{marker}")

    package = repo / "training/distillation/expert_lora"
    sys.path.insert(0, str(package / "src"))
    from expert_lora.data import load_records, validate_records
    from expert_lora.runner import load_config

    for path in sorted((package / "configs").glob("*.json")):
        try:
            config = load_config(path)
            report["configs"][path.stem] = {
                "records": config["expected_records"],
                "passed": True,
            }
        except Exception as exc:
            failures.append(f"{path}: {exc}")
            report["configs"][path.stem] = {"passed": False}

    for name, expected in FIVE_STAGE.items():
        path = training_data / name
        try:
            rows = load(path)
            unique, missing = check_images(rows, training_data)
            passed = len(rows) == expected and missing == 0
            report["five_stage"][name] = {
                "records": len(rows),
                "unique_images": unique,
                "missing_images": missing,
                "passed": passed,
            }
            if not passed:
                failures.append(
                    f"{name}: records={len(rows)} missing_images={missing}"
                )
        except Exception as exc:
            failures.append(f"{name}: {exc}")

    for config_name in EXPERT_CONFIGS if args.include_expert else ():
        try:
            config = load_config(package / "configs" / f"{config_name}.json")
            path = training_data / config["data"]
            rows = load_records(path)
            result = validate_records(
                rows,
                schema=config["data_schema"],
                image_root=training_data,
                check_images=True,
                allow_absolute=False,
            )
            result["expected_records"] = config["expected_records"]
            result["passed"] = (
                result["passed"] and len(rows) == config["expected_records"]
            )
            all_errors = result.get("errors", [])
            result["error_count"] = len(all_errors)
            result["errors"] = all_errors[:20]
            report["expert_data"][config_name] = result
            if not result["passed"]:
                failures.append(f"{config_name}: {all_errors[:3]}")
        except Exception as exc:
            failures.append(f"{config_name}: {exc}")
            report["expert_data"][config_name] = {
                "passed": False,
                "error": str(exc),
            }

    report["passed"] = not failures
    report["failures"] = failures
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
