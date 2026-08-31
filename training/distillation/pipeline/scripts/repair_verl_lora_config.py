#!/usr/bin/env python3
"""Repair VERL LoRA configs that serialize a regex as single characters."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from safetensors import safe_open


LORA_MODULE = re.compile(r"\.([^.]+)\.lora_[AB](?:\.default)?\.weight$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter_dir", type=Path)
    args = parser.parse_args()

    config_path = args.adapter_dir / "adapter_config.json"
    weights_path = args.adapter_dir / "adapter_model.safetensors"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    with safe_open(weights_path, framework="pt", device="cpu") as stream:
        keys = list(stream.keys())

    modules = sorted(
        {
            match.group(1)
            for key in keys
            if (match := LORA_MODULE.search(key)) is not None
        }
    )
    if not modules:
        raise AssertionError(f"No LoRA module names found in {weights_path}")
    if not any(key.endswith("lora_A.weight") for key in keys):
        raise AssertionError("Adapter has no LoRA-A tensors")
    if not any(key.endswith("lora_B.weight") for key in keys):
        raise AssertionError("Adapter has no LoRA-B tensors")

    before = config.get("target_modules")
    config["target_modules"] = modules
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )
    report = {
        "status": "PASS",
        "adapter_dir": str(args.adapter_dir),
        "weight_tensors": len(keys),
        "target_modules_before": before,
        "target_modules_after": modules,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
