#!/usr/bin/env python3
"""Convert a merged full model into a Delta relative to the raw base."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--merged_model", required=True)
    parser.add_argument("--reference_delta", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reference = torch.load(args.reference_delta, map_location="cpu",
                           weights_only=True, mmap=True)
    expected_keys = set(reference)

    base_index = json.loads(Path(args.base_model, "model.safetensors.index.json").read_text())
    base_shards = {}
    for key, filename in base_index["weight_map"].items():
        if key in expected_keys:
            base_shards.setdefault(filename, []).append(key)

    merged_files = sorted(Path(args.merged_model).glob("*.safetensors"))
    if not merged_files:
        raise FileNotFoundError(f"No safetensors found in {args.merged_model}")

    output_delta = {}
    seen = set()
    with safe_open(str(merged_files[0]), framework="pt", device="cpu") as merged:
        base_handles = {}
        try:
            for filename in base_shards:
                base_handles[filename] = safe_open(
                    str(Path(args.base_model, filename)),
                    framework="pt", device="cpu",
                )
            for key in sorted(expected_keys):
                if key not in merged.keys():
                    raise RuntimeError(f"Merged model missing key: {key}")
                filename = base_index["weight_map"][key]
                base = base_handles[filename].get_tensor(key)
                merged_value = merged.get_tensor(key)
                if tuple(base.shape) != tuple(merged_value.shape):
                    raise RuntimeError(
                        f"Shape mismatch for {key}: base={tuple(base.shape)} "
                        f"merged={tuple(merged_value.shape)}"
                    )
                output_delta[key] = (merged_value - base).to(torch.bfloat16)
                seen.add(key)
        finally:
            for handle in base_handles.values():
                handle.__exit__(None, None, None)

    missing = expected_keys - seen
    if missing:
        raise RuntimeError(f"Missing {len(missing)} Delta keys: {sorted(missing)[:5]}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_delta, output)
    output.with_suffix(output.suffix + ".json").write_text(json.dumps({
        "base_model": args.base_model,
        "merged_model": args.merged_model,
        "reference_delta": args.reference_delta,
        "keys": len(output_delta),
        "dtype": "bfloat16",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(output_delta)} tensors -> {output}")


if __name__ == "__main__":
    main()
