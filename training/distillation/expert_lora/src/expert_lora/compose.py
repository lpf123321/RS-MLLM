"""Materialize a base with an optional full-rank delta and/or PEFT LoRA."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from .data import sha256


def _nonempty(path: Path) -> bool:
    return path.exists() and (not path.is_dir() or any(path.iterdir()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--delta", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--processor-source", type=Path)
    parser.add_argument("--max-shard-size", default="5GB")
    args = parser.parse_args()

    paths = [("base", args.base)]
    if args.delta:
        paths.append(("delta", args.delta))
    for label, path in paths:
        if not path.exists():
            raise FileNotFoundError(f"missing {label}: {path}")
    if args.adapter and not args.adapter.exists():
        raise FileNotFoundError(f"missing adapter: {args.adapter}")
    if _nonempty(args.output):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output}")

    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.base,
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tensor_count = 0
    if args.delta:
        delta = torch.load(args.delta, map_location="cpu", weights_only=True)
        if not isinstance(delta, dict) or not delta:
            raise TypeError("full-rank delta must be a non-empty state dict")

        state = model.state_dict()
        unknown = sorted(set(delta) - set(state))
        if unknown:
            raise RuntimeError(
                f"delta contains {len(unknown)} unknown keys; first: {unknown[:10]}"
            )
        mismatches = [
            (name, tuple(value.shape), tuple(state[name].shape))
            for name, value in delta.items()
            if value.shape != state[name].shape
        ]
        if mismatches:
            raise RuntimeError(
                f"delta contains {len(mismatches)} shape mismatches; first: {mismatches[:5]}"
            )

        with torch.no_grad():
            for name, value in delta.items():
                target = state[name]
                target.add_(value.to(dtype=target.dtype, device=target.device))
        tensor_count = len(delta)
        del delta, state
        gc.collect()

    adapter_hash = None
    if args.adapter:
        adapter_weights = args.adapter / "adapter_model.safetensors"
        if adapter_weights.is_file():
            adapter_hash = sha256(adapter_weights)
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
        model = model.merge_and_unload(safe_merge=True)

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        args.output,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    processor_source = args.processor_source or args.base
    processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
    processor.save_pretrained(args.output)

    manifest = {
        "operation": "base" + (" + full_rank_delta" if args.delta else "") + (" + peft_lora" if args.adapter else ""),
        "base_config_sha256": sha256(args.base / "config.json"),
        "delta_file": args.delta.name if args.delta else None,
        "delta_sha256": sha256(args.delta) if args.delta else None,
        "delta_tensors": tensor_count,
        "adapter_weights_sha256": adapter_hash,
        "dtype": "bfloat16",
    }
    (args.output / "composition_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
