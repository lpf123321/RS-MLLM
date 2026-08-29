#!/usr/bin/env python3
"""Merge an Expert Delta and a PEFT LoRA on the raw base model.

General usage (also reused for the grounding expert update):

    W0 + expert_delta + peft_lora -> merged full model
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import Qwen3_5ForConditionalGeneration

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--expert_delta", required=True)
    parser.add_argument("--peft_lora", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for the conversion")

    print("Loading W0 ...", flush=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(args.device)
    model.eval()

    print("Applying Expert Delta ...", flush=True)
    delta = torch.load(args.expert_delta, map_location="cpu", weights_only=True)
    state = model.state_dict()
    missing = []
    with torch.no_grad():
        for name, value in delta.items():
            if name not in state:
                missing.append(name)
                continue
            state[name].add_(value.to(device=state[name].device,
                                      dtype=state[name].dtype))
    if missing:
        raise RuntimeError(f"Expert Delta has {len(missing)} unknown keys: {missing[:5]}")
    print(f"Applied {len(delta)} Expert Delta tensors", flush=True)

    print("Loading and merging PEFT LoRA ...", flush=True)
    peft_model = PeftModel.from_pretrained(model, args.peft_lora,
                                           is_trainable=False)
    merged = peft_model.merge_and_unload(safe_merge=True)
    merged.eval()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output, safe_serialization=True)
    (output / "merge_manifest.json").write_text(json.dumps({
        "base_model": args.base_model,
        "expert_delta": args.expert_delta,
        "peft_lora": args.peft_lora,
        "merge_order": ["raw_base", "expert_delta", "peft_lora"],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved merged model -> {output}", flush=True)


if __name__ == "__main__":
    main()
