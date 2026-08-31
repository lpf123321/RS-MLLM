#!/usr/bin/env python3
"""Smoke test DeltaRouter + token_compression selectors."""
import argparse
from rsmllm.config import MODELS_ROOT as M_ROOT
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.adapters.delta_pruned import DeltaPrunedAdapter
from evaluation.evalsets import vrsbench
from evaluation.main import SYSTEM_PROMPTS

SHARED = os.environ.get("DATA_ROOT", str(Path(__file__).resolve().parent.parent / "datasets"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="M_ROOT/lora_expert/base_model")
    parser.add_argument("--general_lora", default="M_ROOT/lora_expert/lora/general/delta_model.pt")
    parser.add_argument("--grounding_lora", default="M_ROOT/lora_expert/lora/grounding/delta_model.pt")
    parser.add_argument("--change_lora", default="M_ROOT/lora_expert/lora/change/delta_model.pt")
    parser.add_argument("--keep_ratio", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    samples = vrsbench.load_data(f"{SHARED}/VRSBench/vrsbench_eval.jsonl")
    selected = []
    seen = set()
    for sample in samples:
        if sample["task"] not in seen:
            selected.append(sample)
            seen.add(sample["task"])
        if len(selected) == 3:
            break
    batch = [(sample["images"], sample["prompt"]) for sample in selected]

    for method in ("l2norm", "uniform", "divprune"):
        print(f"\n=== method={method} keep_ratio={args.keep_ratio} ===", flush=True)
        adapter = DeltaPrunedAdapter(
            args.model_path,
            general_lora=args.general_lora,
            grounding_lora=args.grounding_lora,
            change_lora=args.change_lora,
            prune_method=method,
            keep_ratio=args.keep_ratio,
        )
        adapter.system_prompt = SYSTEM_PROMPTS["vrsbench"]
        predictions = adapter.batch_generate(batch, batch_size=args.batch_size)
        for (_, prompt), prediction in zip(batch, predictions):
            print(f"Q: {prompt[:70]!r}", flush=True)
            print(f"A: {prediction[:160]!r}", flush=True)

    print("\nDELTA TOKEN COMPRESSION SMOKE TEST DONE", flush=True)


if __name__ == "__main__":
    main()
