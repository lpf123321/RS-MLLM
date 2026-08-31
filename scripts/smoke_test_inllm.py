#!/usr/bin/env python3
"""GPU smoke test for pre-LLM x in-LLM combination (RouterPrunedAdapter).

Runs a few VRSBench samples through several pruning configurations to validate:
  - pre-LLM L2 delete + in-LLM FastV-K mask (combination)
  - pre-LLM L2 delete + in-LLM Clip (drop all visual tokens at layer k)
  - in-LLM only (FastV-K / Clip)

Each config loads the model fresh (the in-LLM forward is monkey-patched at init).

Usage (on a GPU node):
    python scripts/smoke_test_inllm.py \
        --model_path .../lora_expert/base_model \
        --general_lora .../lora/general --grounding_lora .../lora/grounding \
        --change_lora .../lora/change
"""
import argparse
from rsmllm.config import DATA_ROOT
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evalsets import vrsbench
from evaluation.main import SYSTEM_PROMPTS
from evaluation.adapters.router_pruned import RouterPrunedAdapter

SHARED = DATA_ROOT

CONFIGS = [
    ("l2 keep=0.5 + FastV-K keep=0.5 (k=2)",
     dict(prune_method="l2", keep_ratio=0.5, in_llm="k2", in_llm_k=2, in_llm_keep_ratio=0.5)),
    ("l2 keep=0.5 + Clip (k=2)",
     dict(prune_method="l2", keep_ratio=0.5, in_llm="clip", in_llm_k=2)),
    ("FastV-K only keep=0.5 (k=2)",
     dict(prune_method="none", in_llm="k2", in_llm_k=2, in_llm_keep_ratio=0.5)),
    ("Clip only (k=2)",
     dict(prune_method="none", in_llm="clip", in_llm_k=2)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--general_lora", required=True)
    ap.add_argument("--grounding_lora", required=True)
    ap.add_argument("--change_lora", required=True)
    args = ap.parse_args()

    samples = vrsbench.load_data(f"{SHARED}/VRSBench/vrsbench_eval.jsonl")
    seen = {}
    for s in samples:
        seen.setdefault(s["task"], s)
    batch = [(s["images"], s["prompt"]) for s in seen.values()]
    print(f"Smoke test on {len(batch)} samples", flush=True)

    for name, cfg in CONFIGS:
        print(f"\n######## {name} ########", flush=True)
        adapter = RouterPrunedAdapter(
            args.model_path,
            general_lora=args.general_lora,
            grounding_lora=args.grounding_lora,
            change_lora=args.change_lora,
            **cfg,
        )
        adapter.system_prompt = SYSTEM_PROMPTS["vrsbench"]
        preds = adapter.batch_generate(batch, batch_size=1)
        for (_, prompt), p in zip(batch, preds):
            print(f"  Q: {prompt[:60]!r}", flush=True)
            print(f"  A: {p[:140]!r}", flush=True)

    print("\nIN-LLM SMOKE TEST DONE", flush=True)


if __name__ == "__main__":
    main()
