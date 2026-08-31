#!/usr/bin/env python3
"""GPU smoke test: system prompt + forced-<think> ablation for the router adapter.

Compares, at baseline (keep_ratio=1.0, no pruning), two prompt configurations:
  - force_think=True  : the current behavior (add_generation_prompt=False + empty <think>)
  - force_think=False : add_generation_prompt=True + enable_thinking=False

Each sample is run with the dataset's system prompt, so we can check whether the
short-answer degeneration (``\\nuser\\nuser...`` / ``)\\n-``) disappears and whether
MCQ / referring outputs become clean.

Usage (on a GPU node):
    python scripts/smoke_test_router_pruned.py \
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

from evaluation.evalsets import levircc, mme, vrsbench
from evaluation.main import SYSTEM_PROMPTS
from evaluation.adapters.router_pruned import RouterPrunedAdapter

SHARED = DATA_ROOT


def pick_samples(dataset_modules):
    """Return {dataset_name: [samples]} with one sample per task of each dataset."""
    grouped = {}
    for ds_name, mod, path in dataset_modules:
        seen = {}
        for s in mod.load_data(path):
            seen.setdefault(s["task"], s)
        grouped[ds_name] = list(seen.values())
    return grouped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--general_lora", required=True)
    ap.add_argument("--grounding_lora", required=True)
    ap.add_argument("--change_lora", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    adapter = RouterPrunedAdapter(
        args.model_path,
        general_lora=args.general_lora,
        grounding_lora=args.grounding_lora,
        change_lora=args.change_lora,
        prune_method="l2",
        keep_ratio=1.0,
    )

    grouped = pick_samples([
        ("vrsbench", vrsbench, f"{SHARED}/VRSBench/vrsbench_eval.jsonl"),
        ("mme", mme, f"{SHARED}/MME-RealWorld-RS/mme_rs.jsonl"),
        ("levircc", levircc, f"{SHARED}/LEVIR-CC/levircc_test.jsonl"),
    ])

    for ds_name, samples in grouped.items():
        if not samples:
            continue
        adapter.system_prompt = SYSTEM_PROMPTS.get(ds_name, "")
        batch = [(s["images"], s["prompt"]) for s in samples]
        print(f"\n######## {ds_name} (system_prompt={adapter.system_prompt[:40]!r}...) ########",
              flush=True)
        for force_think in (True, False):
            adapter.force_think = force_think
            preds = adapter.batch_generate(batch, batch_size=args.batch_size)
            print(f"\n--- force_think={force_think} ---", flush=True)
            for (_, prompt), p in zip(batch, preds):
                print(f"  Q: {prompt[:70]!r}", flush=True)
                print(f"  A: {p!r}", flush=True)

    print("\nSMOKE TEST DONE", flush=True)


if __name__ == "__main__":
    main()
