#!/usr/bin/env python3
"""GPU smoke test for task-adaptive pruning (RouterPrunedAdapter + task_keep_ratio).

Loads base_model + 3 expert LoRAs once, then runs one sample per task (vqa /
caption / referring / mcq / change), printing the routed task -> expert ->
keep_ratio and the generated answer, to verify the per-task keep_ratio routing.

Usage (on a GPU node):
    python scripts/smoke_test_task_adaptive.py \
        --model_path .../lora_expert/base_model \
        --general_lora .../lora/general --grounding_lora .../lora/grounding \
        --change_lora .../lora/change
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evalsets import levircc, mme, vrsbench
from evaluation.main import SYSTEM_PROMPTS
from evaluation.router import rules
from evaluation.adapters.router_pruned import RouterPrunedAdapter

SHARED = os.environ.get("DATA_ROOT", "/users/u2024311136/shared/shared_datasets")

TASK_KEEP = {"vqa": 0.1, "caption": 0.9, "referring": 0.75, "mcq": 1.0, "change": 0.5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--general_lora", required=True)
    ap.add_argument("--grounding_lora", required=True)
    ap.add_argument("--change_lora", required=True)
    args = ap.parse_args()

    adapter = RouterPrunedAdapter(
        args.model_path,
        general_lora=args.general_lora,
        grounding_lora=args.grounding_lora,
        change_lora=args.change_lora,
        prune_method="l2",
        task_keep_ratio=TASK_KEEP,
    )

    grouped = {}
    for ds_name, mod, path in [
        ("vrsbench", vrsbench, f"{SHARED}/VRSBench/vrsbench_eval.jsonl"),
        ("mme", mme, f"{SHARED}/MME-RealWorld-RS/mme_rs.jsonl"),
        ("levircc", levircc, f"{SHARED}/LEVIR-CC/levircc_test.jsonl"),
    ]:
        for s in mod.load_data(path):
            task = rules.route_task(s["prompt"])
            grouped.setdefault(task, (ds_name, s))

    for task in ("vqa", "caption", "referring", "mcq", "change"):
        if task not in grouped:
            print(f"[task={task}] no sample found, skip", flush=True)
            continue
        ds_name, s = grouped[task]
        adapter.system_prompt = SYSTEM_PROMPTS.get(ds_name, "")
        keep = (adapter.task_keep_ratio or {}).get(task, adapter.keep_ratio)
        expert = rules.TASK_TO_EXPERT.get(task, rules.GENERAL)
        pred = adapter.generate(s["images"], s["prompt"])
        print(f"\n[task={task} -> expert={expert} keep={keep}]", flush=True)
        print(f"  Q: {s['prompt'][:80]!r}", flush=True)
        print(f"  A: {pred[:200]!r}", flush=True)

    print("\nTASK-ADAPTIVE SMOKE TEST DONE", flush=True)


if __name__ == "__main__":
    main()
