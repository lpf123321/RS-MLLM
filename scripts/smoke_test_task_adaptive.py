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
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evalsets import levircc, mme, vrsbench
from evaluation.main import SYSTEM_PROMPTS
from evaluation.router import rules
from evaluation.adapters.router_pruned import RouterPrunedAdapter
from evaluation.router.task_prune_config import THRESHOLD_TASK_PRUNE_CONFIG

SHARED = os.environ.get("DATA_ROOT", str(Path(__file__).resolve().parent.parent / "datasets"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--general_lora", required=True)
    ap.add_argument("--grounding_lora", required=True)
    ap.add_argument("--change_lora", required=True)
    ap.add_argument("--caption_lora", default=None)
    args = ap.parse_args()

    adapter = RouterPrunedAdapter(
        args.model_path,
        general_lora=args.general_lora,
        grounding_lora=args.grounding_lora,
        change_lora=args.change_lora,
        caption_lora=args.caption_lora,
        prune_method="l2",
        task_prune_config=THRESHOLD_TASK_PRUNE_CONFIG,
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
        config = THRESHOLD_TASK_PRUNE_CONFIG.get(task, {})
        method = config.get("method", adapter.prune_method)
        keep = config.get("keep_ratio", adapter.keep_ratio)
        expert = rules.TASK_TO_EXPERT.get(task, rules.GENERAL)
        pred = adapter.generate(s["images"], s["prompt"])
        print(f"\n[task={task} -> expert={expert} method={method} keep={keep}]", flush=True)
        print(f"  Q: {s['prompt'][:80]!r}", flush=True)
        print(f"  A: {pred[:200]!r}", flush=True)

    print("\nTASK-ADAPTIVE SMOKE TEST DONE", flush=True)


if __name__ == "__main__":
    main()
