#!/usr/bin/env python3
"""Run task-adaptive pruning (RouterPrunedAdapter + task_keep_ratio) on 4 datasets.

Compares against the uniform-ratio baseline already produced by run_prune_sweep.
Uses the same random sampling (seed 2026, N per dataset) so the comparison is
apples-to-apples.

Usage:
    python -m evaluation.run_task_adaptive \
        --model_path .../lora_expert/base_model \
        --general_lora .../lora/general --grounding_lora .../lora/grounding \
        --change_lora .../lora/change \
        --random_samples 1000 --sample_seed 2026 --eval_batch_size 32
"""
import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rsmllm.config import DATA_ROOT

from evaluation.evalsets import levircc, mme, vrsbench, xlrs, xlrs_caption, xlrs_grounding
from evaluation.main import SYSTEM_PROMPTS, evaluate
from evaluation.adapters.router_pruned import RouterPrunedAdapter
from evaluation.router.task_prune_config import THRESHOLD_TASK_PRUNE_CONFIG

SHARED = DATA_ROOT
_REPO_ROOT = str(REPO_ROOT)

DATASETS = {
    "vrsbench": (vrsbench, f"{SHARED}/VRSBench/vrsbench_eval.jsonl"),
    "mme": (mme, f"{SHARED}/MME-RealWorld-RS/mme_rs.jsonl"),
    "xlrs": (xlrs, f"{SHARED}/XLRS-Bench-lite/xlrs.jsonl"),
    "levircc": (levircc, f"{SHARED}/LEVIR-CC/levircc_test.jsonl"),
    "xlrs_caption": (xlrs_caption, os.path.join(_REPO_ROOT, "evaluation/data/xlrs_caption.jsonl")),
    "xlrs_grounding": (xlrs_grounding, os.path.join(_REPO_ROOT, "evaluation/data/xlrs_grounding.jsonl")),
}

# task -> (dataset, task) 用于统计各 task 样本数与平均 keep
TASK_SAMPLE_MAP = {
    "vqa": [("vrsbench", "vqa")],
    "caption": [("vrsbench", "caption")],
    "referring": [("vrsbench", "referring")],
    "mcq": [("mme", "vqa"), ("xlrs", "vqa")],
    "change": [("levircc", "caption")],
}

# 当前专家仍可能拆分 Caption，暂不将 Caption 纳入 Delta 重测。
DEFAULT_DATASETS = ["vrsbench", "mme", "xlrs", "levircc", "xlrs_grounding"]


def main():
    parser = argparse.ArgumentParser(description="Task-adaptive pruning evaluation")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--general_lora", required=True)
    parser.add_argument("--grounding_lora", required=True)
    parser.add_argument("--change_lora", required=True)
    parser.add_argument("--caption_lora", default=None)
    parser.add_argument("--prune_method", default="l2", choices=["l2", "divprune"])
    parser.add_argument("--random_samples", type=int, default=1000)
    parser.add_argument("--sample_seed", type=int, default=2026)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=DEFAULT_DATASETS)
    parser.add_argument(
        "--output_dir",
        default=str(REPO_ROOT / "prune" / "output" / "prune_sweep"),
        help="输出目录(相对路径按仓库根解析)",
    )
    args = parser.parse_args()
    os.chdir(REPO_ROOT)

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir = output_dir.resolve()
    os.makedirs(output_dir, exist_ok=True)

    print("Loading task-adaptive multi-expert model ...", flush=True)
    adapter = RouterPrunedAdapter(
        args.model_path,
        general_lora=args.general_lora,
        grounding_lora=args.grounding_lora,
        change_lora=args.change_lora,
        caption_lora=args.caption_lora,
        prune_method=args.prune_method,
        task_prune_config=THRESHOLD_TASK_PRUNE_CONFIG,
    )

    all_results = {}
    for ds_name in args.datasets:
        module, data_path = DATASETS[ds_name]
        if not os.path.exists(data_path):
            print(f"[WARN] data not found, skip {ds_name}: {data_path}", flush=True)
            continue
        adapter.system_prompt = SYSTEM_PROMPTS.get(ds_name, "")
        print(f"[RUN] task-adaptive dataset={ds_name} "
              f"(random {args.random_samples}, seed {args.sample_seed}, batch {args.eval_batch_size})",
              flush=True)
        results, _ = evaluate(
            adapter, module, data_path, 0, args.eval_batch_size,
            random_samples=args.random_samples, sample_seed=args.sample_seed,
        )
        all_results[ds_name] = results

    all_results["_config"] = {
        "prune_method": args.prune_method,
        "task_prune_config": THRESHOLD_TASK_PRUNE_CONFIG,
        "random_samples": args.random_samples,
        "sample_seed": args.sample_seed,
    }
    out = output_dir / "task_adaptive_results.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {out}", flush=True)
    print(json.dumps(all_results, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
