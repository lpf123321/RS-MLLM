"""Offline metric recomputation for saved prediction files.

Recomputes task metrics using the CURRENT (fixed) implementations so that
results computed before commit 1a550af (old CIDEr) can be compared on a
consistent footing.

Usage:
    python evaluation/recompute_metrics.py \
        --predictions outputs/eval_results/sft_stage1_clean/preds_vrsbench_vrsbench.json \
        --evalset vrsbench
    python evaluation/recompute_metrics.py \
        --predictions outputs/eval_results/sft_stage1_clean/preds_levircc.json \
        --evalset levircc
"""

import argparse
import json
import sys
from collections import defaultdict

from evaluation.evalsets import vrsbench, levircc, mme, xlrs

EVALSETS = {
    "vrsbench": (vrsbench, ("vqa", "caption", "referring")),
    "levircc": (levircc, ("caption",)),
    "mme": (mme, ("vqa",)),
    "xlrs": (xlrs, ("vqa",)),
}


def recompute(pred_path: str, evalset_name: str):
    module, allowed_tasks = EVALSETS[evalset_name]
    records = json.load(open(pred_path))
    grouped = defaultdict(lambda: {"references": [], "predictions": []})
    for rec in records:
        task = rec["task"]
        if task not in allowed_tasks:
            continue
        grouped[task]["references"].append(rec["reference"])
        grouped[task]["predictions"].append(rec["prediction"])

    results = {}
    for task, data in grouped.items():
        metrics = module.TASK_METRICS.get(task, [])
        task_result = {}
        for metric in metrics:
            task_result.update(metric.compute(data["references"], data["predictions"]))
        task_result["samples"] = len(data["predictions"])
        results[task] = task_result
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--evalset", required=True, choices=list(EVALSETS.keys()))
    args = ap.parse_args()
    results = recompute(args.predictions, args.evalset)
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
