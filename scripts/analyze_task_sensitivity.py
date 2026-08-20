#!/usr/bin/env python3
"""Compute per-task critical keep-ratio (临界剪枝率) from the R-sweep results.

For each task, finds the smallest keep_ratio whose metric stays within
``(1 - eps)`` of the baseline (keep_ratio = 1.0). Outputs a ``task -> keep_ratio``
table ready to feed ``RouterPrunedAdapter(task_keep_ratio=...)``, and writes it
to ``prune/output/prune_sweep/task_keep_ratio.json``.

Usage:
    python scripts/analyze_task_sensitivity.py [--method l2] [--eps 0.05]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = os.environ.get("SWEEP_DIR", "prune/output/prune_sweep")
RATIOS = [0.1, 0.25, 0.35, 0.5, 0.65, 0.75, 0.9, 1.0]

# task -> [(dataset, task, metric)]。多来源时取最保守（最大的）临界值。
TASKS = [
    ("vqa", "VQA", [("vrsbench", "vqa", "Accuracy")]),
    ("caption", "Caption", [("vrsbench", "caption", "BLEU-4")]),
    ("referring", "Referring", [("vrsbench", "referring", "Acc@0.5")]),
    ("mcq", "MCQ", [
        ("mme", "vqa", "MCQ_Accuracy"),
        ("xlrs", "vqa", "MCQ_Accuracy"),
    ]),
    ("change", "Change", [("levircc", "caption", "CIDEr")]),
]


def ratio_file(method, dataset, r):
    return os.path.join(RESULTS_DIR, method, f"{dataset}_r{int(round(r * 100)):02d}.json")


def load_curve(method, dataset, task, metric):
    xs, ys = [], []
    for r in RATIOS:
        path = ratio_file(method, dataset, r)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        if task in d and metric in d[task]:
            xs.append(r)
            ys.append(d[task][metric])
    return xs, ys


def critical_ratio(xs, ys, eps):
    """Smallest keep_ratio whose metric >= (1-eps) * baseline (r=1.0)."""
    if 1.0 not in xs:
        return None
    base = ys[xs.index(1.0)]
    if base == 0:
        return None
    threshold = (1.0 - eps) * base
    for r, y in zip(xs, ys):
        if y >= threshold:
            return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="l2", choices=["l2", "divprune"])
    ap.add_argument("--eps", type=float, default=0.05)
    args = ap.parse_args()

    print(f"Critical keep-ratio (method={args.method}, eps={args.eps:.0%} drop allowed)\n")
    print(f"{'task':<10} {'source':<22} {'metric':<12} {'baseline':>9} {'critical':>9}")
    print("-" * 66)

    task_keep_ratio = {}
    for task_key, task_name, sources in TASKS:
        crits = []
        for dataset, task, metric in sources:
            xs, ys = load_curve(args.method, dataset, task, metric)
            if not xs:
                continue
            base = ys[xs.index(1.0)] if 1.0 in xs else None
            crit = critical_ratio(xs, ys, args.eps)
            src = f"{dataset}/{metric}"
            base_s = f"{base:.3f}" if base is not None else "n/a"
            crit_s = f"{crit:.2f}" if crit is not None else "n/a"
            print(f"{task_name:<10} {src:<22} {metric:<12} {base_s:>9} {crit_s:>9}")
            if crit is not None:
                crits.append(crit)

            # 详细曲线
            detail = "  ".join(f"r{r:.2f}={y:.3f}" for r, y in zip(xs, ys))
            print(f"           curve: {detail}")

        if crits:
            task_keep_ratio[task_key] = max(crits)  # 多来源取最保守

    print("\n=== task_keep_ratio dict (feed RouterPrunedAdapter) ===")
    print(json.dumps(task_keep_ratio, indent=2))

    eps_s = f"{args.eps:.2f}".rstrip("0").rstrip(".")
    out = os.path.join(RESULTS_DIR, f"task_keep_ratio_{args.method}_eps{eps_s}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"eps": args.eps, "method": args.method, "task_keep_ratio": task_keep_ratio},
                  f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
