#!/usr/bin/env python3
"""Plot the L2 / DivPrune keep-ratio sweep results.

Outputs (into ``prune/output/prune_sweep/``):
  - ``curve_<dataset>.png``   per-dataset detail (L2 vs DivPrune, per task/metric)
  - ``task_sensitivity.png``  normalized per-task sensitivity (L2, relative to
                              baseline keep_ratio=1.0) with a 95% threshold line
  - ``curve_combined.png``    primary metric per dataset side-by-side
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = os.environ.get("SWEEP_DIR", "prune/output/prune_sweep")
RATIOS = [0.1, 0.25, 0.35, 0.5, 0.65, 0.75, 0.9, 1.0]
METHODS = [m.strip() for m in os.environ.get("SWEEP_METHODS", "l2,divprune").split(",") if m.strip()]

METHOD_LABEL = {
    "l2": "L2Norm", "l2norm": "L2Norm", "divprune": "DivPrune",
    "uniform": "Uniform", "random": "Random", "mmtok": "MMTok",
    "scope_l2": "ScopeL2",
}
METHOD_COLOR = {
    "l2": "#1f77b4", "l2norm": "#1f77b4", "divprune": "#d62728",
    "uniform": "#2ca02c", "random": "#9467bd", "mmtok": "#ff7f0e",
    "scope_l2": "#8c564b",
}
METHOD_MARKER = {
    "l2": "o", "l2norm": "o", "divprune": "s", "uniform": "^",
    "random": "x", "mmtok": "D", "scope_l2": "P",
}

# (task, metric, display label) per dataset.
DATASET_METRICS = {
    "vrsbench": [
        ("vqa", "Accuracy", "VQA Accuracy"),
        ("caption", "BLEU-4", "Caption BLEU-4"),
        ("caption", "ROUGE-L", "Caption ROUGE-L"),
        ("referring", "Acc@0.5", "Referring Acc@0.5"),
    ],
    "mme": [
        ("vqa", "MCQ_Accuracy", "MCQ Accuracy"),
        ("vqa", "Accuracy", "Accuracy"),
    ],
    "xlrs": [
        ("vqa", "MCQ_Accuracy", "MCQ Accuracy"),
        ("vqa", "Accuracy", "Accuracy"),
    ],
    "levircc": [
        ("caption", "CIDEr", "CIDEr"),
        ("caption", "BLEU-4", "BLEU-4"),
        ("caption", "ROUGE-L", "ROUGE-L"),
    ],
}

# Task -> (dataset, task, metric) for the normalized sensitivity chart.
TASKS = [
    ("VQA", "vrsbench", "vqa", "Accuracy"),
    ("Caption", "vrsbench", "caption", "BLEU-4"),
    ("Referring", "vrsbench", "referring", "Acc@0.5"),
    ("MCQ", "mme", "vqa", "MCQ_Accuracy"),
    ("Change", "levircc", "caption", "CIDEr"),
]
TASK_COLOR = {
    "VQA": "#1f77b4",
    "Caption": "#2ca02c",
    "Referring": "#d62728",
    "MCQ": "#9467bd",
    "Change": "#ff7f0e",
}


def ratio_file(method, dataset, r):
    return os.path.join(RESULTS_DIR, method, f"{dataset}_r{int(round(r * 100)):02d}.json")


def load_data():
    data = {m: {d: {} for d in DATASET_METRICS} for m in METHODS}
    for m in METHODS:
        for d in DATASET_METRICS:
            for r in RATIOS:
                path = ratio_file(m, d, r)
                if os.path.exists(path):
                    with open(path) as f:
                        data[m][d][r] = json.load(f)
    return data


def series(data, method, dataset, task, metric):
    xs, ys = [], []
    for r in RATIOS:
        res = data[method][dataset].get(r)
        if res and task in res and metric in res[task]:
            xs.append(r)
            ys.append(res[task][metric])
    return xs, ys


def plot_dataset(data, dataset):
    metrics = DATASET_METRICS[dataset]
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), squeeze=False)
    for ax, (task, metric, label) in zip(axes[0], metrics):
        for method in METHODS:
            xs, ys = series(data, method, dataset, task, metric)
            if xs:
                ax.plot(xs, ys, marker=METHOD_MARKER.get(method, "o"),
                        color=METHOD_COLOR.get(method),
                        label=METHOD_LABEL.get(method, method), linewidth=1.5, markersize=4)
        ax.set_title(f"{dataset} — {label}", fontsize=10)
        ax.set_xlabel("keep_ratio")
        ax.set_ylabel(metric)
        ax.set_xticks(RATIOS)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"curve_{dataset}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_task_sensitivity(data, method=None):
    """One line per task, metric normalized to its own baseline (keep_ratio=1.0)."""
    method = method or os.environ.get("SWEEP_PRIMARY_METHOD", METHODS[0])
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name, dataset, task, metric in TASKS:
        xs, ys = series(data, method, dataset, task, metric)
        if not xs:
            continue
        base = None
        norm = []
        for r, y in zip(xs, ys):
            if abs(r - 1.0) < 1e-6:
                base = y
        if base in (None, 0):
            continue
        norm = [y / base for y in ys]
        ax.plot(xs, norm, marker="o", color=TASK_COLOR[name], label=name,
                linewidth=1.8, markersize=5)
    ax.axhline(0.95, color="gray", linestyle="--", linewidth=1, alpha=0.8)
    ax.text(0.10, 0.955, "95% of baseline", color="gray", fontsize=8, va="bottom")
    ax.set_xlabel("keep_ratio")
    ax.set_ylabel("metric / baseline (keep_ratio=1.0)")
    ax.set_xticks(RATIOS)
    ax.set_ylim(0, 1.08)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    ax.set_title(f"Task pruning sensitivity ({METHOD_LABEL.get(method, method)}, normalized to baseline)",
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "task_sensitivity.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_combined(data):
    n = len(DATASET_METRICS)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), squeeze=False)
    for ax, dataset in zip(axes[0], DATASET_METRICS):
        task, metric, label = DATASET_METRICS[dataset][0]
        for method in METHODS:
            xs, ys = series(data, method, dataset, task, metric)
            if xs:
                ax.plot(xs, ys, marker=METHOD_MARKER.get(method, "o"),
                        color=METHOD_COLOR.get(method), label=METHOD_LABEL.get(method, method),
                        linewidth=1.5, markersize=4)
        ax.set_title(f"{dataset} — {label}", fontsize=10)
        ax.set_xlabel("keep_ratio")
        ax.set_ylabel(metric)
        ax.set_xticks(RATIOS)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    labels = ", ".join(METHOD_LABEL.get(m, m) for m in METHODS)
    fig.suptitle(f"{labels} (multi-expert, random 1000, seed 2026)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(RESULTS_DIR, "curve_combined.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    data = load_data()
    found = sum(1 for m in METHODS for d in DATASET_METRICS if data[m][d])
    print(f"Loaded {found} (method, dataset) groups from {RESULTS_DIR}")
    if found == 0:
        print("No results found. Run the sweep first.", file=sys.stderr)
        sys.exit(1)
    for dataset in DATASET_METRICS:
        plot_dataset(data, dataset)
    plot_task_sensitivity(data)
    plot_combined(data)


if __name__ == "__main__":
    main()
