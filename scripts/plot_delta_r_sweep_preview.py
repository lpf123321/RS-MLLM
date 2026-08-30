#!/usr/bin/env python3
"""Preview plots for the completed non-Caption Delta R sweep."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


RESULTS_DIR = os.environ.get("SWEEP_DIR", "prune/output/delta_prune_r_sweep")
OUT_DIR = os.environ.get("PLOT_DIR", RESULTS_DIR)
RATIOS = [0.10, 0.25, 0.35, 0.50, 0.65, 0.75, 0.90, 1.00]
METHODS = ["uniform", "random", "mmtok", "l2norm", "divprune", "scope_l2"]
LABELS = {
    "uniform": "Uniform",
    "random": "Random",
    "mmtok": "MMTok",
    "l2norm": "L2Norm",
    "divprune": "DivPrune",
    "scope_l2": "ScopeL2",
}
COLORS = {
    "uniform": "#4C78A8",
    "random": "#F58518",
    "mmtok": "#54A24B",
    "l2norm": "#E45756",
    "divprune": "#72B7B2",
    "scope_l2": "#B279A2",
}
MARKERS = {
    "uniform": "o",
    "random": "s",
    "mmtok": "^",
    "l2norm": "D",
    "divprune": "v",
    "scope_l2": "P",
}

METRICS = [
    ("vrsbench", "vqa", "Accuracy", "VRSBench VQA"),
    ("vrsbench", "referring", "Acc@0.5", "VRSBench Referring"),
    ("mme", "vqa", "MCQ_Accuracy", "MME MCQ"),
    ("xlrs", "vqa", "MCQ_Accuracy", "XLRS MCQ"),
    ("levircc", "caption", "CIDEr", "LEVIR-CC CIDEr"),
    ("levircc", "caption", "BLEU-4", "LEVIR-CC BLEU-4"),
]


def load_metric(method, ratio, dataset, task, metric):
    path = os.path.join(
        RESULTS_DIR, method, f"{dataset}_r{int(round(ratio * 100)):02d}.json"
    )
    with open(path, encoding="utf-8") as f:
        return json.load(f)[task][metric]


def plot(methods, output_name, normalized=False):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), sharex=True)
    axes = axes.ravel()
    for ax, (dataset, task, metric, ylabel) in zip(axes, METRICS):
        for method in methods:
            ys = [load_metric(method, r, dataset, task, metric) for r in RATIOS]
            if normalized:
                baseline = ys[-1]
                ys = [value / baseline if baseline else 0.0 for value in ys]
            ax.plot(
                RATIOS, ys,
                color=COLORS[method], marker=MARKERS[method],
                linewidth=2.0, markersize=5,
                label=LABELS[method],
            )
        ax.set_xlabel("Keep ratio R")
        ax.set_ylabel(f"{ylabel} / R=1" if normalized else ylabel)
        ax.set_xticks(RATIOS)
        # sharex=True suppresses upper-row tick labels by default.
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(True, alpha=0.25, linewidth=0.7)
        if normalized:
            ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=len(methods),
        frameon=False, bbox_to_anchor=(0.5, -0.005),
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(OUT_DIR, f"{output_name}.png"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, f"{output_name}.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    plot(METHODS, "r_sweep_all_methods")
    plot(["l2norm", "scope_l2"], "r_sweep_l2_scope")
    plot(METHODS, "r_sweep_all_methods_normalized", normalized=True)
    plot(["l2norm", "scope_l2"], "r_sweep_l2_scope_normalized", normalized=True)
    print(f"saved plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
