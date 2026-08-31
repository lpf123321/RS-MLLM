#!/usr/bin/env python3
"""Referring R sweep plots (new Grounding Expert Update weights)."""

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

RESULTS_DIR = os.environ.get("SWEEP_DIR", "prune/output/delta_ground_update_r_sweep")
OUT_DIR = os.environ.get("PLOT_DIR", RESULTS_DIR)
RATIOS = [0.10, 0.25, 0.35, 0.50, 0.65, 0.75, 0.90, 1.00]
METHODS = ["uniform", "random", "mmtok", "l2norm", "divprune", "scope_l2"]
LABELS = {
    "uniform": "Uniform", "random": "Random", "mmtok": "MMTok",
    "l2norm": "L2Norm", "divprune": "DivPrune", "scope_l2": "ScopeL2",
}
COLORS = {
    "uniform": "#4C78A8", "random": "#F58518", "mmtok": "#54A24B",
    "l2norm": "#E45756", "divprune": "#72B7B2", "scope_l2": "#B279A2",
}
MARKERS = {
    "uniform": "o", "random": "s", "mmtok": "^",
    "l2norm": "D", "divprune": "v", "scope_l2": "P",
}
PANELS = [
    ("vrsbench_referring", "Acc@0.5", "VRSBench Referring Acc@0.5"),
    ("xlrs_grounding", "Acc@0.5", "XLRS grounding Acc@0.5"),
]


def load(dataset, ratio, metric):
    path = os.path.join(RESULTS_DIR, "{m}", f"{dataset}_r{int(round(ratio * 100)):02d}.json")
    ys = []
    for method in METHODS:
        with open(path.format(m=method), encoding="utf-8") as f:
            ys.append(json.load(f)["referring"][metric])
    return ys


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, (dataset, metric, ylabel) in zip(axes, PANELS):
        for method in METHODS:
            ys = [json.load(open(
                os.path.join(RESULTS_DIR, method, f"{dataset}_r{int(round(r * 100)):02d}.json"),
                encoding="utf-8"))["referring"][metric] for r in RATIOS]
            ax.plot(RATIOS, ys, color=COLORS[method], marker=MARKERS[method],
                    linewidth=2.0, markersize=5, label=LABELS[method])
        ax.set_xlabel("Keep ratio R")
        ax.set_ylabel(ylabel)
        ax.set_xticks(RATIOS)
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(True, alpha=0.25, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(METHODS),
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(OUT_DIR, "r_sweep_referring.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "r_sweep_referring.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved referring plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
