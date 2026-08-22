#!/usr/bin/env python3
"""Evaluate standalone token-compression methods on Delta expert models.

This is the first unified entry point for the methods listed in
``token_compression``. Fourier is intentionally excluded here because it
changes the visual grid and requires a separate image-grid integration path.

Example:
    python -m evaluation.run_delta_prune_eval \
      --methods uniform random mmtok l2norm divprune scope_l2 \
      --keep_ratios 0.5 0.25 --random_samples 1000 --eval_batch_size 32
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.adapters.delta_pruned import DeltaPrunedAdapter
from evaluation.evalsets import levircc, mme, vrsbench, xlrs
from evaluation.main import SYSTEM_PROMPTS, evaluate

SHARED = os.environ.get("DATA_ROOT", "/users/u2024311136/shared/shared_datasets")
DATASETS = {
    "vrsbench": (vrsbench, f"{SHARED}/VRSBench/vrsbench_eval.jsonl"),
    "mme": (mme, f"{SHARED}/MME-RealWorld-RS/mme_rs.jsonl"),
    "xlrs": (xlrs, f"{SHARED}/XLRS-Bench-lite/xlrs.jsonl"),
    "levircc": (levircc, f"{SHARED}/LEVIR-CC/levircc_test.jsonl"),
}
METHODS = ["uniform", "random", "mmtok", "l2norm", "divprune", "scope_l2"]


def label(r):
    return f"r{int(round(r * 100)):02d}"


def main():
    parser = argparse.ArgumentParser(description="Delta token-compression ablation")
    parser.add_argument("--model_path", default="/users/u2024311136/shared/shared_models/lora_expert/base_model")
    parser.add_argument("--general_lora", default="/users/u2024311136/shared/shared_models/lora_expert/lora/general/delta_model.pt")
    parser.add_argument("--grounding_lora", default="/users/u2024311136/shared/shared_models/lora_expert/lora/grounding/delta_model.pt")
    parser.add_argument("--change_lora", default="/users/u2024311136/shared/shared_models/lora_expert/lora/change/delta_model.pt")
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--keep_ratios", nargs="+", type=float, default=[0.5, 0.25])
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--random_samples", type=int, default=1000)
    parser.add_argument("--sample_seed", type=int, default=2026)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--output_dir", default="prune/output/delta_prune_ablation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for method in args.methods:
        adapter = DeltaPrunedAdapter(
            args.model_path,
            general_lora=args.general_lora,
            grounding_lora=args.grounding_lora,
            change_lora=args.change_lora,
            prune_method=method,
            keep_ratio=1.0,
            pruner_seed=args.sample_seed,
            max_new_tokens=256,
        )
        for keep_ratio in args.keep_ratios:
            adapter._set_uniform_keep_ratio(keep_ratio)
            for dataset in args.datasets:
                module, data_path = DATASETS[dataset]
                if not os.path.exists(data_path):
                    raise FileNotFoundError(data_path)
                adapter.system_prompt = SYSTEM_PROMPTS.get(dataset, "")
                out = os.path.join(args.output_dir, method, f"{dataset}_{label(keep_ratio)}.json")
                if os.path.exists(out):
                    print(f"[SKIP] {out}", flush=True)
                    continue
                print(f"[RUN] method={method} keep={keep_ratio} dataset={dataset}", flush=True)
                results, _ = evaluate(
                    adapter, module, data_path, 0, args.eval_batch_size,
                    random_samples=args.random_samples, sample_seed=args.sample_seed,
                )
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"[DONE] {out}", flush=True)


if __name__ == "__main__":
    main()
