#!/usr/bin/env python3
import argparse
import json
import os

from tqdm import tqdm

from evaluation.adapters import Qwen3VLAdapter
from evaluation.evalsets import vrsbench, mme, xlrs, levircc

DATASETS = {
    "vrsbench": (vrsbench, "output/vrsbench_eval.jsonl"),
    "mme": (mme, "output/mme_rs.jsonl"),
    "xlrs": (xlrs, "output/xlrs.jsonl"),
    "levircc": (levircc, "output/levircc_test.jsonl"),
}


def evaluate(adapter, module, data_path, max_samples):
    samples = module.load_data(data_path)
    if 0 < max_samples < len(samples):
        samples = samples[:max_samples]

    grouped = {}
    for s in samples:
        task = s["task"]
        if task not in grouped:
            grouped[task] = {"references": [], "predictions": []}

    for s in tqdm(samples, desc=f"[{module.NAME}] Inferring"):
        pred = adapter.generate(s["images"], s["prompt"])
        grouped[s["task"]]["predictions"].append(pred)
        grouped[s["task"]]["references"].append(s["references"])

    results = {}
    for task, data in grouped.items():
        metrics = module.TASK_METRICS.get(task, [])
        task_result = {}
        for metric in metrics:
            task_result.update(metric.compute(data["references"], data["predictions"]))
        task_result["samples"] = len(data["predictions"])
        results[task] = task_result
    return results


def print_results(all_results):
    print("\n" + "=" * 90)
    print("  Evaluation Results")
    print("=" * 90)
    for ds_name, tasks in all_results.items():
        print(f"\n  Dataset: {ds_name}")
        print("  " + "-" * 70)
        for task_name, metrics in tasks.items():
            n = metrics.pop("samples", 0)
            parts = [f"  [{task_name}]  samples={n}"]
            for k, v in sorted(metrics.items()):
                parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
            print(" | ".join(parts))


def save_results(all_results, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="RS-MLLM Evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--datasets", type=str, nargs="+",
                        choices=list(DATASETS.keys()) + ["all"], default=["all"])
    parser.add_argument("--data_root", type=str, default="output")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="evaluation/results.json")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = args.data_root if os.path.isabs(args.data_root) else os.path.join(repo_root, args.data_root)
    ds_names = list(DATASETS.keys()) if "all" in args.datasets else args.datasets

    print("Loading model adapter ...")
    adapter = Qwen3VLAdapter(args.model_path, device=args.device)
    print(f"  Model loaded on {adapter.device}")

    all_results = {}
    for ds_name in ds_names:
        module, rel_path = DATASETS[ds_name]
        data_path = os.path.join(data_root, os.path.basename(rel_path))
        if not os.path.exists(data_path):
            print(f"\n[WARN] Data not found: {data_path}, skipping {ds_name}")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Evaluating {ds_name} ({module.NAME})")
        print(f"  Data: {data_path}")
        print(f"{'=' * 60}")

        all_results[ds_name] = evaluate(adapter, module, data_path, args.max_samples)

    print_results(all_results)
    save_results(all_results, args.output)


if __name__ == "__main__":
    main()
