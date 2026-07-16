#!/usr/bin/env python3
import argparse
import json
import os

from tqdm import tqdm

from evaluation.adapters import Qwen3VLAdapter
from evaluation.evalsets import vrsbench, mme, xlrs, levircc

SHARED = "/users/u2024311136/shared/shared_datasets"

SYSTEM_PROMPTS = {
    "vrsbench": "Obey the task prefix: [VQA] answer with a short phrase, [CAP] describe the image in detail, [REF] output bounding box coordinates as {<x1><y1><x2><y2>}.",
    "mme": "Answer EXACTLY in format 'X. (X) FullOptionText', e.g. 'D. (D) White'. Never omit the parenthesized letter.",
    "xlrs": "Answer EXACTLY in format 'X. (X) FullOptionText', e.g. 'A. (A) Some description'. Never omit the parenthesized letter.",
    "levircc": "Describe the changes between the two images concisely.",
}

DATASETS = {
    "vrsbench": (vrsbench, f"{SHARED}/VRSBench/vrsbench_eval.jsonl"),
    "mme": (mme, f"{SHARED}/MME-RealWorld-RS/mme_rs.jsonl"),
    "xlrs": (xlrs, f"{SHARED}/XLRS-Bench-lite/xlrs.jsonl"),
    "levircc": (levircc, f"{SHARED}/LEVIR-CC/levircc_test.jsonl"),
}


def evaluate(adapter, module, data_path, max_samples, eval_batch_size):
    samples = module.load_data(data_path)
    if 0 < max_samples < len(samples):
        samples = samples[:max_samples]

    grouped = {}
    for s in samples:
        task = s["task"]
        if task not in grouped:
            grouped[task] = {"references": [], "predictions": []}

    batch = [(s["images"], s["prompt"]) for s in samples]
    predictions = adapter.batch_generate(batch, batch_size=eval_batch_size)

    # Debug: print first 3 predictions per task
    for i, (s, pred) in enumerate(zip(samples[:3], predictions[:3])):
        print(f"  [DEBUG {s['task']} {i}] ref='{s['references'][0][:80]}'  pred='{pred[:120]}'", flush=True)

    for s, pred in zip(samples, predictions):
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
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--compile_model", action="store_true", default=False)
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = args.data_root if os.path.isabs(args.data_root) else os.path.join(repo_root, args.data_root)
    ds_names = list(DATASETS.keys()) if "all" in args.datasets else args.datasets

    print("Loading model adapter ...", flush=True)
    adapter = Qwen3VLAdapter(args.model_path, device=args.device,
                             compile_model=args.compile_model)
    print(f"  Model loaded on {adapter.device}", flush=True)
    print(f"  Compile model: {args.compile_model}", flush=True)
    print(f"  Eval batch size: {args.eval_batch_size}", flush=True)

    all_results = {}
    for ds_name in ds_names:
        module, data_path = DATASETS[ds_name]
        if not os.path.exists(data_path):
            print(f"\n[WARN] Data not found: {data_path}, skipping {ds_name}")
            continue

        adapter.system_prompt = SYSTEM_PROMPTS.get(ds_name, "")
        print(f"\n{'=' * 60}")
        print(f"  Evaluating {ds_name} ({module.NAME})")
        print(f"  Data: {data_path}")
        print(f"  System: {adapter.system_prompt}")
        print(f"{'=' * 60}")

        all_results[ds_name] = evaluate(adapter, module, data_path, args.max_samples, args.eval_batch_size)

    print_results(all_results)
    save_results(all_results, args.output)


if __name__ == "__main__":
    main()
