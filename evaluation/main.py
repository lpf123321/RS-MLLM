#!/usr/bin/env python3
import argparse
import json
import os
import re

from PIL import Image
from tqdm import tqdm

from evaluation.evalsets import vrsbench, mme, xlrs, levircc

SHARED = "/users/u2024311136/shared/shared_datasets"

SYSTEM_PROMPTS = {
    "vrsbench": "Obey the task prefix:\n- [VQA] Answer with a single word or short phrase only. No extra text.\n- [CAP] Describe the image in detail.\n- [REF] Output ONLY the bounding box in format {<x1><y1><x2><y2>} with integer coordinates 0-99, e.g. {<25><40><33><60>}. No other text.",
    "mme": "Answer EXACTLY in format \"X. (X) FullOptionText\" with the letter repeated in parentheses. Example: \"D. (D) White\". You MUST include the parenthesized letter - never omit it. Output ONLY that line.",
    "xlrs": "Answer EXACTLY in format \"X. (X) FullOptionText\" with the letter repeated in parentheses. Example: \"A. (A) Some description\". You MUST include the parenthesized letter - never omit it. Output ONLY that line.",
    "levircc": "Describe the changes between the two images concisely in 1-2 sentences.",
}

# GeoEyes uses its own built-in system prompt (tool definitions) so these
# are passed as part of the user message instead.
GEOEYES_PROMPTS = {
    "vrsbench": "Follow the task prefix:\n- [VQA] Answer with a single word or short phrase only.\n- [CAP] Describe the image in detail, covering all visible objects and their layout.\n- [REF] Output the bounding box as {<x1><y1><x2><y2>} with integer coordinates 0-99, e.g. {<25><40><33><60>}.",
    "mme": "This is a multiple-choice question. You MUST answer EXACTLY in this format (including the parentheses):\nX. (X) FullOptionText\nExample: D. (D) White\nOutput ONLY ONE line after <answer>.",
    "xlrs": "This is a multiple-choice question. You MUST answer EXACTLY in this format (including the parentheses):\nX. (X) FullOptionText\nExample: A. (A) Some description\nOutput ONLY ONE line after <answer>.",
    "levircc": "Describe the changes between the two images concisely in 1-2 sentences. Focus on what has changed, not what stayed the same.",
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

    # Normalize referring bbox predictions from pixel space to 0-99
    _BBOX_RE = re.compile(r"\{<\s*(\d+)\s*><\s*(\d+)\s*><\s*(\d+)\s*><\s*(\d+)\s*>\}")
    for i, s in enumerate(samples):
        if s["task"] != "referring":
            continue
        m = _BBOX_RE.search(predictions[i])
        if not m:
            continue
        coords = list(map(float, m.groups()))
        if max(coords) <= 99:
            continue
        img_path = s["images"][0]
        try:
            with Image.open(img_path) as img:
                w, h = img.size
            scale_x = 99.0 / max(w, 1)
            scale_y = 99.0 / max(h, 1)
            norm = [int(max(0, min(99, round(x * scale_x if j % 2 == 0 else x * scale_y))))
                    for j, x in enumerate(coords)]
            predictions[i] = "{{<{}><{}><{}><{}>}}".format(*norm)
        except Exception:
            pass

    for s, pred in zip(samples, predictions):
        grouped[s["task"]]["predictions"].append(pred)
        grouped[s["task"]]["references"].append(s["references"])

    # Debug: print first prediction of each task
    seen = set()
    for s, pred in zip(samples, predictions):
        if s["task"] not in seen:
            seen.add(s["task"])
            print(f"  [DEBUG {s['task']}] ref='{s['references'][0][:100]}'  pred='{pred[:150]}'", flush=True)

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
    parser.add_argument("--adapter", type=str, default="qwen3vl",
                        choices=["qwen3vl", "geoeyes"])
    parser.add_argument("--datasets", type=str, nargs="+",
                        choices=list(DATASETS.keys()) + ["all"], default=["all"])
    parser.add_argument("--data_root", type=str, default="output")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="evaluation/results.json")
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--compile_model", action="store_true", default=False)
    parser.add_argument("--vllm_url", type=str, default=None,
                        help="Connect to existing vLLM server (GeoEyes only)")
    parser.add_argument("--vllm_port", type=int, default=None,
                        help="Port for new vLLM server (GeoEyes only)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.92)
    parser.add_argument("--vllm_max_model_len", type=int, default=4096,
                        help="Max model length for vLLM (GeoEyes only)")
    parser.add_argument("--data_path_overrides", type=str, default=None,
                        help='JSON dict overriding dataset data paths, e.g. '
                         '{\\"vrsbench\\": \\"/path/to/split.jsonl\\"}')
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = args.data_root if os.path.isabs(args.data_root) else os.path.join(repo_root, args.data_root)
    ds_names = list(DATASETS.keys()) if "all" in args.datasets else args.datasets

    if args.data_path_overrides:
        overrides = json.loads(args.data_path_overrides)
        for ds_name, path in overrides.items():
            if ds_name in DATASETS:
                mod, _ = DATASETS[ds_name]
                DATASETS[ds_name] = (mod, path)
                print(f"  Overrode data path for {ds_name}: {path}", flush=True)

    print("Loading model adapter ...", flush=True)
    if args.adapter == "geoeyes":
        from evaluation.adapters.geoeyes import GeoEyesAdapter
        adapter = GeoEyesAdapter(
            args.model_path, device=args.device,
            vllm_url=args.vllm_url, vllm_port=args.vllm_port,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
        )
        prompt_map = GEOEYES_PROMPTS
    else:
        from evaluation.adapters.qwen3vl import Qwen3VLAdapter
        adapter = Qwen3VLAdapter(args.model_path, device=args.device,
                                 compile_model=args.compile_model)
        prompt_map = SYSTEM_PROMPTS

    if hasattr(adapter, "device"):
        print(f"  Model loaded on {adapter.device}", flush=True)
    print(f"  Adapter: {args.adapter}", flush=True)
    print(f"  Eval batch size: {args.eval_batch_size}", flush=True)

    all_results = {}
    try:
        for ds_name in ds_names:
            module, data_path = DATASETS[ds_name]
            if not os.path.exists(data_path):
                print(f"\n[WARN] Data not found: {data_path}, skipping {ds_name}")
                continue

            adapter.system_prompt = prompt_map.get(ds_name, "")
            print(f"\n{'=' * 60}")
            print(f"  Evaluating {ds_name} ({module.NAME})")
            print(f"  Data: {data_path}")
            print(f"  System: {adapter.system_prompt}")
            print(f"{'=' * 60}")

            all_results[ds_name] = evaluate(adapter, module, data_path, args.max_samples, args.eval_batch_size)
    finally:
        if all_results:
            print_results(all_results)
            save_results(all_results, args.output)
        if hasattr(adapter, "close"):
            adapter.close()


if __name__ == "__main__":
    main()
