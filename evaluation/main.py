#!/usr/bin/env python3
import argparse
import json
from rsmllm.config import DATA_ROOT
import os
from pathlib import Path
import random
import re

from PIL import Image
from tqdm import tqdm

from evaluation.evalsets import vrsbench, mme, xlrs, levircc, xlrs_caption, xlrs_grounding

SHARED = DATA_ROOT
OLD_DATA_ROOT = os.environ.get("DATA_ROOT_OLD", "")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_prompt(name: str) -> str:
    path = os.path.join(_REPO_ROOT, "evaluation", "prompts", name)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    return ""


_XLRS_CAPTION_PROMPT = _load_prompt("xlrs_caption_en.txt")

SYSTEM_PROMPTS = {
    "vrsbench": "Obey the task prefix:\n- [VQA] Answer with a single word or short phrase only. No extra text.\n- [CAP] Describe the image in detail.\n- [REF] Output ONLY the bounding box in format {<x1><y1><x2><y2>} with integer coordinates 0-100, e.g. {<25><40><33><60>}. No other text.",
    "mme": "Answer EXACTLY in format \"X. (X) FullOptionText\" with the letter repeated in parentheses. Example: \"D. (D) White\". You MUST include the parenthesized letter - never omit it. Output ONLY that line.",
    "xlrs": "Answer EXACTLY in format \"X. (X) FullOptionText\" with the letter repeated in parentheses. Example: \"A. (A) Some description\". You MUST include the parenthesized letter - never omit it. Output ONLY that line.",
    "levircc": "Describe the changes between the two images concisely in 1-2 sentences.",
    "xlrs_caption": _XLRS_CAPTION_PROMPT or "Describe the image in detail.",
    "xlrs_grounding": "Obey the task prefix:\n- [REF] Output the bounding box coordinates of the described object, directly without explanation.",
}

GEOEYES_PROMPTS = {
    "vrsbench": "Obey the task prefix:\n- [VQA] Answer with a single word or short phrase only. No extra text.\n- [CAP] Describe the image in detail.\n- [REF] Output ONLY the bounding box in format {<x1><y1><x2><y2>} with integer coordinates 0-100, e.g. {<25><40><33><60>}. No other text.",
    "mme": "Answer EXACTLY in format \"X. (X) FullOptionText\" with the letter repeated in parentheses. Example: \"D. (D) White\". You MUST include the parenthesized letter - never omit it. Output ONLY that line.",
    "xlrs": "Answer EXACTLY in format \"X. (X) FullOptionText\" with the letter repeated in parentheses. Example: \"A. (A) Some description\". You MUST include the parenthesized letter - never omit it. Output ONLY that line.",
    "levircc": "Describe the changes between the two images concisely in 1-2 sentences.",
}

DATASETS = {
    "vrsbench": (vrsbench, f"{SHARED}/VRSBench/vrsbench_eval.jsonl"),
    "mme": (mme, f"{SHARED}/MME-RealWorld-RS/mme_rs.jsonl"),
    "xlrs": (xlrs, f"{SHARED}/XLRS-Bench-lite/xlrs.jsonl"),
    "levircc": (levircc, f"{SHARED}/LEVIR-CC/levircc_test.jsonl"),
    "xlrs_caption": (xlrs_caption, os.path.join(_REPO_ROOT, "evaluation/data/xlrs_caption.jsonl")),
    "xlrs_grounding": (xlrs_grounding, os.path.join(_REPO_ROOT, "evaluation/data/xlrs_grounding.jsonl")),
}

TASK_MAX_TOKENS = {
    "vqa": 64,
    "referring": 32,
    "caption": 256,
}


def evaluate(
    adapter,
    module,
    data_path,
    max_samples,
    eval_batch_size,
    start_offset=0,
    random_samples=0,
    sample_seed=2026,
    task_max_tokens=None,
    subtask=None,
):
    samples = module.load_data(data_path)
    if start_offset > 0:
        samples = samples[start_offset:]
    if random_samples > 0:
        samples = random.Random(sample_seed).sample(
            samples, min(random_samples, len(samples))
        )
    elif 0 < max_samples < len(samples):
        samples = samples[:max_samples]

    # 子任务过滤: 逐专家评测只评对应任务(如 caption 专家只评 caption)
    if subtask:
        _tag = {
            "caption": "caption",
            "referring": "referring", "bbox": "referring",
            "vqa": "vqa", "mcq": "mcq",
            "change": "change", "change_caption": "change",
        }
        want = _tag.get(subtask)
        if want:
            samples = [s for s in samples if s.get("task") == want]
        if not samples:
            print(f"  [warn] --subtask {subtask} 无匹配样本(任务标签 {want})", flush=True)
            return {}, {}

    # Rewrite image paths for cross-server compatibility
    if OLD_DATA_ROOT:
        for s in samples:
            s["images"] = [img.replace(OLD_DATA_ROOT, SHARED) for img in s["images"]]

    # Group samples by task for per-task max_new_tokens
    task_indices = {}
    for i, s in enumerate(samples):
        task_indices.setdefault(s["task"], []).append(i)

    predictions = [None] * len(samples)
    for task, indices in task_indices.items():
        batch = [(samples[idx]["images"], samples[idx]["prompt"]) for idx in indices]
        token_limits = task_max_tokens or TASK_MAX_TOKENS
        max_tok = token_limits.get(task, adapter.max_new_tokens)
        print(f"  [{task}] max_new_tokens={max_tok}, samples={len(batch)}", flush=True)
        previous_max_new_tokens = adapter.max_new_tokens
        adapter.max_new_tokens = max_tok
        try:
            preds = adapter.batch_generate(batch, batch_size=eval_batch_size)
        finally:
            adapter.max_new_tokens = previous_max_new_tokens
        for idx, pred in zip(indices, preds):
            predictions[idx] = pred

    grouped = {}
    for s in samples:
        task = s["task"]
        if task not in grouped:
            grouped[task] = {"references": [], "predictions": []}

    # Normalize referring bbox predictions from pixel space to 0-100
    # Support {<x1><y1><x2><y2>}, {x1,y1,x2,y2}, and training-style [x1,y1,x2,y2]
    _BBOX_ANGLE_RE = re.compile(r"\{<\s*(\d+)\s*><\s*(\d+)\s*><\s*(\d+)\s*><\s*(\d+)\s*>\}")
    _BBOX_COMMA_RE = re.compile(
        r"\{\s*(\d+(?:\.\d+)?)\s*[,;\s]+\s*(\d+(?:\.\d+)?)\s*[,;\s]+\s*(\d+(?:\.\d+)?)\s*[,;\s]+\s*(\d+(?:\.\d+)?)\s*\}"
    )
    _BBOX_BRACKET_RE = re.compile(
        r"\[\s*(\d+(?:\.\d+)?)\s*[,;\s]+\s*(\d+(?:\.\d+)?)\s*[,;\s]+\s*(\d+(?:\.\d+)?)\s*[,;\s]+\s*(\d+(?:\.\d+)?)\s*\]"
    )
    for i, s in enumerate(samples):
        if s["task"] != "referring":
            continue
        pred = _BBOX_COMMA_RE.sub(r"{<\1><\2><\3><\4>}", predictions[i])
        pred = _BBOX_BRACKET_RE.sub(r"{<\1><\2><\3><\4>}", pred)
        m = _BBOX_ANGLE_RE.search(pred)
        if not m:
            continue
        coords = list(map(float, m.groups()))
        if max(coords) <= 100:
            predictions[i] = pred
            continue
        img_path = s["images"][0]
        try:
            with Image.open(img_path) as img:
                w, h = img.size
            scale_x = 100.0 / max(w, 1)
            scale_y = 100.0 / max(h, 1)
            norm = [int(max(0, min(100, round(x * scale_x if j % 2 == 0 else x * scale_y))))
                    for j, x in enumerate(coords)]
            predictions[i] = "{{<{}><{}><{}><{}>}}".format(*norm)
        except Exception:
            predictions[i] = pred

    for s, pred in zip(samples, predictions):
        grouped[s["task"]]["predictions"].append(pred)
        grouped[s["task"]]["references"].append(s["references"])

    # Debug: print first prediction of each task
    seen = set()
    for s, pred in zip(samples, predictions):
        if s["task"] not in seen:
            seen.add(s["task"])
            ref_preview = str(s["references"][0])[:100] if s["references"] else ""
            pred_preview = str(pred)[:150] if pred is not None else ""
            print(f"  [DEBUG {s['task']}] ref='{ref_preview}'  pred='{pred_preview}'", flush=True)

    results = {}
    per_sample = []
    for task, data in grouped.items():
        metrics = module.TASK_METRICS.get(task, [])
        task_result = {}
        for metric in metrics:
            task_result.update(metric.compute(data["references"], data["predictions"]))
        task_result["samples"] = len(data["predictions"])
        results[task] = task_result
        for refs, pred in zip(data["references"], data["predictions"]):
            per_sample.append({"task": task, "reference": refs, "prediction": pred})
    return results, per_sample


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
                        choices=["qwen3vl", "qwen35vl", "qwen35_2b", "qwen35_pruned",
                                 "qwen35_mmtok", "qwen35_divprune", "qwen35_fourier",
                                 "geoeyes", "router"])
    parser.add_argument("--datasets", type=str, nargs="+",
                        choices=list(DATASETS.keys()) + ["all"], default=["all"])
    parser.add_argument("--data_root", type=str, default="output")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--random_samples", type=int, default=0,
                        help="Randomly select this many samples per dataset")
    parser.add_argument("--sample_seed", type=int, default=2026,
                        help="Seed for --random_samples")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="evaluation/results.json")
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--pruner", type=str,
                        choices=["uniform", "random", "mmtok", "l2norm", "divprune", "scope_l2"],
                        default="mmtok")
    parser.add_argument("--keep_ratio", type=float, default=0.5)
    parser.add_argument("--pruner_seed", type=int, default=2026)
    parser.add_argument("--compile_model", action="store_true", default=False)
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA checkpoint (qwen3vl adapter only)")
    parser.add_argument("--general_lora", type=str, default=None,
                        help="General Understanding LoRA path (router adapter only)")
    parser.add_argument("--grounding_lora", type=str, default=None,
                        help="Grounding LoRA path (router adapter only)")
    parser.add_argument("--change_lora", type=str, default=None,
                        help="Change LoRA path (router adapter only)")
    parser.add_argument("--caption_lora", type=str, default=None,
                        help="Caption LoRA path (router adapter only)")
    parser.add_argument("--image_min_pixels", type=int, default=262144,
                        help="Min pixels for image processing")
    parser.add_argument("--image_max_pixels", type=int, default=1048576,
                        help="Max pixels for image processing")
    parser.add_argument("--save_predictions", type=str, default=None,
                        help="Path to save per-sample predictions JSON")
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
    parser.add_argument("--subtask", type=str, default=None,
                        help="Only evaluate one subtask per dataset "
                             "(vqa / caption / referring / mcq / change); "
                             "e.g. --subtask caption 评 Caption 子任务")
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
    elif args.adapter == "qwen35_divprune":
        from evaluation.adapters.qwen35_divprune import Qwen35DivPruneAdapter
        adapter = Qwen35DivPruneAdapter(
            args.model_path, device=args.device, compile_model=args.compile_model,
            keep_ratio=args.keep_ratio, pruner_seed=args.pruner_seed,
        )
        prompt_map = SYSTEM_PROMPTS
    elif args.adapter == "qwen35_fourier":
        from evaluation.adapters.qwen35_fourier import Qwen35FourierAdapter
        adapter = Qwen35FourierAdapter(
            args.model_path, device=args.device, compile_model=args.compile_model,
            keep_ratio=args.keep_ratio, pruner_seed=args.pruner_seed,
        )
        prompt_map = SYSTEM_PROMPTS
    elif args.adapter == "qwen35_pruned":
        from evaluation.adapters.qwen35_pruned import Qwen35PrunedAdapter
        adapter = Qwen35PrunedAdapter(
            args.model_path, device=args.device, compile_model=args.compile_model,
            keep_ratio=args.keep_ratio, pruner=args.pruner,
            pruner_seed=args.pruner_seed,
        )
        prompt_map = SYSTEM_PROMPTS
    elif args.adapter == "qwen35_mmtok":
        from evaluation.adapters.qwen35_mmtok import Qwen35MMTokAdapter
        adapter = Qwen35MMTokAdapter(
            args.model_path, device=args.device, compile_model=args.compile_model,
            keep_ratio=args.keep_ratio,
        )
        prompt_map = SYSTEM_PROMPTS
    elif args.adapter == "qwen35_2b":
        from evaluation.adapters.qwen35_2b import Qwen35_2BAdapter
        adapter = Qwen35_2BAdapter(args.model_path, device=args.device,
                                   compile_model=args.compile_model)
        prompt_map = SYSTEM_PROMPTS
    elif args.adapter == "qwen35vl":
        from evaluation.adapters.qwen35vl import Qwen35VLAdapter
        adapter = Qwen35VLAdapter(args.model_path, device=args.device,
                                  compile_model=args.compile_model)
        prompt_map = SYSTEM_PROMPTS
    elif args.adapter == "router":
        from evaluation.adapters.router import RouterAdapter
        if not (args.general_lora and args.grounding_lora and args.change_lora):
            parser.error("--adapter router requires --general_lora/--grounding_lora/--change_lora")
        adapter = RouterAdapter(
            args.model_path, device=args.device,
            general_lora=args.general_lora,
            grounding_lora=args.grounding_lora,
            change_lora=args.change_lora,
            caption_lora=args.caption_lora,
            image_min_pixels=args.image_min_pixels,
            image_max_pixels=args.image_max_pixels,
        )
        prompt_map = SYSTEM_PROMPTS
    else:
        from evaluation.adapters.qwen3vl import Qwen3VLAdapter
        adapter = Qwen3VLAdapter(args.model_path, device=args.device,
                                 compile_model=args.compile_model)
        prompt_map = SYSTEM_PROMPTS

    if hasattr(adapter, "device"):
        print(f"  Model loaded on {adapter.device}", flush=True)
    print(f"  Adapter: {args.adapter}", flush=True)
    print(f"  Eval batch size: {args.eval_batch_size}", flush=True)
    if args.random_samples:
        print(f"  Random samples: {args.random_samples}, seed={args.sample_seed}", flush=True)
    if args.adapter == "qwen35_pruned":
        print(f"  Pruner: {args.pruner}, keep_ratio={args.keep_ratio}, seed={args.pruner_seed}", flush=True)

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

            ds_results, ds_preds = evaluate(
                adapter, module, data_path, args.max_samples, args.eval_batch_size,
                random_samples=args.random_samples, sample_seed=args.sample_seed,
                subtask=args.subtask,
            )
            all_results[ds_name] = ds_results
            if args.save_predictions:
                os.makedirs(os.path.dirname(args.save_predictions) or ".", exist_ok=True)
                base, ext = os.path.splitext(args.save_predictions)
                pred_path = f"{base}_{ds_name}{ext}"
                with open(pred_path, "w", encoding="utf-8") as f:
                    json.dump(ds_preds, f, indent=2, ensure_ascii=False)
                print(f"  Predictions saved to {pred_path}", flush=True)

    finally:
        if all_results:
            print_results(all_results)
            # Add efficiency info to output
            if hasattr(adapter, "peak_memory"):
                for ds_name in all_results:
                    all_results[ds_name]["_efficiency"] = {
                        "peak_memory_gb": round(adapter.peak_memory / 1e9, 2),
                        "inference_time_s": round(getattr(adapter, "inference_time", 0), 1),
                        "samples_per_sec": round(getattr(adapter, "inference_samples_per_sec", 0), 2),
                    }
            save_results(all_results, args.output)
        if hasattr(adapter, "close"):
            adapter.close()


if __name__ == "__main__":
    main()
