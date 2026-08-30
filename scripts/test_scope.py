#!/usr/bin/env python3
"""
SCOPE-L2 微量数据集检验 — 端到端测试剪枝管道。
在 4 个数据集各取 3 条样本，分别运行 baseline 和 SCOPE。

用法（SLURM）:
  sbatch slurm_scripts/test_scope.sh

或直接运行:
  python3 scripts/test_scope.py --model_path /home/u2024311149/models/Qwen3.5-4B
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


# ── 数据集路径 ──
DATASETS = {
    "mme":      "datasets_data/mme_rs.jsonl",
    "xlrs":     "datasets_data/xlrs.jsonl",
    "vrsbench": "datasets_data/vrsbench_eval.jsonl",
    "levircc":  "datasets_data/levircc_test.jsonl",
}


def load_samples(data_path, num):
    """从 JSONL 加载 num 条样本，统一为 {images, prompt, references} 格式。"""
    samples = []
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(data_path) as f:
        for line in f:
            if len(samples) >= num:
                break
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            parsed = _parse_sample(raw, repo_root)
            if parsed:
                samples.append(parsed)
    return samples


def _parse_sample(raw, repo_root):
    """将原始 JSONL 行统一为 {images, prompt, references}。"""
    # 兼容 "messages" 格式（MME, XLRS, VRSBench）
    if "messages" in raw:
        msgs = raw["messages"]
        if isinstance(msgs, str):
            try:
                msgs = eval(msgs)
            except Exception:
                return None
        images_list = []
        prompt = ""
        if isinstance(msgs, list):
            for msg in msgs:
                if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                    for item in msg["content"]:
                        if isinstance(item, dict):
                            if item.get("type") == "image":
                                img = item.get("image", "")
                                if not os.path.isabs(img):
                                    img = os.path.join(repo_root, img)
                                if os.path.exists(img):
                                    images_list.append(img)
                            elif item.get("type") == "text":
                                prompt = item.get("text", "")
        if not images_list:
            return None
        references = raw.get("references", raw.get("reference", ""))
        if isinstance(references, str):
            references = [references]
        return {"images": images_list, "prompt": prompt, "references": references}

    # 兼容 legacy "images" / "prompt" 格式（LEVIR-CC）
    if "images" in raw:
        images_list = []
        for img in raw["images"]:
            if not os.path.isabs(img):
                img = os.path.join(repo_root, img)
            if os.path.exists(img):
                images_list.append(img)
        prompt = raw.get("prompt", raw.get("question", ""))
        references = raw.get("references", raw.get("reference", ""))
        if isinstance(references, str):
            references = [references]
        if images_list and prompt:
            return {"images": images_list, "prompt": prompt, "references": references}

    return None


def run_inference(model, processor, sample, max_new_tokens=128):
    """单条样本推理。"""
    messages = [{"role": "user", "content": [
        *[{"type": "image", "image": img} for img in sample["images"]],
        {"type": "text", "text": sample["prompt"]},
    ]}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    text += "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None)
    input_len = inputs["input_ids"].shape[1]
    out = processor.decode(gen[0, input_len:], skip_special_tokens=True)
    return re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/u2024311149/models/Qwen3.5-4B")
    parser.add_argument("--prune_r", type=float, default=0.5)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--datasets", type=str, default="mme,xlrs,vrsbench,levircc")
    parser.add_argument("--baseline", action="store_true", help="Run baseline (no pruning) for comparison")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    ds_list = [d.strip() for d in args.datasets.split(",")]
    print(f"=== SCOPE-L2 微量检验 ===")
    print(f"Model: {args.model_path}")
    print(f"Prune R: {args.prune_r}  (keep {(1-args.prune_r)*100:.0f}%)")
    print(f"Datasets: {ds_list}, samples/dataset: {args.num_samples}")
    print(f"Baseline: {args.baseline}")

    # ── 加载数据 ──
    all_samples = {}
    for name in ds_list:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), DATASETS[name])
        if os.path.exists(path):
            all_samples[name] = load_samples(path, args.num_samples)
            print(f"  [{name}] loaded {len(all_samples[name])}/{args.num_samples} samples")
        else:
            print(f"  [{name}] data NOT FOUND: {path}")
            sys.exit(1)

    # ── 运行模式 ──
    modes = [("scope", args.prune_r)]
    if args.baseline:
        modes = [("baseline", 0)] + modes

    results_all = {}
    for method, r in modes:
        print(f"\n{'='*60}")
        label = f"{method}" if method == "baseline" else f"{method}(R={r})"
        print(f"=== {label} ===")

        # 加载模型
        if method != "baseline":
            from scripts.prune import apply_pruning, enable_pruning
            apply_pruning(model_path=args.model_path)

        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        processor.tokenizer.padding_side = "left"

        if method != "baseline":
            enable_pruning(model, method=method, r=r)

        t0 = time.time()
        results_mode = {}

        for ds_name in ds_list:
            ds_results = []
            for idx, sample in enumerate(all_samples[ds_name]):
                print(f"  [{ds_name}] sample {idx+1}...", end=" ", flush=True)
                try:
                    t_start = time.time()
                    output = run_inference(model, processor, sample, args.max_new_tokens)
                    elapsed = time.time() - t_start
                    ds_results.append({
                        "idx": idx, "prompt": sample["prompt"][:80],
                        "output": output[:300], "time": round(elapsed, 1),
                    })
                    print(f"{elapsed:.1f}s → {output[:80]}")
                except Exception as e:
                    print(f"ERROR: {type(e).__name__}: {e}")
                    ds_results.append({"idx": idx, "error": str(e)})

            results_mode[ds_name] = ds_results

        elapsed_total = time.time() - t0
        print(f"  Total time: {elapsed_total:.1f}s")
        results_all[label] = results_mode

        # 释放模型内存
        del model
        torch.cuda.empty_cache()

    # ── 打印摘要 ──
    print(f"\n{'='*60}")
    print("=== FINAL SUMMARY ===")
    for label, res in results_all.items():
        print(f"\n--- {label} ---")
        for ds_name, entries in res.items():
            ok = sum(1 for e in entries if "error" not in e)
            avg_time = sum(e.get("time", 0) for e in entries) / max(ok, 1)
            print(f"  {ds_name}: {ok}/{len(entries)} ok, avg {avg_time:.1f}s")
            for e in entries:
                out = e.get("output", e.get("error", "?"))[:120].replace("\n", " ")
                print(f"    [{e['idx']}] {out}")

    # ── 保存 ──
    out_path = args.output or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "prune/output/test_scope_results.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results_all, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
