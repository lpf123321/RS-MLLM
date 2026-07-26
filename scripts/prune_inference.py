#!/usr/bin/env python3
"""
独立推理脚本 —— 不依赖 evaluation 框架，直接加载模型 + 推理 + 算指标。

用法:
  python scripts/prune_inference.py \
      --model_path /path/to/Qwen3.5-4B \
      --data_path datasets/shared_datasets/VRSBench/vrsbench_eval.jsonl \
      --prune_method l2 --prune_r 0.5 \
      --max_samples 100
"""
import argparse
import json
import os
import re
import sys
import time

# Ensure repo root in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from collections import defaultdict

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


def load_model(model_path, device="cuda", prune_method=None, prune_r=0.5,
               load_in_4bit=False, load_in_8bit=False):
    """加载模型，可选启用剪枝和量化。"""
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"

    # 剪枝必须在 from_pretrained 前 monkey-patch
    if prune_method and prune_method != "none":
        from scripts.prune import apply_pruning, enable_pruning
        apply_pruning(model_path=model_path)

    # 量化配置（可选，与剪枝正交兼容）
    quant_kwargs = {}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif load_in_8bit:
        from transformers import BitsAndBytesConfig
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        **quant_kwargs,
    )
    model.eval()

    if prune_method and prune_method != "none":
        enable_pruning(model, method=prune_method, r=prune_r)

    return model, processor


def run_batch_inference(model, processor, samples_batch, max_new_tokens=256, disable_thinking=True):
    """批量推理 —— 多个样本一起过 model.generate()。"""
    texts = []
    all_images = []
    for s in samples_batch:
        messages = [{"role": "user", "content": [
            *[{"type": "image", "image": img} for img in s["images"]],
            {"type": "text", "text": s["prompt"]},
        ]}]
        chat_kwargs = {"add_generation_prompt": True}
        if disable_thinking:
            chat_kwargs["enable_thinking"] = False
        texts.append(processor.apply_chat_template(messages, tokenize=False, **chat_kwargs))
        image_inputs, _ = process_vision_info(messages)
        all_images.append(image_inputs)

    inputs = processor(
        text=texts, images=all_images,
        padding=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None,
        )

    outputs = []
    for i in range(len(samples_batch)):
        input_len = (inputs.input_ids[i] != processor.tokenizer.pad_token_id).sum().item()
        out = processor.batch_decode(
            generated_ids[i:i+1, input_len:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        out = re.sub(r"<think>.*?</think>\s*", "", out, flags=re.DOTALL).strip()
        outputs.append(out)

    img_tokens = (inputs.input_ids == model.config.image_token_id).sum().item()
    total_tokens = inputs.input_ids.shape[1]
    torch.cuda.synchronize()
    return outputs, img_tokens, total_tokens


# ============================================================
# 数据集加载
# ============================================================

def load_vrsbench(data_path, max_samples=0):
    """VRSBench: VQA / Caption / Referring，单图。"""
    prefix_map = {"[VQA]": "vqa", "[CAP]": "caption", "[REF]": "referring"}
    samples = []
    with open(data_path) as f:
        for line in f:
            if max_samples and len(samples) >= max_samples:
                break
            d = json.loads(line)
            user = d["messages"][0]["content"]
            assistant = d["messages"][1]["content"][0]["text"]
            prompt = user[-1]["text"]
            images = [c["image"] for c in user if c["type"] == "image"]
            task = "vqa"
            for pfx, t in prefix_map.items():
                if prompt.startswith(pfx):
                    task = t; break
            samples.append({
                "task": task, "images": images, "prompt": prompt,
                "reference": assistant,
            })
    return samples


def load_mcq(data_path, max_samples=0):
    """MME / XLRS: MCQ，单图。"""
    samples = []
    with open(data_path) as f:
        for line in f:
            if max_samples and len(samples) >= max_samples:
                break
            d = json.loads(line)
            user = d["messages"][0]["content"]
            assistant = d["messages"][1]["content"][0]["text"]
            prompt = user[-1]["text"]
            images = [c["image"] for c in user if c["type"] == "image"]
            samples.append({
                "task": "vqa", "images": images, "prompt": prompt,
                "reference": assistant,
            })
    return samples


def load_levircc(data_path, max_samples=0):
    """LEVIR-CC: 双图变化描述。"""
    samples = []
    with open(data_path) as f:
        for line in f:
            if max_samples and len(samples) >= max_samples:
                break
            d = json.loads(line)
            user = d["messages"][0]["content"]
            prompt = user[-1]["text"]
            images = [c["image"] for c in user if c["type"] == "image"]
            refs = d.get("references", [d["messages"][1]["content"][0]["text"]])
            if isinstance(refs, str):
                refs = [refs]
            samples.append({
                "task": "caption", "images": images, "prompt": prompt,
                "reference": refs,
            })
    return samples


# ============================================================
# 指标计算
# ============================================================

def _tokenize(text):
    return re.sub(r"[^\w\s]", " ", str(text).lower()).split()


def compute_accuracy(references, predictions):
    """Exact match accuracy。"""
    correct = sum(1 for r, p in zip(references, predictions) if str(r).strip().lower() == str(p).strip().lower())
    return correct / len(predictions) if predictions else 0


def compute_mcq_accuracy(references, predictions):
    """MCQ letter-only accuracy: extract answer letter。"""
    correct = 0
    for r, p in zip(references, predictions):
        r_letter = re.match(r"^[A-D]", str(r).strip())
        p_letter = re.match(r"^[A-D]", str(p).strip())
        if r_letter and p_letter and r_letter.group() == p_letter.group():
            correct += 1
    return correct / len(predictions) if predictions else 0


def compute_bleu(references, predictions, max_n=4):
    """Multi-reference BLEU。"""
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    smooth = SmoothingFunction().method1
    list_of_refs = []
    for ref in references:
        if isinstance(ref, list):
            list_of_refs.append([_tokenize(r) for r in ref])
        else:
            list_of_refs.append([_tokenize(ref)])
    hyps = [_tokenize(p) for p in predictions]
    bleus = {}
    for n in range(1, max_n + 1):
        weights = [1.0 / n] * n
        try:
            bleus[f"BLEU-{n}"] = corpus_bleu(list_of_refs, hyps, weights=weights, smoothing_function=smooth)
        except Exception:
            bleus[f"BLEU-{n}"] = 0.0
    return bleus


def compute_rouge_l(references, predictions):
    """ROUGE-L。"""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        scores = []
        for ref, pred in zip(references, predictions):
            r = ref[0] if isinstance(ref, list) else ref
            s = scorer.score(str(r), str(pred))
            scores.append(s["rougeL"].fmeasure)
        return np.mean(scores) if scores else 0.0
    except ImportError:
        return 0.0


def compute_referring_acc(references, predictions, threshold=0.5):
    """Referring expression IoU accuracy。"""
    pattern = re.compile(r"\{<\s*(\d+)\s*><\s*(\d+)\s*><\s*(\d+)\s*><\s*(\d+)\s*>\}")
    correct = 0
    for r, p in zip(references, predictions):
        rm = pattern.search(str(r))
        pm = pattern.search(str(p))
        if not rm or not pm:
            continue
        rx = list(map(int, rm.groups()))
        px = list(map(int, pm.groups()))
        # IoU on 0-99 normalized coords
        ix1, iy1 = max(rx[0], px[0]), max(rx[1], px[1])
        ix2, iy2 = min(rx[2], px[2]), min(rx[3], px[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_r = (rx[2] - rx[0]) * (rx[3] - rx[1])
        area_p = (px[2] - px[0]) * (px[3] - px[1])
        union = area_r + area_p - inter
        iou = inter / union if union > 0 else 0
        if iou >= threshold:
            correct += 1
    return correct / len(predictions) if predictions else 0


# ============================================================
# 主流程
# ============================================================

def _compute_results(samples, predictions, ds_type):
    """根据 samples 和 predictions 计算指标。"""
    grouped = defaultdict(lambda: {"references": [], "predictions": []})
    for s, p in zip(samples, predictions):
        grouped[s["task"]]["references"].append(s["reference"])
        grouped[s["task"]]["predictions"].append(p)
    metric_map = {
        "vqa": [("Accuracy", compute_accuracy), ("MCQ_Accuracy", compute_mcq_accuracy)],
        "caption": [],
        "referring": [("Acc@0.5", lambda r, p: compute_referring_acc(r, p, 0.5))],
    }
    results = {}
    for task, data in grouped.items():
        refs, preds = data["references"], data["predictions"]
        task_result = {"samples": len(preds)}
        if task == "caption":
            bleus = compute_bleu(refs, preds)
            task_result.update(bleus)
            task_result["ROUGE-L"] = compute_rouge_l(refs, preds)
        else:
            for name, fn in metric_map.get(task, []):
                task_result[name] = fn(refs, preds)
        results[task] = task_result
    return results

def main():
    parser = argparse.ArgumentParser(description="Token pruning inference")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to JSONL dataset (omit for ad-hoc inference)")
    parser.add_argument("--image", type=str, nargs="*", default=None,
                        help="Ad-hoc: image path(s) for single inference")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Ad-hoc: text prompt for single inference")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["vrsbench", "mme", "xlrs", "levircc"],
                        help="Dataset type (auto-detect if not set)")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples (0 = all)")
    parser.add_argument("--start_offset", type=int, default=0,
                        help="Skip first N samples")
    parser.add_argument("--prune_method", type=str, default=None,
                        choices=["l2", "k2", "divprune", "none"],
                        help="Pruning method")
    parser.add_argument("--prune_r", type=float, default=0.5,
                        help="Pruning ratio (0.5 = keep 50%)")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Use 4-bit quantization")
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="Use 8-bit quantization")
    args = parser.parse_args()

    # Ad-hoc single image inference mode
    if args.image and args.prompt:
        model, processor = load_model(
            args.model_path, args.device,
            prune_method=args.prune_method, prune_r=args.prune_r,
            load_in_4bit=args.load_in_4bit, load_in_8bit=args.load_in_8bit,
        )
        samples = [{"images": args.image, "prompt": args.prompt, "task": "adhoc"}]
        preds, _, _ = run_batch_inference(
            model, processor, samples, args.max_new_tokens
        )
        print(f"\n{'='*60}")
        print(f"  Prompt: {args.prompt}")
        print(f"  Images: {args.image}")
        print(f"  Pruning: {args.prune_method or 'none'} (R={args.prune_r})")
        print(f"{'='*60}")
        print(f"  Response: {preds[0]}")
        return

    if not args.data_path:
        print("ERROR: Specify --data_path for dataset eval, or --image + --prompt for ad-hoc inference.")
        return

    # Auto-detect dataset type from path
    ds_type = args.dataset
    if ds_type is None:
        for k in ["vrsbench", "VRSBench", "mme", "MME", "xlrs", "XLRS", "levircc", "LEVIR"]:
            if k.lower() in args.data_path.lower():
                ds_type = {"vrsbench": "vrsbench", "mme": "mme", "xlrs": "xlrs", "levircc": "levircc"}[k.lower()]
                break
        if ds_type is None:
            ds_type = "vrsbench"
    print(f"Dataset type: {ds_type}")

    # Load data — start_offset applies during loading, max_samples after
    loaders = {"vrsbench": load_vrsbench, "mme": load_mcq, "xlrs": load_mcq, "levircc": load_levircc}
    samples = loaders[ds_type](args.data_path, max_samples=0)  # load all
    if args.start_offset > 0:
        samples = samples[args.start_offset:]
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    if not samples:
        print("ERROR: No samples loaded. Check --start_offset and --max_samples.")
        return
    print(f"Loaded {len(samples)} samples")

    # Load model
    print("Loading model...")
    model, processor = load_model(
        args.model_path, args.device,
        prune_method=args.prune_method, prune_r=args.prune_r,
        load_in_4bit=args.load_in_4bit, load_in_8bit=args.load_in_8bit,
    )
    if args.device == "cuda":
        print(f"GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GiB")

    # Inference in batches
    predictions = []
    img_token_counts = []
    t0 = time.time()
    batch_size = args.batch_size

    for bi in range(0, len(samples), batch_size):
        batch = samples[bi:bi + batch_size]
        preds, n_img, n_total = run_batch_inference(
            model, processor, batch, args.max_new_tokens
        )
        predictions.extend(preds)
        img_token_counts.append(n_img)

        done = min(bi + batch_size, len(samples))
        elapsed = time.time() - t0
        eta = elapsed / done * (len(samples) - done) if done > 0 else 0
        if bi % (batch_size * 10) == 0 or bi == 0:
            print(f"  [{done}/{len(samples)}] {elapsed/done:.2f}s/samp  ETA={eta/3600:.1f}h  img={n_img//len(batch)}  pred={preds[0][:60]}", flush=True)
        if bi % (batch_size * 50) == 0:
            torch.cuda.empty_cache()
            if args.output:
                ckpt = args.output.replace(".json", ".ckpt.json")
                results_snapshot = _compute_results(samples[:done], predictions, ds_type)
                with open(ckpt, "w") as f:
                    json.dump(results_snapshot, f, indent=2)

    elapsed = time.time() - t0
    if len(samples) > 0:
        print(f"Total: {elapsed:.1f}s, avg {elapsed/len(samples):.2f}s/sample")
    if img_token_counts:
        avg_img = np.mean([x for x in img_token_counts if x > 0])
        print(f"Image tokens: avg {avg_img:.0f} per batch")
        if args.prune_method and args.prune_method != "none":
            print(f"  (pruned from {avg_img:.0f} → ~{avg_img*(1-args.prune_r):.0f} per batch)")

    results = _compute_results(samples, predictions, ds_type)

    # Print
    print(f"\n{'='*60}")
    print(f"  Results ({ds_type})")
    print(f"{'='*60}")
    for task, metrics in results.items():
        n = metrics.pop("samples")
        parts = [f"  [{task}] samples={n}"]
        for k, v in sorted(metrics.items()):
            parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
        print(" | ".join(parts))

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
