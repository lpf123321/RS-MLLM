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
               load_in_4bit=False, load_in_8bit=False, lora_path=None):
    """加载模型，可选启用剪枝、量化和 LoRA adapter。"""
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

    # 加载 LoRA adapter（可选）
    if lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        print(f"LoRA adapter loaded: {lora_path}")

    model.eval()

    if prune_method and prune_method != "none":
        enable_pruning(model, method=prune_method, r=prune_r)

    return model, processor


def run_batch_inference(model, processor, samples_batch, max_new_tokens=256):
    """批量推理。预填空 think 块抑制 CoT，system prompt 控制输出格式。"""
    texts = []
    all_images = []
    for s in samples_batch:
        messages = []
        if s.get("system_prompt"):
            messages.append({"role": "system", "content": s["system_prompt"]})
        messages.append({"role": "user", "content": [
            *[{"type": "image", "image": img} for img in s["images"]],
            {"type": "text", "text": s["prompt"]},
        ]})
        # 参考 inference.py 的 build_prompt 格式
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        text += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        texts.append(text)
        # 单独提取图像
        img_msg = [{"role": "user", "content": [{"type": "image", "image": img} for img in s["images"]]}]
        image_inputs, _ = process_vision_info(img_msg)
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
        # 和 inference.py 一致的提取：</think> 之后的内容
        out = out.split("</think>", 1)[1].strip() if "</think>" in out else out.strip()
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


def load_mcq(data_path, dataset_type, max_samples=0):
    """MME / XLRS: MCQ，单图。"""
    prompts = {
        "mme": 'Answer EXACTLY in format "X. (X) FullOptionText" with the letter repeated in parentheses. Example: "D. (D) White". You MUST include the parenthesized letter - never omit it. Output ONLY that line.',
        "xlrs": 'Answer EXACTLY in format "X. (X) FullOptionText" with the letter repeated in parentheses. Example: "A. (A) Some description". You MUST include the parenthesized letter - never omit it. Output ONLY that line.',
    }
    sp = prompts.get(dataset_type, "")
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
                "reference": assistant, "system_prompt": sp,
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
            refs_raw = d.get("references", [d["messages"][1]["content"][0]["text"]])
            # 提取 "raw" 字段（reference 可能是 dict 或 string）
            refs = [r["raw"] if isinstance(r, dict) else r for r in refs_raw]
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
    """MCQ letter-only accuracy: extract answer letter from patterns like 'D.', '(D)', 'D)'."""
    def _extract_letter(text):
        # A. (A) / D) / (D) / 或裸字母 A-D
        m = re.search(r'(?<!\w)([A-Da-d])\s*[.)]', str(text).strip())
        if m:
            return m.group(1).upper()
        # fallback: standalone A-D letter
        m = re.search(r'(?<!\w)([A-Da-d])(?!\w)', str(text).strip())
        return m.group(1).upper() if m else ""
    correct = 0
    for r, p in zip(references, predictions):
        rl = _extract_letter(r)
        pl = _extract_letter(p)
        if rl and pl and rl == pl:
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
        weights = tuple(1.0 / n if i < n else 0.0 for i in range(max_n))
        try:
            bleu = corpus_bleu(list_of_refs, hyps, weights=weights, smoothing_function=smooth)
            if bleu > 1.0:
                bleu = bleu / 100.0
            bleus[f"BLEU-{n}"] = bleu
        except Exception:
            bleus[f"BLEU-{n}"] = 0.0
    return bleus


def compute_rouge_l(references, predictions):
    """ROUGE-L，取多参考中最高分。"""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        scores = []
        for ref, pred in zip(references, predictions):
            refs = ref if isinstance(ref, list) else [ref]
            best = max(scorer.score(str(r), str(pred))["rougeL"].fmeasure for r in refs)
            scores.append(best)
        return np.mean(scores) if scores else 0.0
    except ImportError:
        return 0.0


def compute_cider(references, predictions):
    """CIDEr 指标。"""
    try:
        from collections import defaultdict
        from math import log, sqrt
        all_ref_tokens = [[_tokenize(ref) for ref in (r if isinstance(r, list) else [r])] for r in references]
        all_hyp_tokens = [_tokenize(hyp) for hyp in predictions]
        N = len(references)
        corpus_refs = [t for group in all_ref_tokens for t in group]
        cider_scores = []

        for ngram_n in range(1, 5):
            df = defaultdict(int)
            for tokens in corpus_refs:
                seen = set()
                for i in range(len(tokens) - ngram_n + 1):
                    g = tuple(tokens[i:i + ngram_n])
                    if g not in seen: df[g] += 1; seen.add(g)

            n_scores = []
            for hyp_tokens, ref_groups in zip(all_hyp_tokens, all_ref_tokens):
                if not hyp_tokens: continue
                hyp_ng = {}
                for i in range(len(hyp_tokens) - ngram_n + 1):
                    g = tuple(hyp_tokens[i:i + ngram_n])
                    hyp_ng[g] = hyp_ng.get(g, 0) + 1
                hyp_max = max(hyp_ng.values()) if hyp_ng else 1
                hyp_vec = {g: (c / hyp_max) * (log((N + 1) / (df.get(g, 0) + 1)) + 1) for g, c in hyp_ng.items()}
                ref_avg = []
                for ref_tokens in ref_groups:
                    if not ref_tokens: continue
                    ref_ng = {}
                    for i in range(len(ref_tokens) - ngram_n + 1):
                        g = tuple(ref_tokens[i:i + ngram_n])
                        ref_ng[g] = ref_ng.get(g, 0) + 1
                    ref_max = max(ref_ng.values()) if ref_ng else 1
                    ref_vec = {g: (c / ref_max) * (log((N + 1) / (df.get(g, 0) + 1)) + 1) for g, c in ref_ng.items()}
                    dot = sum(hyp_vec.get(k, 0) * ref_vec.get(k, 0) for k in set(hyp_vec) | set(ref_vec))
                    n1 = sqrt(sum(v ** 2 for v in hyp_vec.values()))
                    n2 = sqrt(sum(v ** 2 for v in ref_vec.values()))
                    ref_avg.append(dot / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0)
                n_scores.append(np.mean(ref_avg) if ref_avg else 0.0)
            cider_scores.append(float(np.mean(n_scores)) if n_scores else 0.0)

        weights = [0.25, 0.25, 0.25, 0.25]
        cider = sum(w * s for w, s in zip(weights, cider_scores))
        # Length penalty
        ref_lens = [len(t) for group in all_ref_tokens for t in group]
        hyp_lens = [len(t) for t in all_hyp_tokens]
        avg_ref_len = float(np.mean(ref_lens)) if ref_lens else 1.0
        avg_hyp_len = float(np.mean(hyp_lens)) if hyp_lens else 0.0
        sigma_val = float(np.std(ref_lens)) if len(ref_lens) > 1 else avg_ref_len / 6.0
        diff = abs(avg_ref_len - avg_hyp_len)
        penalty = np.exp(-(diff ** 2) / (2 * sigma_val ** 2)) if sigma_val > 0 else 1.0
        return float(max(cider * penalty * 10.0, 0.0))
    except Exception:
        return 0.0


def compute_referring_acc(references, predictions, threshold=0.5):
    """Referring expression IoU accuracy，与 evaluation/metrics/referring.py 一致。"""
    _BBOX_ANGLE_RE = re.compile(
        r"\{<\s*(\d+(?:\.\d+)?)\s*><\s*(\d+(?:\.\d+)?)\s*><\s*(\d+(?:\.\d+)?)\s*><\s*(\d+(?:\.\d+)?)\s*>\}"
    )
    _BBOX_COMMA_RE = re.compile(
        r"\{\s*(\d+(?:\.\d+)?)\s*[,;\s]+\s*(\d+(?:\.\d+)?)\s*[,;\s]+\s*(\d+(?:\.\d+)?)\s*[,;\s]+\s*(\d+(?:\.\d+)?)\s*\}"
    )
    def _parse_bbox(text):
        text = _BBOX_COMMA_RE.sub(r"{<\1><\2><\3><\4>}", text)
        m = _BBOX_ANGLE_RE.search(text)
        return tuple(map(float, m.groups())) if m else None
    def _iou(b1, b2):
        x1 = max(min(b1[0], b1[2]), min(b2[0], b2[2]))
        y1 = max(min(b1[1], b1[3]), min(b2[1], b2[3]))
        x2 = min(max(b1[0], b1[2]), max(b2[0], b2[2]))
        y2 = min(max(b1[1], b1[3]), max(b2[1], b2[3]))
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = abs(b1[2] - b1[0]) * abs(b1[3] - b1[1])
        a2 = abs(b2[2] - b2[0]) * abs(b2[3] - b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0
    ious = []
    for r, p in zip(references, predictions):
        refs = r if isinstance(r, list) else [r]
        ref_box = _parse_bbox(refs[0]) if refs else None
        pred_box = _parse_bbox(p)
        ious.append(_iou(ref_box, pred_box) if ref_box and pred_box else 0.0)
    if not ious:
        return 0.0
    return sum(1 for v in ious if v >= threshold) / len(ious)


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
            task_result["CIDEr"] = compute_cider(refs, preds)
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
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Max tokens to generate")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Use 4-bit quantization")
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="Use 8-bit quantization")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA adapter weights")
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
    if ds_type in ("mme", "xlrs"):
        samples = load_mcq(args.data_path, ds_type, max_samples=0)
    else:
        loader = {"vrsbench": load_vrsbench, "levircc": load_levircc}[ds_type]
        samples = loader(args.data_path, max_samples=0)
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
        lora_path=args.lora_path,
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
        # Save checkpoint every 100 samples (frequent for long runs)
        if bi > 0 and done % 100 == 0:
            torch.cuda.empty_cache()
            if args.output:
                ckpt = args.output.replace(".json", ".ckpt.json")
                results_snapshot = _compute_results(samples[:done], predictions, ds_type)
                with open(ckpt, "w") as f:
                    json.dump(results_snapshot, f, indent=2)
        elif done % 200 == 0:
            torch.cuda.empty_cache()

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
