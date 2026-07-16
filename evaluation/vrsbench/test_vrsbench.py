#!/usr/bin/env python3
"""
VRSBench 评估脚本：支持 referring (IoU)、captioning (NLG)、VQA (accuracy) 三种任务。

用法:
    python test_vrsbench.py --model_path /path/to/model --test_data_dir /path/to/VRSBench

依赖:
    pip install nltk rouge_score
    python -m nltk.downloader wordnet punkt punkt_tab
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from math import log, sqrt
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# ============================================================
# 指标计算函数
# ============================================================

def parse_bbox(text: str) -> Optional[Tuple[float, float, float, float]]:
    """从文本中解析 {<x1><y1><x2><y2>} 格式的边界框坐标。"""
    pattern = r"\{<\s*(\d+(?:\.\d+)?)\s*><\s*(\d+(?:\.\d+)?)\s*><\s*(\d+(?:\.\d+)?)\s*><\s*(\d+(?:\.\d+)?)\s*>\}"
    match = re.search(pattern, text)
    if match:
        return tuple(map(float, match.groups()))
    return None


def compute_iou(box1: Tuple[float, ...], box2: Tuple[float, ...]) -> float:
    """计算两个边界框的 IoU。"""
    x1_inter = max(min(box1[0], box1[2]), min(box2[0], box2[2]))
    y1_inter = max(min(box1[1], box1[3]), min(box2[1], box2[3]))
    x2_inter = min(max(box1[0], box1[2]), max(box2[0], box2[2]))
    y2_inter = min(max(box1[1], box1[3]), max(box2[1], box2[3]))
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    area1 = abs(box1[2] - box1[0]) * abs(box1[3] - box1[1])
    area2 = abs(box2[2] - box2[0]) * abs(box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def compute_referring_metrics(references: List[str], predictions: List[str]) -> Dict[str, float]:
    """计算 referring 任务指标：mean_iou, Acc@0.25, Acc@0.5, Acc@0.7。"""
    ious = []
    for ref_text, pred_text in tqdm(zip(references, predictions),
                                     total=len(references),
                                     desc="  计算 IoU",
                                     file=sys.stdout):
        ref_box = parse_bbox(ref_text)
        pred_box = parse_bbox(pred_text)
        if ref_box is not None and pred_box is not None:
            ious.append(compute_iou(ref_box, pred_box))
        else:
            ious.append(0.0)  # 解析失败的样本视为 IoU=0，统一分母

    if not ious:
        return {"mean_iou": 0.0, "Acc@0.25": 0.0, "Acc@0.5": 0.0, "Acc@0.7": 0.0}

    total = len(ious)
    results = {"mean_iou": float(np.mean(ious))}
    for t in [0.25, 0.5, 0.7]:
        results[f"Acc@{t}"] = sum(1 for iou in ious if iou >= t) / total
    return results


def compute_bleu(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """计算 corpus-level BLEU-1/2/3/4。"""
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

    def tokenize(text: str) -> List[str]:
        text = re.sub(r"[^\w\s]", " ", text.lower())
        return text.split()

    list_of_refs = [[tokenize(ref)] for ref in references]
    list_of_hyps = [tokenize(hyp) for hyp in hypotheses]
    smooth = SmoothingFunction().method1
    scores = {}
    for n in [1, 2, 3, 4]:
        weights = tuple(1.0 / n if i < n else 0.0 for i in range(4))
        try:
            score = corpus_bleu(list_of_refs, list_of_hyps, weights=weights, smoothing_function=smooth)
            if score > 1.0:
                score = score / 100.0
            scores[f"BLEU-{n}"] = score
        except Exception:
            scores[f"BLEU-{n}"] = 0.0
    return scores


def compute_rouge_l(references: List[str], hypotheses: List[str]) -> float:
    """计算 ROUGE-L 分数。"""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = []
    for ref, hyp in tqdm(zip(references, hypotheses), total=len(references),
                         desc="  计算 ROUGE-L", file=sys.stdout):
        scores.append(scorer.score(ref, hyp)["rougeL"].fmeasure)
    return float(np.mean(scores)) if scores else 0.0


def compute_meteor(references: List[str], hypotheses: List[str]) -> float:
    """计算 METEOR 分数。"""
    from nltk.translate.meteor_score import meteor_score

    scores = []
    for ref, hyp in tqdm(zip(references, hypotheses), total=len(references),
                         desc="  计算 METEOR", file=sys.stdout):
        if not hyp.strip() or not ref.strip():
            scores.append(0.0)
            continue
        try:
            scores.append(meteor_score([ref.lower().split()], hyp.lower().split()))
        except Exception:
            scores.append(0.0)
    return float(np.mean(scores)) if scores else 0.0


def compute_cider_d(references: List[str], hypotheses: List[str], n: int = 4) -> float:
    """计算 CIDEr-D 分数。"""
    def tokenize(text: str) -> List[str]:
        text = re.sub(r"[^\w\s]", " ", text.lower())
        return text.split()

    def extract_ngrams(tokens: List[str], ngram_n: int) -> Dict[Tuple[str, ...], int]:
        ngrams = {}
        for i in range(len(tokens) - ngram_n + 1):
            gram = tuple(tokens[i:i + ngram_n])
            ngrams[gram] = ngrams.get(gram, 0) + 1
        return ngrams

    def compute_document_frequency(corpus: List[List[str]], ngram_n: int):
        df = defaultdict(int)
        for tokens in corpus:
            for gram in set(tuple(tokens[i:i + ngram_n]) for i in range(len(tokens) - ngram_n + 1)):
                df[gram] += 1
        return df

    def compute_tfidf_vector(tokens: List[str], ngram_n: int, df: Dict, N: int):
        ngrams = extract_ngrams(tokens, ngram_n)
        max_count = max(ngrams.values()) if ngrams else 1
        return {gram: (count / max_count) * (log((N + 1) / (df.get(gram, 0) + 1)) + 1)
                for gram, count in ngrams.items()}

    def cosine_similarity(vec1: Dict, vec2: Dict) -> float:
        if not vec1 or not vec2:
            return 0.0
        dot = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in set(vec1) | set(vec2))
        norm1 = sqrt(sum(v ** 2 for v in vec1.values()))
        norm2 = sqrt(sum(v ** 2 for v in vec2.values()))
        return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

    all_ref_tokens = [tokenize(ref) for ref in references]
    all_hyp_tokens = [tokenize(hyp) for hyp in hypotheses]
    N = len(references)

    cider_scores = []
    for ngram_n in range(1, n + 1):
        df = compute_document_frequency(all_ref_tokens, ngram_n)
        n_scores = []
        for hyp_tokens, ref_tokens in tqdm(zip(all_hyp_tokens, all_ref_tokens),
                                            total=N, desc=f"  计算 CIDEr-D {ngram_n}-gram",
                                            file=sys.stdout):
            hyp_vec = compute_tfidf_vector(hyp_tokens, ngram_n, df, N)
            ref_vec = compute_tfidf_vector(ref_tokens, ngram_n, df, N)
            n_scores.append(cosine_similarity(hyp_vec, ref_vec))
        cider_scores.append(float(np.mean(n_scores)) if n_scores else 0.0)

    weights = [1.0 / n] * n
    cider = sum(w * s for w, s in zip(weights, cider_scores))

    ref_lens = [len(t) for t in all_ref_tokens]
    hyp_lens = [len(t) for t in all_hyp_tokens]
    avg_ref_len = float(np.mean(ref_lens)) if ref_lens else 1.0
    sigma = float(np.std(ref_lens)) if len(ref_lens) > 1 else avg_ref_len / 6.0
    diff = abs(avg_ref_len - float(np.mean(hyp_lens)) if hyp_lens else 0.0)
    gaussian_penalty = np.exp(-(diff ** 2) / (2 * sigma ** 2)) if sigma > 0 else 1.0

    return float(max(cider * gaussian_penalty * 10.0, 0.0))


def compute_caption_metrics(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """计算 caption 任务的全部 NLG 指标。"""
    results = {}
    print("  计算 BLEU...")
    try:
        results.update(compute_bleu(references, hypotheses))
    except Exception:
        for n in [1, 2, 3, 4]:
            results[f"BLEU-{n}"] = 0.0
    print("  计算 ROUGE-L...")
    try:
        results["ROUGE-L"] = compute_rouge_l(references, hypotheses)
    except Exception:
        results["ROUGE-L"] = 0.0
    print("  计算 METEOR...")
    try:
        results["METEOR"] = compute_meteor(references, hypotheses)
    except Exception:
        results["METEOR"] = 0.0
    print("  计算 CIDEr-D...")
    try:
        results["CIDEr-D"] = compute_cider_d(references, hypotheses)
    except Exception:
        results["CIDEr-D"] = 0.0
    return results


def compute_accuracy(references: List[str], predictions: List[str]) -> float:
    """计算精确匹配准确率（用于 VQA 任务）。"""
    if not references:
        return 0.0
    correct = sum(1 for ref, pred in zip(references, predictions)
                  if ref.strip().lower() == pred.strip().lower())
    return correct / len(references)


# ============================================================
# Dataset 定义
# ============================================================

class VRSBenchDataset(Dataset):
    """VRSBench 评估数据集，支持 caption / vqa / referring 三种任务类型。"""

    def __init__(self, data: List[Dict], image_dir: str):
        self.data = data
        self.image_dir = image_dir

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        image_path = os.path.join(self.image_dir, item["image_id"])
        return {
            "image_path": image_path,
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "task_type": item.get("type", "caption"),
            "index": idx,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """整理 batch，保持 list 形式便于逐条推理。"""
    return {
        "image_path": [b["image_path"] for b in batch],
        "question": [b["question"] for b in batch],
        "ground_truth": [b["ground_truth"] for b in batch],
        "task_type": [b["task_type"] for b in batch],
        "index": [b["index"] for b in batch],
    }


# ============================================================
# 模型加载与推理
# ============================================================

def load_model(model_path: str, device: str = "cuda"):
    """加载 Qwen3-VL 模型和 processor。"""
    print(f"加载模型: {model_path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor


def run_inference_batch(model, processor, batch: Dict, device: str = "cuda",
                        max_new_tokens: int = 256) -> List[str]:
    """对一个 batch 的数据运行推理，返回生成的文本列表。"""
    predictions = []
    batch_size = len(batch["image_path"])
    for i, (image_path, question) in enumerate(zip(batch["image_path"], batch["question"])):
        if not os.path.exists(image_path):
            predictions.append("")
            continue
        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question},
                ],
            }]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                               do_sample=False, temperature=None, top_p=None)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )[0]
            predictions.append(output_text.strip())
        except Exception:
            predictions.append("")
    return predictions


# ============================================================
# 主评估流程
# ============================================================

TASK_CONFIG = {
    "referring": {
        "file": "VRSBench_EVAL_referring.json",
        "label": "REFERRING",
        "metrics_fn": "referring",
    },
    "caption": {
        "file": "VRSBench_EVAL_Cap.json",
        "label": "CAPTION",
        "metrics_fn": "caption",
    },
    "vqa": {
        "file": "VRSBench_EVAL_vqa.json",
        "label": "VQA",
        "metrics_fn": "vqa",
    },
}

PROMPT_TEMPLATES = {
    "caption": "<image>\n[caption] Could you describe the contents of this image for me?",
    "vqa": "<image>\n[vqa] {question}",
    "referring": "<image>\n[refer] could you tell me the location for <p>{question}</p>?",
}


def evaluate_task(model, processor, task_key: str, test_data_dir: str,
                  device: str, batch_size: int, max_samples: int,
                  use_prompt_template: bool = False) -> Dict:
    """评估单个任务，返回指标字典。"""
    config = TASK_CONFIG[task_key]
    eval_file = os.path.join(test_data_dir, config["file"])
    if not os.path.exists(eval_file):
        print(f"  评估文件不存在: {eval_file}")
        return {}

    data = json.load(open(eval_file, "r", encoding="utf-8"))
    if max_samples > 0 and max_samples < len(data):
        indices = np.linspace(0, len(data) - 1, max_samples, dtype=int)
        data = [data[i] for i in indices]

    # 应用 prompt 模板
    if use_prompt_template:
        template = PROMPT_TEMPLATES[task_key]
        for item in data:
            item["question"] = template.format(question=item["question"])
        print(f"  已应用 {task_key} prompt 模板: {template[:60]}...")

    # VRSBench eval 图片目录
    image_dir = os.path.join(test_data_dir, "images", "val")
    if not os.path.exists(image_dir):
        image_dir = os.path.join(test_data_dir, "Images_val")

    dataset = VRSBenchDataset(data, image_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    all_predictions = []
    all_references = []

    print(f"  推理中 ({len(dataset)} 样本, batch_size={batch_size})...")
    t_infer_start = time.time()
    for batch in tqdm(dataloader, desc=f"  {config['label']}", file=sys.stdout):
        preds = run_inference_batch(model, processor, batch, device=device)
        all_predictions.extend(preds)
        all_references.extend(batch["ground_truth"])
    t_infer = time.time() - t_infer_start
    print(f"  推理完成，耗时 {t_infer:.1f}s ({t_infer/len(dataset):.2f}s/样本)")

    # 计算指标
    print(f"  计算指标...")
    t_metric_start = time.time()
    if task_key == "referring":
        metrics = compute_referring_metrics(all_references, all_predictions)
    elif task_key == "caption":
        valid = [(r, p) for r, p in zip(all_references, all_predictions) if p.strip()]
        if valid:
            valid_refs, valid_preds = zip(*valid)
            metrics = compute_caption_metrics(list(valid_refs), list(valid_preds))
        else:
            metrics = {k: 0.0 for k in ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L", "METEOR", "CIDEr-D"]}
    else:  # vqa
        metrics = {"Accuracy": compute_accuracy(all_references, all_predictions)}
    t_metric = time.time() - t_metric_start
    print(f"  指标计算完成，耗时 {t_metric:.1f}s")

    metrics["samples"] = len(all_predictions)
    return metrics


def print_results_table(all_metrics: Dict[str, Dict]):
    """以表格形式输出所有任务的评估结果。"""
    print("\n" + "=" * 80)
    print("  VRSBench 评估结果")
    print("=" * 80)

    # 收集所有指标名
    all_keys = []
    for task_metrics in all_metrics.values():
        for k in task_metrics:
            if k != "samples" and k not in all_keys:
                all_keys.append(k)

    # 确定每列宽度
    col_widths = {"task": 14}
    for k in all_keys:
        col_widths[k] = max(len(k), 10)
    total_width = sum(col_widths.values()) + len(col_widths) + 1

    # 分隔线
    sep = "+" + "+".join("-" * w for w in col_widths.values()) + "+"

    # 表头
    print(sep)
    header = "|" + "|".join(k.center(col_widths[k]) for k in ["task"] + all_keys) + "|"
    print(header)
    print(sep.replace("-", "="))

    # 数据行
    task_order = ["referring", "caption", "vqa"]
    for task_key in task_order:
        if task_key not in all_metrics:
            continue
        metrics = all_metrics[task_key]
        label = TASK_CONFIG[task_key]["label"]
        row = f"|{label:<{col_widths['task']}}|"
        for k in all_keys:
            val = metrics.get(k, "-")
            if isinstance(val, float):
                row += f"{val:^{col_widths[k]}.4f}|"
            else:
                row += f"{str(val):^{col_widths[k]}}|"
        print(row)

    print(sep)

    # 样本数汇总
    sample_info = ", ".join(
        f"{TASK_CONFIG[t]['label']}: {all_metrics[t].get('samples', 0)}"
        for t in task_order if t in all_metrics
    )
    print(f"  样本数: {sample_info}")
    print()


def main():
    parser = argparse.ArgumentParser(description="VRSBench 评估")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--test_data_dir", type=str, required=True, help="VRSBench 数据集根目录")
    parser.add_argument("--tasks", type=str, default="all",
                        choices=["referring", "caption", "vqa", "all"],
                        help="评估任务 (默认: all)")
    parser.add_argument("--batch_size", type=int, default=1, help="推理 batch size (默认: 1)")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="每任务最大样本数 (0=全部)")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备")
    parser.add_argument("--output", type=str, default=None,
                        help="结果输出 JSON 路径 (默认: 脚本同目录下 vrsbench_results.json)")
    parser.add_argument("--use_prompt_template", action="store_true", default=False,
                        help="使用训练时的 prompt 模板包装问题 (caption/vqa/referring)")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，切换到 CPU")
        args.device = "cpu"

    tasks = ["referring", "caption", "vqa"] if args.tasks == "all" else [args.tasks]

    print("=" * 80)
    print("  VRSBench 评估")
    print("=" * 80)
    print(f"  模型:     {args.model_path}")
    print(f"  数据:     {args.test_data_dir}")
    print(f"  任务:     {', '.join(tasks)}")
    print(f"  设备:     {args.device}")
    print(f"  Prompt模板: {'是' if args.use_prompt_template else '否'}")
    print()

    model, processor = load_model(args.model_path, device=args.device)

    all_metrics = {}
    for task_key in tasks:
        print(f"\n--- {TASK_CONFIG[task_key]['label']} ---")
        all_metrics[task_key] = evaluate_task(
            model, processor, task_key, args.test_data_dir,
            args.device, args.batch_size, args.max_samples,
            use_prompt_template=args.use_prompt_template,
        )

    del model, processor
    torch.cuda.empty_cache()

    print_results_table(all_metrics)

    # 保存结果到 JSON
    output_path = args.output or os.path.join(os.path.dirname(__file__), "vrsbench_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"结果已保存至 {output_path}")


if __name__ == "__main__":
    main()
