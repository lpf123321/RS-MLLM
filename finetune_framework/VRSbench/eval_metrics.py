#!/usr/bin/env python3
"""
VRSBench 评估脚本：对比 Qwen3-VL-2B-Instruct 在 LoRA 微调前后的 NLG 指标。

指标：BLEU-1/2/3/4, ROUGE-L, METEOR, CIDEr-D

用法:
    # 仅评估 caption（NLG 指标最有意义的任务）
    python eval_metrics.py --task caption --max_samples 500

    # 评估全部三个任务，每任务采样 500 条
    python eval_metrics.py --task all --max_samples 500

    # 仅评估基座模型
    python eval_metrics.py --task caption --base_only --max_samples 500

    # 仅评估微调模型
    python eval_metrics.py --task caption --ft_only --max_samples 500

    # 完整评估（全部 caption 数据，耗时长）
    python eval_metrics.py --task caption

依赖:
    pip install nltk rouge_score pycocoevalcap
    python -m nltk.downloader wordnet punkt punkt_tab
"""

import argparse
import json
import os
import sys
import time
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np

import torch
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

# ============================================================
# 路径配置
# ============================================================
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_MODEL_PATH = os.path.join(REPO_ROOT, "models", "Qwen3-VL-2B-Instruct")
LORA_ADAPTER_PATH = os.path.join(
    REPO_ROOT, "finetune_framework", "VRSbench", "output", "finetune_test"
)
IMAGE_FOLDER = os.path.join(REPO_ROOT, "datasets", "shared_datasets", "VRSBench", "images", "val")
EVAL_DIR = os.path.join(REPO_ROOT, "datasets", "shared_datasets", "VRSBench")

EVAL_FILES = {
    "caption": os.path.join(EVAL_DIR, "VRSBench_EVAL_Cap.json"),
    "vqa": os.path.join(EVAL_DIR, "VRSBench_EVAL_vqa.json"),
    "referring": os.path.join(EVAL_DIR, "VRSBench_EVAL_referring.json"),
}

# 训练时使用的 prompt 前缀（微调模型对此格式敏感）
TRAINING_PROMPTS = {
    "caption": "<image>\n[caption] Could you describe the contents of this image for me?",
    "vqa": "<image>\n[vqa] {question}",
    "referring": "<image>\n[refer] could you tell me the location for <p>{question}</p>?",
}


# ============================================================
# 指标计算
# ============================================================

def parse_bbox(text: str) -> Optional[Tuple[float, float, float, float]]:
    """
    解析 {<x1><y1><x2><y2>} 格式的边界框坐标。

    Args:
        text: 模型输出文本，如 "{<25><40><33><60>}"

    Returns:
        (x1, y1, x2, y2) 归一化坐标 (0-100)，解析失败返回 None
    """
    # 匹配 {<数字><数字><数字><数字>} 模式
    pattern = r"\{<\s*(\d+(?:\.\d+)?)\s*><\s*(\d+(?:\.\d+)?)\s*><\s*(\d+(?:\.\d+)?)\s*><\s*(\d+(?:\.\d+)?)\s*>\}"
    match = re.search(pattern, text)
    if match:
        x1, y1, x2, y2 = map(float, match.groups())
        return (x1, y1, x2, y2)
    return None


def compute_iou(box1: Tuple[float, ...], box2: Tuple[float, ...]) -> float:
    """
    计算两个边界框的 IoU (Intersection over Union)。

    Args:
        box1, box2: (x1, y1, x2, y2) 格式的坐标

    Returns:
        IoU ∈ [0, 1]
    """
    # 确保 x1 < x2, y1 < y2
    x1_inter = max(min(box1[0], box1[2]), min(box2[0], box2[2]))
    y1_inter = max(min(box1[1], box1[3]), min(box2[1], box2[3]))
    x2_inter = min(max(box1[0], box1[2]), max(box2[0], box2[2]))
    y2_inter = min(max(box1[1], box1[3]), max(box2[1], box2[3]))

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    area1 = abs(box1[2] - box1[0]) * abs(box1[3] - box1[1])
    area2 = abs(box2[2] - box2[0]) * abs(box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def compute_referring_metrics(
    references: List[str], predictions: List[str]
) -> Dict[str, float]:
    """
    计算 referring 任务的评估指标：IoU 均值 + Accuracy@τ。

    Args:
        references: ground truth 文本列表，格式 "{<x1><y1><x2><y2>}"
        predictions: 模型预测文本列表

    Returns:
        包含 mean_iou, Acc@0.25, Acc@0.5, Acc@0.7, valid_samples 的字典
    """
    ious = []
    valid_count = 0
    for ref_text, pred_text in zip(references, predictions):
        ref_box = parse_bbox(ref_text)
        pred_box = parse_bbox(pred_text)
        if ref_box is None or pred_box is None:
            continue
        iou = compute_iou(ref_box, pred_box)
        ious.append(iou)
        valid_count += 1

    if not ious:
        return {
            "mean_iou": 0.0,
            "Acc@0.25": 0.0,
            "Acc@0.5": 0.0,
            "Acc@0.7": 0.0,
            "valid_boxes": 0,
        }

    # IoU 阈值对应的准确率
    thresholds = [0.25, 0.5, 0.7]
    acc = {}
    for t in thresholds:
        correct = sum(1 for iou in ious if iou >= t)
        acc[f"Acc@{t}"] = correct / len(predictions) if predictions else 0.0

    acc["mean_iou"] = float(np.mean(ious))
    acc["valid_boxes"] = valid_count
    return acc


def compute_bleu(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """计算 BLEU-1/2/3/4 分数（corpus-level，使用平滑）。"""
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    # tokenize：按空格和标点分词
    def tokenize(text: str) -> List[str]:
        text = re.sub(r"[^\w\s]", " ", text.lower())
        return text.split()

    list_of_references = [[tokenize(ref)] for ref in references]
    list_of_hypotheses = [tokenize(hyp) for hyp in hypotheses]

    smooth = SmoothingFunction().method1
    scores = {}
    # 检查样本数是否足够计算各阶 BLEU
    min_ref_len = min(len(ref[0]) for ref in list_of_references)
    for n in [1, 2, 3, 4]:
        if n <= min_ref_len:
            weights = tuple(1.0 / n if i < n else 0.0 for i in range(4))
            try:
                score = corpus_bleu(
                    list_of_references, list_of_hypotheses,
                    weights=weights, smoothing_function=smooth
                )
            except Exception:
                score = 0.0
            # corpus_bleu 返回百分比形式（0-1 或 0-100 取决于版本）
            if score > 1.0:
                score = score / 100.0
            scores[f"BLEU-{n}"] = score
        else:
            scores[f"BLEU-{n}"] = 0.0
    return scores


def compute_rouge_l(references: List[str], hypotheses: List[str]) -> float:
    """计算 ROUGE-L 分数（基于最长公共子序列）。"""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = []
    for ref, hyp in zip(references, hypotheses):
        score = scorer.score(ref, hyp)
        scores.append(score["rougeL"].fmeasure)
    return float(np.mean(scores)) if scores else 0.0


def compute_meteor(references: List[str], hypotheses: List[str]) -> float:
    """计算 METEOR 分数。"""
    from nltk.translate.meteor_score import meteor_score

    # tokenize by whitespace
    scores = []
    for ref, hyp in zip(references, hypotheses):
        if not hyp.strip() or not ref.strip():
            scores.append(0.0)
            continue
        try:
            # meteor_score expects lists of tokens
            ref_tokens = ref.lower().split()
            hyp_tokens = hyp.lower().split()
            score = meteor_score([ref_tokens], hyp_tokens)
            scores.append(score)
        except Exception:
            scores.append(0.0)
    return float(np.mean(scores)) if scores else 0.0


def compute_cider_d(
    references: List[str], hypotheses: List[str], n: int = 4
) -> float:
    """
    计算 CIDEr-D 分数。

    实现基于原始论文：
    CIDEr-D_n = cosine_similarity(g^n(c), g^n(s)) 带 TF-IDF 加权
    CIDEr-D = sum_n w_n * CIDEr_n * length_penalty
    """
    from math import log, sqrt

    def tokenize(text: str) -> List[str]:
        text = re.sub(r"[^\w\s]", " ", text.lower())
        return text.split()

    def extract_ngrams(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
        ngrams = {}
        for i in range(len(tokens) - n + 1):
            gram = tuple(tokens[i : i + n])
            ngrams[gram] = ngrams.get(gram, 0) + 1
        return ngrams

    def compute_document_frequency(corpus: List[List[str]], n: int):
        """计算每个 n-gram 在多少文档中出现。"""
        df = defaultdict(int)
        for tokens in corpus:
            unique_ngrams = set()
            for i in range(len(tokens) - n + 1):
                unique_ngrams.add(tuple(tokens[i : i + n]))
            for gram in unique_ngrams:
                df[gram] += 1
        return df

    def compute_tfidf_vector(
        tokens: List[str], n: int, df: Dict, N: int
    ) -> Dict[Tuple[str, ...], float]:
        """计算单文档的 TF-IDF n-gram 向量。"""
        ngrams = extract_ngrams(tokens, n)
        vector = {}
        for gram, count in ngrams.items():
            tf = count / max(ngrams[g] for g in ngrams) if ngrams else 0
            idf = log((N + 1) / (df.get(gram, 0) + 1)) + 1
            vector[gram] = tf * idf
        return vector

    def cosine_similarity(
        vec1: Dict, vec2: Dict
    ) -> float:
        """计算两个向量的余弦相似度。"""
        if not vec1 or not vec2:
            return 0.0
        dot = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in set(vec1) | set(vec2))
        norm1 = sqrt(sum(v**2 for v in vec1.values()))
        norm2 = sqrt(sum(v**2 for v in vec2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    # tokenize all
    all_ref_tokens = [tokenize(ref) for ref in references]
    all_hyp_tokens = [tokenize(hyp) for hyp in hypotheses]
    N = len(references)

    # 对每个 n 计算 CIDEr_n
    cider_scores = []
    for ngram_n in range(1, n + 1):
        df = compute_document_frequency(all_ref_tokens, ngram_n)
        n_scores = []
        for hyp_tokens, ref_tokens in zip(all_hyp_tokens, all_ref_tokens):
            hyp_vec = compute_tfidf_vector(hyp_tokens, ngram_n, df, N)
            ref_vec = compute_tfidf_vector(ref_tokens, ngram_n, df, N)
            sim = cosine_similarity(hyp_vec, ref_vec)
            n_scores.append(sim)
        cider_scores.append(float(np.mean(n_scores)) if n_scores else 0.0)

    # 统一权重
    weights = [1.0 / n] * n
    cider = sum(w * s for w, s in zip(weights, cider_scores))

    # 长度惩罚项（CIDEr-D）
    ref_lens = [len(t) for t in all_ref_tokens]
    hyp_lens = [len(t) for t in all_hyp_tokens]
    avg_ref_len = float(np.mean(ref_lens)) if ref_lens else 1.0
    avg_hyp_len = float(np.mean(hyp_lens)) if hyp_lens else 0.0

    # 标准差（用于高斯惩罚）
    if len(ref_lens) > 1:
        sigma = float(np.std(ref_lens))
    else:
        sigma = avg_ref_len / 6.0 if avg_ref_len > 0 else 1.0

    # 长度差异惩罚
    diff = abs(avg_ref_len - avg_hyp_len)
    if sigma > 0:
        gaussian_penalty = np.exp(-(diff**2) / (2 * sigma**2))
    else:
        gaussian_penalty = 1.0

    cider_d = cider * gaussian_penalty * 10.0  # ×10 使数值便于阅读
    return float(max(cider_d, 0.0))


def compute_all_metrics(
    references: List[str], hypotheses: List[str]
) -> Dict[str, float]:
    """计算全部四项指标。"""
    results = {}
    # BLEU
    try:
        bleu_scores = compute_bleu(references, hypotheses)
        results.update(bleu_scores)
    except Exception as e:
        print(f"  ⚠ BLEU 计算失败: {e}")
        for n in [1, 2, 3, 4]:
            results[f"BLEU-{n}"] = 0.0

    # ROUGE-L
    try:
        results["ROUGE-L"] = compute_rouge_l(references, hypotheses)
    except Exception as e:
        print(f"  ⚠ ROUGE-L 计算失败: {e}")
        results["ROUGE-L"] = 0.0

    # METEOR
    try:
        results["METEOR"] = compute_meteor(references, hypotheses)
    except Exception as e:
        print(f"  ⚠ METEOR 计算失败: {e}")
        results["METEOR"] = 0.0

    # CIDEr-D
    try:
        results["CIDEr-D"] = compute_cider_d(references, hypotheses)
    except Exception as e:
        print(f"  ⚠ CIDEr-D 计算失败: {e}")
        results["CIDEr-D"] = 0.0

    return results


# ============================================================
# 模型加载
# ============================================================

def load_model_and_processor(
    base_path: str,
    lora_path: Optional[str] = None,
    device: str = "cuda",
):
    """
    加载 Qwen3-VL 模型和 processor。
    如果提供 lora_path，则加载 LoRA adapter。
    """
    print(f"  加载基座模型: {base_path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    if lora_path and os.path.exists(lora_path):
        print(f"  加载 LoRA adapter: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        print("  LoRA 权重已合并")

    processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)
    return model, processor


# ============================================================
# 推理
# ============================================================

def build_messages(image_path: str, question: str, task_type: str) -> List[Dict]:
    """
    构建 Qwen3-VL 标准 messages 格式。

    对于微调模型，使用训练时的 prompt 格式；对于基座模型，使用自然语言 prompt。
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }
    ]


def run_inference(
    model,
    processor,
    data: List[Dict],
    device: str = "cuda",
    max_new_tokens: int = 256,
) -> List[str]:
    """
    在给定数据上运行模型推理，返回生成的文本列表。
    """
    predictions = []
    for item in tqdm(data, desc="  推理中"):
        image_id = item["image_id"]
        image_path = os.path.join(IMAGE_FOLDER, image_id)

        # 检查图片是否存在
        if not os.path.exists(image_path):
            # 尝试在其他子目录查找
            alt_paths = [
                os.path.join(IMAGE_FOLDER, "..", "Images_val", image_id),
                os.path.join(IMAGE_FOLDER, "..", "Images_train", image_id),
                os.path.join(IMAGE_FOLDER, "..", "images", "val", image_id),
            ]
            image_path = None
            for ap in alt_paths:
                if os.path.exists(ap):
                    image_path = os.path.abspath(ap)
                    break
            if image_path is None:
                predictions.append("")
                continue

        question = item["question"]
        task_type = item.get("type", "caption")

        # 构建 messages
        messages = build_messages(image_path, question, task_type)

        try:
            # 应用 chat template
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # 处理图片
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )

            # 截取生成的部分
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            predictions.append(output_text.strip())

        except Exception as e:
            print(f"\n  ⚠ 推理失败 (image={image_id}): {e}")
            predictions.append("")

    return predictions


# ============================================================
# 主评估流程
# ============================================================

def evaluate_task(
    model,
    processor,
    task_type: str,
    max_samples: int = 0,
    device: str = "cuda",
    use_training_prompt: bool = False,
) -> Dict:
    """
    评估单个任务类型。

    Args:
        model: 模型
        processor: processor
        task_type: "caption" | "vqa" | "referring"
        max_samples: 最大样本数（0 = 全部）
        device: 设备
        use_training_prompt: 是否使用训练时的 prompt 格式

    Returns:
        包含推理结果和指标的字典
    """
    print(f"\n{'='*60}")
    print(f"  任务: {task_type.upper()}")
    print(f"{'='*60}")

    # 加载数据
    eval_file = EVAL_FILES[task_type]
    if not os.path.exists(eval_file):
        print(f"  ❌ 评估文件不存在: {eval_file}")
        return {"error": f"File not found: {eval_file}"}

    data = json.load(open(eval_file, "r"))
    total = len(data)

    # 采样
    if max_samples > 0 and max_samples < total:
        # 分层采样，保证多样性
        indices = np.linspace(0, total - 1, max_samples, dtype=int)
        data = [data[i] for i in indices]
        print(f"  数据: {len(data)}/{total} 样本 (采样)")
    else:
        print(f"  数据: {len(data)} 样本 (全部)")

    # 如果使用训练 prompt，替换 question
    if use_training_prompt and task_type in TRAINING_PROMPTS:
        template = TRAINING_PROMPTS[task_type]
        for item in data:
            item["original_question"] = item.get("question", "")
            item["question"] = template.replace("{question}", item["question"])

    # 收集 ground truth
    references = [item["ground_truth"] for item in data]

    # 推理
    print("  开始推理...")
    start_time = time.time()
    predictions = run_inference(model, processor, data, device=device)
    elapsed = time.time() - start_time
    print(f"  推理完成，耗时: {elapsed:.1f}s ({elapsed/len(data):.2f}s/样本)")

    # 计算指标 — referring 任务使用 IoU 系列指标
    print("  计算指标...")

    if task_type == "referring":
        # referring: 使用 IoU + Accuracy@τ
        metrics = compute_referring_metrics(references, predictions)
        metrics["valid_samples"] = metrics.pop("valid_boxes", 0)
        metrics["total_samples"] = len(predictions)

        print(f"\n  📊 指标 (referring):")
        print(f"     有效预测: {metrics['valid_samples']}/{metrics['total_samples']}")
        for name in ["mean_iou", "Acc@0.25", "Acc@0.5", "Acc@0.7"]:
            val = metrics.get(name, 0.0)
            print(f"     {name:12s}: {val:.4f}")
    else:
        # caption / vqa: 使用 NLG 指标
        valid_indices = [(r, p) for i, (r, p) in enumerate(zip(references, predictions)) if p.strip()]
        if valid_indices:
            valid_refs, valid_preds = zip(*valid_indices)
        else:
            valid_refs, valid_preds = [], []
        valid_count = len(valid_refs)

        if valid_count > 0:
            metrics = compute_all_metrics(list(valid_refs), list(valid_preds))
            metrics["valid_samples"] = valid_count
            metrics["total_samples"] = len(predictions)
        else:
            metrics = {
                "BLEU-1": 0.0, "BLEU-2": 0.0, "BLEU-3": 0.0, "BLEU-4": 0.0,
                "ROUGE-L": 0.0, "METEOR": 0.0, "CIDEr-D": 0.0,
                "valid_samples": 0, "total_samples": len(predictions),
            }

        print(f"\n  📊 指标 ({task_type}):")
        print(f"     有效样本: {metrics['valid_samples']}/{metrics['total_samples']}")
        for name in ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L", "METEOR", "CIDEr-D"]:
            val = metrics.get(name, 0.0)
            print(f"     {name:12s}: {val:.4f}")

    return {
        "task": task_type,
        "metrics": metrics,
        "predictions": predictions[:10],  # 保存前 10 条示例
        "references": references[:10],
    }


def print_comparison_table(
    base_results: Dict,
    ft_results: Dict,
    tasks: List[str],
):
    """打印基座模型 vs 微调模型的对比表。"""
    caption_metrics = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L", "METEOR", "CIDEr-D"]
    referring_metrics = ["mean_iou", "Acc@0.25", "Acc@0.5", "Acc@0.7"]

    print("\n")
    print("=" * 100)
    print("  📊 VRSBench 评估结果：基座模型 vs 微调模型")
    print("=" * 100)

    for task in tasks:
        if task not in base_results and task not in ft_results:
            continue

        # referring 使用不同指标
        metric_names = referring_metrics if task == "referring" else caption_metrics

        print(f"\n┌{'─' * 98}┐")
        print(f"│ 任务: {task.upper():<91}│")
        print(f"├{'─' * 98}┤")

        # 表头
        header = f"│ {'指标':<14} │ {'基座模型':>14} │ {'微调模型':>14} │ {'变化':>14} │ {'提升':>14} │"
        print(header)
        print(f"│{'─' * 16}┼{'─' * 16}┼{'─' * 16}┼{'─' * 16}┼{'─' * 16}│")

        for name in metric_names:
            base_val = base_results.get(task, {}).get("metrics", {}).get(name, 0.0)
            ft_val = ft_results.get(task, {}).get("metrics", {}).get(name, 0.0)
            diff = ft_val - base_val
            if base_val > 0.0001:
                pct = diff / base_val * 100
                pct_str = f"+{pct:.1f}%" if pct >= 0 else f"{pct:.1f}%"
            elif ft_val > 0.0001:
                pct_str = "NEW"
            else:
                pct_str = "—"

            # 用箭头表示提升/下降
            if diff > 0.0001:
                arrow = " ▲"
            elif diff < -0.0001:
                arrow = " ▼"
            else:
                arrow = "  "

            print(
                f"│ {name:<14} │ {base_val:>14.4f} │ {ft_val:>14.4f} │ "
                f"{diff:>+14.4f} │ {pct_str:>13} {arrow}│"
            )

        # 有效样本数
        base_valid = base_results.get(task, {}).get("metrics", {}).get("valid_samples", 0)
        ft_valid = ft_results.get(task, {}).get("metrics", {}).get("valid_samples", 0)
        print(f"│{'─' * 16}┼{'─' * 16}┼{'─' * 16}┼{'─' * 16}┼{'─' * 16}│")
        print(f"│ {'有效样本':<14} │ {base_valid:>14} │ {ft_valid:>14} │ {'':>14} │ {'':>14} │")
        print(f"└{'─' * 98}┘")

    print("\n说明:")
    print("  ▲ = 微调后提升    ▼ = 微调后下降")
    print("  所有指标数值越高越好")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="VRSBench 评估：对比 Qwen3-VL-2B 微调前后的 NLG 指标"
    )
    parser.add_argument(
        "--task", type=str, default="caption",
        choices=["caption", "vqa", "referring", "all"],
        help="评估的任务类型 (默认: caption)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=0,
        help="每任务最大样本数（0=全部），建议 500-1000"
    )
    parser.add_argument(
        "--base_only", action="store_true",
        help="仅评估基座模型"
    )
    parser.add_argument(
        "--ft_only", action="store_true",
        help="仅评估微调模型"
    )
    parser.add_argument(
        "--base_model", type=str, default=BASE_MODEL_PATH,
        help="基座模型路径"
    )
    parser.add_argument(
        "--lora_path", type=str, default=LORA_ADAPTER_PATH,
        help="LoRA adapter 路径"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="推理设备"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="保存 JSON 结果的路径"
    )
    parser.add_argument(
        "--use_training_prompt", action="store_true",
        help="使用训练时的 prompt 格式（对微调模型可能效果更好）"
    )
    args = parser.parse_args()

    # 确定要评估的任务
    if args.task == "all":
        tasks = ["caption", "vqa", "referring"]
    else:
        tasks = [args.task]

    eval_both = not args.base_only and not args.ft_only

    print("=" * 100)
    print("  VRSBench 评估 —— Qwen3-VL-2B-Instruct 微调前后对比")
    print("=" * 100)
    print(f"  基座模型:   {args.base_model}")
    print(f"  LoRA 路径:  {args.lora_path}")
    print(f"  任务:       {', '.join(tasks)}")
    print(f"  采样数:     {args.max_samples if args.max_samples > 0 else '全部'}")
    print(f"  设备:       {args.device}")
    print(f"  Prompt:     {'训练格式' if args.use_training_prompt else '标准格式'}")
    print()

    # 检查 CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA 不可用，切换到 CPU")
        args.device = "cpu"

    # ================================================================
    # 逐模型评估（先基座，再微调，每次只保留一个模型在 GPU 上）
    # ================================================================
    base_results = {}
    ft_results = {}

    for model_label, load_ft in [("基座模型 (微调前)", False), ("微调模型 (微调后)", True)]:
        if model_label == "基座模型 (微调前)" and args.ft_only:
            continue
        if model_label == "微调模型 (微调后)" and args.base_only:
            continue

        print(f"\n{'=' * 60}")
        print(f"  {model_label}")
        print(f"{'=' * 60}")

        model, processor = load_model_and_processor(
            args.base_model,
            lora_path=args.lora_path if load_ft else None,
            device=args.device,
        )

        results_dest = ft_results if load_ft else base_results

        for task in tasks:
            print(f"\n--- 任务: {task} ---")
            results_dest[task] = evaluate_task(
                model, processor, task,
                max_samples=args.max_samples, device=args.device,
                use_training_prompt=args.use_training_prompt,
            )

        # 释放 GPU 内存
        del model, processor
        torch.cuda.empty_cache()

    # 打印对比表
    if eval_both:
        print_comparison_table(base_results, ft_results, tasks)

    # 保存结果
    if args.output:
        output_data = {
            "config": {
                "base_model": args.base_model,
                "lora_path": args.lora_path,
                "tasks": tasks,
                "max_samples": args.max_samples,
            },
            "base_model": base_results,
            "finetuned_model": ft_results,
        }
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n📁 结果已保存到: {args.output}")

    print("\n✅ 评估完成！")


if __name__ == "__main__":
    main()
