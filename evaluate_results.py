#!/usr/bin/env python3
"""Summarize RS-MLLM JSONL evaluation outputs with uncertainty estimates.

This script is intentionally stdlib-only. It consumes JSONL prediction files
written by eval_baseline.py and produces reproducible summary artifacts without
loading models or images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

CAPTION_NOTE = (
    "Caption metrics are lexical proxy metrics implemented without third-party "
    "evaluation packages; METEOR/CIDEr and LLM-as-Judge are not included in this report."
)
REFER_NOTE = (
    "VRSBench refer coordinates are parsed from either 0-100 or Qwen-style 0-1000 output. "
    "accuracy_at_0_5 and accuracy_at_0_7 are IoU-threshold grounding accuracies; invalid boxes score zero."
)
SUMMARY_FIELDS = [
    "run_id",
    "dataset",
    "task",
    "metric",
    "observed",
    "unit_count",
    "row_count",
    "bootstrap_reps",
    "bootstrap_mean",
    "bootstrap_std",
    "ci95_low",
    "ci95_high",
]
UNIT_FIELDS = [
    "run_id",
    "dataset",
    "task",
    "unit_id",
    "metric",
    "value",
    "row_count",
    "reference_count",
    "prediction_variant_count",
]
ACROSS_RUN_FIELDS = [
    "dataset",
    "task",
    "metric",
    "run_count",
    "mean_observed",
    "std_observed",
    "min_observed",
    "max_observed",
]
PAIRED_FIELDS = [
    "base_run_id",
    "candidate_run_id",
    "dataset",
    "task",
    "metric",
    "matched_unit_count",
    "delta_observed",
    "delta_bootstrap_mean",
    "delta_bootstrap_std",
    "delta_ci95_low",
    "delta_ci95_high",
    "prob_delta_gt_0",
]
VRSBENCH_TYPE_FIELDS = [
    "run_id",
    "question_type",
    "observed",
    "correct_count",
    "unit_count",
    "bootstrap_reps",
    "bootstrap_mean",
    "bootstrap_std",
    "ci95_low",
    "ci95_high",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Repeatable RUN_ID:PATH spec. Files sharing RUN_ID are one experiment.",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--compare", help="Optional BASE_RUN_ID:CANDIDATE_RUN_ID")
    parser.add_argument(
        "--vrsbench-vqa-annotations",
        type=Path,
        help=(
            "Optional VRSBench_EVAL_vqa.json path. When provided, write exact "
            "per-question-type accuracy and bootstrap confidence intervals."
        ),
    )
    return parser.parse_args()


def split_spec(spec: str, label: str) -> tuple[str, Path]:
    if spec.count(":") != 1:
        raise ValueError(f"malformed {label} spec: {spec!r}")
    run_id, path_text = spec.split(":", 1)
    if not run_id or not path_text:
        raise ValueError(f"malformed {label} spec: {spec!r}")
    return run_id, Path(path_text)


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc


def line_count(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(str(path))
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def load_runs(run_specs: list[str]) -> dict[str, list[dict[str, Any]]]:
    runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in run_specs:
        run_id, path = split_spec(spec, "--run")
        for line_no, row in iter_jsonl(path):
            row = dict(row)
            row["__source_path"] = str(path)
            row["__line_no"] = line_no
            row["__row_index"] = len(runs[run_id])
            runs[run_id].append(row)
    return dict(runs)


def record_key(row: dict[str, Any], row_index: int) -> str:
    dataset = str(row.get("dataset", ""))
    task = str(row.get("task", ""))
    if dataset == "vrsbench" and task == "caption":
        return str(row.get("image_id", row_index))
    if dataset == "vrsbench" and task in {"vqa", "refer"}:
        return f"{row.get('image_id', row_index)}::{row.get('question', '')}"
    if dataset == "xlrs_lite" and task == "vqa":
        return f"{row.get('question', '')}::{row.get('ground_truth', '')}"
    if dataset == "mme_realworld" and task == "vqa":
        return str(row.get("question_id", row_index))
    if dataset == "levir_cc" and task == "change_caption":
        return str(row.get("image_id", row_index))
    return str(row_index)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9.+-]+", " ", str(text).lower())).strip()


def tokenize(text: str) -> list[str]:
    norm = normalize_text(text)
    return norm.split() if norm else []


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for tok_a in a:
        curr = [0]
        for j, tok_b in enumerate(b, 1):
            if tok_a == tok_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l_f1(prediction: str, references: list[str]) -> float:
    pred_tokens = tokenize(prediction)
    if not pred_tokens:
        return 0.0
    best = 0.0
    for ref in references:
        ref_tokens = tokenize(ref)
        if not ref_tokens:
            continue
        lcs = lcs_len(pred_tokens, ref_tokens)
        if lcs == 0:
            score = 0.0
        else:
            precision = lcs / len(pred_tokens)
            recall = lcs / len(ref_tokens)
            score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def token_f1(prediction: str, references: list[str]) -> float:
    pred_tokens = tokenize(prediction)
    if not pred_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    best = 0.0
    for ref in references:
        ref_tokens = tokenize(ref)
        if not ref_tokens:
            continue
        ref_counts = Counter(ref_tokens)
        overlap = sum((pred_counts & ref_counts).values())
        if overlap == 0:
            score = 0.0
        else:
            precision = overlap / len(pred_tokens)
            recall = overlap / len(ref_tokens)
            score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(0, max(0, len(tokens) - n + 1)))


def sentence_bleu(prediction: str, references: list[str], max_n: int) -> float:
    pred_tokens = tokenize(prediction)
    if not pred_tokens:
        return 0.0
    ref_tokens_list = [tokenize(ref) for ref in references if tokenize(ref)]
    if not ref_tokens_list:
        return 0.0

    precisions: list[float] = []
    for n in range(1, max_n + 1):
        pred_ngrams = ngrams(pred_tokens, n)
        max_ref_counts: Counter[tuple[str, ...]] = Counter()
        for ref_tokens in ref_tokens_list:
            ref_counts = ngrams(ref_tokens, n)
            for gram, count in ref_counts.items():
                max_ref_counts[gram] = max(max_ref_counts[gram], count)
        clipped = sum(min(count, max_ref_counts.get(gram, 0)) for gram, count in pred_ngrams.items())
        total = sum(pred_ngrams.values())
        precisions.append((clipped + 1.0) / (total + 1.0))

    shortest_ref_len = min(len(ref_tokens) for ref_tokens in ref_tokens_list)
    pred_len = len(pred_tokens)
    brevity_penalty = 1.0 if pred_len > shortest_ref_len else math.exp(1 - shortest_ref_len / pred_len)
    log_precision = sum(math.log(p) for p in precisions) / max_n
    return brevity_penalty * math.exp(log_precision)


def make_unit(
    row: dict[str, Any],
    unit_id: str,
    references: list[str] | None = None,
    prediction: str | None = None,
    row_count: int = 1,
    prediction_variant_count: int | None = None,
) -> dict[str, Any]:
    return {
        "dataset": str(row.get("dataset", "")),
        "task": str(row.get("task", "")),
        "unit_id": unit_id,
        "references": references if references is not None else [str(row.get("ground_truth", ""))],
        "prediction": prediction if prediction is not None else str(row.get("prediction", "")),
        "correct": row.get("correct"),
        "row_count": row_count,
        "reference_count": len(references) if references is not None else 1,
        "prediction_variant_count": prediction_variant_count,
    }


def group_units(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[(str(row.get("dataset", "")), str(row.get("task", "")))].append(row)

    units_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, task_rows in grouped_rows.items():
        dataset, task = key
        if dataset == "levir_cc" and task == "change_caption":
            by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
            order: list[str] = []
            for row in task_rows:
                unit_id = record_key(row, int(row.get("__row_index", 0)))
                if unit_id not in by_image:
                    order.append(unit_id)
                by_image[unit_id].append(row)
            units: list[dict[str, Any]] = []
            for unit_id in order:
                rows_for_unit = by_image[unit_id]
                references = [str(r.get("ground_truth", "")) for r in rows_for_unit]
                predictions = [str(r.get("prediction", "")) for r in rows_for_unit if normalize_text(str(r.get("prediction", "")))]
                prediction = predictions[0] if predictions else ""
                variant_count = len(set(predictions))
                units.append(make_unit(rows_for_unit[0], unit_id, references, prediction, len(rows_for_unit), variant_count))
            units_by_key[key] = units
        else:
            units_by_key[key] = [
                make_unit(row, record_key(row, int(row.get("__row_index", i))))
                for i, row in enumerate(task_rows)
            ]
    return units_by_key


def score_accuracy(units: list[dict[str, Any]]) -> list[float]:
    return [1.0 if bool(unit.get("correct")) else 0.0 for unit in units]


def parse_bbox(text: str) -> tuple[float, float, float, float] | None:
    numbers = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", str(text))[:4]]
    if len(numbers) != 4 or not all(math.isfinite(value) for value in numbers):
        return None
    max_abs = max(abs(value) for value in numbers)
    if max_abs <= 1.0:
        scale = 1.0
    elif max_abs <= 100.0:
        scale = 100.0
    elif max_abs <= 1000.0:
        scale = 1000.0
    else:
        return None
    x1, y1, x2, y2 = (min(1.0, max(0.0, value / scale)) for value in numbers)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_w * intersection_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def score_grounding(units: list[dict[str, Any]]) -> dict[str, list[float]]:
    scores = {
        "format_validity": [],
        "mean_iou": [],
        "accuracy_at_0_5": [],
        "accuracy_at_0_7": [],
    }
    for unit in units:
        predicted = parse_bbox(str(unit.get("prediction", "")))
        references = [parse_bbox(str(r)) for r in unit.get("references", [])]
        references = [bbox for bbox in references if bbox is not None]
        valid = predicted is not None and bool(references)
        iou = max((bbox_iou(predicted, reference) for reference in references), default=0.0) if predicted else 0.0
        scores["format_validity"].append(1.0 if valid else 0.0)
        scores["mean_iou"].append(iou)
        scores["accuracy_at_0_5"].append(1.0 if iou >= 0.5 else 0.0)
        scores["accuracy_at_0_7"].append(1.0 if iou >= 0.7 else 0.0)
    return scores


def score_text(units: list[dict[str, Any]]) -> dict[str, list[float]]:
    scores: dict[str, list[float]] = {
        "rouge_l_f1": [],
        "token_f1": [],
        "sentence_bleu1_mean": [],
        "sentence_bleu2_mean": [],
        "sentence_bleu3_mean": [],
        "sentence_bleu4_mean": [],
        "empty_prediction_rate": [],
        "avg_prediction_tokens": [],
    }
    for unit in units:
        prediction = str(unit.get("prediction", ""))
        references = [str(r) for r in unit.get("references", [])]
        scores["rouge_l_f1"].append(rouge_l_f1(prediction, references))
        scores["token_f1"].append(token_f1(prediction, references))
        for max_n in range(1, 5):
            scores[f"sentence_bleu{max_n}_mean"].append(sentence_bleu(prediction, references, max_n))
        scores["empty_prediction_rate"].append(0.0 if normalize_text(prediction) else 1.0)
        scores["avg_prediction_tokens"].append(float(len(tokenize(prediction))))
    return scores


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight


def stable_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256((str(seed) + "|" + "|".join(parts)).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def bootstrap_mean(values: list[float], reps: int, rng: random.Random) -> list[float]:
    if not values or reps <= 0:
        return []
    n = len(values)
    out: list[float] = []
    for _ in range(reps):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        out.append(total / n)
    return out


def summarize_values_from_bootstrap(
    values: list[float], bootstrap: int, boot: list[float]
) -> dict[str, float | int | None]:
    observed = mean(values)
    if not values:
        return {
            "observed": None,
            "unit_count": 0,
            "bootstrap_reps": bootstrap,
            "bootstrap_mean": None,
            "bootstrap_std": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    if not boot:
        return {
            "observed": observed,
            "unit_count": len(values),
            "bootstrap_reps": bootstrap,
            "bootstrap_mean": None,
            "bootstrap_std": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    boot_sorted = sorted(boot)
    return {
        "observed": observed,
        "unit_count": len(values),
        "bootstrap_reps": bootstrap,
        "bootstrap_mean": mean(boot),
        "bootstrap_std": statistics.stdev(boot) if len(boot) > 1 else 0.0,
        "ci95_low": percentile(boot_sorted, 0.025),
        "ci95_high": percentile(boot_sorted, 0.975),
    }


def summarize_values(values: list[float], bootstrap: int, seed: int) -> dict[str, float | int | None]:
    boot = bootstrap_mean(values, bootstrap, random.Random(seed))
    return summarize_values_from_bootstrap(values, bootstrap, boot)


def summarize_metric(
    run_id: str,
    dataset: str,
    task: str,
    metric: str,
    values: list[float],
    row_count: int,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    metric_seed = stable_seed(seed, run_id, dataset, task, metric)
    stat = summarize_values(values, bootstrap, metric_seed)
    return {
        "run_id": run_id,
        "dataset": dataset,
        "task": task,
        "metric": metric,
        "observed": stat["observed"],
        "unit_count": stat["unit_count"],
        "row_count": row_count,
        "bootstrap_reps": stat["bootstrap_reps"],
        "bootstrap_mean": stat["bootstrap_mean"],
        "bootstrap_std": stat["bootstrap_std"],
        "ci95_low": stat["ci95_low"],
        "ci95_high": stat["ci95_high"],
    }


def metrics_for_units(dataset: str, task: str, units: list[dict[str, Any]]) -> dict[str, list[float]]:
    if dataset == "vrsbench" and task == "caption":
        return score_text(units)
    if dataset == "levir_cc" and task == "change_caption":
        return score_text(units)
    if dataset == "vrsbench" and task == "refer":
        return score_grounding(units)
    if units and "correct" in units[0]:
        return {"accuracy": score_accuracy(units)}
    return {}


def build_summaries(
    runs: dict[str, list[dict[str, Any]]], bootstrap: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    unit_score_rows: list[dict[str, Any]] = []
    for run_id, rows in sorted(runs.items()):
        units_by_key = group_units(rows)
        for (dataset, task), units in sorted(units_by_key.items()):
            metric_values = metrics_for_units(dataset, task, units)
            row_count = sum(int(unit.get("row_count", 1)) for unit in units)
            for metric, values in sorted(metric_values.items()):
                summary_rows.append(summarize_metric(run_id, dataset, task, metric, values, row_count, bootstrap, seed))
                for unit, value in zip(units, values):
                    row = {
                        "run_id": run_id,
                        "dataset": dataset,
                        "task": task,
                        "unit_id": unit["unit_id"],
                        "metric": metric,
                        "value": value,
                        "row_count": unit.get("row_count", 1),
                        "reference_count": unit.get("reference_count"),
                        "prediction_variant_count": unit.get("prediction_variant_count"),
                    }
                    unit_score_rows.append(row)
    return summary_rows, unit_score_rows


def build_across_run_summary(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in summary_rows:
        observed = row.get("observed")
        if observed is None:
            continue
        grouped[(row["dataset"], row["task"], row["metric"])].append(float(observed))
    out: list[dict[str, Any]] = []
    for (dataset, task, metric), values in sorted(grouped.items()):
        out.append(
            {
                "dataset": dataset,
                "task": task,
                "metric": metric,
                "run_count": len(values),
                "mean_observed": mean(values),
                "std_observed": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min_observed": min(values),
                "max_observed": max(values),
            }
        )
    return out


def compare_runs(
    unit_score_rows: list[dict[str, Any]], compare_spec: str | None, bootstrap: int, seed: int
) -> list[dict[str, Any]]:
    if not compare_spec:
        return []
    base_run, candidate_run = split_spec(compare_spec, "--compare")
    candidate_run_id = str(candidate_run)
    by_metric: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(dict)
    for row in unit_score_rows:
        run_id = str(row["run_id"])
        if run_id not in {base_run, candidate_run_id}:
            continue
        key = (run_id, row["dataset"], row["task"], row["metric"])
        by_metric[key][str(row["unit_id"])] = float(row["value"])

    metric_keys = sorted({key[1:] for key in by_metric})
    out: list[dict[str, Any]] = []
    for dataset, task, metric in metric_keys:
        base_values = by_metric.get((base_run, dataset, task, metric), {})
        cand_values = by_metric.get((candidate_run_id, dataset, task, metric), {})
        matched = sorted(set(base_values) & set(cand_values))
        if len(matched) < 2:
            out.append(
                {
                    "base_run_id": base_run,
                    "candidate_run_id": candidate_run_id,
                    "dataset": dataset,
                    "task": task,
                    "metric": metric,
                    "matched_unit_count": len(matched),
                    "delta_observed": None,
                    "delta_bootstrap_mean": None,
                    "delta_bootstrap_std": None,
                    "delta_ci95_low": None,
                    "delta_ci95_high": None,
                    "prob_delta_gt_0": None,
                }
            )
            continue
        deltas = [cand_values[unit_id] - base_values[unit_id] for unit_id in matched]
        delta_seed = stable_seed(seed, base_run, candidate_run_id, dataset, task, metric, "delta")
        boot = bootstrap_mean(deltas, bootstrap, random.Random(delta_seed))
        stat = summarize_values_from_bootstrap(deltas, bootstrap, boot)
        prob_gt_zero = (sum(1 for value in boot if value > 0) / len(boot)) if boot else None
        out.append(
            {
                "base_run_id": base_run,
                "candidate_run_id": candidate_run_id,
                "dataset": dataset,
                "task": task,
                "metric": metric,
                "matched_unit_count": len(matched),
                "delta_observed": stat["observed"],
                "delta_bootstrap_mean": stat["bootstrap_mean"],
                "delta_bootstrap_std": stat["bootstrap_std"],
                "delta_ci95_low": stat["ci95_low"],
                "delta_ci95_high": stat["ci95_high"],
                "prob_delta_gt_0": prob_gt_zero,
            }
        )
    return out


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def load_vrsbench_type_lookups(
    path: Path,
) -> tuple[dict[str, str], dict[tuple[str, str, str], str]]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    samples = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise ValueError(f"{path}: expected a JSON array")

    by_question_id: dict[str, str] = {}
    by_content: dict[tuple[str, str, str], str] = {}
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"{path}: annotation {index} is not an object")
        question_type = str(sample.get("type", "")).strip()
        if not question_type:
            raise ValueError(f"{path}: annotation {index} has no type")

        question_id = sample.get("question_id")
        if question_id is not None:
            id_key = str(question_id)
            previous = by_question_id.setdefault(id_key, question_type)
            if previous != question_type:
                raise ValueError(f"{path}: question_id {id_key!r} has conflicting types")

        content_key = (
            str(sample.get("image_id", "")),
            str(sample.get("question", "")),
            str(sample.get("ground_truth", "")),
        )
        previous = by_content.setdefault(content_key, question_type)
        if previous != question_type:
            raise ValueError(f"{path}: duplicate VQA content has conflicting types")
    return by_question_id, by_content


def build_vrsbench_type_summary(
    runs: dict[str, list[dict[str, Any]]],
    annotation_path: Path,
    bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_question_id, by_content = load_vrsbench_type_lookups(annotation_path)
    out: list[dict[str, Any]] = []
    unmatched: list[str] = []
    for run_id, rows in sorted(runs.items()):
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            if row.get("dataset") != "vrsbench" or row.get("task") != "vqa":
                continue
            question_type = None
            question_id = row.get("question_id")
            if question_id is not None:
                question_type = by_question_id.get(str(question_id))
            if question_type is None:
                content_key = (
                    str(row.get("image_id", "")),
                    str(row.get("question", "")),
                    str(row.get("ground_truth", "")),
                )
                question_type = by_content.get(content_key)
            if question_type is None:
                unmatched.append(
                    f"{run_id}:{row.get('image_id', '')}:{row.get('question', '')}"
                )
                continue
            grouped[question_type].append(1.0 if bool(row.get("correct")) else 0.0)

        for question_type, values in sorted(grouped.items()):
            stat = summarize_values(
                values,
                bootstrap,
                stable_seed(seed, run_id, "vrsbench", "vqa", question_type),
            )
            out.append(
                {
                    "run_id": run_id,
                    "question_type": question_type,
                    "observed": stat["observed"],
                    "correct_count": sum(int(value) for value in values),
                    "unit_count": len(values),
                    "bootstrap_reps": bootstrap,
                    "bootstrap_mean": stat["bootstrap_mean"],
                    "bootstrap_std": stat["bootstrap_std"],
                    "ci95_low": stat["ci95_low"],
                    "ci95_high": stat["ci95_high"],
                }
            )

    if unmatched:
        examples = "; ".join(unmatched[:3])
        raise ValueError(
            f"{annotation_path}: failed to match {len(unmatched)} VRSBench VQA rows; "
            f"examples: {examples}"
        )
    return out


def write_vrsbench_type_outputs(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_json(out_dir / "vrsbench_vqa_type_summary.json", rows)
    write_csv(out_dir / "vrsbench_vqa_type_summary.csv", rows, VRSBENCH_TYPE_FIELDS)
    lines = [
        "# VRSBench VQA Question-Type Summary",
        "",
        "| run_id | question_type | accuracy | correct | total | ci95_low | ci95_high |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {run_id} | {question_type} | {observed} | {correct_count} | {unit_count} | {ci95_low} | {ci95_high} |".format(
                run_id=str(row["run_id"]).replace("|", "\\|"),
                question_type=str(row["question_type"]).replace("|", "\\|"),
                observed=format_float(row.get("observed")),
                correct_count=row["correct_count"],
                unit_count=row["unit_count"],
                ci95_low=format_float(row.get("ci95_low")),
                ci95_high=format_float(row.get("ci95_high")),
            )
        )
    lines.append("")
    (out_dir / "vrsbench_vqa_type_summary.md").write_text("\n".join(lines), encoding="utf-8")

def format_float(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_summary_md(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# RS-MLLM Evaluation Summary",
        "",
        "| run_id | dataset | task | metric | observed | unit_count | row_count | ci95_low | ci95_high |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary_rows, key=lambda r: (r["run_id"], r["dataset"], r["task"], r["metric"])):
        lines.append(
            "| {run_id} | {dataset} | {task} | {metric} | {observed} | {unit_count} | {row_count} | {ci95_low} | {ci95_high} |".format(
                run_id=row["run_id"],
                dataset=row["dataset"],
                task=row["task"],
                metric=row["metric"],
                observed=format_float(row.get("observed")),
                unit_count=row.get("unit_count"),
                row_count=row.get("row_count"),
                ci95_low=format_float(row.get("ci95_low")),
                ci95_high=format_float(row.get("ci95_high")),
            )
        )
    lines.extend(["", "## Notes", "", f"- {CAPTION_NOTE}", f"- {REFER_NOTE}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    inputs = []
    for spec in args.run:
        run_id, path = split_spec(spec, "--run")
        inputs.append(
            {
                "run_id": run_id,
                "path": str(path),
                "bytes": path.stat().st_size if path.exists() else None,
                "line_count": line_count(path) if path.exists() else None,
            }
        )
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "timestamp_local": datetime.now().astimezone().isoformat(),
        "argv": sys.argv,
        "python_version": sys.version,
        "run_specs": args.run,
        "inputs": inputs,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "compare": args.compare,
        "vrsbench_vqa_annotations": (
            str(args.vrsbench_vqa_annotations) if args.vrsbench_vqa_annotations else None
        ),
    }
    return manifest


def write_outputs(
    out_dir: Path,
    summary_rows: list[dict[str, Any]],
    unit_score_rows: list[dict[str, Any]],
    across_run_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary_rows)
    write_csv(out_dir / "summary.csv", summary_rows, SUMMARY_FIELDS)
    write_summary_md(out_dir / "summary.md", summary_rows)
    write_jsonl(out_dir / "unit_scores.jsonl", unit_score_rows)
    write_json(out_dir / "run_manifest.json", manifest)
    write_csv(out_dir / "across_run_summary.csv", across_run_rows, ACROSS_RUN_FIELDS)
    if paired_rows:
        write_csv(out_dir / "paired_comparison.csv", paired_rows, PAIRED_FIELDS)


def main() -> int:
    args = parse_args()
    if args.bootstrap < 0:
        raise ValueError("--bootstrap must be >= 0")
    runs = load_runs(args.run)
    summary_rows, unit_score_rows = build_summaries(runs, args.bootstrap, args.seed)
    across_run_rows = build_across_run_summary(summary_rows)
    paired_rows = compare_runs(unit_score_rows, args.compare, args.bootstrap, args.seed)
    vrsbench_type_rows = (
        build_vrsbench_type_summary(
            runs, args.vrsbench_vqa_annotations, args.bootstrap, args.seed
        )
        if args.vrsbench_vqa_annotations
        else []
    )
    manifest = build_manifest(args)
    write_outputs(args.out_dir, summary_rows, unit_score_rows, across_run_rows, paired_rows, manifest)
    if args.vrsbench_vqa_annotations:
        write_vrsbench_type_outputs(args.out_dir, vrsbench_type_rows)
    print(f"Saved evaluation report to: {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
