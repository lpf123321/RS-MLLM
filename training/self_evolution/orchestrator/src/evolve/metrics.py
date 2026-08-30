from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

from .bbox import iou
from .io import read_jsonl


def _first_local_sam_recovery(target: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first successful post-initial SAM call for an initially failed target."""
    if target.get("initial_sam_success"):
        return None
    attempts = sorted(
        (
            attempt for attempt in target.get("sam_attempts", [])
            if attempt.get("stage") != "initial"
        ),
        key=lambda attempt: int(attempt.get("call_index", 0)),
    )
    return next((attempt for attempt in attempts if attempt.get("sam_success")), None)


def _recovered_instances(recovery: dict[str, Any] | None) -> list[dict[str, Any]]:
    if recovery is None:
        return []
    return [
        instance for instance in recovery.get("instances", [])
        if instance.get("retained_by_cvsearch") and instance.get("valid_mask")
        and instance.get("pseudo_bbox_xyxy")
    ]


def _bootstrap(values: list[float], seed: int = 42, repeats: int = 10_000) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    estimates = []
    for _ in range(repeats):
        estimates.append(mean(values[rng.randrange(len(values))] for _ in values))
    estimates.sort()
    return [estimates[int(0.025 * repeats)], estimates[min(repeats - 1, int(0.975 * repeats))]]


def trajectory_metrics(path: str | Path, seed: int = 42) -> dict[str, Any]:
    rows = list(read_jsonl(path))
    correctness = [float(bool(row.get("answer_correct"))) for row in rows]
    illegal = [float(row.get("normalized_prediction") is None) for row in rows]
    pops = [float(row.get("num_pop", 0.0)) for row in rows]
    routes = Counter(str(row.get("search_mode", 3)) for row in rows)
    per_family: dict[str, list[float]] = defaultdict(list)
    for row, value in zip(rows, correctness):
        per_family[row.get("question_family", "unknown")].append(value)
    teacher_rows = [row for row in rows if row.get("teacher_prediction") is not None]
    logprob_rows = [
        row for row in teacher_rows
        if row.get("teacher_gt_logprob") is not None and row.get("student_gt_logprob") is not None
    ]
    gains = [row["teacher_gt_logprob"] - row["student_gt_logprob"] for row in logprob_rows]
    recoveries = [
        recovery
        for row in rows if int(row.get("search_mode", 3)) == 2 and row.get("answer_correct")
        for target in row.get("target_traces", [])
        if (recovery := _first_local_sam_recovery(target)) is not None
        and _recovered_instances(recovery)
    ]
    recovery_stages = Counter(str(recovery.get("stage")) for recovery in recoveries)
    return {
        "samples": len(rows),
        "correct": int(sum(correctness)),
        "accuracy": mean(correctness) if correctness else 0.0,
        "accuracy_ci95": _bootstrap(correctness, seed),
        "illegal_rate": mean(illegal) if illegal else 0.0,
        "mean_pop": mean(pops) if pops else 0.0,
        "routes": dict(sorted(routes.items())),
        "tree_search_samples": int(routes.get("2", 0)),
        "per_question_family_accuracy": {key: mean(values) for key, values in sorted(per_family.items())},
        "teacher_crop_accuracy": mean([
            float(row["teacher_prediction"] == row["gt_answer"]) for row in teacher_rows
        ]) if teacher_rows else 0.0,
        "teacher_logprob_gain_mean": mean(gains) if gains else None,
        "opsd_acceptance_rate": mean([float(row.get("accepted_for_opsd", False)) for row in rows]) if rows else 0.0,
        "sam_pseudo_target_count": len(recoveries),
        "sam_recovered_target_count": len(recoveries),
        "sam_recovered_instance_count": sum(len(_recovered_instances(recovery)) for recovery in recoveries),
        "sam_recovery_stages": dict(sorted(recovery_stages.items())),
    }


def sam_trace_metrics(path: str | Path, threshold: float = 0.5) -> dict[str, Any]:
    rows = list(read_jsonl(path))
    top_ious, recalls, scored = [], [], []
    pseudo_samples = 0
    for row in rows:
        if int(row.get("search_mode", 3)) != 2 or not row.get("answer_correct"):
            continue
        for target in row.get("target_traces", []):
            recovery = _first_local_sam_recovery(target)
            instances = _recovered_instances(recovery)
            if not instances:
                continue
            prediction = target.get("initial_sam_prediction") or {}
            candidates = list(zip(prediction.get("joint_scores", []), prediction.get("boxes_xyxy", [])))
            candidates.sort(key=lambda item: float(item[0]), reverse=True)
            for instance in instances:
                pseudo_xyxy = [float(value) for value in instance["pseudo_bbox_xyxy"]]
                top = iou(candidates[0][1], pseudo_xyxy) if candidates else 0.0
                top_ious.append(top)
                recalls.append(float(any(
                    float(score) >= threshold and iou(box, pseudo_xyxy) >= 0.5
                    for score, box in candidates
                )))
                for score, box in candidates:
                    scored.append((float(score), int(iou(box, pseudo_xyxy) >= 0.5)))
                pseudo_samples += 1
    scored.sort(reverse=True)
    positives = max(1, pseudo_samples)
    true_seen = 0
    precision_sum = 0.0
    for rank, (_, is_true) in enumerate(scored, 1):
        if is_true:
            true_seen += 1
            precision_sum += true_seen / rank
    return {
        "pseudo_samples": pseudo_samples,
        "pseudo_ap50": precision_sum / positives,
        "pseudo_recall_at_0_5": mean(recalls) if recalls else 0.0,
        "pseudo_top1_iou_mean": mean(top_ious) if top_ious else 0.0,
        "pseudo_top1_iou_median": median(top_ious) if top_ious else 0.0,
    }


def mcnemar(left: list[bool], right: list[bool]) -> dict[str, float]:
    if len(left) != len(right):
        raise ValueError("Paired predictions must have the same length")
    b = sum(a and not c for a, c in zip(left, right))
    c = sum((not a) and d for a, d in zip(left, right))
    statistic = (abs(b - c) - 1) ** 2 / (b + c) if b + c else 0.0
    p_value = math.erfc(math.sqrt(statistic / 2.0))
    return {"left_only_correct": b, "right_only_correct": c, "chi2_cc": statistic, "p_value": p_value}


def paired_trajectory_comparison(
    baseline_path: str | Path, candidate_path: str | Path, seed: int = 42,
) -> dict[str, Any]:
    baseline = {row["sample_id"]: bool(row.get("answer_correct")) for row in read_jsonl(baseline_path)}
    candidate = {row["sample_id"]: bool(row.get("answer_correct")) for row in read_jsonl(candidate_path)}
    if baseline.keys() != candidate.keys():
        missing_left = sorted(candidate.keys() - baseline.keys())[:5]
        missing_right = sorted(baseline.keys() - candidate.keys())[:5]
        raise ValueError(f"Unpaired trajectory IDs; baseline_missing={missing_left}, candidate_missing={missing_right}")
    ids = sorted(baseline)
    left = [baseline[sample_id] for sample_id in ids]
    right = [candidate[sample_id] for sample_id in ids]
    deltas = [float(new) - float(old) for old, new in zip(left, right)]
    return {
        "pairs": len(ids),
        "accuracy_delta": mean(deltas) if deltas else 0.0,
        "accuracy_delta_ci95": _bootstrap(deltas, seed),
        "mcnemar": mcnemar(left, right),
    }
