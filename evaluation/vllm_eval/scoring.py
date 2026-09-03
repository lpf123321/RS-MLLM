"""Strict official-compatible and logically cleaned scoring paths."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from caption_metrics import summarize_reported_caption_metrics
from schema import Sample

_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_LETTER_RE = re.compile(r"(?<![A-Z])([A-E])(?![A-Z])")
_ALIASES = {
    "grey": "gray",
    "rectangular": "rectangle",
    "windmill": "wind turbine",
    "aeroplane": "airplane",
}


def normalize_answer(text: str, *, aliases: bool = False) -> str:
    normalized = text.lower()
    if aliases:
        for source, target in _ALIASES.items():
            normalized = re.sub(rf"\b{re.escape(source)}\b", target, normalized)
    normalized = re.sub(r"[^a-z0-9.+-]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def parse_bbox(text: str) -> list[float] | None:
    cleaned = re.sub(r"\b[xy][12]\b", "", text, flags=re.IGNORECASE)
    values = [float(value) for value in _NUMBER_RE.findall(cleaned)]
    if len(values) < 4:
        return None
    coordinates = values[:4]
    maximum = max(coordinates)
    if min(coordinates) < 0 or maximum > 1000:
        return None
    if maximum <= 1:
        coordinates = [value * 100 for value in coordinates]
    elif maximum > 100:
        coordinates = [value / 10 for value in coordinates]
    x1, y1, x2, y2 = coordinates
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x1 == x2 or y1 == y2:
        return None
    return [x1, y1, x2, y2]


def bbox_iou(
    left: list[float], right: list[float], *, inclusive_integer: bool
) -> float:
    increment = 1.0 if inclusive_integer else 0.0
    intersection_width = max(
        0.0, min(left[2], right[2]) - max(left[0], right[0]) + increment
    )
    intersection_height = max(
        0.0, min(left[3], right[3]) - max(left[1], right[1]) + increment
    )
    intersection = intersection_width * intersection_height
    left_area = (left[2] - left[0] + increment) * (left[3] - left[1] + increment)
    right_area = (right[2] - right[0] + increment) * (right[3] - right[1] + increment)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def parse_choice_labels(
    prediction: str, *, allowed: set[str], multiple: bool
) -> list[str]:
    upper = prediction.upper().strip()
    labels = [match for match in _LETTER_RE.findall(upper) if match in allowed]
    if not labels:
        compact = re.sub(r"[\s,;/|+\-]", "", upper.strip(".()[]{}"))
        if re.fullmatch(r"[A-E]+", compact) and set(compact) <= allowed:
            labels = list(compact)
    unique = sorted(set(labels))
    if not multiple:
        return unique if len(unique) == 1 else []
    return unique


def _fallback_choice_text(prediction: str, choices: dict[str, str]) -> list[str]:
    normalized_prediction = normalize_answer(prediction)
    matches = [
        label
        for label, text in choices.items()
        if normalized_prediction == normalize_answer(text)
    ]
    return sorted(matches)


def _choice_score(sample: Sample, prediction: str) -> dict[str, Any]:
    multiple = sample.task_type == "multi_choice"
    parsed = parse_choice_labels(
        prediction, allowed=set(sample.choices), multiple=multiple
    )
    if not parsed:
        parsed = _fallback_choice_text(prediction, sample.choices)
    official = set(parsed) == set(sample.answer_labels)
    if multiple:
        predicted_content = {
            normalize_answer(sample.choices[label], aliases=True) for label in parsed
        }
        answer_content = {
            normalize_answer(sample.choices[label], aliases=True)
            for label in sample.answer_labels
        }
        clean = predicted_content == answer_content
    else:
        clean = bool(parsed) and parsed[0] in set(
            sample.accepted_labels or sample.answer_labels
        )
    return {
        "parsed_answer": parsed,
        "official_correct": official,
        "clean_correct": clean if sample.clean_status in ("keep", "corrected") else None,
        "clean_eligible": sample.clean_status in ("keep", "corrected"),
    }


def _vqa_score(sample: Sample, prediction: str) -> dict[str, Any]:
    reference = sample.references[0]
    official = normalize_answer(prediction) == normalize_answer(reference)
    clean = normalize_answer(prediction, aliases=True) == normalize_answer(
        reference, aliases=True
    )
    return {
        "parsed_answer": normalize_answer(prediction),
        "official_correct": official,
        "clean_correct": clean if sample.clean_status in ("keep", "corrected") else None,
        "clean_eligible": sample.clean_status in ("keep", "corrected"),
    }


def _bbox_score(sample: Sample, prediction: str) -> dict[str, Any]:
    predicted = parse_bbox(prediction)
    reference = parse_bbox(sample.references[0])
    if reference is None:
        return {
            "parsed_answer": predicted,
            "official_iou": 0.0,
            "clean_iou": 0.0
            if sample.clean_status in ("keep", "corrected")
            else None,
            "official_correct": False,
            "clean_correct": False
            if sample.clean_status in ("keep", "corrected")
            else None,
            "clean_eligible": sample.clean_status in ("keep", "corrected"),
            "format_valid": predicted is not None,
        }
    official_iou = (
        bbox_iou(predicted, reference, inclusive_integer=True) if predicted else 0.0
    )
    clean_iou = (
        bbox_iou(predicted, reference, inclusive_integer=False) if predicted else 0.0
    )
    return {
        "parsed_answer": predicted,
        "official_iou": official_iou,
        "clean_iou": clean_iou if sample.clean_status in ("keep", "corrected") else None,
        "official_correct": official_iou >= 0.5,
        "clean_correct": clean_iou >= 0.5 if sample.clean_status in ("keep", "corrected") else None,
        "clean_eligible": sample.clean_status in ("keep", "corrected"),
        "format_valid": predicted is not None,
    }


def score_prediction(sample: Sample, prediction: str) -> dict[str, Any]:
    if sample.task_type in {"single_choice", "multi_choice"}:
        return _choice_score(sample, prediction)
    if sample.task_type == "open_vqa":
        return _vqa_score(sample, prediction)
    if sample.task_type == "bbox":
        return _bbox_score(sample, prediction)
    return {
        "parsed_answer": None,
        "official_correct": None,
        "clean_correct": None,
        "clean_eligible": sample.clean_status in ("keep", "corrected"),
    }


def _accuracy(correct: int, total: int) -> float | None:
    return correct / total if total else None


def summarize_predictions(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[f"{row['sample']['dataset']}/{row['sample']['subtask']}"].append(row)

    def caption_metrics(group_rows: list[dict[str, Any]], *, clean: bool) -> dict:
        caption_rows = [
            row
            for row in group_rows
            if row["sample"]["task_type"] in {"caption", "change_caption"}
            and not row.get("error")
            and (not clean or row["sample"]["clean_status"] in ("keep", "corrected"))
        ]
        if not caption_rows:
            return {}
        return summarize_reported_caption_metrics(
            [row["prediction"] for row in caption_rows],
            [row["sample"]["references"] for row in caption_rows],
        )

    def bbox_metrics(group_rows: list[dict[str, Any]], *, clean: bool) -> dict:
        key = "clean_iou" if clean else "official_iou"
        bbox_rows = [
            row
            for row in group_rows
            if row["score"].get(key) is not None
            and (not clean or row["score"].get("clean_eligible"))
        ]
        if not bbox_rows:
            return {}
        values = [float(row["score"][key]) for row in bbox_rows]
        return {
            "samples": len(values),
            "mean_iou": sum(values) / len(values),
            "accuracy_at_0_5": sum(value >= 0.5 for value in values) / len(values),
            "accuracy_at_0_7": sum(value >= 0.7 for value in values) / len(values),
            "format_validity": sum(
                bool(row["score"].get("format_valid")) for row in bbox_rows
            )
            / len(bbox_rows),
        }

    official_groups: dict[str, Any] = {}
    clean_groups: dict[str, Any] = {}
    errors = sum(bool(row.get("error")) for row in rows)
    truncations = sum(bool(row.get("generation_truncated")) for row in rows)

    for group_name in sorted(groups):
        group_rows = groups[group_name]
        discrete = [
            row
            for row in group_rows
            if row["score"].get("official_correct") is not None
        ]
        official_correct = sum(
            bool(row["score"]["official_correct"]) for row in discrete
        )
        clean_rows = [row for row in discrete if row["score"].get("clean_eligible")]
        clean_correct = sum(
            bool(row["score"].get("clean_correct")) for row in clean_rows
        )
        official_group = {
            "samples": len(group_rows),
            "scoreable_samples": len(discrete),
            "correct": official_correct,
            "accuracy": _accuracy(official_correct, len(discrete)),
            "caption_metrics": caption_metrics(group_rows, clean=False),
        }
        clean_group = {
            "samples": len(group_rows),
            "eligible_scoreable_samples": len(clean_rows),
            "manual_or_excluded": sum(
                row["sample"]["clean_status"] not in ("keep", "corrected") for row in group_rows
            ),
            "correct": clean_correct,
            "accuracy": _accuracy(clean_correct, len(clean_rows)),
            "caption_metrics": caption_metrics(group_rows, clean=True),
        }
        if bbox := bbox_metrics(group_rows, clean=False):
            official_group["bbox_metrics"] = bbox
        if bbox := bbox_metrics(group_rows, clean=True):
            clean_group["bbox_metrics"] = bbox
            # Historical XLRS Grounding experiments use continuous normalized
            # boxes (no integer-coordinate +1 area convention).  Expose that
            # exact report-facing result in both summaries.
            official_group["reported_bbox_metrics"] = bbox
        official_groups[group_name] = official_group
        clean_groups[group_name] = clean_group

    def dataset_summaries(group_summaries: dict[str, Any], *, clean: bool) -> dict:
        output: dict[str, Any] = {}
        dataset_names = sorted({row["sample"]["dataset"] for row in rows})
        for dataset in dataset_names:
            dataset_rows = [row for row in rows if row["sample"]["dataset"] == dataset]
            score_key = "clean_correct" if clean else "official_correct"
            scoreable = [
                row
                for row in dataset_rows
                if row["score"].get(score_key) is not None
                and (not clean or row["score"].get("clean_eligible"))
            ]
            correct = sum(bool(row["score"].get(score_key)) for row in scoreable)
            summary: dict[str, Any] = {
                "samples": len(dataset_rows),
                "scoreable_samples": len(scoreable),
                "correct": correct,
                "micro_accuracy": _accuracy(correct, len(scoreable)),
            }
            if dataset == "xlrs_bench_lite":
                category_scores = [
                    group_summary["accuracy"]
                    for group_name, group_summary in group_summaries.items()
                    if group_name.startswith("xlrs_bench_lite/")
                    and group_summary["accuracy"] is not None
                ]
                summary["macro_accuracy"] = (
                    sum(category_scores) / len(category_scores)
                    if category_scores
                    else None
                )
                summary["macro_categories"] = len(category_scores)
            output[dataset] = summary
        return output

    official_datasets = dataset_summaries(official_groups, clean=False)
    clean_datasets = dataset_summaries(clean_groups, clean=True)
    official_global_total = sum(
        item["scoreable_samples"] for item in official_datasets.values()
    )
    official_global_correct = sum(
        item["correct"] for item in official_datasets.values()
    )
    clean_global_total = sum(
        item["scoreable_samples"] for item in clean_datasets.values()
    )
    clean_global_correct = sum(item["correct"] for item in clean_datasets.values())
    common = {
        "prediction_rows": len(rows),
        "generation_errors": errors,
        "generation_truncations": truncations,
    }
    official = common | {
        "protocol": "source_answer_protocol_with_correct_task_specific_parsers",
        "global_discrete_micro_diagnostic": {
            "correct": official_global_correct,
            "scoreable_samples": official_global_total,
            "accuracy": _accuracy(official_global_correct, official_global_total),
        },
        "datasets": official_datasets,
        "groups": official_groups,
    }
    clean = common | {
        "protocol": "logical_clean_protocol_excluding_manual_review_and_excluded_samples",
        "global_discrete_micro_diagnostic": {
            "correct": clean_global_correct,
            "scoreable_samples": clean_global_total,
            "accuracy": _accuracy(clean_global_correct, clean_global_total),
        },
        "datasets": clean_datasets,
        "groups": clean_groups,
    }
    return official, clean


def prediction_letter_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        sample = row.get("sample", {})
        if sample.get("task_type") not in {"single_choice", "multi_choice"}:
            continue
        parsed = row.get("score", {}).get("parsed_answer")
        allowed = set(sample.get("choices", {}))
        if isinstance(parsed, list) and all(
            isinstance(label, str) and label in allowed for label in parsed
        ):
            counts.update(parsed)
    return dict(sorted(counts.items()))
