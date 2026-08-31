#!/usr/bin/env python3
"""Collect complete Codex results without changing frozen job data or geometry."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


FORBIDDEN_GROUNDING_META = (
    "bounding box", "bbox", "mask", "highlighted", "overlay", "sam3",
)

STYLE_IDENTITY_FIELDS = (
    "profile_id", "source_task_category", "task_type", "language",
    "answer_form", "prompt_pattern", "answer_pattern", "lexical_register",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def enforce_identity(job: dict[str, Any], result: dict[str, Any]) -> None:
    expected = job.get("payload", {}).get("expected_identity", {})
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(
                f"{job['job_id']}: frozen identity mismatch for {key}: "
                f"expected {value!r}, got {result.get(key)!r}"
            )


def answer_letters(value: str) -> str:
    match = re.match(r"^([A-E]+)(?:\.|\s|$)", value.strip())
    if not match:
        raise ValueError(f"cannot parse source-native MCQ answer: {value!r}")
    return match.group(1)


def native_option_labels(instruction: str, expected_count: int | None) -> list[str]:
    """Read the option layouts used by the retained MME/XLRS profiles.

    Official-style prompts occur both as ``(A) one (B) two`` on one line and
    as ``A. (A) one``/``A. one`` on separate lines.  Prefer a candidate that
    exactly matches the frozen option count and order.
    """
    candidates = [
        re.findall(r"\(([A-E])\)\s*", instruction),
        re.findall(r"(?m)^\s*([A-E])\.\s+", instruction),
        re.findall(r"(?<![A-Za-z0-9])([A-E])\.\s+", instruction),
    ]
    if expected_count is not None:
        expected = [chr(ord("A") + index) for index in range(expected_count)]
        for labels in candidates:
            if labels == expected:
                return labels
        return max(candidates, key=len)
    return max(candidates, key=len)


def validate_general_style_and_options(job: dict[str, Any], result: dict[str, Any]) -> str:
    """Validate model claims against the frozen profile and native MCQ shape."""
    profile = job.get("payload", {}).get("style_profile", {})
    audit = result.get("style_audit", {})
    for key in STYLE_IDENTITY_FIELDS:
        if key in profile and audit.get(key) != profile[key]:
            raise ValueError(
                f"{job['job_id']}: style audit changed frozen {key}: "
                f"expected {profile[key]!r}, got {audit.get(key)!r}"
            )
    instruction = str(result["instruction"])
    expected_count = profile.get("option_count")
    labels = native_option_labels(
        instruction, int(expected_count) if expected_count is not None else None
    )
    if expected_count is not None:
        expected_labels = [chr(ord("A") + index) for index in range(int(expected_count))]
        if labels != expected_labels:
            raise ValueError(
                f"{job['job_id']}: expected native choices {expected_labels}, got {labels}"
            )
    letters = answer_letters(str(result["response"]))
    if labels and any(letter not in labels for letter in letters):
        raise ValueError(f"{job['job_id']}: answer {letters!r} is outside choices {labels}")
    return letters


def collect_general(job: dict[str, Any], result: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    review = result["direct_session_review"]
    passed = review.get("mode") == "direct_session" and all(
        review.get(key) is True
        for key in (
            "full_image_answerable", "answer_factually_supported", "answer_complete",
            "style_matched", "no_annotation_leakage",
        )
    )
    if not passed:
        return False, {"job_id": job["job_id"], "reason": "direct_session_review_failed", "result": result}
    letters = validate_general_style_and_options(job, result)
    images = list(job.get("images", []))
    prompt = "<image>\n" * len(images) + result["instruction"]
    row = {
        "id": str(result["sample_id"]),
        "image": images,
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": letters},
        ],
        "generation_audit": {
            "dataset": result["dataset"],
            "source_id": result["source_id"],
            "canonical_source_key": result["canonical_source_key"],
            "style_audit": result["style_audit"],
            "direct_session_review": review,
            "provenance": result.get("_provenance", {}),
        },
    }
    return True, row


def validate_xlrs_description(value: str) -> str | None:
    words = re.findall(r"[A-Za-z0-9]+", value)
    if not 35 <= len(words) <= 100:
        return f"XLRS description has {len(words)} words; expected 35-100"
    sentences = [part for part in re.split(r"[.!?]+", value) if part.strip()]
    if len(sentences) < 2:
        return "XLRS description needs at least two sentences"
    groups = (
        r"\b(?:above|below|upper|lower|top|bottom)\b",
        r"\b(?:left|right|west|east)\b",
        r"\b(?:beside|adjacent|near|next to|between|surrounded|immediately)\b",
        r"\b(?:inside|within|along|through|edge|corner|center|centre)\b",
    )
    if sum(bool(re.search(pattern, value, flags=re.I)) for pattern in groups) < 2:
        return "XLRS description lacks two independent relation types"
    return None


def collect_grounding(job: dict[str, Any], result: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    candidate_id = str(result["candidate_id"])
    if not result["accepted"]:
        if str(result.get("language_text", "")):
            return False, {
                "candidate_id": candidate_id, "status": "rejected",
                "reason": "rejected_result_has_language",
            }
        return False, {"candidate_id": candidate_id, "status": "rejected", "reason": result["reason"]}
    if not all(result[key] is True for key in (
        "box_visually_supported", "description_unique", "source_style_matched"
    )):
        return False, {"candidate_id": candidate_id, "status": "rejected", "reason": "semantic_gate_failed"}
    text = str(result["language_text"]).strip()
    lowered = text.lower()
    if not text or any(token in lowered for token in FORBIDDEN_GROUNDING_META):
        return False, {"candidate_id": candidate_id, "status": "rejected", "reason": "language_meta_leak"}
    profile = job.get("payload", {}).get("dataset_profile")
    if profile == "xlrs":
        error = validate_xlrs_description(text)
        if error:
            return False, {"candidate_id": candidate_id, "status": "rejected", "reason": error}
        accepted = {
            "candidate_id": candidate_id, "status": "accepted", "description": text,
            "reason": result["reason"], "_provenance": result.get("_provenance", {}),
        }
    elif profile == "vrsbench":
        if "[refer]" not in lowered or text.count("<p>") != 1 or text.count("</p>") != 1:
            return False, {"candidate_id": candidate_id, "status": "rejected", "reason": "non_native_vrsbench_question"}
        if not text.startswith("<image>\n"):
            text = "<image>\n" + text
        accepted = {
            "candidate_id": candidate_id, "status": "accepted", "question": text,
            "reason": result["reason"], "_provenance": result.get("_provenance", {}),
        }
    else:
        raise ValueError(f"{candidate_id}: unknown dataset_profile {profile!r}")
    return True, accepted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--kind", choices=("general", "grounding"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    jobs = read_jsonl(args.jobs.expanduser().resolve())
    result_root = args.results_dir.expanduser().resolve()
    expected_ids = [str(job["job_id"]) for job in jobs]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("jobs contain duplicate job_id values")
    actual = {path.stem for path in result_root.glob("*.json") if not path.name.startswith(".")}
    missing = sorted(set(expected_ids) - actual)
    extra = sorted(actual - set(expected_ids))
    if missing or extra:
        raise ValueError(f"result coverage mismatch: missing={missing[:10]} extra={extra[:10]}")

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for job in jobs:
        result = json.loads((result_root / f"{job['job_id']}.json").read_text(encoding="utf-8"))
        enforce_identity(job, result)
        keep, row = collect_general(job, result) if args.kind == "general" else collect_grounding(job, result)
        (accepted if keep else rejected).append(row)

    output = args.output.expanduser().resolve()
    write_jsonl(output, accepted)
    rejected_path = output.with_name(output.stem + ".rejected.jsonl")
    write_jsonl(rejected_path, rejected)
    summary = {
        "kind": args.kind,
        "jobs": len(jobs),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "complete_coverage": True,
        "boxes_modified_by_language_step": 0 if args.kind == "grounding" else None,
    }
    output.with_name(output.stem + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
