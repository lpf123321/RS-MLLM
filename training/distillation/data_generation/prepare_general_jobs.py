#!/usr/bin/env python3
"""Build train-only, annotation-free Codex jobs for general-expert SFT data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROFILE_FIELDS = (
    "profile_id", "source_task_category", "task_type", "language", "option_count",
    "prompt_pattern", "answer_form", "answer_pattern", "lexical_register", "special_rule",
)


def reject_declared_non_train(row: dict[str, Any], context: str) -> None:
    """Fail closed when an input record explicitly identifies a non-train split.

    Some historical manifests predate an explicit ``split`` field, so absence
    cannot be treated as an error.  A present split marker, however, must never
    be allowed to contradict the train-only CLI contract.
    """
    split = row.get("split")
    if split is None and isinstance(row.get("source_record"), dict):
        split = row["source_record"].get("split")
    if split is not None and str(split).strip().lower() != "train":
        raise ValueError(f"{context}: refusing declared non-train split {split!r}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_image(value: str, manifest: Path) -> Path:
    candidate = Path(value).expanduser()
    result = candidate.resolve() if candidate.is_absolute() else (manifest.parent / candidate).resolve()
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def public_source_key(dataset: str, source: dict[str, Any]) -> str:
    value = str(source.get("public_source_key", "")).strip()
    return value or f"{dataset}:train:{source['source_id']}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--question-manifest", type=Path, required=True)
    parser.add_argument("--style-profile", type=Path, required=True)
    parser.add_argument("--split", choices=("train",), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_path = args.source_manifest.expanduser().resolve()
    question_path = args.question_manifest.expanduser().resolve()
    style_path = args.style_profile.expanduser().resolve()
    sources = {str(row["source_id"]): row for row in read_jsonl(source_path)}
    questions = read_jsonl(question_path)
    if len(sources) != len(read_jsonl(source_path)):
        raise ValueError("source_manifest contains duplicate source_id values")
    style_document = json.loads(style_path.read_text(encoding="utf-8"))
    profiles = {
        str(profile["source_task_category"]): {
            key: profile[key] for key in PROFILE_FIELDS if key in profile
        }
        for profile in style_document.get("profiles", [])
    }
    if not profiles:
        raise ValueError("style_profile has no profiles")

    jobs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for slot in questions:
        sample_id = str(slot["sample_id"])
        source_id = str(slot["source_id"])
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        if source_id not in sources:
            raise ValueError(f"question references missing source_id: {source_id}")
        source = sources[source_id]
        reject_declared_non_train(source, f"source {source_id}")
        reject_declared_non_train(slot, f"question {sample_id}")
        dataset = str(slot.get("dataset") or source.get("dataset") or style_document.get("dataset") or "").strip()
        if not dataset:
            raise ValueError(f"{sample_id}: missing dataset")
        category = str(slot["source_task_category"])
        if category not in profiles:
            raise ValueError(f"{sample_id}: no style profile for {category!r}")
        profile = profiles[category]
        images = [str(resolve_image(str(value), source_path)) for value in source.get("original_images", [])]
        if not images:
            raise ValueError(f"{sample_id}: source has no images")
        source_record = source.get("source_record", {})
        dimensions = source_record.get("highres_dimensions") or source_record.get("original_dimensions")
        identity = {
            "dataset": dataset,
            "source_id": source_id,
            "sample_id": sample_id,
            "canonical_source_key": public_source_key(dataset, source),
        }
        prompt = f"""Create exactly one NEW source-native training question and answer by inspecting the attached train image(s).

Required task category: {category}
Required aggregate style profile: {json.dumps(profile, ensure_ascii=False, sort_keys=True)}

Rules:
1. Use only facts directly visible in the full-resolution image(s). The item must be answerable without an unseen crop or hidden annotation.
2. Follow the profile's prompt pattern, option count, answer form, punctuation, terminology, language, and lexical register. For MCQ, include all choices in instruction and match the response format exactly.
3. Select a uniquely describable visible target or region. Do not emit coordinates, bounding boxes, source paths, annotation IDs, or phrases such as highlighted/crop unless such evidence is visibly present in the original image.
4. This must be a newly authored item, not a reconstruction of an official source question. Do not infer any hidden label.
5. Copy the frozen identity fields from the payload exactly. Complete every style_audit field truthfully. Set all review booleans true only after checking the image; a result that cannot pass all checks must not fabricate an answer.
"""
        jobs.append({
            "job_id": sample_id,
            "prompt": prompt,
            "images": images,
            "payload": {
                "split": "train",
                "expected_identity": identity,
                "source_task_category": category,
                "source_l2_category": slot.get("source_l2_category", "default"),
                "image_dimensions": dimensions,
                "style_profile": profile,
                "source_annotations_included": False,
            },
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in jobs),
        encoding="utf-8",
    )
    print(json.dumps({"jobs": len(jobs), "split": "train", "source_annotations_included": False}, indent=2))


if __name__ == "__main__":
    main()
