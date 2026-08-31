#!/usr/bin/env python3
"""Build Codex language-review jobs after SAM3 boxes are frozen."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_path(value: str, base: Path) -> Path:
    candidate = Path(value).expanduser()
    result = candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def official_text(row: dict[str, Any]) -> str:
    value = row.get("question")
    if value is None:
        conversations = row.get("conversations", [])
        value = conversations[0].get("value", "") if conversations else ""
    return str(value).replace("<image>\n", "", 1).strip()


def reject_declared_non_train(row: dict[str, Any], context: str) -> None:
    split = row.get("split")
    if split is not None and str(split).strip().lower() != "train":
        raise ValueError(f"{context}: refusing declared non-train split {split!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--official-train-records", type=Path)
    parser.add_argument("--dataset-profile", choices=("vrsbench", "xlrs"), required=True)
    parser.add_argument("--split", choices=("train",), required=True)
    parser.add_argument("--boxes-are-frozen-postprocessed", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.boxes_are_frozen_postprocessed:
        raise ValueError("Refusing raw candidates: pass --boxes-are-frozen-postprocessed after SAM3 geometry filtering")

    candidates_path = args.candidates.expanduser().resolve()
    image_root = args.image_root.expanduser().resolve()
    examples: dict[str, list[str]] = defaultdict(list)
    if args.official_train_records:
        for index, row in enumerate(read_jsonl(args.official_train_records.expanduser().resolve())):
            reject_declared_non_train(row, f"official record {index}")
            image = str(row.get("image", ""))
            text = official_text(row)
            if image and text:
                examples[image].append(text)

    jobs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in read_jsonl(candidates_path):
        candidate_id = str(candidate["candidate_id"])
        reject_declared_non_train(candidate, f"candidate {candidate_id}")
        if candidate_id in seen:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        seen.add(candidate_id)
        image_name = str(candidate["image"])
        image_path = resolve_path(image_name, image_root)
        overlay_path = resolve_path(str(candidate["overlay_path"]), candidates_path.parent)
        resolve_path(str(candidate["mask_path"]), candidates_path.parent)
        box = candidate.get("bbox_xyxy_px")
        if not isinstance(box, list) or len(box) != 4 or not all(isinstance(value, (int, float)) for value in box):
            raise ValueError(f"{candidate_id}: invalid frozen bbox_xyxy_px")
        x1, y1, x2, y2 = map(float, box)
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"{candidate_id}: degenerate frozen bbox_xyxy_px")
        style_examples = examples.get(image_name, [])[:8]
        if args.dataset_profile == "xlrs":
            language_rule = (
                "If accepted, language_text is only the inner English Description: 35-100 words, at least "
                "two sentences, the exact class or visible appearance, absolute image position, and at least "
                "two independently visible spatial/adjacency relations. Do not include the outer template."
            )
        else:
            language_rule = (
                "If accepted, language_text is one native VRSBench [refer] question containing exactly one "
                "<p>...</p> target description. It must uniquely identify this instance without coordinates."
            )
        prompt = f"""The first attached image is the unmodified train image. The second is a review overlay for exactly one already frozen SAM3 candidate.

Dataset profile: {args.dataset_profile}
Canonical class: {candidate['class']}
Frozen bbox (context only; never edit, replace, normalize, or return it): {json.dumps(box)}
Official train-only style examples from this image: {json.dumps(style_examples, ensure_ascii=False)}

Accept only if the marked pixels visibly belong to the canonical class, form one coherent instance or coherent same-class group, and can be uniquely described. Reject fragments, background, incoherent merges, or uncertainty. The overlay colors and marks are artificial: never describe them as real appearance and never mention SAM3, box, mask, overlay, or highlighting.

{language_rule}
Return candidate_id exactly as provided. Put the proposed text in language_text. When rejected, language_text must be an empty string. The fixed geometry is deliberately absent from the output schema, so it cannot be modified by this language step.
"""
        jobs.append({
            "job_id": candidate_id,
            "prompt": prompt,
            "images": [str(image_path), str(overlay_path)],
            "payload": {
                "split": "train",
                "dataset_profile": args.dataset_profile,
                "expected_identity": {"candidate_id": candidate_id},
                "frozen_candidate": {
                    "candidate_id": candidate_id,
                    "image": image_name,
                    "class": candidate["class"],
                    "bbox_xyxy_px": box,
                    "image_size": candidate.get("image_size"),
                    "score": candidate.get("score"),
                    "mask_path": candidate["mask_path"],
                    "overlay_path": candidate["overlay_path"],
                },
                "boxes_frozen_before_language": True,
                "source_style_examples": style_examples,
            },
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in jobs),
        encoding="utf-8",
    )
    print(json.dumps({"jobs": len(jobs), "split": "train", "boxes_frozen_before_language": True}, indent=2))


if __name__ == "__main__":
    main()
