from __future__ import annotations

import hashlib
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from .answers import build_prompt, split_question_options
from .bbox import area, clip_xyxy, xywh_to_xyxy, xyxy_to_xywh
from .config import workspace
from .io import read_jsonl, stable_id, write_json, write_jsonl


def question_family(question: str) -> str:
    words = re.findall(r"[a-z]+", question.lower())
    if not words:
        return "other"
    if words[0] in {"what", "which", "where", "who", "how", "is", "are", "does", "do"}:
        return " ".join(words[:2])
    return words[0]


def _area_bin(ratio: float) -> int:
    return max(0, min(5, int((math.log10(max(ratio, 1e-8)) + 5.0) // 0.8)))


def _normalize_options(options: Any) -> list[str]:
    if isinstance(options, dict):
        options = [options.get(letter) for letter in "ABCD"]
    if not isinstance(options, list) or len(options) != 4:
        raise ValueError(f"MCQ options must contain exactly A/B/C/D: {options!r}")
    cleaned = []
    for letter, value in zip("ABCD", options):
        text = str(value).strip()
        text = re.sub(rf"^{letter}[.)]\s*", "", text, flags=re.IGNORECASE)
        if not text:
            raise ValueError(f"Empty MCQ option {letter}")
        cleaned.append(text)
    return cleaned


def _normalize_source_item(item: dict[str, Any], root: Path) -> tuple[str, list[str], str, Path, str]:
    if all(key in item for key in ("image_path", "question", "options", "answer")):
        question = str(item["question"]).strip()
        options = _normalize_options(item["options"])
        relative_image = str(item["image_path"])
    elif all(key in item for key in ("problem", "original_images", "answer")):
        question, options = split_question_options(item["problem"])
        relative_image = str(item["original_images"][0])
    else:
        raise ValueError(
            "Each source row must use standard MCQ fields image_path/question/options/answer "
            "or the legacy VOPD problem/original_images/answer schema"
        )
    answer = str(item["answer"]).strip().upper()
    if answer not in "ABCD" or len(answer) != 1:
        raise ValueError(f"Only A/B/C/D answers are supported in schema v1: {answer!r}")
    candidate = Path(relative_image).expanduser()
    image_path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    return question, options, answer, image_path, relative_image


def load_source_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(config["data"]["root"])
    records = []
    seen = set()
    for source_index, item in enumerate(read_jsonl(config["data"]["source_jsonl"])):
        question, options, answer, image_path, relative_image = _normalize_source_item(item, root)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            width, height = image.size
        sample_id = stable_id(relative_image, question, answer)
        if sample_id in seen:
            raise ValueError(f"Duplicate sample id: {sample_id}")
        seen.add(sample_id)
        bbox = clip_xyxy(item["bbox"], width, height) if item.get("bbox") is not None else None
        ratio = area(bbox) / float(width * height) if bbox is not None else None
        record = {
            "sample_id": sample_id,
            "source_index": source_index,
            "image_path": str(image_path),
            "image_relative_path": relative_image,
            "width": width,
            "height": height,
            "question": question,
            "options": options,
            "prompt": build_prompt(question, options),
            "gt_answer": answer,
            "optional_gt_bbox_xyxy": bbox,
            "optional_gt_bbox_xywh": xyxy_to_xywh(bbox) if bbox is not None else None,
            "optional_gt_bbox_area_ratio": ratio,
            "question_family": question_family(question),
            "stratum": question_family(question),
            "source_extra_info": item.get("extra_info", {}),
        }
        records.append(record)
    return records


def stratified_split(records: list[dict[str, Any]], sizes: dict[str, int], seed: int) -> dict[str, list[dict[str, Any]]]:
    if sum(sizes.values()) != len(records):
        raise ValueError(f"Split sizes {sizes} do not sum to {len(records)}")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["stratum"]].append(record)
    ordered = []
    for key in sorted(groups):
        group = groups[key]
        group.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['sample_id']}".encode()).hexdigest())
        ordered.extend(group)
    # Round-robin strata order is then globally re-keyed to hit exact requested counts.
    ordered.sort(key=lambda row: hashlib.sha256(f"split:{seed}:{row['stratum']}:{row['sample_id']}".encode()).hexdigest())
    result = {}
    offset = 0
    for name in ("train", "valid", "test"):
        count = sizes[name]
        result[name] = [dict(row, split=name) for row in ordered[offset:offset + count]]
        offset += count
    return result


def _frozen_split(records: list[dict[str, Any]], manifest_path: str | Path) -> dict[str, list[dict[str, Any]]]:
    manifest = __import__("json").loads(Path(manifest_path).read_text(encoding="utf-8"))
    by_index = {int(row["source_index"]): row for row in records}
    result: dict[str, list[dict[str, Any]]] = {}
    used: set[int] = set()
    for split in ("train", "valid", "test"):
        indices = [int(value) for value in manifest["splits"][split]]
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate source indices in frozen {split} split")
        missing = [index for index in indices if index not in by_index]
        if missing:
            raise ValueError(f"Frozen {split} split references missing source indices: {missing[:5]}")
        overlap = used.intersection(indices)
        if overlap:
            raise ValueError(f"Frozen splits overlap at source indices: {sorted(overlap)[:5]}")
        used.update(indices)
        result[split] = [dict(by_index[index], split=split) for index in indices]
    if len(used) != len(records):
        raise ValueError(f"Frozen splits cover {len(used)} of {len(records)} source records")
    return result


def prepare(config: dict[str, Any]) -> dict[str, Any]:
    root = workspace(config)
    records = load_source_records(config)
    data_cfg = config["data"]
    frozen_manifest = config.get("validated_sam_method", {}).get("split_manifest")
    if frozen_manifest:
        splits = _frozen_split(records, frozen_manifest)
        train_limit = data_cfg.get("train_limit")
        if train_limit is not None:
            splits["train"] = splits["train"][:int(train_limit)]
    else:
        splits = stratified_split(records, {
            "train": int(data_cfg["train_size"]),
            "valid": int(data_cfg["valid_size"]),
            "test": int(data_cfg["test_size"]),
        }, int(config["seed"]))
    actual_sizes = {name: len(rows) for name, rows in splits.items()}
    expected_sizes = {name: int(data_cfg[f"{name}_size"]) for name in ("train", "valid", "test")}
    if actual_sizes != expected_sizes:
        raise ValueError(f"Frozen split sizes {actual_sizes} do not match configured sizes {expected_sizes}")
    split_root = root / "data" / "splits"
    for split, rows in splits.items():
        write_jsonl(split_root / f"{split}.jsonl", rows)
    smoke = {}
    smoke_anchor_ids = data_cfg.get("smoke_anchor_ids", {})
    for split in ("train", "valid"):
        size = int(data_cfg[f"smoke_{split}_size"])
        anchors = [str(value) for value in smoke_anchor_ids.get(split, [])]
        by_id = {str(row["sample_id"]): row for row in splits[split]}
        missing = [sample_id for sample_id in anchors if sample_id not in by_id]
        if missing:
            raise ValueError(f"Smoke {split} anchors are outside the frozen split: {missing}")
        if len(anchors) != len(set(anchors)):
            raise ValueError(f"Duplicate smoke {split} anchors")
        selected = [by_id[sample_id] for sample_id in anchors]
        selected.extend(
            row for row in splits[split]
            if str(row["sample_id"]) not in set(anchors)
        )
        smoke[split] = selected[:size]
    for split, rows in smoke.items():
        write_jsonl(split_root / f"smoke_{split}.jsonl", rows)
    summary = {
        "total": len(records),
        "split_counts": {key: len(value) for key, value in splits.items()},
        "smoke_counts": {key: len(value) for key, value in smoke.items()},
        "smoke_anchor_ids": smoke_anchor_ids,
        "question_families": dict(sorted(Counter(row["question_family"] for row in records).items())),
        "rows_with_optional_gt_bbox": sum(row["optional_gt_bbox_xyxy"] is not None for row in records),
        "bbox_roundtrip_exact": all(
            row["optional_gt_bbox_xyxy"] is None
            or [float(v) for v in row["optional_gt_bbox_xyxy"]] == xywh_to_xyxy(row["optional_gt_bbox_xywh"])
            for row in records
        ),
        "seed": int(config["seed"]),
        "split_source": str(Path(frozen_manifest).resolve()) if frozen_manifest else "generated_stratified_split",
    }
    write_json(root / "manifests" / "data_summary.json", summary)
    return summary
