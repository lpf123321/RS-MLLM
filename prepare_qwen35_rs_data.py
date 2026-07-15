#!/usr/bin/env python3
"""Build deterministic ms-swift SFT splits from the local RS datasets."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "datasets/shared_datasets"
PATH_PREFIXES = (
    "/users/u2024311136/shared/shared_datasets",
    "/users/u2024311136/datasets/shared_datasets",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vrs-cap", type=int, default=1100)
    parser.add_argument("--vrs-vqa", type=int, default=1100)
    parser.add_argument("--levir", type=int, default=1100)
    parser.add_argument("--val-per-source", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "finetune_data/qwen35_rs_pilot_v2",
    )
    return parser.parse_args()


def normalize_image_path(raw: str) -> str:
    for prefix in PATH_PREFIXES:
        if raw.startswith(prefix):
            return str(DATA_ROOT) + raw[len(prefix) :]
    return raw


def content_to_text(content: Any, images: list[str]) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError(f"Unsupported message content type: {type(content)!r}")

    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            raise ValueError(f"Unsupported content item: {item!r}")
        item_type = item.get("type")
        if item_type == "image":
            image = normalize_image_path(str(item["image"]))
            images.append(image)
            parts.append("<image>")
        elif item_type == "text":
            parts.append(str(item.get("text", "")))
        else:
            raise ValueError(f"Unsupported content item type: {item_type!r}")
    return "".join(parts)


def convert_record(record: dict[str, Any]) -> dict[str, Any]:
    images: list[str] = []
    messages = []
    for message in record["messages"]:
        role = str(message["role"])
        text = content_to_text(message["content"], images)
        messages.append({"role": role, "content": text})

    missing = [path for path in images if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing image(s): {missing[:3]}")
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("Every SFT record must end with an assistant response")
    return {"messages": messages, "images": images}


def reservoir_add(
    buckets: dict[str, list[dict[str, Any]]],
    seen: Counter[str],
    key: str,
    item: dict[str, Any],
    limit: int,
    rng: random.Random,
) -> None:
    seen[key] += 1
    bucket = buckets[key]
    if len(bucket) < limit:
        bucket.append(item)
        return
    index = rng.randrange(seen[key])
    if index < limit:
        bucket[index] = item


def load_vrs(
    path: Path,
    limits: dict[str, int],
    rng: random.Random,
) -> dict[str, list[dict[str, Any]]]:
    buckets = {"vrs_cap": [], "vrs_vqa": []}
    seen: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            user_content = record["messages"][0]["content"]
            prompt = "".join(
                str(item.get("text", ""))
                for item in user_content
                if isinstance(item, dict) and item.get("type") == "text"
            )
            if prompt.startswith("[CAP]"):
                key = "vrs_cap"
            elif prompt.startswith("[VQA]"):
                key = "vrs_vqa"
            else:
                continue
            try:
                converted = convert_record(record)
            except (FileNotFoundError, ValueError) as exc:
                raise RuntimeError(f"{path}:{line_number}: {exc}") from exc
            reservoir_add(buckets, seen, key, converted, limits[key], rng)
    return buckets


def load_levir(path: Path, limit: int, rng: random.Random) -> list[dict[str, Any]]:
    bucket: list[dict[str, Any]] = []
    seen_count = 0
    seen_pairs: set[tuple[str, ...]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                converted = convert_record(record)
            except (FileNotFoundError, ValueError) as exc:
                raise RuntimeError(f"{path}:{line_number}: {exc}") from exc
            image_pair = tuple(converted["images"])
            if image_pair in seen_pairs:
                continue
            seen_pairs.add(image_pair)
            seen_count += 1
            if len(bucket) < limit:
                bucket.append(converted)
                continue
            index = rng.randrange(seen_count)
            if index < limit:
                bucket[index] = converted
    return bucket


def split_buckets_image_disjoint(
    buckets: dict[str, list[dict[str, Any]]],
    val_per_source: int,
    rng: random.Random,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    image_sources: dict[str, set[str]] = {}
    for source, rows in buckets.items():
        for row in rows:
            for image in row["images"]:
                image_sources.setdefault(image, set()).add(source)

    train_by_source: dict[str, list[dict[str, Any]]] = {}
    validation_by_source: dict[str, list[dict[str, Any]]] = {}
    for source, rows in buckets.items():
        groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(tuple(row["images"]), []).append(row)
        eligible = [
            group
            for images, group in groups.items()
            if all(image_sources[image] == {source} for image in images)
        ]
        rng.shuffle(eligible)

        reachable: dict[int, list[list[dict[str, Any]]]] = {0: []}
        for group in eligible:
            group_size = len(group)
            for total, selected in list(reachable.items()):
                new_total = total + group_size
                if new_total <= val_per_source and new_total not in reachable:
                    reachable[new_total] = [*selected, group]
            if val_per_source in reachable:
                break
        if val_per_source not in reachable:
            raise RuntimeError(
                f"Cannot build an image-disjoint validation split of "
                f"{val_per_source} rows for {source}"
            )

        validation_groups = reachable[val_per_source]
        validation_keys = {tuple(group[0]["images"]) for group in validation_groups}
        validation_by_source[source] = [row for group in validation_groups for row in group]
        train_by_source[source] = [
            row for row in rows if tuple(row["images"]) not in validation_keys
        ]

    train_images = {
        image
        for rows in train_by_source.values()
        for row in rows
        for image in row["images"]
    }
    validation_images = {
        image
        for rows in validation_by_source.values()
        for row in rows
        for image in row["images"]
    }
    overlap = train_images & validation_images
    if overlap:
        raise RuntimeError(f"Train/validation image leakage: {sorted(overlap)[:3]}")
    return train_by_source, validation_by_source


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.val_per_source < 1:
        raise ValueError("--val-per-source must be positive")
    requested = {
        "vrs_cap": args.vrs_cap,
        "vrs_vqa": args.vrs_vqa,
        "levir": args.levir,
    }
    if any(count <= args.val_per_source for count in requested.values()):
        raise ValueError("Each source count must exceed --val-per-source")

    rng = random.Random(args.seed)
    vrs = load_vrs(
        DATA_ROOT / "VRSBench/vrsbench_train.jsonl",
        {"vrs_cap": args.vrs_cap, "vrs_vqa": args.vrs_vqa},
        rng,
    )
    buckets = {
        **vrs,
        "levir": load_levir(
            DATA_ROOT / "LEVIR-CC/levircc_train.jsonl", args.levir, rng
        ),
    }
    actual = {key: len(rows) for key, rows in buckets.items()}
    if actual != requested:
        raise RuntimeError(f"Insufficient source records: requested={requested}, actual={actual}")

    train_by_source, validation_by_source = split_buckets_image_disjoint(
        buckets, args.val_per_source, rng
    )
    train = [row for rows in train_by_source.values() for row in rows]
    validation = [row for rows in validation_by_source.values() for row in rows]
    rng.shuffle(train)
    rng.shuffle(validation)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    train_path = args.output_dir / "train.jsonl"
    val_path = args.output_dir / "validation.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(val_path, validation)
    summary = {
        "seed": args.seed,
        "requested_by_source": requested,
        "train_by_source": {key: len(rows) for key, rows in train_by_source.items()},
        "validation_by_source": {
            key: len(rows) for key, rows in validation_by_source.items()
        },
        "train_validation_image_overlap": 0,
        "train_total": len(train),
        "validation_total": len(validation),
        "vrs_ref_excluded": True,
        "xlrs_and_mme_excluded_as_evaluation_only": True,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
