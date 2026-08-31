#!/usr/bin/env python3
"""Build a deterministic, stratified 512-example OPD framework demo set."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


PIPELINE_ROOT = Path(os.environ.get("PIPELINE_ROOT", Path(__file__).resolve().parents[1]))
ARTIFACT_ROOT = Path(os.environ.get("ARTIFACT_ROOT", PIPELINE_ROOT / "artifacts"))
SOURCE = PIPELINE_ROOT / "data" / "train" / "unified_sft.json"
OUTPUT = ARTIFACT_ROOT / "opd" / "data" / "demo_train_512.parquet"
MANIFEST = PIPELINE_ROOT / "manifests" / "opd_demo_512_manifest.json"
SEED = 20260809
QUOTAS = {
    ("vrsbench", "[VQA]"): 160,
    ("vrsbench", "[REF]"): 96,
    ("vrsbench", "[CAP]"): 64,
    ("levircc", "other"): 96,
    ("mme", "[MCQ]"): 48,
    ("xlrs", "[MCQ]"): 48,
}

sys.path.insert(0, str(PIPELINE_ROOT))
from prompts import SYSTEM_PROMPTS  # noqa: E402


def task_stratum(record: dict[str, Any]) -> str:
    prompt = record["conversations"][0]["value"]
    for prefix in ("[CAP]", "[REF]", "[VQA]", "[MCQ]"):
        if prefix in prompt:
            return prefix
    return "other"


def stable_key(record: dict[str, Any]) -> bytes:
    return hashlib.sha256(f"{SEED}:{record['id']}".encode()).digest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    records = json.loads(SOURCE.read_text(encoding="utf-8"))
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {
        key: [] for key in QUOTAS
    }
    for record in records:
        key = (record["source_dataset"], task_stratum(record))
        if key in buckets:
            buckets[key].append(record)

    selected: list[dict[str, Any]] = []
    for key, quota in QUOTAS.items():
        candidates = sorted(buckets[key], key=stable_key)
        if len(candidates) < quota:
            raise AssertionError(f"Not enough examples for {key}: {len(candidates)} < {quota}")
        selected.extend(candidates[:quota])

    # Reorder the selected records by the same deterministic hash so every
    # training batch sees a mixture of tasks before VERL's seeded shuffle.
    selected.sort(key=stable_key)
    ids = [record["id"] for record in selected]
    if len(selected) != 512 or len(set(ids)) != 512:
        raise AssertionError("Demo selection must contain exactly 512 unique examples")

    rows = []
    counts: Counter[str] = Counter()
    image_count = 0
    for record in selected:
        dataset = record["source_dataset"]
        images = record["image"] if isinstance(record["image"], list) else [record["image"]]
        missing = [image for image in images if not Path(image).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing images for {record['id']}: {missing}")

        prompt = record["conversations"][0]["value"]
        answer = record["conversations"][1]["value"]
        image_structs = [{"path": image} for image in images]
        rows.append(
            {
                "data_source": "opd_demo_reward_free",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPTS[dataset]},
                    {"role": "user", "content": prompt},
                ],
                "images": image_structs,
                "bbox_images": image_structs,
                "ability": dataset,
                "reward_model": {"ground_truth": answer, "style": "none"},
                "extra_info": {
                    "answer": answer,
                    "question": prompt,
                    "source_extra_info": {"answer": answer, "question": prompt},
                },
            }
        )
        counts[f"{dataset}/{task_stratum(record)}"] += 1
        image_count += len(images)

    message = pa.struct([("content", pa.string()), ("role", pa.string())])
    image = pa.struct([("path", pa.string())])
    source_extra = pa.struct([("answer", pa.string()), ("question", pa.string())])
    schema = pa.schema(
        [
            ("data_source", pa.string()),
            ("prompt", pa.list_(message)),
            ("images", pa.list_(image)),
            ("bbox_images", pa.list_(image)),
            ("ability", pa.string()),
            ("reward_model", pa.struct([("ground_truth", pa.string()), ("style", pa.string())])),
            (
                "extra_info",
                pa.struct(
                    [
                        ("answer", pa.string()),
                        ("question", pa.string()),
                        ("source_extra_info", source_extra),
                    ]
                ),
            ),
        ]
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), OUTPUT, compression="zstd")

    expected_counts = {f"{dataset}/{task}": count for (dataset, task), count in QUOTAS.items()}
    if dict(counts) != expected_counts:
        raise AssertionError(f"Unexpected selection counts: {dict(counts)}")
    manifest = {
        "schema_version": 1,
        "purpose": "small OPD framework effectiveness test",
        "seed": SEED,
        "source": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "output": str(OUTPUT),
        "output_sha256": sha256(OUTPUT),
        "records": len(rows),
        "unique_ids": len(set(ids)),
        "image_references": image_count,
        "counts": dict(sorted(counts.items())),
        "selection_policy": "lowest sha256(seed:record_id) per dataset/task stratum",
        "teacher_image_policy": "same source images as student (no separate teacher crops)",
        "selected_ids": ids,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "selected_ids"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
