#!/usr/bin/env python3
"""Convert the formal SFT subset to Vision-OPD/VERL parquet."""

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
OUTPUT = ARTIFACT_ROOT / "opd" / "data" / "formal_train_18413.parquet"
MANIFEST = PIPELINE_ROOT / "manifests" / "opd_formal_manifest.json"

sys.path.insert(0, str(PIPELINE_ROOT))
from prompts import SYSTEM_PROMPTS  # noqa: E402


def task_stratum(record: dict[str, Any]) -> str:
    prompt = record["conversations"][0]["value"]
    for prefix in ("[CAP]", "[REF]", "[VQA]", "[MCQ]"):
        if prefix in prompt:
            return prefix
    return "other"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    records = json.loads(SOURCE.read_text(encoding="utf-8"))
    if len(records) != 18_413:
        raise AssertionError(f"Expected 18,413 formal SFT records, got {len(records)}")
    ids = [record["id"] for record in records]
    if len(set(ids)) != len(ids):
        raise AssertionError("Formal OPD source contains duplicate record ids")

    rows = []
    counts: Counter[str] = Counter()
    image_count = 0
    for record in records:
        dataset = record["source_dataset"]
        if dataset not in SYSTEM_PROMPTS:
            raise KeyError(f"No system prompt for dataset {dataset!r}")
        images = record["image"] if isinstance(record["image"], list) else [record["image"]]
        missing = [image for image in images if not Path(image).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing images for {record['id']}: {missing}")

        prompt = record["conversations"][0]["value"]
        answer = record["conversations"][1]["value"]
        image_structs = [{"path": image} for image in images]
        rows.append(
            {
                "data_source": "opd_formal_reward_free",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPTS[dataset]},
                    {"role": "user", "content": prompt},
                ],
                "images": image_structs,
                # The datasets do not provide a separate teacher crop.  Both
                # models therefore receive the same source image(s).
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

    manifest = {
        "schema_version": 1,
        "source": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "output": str(OUTPUT),
        "output_sha256": sha256(OUTPUT),
        "records": len(rows),
        "unique_ids": len(set(ids)),
        "image_references": image_count,
        "counts": dict(sorted(counts.items())),
        "order_policy": "identical to unified_sft.json",
        "teacher_image_policy": "same source images as student (no separate teacher crops)",
        "first_ids": ids[:5],
        "last_ids": ids[-5:],
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
