#!/usr/bin/env python3
"""Prepare all available training records for the full Vision-OPD-9B SFT run."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


ROOT = Path(os.environ.get("PIPELINE_ROOT", Path(__file__).resolve().parents[1]))
OUTPUT = ROOT / "data" / "full_train" / "unified_sft_full.json"
MANIFEST = ROOT / "manifests" / "full_sft_manifest.json"
EXCLUSIONS = ROOT / "manifests" / "full_sft_exclusions.json"
SEED = 20260809

sys.path.insert(0, str(ROOT / "scripts"))
from prepare_subsets import (  # noqa: E402
    SOURCES,
    group_records,
    image_paths,
    load_jsonl,
    sha256_file,
    stable_score,
    task_name,
    to_llava,
)


TRAIN_SOURCES = {
    # Preserve official test/validation isolation where official splits exist.
    "vrsbench": "vrs_train",
    "levircc": "levir_train",
    # MME and XLRS are benchmark-only sources without an official train split;
    # the user's full-data request therefore includes every available record.
    "mme": "mme",
    "xlrs": "xlrs",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_image(path: str) -> tuple[str, str | None]:
    """Return a PIL decoding error for an unusable image, otherwise None."""
    if not os.path.isfile(path):
        return path, "missing file"
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as error:  # Preserve the decoder's exact failure in the audit.
        return path, f"{type(error).__name__}: {error}"
    return path, None


def main() -> None:
    loaded = {dataset: load_jsonl(source) for dataset, source in TRAIN_SOURCES.items()}
    source_counts = {dataset: len(records) for dataset, records in loaded.items()}
    expected = {
        "vrsbench": 142_390,
        "levircc": 34_075,
        "mme": 3_736,
        "xlrs": 3_080,
    }
    if source_counts != expected:
        raise AssertionError(f"Unexpected full-data source counts: {source_counts}")

    unique_images = sorted(
        {image for records in loaded.values() for record in records for image in image_paths(record)}
    )
    with ThreadPoolExecutor(max_workers=32) as executor:
        checked_images = executor.map(validate_image, unique_images)
    image_errors = {path: error for path, error in checked_images if error is not None}

    exclusions = []
    task_counts: Counter[str] = Counter()
    image_groups = {}
    included_record_counts: Counter[str] = Counter()
    unified = []
    for dataset, records in loaded.items():
        included_records = []
        for record in records:
            bad_images = [image for image in image_paths(record) if image in image_errors]
            if bad_images:
                exclusions.append(
                    {
                        "id": f"{dataset}:{record['sample_metadata']['source_index']}",
                        "dataset": dataset,
                        "source_index": record["sample_metadata"]["source_index"],
                        "images": bad_images,
                        "errors": {image: image_errors[image] for image in bad_images},
                    }
                )
                continue
            included_records.append(record)
            included_record_counts[dataset] += 1
            task_counts[f"{dataset}/{task_name(record)}"] += 1
            unified.append(to_llava(record, dataset))
        image_groups[dataset] = len(group_records(included_records))

    ids = [record["id"] for record in unified]
    if len(unified) + len(exclusions) != 183_281 or len(set(ids)) != len(ids):
        raise AssertionError(
            "Expected 183,281 included + excluded full-SFT examples, got "
            f"{len(unified)} included / {len(exclusions)} excluded / {len(set(ids))} unique ids"
        )
    unified.sort(key=lambda item: stable_score(SEED, "full_unified_sft", item["id"]))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as stream:
        json.dump(unified, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")

    manifest = {
        "schema_version": 2,
        "purpose": "Vision-OPD-9B LoRA SFT on all available training records",
        "seed": SEED,
        "output": str(OUTPUT),
        "output_sha256": sha256(OUTPUT),
        "records": len(unified),
        "unique_ids": len(set(ids)),
        "source_record_counts": source_counts,
        "included_record_counts": dict(included_record_counts),
        "source_image_groups": image_groups,
        "task_counts": dict(sorted(task_counts.items())),
        "validated_unique_images": len(unique_images),
        "unusable_images": len(image_errors),
        "excluded_records": len(exclusions),
        "exclusions_manifest": str(EXCLUSIONS),
        "source_files": {
            dataset: {
                "path": str(SOURCES[source]),
                "sha256": sha256_file(SOURCES[source]),
            }
            for dataset, source in TRAIN_SOURCES.items()
        },
        "split_policy": {
            "vrsbench": "all records from official train split; official eval excluded",
            "levircc": "all records from official train split; official val/test excluded",
            "mme": "all benchmark records (no official train split)",
            "xlrs": "all benchmark records (no official train split)",
        },
        "order_policy": "sha256(seed, full_unified_sft, record id)",
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    EXCLUSIONS.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "reason": "Records excluded because at least one referenced image failed PIL verification",
                "unusable_images": image_errors,
                "records": exclusions,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
