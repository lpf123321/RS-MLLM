#!/usr/bin/env python3
"""Prepare train-image class-screening jobs for the repository SAM3 pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sam3_pipeline.common import DATASET_OBJECT_CLASSES


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def safe_job_id(image: str) -> str:
    stem = "".join(char if char.isalnum() or char in "._-" else "_" for char in Path(image).name)
    suffix = hashlib.sha256(image.encode()).hexdigest()[:10]
    return f"class_{stem}_{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--dataset-profile", choices=sorted(DATASET_OBJECT_CLASSES), required=True)
    parser.add_argument("--split", choices=("train",), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    allowed = list(DATASET_OBJECT_CLASSES[args.dataset_profile])
    image_root = args.image_root.expanduser().resolve()
    jobs = []
    seen = set()
    for row in read_jsonl(args.selection.expanduser().resolve()):
        if row.get("split") not in (None, "train"):
            raise ValueError(f"refusing declared non-train selection: {row.get('split')!r}")
        image = str(row["image"])
        if image in seen:
            raise ValueError(f"duplicate selected image: {image}")
        seen.add(image)
        image_path = (image_root / image).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        jobs.append({
            "job_id": safe_job_id(image),
            "images": [str(image_path)],
            "prompt": (
                "Inspect this unmodified train image and select every canonical class that is visibly "
                "likely to occur. Select only from this frozen vocabulary: " + json.dumps(allowed) +
                ". An empty list is valid. Do not infer hidden annotations, boxes, masks, or test labels. "
                "Copy the image identity exactly and give a short visual reason."
            ),
            "payload": {
                "split": "train", "dataset_profile": args.dataset_profile,
                "allowed_classes": allowed, "expected_identity": {"image": image},
            },
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in jobs), encoding="utf-8")
    print(json.dumps({"jobs": len(jobs), "dataset_profile": args.dataset_profile, "split": "train"}, indent=2))


if __name__ == "__main__":
    main()
