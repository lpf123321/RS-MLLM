#!/usr/bin/env python3
"""Collect complete fixed-vocabulary Codex class-screening results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam3_pipeline.common import DATASET_OBJECT_CLASSES


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--dataset-profile", choices=sorted(DATASET_OBJECT_CLASSES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    jobs = read_jsonl(args.jobs.expanduser().resolve())
    root = args.results_dir.expanduser().resolve()
    expected_ids = [str(job["job_id"]) for job in jobs]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("duplicate class job IDs")
    actual = {path.stem for path in root.glob("*.json") if not path.name.startswith(".")}
    if actual != set(expected_ids):
        raise ValueError(
            f"result coverage mismatch: missing={sorted(set(expected_ids)-actual)[:10]} "
            f"extra={sorted(actual-set(expected_ids))[:10]}"
        )
    allowed = set(DATASET_OBJECT_CLASSES[args.dataset_profile])
    rows = []
    for job in jobs:
        result = json.loads((root / f"{job['job_id']}.json").read_text(encoding="utf-8"))
        expected_image = job["payload"]["expected_identity"]["image"]
        if result.get("image") != expected_image:
            raise ValueError(f"{job['job_id']}: image identity changed")
        classes = result.get("classes")
        if not isinstance(classes, list) or len(classes) != len(set(classes)):
            raise ValueError(f"{job['job_id']}: classes must be a unique list")
        if any(value not in allowed for value in classes):
            raise ValueError(f"{job['job_id']}: class outside frozen vocabulary")
        rows.append({"image": expected_image, "classes": classes, "reason": str(result.get("reason", ""))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"images": len(rows), "dataset_profile": args.dataset_profile}, indent=2))


if __name__ == "__main__":
    main()
