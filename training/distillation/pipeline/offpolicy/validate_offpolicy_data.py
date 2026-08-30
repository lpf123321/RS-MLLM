#!/usr/bin/env python3
"""Fast integrity audit for the formal off-policy parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--expected-records", type=int, default=18_413)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    table = pq.read_table(args.parquet)
    rows = table.to_pylist()
    ids = [row["trajectory_id"] for row in rows]
    image_mismatches = sum(row["images"] != row["bbox_images"] for row in rows)
    empty_responses = sum(not row["teacher_response_ids"] for row in rows)
    oversized_responses = sum(len(row["teacher_response_ids"]) > 64 for row in rows)
    report = {
        "status": "PASS",
        "parquet": str(args.parquet.resolve()),
        "records": len(rows),
        "unique_trajectory_ids": len(set(ids)),
        "image_mismatches": image_mismatches,
        "empty_responses": empty_responses,
        "oversized_responses": oversized_responses,
    }
    if (
        len(rows) != args.expected_records
        or len(set(ids)) != args.expected_records
        or image_mismatches
        or empty_responses
        or oversized_responses
    ):
        report["status"] = "FAIL"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
