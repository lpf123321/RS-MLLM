#!/usr/bin/env python3
"""Convert finalized XLRS SAM3 examples to the fixed-five trainer schema."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DIMENSIONS_RE = re.compile(r"Given a ([0-9]+) x ([0-9]+) pixel")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    accepted, collapsed = [], []
    for index, line in enumerate(args.synthetic_jsonl.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        question = row["messages"][0]["content"]
        match = DIMENSIONS_RE.search(question)
        if not match:
            raise ValueError(f"row {index}: missing original dimensions")
        width, height = map(int, match.groups())
        box = json.loads(row["messages"][1]["content"])
        if not (
            isinstance(box, list) and len(box) == 4
            and 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height
        ):
            raise ValueError(f"row {index}: invalid pixel box {box}")
        normalized = tuple(round(value / scale, 5) for value, scale in zip(box, (width, height, width, height)))
        if normalized[0] >= normalized[2] or normalized[1] >= normalized[3]:
            collapsed.append({"id": row["id"], "pixel_box": box, "rounded": normalized})
            continue
        answer = "[" + ",".join(f"{value:.5f}" for value in normalized) + "]"
        accepted.append({
            "id": row["id"], "image": [str(Path(row["image"]).resolve())],
            "conversations": [
                {"from": "human", "value": question},
                {"from": "gpt", "value": answer},
            ],
            "loss_weight": 1.0, "weight_group": "sam3_xlrs_fixed5",
            "source_candidate_id": row["candidate_id"], "source_class": row["class"],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(accepted, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {
        "source_records": len(accepted) + len(collapsed), "accepted": len(accepted),
        "collapsed_after_fixed5": collapsed, "coordinate_format": "normalized fixed-five",
        "test_split_opened": False,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
