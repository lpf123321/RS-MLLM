#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

try:
    from .common import image_name, question_answer, read_rows, write_jsonl
except ImportError:  # direct script execution
    from common import image_name, question_answer, read_rows, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--exclude-selection", action="append", default=[],
                        help="Optional selected_images.jsonl files whose image names must be excluded")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    train, _ = read_rows(args.train)
    grouped = defaultdict(list)
    for row in train:
        grouped[image_name(row)].append(row)
    excluded_images = set()
    for exclude_path in args.exclude_selection:
        for line in Path(exclude_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                excluded_images.add(str(json.loads(line)["image"]))
    eligible = sorted(set(grouped) - excluded_images)
    rng = random.Random(args.seed)
    chosen = eligible if args.num_images <= 0 or args.num_images >= len(eligible) else rng.sample(eligible, args.num_images)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    selected, official = [], []
    for image in chosen:
        path = Path(args.image_root) / image
        if not path.is_file():
            raise SystemExit(f"missing train image: {path}")
        selected.append({"image": image, "image_path": str(path.resolve()), "official_records": len(grouped[image])})
        for row in grouped[image]:
            question, answer = question_answer(row)
            official.append({"image": image, "id": row.get("id"), "question": question, "answer": answer})
    write_jsonl(out / "selected_images.jsonl", selected)
    write_jsonl(out / "official_records.jsonl", official)
    (out / "manifest.json").write_text(json.dumps({
        "seed": args.seed, "selected_images": len(selected),
        "official_records": len(official), "train_only": True,
        "test_split_opened": False, "excluded_images": len(excluded_images),
        "exclude_selection": args.exclude_selection,
    }, indent=2) + "\n")
    print(json.dumps({"selected_images": len(selected), "official_records": len(official), "output_dir": str(out.resolve())}, indent=2))


if __name__ == "__main__":
    main()
