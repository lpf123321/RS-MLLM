#!/usr/bin/env python3
"""Convert XLRS visual-grounding split_evals to messages format (official口径).

Reads the classmate's flat jsonl and rewrites it into the ``messages`` format,
keeping the OFFICIAL question and the 0-1 normalized bbox ``[xmin, ymin, xmax,
ymax]`` (no rounding). A minimal ``[REF]`` prefix is prepended only for expert
routing; the bbox stays 0-1 float and is scored by ``GroundingIoU`` (which
normalizes pixel/0-100 predictions to 0-1).

Usage:
    python scripts/preprocess_xlrs_grounding.py [--src ...] [--out ...]
"""
import argparse
import json

DEFAULT_SRC = (
    "M_ROOT/lora_expert/evaluation/split_evals/"
    "xlrs_grounding_test_4096.jsonl"
)
DEFAULT_OUT = "evaluation/data/xlrs_grounding.jsonl"


def bbox_to_text(box):
    """0-1 normalized [x1,y1,x2,y2] -> '[x1, y1, x2, y2]' (keep float precision)."""
    x1, y1, x2, y2 = box
    return "[%s, %s, %s, %s]" % (x1, y1, x2, y2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    n = 0
    with open(args.src, encoding="utf-8") as f_in, \
            open(args.out, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            image = d["image"]
            question = d["question"].strip()
            prompt = "[REF] " + question
            answer = bbox_to_text(d["bbox"])
            record = {
                "messages": [
                    {"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": answer},
                    ]},
                ],
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1

    print(f"wrote {n} samples -> {args.out}")


if __name__ == "__main__":
    main()
