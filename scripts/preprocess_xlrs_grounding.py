#!/usr/bin/env python3
"""Convert XLRS visual-grounding split_evals to VRSBench-style messages jsonl.

Reads the classmate's flat jsonl (0-1 normalized bbox + image dims) and rewrites
it into the VRSBench referring format: ``[REF]`` prefix + a 0-100 integer text
bbox ``{<x1><y1><x2><y2>}``, so the existing ``ReferringAcc`` metric and the task
router ([REF] -> grounding expert) work unchanged.

Usage:
    python scripts/preprocess_xlrs_grounding.py [--src ...] [--out ...]
"""
import argparse
import json

DEFAULT_SRC = (
    "/users/u2024311136/shared/shared_models/lora_expert/evaluation/split_evals/"
    "xlrs_grounding_test_4096.jsonl"
)
DEFAULT_OUT = "evaluation/data/xlrs_grounding.jsonl"


def bbox_to_angle(box):
    """0-1 normalized [x1,y1,x2,y2] -> '{<x1><y1><x2><y2>}' with 0-100 ints."""
    x1, y1, x2, y2 = box
    return "{<%d><%d><%d><%d>}" % (
        round(x1 * 100), round(y1 * 100), round(x2 * 100), round(y2 * 100)
    )


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
            question = d["question"]
            # 提取 "Description: ..." 作为任务描述，前面加 [REF] 前缀
            if "Description: " in question:
                desc = question.split("Description: ", 1)[1].strip()
            else:
                desc = question.strip()
            prompt = "[REF] " + desc
            answer = bbox_to_angle(d["bbox"])
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
