#!/usr/bin/env python3
"""Fix the 2048-res XLRS caption/grounding datasets to the VRSBench-style format.

The datasets_data/ jsonl were produced without the task prefix and with 0-1
normalized bracket bboxes, so they don't work with the existing prefix-based
evalsets / ReferringAcc metric. This rewrites them in place:

  - caption   -> "[CAP] " prefix
  - grounding -> "[REF] " + description (dropping the conflicting coordinate-format
                 preamble), and 0-1 bracket bbox -> 0-100 "{<x1><y1><x2><y2>}"

The system prompt and <think> prefill are NOT baked into the data: they are added
by the router adapter at inference (RS-MLLM convention, see evaluation/adapters/
router.py), so this only needs the prefix + 0-100 bbox to be consistent.

Usage:
    python scripts/fix_xlrs_2048.py
"""
import argparse
import json
import re

BBOX_RE = re.compile(
    r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]"
)


def bbox_to_angle(text: str) -> str:
    m = BBOX_RE.search(text)
    if not m:
        return text
    x1, y1, x2, y2 = map(float, m.groups())
    return "{<%d><%d><%d><%d>}" % (
        round(x1 * 100), round(y1 * 100), round(x2 * 100), round(y2 * 100)
    )


def fix_caption(path: str):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            uc = d["messages"][0]["content"]
            text = uc[-1]["text"]
            if not text.startswith("[CAP]"):
                uc[-1]["text"] = "[CAP] " + text
            records.append(d)
    with open(path, "w", encoding="utf-8") as f:
        for d in records:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"fixed caption: {len(records)} samples -> {path}")


def fix_grounding(path: str):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            uc = d["messages"][0]["content"]
            text = uc[-1]["text"]
            if not text.startswith("[REF]"):
                if "Description: " in text:
                    desc = text.split("Description: ", 1)[1].strip()
                else:
                    desc = text.strip()
                uc[-1]["text"] = "[REF] " + desc
            d["messages"][1]["content"][0]["text"] = bbox_to_angle(
                d["messages"][1]["content"][0]["text"]
            )
            records.append(d)
    with open(path, "w", encoding="utf-8") as f:
        for d in records:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"fixed grounding: {len(records)} samples -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caption", default="datasets_data/xlrs_caption.jsonl")
    ap.add_argument("--grounding", default="datasets_data/xlrs_grounding.jsonl")
    args = ap.parse_args()

    fix_caption(args.caption)
    fix_grounding(args.grounding)


if __name__ == "__main__":
    main()
