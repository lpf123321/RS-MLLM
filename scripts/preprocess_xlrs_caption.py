#!/usr/bin/env python3
"""Convert XLRS caption split_evals (flat jsonl) to VRSBench-style messages jsonl.

Reads the classmate's preprocessed flat jsonl and rewrites it into the same
``messages`` format used by the VRSBench evalset, so the existing caption metrics
(BLEU / ROUGE-L / CIDEr) and the task router ([CAP] -> general expert) work
unchanged.

Usage:
    python scripts/preprocess_xlrs_caption.py [--src ...] [--out ...]
"""
import argparse
import json

DEFAULT_SRC = (
    "/users/u2024311136/shared/shared_models/lora_expert/evaluation/split_evals/"
    "xlrs_caption_en.jsonl"
)
DEFAULT_OUT = "evaluation/data/xlrs_caption.jsonl"


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
            prompt = "[CAP] " + d.get("prompt", "Describe the image in detail.")
            refs = d.get("references", [])
            answer = refs[0] if refs else ""
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
