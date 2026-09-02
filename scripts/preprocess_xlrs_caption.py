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
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rsmllm.config import MODELS_ROOT

DEFAULT_SRC = MODELS_ROOT / "lora_expert" / "evaluation" / "split_evals" / "xlrs_caption_en.jsonl"
DEFAULT_OUT = REPO_ROOT / "evaluation" / "data" / "xlrs_caption.jsonl"


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=DEFAULT_SRC)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    source = resolve_path(args.src)
    output = resolve_path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with source.open(encoding="utf-8") as f_in, output.open("w", encoding="utf-8") as f_out:
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

    print(f"wrote {n} samples -> {output}")


if __name__ == "__main__":
    main()
