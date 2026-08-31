#!/usr/bin/env python3
"""Regenerate the 2048-res XLRS caption/grounding datasets in the routing-compatible format.

The datasets_data/ jsonl were produced without the routing prefix (and, after an
earlier experiment, the grounding bbox was rounded to 0-100). This rewrites them:

  - caption   -> "[CAP] " prefix (the official three-part prompt is added as the
                 system prompt by evaluation/main.py, not baked into the data)
  - grounding -> regenerated from the split_evals (0-1 float bbox + official
                 question), mapping to the local 2048 images and rewriting the
                 question's "Given a {W} x {H} pixel" to the actual resized dims;
                 "[REF] " prefix for expert routing

Usage:
    python scripts/fix_xlrs_2048.py
"""
import argparse
import json
from rsmllm.config import MODELS_ROOT as M_ROOT
import os
import re

SPLIT_EVALS = (
    "M_ROOT/lora_expert/evaluation/split_evals/"
)
PIXEL_DIMS_RE = re.compile(r"Given a \d+ x \d+ pixel")


def bbox_to_text(box):
    x1, y1, x2, y2 = box
    return "[%s, %s, %s, %s]" % (x1, y1, x2, y2)


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


def regenerate_grounding(image_dir: str, out: str):
    """Rebuild the 2048 grounding jsonl from the split_evals (0-1 bbox + question)."""
    from PIL import Image

    src = os.path.join(SPLIT_EVALS, "xlrs_grounding_test_4096.jsonl")
    n = 0
    with open(src, encoding="utf-8") as f_in, open(out, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            filename = os.path.basename(d["image"])
            local_path = os.path.join(image_dir, filename)

            question = d["question"].strip()
            # 用 2048 实际尺寸改写 question 里的像素分辨率
            try:
                with Image.open(local_path) as im:
                    w, h = im.size
                question = PIXEL_DIMS_RE.sub(f"Given a {w} x {h} pixel", question)
            except Exception:
                pass

            record = {
                "messages": [
                    {"role": "user", "content": [
                        {"type": "image", "image": local_path},
                        {"type": "text", "text": "[REF] " + question},
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": bbox_to_text(d["bbox"])},
                    ]},
                ],
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    print(f"regenerated grounding: {n} samples -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caption", default="datasets_data/xlrs_caption.jsonl")
    ap.add_argument("--grounding_images", default="datasets_data/xlrs_grounding_images")
    ap.add_argument("--grounding", default="datasets_data/xlrs_grounding.jsonl")
    args = ap.parse_args()

    fix_caption(args.caption)
    regenerate_grounding(args.grounding_images, args.grounding)


if __name__ == "__main__":
    main()
