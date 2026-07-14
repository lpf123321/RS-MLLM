"""
Preprocess MME-RealWorld-RS (Remote Sensing subset) into Qwen3-VL messages format.

MME-RealWorld-RS has 3,738 MCQ entries over 1,265 unique remote sensing images.
Images are separate PNG files. Multiple questions reference the same image.
Output: output/mme_rs.jsonl (3,738 lines), output/images/mme_*.png (1,265 max)
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path
from collections import defaultdict

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ANNOT_PATH = Path("/users/u2024311136/shared/shared_datasets/MME-RealWorld-RS/MME_RealWorld.json")
IMAGE_ROOT = Path("/users/u2024311136/shared/shared_datasets/MME-RealWorld-RS")
OUTPUT_DIR = Path("/home/u2024311136/RS-MLLM/output")
IMAGE_OUTPUT_DIR = OUTPUT_DIR / "images"
MAX_LONG_EDGE = 1024
LOG_FILE = OUTPUT_DIR / "preprocess_mme.log"


def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(msg, flush=True)


def save_image_once(src_path, dst_path):
    if dst_path.exists():
        return str(dst_path)
    img = Image.open(src_path)
    w, h = img.size
    if max(w, h) > MAX_LONG_EDGE:
        if w >= h:
            new_w = MAX_LONG_EDGE
            new_h = int(h * MAX_LONG_EDGE / w)
        else:
            new_h = MAX_LONG_EDGE
            new_w = int(w * MAX_LONG_EDGE / h)
        img = img.resize((new_w, new_h), Image.BICUBIC)
    img.save(dst_path)
    return str(dst_path)


def build_messages(question, choices, answer, image_path):
    choices_str = "\n".join(choices)
    instruction = f"[MCQ] {question}\n{choices_str}"
    answer_text = None
    for c in choices:
        if c.startswith(f"({answer})"):
            answer_text = c
            break
    gt = f"{answer}. {answer_text}" if answer_text else answer
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": instruction},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": gt}
                ],
            },
        ]
    }


def main():
    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

    log("Loading annotations...")
    with open(ANNOT_PATH) as f:
        data = json.load(f)
    log(f"  Total entries: {len(data)}")

    # Filter Remote Sensing only
    rs_entries = [e for e in data if e.get("Subtask") == "Remote Sensing"]
    log(f"  Remote Sensing entries: {len(rs_entries)}")

    unique_images = set(e["Image"] for e in rs_entries)
    log(f"  Unique image paths: {len(unique_images)}")

    out_path = OUTPUT_DIR / "mme_rs.jsonl"
    total = 0
    saved_imgs = 0
    missing_imgs = 0
    t_start = time.time()

    with open(out_path, "w") as out:
        for entry in rs_entries:
            try:
                rel_path = entry["Image"]
                src_path = IMAGE_ROOT / rel_path

                if not src_path.exists():
                    missing_imgs += 1
                    log(f"  WARN: image not found: {src_path}")
                    continue

                # Save resized image (dedup by filename)
                dst_name = f"mme_{Path(rel_path).name}"
                dst_path = IMAGE_OUTPUT_DIR / dst_name
                image_path = save_image_once(str(src_path), dst_path)
                if image_path:
                    saved_imgs += 1

                question = entry["Text"]
                choices = entry["Answer choices"]
                answer = entry["Ground truth"]

                record = build_messages(question, choices, answer, image_path)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1

                if total % 500 == 0:
                    elapsed = time.time() - t_start
                    log(f"  ... {total}/{len(rs_entries)} entries, unique_imgs={len(list(IMAGE_OUTPUT_DIR.glob('mme_*.png')))}, {elapsed:.0f}s")

            except Exception as e:
                log(f"  ERROR on {entry.get('Question_id', '?')}: {e}\n{traceback.format_exc()}")

    t_total = time.time() - t_start
    log(f"\nDone: {total} lines to {out_path} in {t_total:.1f}s")
    log(f"  Images saved: {saved_imgs} attempts, {len(list(IMAGE_OUTPUT_DIR.glob('mme_*.png')))} unique on disk")
    log(f"  Missing images: {missing_imgs}")

    # Verify
    log("\n=== Verification ===")
    with open(out_path) as f:
        lines = f.readlines()
    log(f"  JSONL lines: {len(lines)}")

    image_files = sorted(IMAGE_OUTPUT_DIR.glob("mme_*.png"))
    log(f"  Images on disk: {len(image_files)}")

    max_w, max_h = 0, 0
    for img_f in image_files:
        img = Image.open(img_f)
        if img.width > max_w:
            max_w = img.width
        if img.height > max_h:
            max_h = img.height
        if img.width > MAX_LONG_EDGE or img.height > MAX_LONG_EDGE:
            log(f"  WARN: {img_f.name} exceeds ({img.width}x{img.height})")
    log(f"  max image size: {max_w}x{max_h}")
    log(f"  all <= {MAX_LONG_EDGE}: {max_w <= MAX_LONG_EDGE and max_h <= MAX_LONG_EDGE}")

    unique_refs = set()
    missing = 0
    for line in lines:
        rec = json.loads(line)
        img_path = rec["messages"][0]["content"][0]["image"]
        unique_refs.add(img_path)
        if not Path(img_path).exists():
            missing += 1
    log(f"  Unique image paths in JSONL: {len(unique_refs)}")
    log(f"  Missing referenced images: {missing}")

    import random
    samples = random.sample(lines, min(3, len(lines)))
    for i, sample in enumerate(samples):
        rec = json.loads(sample)
        uc = rec["messages"][0]["content"]
        ac = rec["messages"][1]["content"][0]["text"]
        img_path = uc[0]["image"]
        instr = uc[1]["text"]
        log(f"\n  sample {i + 1}:")
        log(f"    image: {Path(img_path).name}  exists={Path(img_path).exists()}")
        log(f"    instruction: {instr[:100]}...")
        log(f"    answer: {ac}")
        if Path(img_path).exists():
            img = Image.open(img_path)
            log(f"    size: {img.width}x{img.height}")

    log("\nDone.")


if __name__ == "__main__":
    main()
