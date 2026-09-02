"""
Preprocess XLRS-Bench-lite into Qwen3-VL messages format.

The dataset has 3080 QA pairs over 800 unique images (each image has ~3.85 questions).
Images are embedded in Arrow files (74 shards). The `index` field (0-799) identifies
unique images, NOT samples -- multiple arrow shards contain overlapping indices.

Output: output/xlrs.jsonl (3080 lines)
Images are saved to output/images/ (deduplicated by index, 800 files max).
"""

import json
import io
import os
import sys
import time
import argparse
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rsmllm.config import DATA_ROOT as DATASETS_ROOT

import pyarrow as pa
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

DATASET_ROOT = DATASETS_ROOT / "XLRS-Bench-lite"
ARROW_DIR = DATASET_ROOT / "train"
OUTPUT_DIR = DATASET_ROOT
IMAGE_OUTPUT_DIR = OUTPUT_DIR / "images_resized"
MAX_LONG_EDGE = 1024
LOG_FILE = OUTPUT_DIR / "preprocess_xlrs.log"


def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(msg, flush=True)


def save_image_once(img_bytes, index, output_dir):
    out_path = output_dir / f"xlrs_{index:05d}.png"
    if out_path.exists():
        return str(out_path)
    img = Image.open(io.BytesIO(img_bytes))
    w, h = img.size
    if max(w, h) > MAX_LONG_EDGE:
        if w >= h:
            new_w = MAX_LONG_EDGE
            new_h = int(h * MAX_LONG_EDGE / w)
        else:
            new_h = MAX_LONG_EDGE
            new_w = int(w * MAX_LONG_EDGE / h)
        img = img.resize((new_w, new_h), Image.BICUBIC)
    img.save(out_path)
    return str(out_path)


def build_messages(question, options_list, answer, image_path):
    choices_str = "\n".join(options_list)
    instruction = f"[MCQ] {question}\n{choices_str}"
    answer_text = None
    for opt in options_list:
        if opt.startswith(f"({answer})"):
            answer_text = opt
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


def process_chunk(arrow_files, chunk_idx, total_chunks):
    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
    out_path = OUTPUT_DIR / "xlrs.jsonl"

    mode = "a" if chunk_idx > 0 else "w"
    total = 0
    saved_imgs = 0
    t_start = time.time()

    with open(out_path, mode) as out:
        for af_idx, arrow_file in enumerate(arrow_files):
            samples_in_file = 0
            t_file = time.time()

            try:
                with open(arrow_file, "rb") as f:
                    reader = pa.ipc.open_stream(f)
                    for batch in reader:
                        for i in range(len(batch)):
                            try:
                                question = batch.column("question")[i].as_py()
                                options = batch.column("multi-choice options")[i].as_py()
                                answer = batch.column("answer")[i].as_py()
                                index = batch.column("index")[i].as_py()
                                raw_img = batch.column("image")[i].as_py()

                                image_path = save_image_once(
                                    raw_img[0]["bytes"], index, IMAGE_OUTPUT_DIR
                                )
                                if image_path:
                                    saved_imgs += 1

                                record = build_messages(question, options, answer, image_path)
                                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                                total += 1
                                samples_in_file += 1
                            except Exception as e:
                                log(f"  ERROR row {i} in {arrow_file.name}: {e}\n{traceback.format_exc()}")
            except Exception as e:
                log(f"  ERROR opening {arrow_file.name}: {e}\n{traceback.format_exc()}")

            elapsed = time.time() - t_file
            elapsed_total = time.time() - t_start
            log(f"  [{af_idx+1}/{len(arrow_files)}] {arrow_file.name}: {samples_in_file} in {elapsed:.1f}s "
                f"(total: {total}, unique_imgs: {len(list(IMAGE_OUTPUT_DIR.glob('*.png')))})")

    t_total = time.time() - t_start
    log(f"Chunk done: {total} samples in {t_total:.1f}s, {saved_imgs} save attempts")
    return total


def verify():
    out_path = OUTPUT_DIR / "xlrs.jsonl"
    with open(out_path) as f:
        lines = f.readlines()
    log(f"  JSONL lines: {len(lines)} (expected 3080)")

    image_files = sorted(IMAGE_OUTPUT_DIR.glob("*.png"))
    log(f"  Images on disk: {len(image_files)} (max 800 unique)")

    max_w, max_h = 0, 0
    for img_f in image_files:
        img = Image.open(img_f)
        if img.width > max_w:
            max_w = img.width
        if img.height > max_h:
            max_h = img.height
        if img.width > MAX_LONG_EDGE or img.height > MAX_LONG_EDGE:
            log(f"  WARN: {img_f.name} exceeds limit ({img.width}x{img.height})")
    log(f"  max image size: {max_w}x{max_h}")
    log(f"  all within {MAX_LONG_EDGE}: {max_w <= MAX_LONG_EDGE and max_h <= MAX_LONG_EDGE}")

    # Verify all referenced images exist
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

    # Sample check
    import random
    samples = random.sample(lines, min(3, len(lines)))
    for i, sample in enumerate(samples):
        rec = json.loads(sample)
        uc = rec["messages"][0]["content"]
        ac = rec["messages"][1]["content"][0]["text"]
        img_path = uc[0]["image"]
        instr = uc[1]["text"]
        log(f"\n  sample {i + 1}:")
        log(f"    image: {img_path}  exists={os.path.exists(img_path)}")
        log(f"    instruction: {instr[:120]}...")
        log(f"    answer: {ac}")
        if os.path.exists(img_path):
            img = Image.open(img_path)
            log(f"    size: {img.width}x{img.height}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, nargs=2, default=None)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        log("=== Verification ===")
        verify()
        return

    arrow_files = sorted(ARROW_DIR.glob("data-*.arrow"))

    if args.chunk:
        chunk_idx, total_chunks = args.chunk
        chunk_size = (len(arrow_files) + total_chunks - 1) // total_chunks
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(arrow_files))
        files_to_process = arrow_files[start:end]
        log(f"Chunk {chunk_idx}/{total_chunks}: files {start}-{end-1} ({len(files_to_process)} files)")
        process_chunk(files_to_process, chunk_idx, total_chunks)
    else:
        log(f"Found {len(arrow_files)} arrow files, total ~3080 samples, ~800 unique images")
        process_chunk(arrow_files, 0, 1)
        log("\n=== Verification ===")
        verify()
        log("\nDone.")


if __name__ == "__main__":
    main()
