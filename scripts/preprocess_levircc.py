"""
Preprocess LEVIR-CC dataset into Qwen3-VL messages format.

Outputs:
  output/levircc_train.jsonl  (~34,075 lines)
  output/levircc_val.jsonl    (~6,665 lines)
  output/levircc_test.jsonl   (~9,645 lines)
"""

import json
import os
from rsmllm.config import DATA_ROOT
import random
from pathlib import Path

ANNOT_PATH = Path("$DATA_ROOT/LEVIR-CC/LevirCCcaptions.json")
IMAGE_ROOT = Path("$DATA_ROOT/LEVIR-CC/images")
OUTPUT_DIR = Path("REPO_ROOT/output")

INSTRUCTION = "[CD] Describe the changes between these two images."

EXPECTED_LINES = {
    "train": 34075,
    "val": 1333,
    "test": 1929,
}

# val/test use grouped format: 1 entry per image pair with 5 references
EVAL_SPLITS = {"val", "test"}


def get_image_paths(split: str, filename: str):
    a_path = str(IMAGE_ROOT / split / "A" / filename)
    b_path = str(IMAGE_ROOT / split / "B" / filename)
    return a_path, b_path


def process_split(split: str, data: dict):
    out_path = OUTPUT_DIR / f"levircc_{split}.jsonl"
    total = 0
    skipped = 0

    with open(out_path, "w") as out:
        for img in data["images"]:
            if img["split"] != split:
                continue

            filename = img["filename"]
            a_path, b_path = get_image_paths(split, filename)

            if not os.path.exists(a_path):
                skipped += 1
                continue
            if not os.path.exists(b_path):
                skipped += 1
                continue

            references = [s["raw"].strip() for s in img["sentences"] if s["raw"].strip()]
            if not references:
                skipped += 1
                continue

            if split in EVAL_SPLITS:
                # grouped format: 1 entry per image pair, references for multi-ref eval
                record = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": a_path},
                                {"type": "image", "image": b_path},
                                {"type": "text", "text": INSTRUCTION},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": references[0]},
                            ],
                        },
                    ],
                    "references": references,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1
            else:
                # train: expand each sentence into independent sample
                for caption in references:
                    record = {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": a_path},
                                    {"type": "image", "image": b_path},
                                    {"type": "text", "text": INSTRUCTION},
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": [
                                    {"type": "text", "text": caption},
                                ],
                            },
                        ]
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total += 1

    expected = EXPECTED_LINES.get(split, "?")
    print(f"  {split}: wrote {total} lines to {out_path} (expected ~{expected}, skipped {skipped})")
    return total


def verify(split: str):
    out_path = OUTPUT_DIR / f"levircc_{split}.jsonl"
    with open(out_path) as f:
        lines = f.readlines()

    print(f"\n=== Verify {split} ===")
    print(f"  lines: {len(lines)}")

    all_images_exist = True
    all_refs_ok = True
    for line in lines:
        rec = json.loads(line)
        for item in rec["messages"][0]["content"]:
            if item["type"] == "image":
                if not os.path.exists(item["image"]):
                    print(f"  WARN: missing image {item['image']}")
                    all_images_exist = False
                    break

        if split in EVAL_SPLITS:
            refs = rec.get("references", [])
            if len(refs) != 5:
                print(f"  WARN: expected 5 references, got {len(refs)}")
                all_refs_ok = False

    print(f"  all image paths exist: {all_images_exist}")
    if split in EVAL_SPLITS:
        print(f"  all entries have 5 references: {all_refs_ok}")

    samples = random.sample(lines, min(3, len(lines)))
    for i, sample in enumerate(samples):
        rec = json.loads(sample)
        content = rec["messages"][0]["content"]
        images = [c for c in content if c["type"] == "image"]
        texts = [c for c in content if c["type"] == "text"]
        assistant_text = rec["messages"][1]["content"][0]["text"]
        print(f"\n  sample {i + 1}:")
        print(f"    A image: {images[0]['image']}")
        print(f"    B image: {images[1]['image']}")
        print(f"    instruction: {texts[0]['text']}")
        print(f"    answer: {assistant_text}")
        if split in EVAL_SPLITS:
            refs = rec.get("references", [])
            print(f"    references ({len(refs)}):")
            for j, r in enumerate(refs):
                print(f"      [{j}] {r}")
        print(f"    A exists: {os.path.exists(images[0]['image'])}")
        print(f"    B exists: {os.path.exists(images[1]['image'])}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading annotations...")
    with open(ANNOT_PATH) as f:
        data = json.load(f)
    print(f"  Total image entries: {len(data['images'])}")

    splits = ["train", "val", "test"]
    print("\nProcessing splits...")
    for split in splits:
        process_split(split, data)

    print("\nVerifying...")
    for split in splits:
        verify(split)


if __name__ == "__main__":
    main()
