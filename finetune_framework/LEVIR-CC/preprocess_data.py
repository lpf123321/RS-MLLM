"""
Preprocess LEVIR-CC dataset into Qwen-VL Series Finetune conversations format.

LEVIR-CC is a change detection/captioning dataset with image pairs (A: before, B: after).
Task: describe the changes between two images.

Output format (compatible with Qwen-VL-Series-Finetune):
  [
    {
      "id": "...",
      "image": ["path/to/A.jpg", "path/to/B.jpg"],
      "conversations": [
        {"from": "human", "value": "<image>\n<image>\n[CD] Describe the changes..."},
        {"from": "gpt", "value": "The caption describing changes."}
      ]
    }
  ]

Usage:
  python preprocess_data.py                           # default paths
  python preprocess_data.py --max_samples 100         # quick test with 100 samples
"""

import json
import os
import argparse
from pathlib import Path

# ---- Configurable paths ----
# Set these to match your environment
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_ROOT = REPO_ROOT / "datasets" / "shared_datasets" / "LEVIR-CC"
ANNOT_PATH = DATASET_ROOT / "LevirCCcaptions.json"
IMAGE_ROOT = DATASET_ROOT / "images"
OUTPUT_DIR = Path(__file__).resolve().parent  # same dir as this script by default

INSTRUCTION = "[CD] Describe the changes between these two images."

EXPECTED_COUNTS = {"train": 34075, "val": 6665, "test": 9645}


def get_image_paths(split: str, filename: str):
    """Return (A_path, B_path) relative to IMAGE_ROOT, using cross-platform separators."""
    a_path = f"{split}/A/{filename}"
    b_path = f"{split}/B/{filename}"
    return a_path, b_path


def process_split(split: str, data: dict, max_samples: int = None):
    """Convert one split to conversations format."""
    output = []
    skipped = 0

    for img in data["images"]:
        if img["split"] != split:
            continue

        filename = img["filename"]
        a_path, b_path = get_image_paths(split, filename)

        # Verify images exist on disk
        a_full = IMAGE_ROOT / a_path
        b_full = IMAGE_ROOT / b_path

        if not a_full.exists():
            skipped += 1
            continue
        if not b_full.exists():
            skipped += 1
            continue

        for sentence in img["sentences"]:
            caption = sentence["raw"].strip()
            if not caption:
                continue

            record = {
                "id": f"levircc_{split}_{filename}",
                "image": [a_path, b_path],
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n<image>\n{INSTRUCTION}",
                    },
                    {
                        "from": "gpt",
                        "value": caption,
                    },
                ],
            }
            output.append(record)

            if max_samples and len(output) >= max_samples:
                break

        if max_samples and len(output) >= max_samples:
            break

    return output, skipped


def verify(data: list):
    """Quick sanity check on processed data."""
    import random

    print(f"\n{'='*60}")
    print(f"Verify: {len(data)} samples")
    print(f"{'='*60}")

    samples = random.sample(data, min(3, len(data)))
    for i, rec in enumerate(samples):
        print(f"\n  Sample {i+1}:")
        print(f"    id: {rec['id']}")
        print(f"    images: {rec['image']}")
        print(f"    human: {rec['conversations'][0]['value'][:100]}...")
        print(f"    gpt:   {rec['conversations'][1]['value'][:100]}...")


def main():
    parser = argparse.ArgumentParser(description="Preprocess LEVIR-CC for finetune")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples per split (for quick tests)")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading annotations from: {ANNOT_PATH}")
    with open(ANNOT_PATH) as f:
        data = json.load(f)
    print(f"  Total image entries: {len(data['images'])}")

    splits = ["train", "val", "test"]

    for split in splits:
        print(f"\nProcessing {split}...")
        records, skipped = process_split(split, data, max_samples=args.max_samples)

        out_path = output_dir / f"LEVIR-CC_{split}.json"
        with open(out_path, "w") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        expected = EXPECTED_COUNTS.get(split, "?")
        print(f"  Wrote {len(records)} samples to {out_path} (expected ~{expected}, skipped {skipped})")
        verify(records)


if __name__ == "__main__":
    main()
