"""
Preprocess VRSBench dataset into Qwen3-VL messages format.

Outputs:
  output/vrsbench_train.jsonl  (~142,390 lines)
  output/vrsbench_eval.jsonl   (~62,918 lines)
"""

import json
import os
from pathlib import Path

DATA_ROOT = Path("/users/u2024311136/shared/shared_datasets/VRSBench")
OUTPUT_DIR = Path("/home/u2024311136/RS-MLLM/output")

TASK_PREFIX = {
    "caption": "[CAP]",
    "vqa":     "[VQA]",
    "refer":   "[REF]",
}


def build_image_map():
    image_path_map = {}
    for subdir in ["images/Images_train", "images/val"]:
        d = DATA_ROOT / subdir
        if d.exists():
            for f in os.listdir(d):
                image_path_map[f] = str(d / f)
    return image_path_map


def parse_human(value: str):
    rest = value.replace("<image>\n", "", 1)
    if rest.startswith("[caption]"):
        task = "caption"
        instruction = rest[len("[caption] "):]
    elif rest.startswith("[vqa]"):
        task = "vqa"
        instruction = rest[len("[vqa] "):]
    elif rest.startswith("[refer]"):
        task = "refer"
        instruction = rest[len("[refer] "):]
    else:
        task = "vqa"
        instruction = rest
    return task, instruction


def process_train(image_path_map: dict):
    train_path = DATA_ROOT / "VRSBench_train.json"
    with open(train_path) as f:
        data = json.load(f)

    out_path = OUTPUT_DIR / "vrsbench_train.jsonl"
    skipped = 0
    with open(out_path, "w") as out:
        for entry in data:
            img_file = entry["image"]
            img_path = image_path_map.get(img_file)
            if img_path is None:
                skipped += 1
                continue

            convs = entry["conversations"]
            human_val = convs[0]["value"]
            gpt_val = convs[1]["value"].strip()

            if not gpt_val:
                skipped += 1
                continue

            task, instruction = parse_human(human_val)
            final_instruction = f"{TASK_PREFIX[task]} {instruction}"

            record = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img_path},
                            {"type": "text", "text": final_instruction},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": gpt_val},
                        ],
                    },
                ]
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Train: wrote {out_path}, {len(data) - skipped}/{len(data)} entries (skipped {skipped})")


def process_eval(image_path_map: dict):
    eval_files = [
        ("VRSBench_EVAL_vqa.json", "vqa"),
        ("VRSBench_EVAL_referring.json", "refer"),
        ("VRSBench_EVAL_Cap.json", "caption"),
    ]

    out_path = OUTPUT_DIR / "vrsbench_eval.jsonl"
    total = 0
    with open(out_path, "w") as out:
        for fname, task_type in eval_files:
            fp = DATA_ROOT / fname
            with open(fp) as f:
                data = json.load(f)

            for entry in data:
                img_id = entry["image_id"]
                img_path = image_path_map.get(img_id)
                if img_path is None:
                    continue

                question = entry.get("question", "")
                ground_truth = entry.get("ground_truth", "").strip()

                if not ground_truth:
                    continue

                if task_type == "caption":
                    final_instruction = f"[CAP] {question}"
                elif task_type == "refer":
                    final_instruction = f"[REF] {question}"
                else:
                    final_instruction = f"[VQA] {question}"

                record = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img_path},
                                {"type": "text", "text": final_instruction},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": ground_truth},
                            ],
                        },
                    ]
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1

            print(f"  {fname}: {len(data)} -> {total} (cumulative)")

    print(f"Eval: wrote {out_path}, {total} total entries")


def verify():
    print("\n=== Verification ===")
    train_path = OUTPUT_DIR / "vrsbench_train.jsonl"
    eval_path = OUTPUT_DIR / "vrsbench_eval.jsonl"

    # Count lines
    with open(train_path) as f:
        train_lines = sum(1 for _ in f)
    with open(eval_path) as f:
        eval_lines = sum(1 for _ in f)
    print(f"  train lines: {train_lines} (expected ~142,390)")
    print(f"  eval lines:  {eval_lines} (expected ~62,918)")

    # Check image paths exist
    import random
    for name, path, expected_lines in [
        ("train", train_path, train_lines),
        ("eval", eval_path, eval_lines),
    ]:
        with open(path) as f:
            lines = f.readlines()

        # Check random samples
        samples = random.sample(lines, min(3, len(lines)))
        all_images_exist = True
        for line in lines:
            rec = json.loads(line)
            img_path = rec["messages"][0]["content"][0]["image"]
            if not os.path.exists(img_path):
                print(f"  WARN: missing image {img_path}")
                all_images_exist = False
                break

        print(f"  {name}: all image paths exist = {all_images_exist}")
        print(f"  {name}: sample record:")
        print(json.dumps(json.loads(samples[0]), indent=2, ensure_ascii=False)[:600])
        print()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Building image path map...")
    image_path_map = build_image_map()
    print(f"  Found {len(image_path_map)} images")

    print("\nProcessing training data...")
    process_train(image_path_map)

    print("\nProcessing eval data...")
    process_eval(image_path_map)

    print("\nVerifying...")
    verify()


if __name__ == "__main__":
    main()
