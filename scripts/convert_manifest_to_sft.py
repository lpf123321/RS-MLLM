#!/usr/bin/env python3
"""Convert manifest.jsonl → train_sft.py JSON with 80:20 VRS:LEVIR ratio."""
import json
import os

DATA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = "/users/u2024311009/team_share/rs-mllm-ready-20260725/data/manifest.jsonl"
OUTPUT = os.path.join(DATA_ROOT, "finetune_framework/VRSbench/manifest_sft_train.json")

ALLOWED_STATUSES = {"keep", "corrected"}

converted = []
skipped = {"manual_review": 0, "exclude": 0, "no_answer": 0}

with open(MANIFEST) as f:
    for line in f:
        s = json.loads(line)

        # Only training splits
        ds = s["dataset"]
        split = s["split"]
        if split != "train":
            skipped.setdefault("not_train", 0)
            skipped["not_train"] += 1
            continue

        # Only keep/corrected
        if s["clean_status"] not in ALLOWED_STATUSES:
            skipped[s["clean_status"]] += 1
            continue

        imgs = s["images"]
        num_images = len(imgs)
        image_tags = "\n".join(["<image>"] * num_images)

        if num_images == 1:
            image_field = imgs[0]["path"]
        else:
            image_field = [img["path"] for img in imgs]

        task_type = s["task_type"]
        refs = s.get("references", [])
        answer_labels = s.get("answer_labels", [])

        if task_type == "change_caption":
            # LEVIR: create 1 sample per reference for 80:20 ratio
            if not refs:
                skipped["no_answer"] += 1
                continue
            for ref in refs:
                prompt = s["prompt"]
                converted.append({
                    "id": s["id"],
                    "image": image_field,
                    "conversations": [
                        {"from": "human", "value": f"{image_tags}\n{prompt}"},
                        {"from": "gpt", "value": ref},
                    ],
                })
        elif task_type in ("open_vqa", "caption", "bbox"):
            if not refs:
                skipped["no_answer"] += 1
                continue
            answer = refs[0]
            prompt = s["prompt"]
            converted.append({
                "id": s["id"],
                "image": image_field,
                "conversations": [
                    {"from": "human", "value": f"{image_tags}\n{prompt}"},
                    {"from": "gpt", "value": answer},
                ],
            })
        elif task_type in ("single_choice", "multi_choice"):
            if not answer_labels:
                skipped["no_answer"] += 1
                continue
            answer = ", ".join(answer_labels)
            prompt = s["prompt"]
            converted.append({
                "id": s["id"],
                "image": image_field,
                "conversations": [
                    {"from": "human", "value": f"{image_tags}\n{prompt}"},
                    {"from": "gpt", "value": answer},
                ],
            })
        else:
            skipped[f"unknown_task_{task_type}"] = skipped.get(f"unknown_task_{task_type}", 0) + 1
            continue

with open(OUTPUT, "w") as f:
    json.dump(converted, f, indent=2, ensure_ascii=False)

# Summary
from collections import Counter
task_counts = Counter()
for s in converted:
    ds = s["id"].split("/")[0]
    if ds == "levir_cc":
        task_counts["levir_cc"] += 1
    else:
        task_counts[s["id"].split("/")[2]] += 1

total = len(converted)
vrs = sum(v for k, v in task_counts.items() if k != "levir_cc")
levir = task_counts.get("levir_cc", 0)

print(f"转换完成:")
print(f"  输出:   {OUTPUT}")
print(f"  总样本: {total}")
print(f"  VRSBench: {vrs} ({vrs*100/total:.1f}%)")
print(f"  LEVIR-CC: {levir} ({levir*100/total:.1f}%)")
print(f"")
print(f"跳过分项:")
for k, v in sorted(skipped.items()):
    if v > 0:
        print(f"  {k}: {v}")
print(f"")
print(f"按任务:")
for k, v in sorted(task_counts.items()):
    print(f"  {k}: {v}")
