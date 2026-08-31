#!/usr/bin/env python3
"""从 VRSBench VQA 构造 MCQ 格式训练数据"""
import json
import os
import random

random.seed(42)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO, "finetune_framework/VRSbench/combined_train.json")
OUTPUT_PATH = os.path.join(REPO, "finetune_framework/VRSbench/combined_train_stage2.json")

with open(DATA_PATH) as f:
    data = json.load(f)

# 提取 VQA 样本用于构造 MCQ
vqa = [s for s in data if "[vqa]" in s["conversations"][0]["value"].lower()]
answer_pool = list(set(s["conversations"][1]["value"].strip() for s in vqa))

NUM_MCQ = 5000
mcq_samples = []
rand = random.Random(42)
counts = {"A": 0, "B": 0, "C": 0, "D": 0}

for _ in range(NUM_MCQ):
    s = rand.choice(vqa)
    q = s["conversations"][0]["value"].replace("<image>\n", "")
    correct = s["conversations"][1]["value"].strip()
    distractors = rand.sample([a for a in answer_pool if a.lower() != correct.lower()], 3)
    opts = [correct] + distractors
    rand.shuffle(opts)
    ci = opts.index(correct)
    letter = "ABCD"[ci]
    counts[letter] += 1

    lines = [f"[MCQ] {q}"]
    for i, o in enumerate(opts):
        lines.append(f"({chr(65+i)}) {o}")

    mcq_samples.append({
        "id": f"mcq_synth_{_}",
        "image": s["image"],
        "conversations": [
            {"from": "human", "value": "<image>\n" + "\n".join(lines)},
            {"from": "gpt", "value": f"{letter}. ({letter}) {correct}"},
        ],
    })

merged = data + mcq_samples
with open(OUTPUT_PATH, "w") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

print(f"Stage 2 训练数据生成完成:")
print(f"  原数据: {len(data)} 条")
print(f"  MCQ 合成: {len(mcq_samples)} 条")
print(f"  总计: {len(merged)} 条")
print(f"  A/B/C/D 分布: {counts}")
print(f"  输出: {OUTPUT_PATH}")
