#!/usr/bin/env python3
"""从 VRSBench VQA 确定性构造可直接训练的 MCQ 数据。"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def asset_index(root: Path) -> dict[str, str]:
    """原图 basename -> data/assets 下内容哈希相对路径。"""
    index: dict[str, str] = {}
    for path in root.iterdir():
        if not (path.is_file() or path.is_symlink()):
            continue
        source_name = Path(os.path.realpath(path)).name
        current = index.get(source_name)
        # 早期缓存同时有 40/64 hex 名，固定优先完整 SHA-256。
        if current is None or len(path.stem) > len(Path(current).stem):
            index[source_name] = f"assets/vrsbench/{path.name}"
    return index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="datasets/shared_datasets/VRSBench/VRSBench_train.json")
    ap.add_argument("--assets", default="data/assets/vrsbench")
    ap.add_argument("--output", default="finetune_framework/VRSbench/combined_train_stage2.json")
    ap.add_argument("--num-mcq", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    source = Path(args.input)
    assets = Path(args.assets)
    output = Path(args.output)
    if not source.is_absolute():
        source = REPO / source
    if not assets.is_absolute():
        assets = REPO / assets
    if not output.is_absolute():
        output = REPO / output
    if not source.is_file():
        raise FileNotFoundError(source)
    if not assets.is_dir():
        raise FileNotFoundError(assets)

    with source.open() as f:
        raw = json.load(f)
    image_map = asset_index(assets)
    data = []
    missing = 0
    for sample in raw:
        mapped = image_map.get(Path(sample["image"]).name)
        if mapped is None:
            missing += 1
            continue
        item = dict(sample)
        item["image"] = mapped
        data.append(item)

    vqa = [s for s in data if "[vqa]" in s["conversations"][0]["value"].lower()]
    if not vqa:
        raise ValueError(f"no [vqa] records in {source}")
    answer_pool = sorted({s["conversations"][1]["value"].strip() for s in vqa})
    if len(answer_pool) < 4:
        raise ValueError("need at least four distinct VQA answers")

    rand = random.Random(args.seed)
    counts = {letter: 0 for letter in "ABCD"}
    mcq_samples = []
    for ordinal in range(args.num_mcq):
        sample = rand.choice(vqa)
        question = sample["conversations"][0]["value"].replace("<image>\n", "")
        correct = sample["conversations"][1]["value"].strip()
        candidates = [answer for answer in answer_pool if answer.casefold() != correct.casefold()]
        options = [correct, *rand.sample(candidates, 3)]
        rand.shuffle(options)
        letter = "ABCD"[options.index(correct)]
        counts[letter] += 1
        lines = [f"[MCQ] {question}"]
        lines.extend(f"({chr(65 + idx)}) {answer}" for idx, answer in enumerate(options))
        mcq_samples.append({
            "id": f"mcq_synth_{ordinal:05d}",
            "image": sample["image"],
            "conversations": [
                {"from": "human", "value": "<image>\n" + "\n".join(lines)},
                {"from": "gpt", "value": f"{letter}. ({letter}) {correct}"},
            ],
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(data + mcq_samples, f, indent=2, ensure_ascii=False)
    print("Stage 2 训练数据生成完成:")
    print(f"  可移植原数据: {len(data)} 条（缺少资产 {missing} 条）")
    print(f"  VQA 候选: {len(vqa)} 条")
    print(f"  MCQ 合成: {len(mcq_samples)} 条")
    print(f"  A/B/C/D 分布: {counts}")
    print(f"  输出: {output}")


if __name__ == "__main__":
    main()
