#!/usr/bin/env python3
"""把评估数据集按任务过滤，生成每个 Expert 专用的评估子集 jsonl。

vrsbench 源文件含 [VQA]/[CAP]/[REF] 前缀，MME/XLRS 全是 MCQ（路由到 General），
LEVIR-CC 全是 change caption（路由到 Change）。

输出（与源文件同目录）：
  vrsbench_eval_general.jsonl      vqa + caption 样本
  vrsbench_eval_grounding.jsonl    referring 样本
  mme_eval_general.jsonl / xlrs_eval_general.jsonl   (整文件即 General)
  levircc_eval_change.jsonl       (整文件即 Change)
"""
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rsmllm.config import DATA_ROOT

SHARED = str(DATA_ROOT)
OUT = str(REPO_ROOT / "evaluation" / "split_evals")
os.makedirs(OUT, exist_ok=True)


def iter_jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def dump(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# VRSBench 按前缀拆分
vrs_src = f"{SHARED}/VRSBench/vrsbench_eval.jsonl"
general_rows, grounding_rows = [], []
for d in iter_jsonl(vrs_src):
    text = d["messages"][0]["content"][-1]["text"]
    if text.startswith("[REF]"):
        grounding_rows.append(d)
    else:  # [VQA] / [CAP] 均归 General
        general_rows.append(d)

dump(f"{OUT}/vrsbench_eval_general.jsonl", general_rows)
dump(f"{OUT}/vrsbench_eval_grounding.jsonl", grounding_rows)
print(f"vrsbench general(vqa+caption): {len(general_rows)}")
print(f"vrsbench grounding(referring): {len(grounding_rows)}")

# MME / XLRS / LEVIR-CC 整文件即对应 Expert
for name in ["mme_rs", "xlrs"]:
    src = f"{SHARED}/MME-RealWorld-RS/{name}.jsonl" if name == "mme_rs" else f"{SHARED}/XLRS-Bench-lite/xlrs.jsonl"
    dump(f"{OUT}/{name}_general.jsonl", list(iter_jsonl(src)))
    print(f"{name}_general: copied")

src = f"{SHARED}/LEVIR-CC/levircc_test.jsonl"
dump(f"{OUT}/levircc_change.jsonl", list(iter_jsonl(src)))
print(f"levircc_change: copied")

print(f"OUT: {OUT}")
