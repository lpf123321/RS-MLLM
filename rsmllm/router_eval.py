#!/usr/bin/env python3
"""一键路由专家评测: 按映射表自动跑 4 专家 × 对应子任务.

用法:
  python -m rsmllm.router_eval --quant bf16        # bf16 全量路由评测
  python -m rsmllm.router_eval --quant w8a8 --limit 50   # 小样本验证
  python -m rsmllm.router_eval --list              # 打印映射表

只选量化方式(bf16/w8a8/gptq), 专家与子任务自动分配:
  general  : vrsbench-vqa, mme, xlrs-bench-lite
  grounding: vrsbench-referring, xlrs-bench-grounding-en
  change   : levir-cc
  caption  : vrsbench-caption, xlrs-bench-caption

模型: 量化方式决定用 mmerestore(融合) 还是专家模型:
  bf16: mmerestore_bf16 / w8a8: mmerestore_w8a8 / gptq: mmerestore_gptq
  (专家单独评测用 expert_* 模型, 见 --expert 模式)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "evaluation" / "vllm_eval"
PY = EVAL_DIR / ".venv" / "bin" / "python"

# 路由映射: 专家 -> [(任务名, 源清单, task_type 过滤, 输出标签)]
ROUTE_PLAN = {
    "general": [
        ("vrsbench-vqa", "vrsbench_eval.jsonl", "open_vqa"),
        ("mme", "mme_rs.jsonl", None),
        ("xlrs-bench-lite", "xlrs.jsonl", None),
    ],
    "grounding": [
        ("vrsbench-referring", "vrsbench_eval.jsonl", "bbox"),
        ("xlrs-bench-grounding-en", "xlrs_grounding_test.jsonl", None),
    ],
    "change": [
        ("levir-cc", "levircc_test.jsonl", None),
    ],
    "caption": [
        ("vrsbench-caption", "vrsbench_eval.jsonl", "caption"),
        ("xlrs-bench-caption", "xlrs_caption_en.jsonl", None),
    ],
}

# 模型: 量化方式 + 专家 -> 模型引用(本地路径或 get_model 别名)
# general/ground 用 full 版(含 LoRA); change/caption 无 LoRA(架构如此)。full 优先本地 models/。
_M = REPO_ROOT / "models"
QUANT_EXPERTS = {
    "bf16": {"general": str(_M / "expert_general_full"), "grounding": str(_M / "expert_ground_full"),
             "change": "expert_change", "caption": "expert_caption"},
    "w8a8": {"general": str(_M / "expert_general_full_w8a8"), "grounding": str(_M / "expert_ground_full_w8a8"),
             "change": "expert_change_w8a8", "caption": "expert_caption_w8a8"},
    "gptq": {"general": str(_M / "expert_general_full_gptq"), "grounding": str(_M / "expert_ground_full_gptq"),
             "change": "expert_change_gptq", "caption": "expert_caption_gptq"},
}


# 源清单文件名 -> rsmllm.data 数据集键(prapare_eval 用)
SRC_TO_DATASET = {
    "vrsbench_eval.jsonl": "vrsbench",
    "mme_rs.jsonl": "mme",
    "xlrs.jsonl": "xlrs",
    "xlrs_grounding_test.jsonl": "xlrs_grounding",
    "levircc_test.jsonl": "levircc",
    "xlrs_caption_en.jsonl": "xlrs_caption",
}

# router_eval 任务 -> (evaluation.main 的 datasets, --subtask)
# 依据报告 §7/§5.3 路由映射; subtask 可选 vqa/caption/referring/mcq/change
TASK_TO_MAIN = {
    "vrsbench-vqa": ("vrsbench", "vqa"),
    "mme": ("mme", "mcq"),
    "xlrs-bench-lite": ("xlrs", "mcq"),
    "vrsbench-referring": ("vrsbench", "referring"),
    "xlrs-bench-grounding-en": ("xlrs_grounding", None),
    "levir-cc": ("levircc", "change"),
    "vrsbench-caption": ("vrsbench", "caption"),
    "xlrs-bench-caption": ("xlrs_caption", "caption"),
}


def run_eval(model_path: str, profile: str, dataset: str, subtask: str | None,
             limit: int | None) -> int:
    """经 evaluation.main.py 的 qwen35vl 适配器评测(官方入口, 非 vision_opd 遗留器).

    单专家模型(完整模型) → --adapter qwen35vl --model-path <专家完整模型>
    数据集与子任务: --datasets <ds> [--subtask <subtask>] (--subtask 过滤只评本专家任务)
    """
    main_py = REPO_ROOT / "evaluation" / "main.py"
    cmd = [str(PY), str(main_py),
           "--adapter", "qwen35vl",
           "--model-path", model_path,
           "--datasets", dataset]
    if subtask:
        cmd += ["--subtask", subtask]
    if limit:
        cmd += ["--max-samples", str(limit)]
    out = REPO_ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    cmd += ["--output", str(out / f"{dataset}_{subtask or 'all'}.json")]
    print(f"  → 评测 dataset={dataset} subtask={subtask} ...", flush=True)
    return subprocess.run(cmd, check=False).returncode


def resolve_model(alias: str) -> str:
    """get_model 解析别名到本地缓存路径."""
    from rsmllm.models import get_model
    return get_model(alias)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quant", choices=list(QUANT_EXPERTS), default="bf16",
                    help="量化方式(只允许选这个, 其余自动分配)")
    ap.add_argument("--limit", type=int, help="每任务最多样本(验证用)")
    ap.add_argument("--list", action="store_true", help="打印映射表")
    args = ap.parse_args()

    if args.list:
        for expert, tasks in ROUTE_PLAN.items():
            print(f"{expert}: " + ", ".join(t[0] for t in tasks))
        return 0

    rc = 0
    for expert, tasks in ROUTE_PLAN.items():
        alias = QUANT_EXPERTS[args.quant][expert]
        print(f"[router-eval] 量化 {args.quant} / 专家 {expert} → 模型 {alias}")
        model_path = resolve_model(alias)
        for task_name, src_name, task_type in tasks:
            # 首次自动准备: 图片/清单懒加载
            from rsmllm.data import prepare_eval
            if src_name in SRC_TO_DATASET:
                prepare_eval(SRC_TO_DATASET[src_name])
            dataset, subtask = TASK_TO_MAIN.get(task_name, (None, None))
            if dataset is None:
                print(f"  [skip] {task_name}: 未映射到 evaluation.main 数据集")
                continue
            print(f"\n[{expert}] {task_name} → dataset={dataset} subtask={subtask} ...")
            r = run_eval(model_path, alias, dataset, subtask, args.limit)
            rc = max(rc, r)
    print(f"\n[router-eval] 完成. 结果在 results/ (按数据集+子任务 json)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
