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
MANIFESTS = EVAL_DIR / "manifests"

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

# 模型: 量化方式 + 专家 -> (模型路径, profile 注册名)
# general/ground 用 full 版(含 LoRA); change/caption 无 LoRA(架构如此)。full 优先本地 models/。
_M = REPO_ROOT / "models"
QUANT_EXPERTS = {
    "bf16": {"general": (str(_M / "expert_general_full"), "expert_general_full"),
             "grounding": (str(_M / "expert_ground_full"), "expert_ground_full"),
             "change": ("expert_change", "expert_change"),
             "caption": ("expert_caption", "expert_caption")},
    "w8a8": {"general": (str(_M / "expert_general_full_w8a8"), "expert_general_full_w8a8"),
             "grounding": (str(_M / "expert_ground_full_w8a8"), "expert_ground_full_w8a8"),
             "change": ("expert_change_w8a8", "expert_change_w8a8"),
             "caption": ("expert_caption_w8a8", "expert_caption_w8a8")},
    "gptq": {"general": (str(_M / "expert_general_full_gptq"), "expert_general_full_gptq"),
             "grounding": (str(_M / "expert_ground_full_gptq"), "expert_ground_full_gptq"),
             "change": ("expert_change_gptq", "expert_change_gptq"),
             "caption": ("expert_caption_gptq", "expert_caption_gptq")},
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


def build_subtask_manifest(src_name: str, task_type: str | None, label: str) -> Path:
    """从源 manifest 切出子任务子集(按 task_type), 返回子清单路径."""
    src = MANIFESTS / src_name
    if task_type is None:
        return src
    out = MANIFESTS / f"{Path(src_name).stem}__{label}.jsonl"
    written = 0
    with src.open(encoding="utf-8") as f, out.open("w", encoding="utf-8") as g:
        for line in f:
            d = json.loads(line)
            if d.get("task_type") == task_type:
                g.write(line)
                written += 1
    print(f"  子任务清单: {out.name} ({written} samples)")
    return out


def run_eval(model_path: str, profile: str, manifest: Path, limit: int | None) -> int:
    if limit:
        # 子清单截断(评测器无 --max-samples)
        sub = manifest.with_name(f"{manifest.stem}_n{limit}.jsonl")
        with manifest.open(encoding="utf-8") as f, sub.open("w", encoding="utf-8") as g:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                g.write(line)
        manifest = sub
    cmd = [str(PY), str(EVAL_DIR / "vision_opd_vllm_eval.py"),
           "--manifest", str(manifest),
           "--model", model_path,
           "--model-profile", profile,
           "--min-pixels", "200704",
           "--max-pixels", "2097152",
           "--batch-size", "128",
           "--enforce-eager"]  # vllm CUDA graph 在 bbox 任务卡死; eager 规避
    print(f"  → 评测 {manifest.name} ...", flush=True)
    return subprocess.run(cmd, check=False).returncode


def resolve_model(alias: str) -> str:
    """get_model 解析别名到本地缓存路径."""
    from rsmllm.models import get_model
    return get_model(alias)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quant", choices=list(QUANT_EXPERTS), default="bf16",
                    help="量化方式(只允许选这个, 其余自动分配)")
    ap.add_argument("--experts", nargs="+", choices=list(ROUTE_PLAN),
                    help="只跑指定专家(默认全部; 断点续跑用)")
    ap.add_argument("--limit", type=int, help="每任务最多样本(验证用)")
    ap.add_argument("--list", action="store_true", help="打印映射表")
    args = ap.parse_args()

    if args.list:
        for expert, tasks in ROUTE_PLAN.items():
            print(f"{expert}: " + ", ".join(t[0] for t in tasks))
        return 0

    rc = 0
    experts = args.experts or list(ROUTE_PLAN)
    for expert in experts:
        tasks = ROUTE_PLAN[expert]
        model_ref, profile = QUANT_EXPERTS[args.quant][expert]
        print(f"[router-eval] 量化 {args.quant} / 专家 {expert} → 模型 {model_ref} (profile {profile})")
        model_path = resolve_model(model_ref)
        for task_name, src_name, task_type in tasks:
            # 首次自动准备: 图片下载 + 可移植评测清单构建(vrsbench 等 6 数据集)
            from rsmllm.data import prepare_eval
            if src_name in SRC_TO_DATASET:
                prepare_eval(SRC_TO_DATASET[src_name])
            print(f"\n[{expert}] {task_name} ...")
            manifest = build_subtask_manifest(src_name, task_type, task_name.replace("-", "_"))
            r = run_eval(model_path, profile, manifest, args.limit)
            rc = max(rc, r)
    print(f"\n[router-eval] 完成. 结果在 evaluation/vllm_eval/results/ (按任务独立时间戳目录)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
