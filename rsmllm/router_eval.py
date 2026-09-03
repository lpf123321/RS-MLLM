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

模型: 只选择量化方式; 专家模型与任务自动分配。
  ModelScope 的 general/grounding 专家快照本身已经是
  base+expert-delta+PEFT LoRA 的一次合并结果，再交给 vLLM；量化版使用
  从这些快照生成的 W8A8/GPTQ 产物。change/caption 没有 LoRA。
  不再把 adapter 二次挂载或二次合并。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from rsmllm.config import MODELS_ROOT, REPORT_CONF
from rsmllm.eval_reporting import print_combined_key_metrics

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "evaluation" / "vllm_eval"
PY = EVAL_DIR / ".venv" / "bin" / "python"
if not PY.is_file():
    PY = Path(sys.executable)
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

# 模型: 量化方式 + 专家 -> (模型别名, trusted profile 注册名)
# general/ground 的 canonical 快照已包含一次 PEFT LoRA 合并；不要选旧的
# *_full 目录（它们是历史上的二次合并产物）。change/caption 没有 LoRA。
# 别名而不是硬编码绝对路径，确保新机器会走 ModelScope 懒加载。
QUANT_EXPERTS = {
    "bf16": {"general": ("expert_general", "expert_general"),
             "grounding": ("expert_ground", "expert_ground"),
             "change": ("expert_change", "expert_change"),
             "caption": ("expert_caption", "expert_caption")},
    "w8a8": {"general": ("expert_general_w8a8", "expert_general_w8a8"),
             "grounding": ("expert_ground_w8a8", "expert_ground_w8a8"),
             "change": ("expert_change_w8a8", "expert_change_w8a8"),
             "caption": ("expert_caption_w8a8", "expert_caption_w8a8")},
    "gptq": {"general": ("expert_general_gptq", "expert_general_gptq"),
             "grounding": ("expert_ground_gptq", "expert_ground_gptq"),
             "change": ("expert_change_gptq", "expert_change_gptq"),
             "caption": ("expert_caption_gptq", "expert_caption_gptq")},
}

# Known local artifacts are preferred over a second download.  The alternate
# ~/models names are the existing xd1 conversion outputs; they are selected
# only when their canonical profile is used and the layout is complete.
_LOCAL_ARTIFACT_NAMES = {
    "expert_general": ("expert_general", "general_exp7_merged"),
    "expert_ground": ("expert_ground", "ground_expert_update_merged"),
    "expert_general_w8a8": ("expert_general_w8a8", "expert_general_w8a8_int8"),
    "expert_ground_w8a8": ("expert_ground_w8a8", "expert_ground_w8a8_int8"),
    "expert_general_gptq": ("expert_general_gptq", "expert_general_w4a16_gptq"),
    "expert_ground_gptq": ("expert_ground_gptq", "expert_ground_w4a16_gptq"),
    "expert_change_w8a8": ("expert_change_w8a8", "expert_change_w8a8_int8"),
    "expert_change_gptq": ("expert_change_gptq", "expert_change_w4a16_gptq"),
    "expert_caption_w8a8": ("expert_caption_w8a8", "expert_caption_w8a8_int8"),
    "expert_caption_gptq": ("expert_caption_gptq", "expert_caption_w4a16_gptq"),
}
LOCAL_ARTIFACTS = {
    alias: tuple(
        root / name
        for root in (MODELS_ROOT, Path.home() / "models")
        for name in names
    )
    for alias, names in _LOCAL_ARTIFACT_NAMES.items()
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
    from rsmllm.data import validate_eval_manifest

    src = MANIFESTS / src_name
    validate_eval_manifest(src)
    if task_type is None:
        return src
    out = MANIFESTS / f"{Path(src_name).stem}__{label}.jsonl"
    tmp = out.with_name(f".{out.name}.tmp.{os.getpid()}")
    written = 0
    try:
        with src.open(encoding="utf-8") as f, tmp.open("w", encoding="utf-8") as g:
            for line in f:
                d = json.loads(line)
                if d.get("task_type") == task_type:
                    g.write(line)
                    written += 1
        tmp.replace(out)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    print(f"  子任务清单: {out.name} ({written} samples)")
    return out


def run_eval(
    model_path: str,
    profile: str,
    manifest: Path,
    limit: int | None,
    quantization: str,
    *,
    output_dir: Path | None = None,
) -> int:
    if limit:
        # 子清单截断(评测器无 --max-samples)
        sub = manifest.with_name(f"{manifest.stem}_n{limit}.jsonl")
        with manifest.open(encoding="utf-8") as f, sub.open("w", encoding="utf-8") as g:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                g.write(line)
        manifest = sub
    runtime = evaluation_runtime_config(manifest)
    if output_dir is None:
        output_dir = new_evaluation_output_dir(manifest, profile)
    cmd = [str(PY), str(EVAL_DIR / "vision_opd_vllm_eval.py"),
           "--manifest", str(manifest),
           "--model", model_path,
           "--output-dir", str(output_dir),
           "--model-profile", profile,
           "--quantization", quantization,
           "--min-pixels", str(runtime["min_pixels"]),
           "--max-pixels", str(runtime["max_pixels"]),
           "--batch-size", str(runtime["batch_size"]),
           "--max-model-len", str(runtime["max_model_len"]),
           "--max-num-seqs", str(runtime["max_num_seqs"]),
           "--image-load-workers", str(REPORT_CONF["image_load_workers"]),
           "--gpu-memory-utilization", str(REPORT_CONF["gpu_memory_utilization"]),
           "--enforce-eager"]  # 当前稳定配置；graph 模式的启动耗时须看完整结果判断
    print(f"  → 评测 {manifest.name} ...", flush=True)
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + existing if existing else ""
    )
    result = subprocess.run(
        cmd,
        check=False,
        cwd=str(REPO_ROOT),
        env=env,
    )
    if result.returncode:
        print(
            f"  ✗ 评测子进程失败 (exit={result.returncode}); "
            "记录失败并继续其余任务，最终返回非零。",
            file=sys.stderr,
            flush=True,
        )
    return result.returncode


def new_evaluation_output_dir(
    manifest: Path, profile: str, limit: int | None = None
) -> Path:
    """Return the exact timestamped directory used for one evaluator child."""
    stem = f"{manifest.stem}_n{limit}" if limit else manifest.stem
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return REPO_ROOT / "results" / f"{stem}_{profile}_{stamp}"


def evaluation_runtime_config(manifest: Path) -> dict[str, int]:
    """Return the task-specific runtime shared by route and single modes."""
    # Exp5's published XLRS Grounding score was produced from the 4096 export.
    # Its Transformers adapter attempted to set ``processor.image_max_pixels``
    # but that was a no-op; the inner image processor retained 16,777,216.
    # Reproduce the *actual* experiment here, and record it explicitly instead
    # of continuing to call it a 1024-downsample run.
    historical_xlrs_grounding = manifest.stem.startswith("xlrs_grounding")
    min_pixels = 65_536 if historical_xlrs_grounding else REPORT_CONF["min_pixels"]
    max_pixels = 16_777_216 if historical_xlrs_grounding else REPORT_CONF["max_pixels"]
    max_model_len = 32_768 if historical_xlrs_grounding else REPORT_CONF["max_model_len"]
    batch_size = 4 if historical_xlrs_grounding else REPORT_CONF["batch_size"]
    max_num_seqs = 4 if historical_xlrs_grounding else REPORT_CONF["max_num_seqs"]
    return {
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "max_model_len": max_model_len,
        "batch_size": batch_size,
        "max_num_seqs": max_num_seqs,
    }


def _model_dir_ready(path: Path) -> bool:
    """Return whether a local directory has a loadable model layout."""
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    if (path / "model.safetensors").is_file():
        return True
    index = path / "model.safetensors.index.json"
    if not index.is_file():
        return False
    try:
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return all((path / name).is_file() for name in set(weight_map.values()))


def _print_lineage(path: Path) -> None:
    """Print small provenance markers without reading model weights."""
    for filename in ("merge_manifest.json", "conversion_manifest.json"):
        marker = path / filename
        if not marker.is_file():
            continue
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print(f"  [model] 警告: 无法读取 {marker}", flush=True)
            continue
        if filename == "merge_manifest.json":
            order = data.get("merge_order") or data.get("merge_method")
            print(f"  [model] lineage: {filename} ({order})", flush=True)
        else:
            print(
                f"  [model] lineage: {filename} (source={data.get('model_source')})",
                flush=True,
            )


def _assert_canonical_lineage(path: Path, expert: str, quant: str) -> None:
    """Fail closed if a LoRA expert lacks its one-merge provenance marker."""
    if quant != "bf16" or expert not in {"general", "grounding"}:
        return
    marker = path / "merge_manifest.json"
    if not marker.is_file():
        raise RuntimeError(
            f"{expert} canonical 模型缺少 merge_manifest.json，拒绝评测以免误用未合并或二次合并权重: {path}"
        )
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法解析 canonical 模型合并记录: {marker}") from exc
    order = data.get("merge_order") or []
    if not (data.get("peft_lora") or (isinstance(order, list) and "peft_lora" in order)):
        raise RuntimeError(
            f"{expert} canonical 模型的合并记录未包含 PEFT LoRA，拒绝评测: {marker}"
        )


def resolve_model(expert: str, quant: str) -> str:
    """Resolve one canonical route entry without applying LoRA twice."""
    from rsmllm.models import get_model

    model_ref, _profile = QUANT_EXPERTS[quant][expert]
    for local in LOCAL_ARTIFACTS.get(model_ref, ()):
        if _model_dir_ready(local):
            _assert_canonical_lineage(local, expert, quant)
            print(f"  [model] 使用本地 canonical 产物: {local}", flush=True)
            _print_lineage(local)
            return str(local)

    resolved = Path(get_model(model_ref))
    _assert_canonical_lineage(resolved, expert, quant)
    _print_lineage(resolved)
    return str(resolved)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quant", choices=list(QUANT_EXPERTS), default="bf16",
                    help="量化方式(只允许选这个, 其余自动分配)")
    ap.add_argument("--experts", nargs="+", choices=list(ROUTE_PLAN),
                    help="只跑指定专家(默认全部; 断点续跑用)")
    ap.add_argument("--limit", type=int, help="每任务最多样本(验证用)")
    ap.add_argument("--list", action="store_true", help="打印映射表")
    args = ap.parse_args()

    if args.limit is not None and args.limit < 1:
        ap.error("--limit 必须 >= 1")

    if args.list:
        for expert, tasks in ROUTE_PLAN.items():
            print(f"{expert}: " + ", ".join(t[0] for t in tasks))
        return 0

    rc = 0
    completed_outputs: list[Path] = []
    experts = args.experts or list(ROUTE_PLAN)
    for expert in experts:
        tasks = ROUTE_PLAN[expert]
        model_ref, profile = QUANT_EXPERTS[args.quant][expert]
        print(f"[router-eval] 量化 {args.quant} / 专家 {expert} → 模型 {model_ref} (profile {profile})")
        model_path = resolve_model(expert, args.quant)
        for task_name, src_name, task_type in tasks:
            # 首次自动准备: 图片下载 + 可移植评测清单构建(vrsbench 等 6 数据集)
            from rsmllm.data import prepare_eval
            if src_name in SRC_TO_DATASET:
                prepare_eval(SRC_TO_DATASET[src_name])
            print(f"\n[{expert}] {task_name} ...")
            manifest = build_subtask_manifest(src_name, task_type, task_name.replace("-", "_"))
            output_dir = new_evaluation_output_dir(
                manifest, profile, args.limit
            )
            r = run_eval(
                model_path,
                profile,
                manifest,
                args.limit,
                args.quant,
                output_dir=output_dir,
            )
            if r == 0:
                completed_outputs.append(output_dir)
            rc = max(rc, r)
    print_combined_key_metrics(completed_outputs)
    print(f"\n[router-eval] 完成. 结果在 {REPO_ROOT / 'results'}/ (按任务独立时间戳目录)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
