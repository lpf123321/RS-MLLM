#!/usr/bin/env python3
"""把 PEFT LoRA 合并进真正 delta-only 专家模型, 得到完整模型.

当前发布的 `expert_general` / `expert_ground` 快照已经记录并包含一次
`base + expert_delta + PEFT LoRA` 合并；一键评测直接使用它们。
本脚本只服务于没有该合并记录的自定义 delta-only 输入，并会拒绝对
canonical 快照二次合并，防止 LoRA 被重复应用。

用法:
  # canonical 别名带有合并记录，下面命令会安全拒绝二次合并
  python scripts/merge_lora_to_model.py --expert general
  # 对真正 delta-only 的自定义目录执行一次合并
  python scripts/merge_lora_to_model.py --expert general \
      --base-model /path/to/delta-only --lora-path /path/to/adapter \
      --output /path/to/assembled
  python scripts/merge_lora_to_model.py --all

依赖: 训练环境(.venv, transformers + peft), 模型经 get_model 拉取
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

EXPERTS = {
    "general": ("expert_general", "expert_general_lora", "expert_general_full"),
    "grounding": ("expert_ground", "expert_ground_lora", "expert_ground_full"),
}


def _model_dir_ready(path: Path) -> bool:
    """Recognize a complete flat Transformers model output."""
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


def _has_peft_merge_record(path: Path) -> bool:
    """Whether provenance explicitly says a PEFT adapter was already merged."""
    marker = path / "merge_manifest.json"
    if not marker.is_file():
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise RuntimeError(f"无法解析模型合并记录，拒绝继续以免重复合并: {marker}")
    order = data.get("merge_order") or []
    return bool(data.get("peft_lora")) or (
        isinstance(order, list) and "peft_lora" in order
    )


def _output_for(
    expert: str, output: str | None, *, multiple: bool
) -> Path:
    default = Path("models") / EXPERTS[expert][2]
    if output is None:
        return REPO_ROOT / default
    requested = Path(output).expanduser()
    return requested / EXPERTS[expert][2] if multiple else requested


def _merge_one(
    expert: str,
    *,
    model_dir: Path,
    lora_dir: Path,
    out_dir: Path,
    base_alias: str,
    lora_alias: str,
) -> None:
    if _has_peft_merge_record(model_dir):
        raise RuntimeError(
            f"{model_dir} 已记录 PEFT LoRA 合并，拒绝二次合并；"
            f"请直接使用 canonical 模型别名 {base_alias!r}"
        )
    if out_dir.exists():
        if _model_dir_ready(out_dir):
            print(f"[merge] {expert}: 输出已存在且完整, 跳过 -> {out_dir}")
            return
        if any(out_dir.iterdir()):
            raise RuntimeError(
                f"输出目录存在但不完整, 为避免覆盖用户文件而停止: {out_dir}"
            )

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = out_dir.with_name(f"{out_dir.name}.staging-{os.getpid()}")
    if staging.exists():
        raise RuntimeError(f"临时输出目录已存在, 请确认没有并发合并: {staging}")

    try:
        from peft import PeftModel
        from transformers import AutoProcessor
        from transformers import Qwen3_5ForConditionalGeneration
        import torch

        print(f"[merge] {expert}: base+delta={model_dir}")
        print(f"[merge] {expert}: LoRA={lora_dir}")
        print("[merge] 加载模型 + LoRA ...", flush=True)
        # 必须用完整多模态模型类(含视觉编码器); AutoModelForCausalLM 会丢 visual tower
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(model, str(lora_dir), is_trainable=False)
        print("[merge] 合并 LoRA ...", flush=True)
        model = model.merge_and_unload(safe_merge=True)
        model.eval()

        staging.mkdir(parents=True)
        print(f"[merge] 保存完整模型 -> {staging}", flush=True)
        model.save_pretrained(staging, safe_serialization=True)
        processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
        processor.save_pretrained(staging)
        (staging / "merge_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "expert": expert,
                    "base_model_alias": base_alias,
                    "base_model_path": str(model_dir),
                    "lora_alias": lora_alias,
                    "lora_path": str(lora_dir),
                    "merge_order": ["base_plus_expert_delta", "peft_lora"],
                    "merge_method": "PeftModel.merge_and_unload(safe_merge=True)",
                    "model_class": "Qwen3_5ForConditionalGeneration",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if not _model_dir_ready(staging):
            raise RuntimeError(f"合并输出不完整: {staging}")
        staging.rename(out_dir)
        print(f"[merge] {expert} 完成: {out_dir}", flush=True)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expert", choices=list(EXPERTS) + ["all"], required=True)
    ap.add_argument(
        "--output",
        help="单专家输出目录; --expert all 时作为输出根目录(默认 models/<name>_full)",
    )
    ap.add_argument(
        "--base-model",
        help="显式 delta-only 模型目录; 与 --lora-path 必须同时提供",
    )
    ap.add_argument(
        "--lora-path",
        help="显式 PEFT adapter 目录; 与 --base-model 必须同时提供",
    )
    args = ap.parse_args()

    if bool(args.base_model) != bool(args.lora_path):
        ap.error("--base-model 与 --lora-path 必须同时提供")
    if args.expert == "all" and (args.base_model or args.lora_path):
        ap.error("--expert all 不能与显式 --base-model/--lora-path 一起使用")

    from rsmllm.models import get_model

    targets = list(EXPERTS) if args.expert == "all" else [args.expert]
    for name in targets:
        base_alias, lora_alias, _default_out = EXPERTS[name]
        if args.base_model:
            model_dir = Path(args.base_model).expanduser()
            lora_dir = Path(args.lora_path).expanduser()
            if not model_dir.is_dir():
                raise FileNotFoundError(f"delta-only 模型目录不存在: {model_dir}")
            if not lora_dir.is_dir():
                raise FileNotFoundError(f"LoRA 目录不存在: {lora_dir}")
            base_alias = "explicit_delta_only_path"
            lora_alias = "explicit_lora_path"
        else:
            model_dir = Path(get_model(base_alias))
            lora_dir = Path(get_model(lora_alias))
        out_dir = _output_for(name, args.output, multiple=len(targets) > 1)
        _merge_one(
            name,
            model_dir=model_dir,
            lora_dir=lora_dir,
            out_dir=out_dir,
            base_alias=base_alias,
            lora_alias=lora_alias,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
