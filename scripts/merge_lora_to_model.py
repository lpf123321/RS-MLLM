#!/usr/bin/env python3
"""把 PEFT LoRA 合并进 delta-merged 专家模型, 得到完整专家模型(供评测器离线加载).

队友架构: basemodel + expert(delta) + expert_lora(PEFT)
  - delta 已合并进 ModelScope 的 expert_* 模型(base+delta)
  - LoRA 单独托管(Fun10165/rs-mllm-expert-{general,ground}-lora)
  - 本脚本: 合并 LoRA -> 完整模型, 供评测器(离线 LLM 不支持 LoRA)直接加载

用法:
  python scripts/merge_lora_to_model.py --expert general [--output models/expert_general_full]
  python scripts/merge_lora_to_model.py --expert grounding
  python scripts/merge_lora_to_model.py --all

依赖: 训练环境(.venv, transformers + peft), 模型经 get_model 拉取
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

EXPERTS = {
    "general": ("expert_general", "expert_general_lora", "expert_general_full"),
    "grounding": ("expert_ground", "expert_ground_lora", "expert_ground_full"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expert", choices=list(EXPERTS) + ["all"])
    ap.add_argument("--output", help="输出目录(默认 models/<name>_full)")
    args = ap.parse_args()

    from rsmllm.models import get_model
    from peft import PeftModel
    from transformers import AutoProcessor
    from transformers import Qwen3_5ForConditionalGeneration
    import torch

    targets = list(EXPERTS) if args.expert == "all" else [args.expert]
    for name in targets:
        base_alias, lora_alias, default_out = EXPERTS[name]
        model_dir = get_model(base_alias)
        lora_dir = get_model(lora_alias)
        out_dir = Path(args.output or (REPO_ROOT / "models" / default_out))
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"[merge] {name}: 输出已存在, 跳过 -> {out_dir}")
            continue
        print(f"[merge] {name}: base+delta={model_dir}")
        print(f"[merge] {name}: LoRA={lora_dir}")
        print(f"[merge] 加载模型 + LoRA ...")
        # 必须用完整多模态模型类(含视觉编码器); AutoModelForCausalLM 会丢 visual tower
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True)
        model = PeftModel.from_pretrained(model, lora_dir)
        print(f"[merge] 合并 LoRA ...")
        model = model.merge_and_unload()
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[merge] 保存完整模型 -> {out_dir}")
        model.save_pretrained(out_dir, safe_serialization=True)
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        processor.save_pretrained(out_dir)
        print(f"[merge] {name} 完成: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
