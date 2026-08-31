#!/usr/bin/env python3
"""Load expert model via: base Qwen3.5-4B + expert delta (置换).

单基座 + 完整 delta 置换加载器：
    model = load_base_with_delta(expert='grounding')
    等价于 from_pretrained(outputs/merged/a1_grounding_checkpoint-1992_2048)，
    但共享一份 W0 权重，按任务叠加 full-rank delta。

Delta 文件: outputs/merged/expert_deltas_rel_W0/<expert>_relW0.pt
"""
import os

import torch


def load_base_with_delta(expert, deltas_dir="outputs/merged/expert_deltas_rel_W0"):
    from transformers import Qwen3_5ForConditionalGeneration

    base_path = "models/Qwen3.5-4B"
    delta_path = f"{deltas_dir}/{expert}_relW0.pt"
    if not os.path.exists(delta_path):
        raise FileNotFoundError(delta_path)

    print(f"  loading base {base_path} ...")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        base_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True, low_cpu_mem_usage=True)
    print(f"  applying delta {expert} ...")
    delta = torch.load(delta_path, map_location="cpu")
    model = apply_delta(model, delta)
    return model


def apply_delta(model, delta: dict):
    """把 {weight_name: float_tensor} delta 叠到 model 上（就地）。"""
    sd = model.state_dict()
    applied = 0
    for k, d in delta.items():
        if k not in sd:
            continue
        param = sd[k]
        d = d.to(param.dtype).to(param.device)
        # 就地加（避免重新构造 state_dict 大拷贝）
        sd[k].add_(d)
        applied += 1
    print(f"  applied delta to {applied} tensors")
    return model


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", required=True, choices=["grounding", "change", "general_v2"])
    ap.add_argument("--save", default=None, help="save assembled model to dir")
    args = ap.parse_args()
    m = load_base_with_delta(args.expert)
    if args.save:
        from transformers import AutoProcessor
        os.makedirs(args.save, exist_ok=True)
        proc = AutoProcessor.from_pretrained("models/Qwen3.5-4B", trust_remote_code=True)
        m.save_pretrained(args.save, safe_serialization=True)
        proc.save_pretrained(args.save)
        print("saved to", args.save)
    print("done")
