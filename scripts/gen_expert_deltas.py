#!/usr/bin/env python3
"""Gen delta weights: expert_merged - W0, unified on base Qwen3.5-4B.

生成各专家相对原始基座 W0 的完整 delta，并验证 W0+Δ 能精确还原 merged 模型。

四个专家（均可与 W0 同架构对齐，跳过 mtp 层）：
    grounding : outputs/merged/a1_grounding_checkpoint-1992_2048
    change    : outputs/merged/a2b_change_checkpoint-581
    v2 general: outputs/merged/expert_general_v2
    caption   : outputs/merged/caption_expert_checkpoint-158

用法:
    python scripts/gen_expert_deltas.py --verify
"""
import argparse
import json
import os
from collections import OrderedDict

import torch
from safetensors import safe_open

W0_DIR = "models/Qwen3.5-4B"
OUT_DIR = "outputs/merged/expert_deltas_rel_W0"
MTPSKIP = ("mtp.",)  # W0 独有层

EXPERTS = {
    "grounding": "outputs/merged/a1_grounding_checkpoint-1992_2048",
    "change": "outputs/merged/a2b_change_checkpoint-581",
    "general_v2": "outputs/merged/expert_general_v2",
    "caption": "outputs/merged/caption_expert_checkpoint-158",
}


def load0(k):
    with safe_open(f"{W0_DIR}/{bin_of[k]}", framework="pt", device="cpu") as f:
        return f.get_tensor(k)


def slurp(path, device="cpu"):
    """Load full model tensors from a merged model path.

    If path points to a directory with an index.json, read all shards
    (sharded safetensors). Otherwise treat path as a single safetensors file.
    """
    if os.path.isdir(path):
        idx = json.load(open(f"{path}/model.safetensors.index.json"))["weight_map"]
        out = {}
        for fn in sorted(set(idx.values())):
            with safe_open(f"{path}/{fn}", framework="pt", device=device) as f:
                for k in f.keys():
                    out[k] = f.get_tensor(k)
        return out
    with safe_open(path, framework="pt", device=device) as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    global bin_of
    w0_idx = json.load(open(f"{W0_DIR}/model.safetensors.index.json"))["weight_map"]
    bin_of = w0_idx  # key -> filename

    # 统一 key 集：用 一个专家 的 key 为公共集（相对 W0 取交集，跳过 mtp）
    ref = slurp(EXPERTS["grounding"])
    common = sorted(k for k in ref if not k.startswith(MTPSKIP))
    print(f"expert keys (skipping mtp): {len(common)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    summaries = {}
    for name, path in EXPERTS.items():
        expert = slurp(path)
        delta = OrderedDict()
        maxdiff = 0.0
        for k in common:
            dev = expert[k].float()
            w0 = load0(k).to(dev.dtype)
            d = dev - w0
            delta[k] = d
        torch.save(delta, f"{OUT_DIR}/{name}_relW0.pt")
        # 统计非零
        nz = sum(1 for v in delta.values() if torch.count_nonzero(v) > 0)
        summaries[name] = {"tensors": len(delta), "nonzero": nz}
        print(f"  {name}: delta saved ({nz}/{len(delta)} nonzero tensors)")

    print("---")
    print(f"Deltas written to {OUT_DIR}/  (as .pt state dicts, dtype float32)")

    if args.verify:
        print("\n=== VERIFY: W0 + delta == expert_merged ===")
        for name, path in EXPERTS.items():
            delta = torch.load(f"{OUT_DIR}/{name}_relW0.pt", map_location="cpu")
            expert = slurp(path)
            maxdiff = 0.0
            bad = 0
            for k in common:
                w0 = load0(k).float()
                recon = w0 + delta[k]
                dev = expert[k].float()
                m = (recon - dev).abs().max().item()
                maxdiff = max(maxdiff, m)
                if not torch.allclose(recon, dev, atol=1e-3):
                    bad += 1
            print(f"  {name}: max|recon-expert|={maxdiff:.2e}  mismatched@1e-3={bad}")
        print("  (差异应仅来自 fp16 存储舍入，<1e-2 即可认定还原成功)")


if __name__ == "__main__":
    main()