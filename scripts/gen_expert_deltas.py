#!/usr/bin/env python3
"""Gen delta weights: expert_merged - W0, unified on base Qwen3.5-4B.

生成各专家相对原始基座 W0 的完整 delta，并验证 W0+Δ 能精确还原 merged 模型。

四个专家（均可与 W0 同架构对齐，跳过 mtp 层）：
    expert_ground : models/expert_ground
    expert_change : models/expert_change
    expert_general: models/expert_general
    expert_caption: models/expert_caption

用法:
    python scripts/gen_expert_deltas.py --verify
"""
import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors import safe_open

W0_DIR = "models/Qwen3.5-4B"
OUT_DIR = "models/training35/deltas"
MTPSKIP = ("mtp.",)  # W0 独有层

EXPERTS = {
    "expert_ground": "models/expert_ground",
    "expert_change": "models/expert_change",
    "expert_general": "models/expert_general",
    "expert_caption": "models/expert_caption",
}


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
    ap.add_argument("--base", default=W0_DIR)
    ap.add_argument("--output", default=OUT_DIR)
    ap.add_argument("--force", action="store_true", help="覆盖已有 delta")
    ap.add_argument("--tolerance", type=float, default=1e-2)
    args = ap.parse_args()

    base_dir = args.base
    out_dir = args.output
    w0_idx = json.load(open(f"{base_dir}/model.safetensors.index.json"))["weight_map"]
    base = slurp(base_dir)
    common = sorted(k for k in w0_idx if not k.startswith(MTPSKIP))
    print(f"expert keys (skipping mtp): {len(common)}")

    os.makedirs(out_dir, exist_ok=True)
    summaries = {}
    for name, path in EXPERTS.items():
        delta_path = f"{out_dir}/{name}.pt"
        if os.path.exists(delta_path) and not args.force:
            print(f"  {name}: reuse existing {delta_path}")
            continue
        expert = slurp(path)
        delta = OrderedDict()
        for k in common:
            dev = expert[k]
            w0 = base[k].to(dev.dtype)
            delta[k] = (dev - w0).contiguous()
        temporary = f"{delta_path}.tmp"
        torch.save(delta, temporary)
        os.replace(temporary, delta_path)
        # 统计非零
        nz = sum(1 for v in delta.values() if torch.count_nonzero(v) > 0)
        summaries[name] = {"tensors": len(delta), "nonzero": nz}
        print(f"  {name}: delta saved ({nz}/{len(delta)} nonzero BF16 tensors)")
        del expert, delta

    print("---")
    print(f"Deltas written to {out_dir}/  (as .pt state dicts, dtype bfloat16)")

    if args.verify:
        print("\n=== VERIFY: W0 + delta == expert_merged ===")
        failed = False
        verification = {}
        for name, path in EXPERTS.items():
            delta = torch.load(
                f"{out_dir}/{name}.pt",
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            expert = slurp(path)
            maxdiff = 0.0
            bad = 0
            for k in common:
                w0 = base[k].to(delta[k].dtype)
                recon = (w0 + delta[k]).float()
                dev = expert[k].float()
                error = (recon - dev).abs().max().item()
                maxdiff = max(maxdiff, error)
                if error > args.tolerance:
                    bad += 1
            failed |= bad > 0
            verification[name] = {
                "tensors": len(common),
                "max_abs_reconstruction_error": maxdiff,
                "mismatched_tensors": bad,
            }
            print(f"  {name}: max|recon-expert|={maxdiff:.2e}  mismatched={bad}")
            del expert, delta
        print(f"  tolerance={args.tolerance:g}（仅允许 BF16 舍入差异）")
        report = {
            "base": str(Path(base_dir).resolve()),
            "tolerance": args.tolerance,
            "passed": not failed,
            "experts": verification,
        }
        report_path = Path(out_dir) / "verification.json"
        report_path.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(f"  report={report_path}")
        if failed:
            raise SystemExit("delta verification failed")


if __name__ == "__main__":
    main()