#!/usr/bin/env python3
"""ModelScope 批量上传: RS-MLLM 全部模型 -> Fun10165/* (一模型一仓库).

仓库映射(model_id, 本地目录): 基座/专家 bf16 在 slurm 共享; 量化产物在 ~/models(xd1) 或
~/tiled_rag/compressed_models(slurm). 按实际存在的路径上传.

用法:
  export MODELSCOPE_API_TOKEN=xxx
  python scripts/upload_to_modelscope.py --local-root <root> [--dry-run] [--only base,expert_general,...]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# model_id -> (仓库别名, 若干候选本地路径(按存在性取第一个))
PLAN = {
    "base":                ("Fun10165/qwen-3.5-rs",
                            [Path("/users/u2024311136/shared/shared_models/lora_expert/base_model")]),
    "expert_general":      ("Fun10165/rs-mllm-expert-general",
                            [Path("/users/u2024311136/shared/shared_models/lora_expert/_merge/general_exp7_merged")]),
    "expert_ground":       ("Fun10165/rs-mllm-expert-ground",
                            [Path("/users/u2024311136/shared/shared_models/lora_expert/_merge/ground_expert_update_merged")]),
    "expert_change":       ("Fun10165/rs-mllm-expert-change",
                            [Path("/users/u2024311136/shared/shared_models/lora_expert/_merge/change_merged")]),
    "expert_caption":      ("Fun10165/rs-mllm-expert-caption",
                            [Path("/users/u2024311136/shared/shared_models/lora_expert/_merge/caption_merged")]),
    "mmerestore_bf16":     ("Fun10165/rs-mllm-mmerestore-bf16",
                            [Path("/home/u2024311009/tiled_rag/model_mmerestore")]),
    "mmerestore_w8a8":     ("Fun10165/rs-mllm-mmerestore-w8a8-int8",
                            [Path("/home/u2024311009/tiled_rag/compressed_models/model_mmerestore_w8a8_int8")]),
    "mmerestore_gptq":     ("Fun10165/rs-mllm-mmerestore-w4a16-gptq",
                            [Path("/home/u2024311009/tiled_rag/compressed_models/model_mmerestore_w4a16_gptq")]),
}
# 量化产物(xd1 ~/models 或 slurm compressed_models)
for expert in ("general", "ground", "change", "caption"):
    for qual, tag in (("w8a8", "w8a8-int8"), ("gptq", "w4a16-gptq")):
        d = f"expert_{expert}_{qual}"
        PLAN[d] = (f"Fun10165/rs-mllm-expert-{expert}-{tag}",
                   [Path.home() / f"models/expert_{expert}_{qual}",
                    Path(f"/home/u2024311009/tiled_rag/compressed_models/expert_{expert}_{qual}")])

LOCAL_ROOT_FLAG = True  # 本地候选路径已是绝对路径; --local-root 为附加前缀


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", help="逗号分隔别名子集")
    ap.add_argument("--local-root", help="附加路径前缀(相对模型名时)")
    args = ap.parse_args()

    token = args.token or os.environ.get("MODELSCOPE_API_TOKEN")
    if not token:
        print("缺少 token: --token xxx 或 MODELSCOPE_API_TOKEN", file=sys.stderr)
        return 1

    from modelscope import HubApi
    api = HubApi()
    api.login(token)

    selected = set(args.only.split(",")) if args.only else set(PLAN)
    for name, (model_id, candidates) in PLAN.items():
        if name not in selected:
            continue
        src = None
        for c in candidates:
            if c.is_dir():
                src = c
                break
        if src is None:
            print(f"[skip] {name}: 本地路径均不存在", file=sys.stderr)
            continue
        if args.dry_run:
            n = sum(1 for _ in src.rglob("*") if _.is_file())
            print(f"[dry-run] {name}: {model_id} <- {src} ({n} files)")
            continue
        print(f"[upload] {name}: {model_id} <- {src}", flush=True)
        try:
            api.push_model(model_id=model_id, model_dir=str(src))
            print(f"[ok] {name} done", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[error] {name}: {str(e)[:150]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
