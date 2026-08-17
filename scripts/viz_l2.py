#!/usr/bin/env python3
"""L2 剪枝可视化：对真实图像跑 L2 剪枝，把保留/剪掉的 token 叠加回原图。

复用 SCOPE 的 raw-feature hook 机制（vision block 22 捕获 1024-dim raw 特征），
在 2x2 merge block 级别做 L2-norm top-k 打分，然后用 prune.viz 叠加可视化。

用法:
    python scripts/viz_l2.py [--r 0.5] [--out_dir prune/output/viz_l2]
"""

import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from scripts.prune import apply_pruning
from prune.scope import enable_scope_hooks
from prune.l2norm import L2NormPruner
from prune.viz import visualize_pruning

MODEL_PATH = "/home/u2024311149/models/Qwen3.5-4B"
DATA_ROOT = os.environ.get("DATA_ROOT", "/home/u2024311149/RS-MLLM/datasets/shared_datasets")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r", type=float, default=0.5, help="剪枝比例（0.5=保留50%）")
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--out_dir", default="prune/output/viz_l2")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── 采样若干 MME 样本 ──
    samples = []
    mme_path = f"{DATA_ROOT}/MME-RealWorld-RS/mme_rs.jsonl"
    if os.path.exists(mme_path):
        with open(mme_path) as f:
            for line in f:
                samples.append(json.loads(line))
    else:
        # fallback: 用项目里的 test.png
        samples = [{"messages": json.dumps([{"role": "user", "content": [
            {"type": "image", "image": "/home/u2024311149/RS-MLLM/test.png"},
            {"type": "text", "text": "Describe this image."}]}])}]

    import random
    random.seed(42)
    random.shuffle(samples)
    samples = samples[: args.n_samples]

    # ── 加载模型 ──
    print("[*] 加载模型 ...")
    apply_pruning(model_path=MODEL_PATH)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    backbone = model.model
    enable_scope_hooks(backbone.visual)

    pruner = L2NormPruner()

    print(f"[*] L2 剪枝 R={args.r}（保留 {(1-args.r)*100:.0f}% merged token）")

    for idx, s in enumerate(samples):
        msgs = s["messages"]
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        uc = msgs[0]["content"]
        text = next(c["text"] for c in uc if c["type"] == "text")
        imgs = [c["image"] for c in uc if c["type"] == "image"]

        msg = [{"role": "user", "content": [
            *[{"type": "image", "image": i} for i in imgs],
            {"type": "text", "text": text},
        ]}]
        prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        img_in, _ = process_vision_info(msg)
        inputs = processor(text=[prompt], images=img_in, padding=True, return_tensors="pt")
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        grid_thw = inputs["image_grid_thw"]
        hb = int(grid_thw[0, 1].item() // 2)
        wb = int(grid_thw[0, 2].item() // 2)
        n_merged = hb * wb

        # 跑 vision encoder 拿 raw features
        backbone.visual._scope_features = None
        with torch.no_grad():
            backbone.visual(inputs["pixel_values"], grid_thw=grid_thw)
        sf = backbone.visual._scope_features
        if sf is None:
            print(f"  [{idx}] 未捕获 raw features，跳过")
            continue

        raw = sf[: n_merged * 4]
        bf = raw.view(n_merged, 4, -1).mean(dim=1)  # [n_merged, 1024] block features

        # L2 top-k
        keep_count = max(1, int(n_merged * (1.0 - args.r)))
        keep_mask = pruner.keep_mask(bf, keep_count)  # [n_merged] bool
        grid_mask = keep_mask.view(hb, wb)

        # 可视化
        img_path = imgs[0]
        image = Image.open(img_path).convert("RGB")
        out_path = os.path.join(args.out_dir, f"l2_r{int(args.r*100):02d}_{idx}.png")
        visualize_pruning(
            image, grid_mask,
            title=f"L2 pruning R={args.r} (keep {keep_count}/{n_merged})",
            save_path=out_path,
        )
        print(f"  [{idx}] grid={hb}x{wb}, keep={keep_count}/{n_merged}, saved -> {out_path}")

    print(f"\n[+] 可视化输出: {args.out_dir}/")


if __name__ == "__main__":
    main()
