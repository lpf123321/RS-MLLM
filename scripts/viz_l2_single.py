#!/usr/bin/env python3
"""对单张图片做 L2 剪枝可视化（R 可指定）。

用法:
    python scripts/viz_l2_single.py --image <path> [--r 0.5] [--out prune/output/viz_l2/single]
"""

import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from scripts.prune import apply_pruning
from prune.scope import enable_scope_hooks
from prune.l2norm import L2NormPruner
from prune.viz import visualize_pruning, save_overlay_image


def contrast_stretch(grid, mode="rank", low_p=2.0, high_p=98.0):
    """增强热力图区分度。

    - mode="rank"（默认）: 分位数排名归一化，保证 [0,1] 均匀分布、颜色铺满
      整个色阶、区分度最大（代价是只保留相对排序，丢掉绝对量级感）。
    - mode="percentile": 按百分位裁剪后线性归一化，保留一定的绝对量级感，
      但长尾分布下主体可能被压缩。

    Args:
        grid: numpy 2D 数组（L2 范数网格）。
        mode: "rank" 或 "percentile"。
        low_p / high_p: percentile 模式的裁剪百分位。

    Returns:
        归一化到 [0, 1] 的数组。
    """
    if mode == "rank":
        flat = grid.flatten()
        if flat.size <= 1:
            return np.zeros_like(grid, dtype=float)
        # argsort 两次得到 rank，归一化到 [0,1]
        ranks = np.argsort(np.argsort(flat)).astype(float)
        return (ranks / (flat.size - 1)).reshape(grid.shape)
    # percentile 模式
    lo = np.percentile(grid, low_p)
    hi = np.percentile(grid, high_p)
    if hi - lo < 1e-8:
        return np.zeros_like(grid, dtype=float)
    return np.clip((grid - lo) / (hi - lo), 0.0, 1.0)

MODEL_PATH = "/home/u2024311149/models/Qwen3.5-4B"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="单张图片路径")
    parser.add_argument("--r", type=float, default=0.5, help="剪枝比例（0.5=保留50%）")
    parser.add_argument("--question", default="Describe this image briefly.")
    parser.add_argument("--out_dir", default="prune/output/viz_l2/single")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

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

    # ── 处理单张图 ──
    img_path = args.image
    image = Image.open(img_path).convert("RGB")

    msg = [{"role": "user", "content": [
        {"type": "image", "image": img_path},
        {"type": "text", "text": args.question},
    ]}]
    prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
    img_in, _ = process_vision_info(msg)
    inputs = processor(text=[prompt], images=img_in, padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    grid_thw = inputs["image_grid_thw"]
    hb = int(grid_thw[0, 1].item() // 2)
    wb = int(grid_thw[0, 2].item() // 2)
    n_merged = hb * wb

    # ── 跑 vision encoder 拿 raw features ──
    backbone.visual._scope_features = None
    with torch.no_grad():
        backbone.visual(inputs["pixel_values"], grid_thw=grid_thw)
    sf = backbone.visual._scope_features
    if sf is None:
        print("[!] 未捕获 raw features")
        return

    raw = sf[: n_merged * 4]
    bf = raw.view(n_merged, 4, -1).mean(dim=1)  # [n_merged, 1024] block features
    l2_scores = torch.linalg.vector_norm(bf.float(), ord=2, dim=-1)  # [n_merged]

    keep_count = max(1, int(n_merged * (1.0 - args.r)))
    keep_mask = pruner.keep_mask(bf, keep_count)
    grid_mask = keep_mask.view(hb, wb)
    l2_grid = l2_scores.view(hb, wb).cpu().numpy()

    print(f"[*] R={args.r}, grid={hb}x{wb}, keep={keep_count}/{n_merged}")

    # ── 可视化 ──
    out_path = os.path.join(args.out_dir, f"l2_r{int(args.r*100):02d}.png")
    visualize_pruning(
        image, grid_mask,
        title=f"L2 pruning R={args.r} (keep {keep_count}/{n_merged})",
        save_path=out_path,
    )
    print(f"[+] saved -> {out_path}")

    # 纯图像本体版本（无标题/无文字面板，只有叠加了半透明剪枝掩码的原图）
    pure_path = os.path.join(args.out_dir, f"l2_pure_r{int(args.r*100):02d}.png")
    save_overlay_image(image, grid_mask, pure_path, alpha=0.45)
    print(f"[+] pure image saved -> {pure_path}")

    # 额外：更深蓝色版本（对比用，加深剪枝掩码颜色）
    deep_blue = (0.02, 0.03, 0.2)
    deep_path = os.path.join(args.out_dir, f"l2_pure_deep_r{int(args.r*100):02d}.png")
    save_overlay_image(image, grid_mask, deep_path, alpha=0.45, pruned_color=deep_blue)
    print(f"[+] deep-blue pure image saved -> {deep_path}")

    # 额外保存：只有掩码的纯热力图（不叠加原图，便于观察剪枝空间分布）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid_mask.cpu().numpy(), cmap="Greys", origin="upper", interpolation="nearest")
    ax.set_title(f"Keep mask ({keep_count}/{n_merged})")
    ax.axis("off")
    mask_path = os.path.join(args.out_dir, f"mask_r{int(args.r*100):02d}.png")
    fig.savefig(mask_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] mask saved -> {mask_path}")

    # ── L2 分数热力图（连续色阶，展示每个 block 的 L2 范数）──
    # 对比度拉伸：按 2%~98% 百分位裁剪，拉开中间分布（否则颜色单调）
    l2_norm = contrast_stretch(l2_grid, low_p=2.0, high_p=98.0)

    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(13, 6))

    im_a = ax2a.imshow(l2_norm, cmap="hot", origin="upper", interpolation="nearest")
    ax2a.set_title("L2 norm heatmap (contrast-stretched)", fontsize=10)
    ax2a.axis("off")
    fig2.colorbar(im_a, ax=ax2a, fraction=0.046, pad=0.04, label="L2 norm (norm)")

    im_b = ax2b.imshow(l2_norm, cmap="inferno", origin="upper", interpolation="bicubic")
    ax2b.set_title("L2 norm (smooth)", fontsize=10)
    ax2b.axis("off")
    fig2.colorbar(im_b, ax=ax2b, fraction=0.046, pad=0.04, label="L2 norm (norm)")

    fig2.suptitle(f"L2 saliency heatmap (grid {hb}x{wb})", fontsize=12)
    fig2.tight_layout()
    l2_heat_path = os.path.join(args.out_dir, f"l2_heatmap_r{int(args.r*100):02d}.png")
    fig2.savefig(l2_heat_path, dpi=130, bbox_inches="tight")
    plt.close(fig2)
    print(f"[+] L2 heatmap saved -> {l2_heat_path}")

    # ── 纯 L2 热力图（无任何装饰：无 title/colorbar/坐标轴/白边）──
    # 用 turbo colormap（色阶丰富），高分辨率输出
    fig_pure, ax_pure = plt.subplots(figsize=(wb * 8 / 100, hb * 8 / 100))
    ax_pure.imshow(l2_norm, cmap="turbo", origin="upper", interpolation="nearest")
    ax_pure.axis("off")
    # 去掉所有空白，只保留热力图本体
    fig_pure.subplots_adjust(left=0, right=1, top=1, bottom=0)
    pure_heat_path = os.path.join(args.out_dir, f"l2_pure_heatmap_r{int(args.r*100):02d}.png")
    fig_pure.savefig(pure_heat_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig_pure)
    print(f"[+] pure L2 heatmap saved -> {pure_heat_path}")

    # ── 原图 + L2 热力图叠加（PIL 直接合成，无 colorbar/标题/坐标轴）──
    # 生成多张独立纯图（jet / inferno / turbo），每张只有「原图 + 半透明热力图」
    import matplotlib as mpl

    def _get_cmap(name):
        # 兼容新旧 matplotlib：plt.get_cmap 在老版本也可用
        import matplotlib.pyplot as _plt
        return _plt.get_cmap(name)

    base_pil = image.convert("RGB").resize(image.size)

    overlay_cfgs = [
        ("jet", 0.55),
        ("inferno", 0.6),
        ("turbo", 0.45),
    ]
    for cmap_name, alpha_ov in overlay_cfgs:
        heat_rgba = _get_cmap(cmap_name)(l2_norm)  # [hb, wb, 4]，0-1
        heat_rgb = (heat_rgba[:, :, :3] * 255).astype(np.uint8)
        heat_pil = Image.fromarray(heat_rgb).resize(image.size, Image.BILINEAR)
        overlay = Image.blend(base_pil, heat_pil, alpha=alpha_ov)
        path = os.path.join(
            args.out_dir,
            f"l2_overlay_{cmap_name}_r{int(args.r*100):02d}.png",
        )
        overlay.save(path)
        print(f"[+] L2 overlay ({cmap_name}, alpha={alpha_ov}) saved -> {path}")


if __name__ == "__main__":
    main()
