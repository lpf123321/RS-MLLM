"""Smoke test for the modular pruner + visualization.

Does NOT require a model or GPU — uses synthetic features and a synthetic
image to verify that:

  1. ``L2NormPruner.select`` returns the correct indices (sorted, keeps the
     tokens with the largest L2 norms).
  2. ``prune.viz`` maps a keep-mask back onto the image and saves a PNG.

Usage:
    python scripts/smoke_test_viz.py
"""

import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prune.base import VisualTokenPruner
from prune.l2norm import L2NormPruner
from prune.viz import prune_to_grid_mask, overlay_prune_mask, visualize_pruning

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prune", "output", "viz")
os.makedirs(OUT_DIR, exist_ok=True)


def _make_synthetic_image(grid_h, grid_w, block_px=16):
    """Build a synthetic image with a bright salient region top-left."""
    h, w = grid_h * block_px, grid_w * block_px
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # background gradient
    yy, xx = np.mgrid[0:h, 0:w]
    img[..., 0] = (xx / w * 120).astype(np.uint8)
    img[..., 1] = (yy / h * 120).astype(np.uint8)
    img[..., 2] = 60
    # salient bright block top-left (high L2 norm region)
    img[: block_px * grid_h // 4, : block_px * grid_w // 4] = (255, 255, 255)
    return Image.fromarray(img)


def _make_spatial_features(grid_h, grid_w, dim, seed=0):
    """Synthetic features where top-left block tokens have large L2 norms."""
    rng = np.random.RandomState(seed)
    features = rng.randn(grid_h * grid_w, dim).astype(np.float32)
    # top-left 1/4 x 1/4 blocks are salient (large magnitude)
    for hi in range(grid_h // 4):
        for wi in range(grid_w // 4):
            features[hi * grid_w + wi] *= 8.0
    return torch.from_numpy(features)


def test_l2norm_pruner():
    print("[1/3] Testing L2NormPruner.select ...")
    torch.manual_seed(0)
    features = torch.randn(64, 256)
    pruner = L2NormPruner()

    # keep_count >= N -> return all
    all_idx = pruner.select(features, 100, seed=None)
    assert all_idx.numel() == 64, f"expected 64, got {all_idx.numel()}"
    assert torch.equal(all_idx, torch.arange(64)), "should return all indices"

    # keep top-16 by L2 norm
    keep = pruner.select(features, 16, seed=None)
    assert keep.numel() == 16
    assert torch.equal(keep, torch.sort(keep).values), "indices should be sorted"

    norms = torch.linalg.vector_norm(features.float(), ord=2, dim=-1)
    topk = torch.topk(norms, k=16).indices
    assert set(keep.tolist()) == set(topk.tolist()), "should keep the largest L2 norms"

    # keep_mask helper
    mask = pruner.keep_mask(features, 16)
    assert mask.sum().item() == 16
    assert mask.dtype == torch.bool
    print("  L2NormPruner.select: OK")


def test_prune_to_grid_mask():
    print("[2/3] Testing prune_to_grid_mask ...")
    grid_h, grid_w, dim = 8, 8, 128
    features = _make_spatial_features(grid_h, grid_w, dim)
    pruner = L2NormPruner()

    mask = prune_to_grid_mask(pruner, features, grid_h, grid_w, keep_ratio=0.5)
    assert mask.shape == (grid_h, grid_w)
    assert mask.dtype == torch.bool
    assert mask.sum().item() == 32, f"expected 32 kept, got {mask.sum().item()}"

    # top-left salient region should be mostly kept
    tl_keep = mask[: grid_h // 4, : grid_w // 4].float().mean().item()
    assert tl_keep > 0.9, f"top-left salient region keep rate too low: {tl_keep}"
    print(f"  prune_to_grid_mask: OK (top-left keep rate = {tl_keep:.2f})")


def test_visualization():
    print("[3/3] Testing visualization ...")
    grid_h, grid_w, dim = 8, 8, 128
    image = _make_synthetic_image(grid_h, grid_w)
    features = _make_spatial_features(grid_h, grid_w, dim)
    pruner = L2NormPruner()

    mask = prune_to_grid_mask(pruner, features, grid_h, grid_w, keep_ratio=0.5)

    overlay = overlay_prune_mask(image, mask)
    assert overlay.shape[:2] == (grid_h * 16, grid_w * 16), f"overlay shape {overlay.shape}"
    assert overlay.max() <= 1.0 and overlay.min() >= 0.0

    save_path = os.path.join(OUT_DIR, "smoke_test_l2norm_r50.png")
    visualize_pruning(image, mask, title="L2NormPruner keep 50%", save_path=save_path)
    assert os.path.exists(save_path), f"PNG not written: {save_path}"
    print(f"  visualization: OK -> {save_path}")


def main():
    test_l2norm_pruner()
    test_prune_to_grid_mask()
    test_visualization()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
