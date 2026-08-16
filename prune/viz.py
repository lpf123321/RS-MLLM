"""Visualization for visual token pruning.

Given a ``VisualTokenPruner`` and the visual token features of one image, run
the pruner and overlay the resulting keep/prune decision back onto the image.

Qwen3-VL uses a 2x2 spatial merge, so each "merged" token corresponds to a
``block_px x block_px`` region of the (resized) image. The grid dimensions are
``grid_h = image_grid_thw[1] // 2`` and ``grid_w = image_grid_thw[2] // 2``.
"""

import numpy as np
import torch
from PIL import Image


def prune_to_grid_mask(pruner, features, grid_h, grid_w, keep_ratio, seed=None):
    """Run ``pruner`` and return a bool keep-mask of shape ``[grid_h, grid_w]``.

    Args:
        pruner: a ``VisualTokenPruner`` instance.
        features: ``[N, D]`` visual token features (N = grid_h * grid_w).
        grid_h, grid_w: merged-token grid dimensions.
        keep_ratio: fraction of tokens to retain, in ``(0, 1]``.

    Returns:
        ``torch.BoolTensor`` of shape ``[grid_h, grid_w]`` (True = keep).
    """
    n = features.shape[0]
    if n != grid_h * grid_w:
        raise ValueError(
            f"features has {n} tokens but grid is {grid_h}x{grid_w}={grid_h * grid_w}"
        )
    keep_count = max(1, int(n * keep_ratio))
    keep_idx = pruner.select(features, keep_count, seed=seed)
    mask = torch.zeros(n, dtype=torch.bool, device=features.device)
    mask[keep_idx] = True
    return mask.view(grid_h, grid_w)


def overlay_prune_mask(image, keep_mask, block_px=16):
    """Overlay the keep-mask on ``image``; pruned blocks are marked red/dim.

    Args:
        image: ``PIL.Image.Image`` or numpy array (H, W, 3).
        keep_mask: bool array of shape ``[grid_h, grid_w]`` (or torch tensor).
        block_px: pixel size of each merged-token block in the overlay.

    Returns:
        float numpy array in ``[0, 1]`` with shape ``(grid_h*block_px,
        grid_w*block_px, 3)``.
    """
    if isinstance(image, Image.Image):
        image = image
    else:
        image = Image.fromarray(np.asarray(image).astype(np.uint8))

    keep = np.asarray(keep_mask.cpu() if isinstance(keep_mask, torch.Tensor) else keep_mask)
    grid_h, grid_w = keep.shape

    overlay = (
        np.array(image.resize((grid_w * block_px, grid_h * block_px), Image.NEAREST))
        .astype(float)
        / 255.0
    )
    pruned = ~keep
    for hi in range(grid_h):
        for wi in range(grid_w):
            if pruned[hi, wi]:
                r0, r1 = hi * block_px, (hi + 1) * block_px
                c0, c1 = wi * block_px, (wi + 1) * block_px
                overlay[r0:r1, c0:c1, 0] = 0.9  # red channel up
                overlay[r0:r1, c0:c1, 1:] *= 0.25  # dim green/blue
    return overlay


def visualize_pruning(
    image,
    keep_mask,
    title=None,
    save_path=None,
    block_px=16,
    figsize=(12, 5.5),
):
    """Render a two-panel figure: pruned-token overlay + summary text.

    Args:
        image: ``PIL.Image.Image`` or numpy array.
        keep_mask: bool array ``[grid_h, grid_w]``.
        title: optional suptitle.
        save_path: if given, save the figure to this path (PNG).
        block_px: overlay block pixel size.
        figsize: matplotlib figure size.

    Returns:
        ``(fig, axes)`` matplotlib handles.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keep = np.asarray(keep_mask.cpu() if isinstance(keep_mask, torch.Tensor) else keep_mask)
    grid_h, grid_w = keep.shape
    n_keep = int(keep.sum())
    n_total = grid_h * grid_w

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    overlay = overlay_prune_mask(image, keep, block_px=block_px)
    axes[0].imshow(overlay)
    axes[0].set_title(
        f"Pruned tokens: {n_keep}/{n_total} kept ({n_keep / n_total:.0%})", fontsize=10
    )
    axes[0].axis("off")

    axes[1].axis("off")
    summary = (
        f"Grid: {grid_h}x{grid_w} = {n_total} merged tokens\n"
        f"Kept:   {n_keep} ({n_keep / n_total:.0%})\n"
        f"Pruned: {n_total - n_keep} ({(n_total - n_keep) / n_total:.0%})\n\n"
        f"Red blocks = pruned\n"
        f"Visible blocks = kept"
    )
    axes[1].text(
        0.05,
        0.95,
        summary,
        transform=axes[1].transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#f8f9fa", alpha=0.95),
    )

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return fig, axes
