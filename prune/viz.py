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


def overlay_prune_mask(image, keep_mask, block_px=16, alpha=0.45, pruned_color=(0.08, 0.12, 0.45)):
    """Overlay the keep-mask on ``image``; pruned blocks are marked with a
    semi-transparent red tint so the underlying image stays visible.

    Args:
        image: ``PIL.Image.Image`` or numpy array (H, W, 3).
        keep_mask: bool array of shape ``[grid_h, grid_w]`` (or torch tensor).
        block_px: pixel size of each merged-token block in the overlay.
        alpha: opacity of the pruned-region tint in ``[0, 1]`` (0 = fully
            transparent, 1 = fully opaque).
        pruned_color: RGB tuple in ``[0, 1]`` for the pruned-region tint.

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
    color = np.asarray(pruned_color, dtype=float)
    pruned = ~keep
    for hi in range(grid_h):
        for wi in range(grid_w):
            if pruned[hi, wi]:
                r0, r1 = hi * block_px, (hi + 1) * block_px
                c0, c1 = wi * block_px, (wi + 1) * block_px
                block = overlay[r0:r1, c0:c1]
                overlay[r0:r1, c0:c1] = (1.0 - alpha) * block + alpha * color
    return overlay


def save_overlay_image(
    image,
    keep_mask,
    save_path,
    block_px=16,
    alpha=0.45,
    pruned_color=(0.08, 0.12, 0.45),
):
    """Save ONLY the pruned-token overlay as a plain image (no axes, no text).

    Unlike :func:`visualize_pruning`, this writes a bare PNG containing just the
    image with the semi-transparent prune mask — no matplotlib decorations.

    Args:
        image: ``PIL.Image.Image`` or numpy array.
        keep_mask: bool array ``[grid_h, grid_w]``.
        save_path: output PNG path.
        block_px: overlay block pixel size.
        alpha: opacity of the pruned-region tint in ``[0, 1]``.
        pruned_color: RGB tuple in ``[0, 1]`` for the pruned-region tint.
    """
    overlay = overlay_prune_mask(
        image, keep_mask, block_px=block_px, alpha=alpha, pruned_color=pruned_color
    )
    out_img = Image.fromarray((np.clip(overlay, 0, 1) * 255).astype(np.uint8))
    out_img.save(save_path)
    return out_img


def visualize_pruning(
    image,
    keep_mask,
    title=None,
    save_path=None,
    block_px=16,
    alpha=0.45,
    pruned_color=(0.08, 0.12, 0.45),
    figsize=(12, 5.5),
):
    """Render a two-panel figure: pruned-token overlay + summary text.

    Args:
        image: ``PIL.Image.Image`` or numpy array.
        keep_mask: bool array ``[grid_h, grid_w]``.
        title: optional suptitle.
        save_path: if given, save the figure to this path (PNG).
        block_px: overlay block pixel size.
        alpha: opacity of the pruned-region tint in ``[0, 1]``.
        pruned_color: RGB tuple in ``[0, 1]`` for the pruned-region tint.
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

    overlay = overlay_prune_mask(image, keep, block_px=block_px, alpha=alpha, pruned_color=pruned_color)
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
