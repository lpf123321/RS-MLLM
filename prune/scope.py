"""
SCOPE-L2: Saliency-Coverage Oriented Token Pruning for Qwen3.5-4B.

Uses L2 norm of raw visual features (penultimate layer) as saliency,
replacing CLS attention scores. Operates at 2x2 spatial merge block level
to remain compatible with Qwen3VL's PatchMerger.

Algorithm:
  1. Register forward hook on vision block 22 (penultimate) to capture raw features
  2. Group raw tokens into 2x2 blocks (4 tokens = 1 block, maps 1:1 to merged tokens)
  3. L2 norm of block features as saliency score
  4. Cosine similarity matrix + greedy marginal coverage gain selection
  5. Map selected blocks to merged tokens, prune non-selected from inputs_embeds

Reference: SCOPE (NeurIPS 2025) - https://arxiv.org/abs/2510.24214
"""

import os
import torch


def SCOPE_L2(features, num_select, alpha=None, combined=None):
    """
    Batched block-level SCOPE with L2 norm as saliency.

    Args:
        features: [B, N, D] block-level feature vectors
        num_select: number of blocks to select per batch element
        alpha: saliency exponent (default from env ALPHA=1.0)
        combined: 'multi' or 'add' (default from env COMBINED='multi')

    Returns:
        selected_idx: [B, num_select] selected block indices
        cosine_simi: [B, N, N] cosine similarity matrix
    """
    if alpha is None:
        alpha = float(os.environ.get('ALPHA', '1.0'))
    if combined is None:
        combined = os.environ.get('COMBINED', 'multi')

    B, N, D = features.shape
    device = features.device
    dtype = features.dtype

    saliency = features.norm(dim=-1)

    norm_vectors = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
    cosine_simi = torch.bmm(norm_vectors, norm_vectors.transpose(1, 2))

    selected = torch.zeros(B, N, dtype=torch.bool, device=device)
    selected_idx = torch.empty(B, num_select, dtype=torch.long, device=device)
    cur_max = torch.zeros(B, N, dtype=dtype, device=device)

    saliency_powered = saliency ** alpha

    for i in range(num_select):
        unselected_mask = ~selected
        gains = torch.maximum(
            torch.zeros(1, dtype=dtype, device=device),
            cosine_simi.masked_fill(~unselected_mask.unsqueeze(1), 0) -
            cur_max.unsqueeze(2)
        ).sum(dim=1)

        if combined == 'multi':
            gains = gains * saliency_powered
        elif combined == 'add':
            gains = gains + saliency_powered

        gains = gains.masked_fill(~unselected_mask, float('-inf'))
        best_idx = gains.argmax(dim=1)

        selected[torch.arange(B, device=device), best_idx] = True
        selected_idx[:, i] = best_idx
        cur_max = torch.maximum(cur_max, cosine_simi[torch.arange(B, device=device), best_idx])

    return selected_idx, cosine_simi


def enable_scope_hooks(visual):
    """
    Register forward hook on the vision model's penultimate block.
    Must be called AFTER model is loaded.

    Args:
        visual: Qwen3VLVisionModel instance (model.model.visual)
    """
    scope_layer = visual.config.depth - 2  # 22 for depth=24

    def capture_hook(module, input, output):
        visual._scope_features = output.detach()

    visual.blocks[scope_layer].register_forward_hook(capture_hook)

    if not hasattr(visual, '_scope_features'):
        visual._scope_features = None


def _prune_scope(inputs_embeds, attention_mask, position_ids, visual_pos_masks, r,
                  scope_features, image_grid_thw):
    """
    SCOPE pruning: select merged tokens using raw features from vision layer 22.

    Steps:
      1. Group raw features into 2x2 blocks (mean pool), one block per merged token
      2. Run SCOPE_L2 per image to select keep-(1-r) blocks
      3. Map selected blocks back to merged token positions in inputs_embeds
      4. Remove non-selected visual positions via _apply_keep_masks

    Args:
        inputs_embeds: [B, L, D]  --- after scattering image embeddings
        attention_mask: [B, L] or None
        position_ids: [4, B, L]  (Qwen3.5 uses 3D MRoPE with 4 position dims)
        visual_pos_masks: [B, L]  bool, True at visual token positions
        r: pruning ratio (0.5 = keep 50%)
        scope_features: [total_raw, 1024]  raw features captured at layer 22
        image_grid_thw: [num_images, 3]  grid dimensions per image

    Returns:
        (inputs_embeds, attention_mask, position_ids) with pruned visual tokens
    """
    if visual_pos_masks is None or scope_features is None:
        return inputs_embeds, attention_mask, position_ids

    B, L, hidden_dim = inputs_embeds.shape
    device = inputs_embeds.device

    # ---- per-image token counts ----
    merged_per_image = [(g[0] * g[1] * g[2]).item() // 4 for g in image_grid_thw]
    raw_per_image = [m * 4 for m in merged_per_image]

    # If images in this batch have different dimensions, skip pruning to avoid
    # garbage outputs caused by mixed-size position embedding interactions.
    if len(set(merged_per_image)) > 1:
        return inputs_embeds, attention_mask, position_ids

    total_raw = scope_features.shape[0]
    if total_raw < sum(raw_per_image):
        return inputs_embeds, attention_mask, position_ids

    # ---- map images → samples via cumulative merged token counts ----
    per_sample_counts = visual_pos_masks.sum(dim=1).tolist()
    image_to_sample = []
    sidx = 0
    sacc = 0
    for ni in merged_per_image:
        image_to_sample.append(min(sidx, B - 1))
        sacc += ni
        if sidx < B and sacc >= per_sample_counts[sidx]:
            sidx = min(sidx + 1, B - 1)
            sacc = 0

    # ---- SCOPE per image → select merged tokens ----
    raw_off = 0
    sel_per_image = []

    for ni in merged_per_image:
        nr = ni * 4
        if raw_off + nr > total_raw:
            sel_per_image.append(torch.ones(ni, dtype=torch.bool, device=device))
            raw_off += nr
            continue

        img_raw = scope_features[raw_off:raw_off + nr]
        img_raw = img_raw[:ni * 4]
        block_feat = img_raw.view(ni, 4, -1).mean(dim=1)   # [ni, 1024]

        num_keep = max(1, int(ni * (1.0 - r)))
        num_keep = min(num_keep, ni)

        if num_keep < ni:
            sel_blk, _ = SCOPE_L2(block_feat.unsqueeze(0), num_keep)
            m = torch.zeros(ni, dtype=torch.bool, device=device)
            m[sel_blk[0]] = True
            sel_per_image.append(m)
        else:
            sel_per_image.append(torch.ones(ni, dtype=torch.bool, device=device))

        raw_off += nr

    # ---- per-sample keep masks ----
    keep_masks = []
    for b in range(B):
        vm = visual_pos_masks[b]
        nv = vm.sum().item()
        if nv == 0:
            keep_masks.append(torch.ones(L, dtype=torch.bool, device=device))
            continue

        merged_list = [sel_per_image[i] for i in range(len(merged_per_image))
                       if image_to_sample[i] == b]
        if not merged_list:
            keep_masks.append(torch.ones(L, dtype=torch.bool, device=device))
            continue

        combined = torch.cat(merged_list, dim=0)
        if combined.shape[0] != nv:
            keep_masks.append(torch.ones(L, dtype=torch.bool, device=device))
            continue

        keep_pos = torch.where(vm)[0][combined]
        kp = ~vm.clone()
        kp[keep_pos] = True
        keep_masks.append(kp)

    return _apply_keep_masks(inputs_embeds, attention_mask, position_ids, keep_masks)


# ---------------------------------------------------------------------------
#  Utility: delegates to scripts/prune.py:_apply_keep_masks for consistency
# ---------------------------------------------------------------------------
from scripts.prune import _apply_keep_masks
