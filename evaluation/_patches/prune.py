"""
Token 剪枝集成模块 —— 通过参数切换不同剪枝方法。

方法：
  - "l2"       : FastV-L2Norm，输入侧 L2 范数 top-k 选择后直接删除
  - "k2"       : FastV-K，第 1 层后 L2-norm 打分，第 2 层起 mask
  - "divprune" : DivPrune，余弦多样性贪心选择后直接删除

用法：
  from evaluation._patches.prune import apply_pruning, enable_pruning

  apply_pruning()                         # 必须先调用，monkey-patch Qwen3_5Model.forward
  model = Qwen3_5ForConditionalGeneration.from_pretrained(...)
  enable_pruning(model, method="l2", r=0.5)   # 启用剪枝
"""
import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as qwen35
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextModel,
    Qwen3_5ModelOutputWithPast,
    create_causal_mask,
)
from transformers.cache_utils import DynamicCache

# ============================================================
# 剪枝方法注册表
# ============================================================

PRUNE_METHODS = ["l2", "k2", "divprune"]


def enable_pruning(model, method: str = "l2", r: float = 0.5, k: int = 2):
    """对已加载的模型启用指定剪枝方法。

    Args:
        model: Qwen3_5ForConditionalGeneration 实例。
        method: "l2" | "k2" | "divprune"
        r: 图像 token 剪枝比例（0.5 = 保留 50%）
        k: FastV-K 的层数（仅 method="k2" 时生效）
    """
    if method not in PRUNE_METHODS:
        raise ValueError(f"Unknown prune method: {method}. Choose from {PRUNE_METHODS}")

    backbone = model.model  # Qwen3_5Model
    if not hasattr(backbone, "language_model"):
        backbone = getattr(model, "model", model)

    if method == "k2":
        # K=2 需要 monkey-patch TextModel.forward
        text_model = backbone.language_model
        text_model._fastv_enabled = True
        text_model._fastv_k = k
        text_model._fastv_r = r
        _K2_ORIGINAL_FORWARD = Qwen3_5TextModel.forward
        text_model.forward = _qwen35_text_k2_forward.__get__(text_model, Qwen3_5TextModel)
        if not hasattr(text_model, "_k2_original_forward"):
            text_model._k2_original_forward = _K2_ORIGINAL_FORWARD
    else:
        # l2 / divprune: pre-LLM 剪枝，设置 Qwen3_5Model 的标志位
        backbone._prune_method = method
        backbone._prune_r = r
    print(f"Pruning enabled: method={method}, R={r} (keep {(1-r)*100:.0f}%)")


def disable_pruning(model):
    backbone = model.model
    if not hasattr(backbone, "language_model"):
        backbone = getattr(model, "model", model)
    backbone._prune_method = None
    backbone._prune_r = 0
    # 如果 K2 的 forward 打过补丁，恢复
    if hasattr(backbone, "language_model"):
        text_model = backbone.language_model
        if hasattr(text_model, "_k2_original_forward"):
            text_model.forward = text_model._k2_original_forward.__get__(
                text_model, Qwen3_5TextModel
            )


# ============================================================
# Monkey-patch Qwen3_5Model.forward（传入 visual_pos_masks + 剪枝）
# ============================================================

_ORIGINAL_QWEN35_FORWARD = None


def apply_pruning():
    """必须在加载模型前调用！monkey-patch Qwen3_5Model.forward。"""
    global _ORIGINAL_QWEN35_FORWARD
    if _ORIGINAL_QWEN35_FORWARD is not None:
        return  # 已经 patch 过
    _ORIGINAL_QWEN35_FORWARD = qwen35.Qwen3_5Model.forward
    qwen35.Qwen3_5Model.forward = _qwen35_patched_forward


# ---- 辅助函数 ----

def _flatten_vision_features(vision_outputs):
    pooled = getattr(vision_outputs, "pooler_output", vision_outputs)
    if isinstance(pooled, torch.Tensor):
        return pooled
    if isinstance(pooled, (tuple, list)):
        return torch.cat(list(pooled), dim=0)
    raise TypeError(f"Unsupported vision output type: {type(vision_outputs)!r}")


def _make_dummy_qwen3_visual_inputs(visual):
    dummy_grid = torch.tensor([[1, 32, 32]], device=visual.device)
    patch_embed = visual.patch_embed
    patch_dim = (
        patch_embed.in_channels
        * patch_embed.temporal_patch_size
        * patch_embed.patch_size
        * patch_embed.patch_size
    )
    num_patches = int(dummy_grid.prod().item())
    dummy_pixel = torch.zeros((num_patches, patch_dim), device=visual.device, dtype=visual.dtype)
    return dummy_pixel, dummy_grid


def _expand_video_grid_to_frames(video_grid_thw):
    if video_grid_thw is None:
        return None
    frame_grids = []
    for grid in video_grid_thw:
        num_frames = int(grid[0].item())
        per_frame_grid = grid.unsqueeze(0).expand(num_frames, -1).clone()
        per_frame_grid[:, 0] = 1
        frame_grids.append(per_frame_grid)
    if not frame_grids:
        return video_grid_thw
    return torch.cat(frame_grids, dim=0)


# ---- Patched Qwen3_5Model.forward ----

def _qwen35_patched_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    mm_token_type_ids=None,
    **kwargs,
):
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    image_mask = None
    video_mask = None

    if pixel_values is None and pixel_values_videos is None:
        dummy_pixel, dummy_grid = _make_dummy_qwen3_visual_inputs(self.visual)
        image_outputs = self.get_image_features(dummy_pixel, dummy_grid, return_dict=True)
        image_embeds = _flatten_vision_features(image_outputs).to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds += image_embeds.mean() * 0

    if pixel_values is not None:
        image_outputs = self.get_image_features(pixel_values, image_grid_thw, return_dict=True)
        image_embeds = _flatten_vision_features(image_outputs).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_outputs = self.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True)
        video_embeds = _flatten_vision_features(video_outputs).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    if image_mask is not None or video_mask is not None:
        img_bool = image_mask[..., 0].to(dtype=torch.bool) if image_mask is not None else torch.zeros(
            inputs_embeds.shape[0], inputs_embeds.shape[1], dtype=torch.bool, device=inputs_embeds.device
        )
        vid_bool = video_mask[..., 0].to(dtype=torch.bool) if video_mask is not None else torch.zeros(
            inputs_embeds.shape[0], inputs_embeds.shape[1], dtype=torch.bool, device=inputs_embeds.device
        )
        visual_pos_masks = img_bool | vid_bool
    else:
        visual_pos_masks = None

    if position_ids is None:
        rope_video_grid_thw = _expand_video_grid_to_frames(video_grid_thw)
        position_ids = self.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=rope_video_grid_thw,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
        )

    # ---- Pruning (pre-LLM) ----
    prune_method = getattr(self, "_prune_method", None)
    if prune_method in ("l2", "divprune"):
        r = self._prune_r
        if prune_method == "l2":
            inputs_embeds, attention_mask, position_ids = _prune_l2norm(
                inputs_embeds, attention_mask, position_ids, visual_pos_masks, r
            )
        else:
            inputs_embeds, attention_mask, position_ids = _prune_divprune(
                inputs_embeds, attention_mask, position_ids, visual_pos_masks, r
            )
        visual_pos_masks = None

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        visual_pos_masks=visual_pos_masks,
        **kwargs,
    )

    return Qwen3_5ModelOutputWithPast(
        **outputs,
        rope_deltas=self.rope_deltas,
    )


# ============================================================
# 剪枝函数
# ============================================================

def _prune_l2norm(inputs_embeds, attention_mask, position_ids, visual_pos_masks, r):
    """L2 范数剪枝：保留 L2-norm 最高的 top-(1-r)% 图像 token。"""
    if visual_pos_masks is None:
        return inputs_embeds, attention_mask, position_ids

    batch_size, seq_len, hidden_dim = inputs_embeds.shape
    device = inputs_embeds.device
    keep_masks = []

    for b in range(batch_size):
        img_mask = visual_pos_masks[b]
        num_img = img_mask.sum().item()
        if num_img == 0:
            keep_masks.append(torch.ones(seq_len, dtype=torch.bool, device=device))
            continue
        num_keep = max(1, int(num_img * (1.0 - r)))
        img_hidden = inputs_embeds[b, img_mask, :]
        l2_norm = torch.norm(img_hidden.float(), dim=-1)
        _, top_local = torch.topk(l2_norm, k=min(num_keep, num_img))
        img_positions = torch.where(img_mask)[0]
        keep_global = img_positions[top_local]
        keep_mask = ~img_mask.clone()
        keep_mask[keep_global] = True
        keep_masks.append(keep_mask)

    return _apply_keep_masks(inputs_embeds, attention_mask, position_ids, keep_masks)


def _prune_divprune(inputs_embeds, attention_mask, position_ids, visual_pos_masks, r):
    """DivPrune 多样性剪枝：贪心 max-min 选择。"""
    if visual_pos_masks is None:
        return inputs_embeds, attention_mask, position_ids

    batch_size, seq_len, hidden_dim = inputs_embeds.shape
    device = inputs_embeds.device
    keep_masks = []

    for b in range(batch_size):
        img_mask = visual_pos_masks[b]
        num_img = img_mask.sum().item()
        if num_img == 0:
            keep_masks.append(torch.ones(seq_len, dtype=torch.bool, device=device))
            continue
        num_keep = max(1, int(num_img * (1.0 - r)))
        img_hidden = inputs_embeds[b, img_mask, :]
        top_local = _divprune_greedy_select(img_hidden, num_keep)
        img_positions = torch.where(img_mask)[0]
        keep_global = img_positions[top_local]
        keep_mask = ~img_mask.clone()
        keep_mask[keep_global] = True
        keep_masks.append(keep_mask)

    return _apply_keep_masks(inputs_embeds, attention_mask, position_ids, keep_masks)


def _divprune_greedy_select(features, num_keep):
    """贪心最大-最小多样性选择。"""
    N = features.shape[0]
    if num_keep >= N:
        return torch.arange(N, device=features.device)
    normed = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
    cos_dist = 1.0 - torch.mm(normed, normed.t())
    cos_dist.fill_diagonal_(float("inf"))
    selected = torch.empty(num_keep, dtype=torch.long, device=features.device)
    for i in range(num_keep):
        if i == 0:
            best_idx = torch.argmax(cos_dist.min(dim=1).values)
        else:
            min_dist = cos_dist[selected[:i], :].min(dim=0).values
            min_dist[selected[:i]] = float("-inf")
            best_idx = torch.argmax(min_dist)
        selected[i] = best_idx
    return selected


def _apply_keep_masks(inputs_embeds, attention_mask, position_ids, keep_masks):
    """根据 keep_masks 从序列中删除 token，并 padding 对齐 batch。"""
    batch_size, _, hidden_dim = inputs_embeds.shape
    device = inputs_embeds.device
    max_len = max(m.sum().item() for m in keep_masks)

    new_embeds, new_masks, new_pos = [], [], []
    for b in range(batch_size):
        km = keep_masks[b]
        cur_len = km.sum().item()
        pad_len = max_len - cur_len
        new_embeds.append(torch.cat([
            inputs_embeds[b, km, :],
            torch.zeros(pad_len, hidden_dim, dtype=inputs_embeds.dtype, device=device),
        ]))
        if attention_mask is not None:
            new_masks.append(torch.cat([
                attention_mask[b, km],
                torch.zeros(pad_len, dtype=attention_mask.dtype, device=device),
            ]))
        new_pos.append(torch.cat([
            position_ids[:, b, km],
            torch.zeros(4, pad_len, dtype=position_ids.dtype, device=device),
        ], dim=-1))

    inputs_embeds = torch.stack(new_embeds, dim=0)
    attention_mask = torch.stack(new_masks, dim=0) if attention_mask is not None else None
    position_ids = torch.stack(new_pos, dim=1)
    return inputs_embeds, attention_mask, position_ids


# ============================================================
# FastV-K（LLM 内部 mask 模式）
# ============================================================

def _qwen35_text_k2_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    **kwargs,
):
    """FastV-K 版本的 Qwen3_5TextModel.forward。"""
    K = getattr(self, "_fastv_k", 2)
    R = getattr(self, "_fastv_r", 0.5)
    visual_pos_masks = kwargs.get("visual_pos_masks", None)

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    if position_ids is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)
    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = None

    seq_length = inputs_embeds.shape[1]
    # Decode step: fallback
    if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
        if past_key_values.get_seq_length() > 0:
            return _K2_ORIGINAL_FORWARD(self, input_ids=input_ids, attention_mask=attention_mask,
                                         position_ids=position_ids, past_key_values=past_key_values,
                                         inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)

    causal_mask = create_causal_mask(config=self.config, inputs_embeds=inputs_embeds,
                                     attention_mask=attention_mask, past_key_values=past_key_values,
                                     position_ids=text_position_ids)
    linear_attn_mask = self._update_linear_attn_mask(attention_mask, past_key_values)
    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    score_layer = K - 1
    if score_layer < 0 or visual_pos_masks is None:
        for i, dec in enumerate(self.layers[:self.config.num_hidden_layers]):
            lm = linear_attn_mask if self.config.layer_types[i] == "linear_attention" else causal_mask
            hidden_states = dec(hidden_states, position_embeddings=position_embeddings, attention_mask=lm,
                                position_ids=text_position_ids, past_key_values=past_key_values,
                                use_cache=use_cache, **kwargs)
        hidden_states = self.norm(hidden_states)
        return Qwen3_5ModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)

    img_mask = visual_pos_masks
    pruned_causal_mask = None
    pruned_linear_mask = None
    batch_size = inputs_embeds.shape[0]

    for i, dec in enumerate(self.layers[:self.config.num_hidden_layers]):
        if i < score_layer:
            lm = linear_attn_mask if self.config.layer_types[i] == "linear_attention" else causal_mask
            hidden_states = dec(hidden_states, position_embeddings=position_embeddings, attention_mask=lm,
                                position_ids=text_position_ids, past_key_values=past_key_values,
                                use_cache=use_cache, **kwargs)
        elif i == score_layer:
            lm = linear_attn_mask if self.config.layer_types[i] == "linear_attention" else causal_mask
            hidden_states = dec(hidden_states, position_embeddings=position_embeddings, attention_mask=lm,
                                position_ids=text_position_ids, past_key_values=past_key_values,
                                use_cache=use_cache, **kwargs)
            # L2-norm pruning decision
            pruned_attn_mask_2d = torch.ones((batch_size, seq_length), dtype=torch.bool, device=inputs_embeds.device)
            for b in range(batch_size):
                bim = img_mask[b]
                num_img = bim.sum().item()
                if num_img == 0:
                    continue
                num_keep = max(1, int(num_img * (1.0 - R)))
                ih = hidden_states[b, bim, :]
                l2 = torch.norm(ih.float(), dim=-1)
                _, tl = torch.topk(l2, k=min(num_keep, num_img))
                ip = torch.where(bim)[0]
                tg = ip[tl]
                pruned_attn_mask_2d[b, bim] = False
                pruned_attn_mask_2d[b, tg] = True
            pruned_causal_mask = create_causal_mask(config=self.config, inputs_embeds=inputs_embeds,
                                                     attention_mask=pruned_attn_mask_2d,
                                                     past_key_values=past_key_values,
                                                     position_ids=text_position_ids)
            pruned_linear_mask = self._update_linear_attn_mask(pruned_attn_mask_2d, past_key_values)
        elif i >= K:
            lm = pruned_linear_mask if self.config.layer_types[i] == "linear_attention" else pruned_causal_mask
            hidden_states = dec(hidden_states, position_embeddings=position_embeddings, attention_mask=lm,
                                position_ids=text_position_ids, past_key_values=past_key_values,
                                use_cache=use_cache, **kwargs)
        else:
            lm = linear_attn_mask if self.config.layer_types[i] == "linear_attention" else causal_mask
            hidden_states = dec(hidden_states, position_embeddings=position_embeddings, attention_mask=lm,
                                position_ids=text_position_ids, past_key_values=past_key_values,
                                use_cache=use_cache, **kwargs)

    hidden_states = self.norm(hidden_states)
    return Qwen3_5ModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


_K2_ORIGINAL_FORWARD = Qwen3_5TextModel.forward
