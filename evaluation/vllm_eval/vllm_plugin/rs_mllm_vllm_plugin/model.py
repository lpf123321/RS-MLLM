"""Qwen3.5-VL vLLM model with pre-LLM image-token selection.

This uses vLLM 0.26's multimodal-pruning scheduler plumbing but supplies the
Qwen3.5 image pruning and MRoPE recomputation that upstream currently disables.
The pruning method and keep ratio are process-level settings, which is safe for
RS-MLLM because every evaluator subprocess / resident expert has one policy.
"""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5ProcessingInfo,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.evs import compute_mrope_for_media
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import PromptReplacement, PromptUpdate

METHOD_ENV = "RSM_VLLM_PRUNE_METHOD"
KEEP_RATIO_ENV = "RSM_VLLM_PRUNE_KEEP_RATIO"
SUPPORTED_METHODS = frozenset({"l2norm", "scope_l2"})
logger = init_logger(__name__)


def _settings() -> tuple[str, float]:
    method = os.environ.get(METHOD_ENV, "").strip().lower()
    try:
        keep_ratio = float(os.environ.get(KEEP_RATIO_ENV, ""))
    except ValueError as exc:
        raise RuntimeError(f"Invalid {KEEP_RATIO_ENV}") from exc
    if method not in SUPPORTED_METHODS:
        raise RuntimeError(
            f"Unsupported vLLM pruning method {method!r}; "
            f"expected one of {sorted(SUPPORTED_METHODS)}"
        )
    if not 0.0 < keep_ratio < 1.0:
        raise RuntimeError(f"{KEEP_RATIO_ENV} must be in (0, 1), got {keep_ratio}")
    return method, keep_ratio


def select_token_indices(
    features: torch.Tensor, method: str, keep_ratio: float
) -> torch.LongTensor:
    """Run the same selectors used by the Transformers pruning adapter."""
    token_count = int(features.shape[0])
    keep_count = min(token_count, max(1, int(token_count * keep_ratio)))
    if keep_count == token_count:
        return torch.arange(token_count, device=features.device)
    if method == "l2norm":
        scores = torch.linalg.vector_norm(features.float(), ord=2, dim=-1)
        indices = torch.topk(
            scores, k=keep_count, largest=True, sorted=False
        ).indices
        return torch.sort(indices).values
    if method != "scope_l2":
        raise ValueError(f"Unknown pruning method: {method}")

    feature_float = features.float()
    normalized = torch.nn.functional.normalize(feature_float, dim=-1)
    similarity = normalized @ normalized.T
    saliency = torch.linalg.vector_norm(feature_float, ord=2, dim=-1)
    saliency = saliency / saliency.max().clamp_min(
        torch.finfo(saliency.dtype).eps
    )
    selected: list[torch.Tensor] = []
    available = torch.ones(token_count, dtype=torch.bool, device=features.device)
    coverage = torch.zeros(
        token_count, dtype=similarity.dtype, device=features.device
    )
    for _ in range(keep_count):
        gains = (similarity - coverage[:, None]).clamp_min(0).sum(dim=0)
        scores = gains * saliency
        scores.masked_fill_(~available, -torch.inf)
        chosen = scores.argmax()
        selected.append(chosen)
        available[chosen] = False
        coverage = torch.maximum(coverage, similarity[:, chosen])
    return torch.sort(torch.stack(selected)).values


def pruned_image_mrope_input_positions(
    input_tokens: list[int],
    mm_features: Sequence[Any],
    config: Any,
) -> tuple[torch.Tensor, int]:
    """Build a shape-correct provisional MRoPE tensor for pruned images.

    Upstream Qwen3-VL derives an image span from the unpruned grid.  Once the
    processor shortens an image placeholder, that assumption makes the first
    image overlap the second one.  The actual selected spatial positions are
    not available until the vision encoder runs, so here we use a provisional
    prefix of the grid with the *placeholder* length.  vLLM replaces it with
    the selected positions in ``recompute_mrope_positions`` before the LLM.
    """
    pieces: list[np.ndarray] = []
    sequence_start = 0
    position_start = 0
    merge_size = int(config.vision_config.spatial_merge_size)

    for feature in sorted(mm_features, key=lambda item: item.mm_position.offset):
        if feature.modality != "image":
            raise RuntimeError(
                "RS-MLLM Qwen3.5 Token 剪枝当前只支持图片输入，不支持视频"
            )
        offset = int(feature.mm_position.offset)
        actual_tokens = int(feature.mm_position.length)
        text_len = offset - sequence_start
        if text_len < 0:
            raise RuntimeError("overlapping multimodal placeholders")
        if text_len:
            text_positions = (
                np.broadcast_to(np.arange(text_len), (3, text_len))
                + position_start
            )
            pieces.append(text_positions)
            position_start = int(text_positions.max()) + 1

        t, h, w = feature.data["image_grid_thw"].data.tolist()
        grid = np.indices(
            (int(t), int(h) // merge_size, int(w) // merge_size)
        ).reshape(3, -1)
        if actual_tokens > grid.shape[1]:
            raise RuntimeError(
                "image placeholder is longer than the unpruned visual grid: "
                f"placeholder={actual_tokens}, grid={grid.shape[1]}"
            )
        grid = grid[:, :actual_tokens] + position_start
        pieces.append(grid)
        if grid.size:
            position_start = int(grid.max()) + 1
        sequence_start = offset + actual_tokens

    trailing_len = len(input_tokens) - sequence_start
    if trailing_len < 0:
        raise RuntimeError("multimodal placeholder extends past the prompt")
    if trailing_len:
        pieces.append(
            np.broadcast_to(np.arange(trailing_len), (3, trailing_len))
            + position_start
        )
    positions = np.concatenate(pieces, axis=1) if pieces else np.empty((3, 0))
    if positions.shape[1] != len(input_tokens):
        raise RuntimeError(
            "provisional MRoPE length does not match prompt length: "
            f"positions={positions.shape[1]}, prompt={len(input_tokens)}"
        )
    delta = int(positions.max() + 1 - len(input_tokens)) if positions.size else 0
    return torch.from_numpy(positions), delta


def normalize_image_prefill_boundary(
    input_ids: list[int] | torch.Tensor,
    num_computed_tokens: int,
    vision_start_token_id: int,
) -> int:
    """Include ``vision_start`` when a prefill chunk begins at image tokens.

    vLLM's EVS helper correctly handles chunks that start before an image or
    inside an already-started image.  It misses the exact boundary where the
    previous chunk ended immediately after ``vision_start``: no image token has
    been seen yet, but there is also no *next* vision-start marker to find.  In
    that one case, rewind one position so the helper follows its normal
    start-of-media branch.  The marker itself is not overwritten.
    """
    # During decode vLLM counts generated tokens as computed, while input_ids
    # still contains only the prompt.  The upstream helper already treats that
    # state as fully processed, so there is no boundary to normalize.
    if num_computed_tokens <= 0 or num_computed_tokens > len(input_ids):
        return num_computed_tokens
    previous = input_ids[num_computed_tokens - 1]
    if isinstance(previous, torch.Tensor):
        previous = int(previous.item())
    if previous == vision_start_token_id:
        return num_computed_tokens - 1
    return num_computed_tokens


class RSMQwen3_5MultiModalProcessor(Qwen3VLMultiModalProcessor):
    """Make image placeholder counts match the process-level keep ratio."""

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        updates = list(
            super()._get_prompt_updates(
                mm_items, hf_processor_mm_kwargs, out_mm_kwargs
            )
        )
        _method, keep_ratio = _settings()
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        merge_length = image_processor.merge_size**2

        def image_replacement(item_idx: int) -> list[int]:
            out_item = out_mm_kwargs["image"][item_idx]
            grid_thw = out_item["image_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)
            token_count = int(grid_thw.prod()) // merge_length
            keep_count = min(token_count, max(1, int(token_count * keep_ratio)))
            return [hf_processor.image_token_id] * keep_count

        replacement = PromptReplacement(
            modality="image",
            target=hf_processor.image_token,
            replacement=image_replacement,
        )
        return [replacement if update.modality == "image" else update for update in updates]


@MULTIMODAL_REGISTRY.register_processor(
    RSMQwen3_5MultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class RSMQwen3_5ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """vLLM-native Qwen3.5-VL image pruning with corrected MRoPE."""

    supports_multimodal_pruning = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.prune_method, self.prune_keep_ratio = _settings()
        configured_prune_ratio = vllm_config.model_config.multimodal_config.video_pruning_rate
        expected = 1.0 - self.prune_keep_ratio
        if configured_prune_ratio is None or abs(configured_prune_ratio - expected) > 1e-9:
            raise RuntimeError(
                "vLLM multimodal pruning rate does not match RS-MLLM policy: "
                f"configured={configured_prune_ratio}, expected={expected}"
            )
        self.video_pruning_rate = configured_prune_ratio
        self.is_multimodal_pruning_enabled = True
        self._pruning_shape_logged = False
        logger.info(
            "RS-MLLM Qwen3.5 visual-token pruning enabled: method=%s keep_ratio=%.4f",
            self.prune_method,
            self.prune_keep_ratio,
        )
        print(
            "[rs-mllm-pruning] Qwen3.5 plugin active: "
            f"method={self.prune_method}, keep_ratio={self.prune_keep_ratio:.4f}",
            flush=True,
        )

    def _postprocess_image_embeds_evs(
        self,
        image_embeds_split: tuple[torch.Tensor, ...],
        image_input: dict[str, Any],
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        merge_size = self.visual.spatial_merge_size
        outputs: list[torch.Tensor] = []
        for embeddings, size in zip(image_embeds_split, grid_thw.tolist()):
            original_count = int(embeddings.shape[0])
            positions = compute_mrope_for_media(size, merge_size).to(
                embeddings.device, non_blocking=True
            )
            indices = select_token_indices(
                embeddings, self.prune_method, self.prune_keep_ratio
            )
            embeddings = embeddings[indices]
            positions = positions[indices]
            # Qwen3-VL's pruning protocol uses five trailing channels:
            # [t, h, w, is_vision_start, is_video]. Images set both flags to 0.
            positions = torch.cat(
                [positions, torch.zeros_like(positions[:, :1])], dim=1
            )
            outputs.append(torch.cat([embeddings, positions], dim=1))
            if not self._pruning_shape_logged:
                logger.info(
                    "RS-MLLM image visual tokens: %d -> %d (method=%s)",
                    original_count,
                    int(embeddings.shape[0]),
                    self.prune_method,
                )
                print(
                    "[rs-mllm-pruning] image visual tokens: "
                    f"{original_count} -> {int(embeddings.shape[0])}",
                    flush=True,
                )
                self._pruning_shape_logged = True
        return tuple(outputs)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[Any],
    ) -> tuple[torch.Tensor, int]:
        return pruned_image_mrope_input_positions(
            input_tokens, mm_features, self.config
        )

    def embed_multimodal(self, **kwargs: object):
        if "pixel_values_videos" in kwargs or "video_embeds" in kwargs:
            raise RuntimeError(
                "RS-MLLM Qwen3.5 Token 剪枝当前只支持图片输入，不支持视频"
            )
        return super().embed_multimodal(**kwargs)

    def recompute_mrope_positions(
        self,
        input_ids: list[int] | torch.Tensor,
        multimodal_embeddings: Sequence[torch.Tensor],
        mrope_positions: torch.LongTensor,
        num_computed_tokens: int,
    ) -> tuple[Sequence[torch.Tensor], torch.Tensor, int]:
        normalized_boundary = normalize_image_prefill_boundary(
            input_ids,
            num_computed_tokens,
            self.config.vision_start_token_id,
        )
        return self._recompute_mrope_positions(
            input_ids=input_ids,
            multimodal_embeddings=multimodal_embeddings,
            mrope_positions=mrope_positions,
            num_computed_tokens=normalized_boundary,
            image_token_id=self.config.image_token_id,
            video_token_id=self.config.video_token_id,
            vision_start_token_id=self.config.vision_start_token_id,
        )
