import math

import torch
import torch.nn.functional as functional

from evaluation.adapters.qwen35_pruned import Qwen35PrunedAdapter


class Qwen35DivPruneAdapter(Qwen35PrunedAdapter):
    """Qwen3.5-compatible DivPrune: L2 centers plus cosine-based token merging."""

    def __init__(self, *args, keep_ratio=0.5, pruner_seed=2026, **kwargs):
        super().__init__(
            *args,
            keep_ratio=keep_ratio,
            pruner="l2norm",
            pruner_seed=pruner_seed,
            **kwargs,
        )
        self.pruner_name = "divprune_l2"

    @staticmethod
    def _merge_features(features, keep_count):
        count = features.shape[0]
        if keep_count >= count:
            return torch.arange(count, device=features.device), features

        saliency = features.float().norm(dim=-1)
        centers = torch.topk(saliency, k=keep_count, largest=True).indices
        centers = torch.sort(centers).values

        all_indices = torch.arange(count, device=features.device)
        dropped_mask = torch.ones(count, dtype=torch.bool, device=features.device)
        dropped_mask[centers] = False
        dropped = all_indices[dropped_mask]

        # Each dropped token is merged into its nearest retained center.
        normalized = functional.normalize(features.float(), dim=-1)
        assignment = (normalized[dropped] @ normalized[centers].T).argmax(dim=-1)

        weights = saliency.clamp_min(torch.finfo(saliency.dtype).eps)
        merged_sum = features[centers].float() * weights[centers, None]
        merged_weight = weights[centers].clone()

        merged_sum.index_add_(
            0,
            assignment,
            features[dropped].float() * weights[dropped, None],
        )
        merged_weight.index_add_(0, assignment, weights[dropped])

        merged = (merged_sum / merged_weight[:, None]).to(dtype=features.dtype)
        return centers, merged

    def _prune_inputs(self, inputs):
        input_ids = inputs["input_ids"]
        if input_ids.shape[0] != 1:
            raise ValueError("DivPrune inference requires --eval_batch_size 1.")

        image_token_id = self.model.config.image_token_id
        image_mask = input_ids.eq(image_token_id)
        image_positions = image_mask[0].nonzero(as_tuple=False).flatten()
        if image_positions.numel() == 0:
            return None

        vision = self.model.model.get_image_features(
            inputs["pixel_values"],
            inputs["image_grid_thw"],
            return_dict=True,
        )
        pooled = vision.pooler_output
        pooled_features = [pooled] if isinstance(pooled, torch.Tensor) else list(pooled)

        expected = sum(feature.shape[0] for feature in pooled_features)
        if expected != image_positions.numel():
            raise RuntimeError(
                f"Visual feature count {expected} != image placeholder count "
                f"{image_positions.numel()}."
            )

        selected_positions = []
        selected_features = []
        offset = 0

        for feature in pooled_features:
            count = feature.shape[0]
            keep_count = max(1, math.ceil(count * self.keep_ratio))
            local_indices, merged_features = self._merge_features(feature, keep_count)

            positions = image_positions[offset:offset + count]
            selected_positions.append(positions.index_select(0, local_indices))
            selected_features.append(merged_features)
            offset += count

        selected_positions = torch.cat(selected_positions)
        selected_features = torch.cat(selected_features)

        full_position_ids, _ = self.model.model.get_rope_index(
            input_ids=input_ids,
            mm_token_type_ids=inputs["mm_token_type_ids"],
            image_grid_thw=inputs["image_grid_thw"],
            attention_mask=inputs["attention_mask"],
        )

        inputs_embeds = self.model.get_input_embeddings()(input_ids).clone()
        inputs_embeds[0, selected_positions] = selected_features

        keep_mask = ~image_mask
        keep_mask[0, selected_positions] = True
        keep_mask = keep_mask[0]

        pruned = {
            "input_ids": input_ids[:, keep_mask],
            "inputs_embeds": inputs_embeds[:, keep_mask],
            "attention_mask": inputs["attention_mask"][:, keep_mask],
            "mm_token_type_ids": inputs["mm_token_type_ids"][:, keep_mask],
            "position_ids": full_position_ids[..., keep_mask],
        }

        max_position = pruned["position_ids"].amax(dim=0).amax(dim=-1, keepdim=True)
        self.model.model.rope_deltas = (
            max_position + 1 - pruned["attention_mask"].sum(dim=-1, keepdim=True)
        )
        return pruned