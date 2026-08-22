"""DivPrune visual-token selection."""

from typing import Optional

import torch
import torch.nn.functional as functional

from .base import TokenPruner, validate_selection_inputs


class DivPruneTokenPruner(TokenPruner):
    """Select a greedy max-min diverse subset using cosine distance."""

    name = "divprune"

    def select(
        self, features: torch.Tensor, keep_count: int, seed: Optional[int] = None
    ) -> torch.LongTensor:
        del seed
        token_count = validate_selection_inputs(features, keep_count)
        if keep_count == token_count:
            return torch.arange(token_count, device=features.device)

        normalized = functional.normalize(features.float(), dim=-1)
        distances = 1.0 - normalized @ normalized.transpose(0, 1)
        distances.fill_diagonal_(float("inf"))

        first = distances.min(dim=0).values.argmax()
        selected = torch.empty(keep_count, dtype=torch.long, device=features.device)
        selected[0] = first
        selected_mask = torch.zeros(token_count, dtype=torch.bool, device=features.device)
        selected_mask[first] = True
        minimum_distances = distances[first].clone()

        for index in range(1, keep_count):
            scores = minimum_distances.masked_fill(selected_mask, float("-inf"))
            next_index = scores.argmax()
            selected[index] = next_index
            selected_mask[next_index] = True
            minimum_distances = torch.minimum(minimum_distances, distances[next_index])

        return torch.sort(selected).values
