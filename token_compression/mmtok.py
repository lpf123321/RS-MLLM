"""MMTok-style greedy coverage selection."""

from typing import Optional

import torch
import torch.nn.functional as functional

from .base import TokenPruner, validate_selection_inputs


class MMTokTokenPruner(TokenPruner):
    """Select a subset with greedy maximum cosine-similarity coverage."""

    name = "mmtok"

    def select(
        self, features: torch.Tensor, keep_count: int, seed: Optional[int] = None
    ) -> torch.LongTensor:
        del seed
        token_count = validate_selection_inputs(features, keep_count)
        if keep_count == token_count:
            return torch.arange(token_count, device=features.device)

        normalized = functional.normalize(features.float(), dim=-1)
        similarity = normalized @ normalized.T
        selected = []
        available = torch.ones(token_count, dtype=torch.bool, device=features.device)

        first = similarity.sum(dim=0).argmax()
        selected.append(first)
        available[first] = False
        coverage = similarity[:, first]

        for _ in range(1, keep_count):
            gains = torch.maximum(coverage[:, None], similarity).sum(dim=0)
            gains -= coverage.sum()
            gains.masked_fill_(~available, -torch.inf)
            chosen = gains.argmax()
            selected.append(chosen)
            available[chosen] = False
            coverage = torch.maximum(coverage, similarity[:, chosen])

        return torch.sort(torch.stack(selected)).values
