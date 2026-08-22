"""Uniform visual-token selection."""

from typing import Optional

import torch

from .base import TokenPruner, validate_selection_inputs


class UniformTokenPruner(TokenPruner):
    """Keep evenly spaced visual tokens."""

    name = "uniform"

    def select(
        self, features: torch.Tensor, keep_count: int, seed: Optional[int] = None
    ) -> torch.LongTensor:
        del seed
        token_count = validate_selection_inputs(features, keep_count)
        if keep_count == token_count:
            return torch.arange(token_count, device=features.device)

        return torch.linspace(
            0, token_count - 1, steps=keep_count, device=features.device
        ).round().long()
