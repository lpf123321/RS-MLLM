"""Seeded random visual-token selection."""

from typing import Optional

import torch

from .base import TokenPruner, validate_selection_inputs


class RandomTokenPruner(TokenPruner):
    """Keep a reproducible random subset of visual tokens."""

    name = "random"

    def select(
        self, features: torch.Tensor, keep_count: int, seed: Optional[int] = None
    ) -> torch.LongTensor:
        token_count = validate_selection_inputs(features, keep_count)
        if keep_count == token_count:
            return torch.arange(token_count, device=features.device)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(0 if seed is None else seed)
        indices = torch.randperm(token_count, generator=generator)[:keep_count]
        return torch.sort(indices.to(features.device)).values
