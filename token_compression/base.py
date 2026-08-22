"""Shared interface and validation for visual-token selectors."""

from abc import ABC, abstractmethod
from typing import Optional

import torch


class TokenPruner(ABC):
    """Select visual-token indices while preserving their original order."""

    name = "base"

    @abstractmethod
    def select(
        self,
        features: torch.Tensor,
        keep_count: int,
        seed: Optional[int] = None,
    ) -> torch.LongTensor:
        """Return sorted token indices to retain."""


def validate_selection_inputs(features: torch.Tensor, keep_count: int) -> int:
    """Validate a token-feature matrix and return its number of tokens."""
    if features.ndim != 2:
        raise ValueError(
            "features must have shape [num_tokens, hidden_size], "
            f"got {tuple(features.shape)}."
        )

    token_count = features.shape[0]
    if not 1 <= keep_count <= token_count:
        raise ValueError(
            f"keep_count must be in [1, {token_count}], got {keep_count}."
        )
    return token_count
