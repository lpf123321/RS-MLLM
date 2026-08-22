"""L2-norm visual-token compression.

This module ranks visual tokens by the L2 norm of each token feature and
retains the highest-scoring tokens. It only requires PyTorch.
"""

from typing import Optional

import torch

from .base import TokenPruner, validate_selection_inputs


def select_l2_tokens(features: torch.Tensor, keep_count: int) -> torch.LongTensor:
    """Return sorted indices of the ``keep_count`` strongest token features.

    Args:
        features: Token features with shape ``[num_tokens, hidden_size]``.
        keep_count: Number of tokens to retain, in ``[1, num_tokens]``.

    Returns:
        A one-dimensional ``torch.LongTensor`` of sorted token indices on the
        same device as ``features``.
    """
    token_count = validate_selection_inputs(features, keep_count)

    if keep_count == token_count:
        return torch.arange(token_count, device=features.device)

    scores = torch.linalg.vector_norm(features.float(), ord=2, dim=-1)
    indices = torch.topk(scores, k=keep_count, largest=True, sorted=False).indices
    return torch.sort(indices).values


class L2NormTokenPruner(TokenPruner):
    """Select visual tokens by descending feature L2 norm.

    The ``seed`` parameter is accepted to match common token-pruner interfaces;
    L2 selection is deterministic and does not use it.
    """

    name = "l2norm"

    def select(
        self,
        features: torch.Tensor,
        keep_count: int,
        seed: Optional[int] = None,
    ) -> torch.LongTensor:
        del seed
        return select_l2_tokens(features, keep_count)
