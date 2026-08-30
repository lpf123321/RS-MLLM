"""SCOPE token selection with L2 saliency."""

from typing import Optional

import torch
import torch.nn.functional as functional

from .base import TokenPruner, validate_selection_inputs


class ScopeL2TokenPruner(TokenPruner):
    """Balance L2-norm saliency with cosine-similarity set coverage."""

    name = "scope_l2"

    def select(
        self, features: torch.Tensor, keep_count: int, seed: Optional[int] = None
    ) -> torch.LongTensor:
        del seed
        token_count = validate_selection_inputs(features, keep_count)
        if keep_count == token_count:
            return torch.arange(token_count, device=features.device)

        feature_float = features.float()
        normalized = functional.normalize(feature_float, dim=-1)
        similarity = normalized @ normalized.T
        saliency = torch.linalg.vector_norm(feature_float, ord=2, dim=-1)
        saliency = saliency / saliency.max().clamp_min(torch.finfo(saliency.dtype).eps)

        selected = []
        available = torch.ones(token_count, dtype=torch.bool, device=features.device)
        coverage = torch.zeros(token_count, dtype=similarity.dtype, device=features.device)

        for _ in range(keep_count):
            gains = (similarity - coverage[:, None]).clamp_min(0).sum(dim=0)
            scores = gains * saliency
            scores.masked_fill_(~available, -torch.inf)
            chosen = scores.argmax()
            selected.append(chosen)
            available[chosen] = False
            coverage = torch.maximum(coverage, similarity[:, chosen])

        return torch.sort(torch.stack(selected)).values
