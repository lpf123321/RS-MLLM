import torch
import torch.nn.functional as functional

from .base import VisualTokenPruner


class ScopeL2Pruner(VisualTokenPruner):
    """
    Qwen3.5-compatible SCOPE.

    Saliency: visual feature L2 norm.
    Coverage: marginal gain under cosine-similarity set coverage.
    """

    name = "scope_l2"

    def select(self, features, keep_count, seed):
        count = features.shape[0]
        if keep_count >= count:
            return torch.arange(count, device=features.device)

        feature_float = features.float()
        normalized = functional.normalize(feature_float, dim=-1)
        similarity = normalized @ normalized.T

        saliency = torch.linalg.vector_norm(feature_float, ord=2, dim=-1)
        saliency = saliency / saliency.max().clamp_min(torch.finfo(saliency.dtype).eps)

        selected = []
        available = torch.ones(count, dtype=torch.bool, device=features.device)
        coverage = torch.zeros(count, dtype=similarity.dtype, device=features.device)

        for _ in range(keep_count):
            gains = (similarity - coverage[:, None]).clamp_min(0).sum(dim=0)
            scores = gains * saliency
            scores.masked_fill_(~available, -torch.inf)

            chosen = scores.argmax()
            selected.append(chosen)
            available[chosen] = False
            coverage = torch.maximum(coverage, similarity[:, chosen])

        return torch.sort(torch.stack(selected)).values