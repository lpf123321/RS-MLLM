import torch

from .base import VisualTokenPruner


class L2NormPruner(VisualTokenPruner):
    """Retain visual tokens with the largest feature L2 norms."""

    name = "l2norm"

    def select(self, features, keep_count, seed):
        count = features.shape[0]
        if keep_count >= count:
            return torch.arange(count, device=features.device)

        scores = torch.linalg.vector_norm(features.float(), ord=2, dim=-1)
        indices = torch.topk(
            scores,
            k=keep_count,
            largest=True,
            sorted=False,
        ).indices
        return torch.sort(indices).values
