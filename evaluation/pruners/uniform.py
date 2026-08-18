import torch

from .base import VisualTokenPruner


class UniformPruner(VisualTokenPruner):
    name = "uniform"

    def select(self, features, keep_count, seed):
        count = features.shape[0]
        if keep_count >= count:
            return torch.arange(count, device=features.device)

        indices = torch.linspace(
            0, count - 1, steps=keep_count, device=features.device
        ).round().long()
        return torch.unique(indices, sorted=True)