import torch

from .base import VisualTokenPruner


class RandomPruner(VisualTokenPruner):
    name = "random"

    def select(self, features, keep_count, seed):
        count = features.shape[0]
        if keep_count >= count:
            return torch.arange(count, device=features.device)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        indices = torch.randperm(count, generator=generator)[:keep_count]
        return torch.sort(indices.to(features.device)).values