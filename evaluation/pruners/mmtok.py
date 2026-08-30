import torch
import torch.nn.functional as functional

from .base import VisualTokenPruner


class MMTokPruner(VisualTokenPruner):
    name = "mmtok"

    def select(self, features, keep_count, seed):
        count = features.shape[0]
        if keep_count >= count:
            return torch.arange(count, device=features.device)

        normalized = functional.normalize(features.float(), dim=-1)
        similarity = normalized @ normalized.T

        selected = []
        available = torch.ones(count, dtype=torch.bool, device=features.device)

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