import torch

from .base import VisualTokenPruner


class DivPrunePruner(VisualTokenPruner):
    """Select a max-min diverse subset of visual features.

    This follows DivPrune's greedy farthest-first selection using cosine
    distance.  Features are selected as-is; DivPrune does not merge dropped
    tokens into the retained embeddings.
    """

    name = "divprune"

    def select(self, features, keep_count, seed):
        count = features.shape[0]
        if keep_count >= count:
            return torch.arange(count, device=features.device)
        if keep_count < 1:
            raise ValueError("keep_count must be positive.")

        normalized = torch.nn.functional.normalize(features.float(), dim=-1)
        distances = 1.0 - normalized @ normalized.transpose(0, 1)

        # Pick the token furthest from its nearest other token, then greedily
        # maximize each candidate's minimum distance to the selected subset.
        distances.fill_diagonal_(float("inf"))
        first = distances.min(dim=0).values.argmax()

        selected = torch.empty(keep_count, dtype=torch.long, device=features.device)
        selected[0] = first
        selected_mask = torch.zeros(count, dtype=torch.bool, device=features.device)
        selected_mask[first] = True
        minimum_distances = distances[first].clone()

        for index in range(1, keep_count):
            scores = minimum_distances.masked_fill(selected_mask, float("-inf"))
            next_index = scores.argmax()
            selected[index] = next_index
            selected_mask[next_index] = True
            minimum_distances = torch.minimum(minimum_distances, distances[next_index])

        return torch.sort(selected).values
