"""Visual token pruner base class.

Each pruner takes a batch of visual token features ``[N, D]`` and returns the
sorted indices of the tokens to keep. This decouples the scoring logic from the
model-specific forward-pass integration, so every pruner can be visualized
through the same interface (see ``prune.viz``).
"""

import torch


class VisualTokenPruner:
    """Abstract base class for training-free visual token pruning.

    Subclasses implement :meth:`select`, which scores tokens and returns the
    indices of the ``keep_count`` tokens to retain (sorted ascending).
    """

    name = "base"

    def select(self, features: torch.Tensor, keep_count: int, seed=None) -> torch.Tensor:
        """Select ``keep_count`` tokens to keep.

        Args:
            features: ``[N, D]`` visual token features (N tokens, D dims).
            keep_count: number of tokens to retain.
            seed: optional random seed (for stochastic pruners).

        Returns:
            Sorted ``torch.LongTensor`` of shape ``[keep_count]`` giving the
            indices of retained tokens.
        """
        raise NotImplementedError

    def __call__(self, features: torch.Tensor, keep_count: int, seed=None) -> torch.Tensor:
        return self.select(features, keep_count, seed=seed)

    def keep_mask(self, features: torch.Tensor, keep_count: int, seed=None) -> torch.Tensor:
        """Return a bool mask of shape ``[N]`` where True means keep."""
        n = features.shape[0]
        mask = torch.zeros(n, dtype=torch.bool, device=features.device)
        keep = self.select(features, keep_count, seed=seed)
        mask[keep] = True
        return mask
