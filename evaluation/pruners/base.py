from abc import ABC, abstractmethod

import torch


class VisualTokenPruner(ABC):
    name = "base"

    @abstractmethod
    def select(
        self,
        features: torch.Tensor,
        keep_count: int,
        seed: int,
    ) -> torch.LongTensor:
        """Return sorted visual-token indices to retain."""