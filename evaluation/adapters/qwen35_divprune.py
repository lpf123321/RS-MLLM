from evaluation.adapters.qwen35_pruned import Qwen35PrunedAdapter


class Qwen35DivPruneAdapter(Qwen35PrunedAdapter):
    """Qwen3.5 adapter using DivPrune's max-min diversity selection."""

    def __init__(self, *args, keep_ratio=0.5, pruner_seed=2026, **kwargs):
        super().__init__(
            *args,
            keep_ratio=keep_ratio,
            pruner="divprune",
            pruner_seed=pruner_seed,
            **kwargs,
        )
        self.pruner_name = "divprune"
