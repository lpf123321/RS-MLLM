"""Delta-expert router with standalone token-compression selectors."""

from evaluation.adapters.router import RouterAdapter
from evaluation.router import rules


class DeltaPrunedAdapter(RouterAdapter):
    """RouterAdapter + batched pre-LLM token_compression pruning.

    The parent loads either legacy LoRA directories or the current full-rank
    ``delta_model.pt`` experts. This adapter adds the selector registry for
    uniform or task-adaptive token retention.
    """

    def __init__(
        self,
        model_path: str,
        general_lora: str = None,
        grounding_lora: str = None,
        change_lora: str = None,
        expert_lora: dict = None,
        prune_method: str = "l2norm",
        keep_ratio: float = 1.0,
        task_keep_ratio: dict = None,
        pruner_seed: int = 2026,
        **kwargs,
    ):
        from scripts.prune import apply_pruning
        from token_compression import build_token_pruner

        apply_pruning(model_path=model_path)
        if prune_method not in ("none", "fourier", "uniform", "random", "mmtok",
                                "l2norm", "divprune", "scope_l2"):
            raise ValueError(f"Unknown compression method: {prune_method}")

        super().__init__(
            model_path,
            general_lora=general_lora,
            grounding_lora=grounding_lora,
            change_lora=change_lora,
            expert_lora=expert_lora,
            **kwargs,
        )
        self.prune_method = prune_method
        self.keep_ratio = keep_ratio
        self.task_keep_ratio = dict(task_keep_ratio) if task_keep_ratio else None
        self.pruner_seed = pruner_seed
        self._token_pruner = (
            None if prune_method in ("none", "fourier") else build_token_pruner(prune_method)
        )
        self._set_uniform_keep_ratio(keep_ratio)

    def _backbone(self):
        node = self.model
        for _ in range(6):
            if hasattr(node, "language_model"):
                return node
            nxt = getattr(node, "model", None)
            if nxt is None or nxt is node:
                break
            node = nxt
        raise RuntimeError("Could not locate Qwen3_5Model (language_model)")

    def _set_uniform_keep_ratio(self, keep_ratio: float):
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
        self.keep_ratio = keep_ratio
        self._apply_group_ratio(keep_ratio)

    def _apply_group_ratio(self, keep_ratio: float):
        backbone = self._backbone()
        if self.prune_method in ("none", "fourier") or keep_ratio >= 1.0:
            backbone._prune_method = None
            backbone._prune_r = 0.0
            return
        backbone._prune_method = "token_compression"
        backbone._prune_r = 1.0 - keep_ratio
        backbone._token_pruner = self._token_pruner
        backbone._token_pruner_seed = self.pruner_seed

    def _group_key(self, prompt: str) -> str:
        if self.task_keep_ratio is not None:
            return rules.route_task(prompt)
        return rules.route(prompt)

    def _prepare_group(self, key: str):
        if self.task_keep_ratio is None:
            self._switch(key)
            self._apply_group_ratio(self.keep_ratio)
            return
        task = key
        self._switch(self._task_to_expert.get(task, rules.GENERAL))
        self._apply_group_ratio(self.task_keep_ratio.get(task, self.keep_ratio))
