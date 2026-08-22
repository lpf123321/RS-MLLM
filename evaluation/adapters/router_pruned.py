"""Multi-expert router adapter with pre-LLM visual-token pruning.

Combines the RouterAdapter (single Qwen3.5-4B base + N expert LoRAs, routed by
prompt) with the batched pre-LLM pruning from ``scripts/prune.py``.  The pruning
is applied in the monkey-patched ``Qwen3_5Model.forward``, so it runs on the
already-batched inputs and supports ``eval_batch_size > 1`` (e.g. 32).

Two modes:
  - uniform: one ``keep_ratio`` for every task (the default).
  - task-adaptive: ``task_keep_ratio`` maps each task (vqa/caption/mcq/referring/
    change) to its own keep ratio; samples are grouped by task and each group is
    generated with its own expert LoRA + keep ratio.

The visual features (``pooler_output``) are expert-independent because the LoRA
adapters only target the language-model projections, so pruning is computed once
per image and is orthogonal to the expert routing.
"""

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from evaluation.adapters.router import RouterAdapter
from evaluation.router import rules


class RouterPrunedAdapter(RouterAdapter):
    def __init__(
        self,
        model_path: str,
        general_lora: str = None,
        grounding_lora: str = None,
        change_lora: str = None,
        expert_lora: dict = None,
        device: str = "cuda",
        max_new_tokens: int = 256,
        system_prompt: str = "",
        image_min_pixels: int = None,
        image_max_pixels: int = None,
        prune_method: str = "l2",
        keep_ratio: float = 1.0,
        task_keep_ratio: dict = None,
        in_llm: str = None,
        in_llm_k: int = 2,
        in_llm_keep_ratio: float = None,
        force_think: bool = False,
    ):
        if prune_method not in ("l2", "divprune", "none"):
            raise ValueError(f"prune_method must be 'l2'/'divprune'/'none', got {prune_method!r}")
        if in_llm not in (None, "k2", "clip"):
            raise ValueError(f"in_llm must be None/'k2'/'clip', got {in_llm!r}")

        # Patch Qwen3_5Model.forward before the model is loaded (batched pruning).
        from scripts.prune import apply_pruning

        apply_pruning(model_path=model_path)

        super().__init__(
            model_path,
            general_lora=general_lora,
            grounding_lora=grounding_lora,
            change_lora=change_lora,
            expert_lora=expert_lora,
            device=device,
            max_new_tokens=max_new_tokens,
            system_prompt=system_prompt,
            image_min_pixels=image_min_pixels,
            image_max_pixels=image_max_pixels,
            force_think=force_think,
        )

        self.prune_method = prune_method
        self.keep_ratio = keep_ratio
        self.task_keep_ratio = dict(task_keep_ratio) if task_keep_ratio else None
        self.in_llm = in_llm
        self.in_llm_k = in_llm_k
        self.in_llm_keep_ratio = (
            in_llm_keep_ratio if in_llm_keep_ratio is not None else keep_ratio
        )
        self.set_keep_ratio(keep_ratio)
        if in_llm is not None:
            self._enable_in_llm()

    def _backbone(self):
        """Return the Qwen3_5Model instance (the object the patched forward runs on)."""
        node = self.model
        for _ in range(6):
            if hasattr(node, "language_model"):
                return node
            nxt = getattr(node, "model", None)
            if nxt is None or nxt is node:
                break
            node = nxt
        raise RuntimeError("Could not locate Qwen3_5Model (language_model) in the model")

    def _text_model(self):
        return self._backbone().language_model

    def _enable_in_llm(self):
        from scripts.prune import enable_inllm_on_text_model

        enable_inllm_on_text_model(
            self._text_model(), self.in_llm, self.in_llm_k,
            1.0 - self.in_llm_keep_ratio,
        )

    def set_in_llm_keep_ratio(self, keep_ratio: float):
        """运行时调整 in-LLM 保留比例（无需重新 patch）。"""
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
        self.in_llm_keep_ratio = keep_ratio
        from scripts.prune import set_inllm_ratio

        set_inllm_ratio(self._text_model(), 1.0 - keep_ratio)

    def _apply_prune_ratio(self, keep_ratio: float):
        backbone = self._backbone()
        if self.prune_method == "none" or keep_ratio >= 1.0:
            backbone._prune_method = None
            backbone._prune_r = 0.0
        else:
            backbone._prune_method = self.prune_method
            backbone._prune_r = 1.0 - keep_ratio

    def set_keep_ratio(self, keep_ratio: float):
        """Set the retention ratio for uniform mode; ``keep_ratio=1.0`` disables pruning."""
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
        self.keep_ratio = keep_ratio
        self._apply_prune_ratio(keep_ratio)

    def _keep_ratio_for(self, task: str) -> float:
        if self.task_keep_ratio is not None:
            return self.task_keep_ratio.get(task, self.keep_ratio)
        return self.keep_ratio

    def _group_key(self, prompt: str) -> str:
        """Task-adaptive 模式按 task 分组；否则按 expert 分组。"""
        if self.task_keep_ratio is not None:
            return rules.route_task(prompt)
        return rules.route(prompt)

    def _prepare_group(self, key: str):
        """每组生成前：切到对应 expert 并设置该组的 keep_ratio。"""
        if self.task_keep_ratio is not None:
            # key 是 task
            task = key
            expert = self._task_to_expert.get(task, rules.GENERAL)
            self._switch(expert)
            self._apply_prune_ratio(self._keep_ratio_for(task))
        else:
            # key 是 expert
            self._switch(key)
