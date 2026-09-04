"""Registration hook loaded by vLLM in the API process and engine workers."""
from __future__ import annotations

import os

PLUGIN_ENV = "RSM_VLLM_TOKEN_PRUNING"


def register() -> None:
    """Override Qwen3.5-VL only for explicitly enabled pruning processes."""
    if os.environ.get(PLUGIN_ENV) != "1":
        return
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model(
        "Qwen3_5ForConditionalGeneration",
        "rs_mllm_vllm_plugin.model:RSMQwen3_5ForConditionalGeneration",
    )

