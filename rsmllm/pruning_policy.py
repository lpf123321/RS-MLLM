"""Shared task-adaptive visual-token pruning policy for vLLM entry points."""
from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import metadata

from evaluation.router.task_prune_config import THRESHOLD_TASK_PRUNE_CONFIG

VLLM_PLUGIN_ENTRYPOINT = "rs_mllm_token_pruning"
PRUNING_ENV_ENABLED = "RSM_VLLM_TOKEN_PRUNING"
PRUNING_ENV_METHOD = "RSM_VLLM_PRUNE_METHOD"
PRUNING_ENV_KEEP_RATIO = "RSM_VLLM_PRUNE_KEEP_RATIO"

# The current policy assigns the same setting to VQA and MCQ, so one fixed
# pruning configuration per resident expert is sufficient for online serving.
EXPERT_POLICY_TASK = {
    "general": "vqa",
    "grounding": "referring",
    "change": "change",
    "caption": "caption",
}

ROUTE_TASK = {
    "vrsbench-vqa": "vqa",
    "mme": "mcq",
    "xlrs-bench-lite": "mcq",
    "vrsbench-referring": "referring",
    "xlrs-bench-grounding-en": "referring",
    "levir-cc": "change",
    "vrsbench-caption": "caption",
    "xlrs-bench-caption": "caption",
}


@dataclass(frozen=True)
class PruningSpec:
    task: str
    method: str
    keep_ratio: float
    policy: str

    @property
    def enabled(self) -> bool:
        return self.keep_ratio < 1.0

    @property
    def prune_ratio(self) -> float:
        return 1.0 - self.keep_ratio


def parse_keep_ratio(value: str | float) -> str | float:
    """Parse ``adaptive`` or a visual-token retention ratio in ``(0, 1]``."""
    if isinstance(value, str) and value.strip().lower() == "adaptive":
        return "adaptive"
    try:
        ratio = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("保留率必须是 adaptive 或 (0, 1] 内的数字") from exc
    if not 0.0 < ratio <= 1.0:
        raise ValueError("视觉 Token 保留率必须在 (0, 1] 内")
    return ratio


def pruning_spec(task: str, requested_keep_ratio: str | float) -> PruningSpec:
    """Keep the task's method while resolving a user-selected keep ratio."""
    if task not in THRESHOLD_TASK_PRUNE_CONFIG:
        raise ValueError(f"未知剪枝任务: {task}")
    requested = parse_keep_ratio(requested_keep_ratio)
    default = THRESHOLD_TASK_PRUNE_CONFIG[task]
    if requested == "adaptive":
        keep_ratio = float(default["keep_ratio"])
        policy = "task_adaptive"
    else:
        keep_ratio = float(requested)
        policy = "task_method_user_keep_ratio"
    return PruningSpec(
        task=task,
        method=str(default["method"]),
        keep_ratio=keep_ratio,
        policy=policy,
    )


def expert_pruning_spec(
    expert: str, requested_keep_ratio: str | float
) -> PruningSpec:
    try:
        task = EXPERT_POLICY_TASK[expert]
    except KeyError as exc:
        raise ValueError(f"未知专家: {expert}") from exc
    return pruning_spec(task, requested_keep_ratio)


def configure_vllm_pruning_env(spec: PruningSpec, env: dict[str, str]) -> None:
    """Configure one vLLM process; disabled runs retain the stock model."""
    for key in (
        PRUNING_ENV_ENABLED,
        PRUNING_ENV_METHOD,
        PRUNING_ENV_KEEP_RATIO,
    ):
        env.pop(key, None)
    if not spec.enabled:
        return
    env[PRUNING_ENV_ENABLED] = "1"
    env[PRUNING_ENV_METHOD] = spec.method
    env[PRUNING_ENV_KEEP_RATIO] = format(spec.keep_ratio, ".12g")
    if "VLLM_PLUGINS" in env:
        allowed = [item for item in env["VLLM_PLUGINS"].split(",") if item]
        if VLLM_PLUGIN_ENTRYPOINT not in allowed:
            allowed.append(VLLM_PLUGIN_ENTRYPOINT)
        env["VLLM_PLUGINS"] = ",".join(allowed)


def vllm_plugin_available() -> bool:
    """Return whether the pinned evaluation environment installed our plugin."""
    points = metadata.entry_points()
    selected = points.select(group="vllm.general_plugins")
    return any(point.name == VLLM_PLUGIN_ENTRYPOINT for point in selected)


def require_vllm_plugin() -> None:
    if not vllm_plugin_available():
        raise RuntimeError(
            "vLLM Token 剪枝插件未安装；请重新运行 "
            "bash evaluation/vllm_eval/setup_env.sh"
        )


def activate_current_process(spec: PruningSpec) -> None:
    """Set pruning env before importing vLLM in the current process."""
    configure_vllm_pruning_env(spec, os.environ)
    if spec.enabled:
        require_vllm_plugin()
