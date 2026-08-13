"""任务路由规则（对应 TaskExpertLoRA.md §6）。

匹配顺序：任务前缀 → 关键词 → 默认 General Understanding Expert。
"""
import re

CHANGE_KEYWORDS = (
    "change", "difference", "before", "after", "compare", "changed"
)
GROUNDING_KEYWORDS = (
    "where", "location", "position", "find", "locate", "region"
)

GENERAL = "general"
GROUNDING = "grounding"
CHANGE = "change"


def _has_prefix(text: str):
    """识别评估/训练侧任务前缀（第一优先级）。返回 expert 名或 None。"""
    t = text.strip()
    if t.startswith("[REF]"):
        return GROUNDING
    if t.startswith("[CAP]") or t.startswith("[VQA]"):
        return GENERAL
    return None


def _has_mcq(text: str):
    """识别选择题格式（MME/XLRS 用 [MCQ] 前缀或选项列表）。"""
    return "[MCQ]" in text or re.search(r"\n\(A\) .+\n\(B\) .+", text) is not None


def route(prompt: str) -> str:
    """根据 prompt 返回 expert 名。"""
    t = (prompt or "").strip()
    if not t:
        return GENERAL

    pref = _has_prefix(t)
    if pref:
        return pref

    low = t.lower()
    if _has_mcq(t):
        return GENERAL

    if any(kw in low for kw in CHANGE_KEYWORDS):
        return CHANGE
    if any(kw in low for kw in GROUNDING_KEYWORDS):
        return GROUNDING
    return GENERAL
