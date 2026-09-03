"""任务路由规则（对应 TaskExpertLoRA.md §6）。

匹配顺序：任务前缀 → 关键词 → 默认 General Understanding Expert。

同时提供比 expert 更细的 task 级识别，供任务自适应剪枝使用：
  - ``route(prompt)``       -> expert 名（general / grounding / change）
  - ``route_task(prompt)``  -> task 名（vqa / caption / mcq / referring / change）

task -> expert 的映射集中在 ``TASK_TO_EXPERT``，将来 expert 拆分（如 general
拆成 vqa/caption 两个 expert）只需改这一张表，不改识别逻辑。
"""
import re

CHANGE_KEYWORDS = (
    "change", "difference", "before", "after", "compare", "changed",
    "变化", "改变", "差异", "前后", "对比",
)
GROUNDING_KEYWORDS = (
    "where", "location", "position", "find", "locate", "region",
    "哪里", "位置", "定位", "找到", "区域", "坐标", "框出",
)
CAPTION_KEYWORDS = (
    "describe", "description", "caption", "captioning", "detailed description",
    "overall description", "partitioned description", "comprehensive inference",
    "describe the image", "generate a caption", "image caption",
    "描述", "图像描述", "图片内容", "场景描述",
)
CAPTION_INSTRUCTION_MARKERS = (
    "description is divided into three parts",
    "partitioned description",
    "comprehensive inference",
    "generating a detailed description",
)
GROUNDING_INSTRUCTION_MARKERS = (
    "identify the bounding box",
    "answer directly with the bounding box",
    "return the bounding box",
)

# expert 名
GENERAL = "general"
GROUNDING = "grounding"
CHANGE = "change"
CAPTION_EXPERT = "caption"

# task 名（比 expert 更细）
VQA = "vqa"
CAPTION = "caption"
MCQ = "mcq"
REFERRING = "referring"
CHANGE_TASK = "change"

# task -> expert 映射（将来 general 拆成 vqa/caption 两个 expert，只改这里）
TASK_TO_EXPERT = {
    VQA: GENERAL,
    CAPTION: CAPTION_EXPERT,
    MCQ: GENERAL,
    REFERRING: GROUNDING,
    CHANGE_TASK: CHANGE,
}


def _has_prefix(text: str):
    """识别评估/训练侧任务前缀（第一优先级）。返回 expert 名或 None。"""
    t = text.strip()
    if t.startswith("[REF]"):
        return GROUNDING
    if t.startswith("[CAP]"):
        return CAPTION_EXPERT
    if t.startswith("[VQA]"):
        return GENERAL
    return None


def _has_task_prefix(text: str):
    """识别任务前缀（第一优先级）。返回 task 名或 None。"""
    t = text.strip()
    if t.startswith("[REF]"):
        return REFERRING
    if t.startswith("[CAP]"):
        return CAPTION
    if t.startswith("[VQA]"):
        return VQA
    if t.startswith("[CD]"):
        return CHANGE_TASK
    return None


def _has_mcq(text: str):
    """识别选择题格式（MME/XLRS 用 [MCQ] 前缀或选项列表）。"""
    return "[MCQ]" in text or re.search(r"\n\(A\) .+\n\(B\) .+", text) is not None


def route_task(prompt: str) -> str:
    """根据 prompt 返回 task 名（标签识别 → 关键词识别 → 默认分支）。"""
    t = (prompt or "").strip()
    if not t:
        return VQA

    pref = _has_task_prefix(t)
    if pref:
        return pref

    low = t.lower()
    # The official XLRS Caption instruction contains words such as "after";
    # identify its explicit caption structure before change keyword matching.
    if any(marker in low for marker in CAPTION_INSTRUCTION_MARKERS):
        return CAPTION
    if _has_mcq(t):
        return MCQ
    # Grounding descriptions may mention visual "differences" or comparisons;
    # an explicit bbox instruction must win over incidental change words.
    if any(marker in low for marker in GROUNDING_INSTRUCTION_MARKERS):
        return REFERRING
    if any(kw in low for kw in CHANGE_KEYWORDS):
        return CHANGE_TASK
    if any(kw in low for kw in GROUNDING_KEYWORDS):
        return REFERRING
    if any(kw in low for kw in CAPTION_KEYWORDS):
        return CAPTION
    return VQA


def route(prompt: str) -> str:
    """根据 prompt 返回 expert 名（task -> expert 映射的兜底）。"""
    return TASK_TO_EXPERT.get(route_task(prompt), GENERAL)
