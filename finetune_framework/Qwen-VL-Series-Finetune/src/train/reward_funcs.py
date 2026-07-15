import re
from typing import List, Optional, Tuple

try:
    from math_verify import LatexExtractionConfig, parse, verify
    from latex2sympy2_extended import NormalizationConfig
except ImportError:
    LatexExtractionConfig = None
    NormalizationConfig = None
    parse = None
    verify = None


# ---------------------------------------------------------------------------
#  Bounding-box helpers (shared across referring reward functions)
# ---------------------------------------------------------------------------

_BBOX_PATTERN = re.compile(
    r"\{<\s*(-?\d+(?:\.\d+)?)\s*><\s*(-?\d+(?:\.\d+)?)\s*><\s*(-?\d+(?:\.\d+)?)\s*><\s*(-?\d+(?:\.\d+)?)\s*>\}"
)


def _parse_bbox(text: str) -> Optional[Tuple[float, float, float, float]]:
    """Parse a {<x1><y1><x2><y2>} coordinate string.

    Returns ``(x1, y1, x2, y2)`` or ``None``.
    """
    m = _BBOX_PATTERN.search(text)
    if m is None:
        return None
    return tuple(float(v) for v in m.groups())  # type: ignore[return-value]


def _compute_iou(box_a: Tuple[float, ...], box_b: Tuple[float, ...]) -> float:
    """Intersection-over-Union for two bboxes in XYXY format.

    Coordinates are expected in [0, 100] normalized space.  The function
    handles unordered x1/x2 or y1/y2 gracefully.
    """
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b

    # ensure x1 < x2, y1 < y2
    x1_a, x2_a = min(x1_a, x2_a), max(x1_a, x2_a)
    y1_a, y2_a = min(y1_a, y2_a), max(y1_a, y2_a)
    x1_b, x2_b = min(x1_b, x2_b), max(x1_b, x2_b)
    y1_b, y2_b = min(y1_b, y2_b), max(y1_b, y2_b)

    inter_x1 = max(x1_a, x1_b)
    inter_y1 = max(y1_a, y1_b)
    inter_x2 = min(x2_a, x2_b)
    inter_y2 = min(y2_a, y2_b)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def _extract_text(completion) -> str:
    """Normalise a single completion to plain text.

    Completions may be:
    * plain ``str``
    * ``[{"role": "assistant", "content": "…"}]`` (conversational)
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and len(completion) > 0:
        return completion[0].get("content", "")
    return str(completion)


# ---------------------------------------------------------------------------
#  Reward functions  (every public function whose name ends with ``_reward``
#  is auto-discovered by ``load_reward_funcs``.)
# ---------------------------------------------------------------------------

def accuracy_reward(completions, assistant, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    rewards = []

    for completion, sol in zip(completions, assistant):
        if parse is None or verify is None or LatexExtractionConfig is None or NormalizationConfig is None:
            rewards.append(float(completion.strip().lower() == sol.strip().lower()))
            continue

        try:
            gold_parsed = parse(sol, extraction_mode="first_match")
        except Exception as e:
            gold_parsed = []

        if len(gold_parsed) != 0:
            # Try parsing predicted answer too
            try:
                answer_parsed = parse(
                    completion,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                boxed="all",
                                units=True,
                            ),
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {completion}, gold: {sol}")
                reward = None
        else:
            # fallback to text match
            reward = float(completion.strip().lower() == sol.strip().lower())

        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
    rewards = [1.0 if match else 0.0 for match in matches]
    return rewards


def iou_reward(completions, assistant, **kwargs) -> List[float]:
    """Reward for referring / bounding-box tasks.

    Parses ``{<x1><y1><x2><y2>}`` from both the model output and the ground
    truth, then returns the IoU ∈ [0, 1] as the reward.  If parsing fails for
    a sample the reward is 0.0.
    """
    rewards: List[float] = []
    for comp, gt_text in zip(completions, assistant):
        pred_text = _extract_text(comp)

        pred_box = _parse_bbox(pred_text)
        gt_box = _parse_bbox(gt_text)

        if pred_box is None or gt_box is None:
            rewards.append(0.0)
            continue

        iou = _compute_iou(pred_box, gt_box)
        rewards.append(iou)

    return rewards


def iou_format_reward(completions, assistant, **kwargs) -> List[float]:
    """Binary reward: 1.0 if the completion contains a well-formed bbox, else 0.0.

    This can be used together with ``iou_reward`` (set appropriate
    ``--reward_weights`` in the GRPO config) to encourage the model to
    stay in the coordinate format while also optimising IoU.
    """
    rewards: List[float] = []
    for comp in completions:
        pred_text = _extract_text(comp)
        rewards.append(1.0 if _parse_bbox(pred_text) is not None else 0.0)
    return rewards
