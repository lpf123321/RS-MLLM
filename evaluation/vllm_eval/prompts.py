"""Task-specific prompts that make the expected answer format explicit."""

from __future__ import annotations

from collections.abc import Mapping


def _choice_lines(choices: Mapping[str, str]) -> str:
    return "\n".join(f"({label}) {text}" for label, text in choices.items())


def caption_prompt(question: str) -> str:
    return (
        f"{question.strip().rstrip('.')} in one factual paragraph. "
        "Return only the caption, using at most 100 words and at most 6 sentences. "
        "Mention only objects, attributes, counts, and spatial relations supported by the image. "
        "Do not repeat any sentence or phrase."
    )


def vqa_prompt(question: str) -> str:
    return (
        f"{question.strip()}\n"
        "Return only the concise answer. Do not add an explanation or repeat the question."
    )


def bbox_prompt(question: str) -> str:
    return (
        f"Locate the following target in the remote-sensing image: {question.strip()}\n"
        "Return exactly one bounding box as [x1, y1, x2, y2]. Coordinates are integers "
        "normalized to the 0-100 range, measured from the image's top-left corner."
    )


def choice_prompt(
    question: str,
    choices: Mapping[str, str],
    *,
    multiple: bool,
    temporal_pair: bool = False,
) -> str:
    prefix = ""
    if temporal_pair:
        prefix = (
            "Image 1 is the earlier/before view and Image 2 is the later/after view. "
            "Compare them in that order.\n"
        )
    if multiple:
        instruction = (
            "One or more options may be correct. Return every correct option letter exactly once, "
            "in alphabetical order, with no explanation (example: ACD)."
        )
    else:
        allowed = ", ".join(choices)
        instruction = (
            f"Your entire response must be exactly one of these option letters: {allowed}. "
            "Do not include the option text, an explanation, or any other characters."
        )
    return f"{prefix}{question.strip()}\n{_choice_lines(choices)}\n{instruction}"


def change_caption_prompt() -> str:
    return (
        "Image 1 is A/pre/before and Image 2 is B/post/after. Describe only semantic changes "
        "from Image 1 to Image 2 in one short sentence. Ignore illumination, color tone, sensor, "
        "compression, and slight registration differences. If no semantic change occurred, state that."
    )
