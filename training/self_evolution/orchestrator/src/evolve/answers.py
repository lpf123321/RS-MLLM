from __future__ import annotations

import re

REMOVE_HINT = "Only focus on the objects inside the red bounding box in the image to answer this question."


def clean_problem(problem: str) -> str:
    text = (problem or "").replace("<image>", "").replace(REMOVE_HINT, "")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    text = re.sub(r"Answer with the option's letter from the given choices\.?", "", text, flags=re.I)
    return text.strip()


def split_question_options(problem: str) -> tuple[str, list[str]]:
    text = clean_problem(problem)
    matches = list(re.finditer(r"(?m)^\s*([A-D])\.\s*(.+?)\s*$", text))
    if len(matches) != 4:
        raise ValueError(f"Expected four A-D choices, found {len(matches)}")
    stem = text[:matches[0].start()].strip()
    options = [match.group(2).strip() for match in matches]
    return stem, options


def build_prompt(question: str, options: list[str]) -> str:
    option_text = "\n".join(f"{letter}. {value}" for letter, value in zip("ABCD", options))
    return f"<image>\n{question}\n\n{option_text}\n\nRespond with only the correct option letter."


def normalize_answer(value: object) -> str | None:
    if isinstance(value, int) and 0 <= value <= 3:
        return "ABCD"[value]
    text = str(value or "").strip().upper()
    patterns = [r"^\s*([A-D])\s*$", r"(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*\(?([A-D])\)?"]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def normalize_teacher_answer(value: object) -> str | None:
    """Parse an unambiguous choice for filtering without relaxing evaluation."""
    strict = normalize_answer(value)
    if strict is not None:
        return strict
    text = str(value or "").strip().upper()
    match = re.match(r"^\s*([A-D])\s*[\.)\]:-]\s*\S", text)
    return match.group(1) if match else None


def evaluate_answer(prediction: object, reference: object, evaluator: str = "mcq_abcd") -> bool:
    """Stable evaluator seam; open-text CIDEr/BLEU adapters will plug in here later."""
    if evaluator != "mcq_abcd":
        raise NotImplementedError(f"Answer evaluator is not implemented yet: {evaluator}")
    normalized_prediction = normalize_answer(prediction)
    normalized_reference = normalize_answer(reference)
    return normalized_prediction is not None and normalized_prediction == normalized_reference
