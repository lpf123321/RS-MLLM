"""Shared system prompts used by OPD data construction and trajectory replay."""

SYSTEM_PROMPTS = {
    "vrsbench": (
        "Obey the task prefix:\n"
        "- [VQA] Answer with a single word or short phrase only. No extra text.\n"
        "- [CAP] Describe the image in detail.\n"
        "- [REF] Output ONLY the bounding box in format {<x1><y1><x2><y2>} "
        "with integer coordinates 0-99. No other text."
    ),
    "mme": (
        'Answer EXACTLY in format "X. (X) FullOptionText" with the letter repeated '
        "in parentheses. Output ONLY that line."
    ),
    "xlrs": (
        'Answer EXACTLY in format "X. (X) FullOptionText" with the letter repeated '
        "in parentheses. Output ONLY that line."
    ),
    "levircc": "Describe the changes between the two images concisely in 1-2 sentences.",
}
