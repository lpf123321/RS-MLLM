import json

from evaluation.metrics import BLEU, CIDEr, ROUGEL


NAME = "LEVIR-CC"

TASK_METRICS = {
    "caption": [BLEU(max_n=4), ROUGEL(), CIDEr()],
}


def _reference_to_text(value):
    """Extract one textual reference from the JSONL chat-format annotation."""
    if isinstance(value, str):
        return value.strip()

    if isinstance(value, dict):
        for key in ("raw", "text", "value"):
            text = value.get(key)
            if isinstance(text, str):
                return text.strip()

        if "content" in value:
            return _reference_to_text(value["content"])

    if isinstance(value, list):
        parts = [_reference_to_text(item) for item in value]
        parts = [part for part in parts if part]
        return " ".join(parts).strip()

    raise TypeError(
        f"Unsupported LEVIR-CC reference type: {type(value).__name__}"
    )


def load_data(data_path: str):
    samples = []

    with open(data_path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            user_content = record["messages"][0]["content"]
            prompt = user_content[-1]["text"]
            image_paths = [
                item["image"]
                for item in user_content
                if item["type"] == "image"
            ]

            raw_references = record.get("references")
            if raw_references is None:
                raw_references = record["messages"][1]["content"]

            if isinstance(raw_references, (str, dict)):
                raw_references = [raw_references]

            references = [
                _reference_to_text(reference)
                for reference in raw_references
            ]
            references = [reference for reference in references if reference]

            if not references:
                raise ValueError(
                    f"Empty LEVIR-CC references at line {line_number}"
                )

            samples.append({
                "task": "caption",
                "images": image_paths,
                "prompt": prompt,
                "references": references,
            })

    return samples
