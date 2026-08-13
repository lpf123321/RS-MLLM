import json

from evaluation.metrics import BLEU, CIDEr, ROUGEL

NAME = "LEVIR-CC"

TASK_METRICS = {"caption": [BLEU(max_n=4), ROUGEL(), CIDEr()]}


def load_data(data_path: str):
    samples = []
    with open(data_path) as f:
        for line in f:
            d = json.loads(line)
            user_content = d["messages"][0]["content"]
            text = user_content[-1]["text"]
            image_paths = [c["image"] for c in user_content if c["type"] == "image"]
            references = [r["raw"] if isinstance(r, dict) else r for r in d.get("references", [d["messages"][1]["content"][0]["text"]])]
            samples.append({
                "task": "caption",
                "images": image_paths,
                "prompt": text,
                "references": references,
            })
    return samples
