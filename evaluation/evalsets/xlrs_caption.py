import json

from evaluation.metrics import BLEU, CIDEr, ROUGEL

NAME = "XLRS-Bench caption"

TASK_METRICS = {
    "caption": [BLEU(max_n=4), ROUGEL(), CIDEr()],
}


def load_data(data_path: str):
    samples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            user_content = d["messages"][0]["content"]
            text = user_content[-1]["text"]
            answer = d["messages"][1]["content"][0]["text"]
            image_paths = [c["image"] for c in user_content if c["type"] == "image"]
            samples.append({
                "task": "caption",
                "images": image_paths,
                "prompt": text,
                "references": [answer],
            })
    return samples
