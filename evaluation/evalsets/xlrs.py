import json

from evaluation.metrics import Accuracy

NAME = "XLRS-Bench-lite"

TASK_METRICS = {"vqa": [Accuracy()]}


def load_data(data_path: str):
    samples = []
    with open(data_path) as f:
        for line in f:
            d = json.loads(line)
            user_content = d["messages"][0]["content"]
            text = user_content[-1]["text"]
            answer = d["messages"][1]["content"][0]["text"]
            image_paths = [c["image"] for c in user_content if c["type"] == "image"]
            samples.append({
                "task": "vqa",
                "images": image_paths,
                "prompt": text,
                "references": [answer],
            })
    return samples
