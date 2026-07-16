import json

from evaluation.metrics import Accuracy, BLEU, CIDEr, ReferringAcc, ROUGEL

NAME = "VRSBench"

TASK_METRICS = {
    "vqa": [Accuracy()],
    "caption": [BLEU(max_n=4), ROUGEL(), CIDEr()],
    "referring": [ReferringAcc(thresholds=[0.25, 0.5, 0.7])],
}

_PREFIX_MAP = {"[VQA]": "vqa", "[CAP]": "caption", "[REF]": "referring"}


def load_data(data_path: str):
    samples = []
    with open(data_path) as f:
        for line in f:
            d = json.loads(line)
            user_content = d["messages"][0]["content"]
            text = user_content[-1]["text"]
            answer = d["messages"][1]["content"][0]["text"]
            image_paths = [c["image"] for c in user_content if c["type"] == "image"]
            for prefix, task in _PREFIX_MAP.items():
                if text.startswith(prefix):
                    samples.append({
                        "task": task,
                        "images": image_paths,
                        "prompt": text,
                        "references": [answer],
                    })
                    break
    return samples
