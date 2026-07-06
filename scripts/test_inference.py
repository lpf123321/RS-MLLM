"""
Quick test: run VQA / CAP / REF samples through Qwen3-VL-2B-Instruct
to verify the model can understand the preprocessed data format.
"""

import json
import os
import sys
import torch
from transformers import Qwen3VLProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

MODEL_PATH = "models/Qwen3-VL-2B-Instruct"
TRAIN_JSONL = "output/vrsbench_train.jsonl"


def find_samples(jsonl_path: str):
    samples = {}
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            text = rec["messages"][0]["content"][1]["text"]
            for task in ["VQA", "CAP", "REF"]:
                prefix = f"[{task}]"
                if text.startswith(prefix) and task not in samples:
                    samples[task] = rec
                    break
            if len(samples) == 3:
                break
    return samples


def run_inference(messages, processor, model):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=256)
    input_len = inputs.input_ids.shape[1]
    output = processor.batch_decode(
        generated_ids[:, input_len:], skip_special_tokens=True
    )[0].strip()
    return output


def main():
    print("Loading model and processor...", flush=True)
    processor = Qwen3VLProcessor.from_pretrained(MODEL_PATH)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"Model device: {model.device}", flush=True)

    print("Finding samples...", flush=True)
    samples = find_samples(TRAIN_JSONL)
    print(f"Found: {list(samples.keys())}", flush=True)

    for task in ["VQA", "CAP", "REF"]:
        if task not in samples:
            print(f"\n[{task}] No sample found, skipping.", flush=True)
            continue

        rec = samples[task]
        messages = rec["messages"]
        answer = messages[1]["content"][0]["text"]
        img_path = messages[0]["content"][0]["image"]

        print(f"\n{'='*60}", flush=True)
        print(f"[{task}] image: {img_path}", flush=True)
        print(f"[{task}] instruction: {messages[0]['content'][1]['text']}", flush=True)
        print(f"[{task}] expected:  {answer}", flush=True)
        print("-" * 40, flush=True)
        output = run_inference(messages, processor, model)
        print(f"[{task}] predicted: {output}", flush=True)
        print(f"[{task}] match: {'YES' if output.strip().lower() == answer.strip().lower() else 'similarity check needed'}", flush=True)


if __name__ == "__main__":
    main()
