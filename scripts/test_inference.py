"""
Quick test: run LEVIR-CC samples through Qwen3-VL-2B-Instruct
to verify the model can handle dual-image change description.
"""

import json
import os
import sys
import torch
from transformers import Qwen3VLProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

MODEL_PATH = "models/Qwen3-VL-2B-Instruct"
TRAIN_JSONL = "output/levircc_train.jsonl"


def find_samples(jsonl_path: str):
    seen_images = set()
    no_change = None
    change = None

    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            caption = rec["messages"][1]["content"][0]["text"]
            img_a = rec["messages"][0]["content"][0]["image"]
            if img_a in seen_images:
                continue
            seen_images.add(img_a)

            is_no_change = any(kw in caption for kw in ["no difference", "no change", "same as before", "nothing has changed", "seem identical"])

            if is_no_change and no_change is None:
                no_change = (i, rec, is_no_change)
            elif not is_no_change and change is None:
                change = (i, rec, is_no_change)

            if no_change is not None and change is not None:
                break

    samples = [s for s in [no_change, change] if s is not None]
    return samples[:2]


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
    print(f"Found {len(samples)} samples", flush=True)

    # print one sample's full messages for debug
    print(json.dumps(samples[0][1]["messages"], indent=2, ensure_ascii=False)[:500], flush=True)

    for idx, rec, is_no_change in samples:
        messages = rec["messages"]
        answer = messages[1]["content"][0]["text"]
        content = messages[0]["content"]
        img_a = content[0]["image"]
        img_b = content[1]["image"]
        instruction = content[2]["text"]

        label = "NO CHANGE" if is_no_change else "CHANGE"

        print(f"\n{'='*60}", flush=True)
        print(f"[Sample {idx}] {label}", flush=True)
        print(f"  image A: {img_a}", flush=True)
        print(f"  image B: {img_b}", flush=True)
        print(f"  instruction: {instruction}", flush=True)
        print(f"  expected:  {answer}", flush=True)
        print("-" * 40, flush=True)
        output = run_inference(messages, processor, model)
        print(f"  predicted: {output}", flush=True)


if __name__ == "__main__":
    main()
