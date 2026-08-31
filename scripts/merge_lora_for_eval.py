#!/usr/bin/env python3
"""Merge LoRA checkpoint into base model and save, then run evaluation."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../finetune_framework/Qwen-VL-Series-Finetune/src"))

from utils import load_pretrained_model, get_model_name_from_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", required=True)
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--save-path", required=True)
    args = parser.parse_args()

    model_name = get_model_name_from_path(args.lora_path)
    processor, model = load_pretrained_model(
        model_path=args.lora_path,
        model_base=args.model_base,
        model_name=model_name,
        device_map="auto",
    )

    os.makedirs(args.save_path, exist_ok=True)
    model.save_pretrained(args.save_path, safe_serialization=True)
    processor.save_pretrained(args.save_path)
    print(f"Merged model saved to {args.save_path}")


if __name__ == "__main__":
    main()
