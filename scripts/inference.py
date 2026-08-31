#!/usr/bin/env python3
"""
RS-MLLM 推理脚本 —— 支持 LoRA + 剪枝 + 四数据集评估。

用法:
  python scripts/inference.py --model_path /path/to/model --datasets all
  python scripts/inference.py --model_path /path/to/model --lora_path /path/to/lora --prune_method l2 --prune_r 0.5
  python scripts/inference.py --model_path /path/to/model --image img.png --text "描述这张图" --dataset vrsbench
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

# Ensure repo root in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

try:
    from evaluation.main import SYSTEM_PROMPTS
except ImportError:
    SYSTEM_PROMPTS = {
        "vrsbench": "Obey the task prefix:\n- [VQA] Answer with a single word or short phrase only. No extra text.\n- [CAP] Describe the image in detail.\n- [REF] Output ONLY the bounding box in format {<x1><y1><x2><y2>} with integer coordinates 0-99, e.g. {<25><40><33><60>}. No other text.",
        "mme": 'Answer EXACTLY in format "X. (X) FullOptionText" with the letter repeated in parentheses. Example: "D. (D) White". You MUST include the parenthesized letter - never omit it. Output ONLY that line.',
        "xlrs": 'Answer EXACTLY in format "X. (X) FullOptionText" with the letter repeated in parentheses. Example: "A. (A) Some description". You MUST include the parenthesized letter - never omit it. Output ONLY that line.',
        "levircc": "Describe the changes between the two images concisely in 1-2 sentences.",
    }

SHARED = os.environ.get("DATA_ROOT", str(Path(__file__).resolve().parent.parent / "datasets"))
_BBOX_ANGLE_RE = re.compile(r"\{<\s*(\d+(?:\.\d+)?)\s*><\s*(\d+(?:\.\d+)?)\s*><\s*(\d+(?:\.\d+)?)\s*><\s*(\d+(?:\.\d+)?)\s*>\}")
_BBOX_COMMA_RE = re.compile(
    r"\{\s*(\d+(?:\.\d+)?)\s*[,;\s]+\s*(\d+(?:\.\d+)?)\s*[,;\s]+\s*(\d+(?:\.\d+)?)\s*[,;\s]+\s*(\d+(?:\.\d+)?)\s*\}"
)
TASK_MAX_TOKENS = {"vqa": 64, "referring": 32, "caption": 256}


class ModelEngine:
    def __init__(self, model_path: str, lora_path: Optional[str] = None,
                 device: str = "cuda", prune_method: Optional[str] = None,
                 prune_r: float = 0.5):
        self.device = device if device == "cuda" and torch.cuda.is_available() else "cpu"

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.processor.tokenizer.padding_side = "left"

        # 剪枝 monkey-patch（在 from_pretrained 前）
        if prune_method and prune_method != "none":
            from scripts.prune import apply_pruning, enable_pruning
            apply_pruning(model_path=model_path)

        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to(self.device)
        if lora_path:
            self.model = PeftModel.from_pretrained(self.model, lora_path).merge_and_unload().to(self.device)
        self.model.eval()

        if prune_method and prune_method != "none":
            enable_pruning(self.model, method=prune_method, r=prune_r)

    @staticmethod
    def extract_answer(text: str) -> str:
        if "</think>" in text:
            return text.split("</think>", 1)[1].strip()
        return text.strip()

    def build_prompt(self, images: List[str], user_text: str, system_prompt: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content":
            [{"type": "image", "image": img} for img in images]
            + [{"type": "text", "text": user_text}],
        })
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        text += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return text

    @torch.no_grad()
    def batch_generate(self, batch: List[Tuple[List[str], str, str]],
                       max_new_tokens: int = 256, batch_size: int = 32) -> List[str]:
        results = []
        for i in tqdm(range(0, len(batch), batch_size), desc="Inference", unit="batch"):
            sub = batch[i:i + batch_size]
            texts, all_images = [], []
            for images, user_text, system_prompt in sub:
                prompt = self.build_prompt(images, user_text, system_prompt)
                msg = [{"role": "user", "content": [{"type": "image", "image": img} for img in images]}]
                image_inputs, _ = process_vision_info(msg)
                texts.append(prompt)
                all_images.append(image_inputs)
            inputs = self.processor(
                text=texts, images=all_images, padding=True, return_tensors="pt",
            ).to(self.model.device)
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=None, top_p=None,
            )
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
            outputs = self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )
            results.extend(self.extract_answer(t) for t in outputs)
        return results


# ---- Benchmark 定义 ----

class VRSBench:
    name = "vrsbench"

    @staticmethod
    def load(data_path, max_samples=0):
        prefix_map = {"[VQA]": "vqa", "[CAP]": "caption", "[REF]": "referring"}
        samples = []
        with open(data_path) as f:
            for line in f:
                if max_samples and len(samples) >= max_samples: break
                d = json.loads(line)
                uc = d["messages"][0]["content"]
                text = next(c["text"] for c in uc if c["type"] == "text")
                images = [c["image"] for c in uc if c["type"] == "image"]
                task = "vqa"
                for pfx, t in prefix_map.items():
                    if text.startswith(pfx): task = t; break
                samples.append({
                    "task": task, "images": images, "prompt": text,
                    "references": [d["messages"][1]["content"][0]["text"]],
                })
        return samples

    @staticmethod
    def group_by_task(samples):
        groups = {}
        for s in samples:
            groups.setdefault(s["task"], []).append(s)
        return groups

    @staticmethod
    def normalize(prediction, sample):
        if sample["task"] != "referring":
            return prediction
        pred = _BBOX_COMMA_RE.sub(r"{<\1><\2><\3><\4>}", prediction)
        m = _BBOX_ANGLE_RE.search(pred)
        if not m: return prediction
        coords = list(map(float, m.groups()))
        if max(coords) <= 99: return pred
        try:
            with Image.open(sample["images"][0]) as img: w, h = img.size
            sx, sy = 99.0 / max(w, 1), 99.0 / max(h, 1)
            norm = [int(max(0, min(99, round(x * sx if j % 2 == 0 else x * sy)))) for j, x in enumerate(coords)]
            return "{{<{}><{}><{}><{}>}}".format(*norm)
        except Exception:
            return pred


class LEVIRCC:
    name = "levircc"

    @staticmethod
    def load(data_path, max_samples=0):
        samples = []
        with open(data_path) as f:
            for line in f:
                if max_samples and len(samples) >= max_samples: break
                d = json.loads(line)
                uc = d["messages"][0]["content"]
                text = next(c["text"] for c in uc if c["type"] == "text")
                images = [c["image"] for c in uc if c["type"] == "image"]
                refs = d.get("references", [d["messages"][1]["content"][0]["text"]])
                samples.append({
                    "task": "caption", "images": images, "prompt": text, "references": refs,
                })
        return samples


class MME:
    name = "mme"

    @staticmethod
    def load(data_path, max_samples=0):
        samples = []
        with open(data_path) as f:
            for line in f:
                if max_samples and len(samples) >= max_samples: break
                d = json.loads(line)
                uc = d["messages"][0]["content"]
                text = next(c["text"] for c in uc if c["type"] == "text")
                images = [c["image"] for c in uc if c["type"] == "image"]
                samples.append({
                    "task": "vqa", "images": images, "prompt": text,
                    "references": [d["messages"][1]["content"][0]["text"]],
                })
        return samples


class XLRS:
    name = "xlrs"

    @staticmethod
    def load(data_path, max_samples=0):
        samples = []
        with open(data_path) as f:
            for line in f:
                if max_samples and len(samples) >= max_samples: break
                d = json.loads(line)
                uc = d["messages"][0]["content"]
                text = next(c["text"] for c in uc if c["type"] == "text")
                images = [c["image"] for c in uc if c["type"] == "image"]
                samples.append({
                    "task": "vqa", "images": images, "prompt": text,
                    "references": [d["messages"][1]["content"][0]["text"]],
                })
        return samples


BENCHMARKS = {
    "vrsbench": (VRSBench(), SYSTEM_PROMPTS.get("vrsbench", ""), f"{SHARED}/VRSBench/vrsbench_eval.jsonl"),
    "levircc": (LEVIRCC(), SYSTEM_PROMPTS.get("levircc", ""), f"{SHARED}/LEVIR-CC/levircc_test.jsonl"),
    "mme": (MME(), SYSTEM_PROMPTS.get("mme", ""), f"{SHARED}/MME-RealWorld-RS/mme_rs.jsonl"),
    "xlrs": (XLRS(), SYSTEM_PROMPTS.get("xlrs", ""), f"{SHARED}/XLRS-Bench-lite/xlrs.jsonl"),
}


def main():
    parser = argparse.ArgumentParser(description="RS-MLLM Inference")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--prune_method", default=None, choices=["l2", "k2", "divprune", "scope", "none"])
    parser.add_argument("--prune_r", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--start_offset", type=int, default=0)
    parser.add_argument("--output_dir", default="outputs/inference_results")
    parser.add_argument("--datasets", nargs="+", choices=list(BENCHMARKS.keys()) + ["all"], default=["all"])
    parser.add_argument("--image", nargs="*", help="Image path(s) for single inference")
    parser.add_argument("--text", help="Text prompt for single inference")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    engine = ModelEngine(args.model_path, lora_path=args.lora_path, device=args.device,
                         prune_method=args.prune_method, prune_r=args.prune_r)

    if args.image and args.text:
        bench, sp, _ = BENCHMARKS["vrsbench"]
        pred = engine.batch_generate([(args.image, args.text, sp)], max_new_tokens=256, batch_size=1)
        print(json.dumps({"prediction": pred[0]}, indent=2))
        return

    ds_names = list(BENCHMARKS.keys()) if "all" in args.datasets else args.datasets
    for ds_name in ds_names:
        bench, sp, data_path = BENCHMARKS[ds_name]
        if not os.path.exists(data_path):
            print(f"[WARN] {data_path} not found, skip {ds_name}")
            continue

        print(f"\n{'='*60}")
        print(f"  {ds_name} ({bench.name})")
        print(f"{'='*60}")

        samples = bench.load(data_path, max_samples=0)
        if args.start_offset > 0:
            samples = samples[args.start_offset:]
        if args.max_samples > 0:
            samples = samples[:args.max_samples]

        # Sort samples by image size to keep uniform grid_thw within each batch
        # (required by SCOPE pruning: mixed image sizes in one batch cause garbage)
        from PIL import Image
        try:
            samples.sort(key=lambda s: Image.open(s["images"][0]).size if s["images"] else (0, 0))
            print(f"  Sorted {len(samples)} samples by image size for uniform batching")
        except Exception:
            pass
        groups = bench.group_by_task(samples) if hasattr(bench, "group_by_task") else {"default": samples}

        predictions = [None] * len(samples)
        offset = 0
        for task_key, group in groups.items():
            tok = TASK_MAX_TOKENS.get(task_key if task_key != "default" else group[0].get("task", "vqa"), 256)
            batch = [(s["images"], s["prompt"], sp) for s in group]
            start = time.time()
            preds = engine.batch_generate(batch, max_new_tokens=tok, batch_size=args.batch_size)
            elapsed = time.time() - start
            print(f"  [{task_key}] {len(preds)} samples in {elapsed:.1f}s ({len(preds)/elapsed:.1f} samp/s)")
            for i, pred in enumerate(preds):
                predictions[offset + i] = bench.normalize(pred, group[i]) if hasattr(bench, "normalize") else pred
            offset += len(group)

        out_path = os.path.join(args.output_dir, f"predictions_{ds_name}.json")
        out = [{"task": s["task"], "images": s["images"], "prompt": s["prompt"],
                "prediction": p, "references": s["references"]}
               for s, p in zip(samples, predictions)]
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
