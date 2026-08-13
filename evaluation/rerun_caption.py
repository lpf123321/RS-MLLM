"""Re-run caption inference only, saving predictions for offline metric recompute.

Only regenerates the VRSBench caption split (9350 samples) so that the Router /
expert caption CIDEr can be recomputed with the fixed metric implementation.

Usage:
    python evaluation/rerun_caption.py \
        --adapter router \
        --model_path models/Qwen3.5-4B \
        --general_lora outputs/checkpoints/expert_general_v2 \
        --grounding_lora outputs/checkpoints/expert_grounding \
        --change_lora outputs/checkpoints/expert_change \
        --output outputs/eval_results/rerun_vrsbench_caption.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evalsets import vrsbench
from evaluation.router import rules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="router")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--general_lora", default=None)
    ap.add_argument("--grounding_lora", default=None)
    ap.add_argument("--change_lora", default=None)
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--eval_batch_size", type=int, default=4)
    ap.add_argument("--max_samples", type=int, default=0)
    args = ap.parse_args()

    SHARED = os.environ.get("DATA_ROOT", "/users/u2024311136/shared/shared_datasets")
    data_path = args.data_path or f"{SHARED}/VRSBench/vrsbench_eval.jsonl"

    from evaluation.adapters.router import RouterAdapter
    adapter = RouterAdapter(
        args.model_path,
        general_lora=args.general_lora,
        grounding_lora=args.grounding_lora,
        change_lora=args.change_lora,
        device="cuda",
        system_prompt="Obey the task prefix:\n- [VQA] Answer with a single word or short phrase only. No extra text.\n- [CAP] Describe the image in detail.\n- [REF] Output ONLY the bounding box in format {<x1><y1><x2><y2>} with integer coordinates 0-100, e.g. {<25><40><33><60>}. No other text.",
    )

    samples = vrsbench.load_data(data_path)
    captions = [s for s in samples if s["task"] == "caption"]
    if args.max_samples > 0:
        captions = captions[: args.max_samples]
    print(f"  Caption samples: {len(captions)}", flush=True)

    batch = [(s["images"], s["prompt"]) for s in captions]
    preds = adapter.batch_generate(batch, batch_size=args.eval_batch_size, max_new_tokens=256)

    records = [
        {"task": "caption", "reference": s["references"], "prediction": p}
        for s, p in zip(captions, preds)
    ]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)
    print(f"  Saved {len(records)} caption predictions to {args.output}", flush=True)


if __name__ == "__main__":
    main()
