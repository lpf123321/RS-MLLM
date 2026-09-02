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
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rsmllm.config import DATA_ROOT

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
    os.chdir(REPO_ROOT)
    model_path = Path(args.model_path).expanduser()
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path
    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    data_override = Path(args.data_path).expanduser() if args.data_path else None
    if data_override is not None and not data_override.is_absolute():
        data_override = REPO_ROOT / data_override

    SHARED = DATA_ROOT
    data_path = str(data_override) if data_override is not None else f"{SHARED}/VRSBench/vrsbench_eval.jsonl"

    from evaluation.adapters.router import RouterAdapter
    adapter = RouterAdapter(
        str(model_path),
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)
    print(f"  Saved {len(records)} caption predictions to {output_path}", flush=True)


if __name__ == "__main__":
    main()
