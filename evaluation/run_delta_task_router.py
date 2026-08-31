#!/usr/bin/env python3
"""Evaluate the task-adaptive Delta pruning Router.

The task-to-expert rules intentionally follow the shared lora_expert router:
VQA/MCQ -> general, referring -> grounding, change -> change, caption -> caption.
The General expert can be replaced with the merged exp7 Delta.
"""

import argparse
import json
from rsmllm.config import MODELS_ROOT as M_ROOT
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.adapters.delta_pruned import DeltaPrunedAdapter
from evaluation.evalsets import levircc, mme, vrsbench, xlrs, xlrs_caption, xlrs_grounding
from evaluation.main import SYSTEM_PROMPTS, evaluate
from evaluation.router.task_prune_config import CANDIDATE_TASK_PRUNE_CONFIG

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED = os.environ.get("DATA_ROOT", str(Path(__file__).resolve().parent.parent / "datasets"))

DATASETS = {
    "vrsbench": (vrsbench, f"{SHARED}/VRSBench/vrsbench_eval.jsonl"),
    "mme": (mme, f"{SHARED}/MME-RealWorld-RS/mme_rs.jsonl"),
    "xlrs": (xlrs, f"{SHARED}/XLRS-Bench-lite/xlrs.jsonl"),
    "levircc": (levircc, f"{SHARED}/LEVIR-CC/levircc_test.jsonl"),
    "xlrs_caption": (xlrs_caption, os.path.join(ROOT, "evaluation/data/xlrs_caption.jsonl")),
    "xlrs_grounding": (xlrs_grounding, os.path.join(ROOT, "evaluation/data/xlrs_grounding.jsonl")),
}


class XLRSInstructionCaption:
    NAME = xlrs_caption.NAME
    TASK_METRICS = xlrs_caption.TASK_METRICS

    @staticmethod
    def load_data(path):
        samples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                samples.append({
                    "task": "caption",
                    "images": [item["image"]],
                    "prompt": item.get("prompt", "Describe the image in detail."),
                    "references": list(item.get("references", [])),
                })
        with open(os.path.join(ROOT, "evaluation/prompts/xlrs_caption_en.txt"), encoding="utf-8") as f:
            instruction = f.read().strip()
        for sample in samples:
            sample["prompt"] = instruction
        return samples


DATASETS["xlrs_caption"] = (
    XLRSInstructionCaption,
    "M_ROOT/lora_expert/evaluation/split_evals/xlrs_caption_en.jsonl",
)


def main():
    parser = argparse.ArgumentParser(description="Delta task-adaptive pruning Router")
    parser.add_argument("--model_path", default="M_ROOT/lora_expert/base_model")
    parser.add_argument("--general_lora", default="prune/output/new_experts/general_exp7_delta.pt")
    parser.add_argument("--grounding_lora", default="M_ROOT/lora_expert/lora/grounding/delta_model.pt")
    parser.add_argument("--change_lora", default="M_ROOT/lora_expert/lora/change/delta_model.pt")
    parser.add_argument("--caption_lora", default="M_ROOT/lora_expert/lora/caption/delta_model.pt")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--random_samples", type=int, default=1000)
    parser.add_argument("--sample_seed", type=int, default=2026)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--output_dir", default="prune/output/delta_exp7_task_router")
    args = parser.parse_args()

    adapter = DeltaPrunedAdapter(
        args.model_path,
        general_lora=args.general_lora,
        grounding_lora=args.grounding_lora,
        change_lora=args.change_lora,
        caption_lora=args.caption_lora,
        prune_method="l2norm",
        keep_ratio=1.0,
        task_prune_config=CANDIDATE_TASK_PRUNE_CONFIG,
        pruner_seed=args.sample_seed,
        max_new_tokens=256,
        force_think=True,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {"_config": CANDIDATE_TASK_PRUNE_CONFIG}
    for dataset in args.datasets:
        module, data_path = DATASETS[dataset]
        if not os.path.exists(data_path):
            raise FileNotFoundError(data_path)
        adapter.system_prompt = SYSTEM_PROMPTS.get(dataset, "")
        print(f"[RUN] router dataset={dataset}", flush=True)
        results, _ = evaluate(
            adapter, module, data_path, 0, args.eval_batch_size,
            random_samples=args.random_samples, sample_seed=args.sample_seed,
            task_max_tokens={"caption": 550, "referring": 64},
        )
        all_results[dataset] = results
        with open(os.path.join(args.output_dir, f"{dataset}.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.output_dir, "router_config.json"), "w", encoding="utf-8") as f:
        json.dump(all_results["_config"], f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
