#!/usr/bin/env python3
"""Evaluate the task-adaptive Delta pruning Router.

The task-to-expert rules intentionally follow the shared lora_expert router:
VQA/MCQ -> general, referring -> grounding, change -> change, caption -> caption.
The General expert can be replaced with the merged exp7 Delta.
"""

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rsmllm.config import MODELS_ROOT as M_ROOT
from rsmllm.config import DATA_ROOT

from evaluation.adapters.delta_pruned import DeltaPrunedAdapter
from evaluation.evalsets import levircc, mme, vrsbench, xlrs, xlrs_caption, xlrs_grounding
from evaluation.main import SYSTEM_PROMPTS, evaluate
from evaluation.router.task_prune_config import CANDIDATE_TASK_PRUNE_CONFIG

SHARED = DATA_ROOT

DATASETS = {
    "vrsbench": (vrsbench, f"{SHARED}/VRSBench/vrsbench_eval.jsonl"),
    "mme": (mme, f"{SHARED}/MME-RealWorld-RS/mme_rs.jsonl"),
    "xlrs": (xlrs, f"{SHARED}/XLRS-Bench-lite/xlrs.jsonl"),
    "levircc": (levircc, f"{SHARED}/LEVIR-CC/levircc_test.jsonl"),
    "xlrs_caption": (xlrs_caption, str(REPO_ROOT / "evaluation" / "data" / "xlrs_caption.jsonl")),
    "xlrs_grounding": (xlrs_grounding, str(REPO_ROOT / "evaluation" / "data" / "xlrs_grounding.jsonl")),
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
        with (REPO_ROOT / "evaluation" / "prompts" / "xlrs_caption_en.txt").open(encoding="utf-8") as f:
            instruction = f.read().strip()
        for sample in samples:
            sample["prompt"] = instruction
        return samples


DATASETS["xlrs_caption"] = (
    XLRSInstructionCaption,
    str(M_ROOT / "lora_expert" / "evaluation" / "split_evals" / "xlrs_caption_en.jsonl"),
)


def main():
    parser = argparse.ArgumentParser(description="Delta task-adaptive pruning Router")
    parser.add_argument("--model_path", default=str(M_ROOT / "lora_expert" / "base_model"))
    parser.add_argument("--general_lora", default=str(REPO_ROOT / "prune" / "output" / "new_experts" / "general_exp7_delta.pt"))
    parser.add_argument("--grounding_lora", default=str(M_ROOT / "lora_expert" / "lora" / "grounding" / "delta_model.pt"))
    parser.add_argument("--change_lora", default=str(M_ROOT / "lora_expert" / "lora" / "change" / "delta_model.pt"))
    parser.add_argument("--caption_lora", default=str(M_ROOT / "lora_expert" / "lora" / "caption" / "delta_model.pt"))
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--random_samples", type=int, default=1000)
    parser.add_argument("--sample_seed", type=int, default=2026)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument(
        "--output_dir",
        default=str(REPO_ROOT / "prune" / "output" / "delta_exp7_task_router"),
        help="输出目录(相对路径按仓库根解析)",
    )
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir = output_dir.resolve()
    os.makedirs(output_dir, exist_ok=True)

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
        with (output_dir / f"{dataset}.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    with (output_dir / "router_config.json").open("w", encoding="utf-8") as f:
        json.dump(all_results["_config"], f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
