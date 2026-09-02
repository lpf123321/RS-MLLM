#!/usr/bin/env python3
"""One-sample-per-task smoke test for the Delta task Router."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rsmllm.config import DATA_ROOT, MODELS_ROOT as M_ROOT

from evaluation.adapters.delta_pruned import DeltaPrunedAdapter
from evaluation.evalsets import levircc, mme, vrsbench, xlrs, xlrs_grounding
from evaluation.router import rules
from evaluation.router.task_prune_config import CANDIDATE_TASK_PRUNE_CONFIG

SHARED = str(DATA_ROOT)
GENERAL = str(REPO_ROOT / "prune" / "output" / "new_experts" / "general_exp7_delta.pt")
GROUNDING = str(M_ROOT / "lora_expert" / "lora" / "grounding" / "delta_model.pt")
CHANGE = str(M_ROOT / "lora_expert" / "lora" / "change" / "delta_model.pt")
CAPTION = str(M_ROOT / "lora_expert" / "lora" / "caption" / "delta_model.pt")


def first(module, path, task=None):
    for sample in module.load_data(path):
        if task is None or sample["task"] == task:
            return sample
    raise RuntimeError(f"No sample found in {path}")


def main():
    samples = [
        first(vrsbench, f"{SHARED}/VRSBench/vrsbench_eval.jsonl", "vqa"),
        first(vrsbench, f"{SHARED}/VRSBench/vrsbench_eval.jsonl", "caption"),
        first(vrsbench, f"{SHARED}/VRSBench/vrsbench_eval.jsonl", "referring"),
        first(mme, f"{SHARED}/MME-RealWorld-RS/mme_rs.jsonl"),
        first(xlrs, f"{SHARED}/XLRS-Bench-lite/xlrs.jsonl"),
        first(levircc, f"{SHARED}/LEVIR-CC/levircc_test.jsonl"),
        first(xlrs_grounding, str(REPO_ROOT / "evaluation" / "data" / "xlrs_grounding.jsonl")),
    ]
    with (REPO_ROOT / "evaluation" / "prompts" / "xlrs_caption_en.txt").open(encoding="utf-8") as f:
        caption_instruction = f.read().strip()
    samples[1]["prompt"] = caption_instruction

    adapter = DeltaPrunedAdapter(
        str(M_ROOT / "lora_expert" / "base_model"),
        general_lora=GENERAL, grounding_lora=GROUNDING,
        change_lora=CHANGE, caption_lora=CAPTION,
        prune_method="l2norm", keep_ratio=1.0,
        task_prune_config=CANDIDATE_TASK_PRUNE_CONFIG,
        force_think=True, max_new_tokens=256,
    )

    for sample in samples:
        task = rules.route_task(sample["prompt"])
        expert = rules.TASK_TO_EXPERT[task]
        keep = CANDIDATE_TASK_PRUNE_CONFIG[task]["keep_ratio"]
        adapter.system_prompt = "" if task == "caption" else ""
        max_tokens = 550 if task == "caption" else 64 if task in ("vqa", "mcq") else 64
        pred = adapter.batch_generate([(sample["images"], sample["prompt"])],
                                       batch_size=1, max_new_tokens=max_tokens)[0]
        print(json.dumps({
            "task": task, "expert": expert, "method": CANDIDATE_TASK_PRUNE_CONFIG[task]["method"],
            "keep_ratio": keep, "prediction": pred[:300],
        }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
