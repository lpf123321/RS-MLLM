#!/usr/bin/env python3
"""Smoke test DeltaFourierAdapter on one sample per VRSBench task."""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rsmllm.config import MODELS_ROOT as M_ROOT
from rsmllm.config import DATA_ROOT

from evaluation.adapters.delta_fourier import DeltaFourierAdapter
from evaluation.evalsets import vrsbench
from evaluation.main import SYSTEM_PROMPTS

SHARED = DATA_ROOT


def main():
    samples = vrsbench.load_data(f"{SHARED}/VRSBench/vrsbench_eval.jsonl")
    selected = {}
    for sample in samples:
        selected.setdefault(sample["task"], sample)

    adapter = DeltaFourierAdapter(
        str(M_ROOT / "lora_expert" / "base_model"),
        general_lora=str(M_ROOT / "lora_expert" / "lora" / "general" / "delta_model.pt"),
        grounding_lora=str(M_ROOT / "lora_expert" / "lora" / "grounding" / "delta_model.pt"),
        change_lora=str(M_ROOT / "lora_expert" / "lora" / "change" / "delta_model.pt"),
        keep_ratio=0.5,
    )
    adapter.system_prompt = SYSTEM_PROMPTS["vrsbench"]
    for sample in selected.values():
        prediction = adapter.generate(sample["images"], sample["prompt"])
        print(f"task={sample['task']} prompt={sample['prompt'][:60]!r}")
        print(f"prediction={prediction[:160]!r}")
    print("DELTA FOURIER SMOKE TEST DONE")


if __name__ == "__main__":
    main()
