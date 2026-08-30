#!/usr/bin/env python3
"""Smoke test: inspect the random sampling distribution used by the prune sweep.

Checks, for each of the 4 datasets:
  - total samples in the full jsonl
  - the random subset (N samples, seed) task distribution
  - the router expert distribution (which expert each prompt routes to)
  - image-path existence on a small prefix of the subset

No GPU/model required. Run with the rs_mllm conda env.
"""
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evalsets import levircc, mme, vrsbench, xlrs
from evaluation.router import rules

SHARED = os.environ.get("DATA_ROOT", "/users/u2024311136/shared/shared_datasets")

DATASETS = {
    "vrsbench": (vrsbench, f"{SHARED}/VRSBench/vrsbench_eval.jsonl"),
    "mme": (mme, f"{SHARED}/MME-RealWorld-RS/mme_rs.jsonl"),
    "xlrs": (xlrs, f"{SHARED}/XLRS-Bench-lite/xlrs.jsonl"),
    "levircc": (levircc, f"{SHARED}/LEVIR-CC/levircc_test.jsonl"),
}

N = int(os.environ.get("N_SAMPLES", "1000"))
SEED = int(os.environ.get("SAMPLE_SEED", "2026"))


def main():
    for name, (mod, path) in DATASETS.items():
        samples = mod.load_data(path)
        subset = random.Random(SEED).sample(samples, min(N, len(samples)))

        task_dist = Counter(s["task"] for s in subset)
        expert_dist = Counter(rules.route(s["prompt"]) for s in subset)

        missing = 0
        checked = 0
        for s in subset[:50]:
            for img in s["images"]:
                checked += 1
                if not os.path.exists(img):
                    missing += 1

        print(f"\n[{name}] total={len(samples)}  sampled={len(subset)}")
        print(f"  tasks          : {dict(task_dist)}")
        print(f"  routed experts : {dict(expert_dist)}")
        print(f"  image-path check (first 50 samples): {missing}/{checked} missing")


if __name__ == "__main__":
    main()
