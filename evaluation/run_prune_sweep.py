#!/usr/bin/env python3
"""Run the L2 / DivPrune keep-ratio sweep on the multi-expert router model.

Loads the model once (base + 3 expert LoRAs) and loops over methods x keep-ratio
x datasets, reusing the existing ``evaluate`` from ``evaluation.main``.  Each
(method, ratio, dataset) result is written to a separate JSON file so the run
can be resumed; ``summary.json`` aggregates everything for plotting.

Example:
    python -m evaluation.run_prune_sweep \
        --model_path M_ROOT/lora_expert/base_model \
        --general_lora .../lora/general --grounding_lora .../lora/grounding \
        --change_lora .../lora/change \
        --methods l2 divprune \
        --ratios 0.1 0.25 0.35 0.5 0.65 0.75 0.9 1.0 \
        --datasets vrsbench mme xlrs levircc \
        --random_samples 1000 --sample_seed 2026 --eval_batch_size 32
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

from evaluation.evalsets import levircc, mme, vrsbench, xlrs, xlrs_caption, xlrs_grounding
from evaluation.main import SYSTEM_PROMPTS, evaluate
from evaluation.adapters.router_pruned import RouterPrunedAdapter

SHARED = DATA_ROOT
_REPO_ROOT = str(REPO_ROOT)

DATASETS = {
    "vrsbench": (vrsbench, f"{SHARED}/VRSBench/vrsbench_eval.jsonl"),
    "mme": (mme, f"{SHARED}/MME-RealWorld-RS/mme_rs.jsonl"),
    "xlrs": (xlrs, f"{SHARED}/XLRS-Bench-lite/xlrs.jsonl"),
    "levircc": (levircc, f"{SHARED}/LEVIR-CC/levircc_test.jsonl"),
    "xlrs_caption": (xlrs_caption, os.path.join(_REPO_ROOT, "evaluation/data/xlrs_caption.jsonl")),
    "xlrs_grounding": (xlrs_grounding, os.path.join(_REPO_ROOT, "evaluation/data/xlrs_grounding.jsonl")),
}

DEFAULT_RATIOS = [0.1, 0.25, 0.35, 0.5, 0.65, 0.75, 0.9, 1.0]
DEFAULT_METHODS = ["l2", "divprune"]
DEFAULT_DATASETS = ["vrsbench", "mme", "xlrs", "levircc"]


def ratio_label(r: float) -> str:
    return f"r{int(round(r * 100)):02d}"


def main():
    parser = argparse.ArgumentParser(description="L2/DivPrune keep-ratio sweep (multi-expert)")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--general_lora", required=True)
    parser.add_argument("--grounding_lora", required=True)
    parser.add_argument("--change_lora", required=True)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--ratios", nargs="+", type=float, default=DEFAULT_RATIOS)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--random_samples", type=int, default=1000)
    parser.add_argument("--sample_seed", type=int, default=2026)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--output_dir",
        default=str(REPO_ROOT / "prune" / "output" / "prune_sweep"),
        help="输出目录(相对路径按仓库根解析)",
    )
    args = parser.parse_args()
    os.chdir(REPO_ROOT)

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir = output_dir.resolve()
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading multi-expert model with pruning: methods={args.methods}", flush=True)
    adapter = RouterPrunedAdapter(
        args.model_path,
        general_lora=args.general_lora,
        grounding_lora=args.grounding_lora,
        change_lora=args.change_lora,
        max_new_tokens=args.max_new_tokens,
        prune_method=args.methods[0],
        keep_ratio=1.0,
    )

    for method in args.methods:
        adapter.prune_method = method
        for r in args.ratios:
            adapter.set_keep_ratio(r)
            label = ratio_label(r)
            for ds_name in args.datasets:
                out_path = output_dir / method / f"{ds_name}_{label}.json"
                if os.path.exists(out_path):
                    print(f"[SKIP] method={method} R={r} {ds_name}", flush=True)
                    continue

                module, data_path = DATASETS[ds_name]
                if not os.path.exists(data_path):
                    print(f"[WARN] data not found, skip {ds_name}: {data_path}", flush=True)
                    continue

                adapter.system_prompt = SYSTEM_PROMPTS.get(ds_name, "")
                print(f"[RUN] method={method} keep_ratio={r} dataset={ds_name} "
                      f"(random {args.random_samples}, seed {args.sample_seed}, batch {args.eval_batch_size})",
                      flush=True)
                results, _ = evaluate(
                    adapter, module, data_path, 0, args.eval_batch_size,
                    random_samples=args.random_samples, sample_seed=args.sample_seed,
                )

                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"  -> saved {out_path}", flush=True)

    print(f"DONE. results under {output_dir}", flush=True)


if __name__ == "__main__":
    main()
