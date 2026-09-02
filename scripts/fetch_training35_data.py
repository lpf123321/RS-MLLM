#!/usr/bin/env python3
"""Download and stage the two README 3.5 JSON bundles from ModelScope."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

FIVE_STAGE_MARKERS = (
    "manifest_sft_train.json",
    "g_a2_mix.json",
    "a1_domainalign.json",
    "a2_change_mix.json",
    "expert_data_caption.jsonl",
)
EXPERT_MARKERS = (
    "general/exp3_mme3736_xlrs3080.json",
    "general/exp7_weighted15480.json",
    "grounding/bootstrap_vrsnew942.json",
    "grounding/exp1_washed_vrs36285.json",
    "grounding/exp4_xlrs6611.json",
    "grounding/exp5_all43838.json",
    "IMAGE_INDEX.jsonl",
)


def present(root: Path, markers: tuple[str, ...]) -> bool:
    return all((root / marker).is_file() for marker in markers)


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--five-stage-repo", default="yasumi/rs-mllm-datasets")
    parser.add_argument(
        "--expert-repo", default="Uchitachi/RS-MLLM-Distillation-Data"
    )
    parser.add_argument(
        "--five-stage-output", type=Path, default=repo / "finetune_framework/VRSbench"
    )
    parser.add_argument(
        "--expert-output",
        type=Path,
        default=repo / "datasets" / "training35" / ".source",
    )
    parser.add_argument(
        "--cache-root", type=Path, default=repo / ".models" / "training35-data-cache"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="show repositories and outputs only"
    )
    args = parser.parse_args()

    plan = {
        "five_stage": {
            "repo": args.five_stage_repo,
            "output": str(args.five_stage_output.resolve()),
            "ready": present(args.five_stage_output, FIVE_STAGE_MARKERS),
        },
        "expert_lora": {
            "repo": args.expert_repo,
            "output": str(args.expert_output.resolve()),
            "ready": present(args.expert_output, EXPERT_MARKERS),
        },
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    if os.environ.get("MODELSCOPE_OFFLINE") and not all(
        item["ready"] for item in plan.values()
    ):
        raise SystemExit("required JSON is missing and MODELSCOPE_OFFLINE=1")

    sys.path.insert(0, str(repo))
    if not plan["five_stage"]["ready"]:
        from rsmllm.data import get_dataset, restore_training_data

        downloaded = get_dataset(
            args.five_stage_repo, cache_dir=str(args.cache_root / "five_stage")
        )
        restore_training_data(
            dataset_root=downloaded, dest_root=str(args.five_stage_output)
        )

    if not plan["expert_lora"]["ready"]:
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except ImportError as exc:
            raise SystemExit("modelscope is required: pip install modelscope") from exc
        args.expert_output.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            model_id=args.expert_repo,
            repo_type="dataset",
            local_dir=str(args.expert_output),
            token=os.environ.get("MODELSCOPE_API_TOKEN") or None,
        )

    missing = [
        str(args.five_stage_output / marker)
        for marker in FIVE_STAGE_MARKERS
        if not (args.five_stage_output / marker).is_file()
    ]
    missing.extend(
        str(args.expert_output / marker)
        for marker in EXPERT_MARKERS
        if not (args.expert_output / marker).is_file()
    )
    if missing:
        raise SystemExit("download finished but required files are missing:\n  " + "\n  ".join(missing))
    plan["five_stage"]["ready"] = present(args.five_stage_output, FIVE_STAGE_MARKERS)
    plan["expert_lora"]["ready"] = present(args.expert_output, EXPERT_MARKERS)
    print(json.dumps({"passed": True, **plan}, indent=2))


if __name__ == "__main__":
    main()
