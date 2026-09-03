#!/usr/bin/env python3
"""Build README 3.6 image links from datasets/shared_datasets/.

The command never copies the full upstream datasets. Five-stage images become
content-addressed symlinks under data/assets; expert-LoRA images become verified
hard links (with a copy fallback) under datasets/training35/.source/images.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SOURCE_DIRS = {
    "mme_realworld_rs": "MME-RealWorld-RS",
    "vrsbench": "VRSBench",
    "xlrs": "XLRS-Bench-lite",
    "xlrs_grounding": "XLRS-Bench_visual_grounding_en",
}
FIVE_STAGE_DIRS = {
    "vrsbench": "VRSBench",
    "levir_cc": "LEVIR-CC",
}
VRS_RECOVER_SHA = "26ab63466cc4f113e6b5e446792eef7ecd3e4a0f7e5a708f3b889ec30cacc7e0"
VRS_RECOVER_MEMBER = "Images_train/P7581_0003.png"


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root", type=Path, default=repo / "datasets" / "shared_datasets"
    )
    parser.add_argument(
        "--expert-source",
        type=Path,
        default=repo / "datasets" / "training35" / ".source",
    )
    parser.add_argument("--assets-out", type=Path, default=repo / "data" / "assets")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify expert image hashes without creating expert image links",
    )
    parser.add_argument(
        "--include-expert",
        action="store_true",
        help="also materialize optional General/Grounding continuation images",
    )
    parser.add_argument(
        "--expert-smoke-samples",
        type=int,
        default=0,
        help="materialize only the first N records of each retained expert run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print required directories and commands without reading images",
    )
    args = parser.parse_args()
    if args.expert_smoke_samples < 0:
        raise ValueError("--expert-smoke-samples must be non-negative")
    if args.expert_smoke_samples and not args.include_expert:
        raise ValueError("--expert-smoke-samples requires --include-expert")

    raw_root = args.raw_root.expanduser().resolve()
    expert_source = args.expert_source.expanduser().resolve()
    assets_out = args.assets_out.expanduser().resolve()
    required = {
        "vrsbench": raw_root / SOURCE_DIRS["vrsbench"],
        "levir_cc": raw_root / FIVE_STAGE_DIRS["levir_cc"],
    }
    if args.include_expert:
        required.update(
            {
                "mme_realworld_rs": raw_root / SOURCE_DIRS["mme_realworld_rs"],
                "xlrs": raw_root / SOURCE_DIRS["xlrs"],
                "xlrs_grounding": raw_root / SOURCE_DIRS["xlrs_grounding"],
            }
        )
    export_xlrs_command = [
        sys.executable,
        str(repo / "scripts" / "export_xlrs_grounding_images.py"),
        "--dataset",
        str(raw_root / SOURCE_DIRS["xlrs_grounding"] / "train"),
        "--output",
        str(raw_root / SOURCE_DIRS["xlrs_grounding"] / "images_exported_train_4096"),
    ]

    build_command = [
        sys.executable,
        str(repo / "scripts" / "build_assets_from_raw.py"),
        "--raw",
        str(required["vrsbench"]),
        "--sub",
        "vrsbench",
        "--raw",
        str(required["levir_cc"]),
        "--sub",
        "levir_cc",
        "--out",
        str(assets_out),
        "--jobs",
        str(args.jobs),
    ]
    recover_output = assets_out / "vrsbench" / f"{VRS_RECOVER_SHA}.png"
    recover_command = [
        sys.executable,
        str(repo / "scripts" / "recover_asset_from_zip.py"),
        "--zip",
        str(required["vrsbench"] / "Images_train.zip"),
        "--member",
        VRS_RECOVER_MEMBER,
        "--sha256",
        VRS_RECOVER_SHA,
        "--output",
        str(recover_output),
    ]
    materialize_command = [
        sys.executable,
        str(
            repo
            / "training"
            / "distillation"
            / "expert_lora"
            / "scripts"
            / "materialize_images.py"
        ),
        "--dataset-root",
        str(expert_source),
    ]
    for name in SOURCE_DIRS:
        materialize_command.extend(
            ["--source", f"{name}={raw_root / SOURCE_DIRS[name]}"]
        )
    if args.verify_only:
        materialize_command.append("--verify-only")
    if args.expert_smoke_samples:
        for relative in (
            "general/exp7_weighted15480.json",
            "grounding/bootstrap_vrsnew942.json",
            "grounding/exp5_all43838.json",
        ):
            materialize_command.extend(
                ["--required-from", str(expert_source / relative)]
            )
        materialize_command.extend(
            ["--limit-records", str(args.expert_smoke_samples)]
        )

    report = {
        "raw_root": str(raw_root),
        "required": {
            name: {"path": str(path), "exists": path.is_dir()}
            for name, path in required.items()
        },
        "include_expert": args.include_expert,
        "expert_index": {
            "path": str(expert_source / "IMAGE_INDEX.jsonl"),
            "exists": (expert_source / "IMAGE_INDEX.jsonl").is_file(),
        },
        "commands": (
            [recover_command, build_command]
            if not args.include_expert
            else [recover_command, export_xlrs_command, build_command, materialize_command]
        ),
    }
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    missing = [str(path) for path in required.values() if not path.is_dir()]
    if missing:
        raise SystemExit(
            "missing official dataset directories:\n  "
            + "\n  ".join(missing)
            + "\nPlace the original datasets under datasets/shared_datasets/ "
            "as documented in README 3.6."
        )
    if args.include_expert and not (expert_source / "IMAGE_INDEX.jsonl").is_file():
        raise SystemExit(
            f"missing {expert_source / 'IMAGE_INDEX.jsonl'}; download the adapted "
            "JSON bundle from ModelScope first"
        )

    if not recover_output.is_file():
        archive = required["vrsbench"] / "Images_train.zip"
        if not archive.is_file():
            raise SystemExit(
                f"missing {archive}; it is required to recover the known "
                f"zero-byte {VRS_RECOVER_MEMBER} asset"
            )
        subprocess.run(recover_command, check=True)
    if args.include_expert:
        subprocess.run(export_xlrs_command, check=True)
    subprocess.run(build_command, check=True)
    if args.include_expert:
        subprocess.run(materialize_command, check=True)


if __name__ == "__main__":
    main()
