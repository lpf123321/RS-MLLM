#!/usr/bin/env python3
"""Build all README 3.5 image links from official datasets under datasets/.

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


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=repo / "datasets")
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
        "--dry-run",
        action="store_true",
        help="print required directories and commands without reading images",
    )
    args = parser.parse_args()

    raw_root = args.raw_root.expanduser().resolve()
    expert_source = args.expert_source.expanduser().resolve()
    assets_out = args.assets_out.expanduser().resolve()
    required = {
        "mme_realworld_rs": raw_root / SOURCE_DIRS["mme_realworld_rs"],
        "vrsbench": raw_root / SOURCE_DIRS["vrsbench"],
        "xlrs": raw_root / SOURCE_DIRS["xlrs"],
        "xlrs_grounding": raw_root / SOURCE_DIRS["xlrs_grounding"],
        "levir_cc": raw_root / FIVE_STAGE_DIRS["levir_cc"],
    }
    export_xlrs_command = [
        sys.executable,
        str(repo / "scripts" / "export_xlrs_grounding_images.py"),
        "--dataset",
        str(required["xlrs_grounding"] / "train"),
        "--output",
        str(required["xlrs_grounding"] / "images_exported_train_4096"),
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
        materialize_command.extend(["--source", f"{name}={required[name]}"])
    if args.verify_only:
        materialize_command.append("--verify-only")

    report = {
        "raw_root": str(raw_root),
        "required": {
            name: {"path": str(path), "exists": path.is_dir()}
            for name, path in required.items()
        },
        "expert_index": {
            "path": str(expert_source / "IMAGE_INDEX.jsonl"),
            "exists": (expert_source / "IMAGE_INDEX.jsonl").is_file(),
        },
        "commands": [export_xlrs_command, build_command, materialize_command],
    }
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    missing = [str(path) for path in required.values() if not path.is_dir()]
    if missing:
        raise SystemExit(
            "missing official dataset directories:\n  "
            + "\n  ".join(missing)
            + "\nPlace the original datasets under datasets/ as documented in README 3.5."
        )
    if not (expert_source / "IMAGE_INDEX.jsonl").is_file():
        raise SystemExit(
            f"missing {expert_source / 'IMAGE_INDEX.jsonl'}; download the adapted "
            "JSON bundle from ModelScope first"
        )

    subprocess.run(export_xlrs_command, check=True)
    subprocess.run(build_command, check=True)
    subprocess.run(materialize_command, check=True)


if __name__ == "__main__":
    main()
