#!/usr/bin/env python3
"""Stage the canonical README 3.5 dataset layout with validated symlinks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

FIVE_STAGE = {
    "stage1_clean.json": "manifest_sft_train.json",
    "ga2_general.json": "g_a2_mix.json",
    "a1_grounding.json": "a1_domainalign.json",
    "a2b_change.json": "a2_change_mix.json",
    "caption.jsonl": "expert_data_caption.jsonl",
}
EXPERT_DATA = {
    "expert_general_lora_mcq.json": "general/exp3_mme3736_xlrs3080.json",
    "expert_general_lora.json": "general/exp7_weighted15480.json",
    "expert_ground_lora_bootstrap.json": "grounding/bootstrap_vrsnew942.json",
    "expert_ground_lora_vrs.json": "grounding/exp1_washed_vrs36285.json",
    "expert_ground_lora_xlrs.json": "grounding/exp4_xlrs6611.json",
    "expert_ground_lora.json": "grounding/exp5_all43838.json",
}
RAW_DATASETS = (
    "LEVIR-CC",
    "MME-RealWorld-RS",
    "VRSBench",
    "XLRS-Bench-lite",
    "XLRS-Bench_caption_en",
    "XLRS-Bench_visual_grounding_en",
)


def ensure_link(link: Path, target: Path) -> None:
    target = target.expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(target)
    if link.is_symlink():
        if link.resolve() != target:
            raise FileExistsError(f"conflicting link: {link} -> {link.resolve()} (wanted {target})")
        return
    if link.exists():
        raise FileExistsError(f"refusing to replace existing path: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    relative = os.path.relpath(target, link.parent.resolve())
    link.symlink_to(relative, target_is_directory=target.is_dir())


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--five-stage-source", type=Path, default=repo / "finetune_framework/VRSbench")
    parser.add_argument("--expert-source", type=Path, default=repo / "datasets/training35/.source")
    parser.add_argument("--raw-source", type=Path, default=repo / "datasets/shared_datasets")
    parser.add_argument("--output", type=Path, default=repo / "datasets/training35")
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    links: dict[str, str] = {}
    for name, source_name in FIVE_STAGE.items():
        source = args.five_stage_source / source_name
        ensure_link(output / name, source)
        links[name] = str(source.resolve())
    for name, source_name in EXPERT_DATA.items():
        source = args.expert_source / source_name
        ensure_link(output / name, source)
        links[name] = str(source.resolve())

    ensure_link(output / "assets", repo / "data/assets")
    ensure_link(output / "images", args.expert_source / "images")
    for name in ("ASSET_MANIFEST.json", "IMAGE_INDEX.jsonl", "LICENSES.md", "SHA256SUMS"):
        source = args.expert_source / name
        if source.is_file():
            ensure_link(output / name, source)

    raw_source = args.raw_source.expanduser().resolve()
    for name in RAW_DATASETS:
        source = raw_source / name
        if source.exists():
            ensure_link(repo / "datasets" / name, source)

    manifest = {
        "schema_version": 1,
        "layout": "datasets/training35",
        "links": links,
        "assets": str((repo / "data/assets").resolve()),
        "expert_images": str((args.expert_source / "images").resolve()),
        "raw_source": str(raw_source),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "STAGING_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "files": len(links), "passed": True}, indent=2))


if __name__ == "__main__":
    main()
