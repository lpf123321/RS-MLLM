#!/usr/bin/env python3
"""Stage the canonical README 3.6 dataset layout with portable paths."""
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
SHARED_MARKER = "/shared_datasets/"


def ensure_link(link: Path, target: Path) -> None:
    target = target.expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(target)
    if link.is_symlink():
        if link.resolve() != target:
            raise FileExistsError(
                f"conflicting link: {link} -> {link.resolve()} (wanted {target})"
            )
        return
    if link.exists():
        raise FileExistsError(f"refusing to replace existing path: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    relative = os.path.relpath(target, link.parent.resolve())
    link.symlink_to(relative, target_is_directory=target.is_dir())


def stage_five_stage_file(destination: Path, source: Path) -> int:
    """Link portable JSON, or materialize a normalized copy if paths are absolute."""
    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"expected JSON array: {source}")
    rewritten = 0
    for row_number, row in enumerate(rows):
        images = row.get("image", [])
        was_string = isinstance(images, str)
        values = [images] if was_string else images
        if not isinstance(values, list):
            raise ValueError(f"invalid image field in {source}, row {row_number}")
        normalized: list[object] = []
        for value in values:
            if not isinstance(value, str) or not Path(value).is_absolute():
                normalized.append(value)
                continue
            slash_value = value.replace("\\", "/")
            if SHARED_MARKER not in slash_value:
                raise ValueError(
                    f"refusing unknown absolute image path in {source}, row {row_number}: {value}"
                )
            suffix = slash_value.split(SHARED_MARKER, 1)[1]
            normalized.append(f"../shared_datasets/{suffix}")
            rewritten += 1
        row["image"] = normalized[0] if was_string else normalized

    if not rewritten:
        ensure_link(destination, source)
        return 0

    if destination.is_symlink():
        destination.unlink()
    elif destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        # Migrate files produced by the older layout, which resolved raw data
        # through datasets/<name> instead of datasets/shared_datasets/<name>.
        legacy = json.loads(json.dumps(rows))
        for row in legacy:
            images = row.get("image", [])
            values = [images] if isinstance(images, str) else images
            if isinstance(values, list):
                migrated = [
                    value.replace("../shared_datasets/", "../", 1)
                    if isinstance(value, str)
                    and value.startswith("../shared_datasets/")
                    else value
                    for value in values
                ]
                row["image"] = migrated[0] if isinstance(images, str) else migrated
        if existing != rows and existing != legacy:
            raise FileExistsError(f"refusing to replace modified staged JSON: {destination}")
        if existing == rows:
            return rewritten
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(rows, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return rewritten


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--five-stage-source", type=Path, default=repo / "finetune_framework/VRSbench"
    )
    parser.add_argument(
        "--expert-source",
        type=Path,
        default=repo / "datasets/training35/.source",
    )
    parser.add_argument(
        "--raw-source", type=Path, default=repo / "datasets" / "shared_datasets"
    )
    parser.add_argument("--output", type=Path, default=repo / "datasets/training35")
    parser.add_argument(
        "--include-expert",
        action="store_true",
        help="also stage optional General/Grounding continuation data and images",
    )
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    links: dict[str, str] = {}
    rewrites: dict[str, int] = {}
    for name, source_name in FIVE_STAGE.items():
        source = args.five_stage_source / source_name
        if not source.is_file():
            raise FileNotFoundError(source)
        rewrites[name] = stage_five_stage_file(output / name, source)
        links[name] = str(source.resolve())
    ensure_link(output / "assets", repo / "data/assets")
    if args.include_expert:
        for name, source_name in EXPERT_DATA.items():
            source = args.expert_source / source_name
            ensure_link(output / name, source)
            links[name] = str(source.resolve())
        ensure_link(output / "images", args.expert_source / "images")
        for name in ("ASSET_MANIFEST.json", "IMAGE_INDEX.jsonl", "LICENSES.md", "SHA256SUMS"):
            source = args.expert_source / name
            if source.is_file():
                ensure_link(output / name, source)

    raw_source = args.raw_source.expanduser().resolve()

    manifest = {
        "schema_version": 1,
        "layout": "datasets/training35",
        "sources": links,
        "rewritten_absolute_images": rewrites,
        "assets": str((repo / "data/assets").resolve()),
        "expert_images": str((args.expert_source / "images").resolve()),
        "include_expert": args.include_expert,
        "raw_source": str(raw_source),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "STAGING_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "files": len(links),
                "rewritten_absolute_images": sum(rewrites.values()),
                "passed": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
