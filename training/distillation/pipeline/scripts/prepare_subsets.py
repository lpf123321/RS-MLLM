#!/usr/bin/env python3
"""Build deterministic, leakage-aware subsets for the OPD validation pipeline.

The source JSONL files contain paths from a different machine and LEVIR-CC
stores captions as dictionaries.  This script repairs both issues, samples by
image group, and writes both standard JSON arrays and evaluation JSONL, plus a
unified LLaVA-style SFT file.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


DEFAULT_SEED = 20260809
OLD_DATA_ROOT = os.environ.get("OLD_DATA_ROOT", "/users/u2024311136/shared/shared_datasets")
LOCAL_DATA_ROOT = os.environ.get("DATA_ROOT", "./datasets")

SOURCES = {
    "vrs_train": Path(f"{LOCAL_DATA_ROOT}/VRSBench/vrsbench_train.jsonl"),
    "vrs_eval": Path(f"{LOCAL_DATA_ROOT}/VRSBench/vrsbench_eval.jsonl"),
    "mme": Path(f"{LOCAL_DATA_ROOT}/MME-RealWorld-RS/mme_rs.jsonl"),
    "levir_train": Path(f"{LOCAL_DATA_ROOT}/LEVIR-CC/levircc_train.jsonl"),
    "levir_val": Path(f"{LOCAL_DATA_ROOT}/LEVIR-CC/levircc_val.jsonl"),
    "levir_test": Path(f"{LOCAL_DATA_ROOT}/LEVIR-CC/levircc_test.jsonl"),
    "xlrs": Path(f"{LOCAL_DATA_ROOT}/XLRS-Bench-lite/xlrs.jsonl"),
}

PREFIX_REPLACEMENTS = {
    "[caption]": "[CAP]",
    "[cap]": "[CAP]",
    "[refer]": "[REF]",
    "[ref]": "[REF]",
    "[vqa]": "[VQA]",
    "[cd]": "[CD]",
    "[mcq]": "[MCQ]",
}

TASK_BY_PREFIX = {
    "[CAP]": "caption",
    "[REF]": "referring",
    "[VQA]": "vqa",
    "[CD]": "change_caption",
    "[MCQ]": "mcq",
}

MCQ_LETTER_RE = re.compile(r"(?<!\w)([A-Ea-e])\s*[.)]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("PIPELINE_ROOT", Path(__file__).resolve().parents[1])),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--official-fraction",
        type=float,
        default=0.10,
        help="Fraction sampled from datasets with official train/eval splits.",
    )
    parser.add_argument(
        "--single-train-fraction",
        type=float,
        default=0.10,
        help="Training fraction for benchmark-only MME/XLRS data.",
    )
    parser.add_argument(
        "--single-eval-fraction",
        type=float,
        default=0.10,
        help="Disjoint held-out fraction for benchmark-only MME/XLRS data.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_score(seed: int, namespace: str, value: str) -> str:
    payload = f"{seed}|{namespace}|{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def nearest_count(total: int, fraction: float) -> int:
    if total == 0 or fraction <= 0:
        return 0
    return max(1, min(total, math.floor(total * fraction + 0.5)))


def normalize_text(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("raw") or " ".join(value.get("tokens", []))
    if not isinstance(value, str):
        raise TypeError(f"Expected textual answer/reference, got {type(value).__name__}")
    return " ".join(value.strip().split())


def normalize_prompt(text: str) -> str:
    stripped = text.strip()
    for old_prefix, new_prefix in PREFIX_REPLACEMENTS.items():
        if stripped.lower().startswith(old_prefix):
            return new_prefix + stripped[len(old_prefix) :]
    return stripped


def normalize_record(raw: dict[str, Any], dataset: str, source_index: int) -> dict[str, Any]:
    record = copy.deepcopy(raw)
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"{dataset}:{source_index}: missing user/assistant messages")

    user_content = messages[0].get("content")
    assistant_content = messages[1].get("content")
    if not isinstance(user_content, list) or not isinstance(assistant_content, list):
        raise ValueError(f"{dataset}:{source_index}: invalid message content")

    image_count = 0
    for part in user_content:
        if part.get("type") == "image":
            part["image"] = part["image"].replace(OLD_DATA_ROOT, LOCAL_DATA_ROOT)
            image_count += 1
    if image_count == 0:
        raise ValueError(f"{dataset}:{source_index}: no images")

    user_content[-1]["text"] = normalize_prompt(user_content[-1]["text"])
    assistant_content[0]["text"] = normalize_text(assistant_content[0]["text"])

    if "references" in record:
        record["references"] = [normalize_text(item) for item in record["references"]]

    record["sample_metadata"] = {
        "dataset": dataset,
        "source_index": source_index,
    }
    return record


def load_jsonl(source_name: str) -> list[dict[str, Any]]:
    path = SOURCES[source_name]
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            if line.strip():
                records.append(normalize_record(json.loads(line), source_name, source_index))
    return records


def image_paths(record: dict[str, Any]) -> list[str]:
    return [
        part["image"]
        for part in record["messages"][0]["content"]
        if part.get("type") == "image"
    ]


def group_key(record: dict[str, Any]) -> str:
    return "\u241f".join(image_paths(record))


def group_records(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[group_key(record)].append(record)
    return dict(grouped)


def select_group_fraction(
    records: list[dict[str, Any]], fraction: float, seed: int, namespace: str
) -> tuple[list[dict[str, Any]], set[str]]:
    grouped = group_records(records)
    ordered_keys = sorted(grouped, key=lambda key: stable_score(seed, namespace, key))
    chosen = set(ordered_keys[: nearest_count(len(ordered_keys), fraction)])
    selected = [record for record in records if group_key(record) in chosen]
    return selected, chosen


def split_single_source(
    records: list[dict[str, Any]],
    train_fraction: float,
    eval_fraction: float,
    seed: int,
    namespace: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str], set[str]]:
    grouped = group_records(records)
    ordered_keys = sorted(grouped, key=lambda key: stable_score(seed, namespace, key))
    train_count = nearest_count(len(ordered_keys), train_fraction)
    eval_count = nearest_count(len(ordered_keys), eval_fraction)
    if train_count + eval_count > len(ordered_keys):
        raise ValueError(f"Requested split exceeds source size for {namespace}")
    train_keys = set(ordered_keys[:train_count])
    eval_keys = set(ordered_keys[train_count : train_count + eval_count])
    train_records = [record for record in records if group_key(record) in train_keys]
    eval_records = [record for record in records if group_key(record) in eval_keys]
    return train_records, eval_records, train_keys, eval_keys


def answer_text(record: dict[str, Any]) -> str:
    return record["messages"][1]["content"][0]["text"]


def prompt_text(record: dict[str, Any]) -> str:
    return record["messages"][0]["content"][-1]["text"]


def task_name(record: dict[str, Any]) -> str:
    prompt = prompt_text(record)
    for prefix, name in TASK_BY_PREFIX.items():
        if prompt.startswith(prefix):
            return name
    return "unknown"


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(records), handle, ensure_ascii=False, indent=2)


def to_llava(record: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    images = image_paths(record)
    prompt = prompt_text(record)
    image_tokens = "\n".join("<image>" for _ in images)
    source_index = record["sample_metadata"]["source_index"]
    item: dict[str, Any] = {
        "id": f"{dataset_name}:{source_index}",
        "image": images[0] if len(images) == 1 else images,
        "conversations": [
            {"from": "human", "value": f"{image_tokens}\n{prompt}"},
            {"from": "gpt", "value": answer_text(record)},
        ],
        "source_dataset": dataset_name,
    }
    return item


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    task_counts = Counter(task_name(record) for record in records)
    answer_letters = Counter()
    missing_images = []
    for record in records:
        match = MCQ_LETTER_RE.search(answer_text(record))
        if match:
            answer_letters[match.group(1).upper()] += 1
        for path in image_paths(record):
            if not os.path.isfile(path):
                missing_images.append(path)
    return {
        "records": len(records),
        "image_groups": len(group_records(records)),
        "task_counts": dict(sorted(task_counts.items())),
        "answer_letter_counts": dict(sorted(answer_letters.items())),
        "missing_image_references": len(missing_images),
        "missing_image_examples": missing_images[:10],
    }


def verify_images(records_by_split: dict[str, list[dict[str, Any]]], seed: int) -> dict[str, Any]:
    results = {}
    for split_name, records in records_by_split.items():
        unique_paths = sorted({path for record in records for path in image_paths(record)})
        ordered = sorted(unique_paths, key=lambda path: stable_score(seed, split_name, path))
        checked = []
        failures = []
        for path in ordered[: min(25, len(ordered))]:
            try:
                with Image.open(path) as image:
                    image.verify()
                checked.append(path)
            except Exception as exc:  # pragma: no cover - audit output captures exact failure
                failures.append({"path": path, "error": repr(exc)})
        results[split_name] = {
            "unique_images": len(unique_paths),
            "decoded_images_checked": len(checked),
            "decode_failures": failures,
        }
    return results


def main() -> None:
    args = parse_args()
    if not 0.10 <= args.official_fraction <= 0.20:
        raise ValueError("official-fraction must be between 0.10 and 0.20")
    combined_single_fraction = args.single_train_fraction + args.single_eval_fraction
    if not 0.10 <= combined_single_fraction <= 0.20:
        raise ValueError("MME/XLRS selected train+eval fraction must be between 0.10 and 0.20")

    for source_name, source_path in SOURCES.items():
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source {source_name}: {source_path}")

    loaded = {name: load_jsonl(name) for name in SOURCES}

    vrs_train, vrs_train_keys = select_group_fraction(
        loaded["vrs_train"], args.official_fraction, args.seed, "vrs_train"
    )
    vrs_eval, vrs_eval_keys = select_group_fraction(
        loaded["vrs_eval"], args.official_fraction, args.seed, "vrs_eval"
    )
    levir_train, levir_train_keys = select_group_fraction(
        loaded["levir_train"], args.official_fraction, args.seed, "levir_train"
    )
    levir_val, levir_val_keys = select_group_fraction(
        loaded["levir_val"], args.official_fraction, args.seed, "levir_val"
    )
    levir_eval, levir_eval_keys = select_group_fraction(
        loaded["levir_test"], args.official_fraction, args.seed, "levir_test"
    )
    mme_train, mme_eval, mme_train_keys, mme_eval_keys = split_single_source(
        loaded["mme"],
        args.single_train_fraction,
        args.single_eval_fraction,
        args.seed,
        "mme",
    )
    xlrs_train, xlrs_eval, xlrs_train_keys, xlrs_eval_keys = split_single_source(
        loaded["xlrs"],
        args.single_train_fraction,
        args.single_eval_fraction,
        args.seed,
        "xlrs",
    )

    if mme_train_keys & mme_eval_keys:
        raise AssertionError("MME train/eval image leakage")
    if xlrs_train_keys & xlrs_eval_keys:
        raise AssertionError("XLRS train/eval image leakage")

    train_splits = {
        "vrsbench": vrs_train,
        "mme": mme_train,
        "levircc": levir_train,
        "xlrs": xlrs_train,
    }
    eval_splits = {
        "vrsbench": vrs_eval,
        "mme": mme_eval,
        "levircc": levir_eval,
        "xlrs": xlrs_eval,
    }
    validation_splits = {"levircc": levir_val}

    for name, records in train_splits.items():
        write_jsonl(args.output_root / "data" / "train" / f"{name}.jsonl", records)
        write_json(args.output_root / "data" / "train" / f"{name}.json", records)
    for name, records in eval_splits.items():
        write_jsonl(args.output_root / "data" / "eval" / f"{name}.jsonl", records)
        write_json(args.output_root / "data" / "eval" / f"{name}.json", records)
    for name, records in validation_splits.items():
        write_jsonl(args.output_root / "data" / "validation" / f"{name}.jsonl", records)
        write_json(args.output_root / "data" / "validation" / f"{name}.json", records)

    unified_sft = []
    for dataset_name, records in train_splits.items():
        unified_sft.extend(to_llava(record, dataset_name) for record in records)
    unified_sft.sort(
        key=lambda item: stable_score(args.seed, "unified_sft", item["id"])
    )
    sft_path = args.output_root / "data" / "train" / "unified_sft.json"
    with sft_path.open("w", encoding="utf-8") as handle:
        json.dump(unified_sft, handle, ensure_ascii=False, indent=2)

    all_splits = {
        **{f"train/{name}": records for name, records in train_splits.items()},
        **{f"eval/{name}": records for name, records in eval_splits.items()},
        **{f"validation/{name}": records for name, records in validation_splits.items()},
    }
    summaries = {name: summarize_records(records) for name, records in all_splits.items()}
    image_verification = verify_images(all_splits, args.seed)

    source_summaries = {
        name: {
            "path": str(SOURCES[name]),
            "sha256": sha256_file(SOURCES[name]),
            **summarize_records(records),
        }
        for name, records in loaded.items()
    }

    group_counts = {
        "vrs_train": len(vrs_train_keys),
        "vrs_eval": len(vrs_eval_keys),
        "levir_train": len(levir_train_keys),
        "levir_val": len(levir_val_keys),
        "levir_eval": len(levir_eval_keys),
        "mme_train": len(mme_train_keys),
        "mme_eval": len(mme_eval_keys),
        "xlrs_train": len(xlrs_train_keys),
        "xlrs_eval": len(xlrs_eval_keys),
    }

    manifest = {
        "schema_version": 2,
        "seed": args.seed,
        "sampling": {
            "official_fraction": args.official_fraction,
            "single_source_train_fraction": args.single_train_fraction,
            "single_source_eval_fraction": args.single_eval_fraction,
            "unit": "unique image group; all QAs/captions for a selected image group are retained",
            "group_counts": group_counts,
        },
        "path_rewrite": {"from": OLD_DATA_ROOT, "to": LOCAL_DATA_ROOT},
        "sources": source_summaries,
        "outputs": summaries,
        "output_files": {
            split_name: {
                "json": str(args.output_root / "data" / split_name) + ".json",
                "jsonl": str(args.output_root / "data" / split_name) + ".jsonl",
            }
            for split_name in summaries
        },
        "unified_sft_records": len(unified_sft),
        "image_decode_audit": image_verification,
        "leakage_checks": {
            "mme_train_eval_shared_image_groups": len(mme_train_keys & mme_eval_keys),
            "xlrs_train_eval_shared_image_groups": len(xlrs_train_keys & xlrs_eval_keys),
            "vrs_uses_official_disjoint_train_eval_sources": True,
            "levir_uses_official_disjoint_train_val_test_sources": True,
        },
    }

    manifest_path = args.output_root / "manifests" / "sampling_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    report_path = args.output_root / "reports" / "data_preparation_report.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# Data preparation report\n\n")
        handle.write(f"- Seed: `{args.seed}`\n")
        handle.write(f"- Official split sampling fraction: `{args.official_fraction:.0%}`\n")
        handle.write(
            "- MME/XLRS selection: "
            f"`{args.single_train_fraction:.0%}` train + "
            f"`{args.single_eval_fraction:.0%}` disjoint evaluation\n"
        )
        handle.write("- Sampling unit: unique image group (multi-image pairs stay together)\n")
        handle.write("- Images are referenced in place; no input images are copied.\n")
        handle.write(f"- Stale source paths were rewritten to `{LOCAL_DATA_ROOT}`.\n")
        handle.write("- LEVIR-CC dictionary captions/references were normalized to their `raw` text.\n\n")
        handle.write("## Output counts\n\n")
        handle.write("| Split | Records | Image groups | Tasks | Missing images |\n")
        handle.write("|---|---:|---:|---|---:|\n")
        for name, summary in sorted(summaries.items()):
            tasks = ", ".join(f"{key}={value}" for key, value in summary["task_counts"].items())
            handle.write(
                f"| {name} | {summary['records']} | {summary['image_groups']} | "
                f"{tasks} | {summary['missing_image_references']} |\n"
            )
        handle.write(f"\nUnified SFT records: **{len(unified_sft)}**\n")
        handle.write("\nFull hashes, distributions, decode checks, and leakage checks are in "
                     "`manifests/sampling_manifest.json`.\n")

    print(json.dumps({
        "manifest": str(manifest_path),
        "report": str(report_path),
        "unified_sft": str(sft_path),
        "outputs": summaries,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
