#!/usr/bin/env python3
"""Build the portable ModelScope dataset bundle from the three retained runs.

The command is intentionally non-destructive: OUTPUT must be absent or empty.
It never packages test answers, merged checkpoints, caches, or machine paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable


METHODS = {
    "tree_node_weak_box": "qwen3vl_2b_bbox_free_one_fifth",
    "strict_local_recovery": "qwen3vl_2b_sam_recovered_multi_one_fifth",
    "fine_trace_only": "qwen3vl_2b_fine_sam_trace_one_fifth",
}
TRAJECTORIES = (
    "train.jsonl", "valid.jsonl", "valid_sam_only.jsonl",
    "valid_mllm_only.jsonl", "valid_both.jsonl",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_content(source: Path, output: Path, subdir: str) -> str:
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = sha256(source)
    suffix = source.suffix.lower() or ".bin"
    relative = Path(subdir) / f"{digest[:24]}{suffix}"
    destination = output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if sha256(destination) != digest:
            raise ValueError(f"Hash collision while staging {source}")
    else:
        shutil.copy2(source, destination)
    return relative.as_posix()


def scrub_machine_paths(value: Any) -> Any:
    if isinstance(value, str) and Path(value).is_absolute():
        return f"redacted-local-path://{Path(value).name}"
    if isinstance(value, list):
        return [scrub_machine_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: scrub_machine_paths(item) for key, item in value.items()}
    return value


def portable_split(
    source: Path, split: str, output: Path, image_by_sample: dict[str, str],
) -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl(source):
        sample_id = str(row["sample_id"])
        if sample_id in image_by_sample:
            raise ValueError(f"Split overlap at sample_id={sample_id}")
        relative = copy_content(Path(row["image_path"]), output, "vision_opd/images")
        image_by_sample[sample_id] = relative
        portable = scrub_machine_paths(row)
        portable.pop("image_path", None)
        portable.update({
            "image_relative_path": relative.removeprefix("vision_opd/"),
            "split": split,
            # The released bbox-free reproduction cannot accidentally read GT.
            "optional_gt_bbox_xyxy": None,
            "optional_gt_bbox_xywh": None,
            "optional_gt_bbox_area_ratio": None,
        })
        rows.append(portable)
    return rows


def sealed_test_ids(source: Path) -> list[dict[str, Any]]:
    sealed = []
    for row in read_jsonl(source):
        # Do not publish sample_id: schema-v1 sample IDs included the answer in
        # their preimage and could therefore be brute-forced over A/B/C/D.
        payload = "\0".join((
            "rs-mllm-sealed-test-v1", str(row["source_index"]),
            str(row["image_relative_path"]), str(row["question"]),
        ))
        sealed.append({
            "sealed_id": hashlib.sha256(payload.encode()).hexdigest(),
            "source_index_sha256": hashlib.sha256(str(row["source_index"]).encode()).hexdigest(),
        })
    return sealed


def normalize_masks(value: Any, output: Path, method: str) -> Any:
    if isinstance(value, list):
        return [normalize_masks(item, output, method) for item in value]
    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            if key in {"mask_path", "pseudo_mask_path"} and item:
                source = Path(str(item))
                normalized[key] = (
                    copy_content(source, output, f"audit_masks/{method}") if source.is_file() else None
                )
            else:
                normalized[key] = normalize_masks(item, output, method)
        return normalized
    return value


def portable_trajectory(
    row: dict[str, Any], output: Path, method: str, image_by_sample: dict[str, str],
) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    if sample_id not in image_by_sample:
        raise ValueError(f"Trajectory references a non-published split row: {sample_id}")
    portable = normalize_masks(row, output, method)
    portable["image_path"] = image_by_sample[sample_id]
    teacher = row.get("teacher_image_path")
    if teacher:
        teacher_path = Path(str(teacher))
        portable["teacher_image_path"] = (
            copy_content(teacher_path, output, "teacher_crops") if teacher_path.is_file() else None
        )
    return scrub_machine_paths(portable)


def stage_negatives(source: Path, output: Path) -> int:
    rows = []
    for row in read_jsonl(source):
        portable = dict(row)
        portable["image_path"] = f"portable-source://{row['source_id']}"
        portable["verifier_model"] = "Qwen/Qwen3-VL-8B-Instruct (historical frozen verification)"
        rows.append(scrub_machine_paths(portable))
    write_jsonl(output, rows)
    return len(rows)


def stage_coco(
    source: Path, destination: Path, output: Path, image_by_sample: dict[str, str], method: str,
) -> tuple[int, int]:
    coco = json.loads(source.read_text(encoding="utf-8"))
    by_image_id = {int(image["id"]): image for image in coco["images"]}
    for image in coco["images"]:
        relative = image_by_sample[str(image["source_id"])]
        image["file_name"] = "../../../../" + relative
    for annotation in coco["annotations"]:
        mask_path = annotation.get("pseudo_mask_path")
        if mask_path:
            source_mask = Path(str(mask_path))
            annotation["pseudo_mask_path"] = (
                copy_content(source_mask, output, f"audit_masks/{method}")
                if source_mask.is_file() else None
            )
        if int(annotation["image_id"]) not in by_image_id:
            raise ValueError(f"COCO annotation references missing image: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, scrub_machine_paths(coco))
    return len(coco["images"]), len(coco["annotations"])


def make_opsd_examples(trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in trajectories:
        if not row.get("accepted_for_opsd"):
            continue
        rows.append({
            "data_source": "cvsearch_self_evolution",
            "prompt": [{"role": "user", "content": row["prompt"]}],
            "image_relative_path": row["image_path"],
            "teacher_image_relative_path": row["teacher_image_path"],
            "ability": "visual_question_answering",
            "reward_model": {"style": "none", "ground_truth": row["gt_answer"]},
            "extra_info": {
                "sample_id": row["sample_id"], "answer": row["gt_answer"],
                "question": row["question"], "search_bboxes_xyxy": row["searched_bboxes_xyxy"],
                "targets": row["extracted_targets"], "search_mode": row["search_mode"],
                "num_pop": row["num_pop"], "final_mllm_crop": row.get("final_mllm_crop"),
                "final_mllm_pixel_sha256": row.get("final_mllm_pixel_sha256"),
            },
        })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--spacy-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    workspaces = {name: (args.runs_root / local).resolve() for name, local in METHODS.items()}
    for workspace in workspaces.values():
        if not workspace.is_dir():
            raise FileNotFoundError(workspace)

    image_by_sample: dict[str, str] = {}
    split_root = workspaces["tree_node_weak_box"] / "data/splits"
    train = portable_split(split_root / "train.jsonl", "train", output, image_by_sample)
    valid = portable_split(split_root / "valid.jsonl", "valid", output, image_by_sample)
    write_jsonl(output / "vision_opd/splits/train.jsonl", train)
    write_jsonl(output / "vision_opd/splits/valid.jsonl", valid)
    sealed = sealed_test_ids(split_root / "test.jsonl")
    write_json(output / "vision_opd/splits/test_ids.sealed.json", sealed)

    if not args.spacy_model.is_dir():
        raise FileNotFoundError(args.spacy_model)
    shutil.copytree(args.spacy_model, output / "runtime/en_core_web_sm-3.8.0", dirs_exist_ok=True)

    method_summary: dict[str, Any] = {}
    canonical_opsd = None
    for method, workspace in workspaces.items():
        method_root = output / "methods" / method
        trajectory_root = workspace / "data/trajectories/full/round_00"
        counts: dict[str, int] = {}
        normalized_train = None
        for name in TRAJECTORIES:
            source = trajectory_root / name
            if not source.is_file():
                raise FileNotFoundError(source)
            rows = [portable_trajectory(row, output, method, image_by_sample) for row in read_jsonl(source)]
            write_jsonl(method_root / "trajectories" / name, rows)
            counts[name] = len(rows)
            if name == "train.jsonl":
                normalized_train = rows
        opsd = make_opsd_examples(normalized_train or [])
        write_jsonl(method_root / "opsd_examples.jsonl", opsd)
        if canonical_opsd is None:
            canonical_opsd = opsd
            write_jsonl(output / "common/opsd_examples.jsonl", opsd)
        elif [row["extra_info"]["sample_id"] for row in opsd] != [
            row["extra_info"]["sample_id"] for row in canonical_opsd
        ]:
            raise ValueError(f"OPSD membership differs for {method}")

        negatives = {}
        coco_counts = {}
        for split in ("train", "valid"):
            source_round = workspace / "data/sam_coco/full/round_00"
            negatives[split] = stage_negatives(
                source_round / f"{split}_negatives.jsonl",
                output / "frozen_negatives" / method / f"{split}.jsonl",
            )
            coco_counts[split] = stage_coco(
                source_round / split / "_annotations.coco.json",
                method_root / "sam_coco" / split / "_annotations.coco.json",
                output, image_by_sample, method,
            )
            rejections = source_round / f"{split}_pseudo_bbox_rejections.jsonl"
            if rejections.is_file():
                write_jsonl(
                    method_root / "sam_coco" / f"{split}_pseudo_bbox_rejections.jsonl",
                    [scrub_machine_paths(row) for row in read_jsonl(rejections)],
                )

        results_root = method_root / "reference_results"
        for source in (
            workspace / "runs/full/round_00/metrics.json",
            workspace / "runs/full/round_00/promotion.json",
            workspace / "runs/full/state.json",
        ):
            if source.is_file():
                write_json(results_root / source.name, scrub_machine_paths(json.loads(source.read_text())))
        method_summary[method] = {
            "trajectories": counts, "opsd_examples": len(opsd),
            "frozen_negatives": negatives, "sam_coco": coco_counts,
        }

    card = """---
license: apache-2.0
task_categories:
- visual-question-answering
- object-detection
language:
- en
tags:
- RS-MLLM
- CVSearch
- OPSD
- SAM3
---
# RS-MLLM Self-Evolution Data

Portable train/valid inputs and frozen artifacts for the three Qwen3-VL-2B + SAM3
CVSearch self-evolution variants: tree-node weak box, strict local recovery, and fine
trace-only. The bundle contains no test labels, no merged base weights, and no local
machine paths. See the RS-MLLM repository for provenance, licenses, training commands,
selection rules, and limitations. `test_ids.sealed.json` uses an answer-independent ID.
"""
    (output / "README.md").write_text(card, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "base_models": ["Qwen/Qwen3-VL-2B-Instruct", "facebook/sam3"],
        "split_counts": {"train": len(train), "valid": len(valid), "test_labels": 0, "sealed_test_ids": len(sealed)},
        "method_summary": method_summary,
        "contains_absolute_paths": False,
        "contains_test_answers": False,
    }
    write_json(output / "ASSET_MANIFEST.json", manifest)

    files = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    with (output / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in files:
            handle.write(f"{sha256(path)}  {path.relative_to(output).as_posix()}\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
