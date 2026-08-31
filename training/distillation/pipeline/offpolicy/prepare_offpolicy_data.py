#!/usr/bin/env python3
"""Join cached teacher trajectories to the formal OPD data and emit VERL parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


PIPELINE_ROOT = Path(os.environ.get("PIPELINE_ROOT", Path(__file__).resolve().parents[1]))
ARTIFACT_ROOT = Path(os.environ.get("ARTIFACT_ROOT", PIPELINE_ROOT / "artifacts"))


SYSTEM_PROMPTS = {
    "vrsbench": "Obey the task prefix:\n- [VQA] Answer with a single word or short phrase only. No extra text.\n- [CAP] Describe the image in detail.\n- [REF] Output ONLY the bounding box in format {<x1><y1><x2><y2>} with integer coordinates 0-99, e.g. {<25><40><33><60>}. No other text.",
    "mme": "Answer EXACTLY in format \"X. (X) FullOptionText\" with the letter repeated in parentheses. Example: \"D. (D) White\". You MUST include the parenthesized letter - never omit it. Output ONLY that line.",
    "xlrs": "Answer EXACTLY in format \"X. (X) FullOptionText\" with the letter repeated in parentheses. Example: \"A. (A) Some description\". You MUST include the parenthesized letter - never omit it. Output ONLY that line.",
    "levircc": "Describe the changes between the two images concisely in 1-2 sentences.",
}

PREFIX_RE = re.compile(r"^(?:<image>\n)+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=PIPELINE_ROOT / "data" / "train" / "unified_sft.json",
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=ARTIFACT_ROOT / "offpolicy" / "trajectories" / "full_sft_vision_opd_9b",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_ROOT / "offpolicy" / "data" / "formal_teacher_trajectories_18413.parquet",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PIPELINE_ROOT / "manifests" / "offpolicy_formal_manifest.json",
    )
    parser.add_argument("--expected-records", type=int, default=18_413)
    parser.add_argument("--student-model", type=Path, default=Path(os.environ.get("STUDENT_MODEL", "student-model")))
    parser.add_argument(
        "--teacher-model",
        type=Path,
        default=ARTIFACT_ROOT / "sft" / "full-vision-opd-9b-sft-merged",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_without_image_tokens(prompt: str) -> str:
    return PREFIX_RE.sub("", prompt)


def load_trajectories(directory: Path) -> tuple[dict[str, dict[str, Any]], list[Path]]:
    paths = sorted(
        path
        for path in directory.glob("*.jsonl")
        if not path.name.endswith("predictions.jsonl")
    )
    if not paths:
        raise FileNotFoundError(f"No trajectory JSONL shards found in {directory}")

    trajectories: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                record = json.loads(line)
                trajectory_id = record["trajectory_id"]
                if trajectory_id in trajectories:
                    raise ValueError(f"Duplicate trajectory_id {trajectory_id!r} in {path}:{line_number}")
                trajectories[trajectory_id] = record
    return trajectories, paths


def assert_tokenizers_identical(student_model: Path, teacher_model: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    student = AutoTokenizer.from_pretrained(student_model, trust_remote_code=True)
    teacher = AutoTokenizer.from_pretrained(teacher_model, trust_remote_code=True)
    student_vocab = student.get_vocab()
    teacher_vocab = teacher.get_vocab()
    if student_vocab != teacher_vocab:
        only_student = sorted(set(student_vocab) - set(teacher_vocab))[:8]
        only_teacher = sorted(set(teacher_vocab) - set(student_vocab))[:8]
        raise ValueError(
            "Teacher/student vocabularies differ; cached teacher token ids cannot be replayed safely. "
            f"only_student={only_student}, only_teacher={only_teacher}"
        )
    payload = json.dumps(sorted(student_vocab.items()), ensure_ascii=False, separators=(",", ":"))
    return {
        "vocab_size": len(student_vocab),
        "vocab_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "student_tokenizer": str(student_model.resolve()),
        "teacher_tokenizer": str(teacher_model.resolve()),
    }


def main() -> None:
    args = parse_args()
    source_records = json.loads(args.source.read_text(encoding="utf-8"))
    if len(source_records) != args.expected_records:
        raise ValueError(f"Expected {args.expected_records} source records, found {len(source_records)}")
    trajectories, trajectory_paths = load_trajectories(args.trajectory_dir)
    if len(trajectories) != args.expected_records:
        raise ValueError(f"Expected {args.expected_records} unique trajectories, found {len(trajectories)}")

    tokenizer_audit = assert_tokenizers_identical(args.student_model, args.teacher_model)
    vocab_size = tokenizer_audit["vocab_size"]
    rows = []
    counts: Counter[str] = Counter()
    response_lengths = []
    source_ids = set()

    for row_index, source in enumerate(source_records):
        trajectory_id = source["id"]
        if trajectory_id in source_ids:
            raise ValueError(f"Duplicate source id: {trajectory_id}")
        source_ids.add(trajectory_id)
        trajectory = trajectories.get(trajectory_id)
        if trajectory is None:
            raise KeyError(f"Missing teacher trajectory: {trajectory_id}")

        dataset = source["source_dataset"]
        if trajectory["dataset"] != dataset:
            raise ValueError(f"Dataset mismatch for {trajectory_id}")
        images = source["image"] if isinstance(source["image"], list) else [source["image"]]
        if images != trajectory["images"]:
            raise ValueError(f"Image mismatch for {trajectory_id}")
        prompt = source["conversations"][0]["value"]
        if prompt_without_image_tokens(prompt) != trajectory["prompt"]:
            raise ValueError(f"Prompt mismatch for {trajectory_id}")

        response_ids = [int(item) for item in trajectory["teacher_response_ids"]]
        if not response_ids:
            raise ValueError(f"Empty teacher response ids for {trajectory_id}")
        invalid_ids = [item for item in response_ids if item < 0 or item >= vocab_size]
        if invalid_ids:
            raise ValueError(f"Invalid teacher token ids for {trajectory_id}: {invalid_ids[:8]}")
        if len(response_ids) > 64:
            raise ValueError(f"Teacher response exceeds 64 tokens for {trajectory_id}: {len(response_ids)}")

        answer = source["conversations"][1]["value"]
        image_structs = [{"path": image_path} for image_path in images]
        rows.append(
            {
                "data_source": "offpolicy_fixed_teacher_reward_free",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPTS[dataset]},
                    {"role": "user", "content": prompt},
                ],
                "images": image_structs,
                "bbox_images": image_structs,
                "ability": dataset,
                "reward_model": {"ground_truth": answer, "style": "none"},
                "extra_info": {
                    "answer": answer,
                    "question": prompt,
                    "index": row_index,
                    "trajectory_id": trajectory_id,
                    "source_extra_info": {"answer": answer, "question": prompt},
                },
                "trajectory_id": trajectory_id,
                "teacher_response": trajectory["teacher_response"],
                "teacher_response_ids": response_ids,
            }
        )
        counts[dataset] += 1
        response_lengths.append(len(response_ids))

    extras = set(trajectories) - source_ids
    if extras:
        raise ValueError(f"Found trajectories not present in source: {sorted(extras)[:8]}")

    message = pa.struct([("content", pa.string()), ("role", pa.string())])
    image = pa.struct([("path", pa.string())])
    source_extra = pa.struct([("answer", pa.string()), ("question", pa.string())])
    schema = pa.schema(
        [
            ("data_source", pa.string()),
            ("prompt", pa.list_(message)),
            ("images", pa.list_(image)),
            ("bbox_images", pa.list_(image)),
            ("ability", pa.string()),
            ("reward_model", pa.struct([("ground_truth", pa.string()), ("style", pa.string())])),
            (
                "extra_info",
                pa.struct(
                    [
                        ("answer", pa.string()),
                        ("question", pa.string()),
                        ("index", pa.int64()),
                        ("trajectory_id", pa.string()),
                        ("source_extra_info", source_extra),
                    ]
                ),
            ),
            ("trajectory_id", pa.string()),
            ("teacher_response", pa.string()),
            ("teacher_response_ids", pa.list_(pa.int64())),
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), args.output, compression="zstd")

    manifest = {
        "schema_version": 1,
        "definition": "teacher-generated trajectories; teacher and student distributions evaluated on identical teacher prefixes",
        "source": str(args.source.resolve()),
        "source_sha256": sha256_file(args.source),
        "trajectory_shards": [str(path.resolve()) for path in trajectory_paths],
        "trajectory_shard_sha256": {str(path.resolve()): sha256_file(path) for path in trajectory_paths},
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "records": len(rows),
        "unique_trajectory_ids": len(source_ids),
        "counts": dict(sorted(counts.items())),
        "response_tokens": {
            "min": min(response_lengths),
            "max": max(response_lengths),
            "mean": sum(response_lengths) / len(response_lengths),
        },
        "trajectory_source": "fixed full-data-SFT Vision-OPD-9B teacher",
        "teacher_image_policy": "same source images as student",
        "order_policy": "identical to unified_sft.json",
        "tokenizer_audit": tokenizer_audit,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
