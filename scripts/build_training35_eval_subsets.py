#!/usr/bin/env python3
"""Build deterministic fixed-ID evaluation subsets for README 3.5 acceptance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SPECS = {
    "mme": ("MME-RealWorld-RS/mme_rs.jsonl", None),
    "xlrs": ("XLRS-Bench-lite/xlrs.jsonl", None),
    "vrsbench_vqa": ("VRSBench/vrsbench_eval.jsonl", "[VQA]"),
    "vrsbench_referring": ("VRSBench/vrsbench_eval.jsonl", "[REF]"),
}


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sample_id(record: dict[str, object]) -> str:
    explicit = record.get("id")
    if isinstance(explicit, str) and explicit:
        return explicit
    return hashlib.sha256(canonical(record).encode()).hexdigest()


def user_text(record: dict[str, object]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("record has no messages")
    content = messages[0].get("content")
    if not isinstance(content, list):
        raise ValueError("user content must be a list")
    for item in reversed(content):
        if isinstance(item, dict) and item.get("type") == "text":
            return str(item.get("text", ""))
    raise ValueError("record has no user text")


def normalize_record(record: dict[str, object]) -> dict[str, object]:
    if "messages" in record:
        return record
    identifier = record.get("id") or record.get("question_id")
    image = record.get("image")
    question = record.get("question")
    bbox = record.get("bbox")
    if not identifier or not image or not question or not isinstance(bbox, list):
        raise ValueError("legacy grounding record is missing id/image/question/bbox")
    coordinates = [
        f"{float(value):.16f}".rstrip("0").rstrip(".") or "0"
        for value in bbox
    ]
    answer = "[" + ",".join(coordinates) + "]"
    return {
        "id": str(identifier),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image)},
                    {"type": "text", "text": str(question)},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ],
    }


def load_jsonl(path: Path, prefix: str | None) -> list[tuple[int, dict[str, object]]]:
    selected = []
    sources = sorted(path.glob("part*.jsonl")) if path.is_dir() else [path]
    if not sources:
        raise FileNotFoundError(f"no JSONL source found under {path}")
    global_line = 0
    for source in sources:
        with source.open(encoding="utf-8") as handle:
            for source_line, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                global_line += 1
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{source}:{source_line}: expected object")
                record = normalize_record(record)
                if prefix is None or user_text(record).startswith(prefix):
                    selected.append((global_line, record))
    return selected


def load_xlrs_arrow(path: Path) -> list[tuple[int, dict[str, object]]]:
    from datasets import Dataset

    dataset = Dataset.load_from_disk(str(path))
    table = dataset.data
    image_paths = table.column("image").flatten()[1]
    identifiers = table.column("question_id")
    questions = table.column("question")
    bboxes = table.column("bbox")
    image_root = path.parent / "images_exported_test_4096"
    image_numbers: dict[str, int] = {}
    selected = []
    for index in range(len(dataset)):
        original_path = str(image_paths[index].as_py())
        if original_path not in image_numbers:
            image_numbers[original_path] = len(image_numbers)
        image = image_root / f"xlrs_vg_{image_numbers[original_path]:05d}.jpg"
        record = normalize_record(
            {
                "id": identifiers[index].as_py(),
                "image": str(image),
                "question": questions[index].as_py(),
                "bbox": bboxes[index].as_py(),
            }
        )
        selected.append((index + 1, record))
    return selected


def load_source(path: Path, prefix: str | None) -> list[tuple[int, dict[str, object]]]:
    if path.is_dir() and (path / "state.json").is_file():
        if prefix is not None:
            raise ValueError("Arrow grounding source does not support task prefixes")
        return load_xlrs_arrow(path)
    return load_jsonl(path, prefix)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shared-root",
        type=Path,
        default=repo / "datasets" / "shared_datasets",
    )
    parser.add_argument("--xlrs-grounding", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo / "outputs" / "training35" / "eval_subsets",
    )
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive")

    sources = {
        name: (args.shared_root / relative, prefix)
        for name, (relative, prefix) in SPECS.items()
    }
    sources["xlrs_grounding"] = (args.xlrs_grounding, None)
    args.output.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "seed": args.seed,
        "requested_per_dataset": args.samples,
        "subsets": {},
    }
    for name, (source, prefix) in sources.items():
        rows = load_source(source.resolve(), prefix)
        ranked = sorted(
            enumerate(rows),
            key=lambda item: hashlib.sha256(
                f"{args.seed}:{sample_id(item[1][1])}".encode()
            ).hexdigest(),
        )
        chosen = ranked[: min(args.samples, len(ranked))]
        missing_images = []
        for _, (_, record) in chosen:
            content = record["messages"][0]["content"]
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    image = Path(str(item["image"])).expanduser()
                    if not image.is_file():
                        missing_images.append(str(image))
        if missing_images:
            raise FileNotFoundError(
                f"{name}: {len(missing_images)} selected image(s) are missing; "
                f"first: {missing_images[:3]}"
            )
        output = args.output / f"{name}_{len(chosen)}.jsonl"
        with output.open("w", encoding="utf-8") as handle:
            for _, (_, record) in chosen:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        ids = [
            {
                "sample_id": sample_id(record),
                "eligible_index": eligible_index,
                "source_line": line_number,
                "selection_sha256": hashlib.sha256(
                    f"{args.seed}:{sample_id(record)}".encode()
                ).hexdigest(),
            }
            for eligible_index, (line_number, record) in chosen
        ]
        ids_path = args.output / f"{name}_{len(chosen)}.ids.json"
        ids_path.write_text(json.dumps(ids, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report["subsets"][name] = {
            "source": str(source.resolve()),
            "eligible": len(rows),
            "selected": len(chosen),
            "jsonl": str(output.resolve()),
            "jsonl_sha256": sha256(output),
            "ids": str(ids_path.resolve()),
            "ids_sha256": sha256(ids_path),
        }
    manifest = args.output / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
