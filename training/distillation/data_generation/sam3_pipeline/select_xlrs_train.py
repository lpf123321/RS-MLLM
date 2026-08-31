#!/usr/bin/env python3
"""Inspect XLRS Arrow train and materialize a train-only construction set."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pyarrow as pa

try:
    from .common import object_classes, safe_id, sha256, write_jsonl
except ImportError:  # direct script execution
    from common import object_classes, safe_id, sha256, write_jsonl


def shard_paths(split_dir: Path) -> list[Path]:
    state = json.loads((split_dir / "state.json").read_text(encoding="utf-8"))
    return [split_dir / item["filename"] for item in state["_data_files"]]


def iter_rows(split_dir: Path, wanted_images: set[str] | None = None):
    global_index = 0
    for shard in shard_paths(split_dir):
        with pa.memory_map(str(shard), "r") as source:
            reader = pa.ipc.open_stream(source)
            for batch in reader:
                names = {name: batch.schema.get_field_index(name) for name in batch.schema.names}
                images = batch.column(names["image"])
                paths = images.field("path")
                payloads = images.field("bytes")
                for index in range(batch.num_rows):
                    image = Path(paths[index].as_py()).name
                    if wanted_images is None or image in wanted_images:
                        yield {
                            "row_index": global_index,
                            "image": image,
                            "image_bytes": payloads[index].as_py() if wanted_images is not None else None,
                            "image_width": int(round(batch.column(names["image_width"])[index].as_py())),
                            "image_height": int(round(batch.column(names["image_height"])[index].as_py())),
                            "question_id": batch.column(names["question_id"])[index].as_py(),
                            "question": batch.column(names["question"])[index].as_py(),
                            "answer": batch.column(names["answer"])[index].as_py(),
                            "bbox": batch.column(names["bbox"])[index].as_py(),
                            "category": batch.column(names["category"])[index].as_py(),
                            "arrow_shard": str(shard.resolve()),
                        }
                    global_index += 1


def infer_class(text: str) -> str | None:
    first = text.lower().split(".", 1)[0]
    rules = [
        ("swimming-pool", ("swimming pool", " pool")),
        ("tennis-court", ("tennis court",)),
        ("basketball-court", ("basketball court",)),
        ("soccer-field", ("soccer field", "football field")),
        ("sports-field", ("sports field", "sports area", "sports zone", "playground")),
        ("parking-lot", ("parking lot", "parking area")),
        ("ship", (" boat", "ship", "vessel")),
        ("vehicle", (" car", "vehicle", " truck", " sedan", " van")),
        ("lake", (" lake", " pond")),
        ("lawn", (" lawn", "green space", "grassy area", "grass patch")),
        ("park", (" park", " plaza", "square")),
        ("settlement", (" town", "village", "settlement")),
        ("farmland", ("farmland", " field", "grassland")),
        ("forest", (" forest", "woodland")),
        ("road", (" road", "intersection", "roundabout", "highway")),
        ("building", ("building", " house", " roof", "rooftop", "structure")),
    ]
    matches = []
    for order, (label, terms) in enumerate(rules):
        positions = [first.find(term) for term in terms if first.find(term) >= 0]
        if positions:
            matches.append((min(positions), order, label))
    return min(matches)[2] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    train_dir, out = Path(args.train_dir), Path(args.output_dir)
    train_meta = list(iter_rows(train_dir))
    grouped = defaultdict(list)
    for row in train_meta:
        grouped[row["image"]].append(row)
    ranked = sorted(grouped, key=lambda image: hashlib.sha256(f"{args.seed}:{image}".encode()).hexdigest())
    selected = ranked[: args.num_images]
    selected_set = set(selected)
    out.mkdir(parents=True, exist_ok=True)
    image_root = out / "images"
    image_root.mkdir(exist_ok=True)

    first_payload = {}
    for row in iter_rows(train_dir, selected_set):
        first_payload.setdefault(row["image"], row["image_bytes"])
    if set(first_payload) != selected_set:
        raise SystemExit("failed to recover every selected Arrow image")
    for image, payload in first_payload.items():
        (image_root / image).write_bytes(payload)

    official = []
    selection = []
    for image in selected:
        records = []
        for row in grouped[image]:
            x1, y1, x2, y2 = [float(value) for value in row["bbox"]]
            record = {key: value for key, value in row.items() if key != "image_bytes"}
            record["bbox_xyxy_px"] = [x1 * row["image_width"], y1 * row["image_height"], x2 * row["image_width"], y2 * row["image_height"]]
            record["inferred_class"] = infer_class(row["answer"])
            records.append(record)
            official.append(record)
        selection.append({
            "image": image,
            "image_path": str((image_root / image).resolve()),
            "official_record_indices": [record["row_index"] for record in records],
            "official_records": len(records),
        })
    write_jsonl(out / "selected_images.jsonl", selection)
    write_jsonl(out / "official_records.jsonl", official)
    profile = {
        "dataset_profile": "xlrs", "split": "train", "train_dir": str(train_dir.resolve()),
        "train_records": len(train_meta), "train_unique_images": len(grouped),
        "split_isolation_mode": "generation reads train Arrow only; no test path or test column is opened",
        "selected_images": len(selection),
        "selected_official_records": len(official), "allowed_classes": list(object_classes("xlrs")),
        "question_contract": "official prefix ending in Description: plus unchanged generated inner description",
        "answer_contract": "[xmin, ymin, xmax, ymax] absolute pixels",
        "source": "embedded highest-resolution Arrow train bytes",
        "train_state_sha256": sha256(train_dir / "state.json"),
        "test_split_opened": False,
    }
    (out / "format_profile.json").write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
