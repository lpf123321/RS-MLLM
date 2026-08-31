#!/usr/bin/env python3
"""Convert an internal image-complete staging bundle into a publishable thin bundle."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


XLRS_CANONICAL_PREVIEW = {
    "kind": "pillow_jpeg_preview",
    "mode": "RGB",
    "max_edge": 1024,
    "resampling": "LANCZOS",
    "reducing_gap": 2.0,
    "use_draft": False,
    "quality": 90,
    "optimize": False,
}


def transform_blob(blob: bytes, spec: dict[str, object]) -> bytes:
    from PIL import Image

    max_edge = int(spec["max_edge"])
    with Image.open(io.BytesIO(blob)) as source:
        image = source.convert(str(spec["mode"]))
        image.thumbnail(
            (max_edge, max_edge),
            getattr(Image.Resampling, str(spec["resampling"])),
            reducing_gap=float(spec["reducing_gap"]),
        )
        output = io.BytesIO()
        image.save(
            output,
            format="JPEG",
            quality=int(spec["quality"]),
            optimize=bool(spec["optimize"]),
        )
        image.close()
    return output.getvalue()


def canonical_xlrs_previews(
    root: Path,
    questions_by_path: dict[str, str],
) -> dict[str, tuple[str, int, str]]:
    """Return old portable path -> (new hash, bytes, question selector)."""
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to canonicalize XLRS previews") from exc
    wanted = {question: path for path, question in questions_by_path.items()}
    results: dict[str, tuple[str, int, str]] = {}
    scan_root = root / "train" if (root / "train").is_dir() else root
    for arrow_path in sorted(scan_root.rglob("*.arrow")):
        if not wanted:
            break
        print(f"canonicalizing official Arrow shard: {arrow_path}")
        with pa.memory_map(str(arrow_path), "r") as source:
            reader = pa.ipc.open_stream(source)
            question_index = reader.schema.get_field_index("question")
            image_index = reader.schema.get_field_index("image")
            for batch in reader:
                questions = batch.column(question_index).to_pylist()
                images = batch.column(image_index)
                for number, question in enumerate(questions):
                    old_path = wanted.get(str(question))
                    if old_path is None:
                        continue
                    value = images[number].as_py()
                    blob = value.get("bytes") if isinstance(value, dict) else None
                    if not isinstance(blob, bytes):
                        raise ValueError(f"official row has no embedded image bytes: {question!r}")
                    payload = transform_blob(blob, XLRS_CANONICAL_PREVIEW)
                    results[old_path] = (hashlib.sha256(payload).hexdigest(), len(payload), str(question))
                    del wanted[str(question)]
    if wanted:
        raise ValueError(f"failed to locate {len(wanted)} XLRS questions in official train shards")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xlrs-grounding-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if not path.is_file() or relative.parts[0] == "images" or path.name == "SHA256SUMS":
            continue
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)

    manifest_path = output / "ASSET_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["contains_third_party_images"] = False
    manifest["image_materialization_required"] = True
    manifest["required_visibility"] = "private"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    index_path = output / "IMAGE_INDEX.jsonl"
    exp4_path = output / "grounding/exp4_xlrs6611.json"
    exp4 = json.loads(exp4_path.read_text(encoding="utf-8"))
    questions_by_path: dict[str, str] = {}
    for record in exp4:
        image_path = str(record["image"][0])
        question = str(record["conversations"][0]["value"]).removeprefix("<image>\n")
        questions_by_path.setdefault(image_path, question)
    grounding_rows = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if str(row.get("upstream_path", "")).startswith("images_exported_train/"):
            grounding_rows.append(row)
    selected_questions = {
        str(row["path"]): questions_by_path[str(row["path"])] for row in grounding_rows
    }
    canonical = canonical_xlrs_previews(
        args.xlrs_grounding_root.expanduser().resolve(), selected_questions
    )
    path_rewrites = {
        old_path: f"images/{digest[:24]}.jpg"
        for old_path, (digest, _, _) in canonical.items()
    }
    for data_path in sorted((output / "grounding").glob("*.json")):
        records = json.loads(data_path.read_text(encoding="utf-8"))
        for record in records:
            record["image"] = [path_rewrites.get(str(value), str(value)) for value in record["image"]]
        data_path.write_text(json.dumps(records, ensure_ascii=False) + "\n", encoding="utf-8")

    rewritten_by_path: dict[str, dict[str, object]] = {}
    transformed = 0
    canonicalized = 0
    for line in index_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        upstream_path = str(row.get("upstream_path", ""))
        old_path = str(row["path"])
        if upstream_path.startswith("images_exported_train/"):
            digest, byte_count, question = canonical[old_path]
            row["historical_sha256"] = row["sha256"]
            row["historical_bytes"] = row["bytes"]
            row["sha256"] = digest
            row["bytes"] = byte_count
            row["path"] = path_rewrites[old_path]
            row["transform"] = XLRS_CANONICAL_PREVIEW
            row["source_selector"] = {"field": "question", "value": question}
            canonicalized += 1
        elif row.get("upstream_dataset") == "xlrs" and not upstream_path.startswith("xlrs_"):
            row["upstream_dataset"] = "xlrs_grounding"
            row["transform"] = {
                "kind": "pillow_jpeg_preview",
                "mode": "RGB",
                "max_edge": 1280,
                "resampling": "LANCZOS",
                "quality": 82,
                "optimize": True,
            }
            transformed += 1
        rewritten_by_path.setdefault(str(row["path"]), row)
    if transformed != 180:
        raise ValueError(f"expected 180 XLRS gallery previews in retained bundle, found {transformed}")
    if canonicalized != len(grounding_rows):
        raise ValueError(f"expected {len(grounding_rows)} canonical XLRS previews, found {canonicalized}")
    with index_path.open("w", encoding="utf-8") as handle:
        for row in sorted(rewritten_by_path.values(), key=lambda value: str(value["path"])):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest["datasets"]["grounding_exp4"]["portable_sha256"] = sha256(exp4_path)
    manifest["datasets"]["grounding_exp5"]["portable_sha256"] = sha256(
        output / "grounding/exp5_all43838.json"
    )
    manifest["canonicalized_xlrs_previews"] = canonicalized
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# RS-MLLM Distillation Data (thin bundle)\n\n"
        "This repository stores retained training annotations and an auditable `IMAGE_INDEX.jsonl`; "
        "it intentionally does not redistribute upstream images. Use "
        "`training/distillation/expert_lora/scripts/download_official_datasets.sh` from the code "
        "repository to download official sources and reconstruct every image by SHA-256.\n",
        encoding="utf-8",
    )
    (output / "LICENSES.md").write_text(
        "# Upstream data terms\n\n"
        "No upstream image is redistributed in this thin bundle. Reproducers obtain data from "
        "the official repositories and must follow their terms. MME-RealWorld is academic-only "
        "and forbids redistribution without prior approval; VRSBench includes academic-use-only "
        "DOTA imagery; XLRS image terms vary by source and Toronto imagery may not be distributed "
        "to third parties.\n",
        encoding="utf-8",
    )
    files = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    with (output / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in files:
            handle.write(f"{sha256(path)}  {path.relative_to(output).as_posix()}\n")
    print(json.dumps({"output": str(output), "files": len(files), "bytes": sum(p.stat().st_size for p in files)}))


if __name__ == "__main__":
    main()
