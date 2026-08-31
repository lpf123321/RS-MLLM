#!/usr/bin/env python3
"""Stage the five retained Expert datasets with an auditable image index.

Images are omitted by default because the upstream datasets already publish
them and some of their licenses prohibit third-party redistribution.  Pass
``--include-images`` only for an access-controlled internal backup after a
license review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


SPECS = {
    "general/exp3_mme3736_xlrs3080.json": ("general_exp3", "mcq", 6816),
    "grounding/bootstrap_vrsnew942.json": ("grounding_bootstrap_942", "vrs_grounding", 942),
    "grounding/exp1_washed_vrs36285.json": ("grounding_exp1", "vrs_grounding", 36285),
    "grounding/exp4_xlrs6611.json": ("grounding_exp4", "xlrs_grounding", 6611),
    "grounding/exp5_all43838.json": ("grounding_exp5", "mixed_grounding", 43838),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"expected JSON array: {path}")
    return value


XLRS_GALLERY_TRANSFORM = {
    "kind": "pillow_jpeg_preview",
    "mode": "RGB",
    "max_edge": 1280,
    "resampling": "LANCZOS",
    "quality": 82,
    "optimize": True,
}


def classify(path: Path, roots: dict[str, Path]) -> tuple[str, str, dict[str, Any] | None]:
    resolved = path.resolve()
    text = resolved.as_posix().lower()
    if "sam3_grounding_skill_xlrs" in text and "/final_clean/assets/" in text:
        return "xlrs_grounding", resolved.name, XLRS_GALLERY_TRANSFORM
    for name, root in roots.items():
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        return name, relative.as_posix(), None
    if "xlrs" in text:
        return "xlrs", resolved.name, None
    if "vrsbench" in text or "sam3_grounding_skill_5000" in text:
        return "vrsbench", resolved.name, None
    if "mme" in text:
        return "mme_realworld_rs", resolved.name, None
    raise ValueError(f"cannot assign upstream dataset provenance: {resolved}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--general", type=Path, required=True)
    parser.add_argument("--grounding-bootstrap", type=Path, required=True)
    parser.add_argument("--grounding-exp1", type=Path, required=True)
    parser.add_argument("--grounding-exp4", type=Path, required=True)
    parser.add_argument("--grounding-exp5", type=Path, required=True)
    parser.add_argument("--mme-root", type=Path, required=True)
    parser.add_argument("--vrsbench-root", type=Path, required=True)
    parser.add_argument("--xlrs-lite-root", type=Path, required=True)
    parser.add_argument("--xlrs-grounding-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-images",
        action="store_true",
        help="copy image blobs into the bundle (off by default; requires license review)",
    )
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    sources = {
        "general/exp3_mme3736_xlrs3080.json": args.general.expanduser().resolve(),
        "grounding/bootstrap_vrsnew942.json": args.grounding_bootstrap.expanduser().resolve(),
        "grounding/exp1_washed_vrs36285.json": args.grounding_exp1.expanduser().resolve(),
        "grounding/exp4_xlrs6611.json": args.grounding_exp4.expanduser().resolve(),
        "grounding/exp5_all43838.json": args.grounding_exp5.expanduser().resolve(),
    }
    roots = {
        "mme_realworld_rs": args.mme_root.expanduser().resolve(),
        "vrsbench": args.vrsbench_root.expanduser().resolve(),
        "xlrs_lite": args.xlrs_lite_root.expanduser().resolve(),
        "xlrs_grounding": args.xlrs_grounding_root.expanduser().resolve(),
    }
    path_cache: dict[Path, str] = {}
    content_cache: dict[str, str] = {}
    image_index: dict[str, dict[str, Any]] = {}
    dataset_manifest: dict[str, Any] = {}

    for relative, source in sources.items():
        name, schema, expected = SPECS[relative]
        rows = load(source)
        if len(rows) != expected:
            raise ValueError(f"{name}: expected {expected} rows, got {len(rows)}")
        portable = json.loads(json.dumps(rows, ensure_ascii=False))
        for row in portable:
            images = row.get("image")
            if not isinstance(images, list) or not images:
                raise ValueError(f"{name}: row {row.get('id')} has no image list")
            rewritten = []
            for value in images:
                path = Path(str(value)).expanduser().resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
                if path not in path_cache:
                    digest = sha256(path)
                    suffix = path.suffix.lower() or ".bin"
                    destination_relative = content_cache.get(digest)
                    if destination_relative is None:
                        destination_relative = f"images/{digest[:24]}{suffix}"
                        if args.include_images:
                            destination = output / destination_relative
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(path, destination)
                        content_cache[digest] = destination_relative
                    upstream_dataset, upstream_path, transform = classify(path, roots)
                    path_cache[path] = destination_relative
                    index_row = {
                        "sha256": digest,
                        "bytes": path.stat().st_size,
                        "upstream_dataset": upstream_dataset,
                        "upstream_path": upstream_path,
                    }
                    if transform is not None:
                        index_row["transform"] = transform
                    image_index.setdefault(destination_relative, index_row)
                rewritten.append(path_cache[path])
            row["image"] = rewritten
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(portable, ensure_ascii=False) + "\n", encoding="utf-8")
        dataset_manifest[name] = {
            "path": relative, "schema": schema, "records": len(portable),
            "retained_source_sha256": sha256(source), "portable_sha256": sha256(destination),
        }

    index_path = output / "IMAGE_INDEX.jsonl"
    with index_path.open("w", encoding="utf-8") as handle:
        for relative, row in sorted(image_index.items()):
            handle.write(json.dumps({"path": relative, **row}, ensure_ascii=False, sort_keys=True) + "\n")

    (output / "LICENSES.md").write_text(
        "# Upstream data licenses\n\n"
        "This bundle does not redistribute upstream images. Reproducers must obtain them from "
        "the official repositories and comply with every upstream condition.\n\n"
        "- MME-RealWorld permits academic research only and forbids redistribution without prior approval.\n"
        "- VRSBench text is CC BY 4.0; some DOTA imagery is academic-use-only.\n"
        "- XLRS-Bench annotations are CC BY-NC 4.0; image licenses vary by source, and Toronto "
        "imagery must not be redistributed to third parties.\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# RS-MLLM Distillation Data\n\n"
        "Exact retained General/Grounding Expert training records with content-addressed image "
        "paths. Upstream images are intentionally omitted. Run the repository's "
        "`download_official_datasets.sh` and `materialize_images.py` workflow to obtain and "
        "hash-verify them. The five source JSON SHA256 values are preserved in "
        "`ASSET_MANIFEST.json`; see `LICENSES.md`.\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1, "datasets": dataset_manifest,
        "unique_source_paths": len(path_cache), "unique_image_blobs": len(content_cache),
        "image_index": "IMAGE_INDEX.jsonl", "contains_absolute_paths": False,
        "contains_third_party_images": args.include_images,
        "image_materialization_required": not args.include_images,
        "required_visibility": "private",
    }
    (output / "ASSET_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    files = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    with (output / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in files:
            handle.write(f"{sha256(path)}  {path.relative_to(output).as_posix()}\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
