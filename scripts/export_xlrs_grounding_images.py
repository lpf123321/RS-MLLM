#!/usr/bin/env python3
"""Export the 844 XLRS train images used by README 3.5 from HF Arrow data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_IMAGES = 844
MAX_EDGE = 4096


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
        "--dataset",
        type=Path,
        default=repo / "datasets" / "XLRS-Bench_visual_grounding_en" / "train",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo
        / "datasets"
        / "XLRS-Bench_visual_grounding_en"
        / "images_exported_train_4096",
    )
    parser.add_argument("--limit", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    expected = args.limit or EXPECTED_IMAGES
    expected_paths = [output / f"xlrs_vg_{index:05d}.jpg" for index in range(expected)]
    ready = all(path.is_file() and path.stat().st_size > 0 for path in expected_paths)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dataset": str(dataset_root),
                    "output": str(output),
                    "expected_images": expected,
                    "ready": ready,
                    "transform": {
                        "max_edge": MAX_EDGE,
                        "resampling": "LANCZOS",
                        "format": "JPEG",
                        "quality": 90,
                    },
                },
                indent=2,
            )
        )
        return
    if ready:
        print(f"XLRS grounding exports already ready: {expected} -> {output}")
        return
    if not (dataset_root / "state.json").is_file():
        raise SystemExit(
            f"missing Hugging Face save_to_disk dataset: {dataset_root}/state.json"
        )

    try:
        from datasets import Dataset
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("datasets and Pillow are required to export XLRS images") from exc

    dataset = Dataset.load_from_disk(str(dataset_root))
    images = dataset.data.column("image")
    output.mkdir(parents=True, exist_ok=True)
    Image.MAX_IMAGE_PIXELS = None
    image_numbers: dict[str, int] = {}
    manifest: list[dict[str, object]] = []
    for row_number in range(len(dataset)):
        value = images[row_number].as_py()
        if not isinstance(value, dict) or not isinstance(value.get("path"), str):
            raise ValueError(f"invalid image field in Arrow row {row_number}")
        original_path = value["path"]
        if original_path in image_numbers:
            continue
        image_number = len(image_numbers)
        image_numbers[original_path] = image_number
        if image_number >= expected:
            break
        destination = output / f"xlrs_vg_{image_number:05d}.jpg"
        payload = value.get("bytes")
        if not destination.is_file() or destination.stat().st_size == 0:
            if not isinstance(payload, (bytes, bytearray, memoryview)):
                raise ValueError(f"missing embedded image bytes in Arrow row {row_number}")
            import io

            with Image.open(io.BytesIO(bytes(payload))) as source:
                image = source.convert("RGB")
                width, height = image.size
                if max(width, height) > MAX_EDGE:
                    scale = MAX_EDGE / max(width, height)
                    size = (int(width * scale), int(height * scale))
                    image = image.resize(size, Image.Resampling.LANCZOS)
                image.save(destination, format="JPEG", quality=90)
                image.close()
        manifest.append(
            {
                "index": image_number,
                "upstream_path": original_path,
                "output": destination.name,
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )

    if len(image_numbers) < expected or not all(path.is_file() for path in expected_paths):
        raise SystemExit(
            f"expected {expected} unique XLRS images, found {len(image_numbers)}"
        )
    (output / "EXPORT_MANIFEST.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "expected_images": expected,
                "max_edge": MAX_EDGE,
                "resampling": "LANCZOS",
                "jpeg_quality": 90,
                "images": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"XLRS grounding exports created: {expected} -> {output}")


if __name__ == "__main__":
    main()
