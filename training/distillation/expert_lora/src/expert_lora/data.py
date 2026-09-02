"""Validation and path normalization for the retained Expert SFT datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable


MCQ_ANSWER = re.compile(r"^[A-E]+$")
VRS_BOX = re.compile(r"^\{<(\d+)><(\d+)><(\d+)><(\d+)>\}$")
XLRS_BOX = re.compile(
    r"^\[((?:0|1)\.\d{5}),((?:0|1)\.\d{5}),"
    r"((?:0|1)\.\d{5}),((?:0|1)\.\d{5})\]$"
)
SCHEMAS = {"mcq", "general_mixed", "vrs_grounding", "xlrs_grounding", "mixed_grounding"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_records(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"expected a JSON array of objects: {path}")
    if not value:
        raise ValueError(f"dataset is empty: {path}")
    return value


def resolve_image_path(
    image: str,
    image_root: Path,
    *,
    allow_absolute: bool = False,
) -> Path:
    path = Path(image)
    if path.is_absolute():
        if not allow_absolute:
            raise ValueError(f"absolute image path is not portable: {image}")
        return path
    return image_root / path


def _validate_answer(answer: str, schema: str) -> None:
    if schema in {"mcq", "general_mixed"}:
        if not MCQ_ANSWER.fullmatch(answer):
            if schema == "mcq":
                raise ValueError(f"invalid MCQ answer: {answer!r}")
            # Exp7 also contains free-form VRSBench VQA answers.
        return

    vrs_match = VRS_BOX.fullmatch(answer)
    xlrs_match = XLRS_BOX.fullmatch(answer)
    if schema == "vrs_grounding" and not vrs_match:
        raise ValueError(f"invalid VRS grounding answer: {answer!r}")
    if schema == "xlrs_grounding" and not xlrs_match:
        raise ValueError(f"invalid fixed-five XLRS answer: {answer!r}")
    if schema == "mixed_grounding" and not (vrs_match or xlrs_match):
        raise ValueError(f"invalid mixed grounding answer: {answer!r}")

    if vrs_match:
        x1, y1, x2, y2 = map(int, vrs_match.groups())
        if not (0 <= x1 < x2 <= 100 and 0 <= y1 < y2 <= 100):
            raise ValueError(f"invalid VRS grounding box: {answer!r}")
    if xlrs_match:
        x1, y1, x2, y2 = map(float, xlrs_match.groups())
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise ValueError(f"invalid XLRS grounding box: {answer!r}")


def validate_records(
    records: Iterable[dict],
    *,
    schema: str,
    image_root: Path,
    check_images: bool = True,
    allow_absolute: bool = False,
) -> dict:
    if schema not in SCHEMAS:
        raise ValueError(f"unknown schema {schema!r}; choose from {sorted(SCHEMAS)}")

    errors: list[str] = []
    ids: set[str] = set()
    image_paths: set[Path] = set()
    rows = list(records)
    for index, row in enumerate(rows):
        try:
            row_id = row.get("id")
            if not isinstance(row_id, str) or not row_id:
                raise ValueError("id must be a non-empty string")
            if row_id in ids:
                raise ValueError(f"duplicate id: {row_id}")
            ids.add(row_id)

            images = row.get("image")
            if not isinstance(images, list) or not images or not all(
                isinstance(image, str) and image for image in images
            ):
                raise ValueError("image must be a non-empty list of strings")

            conversations = row.get("conversations")
            if not isinstance(conversations, list) or len(conversations) != 2:
                raise ValueError("conversations must contain exactly two messages")
            if [item.get("from") for item in conversations] != ["human", "gpt"]:
                raise ValueError("conversation roles must be human then gpt")
            prompt = conversations[0].get("value")
            answer = conversations[1].get("value")
            if not isinstance(prompt, str) or not isinstance(answer, str) or not answer.strip():
                raise ValueError("prompt and answer must be non-empty strings")
            if prompt.count("<image>") != len(images):
                raise ValueError(
                    f"image-token mismatch: {len(images)} images but "
                    f"{prompt.count('<image>')} tokens"
                )
            _validate_answer(answer.strip(), schema)

            weight = float(row.get("loss_weight", 1.0))
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("loss_weight must be finite and positive")

            for image in images:
                resolved = resolve_image_path(
                    image, image_root, allow_absolute=allow_absolute
                )
                image_paths.add(resolved)
                if check_images and not resolved.is_file():
                    raise FileNotFoundError(resolved)
        except (KeyError, TypeError, ValueError, FileNotFoundError) as error:
            errors.append(f"row {index}: {error}")

    return {
        "records": len(rows),
        "unique_ids": len(ids),
        "unique_images": len(image_paths),
        "schema": schema,
        "portable_paths": not any(
            Path(path).is_absolute()
            for row in rows
            for path in row.get("image", [])
        ),
        "passed": not errors,
        "errors": errors[:100],
    }


def _parse_mappings(values: list[str]) -> list[tuple[str, str]]:
    mappings: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"mapping must use OLD=NEW syntax: {value!r}")
        old, new = value.split("=", 1)
        old = old.rstrip("/")
        new = new.strip("/")
        if not old or not new or Path(new).is_absolute():
            raise ValueError(f"invalid portable mapping: {value!r}")
        mappings.append((old, new))
    return sorted(mappings, key=lambda item: len(item[0]), reverse=True)


def normalize_paths(records: list[dict], mappings: list[tuple[str, str]]) -> list[dict]:
    normalized = json.loads(json.dumps(records, ensure_ascii=False))
    for index, row in enumerate(normalized):
        rewritten = []
        for image in row.get("image", []):
            if not Path(image).is_absolute():
                rewritten.append(image)
                continue
            for old, new in mappings:
                if image == old or image.startswith(old + "/"):
                    suffix = image[len(old) :].lstrip("/")
                    rewritten.append(str(Path(new) / suffix))
                    break
            else:
                raise ValueError(f"row {index}: no mapping for absolute path {image}")
        row["image"] = rewritten
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--data", required=True, type=Path)
    validate.add_argument("--schema", required=True, choices=sorted(SCHEMAS))
    validate.add_argument("--image-root", required=True, type=Path)
    validate.add_argument("--expected-records", type=int)
    validate.add_argument("--skip-image-check", action="store_true")
    validate.add_argument("--allow-absolute-paths", action="store_true")

    normalize = subparsers.add_parser("normalize")
    normalize.add_argument("--input", required=True, type=Path)
    normalize.add_argument("--output", required=True, type=Path)
    normalize.add_argument("--map", action="append", default=[], metavar="OLD=NEW")
    normalize.add_argument("--manifest", type=Path)
    normalize.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate":
        records = load_records(args.data)
        report = validate_records(
            records,
            schema=args.schema,
            image_root=args.image_root,
            check_images=not args.skip_image_check,
            allow_absolute=args.allow_absolute_paths,
        )
        report["data"] = str(args.data)
        report["sha256"] = sha256(args.data)
        if args.expected_records is not None and len(records) != args.expected_records:
            report["passed"] = False
            report["errors"].append(
                f"expected {args.expected_records} records, found {len(records)}"
            )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["passed"]:
            raise SystemExit(1)
        return

    if args.output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    mappings = _parse_mappings(args.map)
    records = load_records(args.input)
    normalized = normalize_paths(records, mappings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(normalized, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "records": len(normalized),
        "input_sha256": sha256(args.input),
        "output_sha256": sha256(args.output),
        "mappings": [{"old": old, "new": new} for old, new in mappings],
    }
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
