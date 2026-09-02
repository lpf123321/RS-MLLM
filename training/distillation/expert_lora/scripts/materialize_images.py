#!/usr/bin/env python3
"""Rebuild content-addressed images from user-downloaded official datasets."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_source(value: str) -> tuple[str, Path]:
    name, separator, root = value.partition("=")
    if not separator or not name or not root:
        raise argparse.ArgumentTypeError("--source must be NAME=/absolute/or/relative/root")
    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"source root is not a directory: {path}")
    return name, path


def load_index(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"path", "sha256", "bytes", "upstream_dataset", "upstream_path"}
            if not isinstance(row, dict) or not required.issubset(row):
                raise ValueError(f"invalid image index row {line_number}")
            rows.append(row)
    return rows


def candidate_index(root: Path, needed_names: set[str]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)
    for current, directories, files in os.walk(root):
        directories[:] = [
            name for name in directories
            if not name.startswith(".") and "test" not in name.lower()
        ]
        for name in files:
            if name in needed_names:
                result[name].append(Path(current) / name)
    return result


def verified_candidate(candidates: Iterable[Path], expected_bytes: int, expected_hash: str) -> Path | None:
    for path in candidates:
        if path.is_file() and path.stat().st_size == expected_bytes and sha256(path) == expected_hash:
            return path
    return None


def binary_values(value: object) -> Iterable[bytes]:
    if isinstance(value, (bytes, bytearray, memoryview)):
        yield bytes(value)
    elif isinstance(value, dict):
        image_bytes = value.get("bytes")
        if isinstance(image_bytes, (bytes, bytearray, memoryview)):
            yield bytes(image_bytes)
        else:
            for nested in value.values():
                yield from binary_values(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from binary_values(nested)


def arrow_blobs(root: Path, needed_sizes: set[int], include_all: bool) -> Iterable[bytes]:
    # Training reconstruction must never read benchmark test shards. Official
    # save_to_disk repositories put them under top-level train/ and test/.
    scan_root = root / "train" if (root / "train").is_dir() else root
    arrow_paths = sorted(scan_root.rglob("*.arrow"))
    if not arrow_paths:
        return
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise RuntimeError(
            f"{root} contains Arrow files; install pyarrow to extract embedded images"
        ) from exc
    for arrow_path in arrow_paths:
        print(f"scanning official Arrow shard: {arrow_path}")
        with pa.memory_map(str(arrow_path), "r") as source:
            reader = pa.ipc.open_stream(source)
            image_index = reader.schema.get_field_index("image")
            if image_index < 0:
                continue
            for batch in reader:
                column = batch.column(image_index)
                for row_number in range(len(column)):
                    for blob in binary_values(column[row_number].as_py()):
                        if include_all or len(blob) in needed_sizes:
                            yield blob


def transform_blob(blob: bytes, spec: object) -> bytes:
    if not isinstance(spec, dict) or spec.get("kind") != "pillow_jpeg_preview":
        raise ValueError(f"unsupported image transform: {spec!r}")
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to reproduce transformed preview images") from exc
    resampling_name = str(spec.get("resampling", "LANCZOS"))
    resampling = getattr(Image.Resampling, resampling_name)
    max_edge = int(spec["max_edge"])
    with Image.open(io.BytesIO(blob)) as source:
        if bool(spec.get("use_draft", True)):
            source.draft(str(spec.get("mode", "RGB")), (max_edge, max_edge))
        image = source.convert(str(spec.get("mode", "RGB")))
        image.thumbnail(
            (max_edge, max_edge),
            resampling,
            reducing_gap=float(spec.get("reducing_gap", 2.0)),
        )
        output = io.BytesIO()
        image.save(
            output,
            format="JPEG",
            quality=int(spec.get("quality", 82)),
            optimize=bool(spec.get("optimize", True)),
        )
        image.close()
        return output.getvalue()


def resolve_selected_arrow(
    root: Path,
    pending: dict[str, dict[str, object]],
    dataset_root: Path,
    verify_only: bool,
) -> int:
    """Resolve transformed Arrow images by an exact official-record selector."""
    selected: dict[tuple[str, str], list[tuple[str, dict[str, object]]]] = defaultdict(list)
    for digest, row in pending.items():
        selector = row.get("source_selector")
        if (
            isinstance(selector, dict)
            and selector.get("field") == "question"
        ):
            key = ("question", str(selector["value"]))
        else:
            basename = Path(str(row["upstream_path"])).name
            key = ("path_stem", basename.split(".", 1)[0])
        selected[key].append((digest, row))
    if not selected:
        return 0
    scan_root = root / "train" if (root / "train").is_dir() else root
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to reconstruct selected XLRS images") from exc
    created = 0
    for arrow_path in sorted(scan_root.rglob("*.arrow")):
        if not selected:
            break
        print(f"scanning selected official Arrow shard: {arrow_path}")
        with pa.memory_map(str(arrow_path), "r") as source:
            reader = pa.ipc.open_stream(source)
            question_index = reader.schema.get_field_index("question")
            path_index = reader.schema.get_field_index("path")
            image_index = reader.schema.get_field_index("image")
            if image_index < 0:
                continue
            for batch in reader:
                questions = batch.column(question_index).to_pylist() if question_index >= 0 else []
                paths = batch.column(path_index).to_pylist() if path_index >= 0 else []
                images = batch.column(image_index)
                for row_number in range(len(images)):
                    keys: list[tuple[str, str]] = []
                    if question_index >= 0:
                        keys.append(("question", str(questions[row_number])))
                    if path_index >= 0:
                        basename = Path(str(paths[row_number])).name
                        keys.append(("path_stem", basename.split(".", 1)[0]))
                    matches = [
                        item
                        for key in keys
                        for item in selected.get(key, ())
                    ]
                    if not matches:
                        continue
                    blobs = list(binary_values(images[row_number].as_py()))
                    if len(blobs) != 1:
                        raise ValueError(f"selector resolved {len(blobs)} image blobs")
                    blob = blobs[0]
                    for key in keys:
                        candidates = selected.get(key)
                        if not candidates:
                            continue
                        for digest, row in list(candidates):
                            payload = transform_blob(blob, row["transform"])
                            actual = hashlib.sha256(payload).hexdigest()
                            if actual != digest or len(payload) != int(row["bytes"]):
                                continue
                            if not verify_only:
                                destination = dataset_root / str(row["path"])
                                destination.parent.mkdir(parents=True, exist_ok=True)
                                destination.write_bytes(payload)
                            pending.pop(digest, None)
                            candidates.remove((digest, row))
                            created += 1
                        if not candidates:
                            selected.pop(key, None)
    return created


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        type=parse_source,
        required=True,
        metavar="NAME=ROOT",
        help="repeat for mme_realworld_rs, vrsbench, xlrs, and xlrs_grounding",
    )
    parser.add_argument("--copy", action="store_true", help="copy instead of hard-linking when possible")
    parser.add_argument("--verify-only", action="store_true", help="verify sources without materializing files")
    parser.add_argument(
        "--resume-verified",
        action="store_true",
        help="skip existing files after a prior SHA-verified run; never use for first construction",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    rows = load_index(dataset_root / "IMAGE_INDEX.jsonl")
    sources = dict(args.source)
    expected_names = {str(row["upstream_dataset"]) for row in rows}
    missing_roots = sorted(expected_names - sources.keys())
    if missing_roots:
        raise ValueError(f"missing --source mappings: {', '.join(missing_roots)}")

    names_by_source: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        names_by_source[str(row["upstream_dataset"])].add(Path(str(row["upstream_path"])).name)
    basename_indexes = {
        name: candidate_index(sources[name], names)
        for name, names in names_by_source.items()
    }

    unresolved: dict[str, dict[str, object]] = {}
    reused = created = 0
    for number, row in enumerate(rows, 1):
        destination = dataset_root / str(row["path"])
        expected_hash = str(row["sha256"])
        expected_bytes = int(row["bytes"])
        if args.resume_verified and destination.is_file():
            reused += 1
            continue
        if destination.is_file() and destination.stat().st_size == expected_bytes and sha256(destination) == expected_hash:
            reused += 1
            continue

        source_name = str(row["upstream_dataset"])
        upstream_path = Path(str(row["upstream_path"]))
        exact = sources[source_name] / upstream_path
        candidates = [exact, *basename_indexes[source_name].get(upstream_path.name, [])]
        source = verified_candidate(dict.fromkeys(candidates), expected_bytes, expected_hash)
        transformed_payload = None
        if source is None and "transform" in row:
            for candidate in dict.fromkeys(candidates):
                if not candidate.is_file():
                    continue
                payload = transform_blob(candidate.read_bytes(), row["transform"])
                if len(payload) == expected_bytes and hashlib.sha256(payload).hexdigest() == expected_hash:
                    transformed_payload = payload
                    break
        if source is None and transformed_payload is None:
            unresolved[expected_hash] = row
            continue
        if not args.verify_only:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                destination.unlink()
            if transformed_payload is not None:
                destination.write_bytes(transformed_payload)
            elif args.copy:
                shutil.copy2(source, destination)
            else:
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
        created += 1
        if number % 1000 == 0:
            print(f"checked {number}/{len(rows)} images")

    # XLRS official repositories store images inside Arrow shards. Scan only
    # unresolved hashes and stop each source as soon as all of its images are found.
    for source_name, source_root in sources.items():
        pending = {
            digest: row for digest, row in unresolved.items()
            if str(row["upstream_dataset"]) == source_name
        }
        if not pending:
            continue
        selected_created = resolve_selected_arrow(
            source_root, pending, dataset_root, args.verify_only
        )
        created += selected_created
        for digest in list(unresolved):
            if digest not in pending and str(unresolved[digest]["upstream_dataset"]) == source_name:
                del unresolved[digest]
        if not pending:
            continue
        needed_sizes = {
            int(row["bytes"]) for row in pending.values()
            if "transform" not in row
        }
        transform_specs = {
            json.dumps(row["transform"], sort_keys=True): row["transform"]
            for row in pending.values() if "transform" in row
        }
        seen_raw: set[str] = set()
        for blob in arrow_blobs(source_root, needed_sizes, bool(transform_specs)):
            digest = hashlib.sha256(blob).hexdigest()
            candidates = [(digest, blob)] if digest in pending else []
            if transform_specs and digest not in seen_raw:
                seen_raw.add(digest)
                for spec in transform_specs.values():
                    transformed = transform_blob(blob, spec)
                    transformed_digest = hashlib.sha256(transformed).hexdigest()
                    if transformed_digest in pending:
                        candidates.append((transformed_digest, transformed))
            for matched_digest, payload in candidates:
                row = pending.get(matched_digest)
                if row is None:
                    continue
                destination = dataset_root / str(row["path"])
                if not args.verify_only:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(payload)
                created += 1
                del unresolved[matched_digest]
                del pending[matched_digest]
            if not pending:
                break

    failures = [
        f"{row['upstream_dataset']}:{row['upstream_path']} sha256={digest}"
        for digest, row in sorted(unresolved.items())
    ]

    report = {
        "indexed": len(rows),
        "already_materialized": reused,
        "verified_from_upstream": created,
        "missing_or_mismatched": len(failures),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        preview = "\n".join(failures[:20])
        raise SystemExit(f"failed to resolve {len(failures)} image(s); first failures:\n{preview}")


if __name__ == "__main__":
    main()
