#!/usr/bin/env python3
"""Safely extract official MME split tarballs and VRSBench image archives."""
from __future__ import annotations

import argparse
import io
import tarfile
import zipfile
from pathlib import Path


def safe_target(root: Path, name: str) -> Path:
    target = (root / name).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"archive member escapes output root: {name}") from exc
    return target


class SplitReader(io.RawIOBase):
    def __init__(self, parts: list[Path]):
        self.parts = parts
        self.index = 0
        self.handle = parts[0].open("rb")

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray) -> int:
        view = memoryview(buffer)
        total = 0
        while total < len(view):
            count = self.handle.readinto(view[total:])
            if count:
                total += count
                continue
            self.handle.close()
            self.index += 1
            if self.index >= len(self.parts):
                break
            self.handle = self.parts[self.index].open("rb")
        return total

    def close(self) -> None:
        self.handle.close()
        super().close()


def extract_split_tar(parts: list[Path], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with SplitReader(parts) as raw, io.BufferedReader(raw, buffer_size=8 << 20) as stream:
        with tarfile.open(fileobj=stream, mode="r|gz") as archive:
            for member in archive:
                safe_target(output, member.name)
                if member.issym() or member.islnk():
                    raise ValueError(f"archive links are not accepted: {member.name}")
                archive.extract(member, path=output, filter="data")


def extract_zip(path: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            safe_target(output, member.filename)
        archive.extractall(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mme-parts-dir", type=Path, required=True)
    parser.add_argument("--mme-output", type=Path, required=True)
    parser.add_argument("--vrs-zip", type=Path, required=True)
    parser.add_argument("--vrs-output", type=Path, required=True)
    args = parser.parse_args()

    parts = sorted(args.mme_parts_dir.glob("remote_sensing.tar.gz.part_*"))
    if len(parts) != 12:
        raise ValueError(f"expected 12 MME remote-sensing parts, found {len(parts)}")
    mme_marker = args.mme_output / ".extract_complete"
    if not mme_marker.is_file():
        extract_split_tar(parts, args.mme_output)
        mme_marker.touch()
    vrs_marker = args.vrs_output / ".extract_complete"
    if not vrs_marker.is_file():
        extract_zip(args.vrs_zip, args.vrs_output)
        vrs_marker.touch()


if __name__ == "__main__":
    main()
