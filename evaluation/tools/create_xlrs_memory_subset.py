#!/usr/bin/env python3
"""Create the canonical fixed XLRS subset used by memory benchmarks."""

import argparse
import hashlib
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    lines = source_bytes.decode("utf-8").splitlines()
    if args.count <= 0 or args.count > len(lines):
        parser.error(f"--count must be in [1, {len(lines)}]")

    indices = random.Random(args.seed).sample(range(len(lines)), args.count)
    subset = [lines[index] for index in indices]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(subset) + "\n", encoding="utf-8")
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(
        json.dumps(
            {
                "source_file": args.source.name,
                "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "source_records": len(lines),
                "subset_records": len(indices),
                "selection": "random.Random(seed).sample(range(source_records), count)",
                "seed": args.seed,
                "indices": indices,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
