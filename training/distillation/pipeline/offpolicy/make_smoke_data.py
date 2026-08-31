#!/usr/bin/env python3
"""Create a deterministic one-batch parquet for an end-to-end smoke run."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rows", type=int, default=8)
    args = parser.parse_args()
    if args.rows <= 0:
        raise ValueError("--rows must be positive")
    table = pq.read_table(args.input)
    if table.num_rows < args.rows:
        raise ValueError(f"Input has only {table.num_rows} rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.slice(0, args.rows), args.output, compression="zstd")
    print(f"Wrote {args.rows} smoke records to {args.output}")


if __name__ == "__main__":
    main()
