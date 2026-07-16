#!/usr/bin/env python3
"""Run one real Count Anything inference and save its structured result."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visualization", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    checkpoint = args.checkpoint.resolve()
    image = args.image.resolve()
    for path in (repo, checkpoint, image):
        if not path.exists():
            raise FileNotFoundError(path)

    sys.path.insert(0, str(repo))
    from count_anything import CountAnything

    start = time.perf_counter()
    model = CountAnything(
        checkpoint,
        output_dir=args.output.parent / "count_anything_runs",
        python_executable=sys.executable,
    )
    results = model(image, args.query)
    elapsed_s = time.perf_counter() - start
    if len(results) != 1:
        raise RuntimeError(f"Expected one Count Anything result, got {len(results)}")

    result = results[0]
    args.visualization.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.visualization)
    payload = {
        "model": "MengqiLei/count-anything",
        "image": str(image),
        "query": args.query,
        "count": result.count,
        "point_count": len(result.pred_points),
        "elapsed_s": round(elapsed_s, 3),
        "visualization": str(args.visualization),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
