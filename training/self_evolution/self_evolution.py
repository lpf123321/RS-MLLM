#!/usr/bin/env python3
"""Repository-level entry point for the CVSearch self-evolution orchestrator."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "orchestrator" / "src"))

from evolve.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
