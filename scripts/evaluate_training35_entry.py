#!/usr/bin/env python3
"""Run the shared evaluator with a non-truncating Training 3.5 token limit."""
from __future__ import annotations

from evaluation import main as evaluation_main


def main() -> None:
    # Four five-decimal coordinates can exceed the shared 32-token referring
    # limit.  Keep this override local to Training 3.5 so 3.3/3.4 remain
    # untouched, while allowing the closing bracket to be generated.
    evaluation_main.TASK_MAX_TOKENS = {
        **evaluation_main.TASK_MAX_TOKENS,
        "referring": 64,
    }
    evaluation_main.main()


if __name__ == "__main__":
    main()
