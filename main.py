from __future__ import annotations

import src.runtime_bootstrap  # noqa: F401

import argparse
import sys

from src.cli.app import CLIApp


def _normalize_args(argv: list[str]) -> list[str]:
    """Normalize command-line arguments to support flexible evaluation sample size flags:
    e.g. --evaluate 5, --evaluate - 5, --evaluate -5, or --sample-size 5
    """
    normalized: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--evaluate":
            normalized.append("--evaluate")
            if i + 2 < len(argv) and argv[i + 1] == "-" and argv[i + 2].isdigit():
                normalized.extend(["--sample-size", argv[i + 2]])
                i += 2
            elif i + 1 < len(argv) and argv[i + 1].startswith("-") and argv[i + 1][1:].isdigit():
                normalized.extend(["--sample-size", argv[i + 1][1:]])
                i += 1
            elif i + 1 < len(argv) and argv[i + 1].isdigit():
                normalized.extend(["--sample-size", argv[i + 1]])
                i += 1
        else:
            normalized.append(arg)
        i += 1
    return normalized


def main(argv: list[str] | None = None) -> None:
    raw_args = sys.argv[1:] if argv is None else argv
    normalized_args = _normalize_args(raw_args)

    parser = argparse.ArgumentParser(description="Run the PopQA-RAG-Judge CLI.")
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuild the FAISS index from the PopQA dataset before starting the CLI.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run the evaluation pipeline and exit instead of starting the interactive CLI.",
    )
    parser.add_argument(
        "--sample-size",
        "-s",
        type=int,
        default=None,
        help="Number of dataset samples to evaluate (e.g., --evaluate 5 or --evaluate - 5).",
    )
    args = parser.parse_args(normalized_args)

    app = CLIApp()
    if args.evaluate:
        app.run_evaluation(force_rebuild=args.rebuild_index, sample_size=args.sample_size)
    else:
        app.run(force_rebuild=args.rebuild_index)


if __name__ == "__main__":
    main()
