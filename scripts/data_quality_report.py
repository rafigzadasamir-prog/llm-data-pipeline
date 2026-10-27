#!/usr/bin/env python
"""Standalone data-quality report over whatever is currently stored in SQLite.

Usage:
    python scripts/run_pipeline.py            # populate the database first
    python scripts/data_quality_report.py      # then report on it

Exits non-zero if any sanity check fails, so it can be wired into CI as a
gate on top of the pipeline run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_data_pipeline.quality import build_quality_report  # noqa: E402
from llm_data_pipeline.storage import TicketStore  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path",
        default=REPO_ROOT / "data" / "tickets.db",
        type=Path,
        help="Path to the SQLite database written by scripts/run_pipeline.py.",
    )
    args = parser.parse_args(argv)

    if not args.db_path.exists():
        print(f"No database found at {args.db_path}. Run scripts/run_pipeline.py first.", file=sys.stderr)
        return 1

    with TicketStore(args.db_path) as store:
        tickets = store.all_tickets()

    if not tickets:
        print(f"Database at {args.db_path} contains no tickets.", file=sys.stderr)
        return 1

    report = build_quality_report(tickets)
    print(report.format())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
