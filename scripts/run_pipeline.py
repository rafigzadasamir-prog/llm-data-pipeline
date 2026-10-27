#!/usr/bin/env python
"""Entrypoint: runs ingest -> extract -> validate -> store -> report end to end.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --data-path data/raw_tickets.json --db-path data/tickets.db
    python scripts/run_pipeline.py --force-rule-based

The extractor is chosen automatically: the real `LLMExtractor` is used when
`ANTHROPIC_API_KEY` is set in the environment, otherwise the deterministic
`RuleBasedExtractor` fallback is used - no key, no network, no cost.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_data_pipeline.extractors import Extractor, LLMExtractor, RuleBasedExtractor  # noqa: E402
from llm_data_pipeline.pipeline.orchestrator import Pipeline  # noqa: E402
from llm_data_pipeline.pipeline.stages import (  # noqa: E402
    PipelineContext,
    ingest_stage,
    make_extract_stage,
    make_store_stage,
    report_stage,
    validate_stage,
)


def _build_extractor(force_rule_based: bool) -> Extractor:
    if not force_rule_based and os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY detected - using LLMExtractor (real Anthropic API calls).")
        return LLMExtractor()
    print("No ANTHROPIC_API_KEY (or --force-rule-based set) - using RuleBasedExtractor (offline).")
    return RuleBasedExtractor()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        default=REPO_ROOT / "data" / "raw_tickets.json",
        type=Path,
        help="Path to the bundled raw-records JSON file.",
    )
    parser.add_argument(
        "--db-path",
        default=REPO_ROOT / "data" / "tickets.db",
        type=Path,
        help="Path to the SQLite database file to write validated tickets to.",
    )
    parser.add_argument(
        "--force-rule-based",
        action="store_true",
        help="Use RuleBasedExtractor even if ANTHROPIC_API_KEY is set.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO-level stage-timing logs.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    extractor = _build_extractor(args.force_rule_based)

    pipeline = (
        Pipeline()
        .add_stage("ingest", ingest_stage)
        .add_stage("extract", make_extract_stage(extractor))
        .add_stage("validate", validate_stage)
        .add_stage("store", make_store_stage())
        .add_stage("report", report_stage)
    )

    context = PipelineContext(data_path=args.data_path, db_path=args.db_path)

    try:
        run = pipeline.run(context)
    except Exception as exc:  # noqa: BLE001 - top-level entrypoint, report and exit non-zero
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1

    print()
    print("Stage timings")
    print("-------------")
    for timing in run.timings:
        print(f"  {timing.name:<10} {timing.duration_seconds * 1000:7.2f} ms  [{timing.status}]")

    result_ctx = run.value
    print()
    if result_ctx.valid_tickets:
        example = result_ctx.valid_tickets[0]
        example_record = next(r for r in result_ctx.raw_records if r.id == example.ticket_id)
        print("Example extraction")
        print("-------------------")
        print(f"input  ({example_record.id}): {example_record.text}")
        print(f"output : {example.model_dump_json(indent=2)}")
        print()

    print(result_ctx.summary.format())

    if result_ctx.rejected:
        print()
        print("Rejected records:")
        for rejected in result_ctx.rejected:
            print(f"  {rejected.ticket_id}: {rejected.reason}")

    print()
    print(f"Stored {result_ctx.stored_count} record(s) in {args.db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
