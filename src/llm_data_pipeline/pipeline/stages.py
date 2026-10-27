"""Stage functions for the ingest -> extract -> validate -> store -> report flow.

Each stage is a plain function ``PipelineContext -> PipelineContext`` so it
can be threaded through :class:`llm_data_pipeline.pipeline.orchestrator.Pipeline`.
Stages that need configuration (which extractor, which DB file) are built
by small factory functions that close over that configuration and return
the actual stage callable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from llm_data_pipeline.extractors.base import Extractor
from llm_data_pipeline.ingestion import load_raw_records
from llm_data_pipeline.schema import ExtractedTicket, RawRecord
from llm_data_pipeline.storage import TicketStore


@dataclass
class RejectedRecord:
    """A record that did not make it into storage, and why."""

    ticket_id: str
    reason: str


@dataclass
class PipelineSummary:
    """Headline numbers printed after a pipeline run."""

    processed: int
    valid: int
    rejected: int
    null_rates: dict[str, float]
    category_counts: dict[str, int]

    def format(self) -> str:
        lines = [
            "Pipeline summary",
            "-----------------",
            f"records processed : {self.processed}",
            f"records valid      : {self.valid}",
            f"records rejected   : {self.rejected}",
            "",
            "null rates by field (share of valid records with a missing value):",
        ]
        for field_name, rate in self.null_rates.items():
            lines.append(f"  {field_name:<20} {rate:.1%}")
        lines.append("")
        lines.append("category distribution (valid records):")
        for category, count in sorted(self.category_counts.items(), key=lambda kv: kv[0]):
            lines.append(f"  {category:<20} {count}")
        return "\n".join(lines)


@dataclass
class PipelineContext:
    """State threaded through every pipeline stage."""

    data_path: Path
    db_path: Path
    raw_records: list[RawRecord] = field(default_factory=list)
    extraction_results: list[tuple[RawRecord, dict | None, str | None]] = field(default_factory=list)
    valid_tickets: list[ExtractedTicket] = field(default_factory=list)
    rejected: list[RejectedRecord] = field(default_factory=list)
    stored_count: int = 0
    summary: PipelineSummary | None = None


def ingest_stage(ctx: PipelineContext) -> PipelineContext:
    ctx.raw_records = load_raw_records(ctx.data_path)
    return ctx


def make_extract_stage(extractor: Extractor) -> Callable[[PipelineContext], PipelineContext]:
    def _stage(ctx: PipelineContext) -> PipelineContext:
        ctx.extraction_results = extractor.extract_many(ctx.raw_records)
        return ctx

    return _stage


# Every field extractors are expected to produce. Used by the validate
# stage to tell "extractor crashed" apart from "extractor returned a
# field set that fails schema validation".
_EXPECTED_FIELDS = ("category", "sentiment", "urgency", "mentioned_product", "summary")


def validate_stage(ctx: PipelineContext) -> PipelineContext:
    valid: list[ExtractedTicket] = []
    rejected: list[RejectedRecord] = []
    for record, fields, extraction_error in ctx.extraction_results:
        if extraction_error is not None:
            rejected.append(RejectedRecord(record.id, f"extraction failed: {extraction_error}"))
            continue
        try:
            ticket = ExtractedTicket(ticket_id=record.id, **(fields or {}))
        except ValidationError as exc:
            rejected.append(RejectedRecord(record.id, f"schema validation failed: {exc}"))
            continue
        valid.append(ticket)
    ctx.valid_tickets = valid
    ctx.rejected = rejected
    return ctx


def make_store_stage(db_path_override: Path | None = None) -> Callable[[PipelineContext], PipelineContext]:
    def _stage(ctx: PipelineContext) -> PipelineContext:
        db_path = db_path_override or ctx.db_path
        with TicketStore(db_path) as store:
            ctx.stored_count = store.insert_many(ctx.valid_tickets)
        return ctx

    return _stage


def report_stage(ctx: PipelineContext) -> PipelineContext:
    ctx.summary = build_summary(ctx)
    return ctx


def build_summary(ctx: PipelineContext) -> PipelineSummary:
    valid = ctx.valid_tickets
    total_valid = len(valid)

    null_rates: dict[str, float] = {}
    for field_name in _EXPECTED_FIELDS:
        if total_valid == 0:
            null_rates[field_name] = 0.0
            continue
        null_count = sum(1 for t in valid if getattr(t, field_name) in (None, ""))
        null_rates[field_name] = null_count / total_valid

    category_counts: dict[str, int] = {}
    for ticket in valid:
        category_counts[ticket.category.value] = category_counts.get(ticket.category.value, 0) + 1

    return PipelineSummary(
        processed=len(ctx.raw_records),
        valid=total_valid,
        rejected=len(ctx.rejected),
        null_rates=null_rates,
        category_counts=category_counts,
    )
