"""The orchestrator itself, and the full ingest->extract->validate->store->report
flow wired together with RuleBasedExtractor (no network, no API key).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_data_pipeline.extractors.rule_based import RuleBasedExtractor
from llm_data_pipeline.pipeline.orchestrator import Pipeline, PipelineStageError
from llm_data_pipeline.pipeline.stages import (
    PipelineContext,
    ingest_stage,
    make_extract_stage,
    make_store_stage,
    report_stage,
    validate_stage,
)
from llm_data_pipeline.storage import TicketStore


def test_pipeline_runs_stages_in_order_and_records_timings() -> None:
    pipeline = (
        Pipeline()
        .add_stage("double", lambda x: x * 2)
        .add_stage("add_one", lambda x: x + 1)
    )
    run = pipeline.run(5)
    assert run.value == 11
    assert [t.name for t in run.timings] == ["double", "add_one"]
    assert all(t.status == "ok" for t in run.timings)
    assert all(t.duration_seconds >= 0 for t in run.timings)


def test_pipeline_stops_and_reports_failure_on_stage_exception() -> None:
    def boom(_: int) -> int:
        raise ValueError("kaboom")

    pipeline = Pipeline().add_stage("first", lambda x: x + 1).add_stage("boom", boom).add_stage(
        "never_runs", lambda x: x + 100
    )

    with pytest.raises(PipelineStageError) as exc_info:
        pipeline.run(1)

    assert exc_info.value.stage_name == "boom"
    assert "kaboom" in str(exc_info.value)


def _write_dataset(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {"id": "TCK-0001", "text": "I want to cancel my CloudSync Pro subscription immediately."},
                {"id": "TCK-0002", "text": "Thanks so much, the TaskFlow update looks great!"},
                {"id": "TCK-0003", "text": "My package tracking number has not updated in days."},
            ]
        ),
        encoding="utf-8",
    )


def test_full_pipeline_end_to_end_with_rule_based_extractor(tmp_path: Path) -> None:
    data_path = tmp_path / "raw.json"
    db_path = tmp_path / "tickets.db"
    _write_dataset(data_path)

    extractor = RuleBasedExtractor()
    pipeline = (
        Pipeline()
        .add_stage("ingest", ingest_stage)
        .add_stage("extract", make_extract_stage(extractor))
        .add_stage("validate", validate_stage)
        .add_stage("store", make_store_stage())
        .add_stage("report", report_stage)
    )

    context = PipelineContext(data_path=data_path, db_path=db_path)
    run = pipeline.run(context)
    result = run.value

    assert [t.name for t in run.timings] == ["ingest", "extract", "validate", "store", "report"]
    assert all(t.status == "ok" for t in run.timings)

    assert len(result.raw_records) == 3
    assert len(result.valid_tickets) == 3
    assert len(result.rejected) == 0
    assert result.stored_count == 3
    assert result.summary is not None
    assert result.summary.processed == 3
    assert result.summary.valid == 3
    assert result.summary.rejected == 0

    with TicketStore(db_path) as store:
        assert store.count() == 3
        stored = store.get_ticket("TCK-0001")
    assert stored is not None
    assert stored.category.value == "cancellation"


def test_validate_stage_rejects_records_with_extraction_errors(tmp_path: Path) -> None:
    from llm_data_pipeline.pipeline.stages import validate_stage
    from llm_data_pipeline.schema import RawRecord

    good_record = RawRecord(id="TCK-0001", text="Please cancel my account.")
    bad_record = RawRecord(id="TCK-0002", text="Some ticket text.")

    context = PipelineContext(data_path=tmp_path / "unused.json", db_path=tmp_path / "unused.db")
    context.raw_records = [good_record, bad_record]
    context.extraction_results = [
        (
            good_record,
            {
                "category": "cancellation",
                "sentiment": "neutral",
                "urgency": "medium",
                "mentioned_product": None,
                "summary": "Customer wants to cancel.",
            },
            None,
        ),
        (bad_record, None, "simulated extractor failure"),
    ]

    result = validate_stage(context)

    assert len(result.valid_tickets) == 1
    assert result.valid_tickets[0].ticket_id == "TCK-0001"
    assert len(result.rejected) == 1
    assert result.rejected[0].ticket_id == "TCK-0002"
    assert "simulated extractor failure" in result.rejected[0].reason


def test_validate_stage_rejects_records_failing_schema_validation(tmp_path: Path) -> None:
    from llm_data_pipeline.schema import RawRecord

    record = RawRecord(id="TCK-0003", text="Some ticket text.")
    context = PipelineContext(data_path=tmp_path / "unused.json", db_path=tmp_path / "unused.db")
    context.raw_records = [record]
    context.extraction_results = [
        (
            record,
            {
                "category": "not_a_real_category",
                "sentiment": "neutral",
                "urgency": "medium",
                "mentioned_product": None,
                "summary": "A summary.",
            },
            None,
        )
    ]

    result = validate_stage(context)

    assert result.valid_tickets == []
    assert len(result.rejected) == 1
    assert result.rejected[0].ticket_id == "TCK-0003"
    assert "schema validation failed" in result.rejected[0].reason
