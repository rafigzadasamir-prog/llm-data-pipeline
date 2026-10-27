"""Data-quality report logic: completeness, distributions, sanity checks."""

from __future__ import annotations

from llm_data_pipeline.quality import build_quality_report, compute_completeness, run_sanity_checks
from llm_data_pipeline.schema import ExtractedTicket


def _ticket(ticket_id: str, **overrides: object) -> ExtractedTicket:
    defaults: dict[str, object] = {
        "ticket_id": ticket_id,
        "category": "billing",
        "sentiment": "neutral",
        "urgency": "medium",
        "mentioned_product": "PayEase",
        "summary": "Some summary.",
    }
    defaults.update(overrides)
    return ExtractedTicket(**defaults)


def test_compute_completeness_all_fields_present() -> None:
    tickets = [_ticket("T1"), _ticket("T2")]
    completeness = compute_completeness(tickets)
    assert completeness["category"] == 1.0
    assert completeness["sentiment"] == 1.0
    assert completeness["urgency"] == 1.0
    assert completeness["summary"] == 1.0
    assert completeness["mentioned_product"] == 1.0


def test_compute_completeness_handles_missing_optional_field() -> None:
    tickets = [
        _ticket("T1", mentioned_product="DataVault"),
        _ticket("T2", mentioned_product=None),
        _ticket("T3", mentioned_product=None),
        _ticket("T4", mentioned_product=None),
    ]
    completeness = compute_completeness(tickets)
    assert completeness["mentioned_product"] == 0.25
    # required fields are untouched by the missing optional field
    assert completeness["category"] == 1.0


def test_compute_completeness_empty_list_returns_zero_for_every_field() -> None:
    completeness = compute_completeness([])
    assert completeness["category"] == 0.0
    assert completeness["mentioned_product"] == 0.0


def test_run_sanity_checks_passes_for_well_formed_tickets() -> None:
    tickets = [_ticket("T1"), _ticket("T2", category="technical_issue", urgency="critical")]
    assert run_sanity_checks(tickets) == []


def test_build_quality_report_category_and_sentiment_distribution() -> None:
    tickets = [
        _ticket("T1", category="billing", sentiment="negative"),
        _ticket("T2", category="billing", sentiment="positive"),
        _ticket("T3", category="shipping", sentiment="negative"),
    ]
    report = build_quality_report(tickets)

    assert report.total_records == 3
    assert report.category_distribution == {"billing": 2, "shipping": 1}
    assert report.sentiment_distribution == {"negative": 2, "positive": 1}
    assert report.passed is True


def test_build_quality_report_format_contains_key_sections() -> None:
    report = build_quality_report([_ticket("T1")])
    text = report.format()
    assert "Data Quality Report" in text
    assert "Completeness by field" in text
    assert "Category distribution" in text
    assert "Sentiment distribution" in text
    assert "Urgency distribution" in text
    assert "Sanity checks" in text
    assert "all checks passed" in text


def test_quality_report_on_empty_ticket_list() -> None:
    report = build_quality_report([])
    assert report.total_records == 0
    assert report.category_distribution == {}
    assert report.passed is True
