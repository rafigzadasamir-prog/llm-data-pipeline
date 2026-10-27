"""Validation behavior of the Pydantic schema: valid and invalid cases."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_data_pipeline.schema import ExtractedTicket, RawRecord, Sentiment, TicketCategory, Urgency


def test_valid_extracted_ticket_round_trips_enum_values() -> None:
    ticket = ExtractedTicket(
        ticket_id="TCK-0001",
        category="billing",
        sentiment="negative",
        urgency="critical",
        mentioned_product="CloudSync Pro",
        summary="Customer was double-charged and wants a refund.",
    )
    assert ticket.category is TicketCategory.BILLING
    assert ticket.sentiment is Sentiment.NEGATIVE
    assert ticket.urgency is Urgency.CRITICAL
    assert ticket.mentioned_product == "CloudSync Pro"


def test_valid_extracted_ticket_allows_missing_product() -> None:
    ticket = ExtractedTicket(
        ticket_id="TCK-0002",
        category="general_inquiry",
        sentiment="neutral",
        urgency="medium",
        mentioned_product=None,
        summary="Customer asked about available plans.",
    )
    assert ticket.mentioned_product is None


@pytest.mark.parametrize("sentinel", ["", "   ", "none", "None", "NONE"])
def test_blank_or_none_sentinel_product_normalizes_to_none(sentinel: str) -> None:
    ticket = ExtractedTicket(
        ticket_id="TCK-0003",
        category="general_inquiry",
        sentiment="neutral",
        urgency="low",
        mentioned_product=sentinel,
        summary="A generic inquiry.",
    )
    assert ticket.mentioned_product is None


def test_invalid_category_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ExtractedTicket(
            ticket_id="TCK-0004",
            category="not_a_real_category",
            sentiment="neutral",
            urgency="medium",
            summary="Some summary.",
        )


def test_invalid_urgency_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ExtractedTicket(
            ticket_id="TCK-0005",
            category="billing",
            sentiment="neutral",
            urgency="super-duper-urgent",
            summary="Some summary.",
        )


def test_blank_summary_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ExtractedTicket(
            ticket_id="TCK-0006",
            category="billing",
            sentiment="neutral",
            urgency="low",
            summary="   ",
        )


def test_missing_required_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ExtractedTicket(
            ticket_id="TCK-0007",
            category="billing",
            sentiment="neutral",
            # urgency missing
            summary="Some summary.",
        )


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ExtractedTicket(
            ticket_id="TCK-0008",
            category="billing",
            sentiment="neutral",
            urgency="low",
            summary="Some summary.",
            unexpected_field="surprise",
        )


def test_raw_record_rejects_blank_text() -> None:
    with pytest.raises(ValidationError):
        RawRecord(id="TCK-0009", text="   ")


def test_raw_record_rejects_empty_id() -> None:
    with pytest.raises(ValidationError):
        RawRecord(id="", text="Some ticket text.")


def test_raw_record_accepts_valid_input() -> None:
    record = RawRecord(id="TCK-0010", text="My app crashed again today.")
    assert record.id == "TCK-0010"
    assert record.text == "My app crashed again today."
