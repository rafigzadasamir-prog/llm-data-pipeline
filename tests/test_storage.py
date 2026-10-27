"""SQLite storage layer: insert + query round-trip behavior."""

from __future__ import annotations

from pathlib import Path

from llm_data_pipeline.schema import ExtractedTicket
from llm_data_pipeline.storage import TicketStore


def _make_ticket(ticket_id: str, **overrides: object) -> ExtractedTicket:
    defaults: dict[str, object] = {
        "ticket_id": ticket_id,
        "category": "billing",
        "sentiment": "negative",
        "urgency": "high",
        "mentioned_product": "PayEase",
        "summary": "A billing complaint.",
    }
    defaults.update(overrides)
    return ExtractedTicket(**defaults)


def test_insert_and_get_round_trip(tmp_path: Path) -> None:
    db_path = tmp_path / "tickets.db"
    ticket = _make_ticket("TCK-0001")

    with TicketStore(db_path) as store:
        store.insert_ticket(ticket)
        fetched = store.get_ticket("TCK-0001")

    assert fetched is not None
    assert fetched.ticket_id == ticket.ticket_id
    assert fetched.category == ticket.category
    assert fetched.sentiment == ticket.sentiment
    assert fetched.urgency == ticket.urgency
    assert fetched.mentioned_product == ticket.mentioned_product
    assert fetched.summary == ticket.summary


def test_get_missing_ticket_returns_none(tmp_path: Path) -> None:
    db_path = tmp_path / "tickets.db"
    with TicketStore(db_path) as store:
        assert store.get_ticket("does-not-exist") is None


def test_insert_many_and_all_tickets(tmp_path: Path) -> None:
    db_path = tmp_path / "tickets.db"
    tickets = [_make_ticket(f"TCK-{i:04d}") for i in range(5)]

    with TicketStore(db_path) as store:
        inserted = store.insert_many(tickets)
        assert inserted == 5
        assert store.count() == 5
        all_rows = store.all_tickets()

    assert sorted(t.ticket_id for t in all_rows) == [t.ticket_id for t in tickets]


def test_insert_many_with_empty_list_is_a_no_op(tmp_path: Path) -> None:
    db_path = tmp_path / "tickets.db"
    with TicketStore(db_path) as store:
        assert store.insert_many([]) == 0
        assert store.count() == 0


def test_null_mentioned_product_round_trips_as_none(tmp_path: Path) -> None:
    db_path = tmp_path / "tickets.db"
    ticket = _make_ticket("TCK-0002", mentioned_product=None)

    with TicketStore(db_path) as store:
        store.insert_ticket(ticket)
        fetched = store.get_ticket("TCK-0002")

    assert fetched is not None
    assert fetched.mentioned_product is None


def test_upsert_replaces_existing_row_on_conflicting_id(tmp_path: Path) -> None:
    db_path = tmp_path / "tickets.db"
    original = _make_ticket("TCK-0003", category="billing", summary="Original summary.")
    updated = _make_ticket("TCK-0003", category="technical_issue", summary="Updated summary.")

    with TicketStore(db_path) as store:
        store.insert_ticket(original)
        store.insert_ticket(updated)
        assert store.count() == 1
        fetched = store.get_ticket("TCK-0003")

    assert fetched is not None
    assert fetched.category.value == "technical_issue"
    assert fetched.summary == "Updated summary."


def test_store_persists_across_separate_connections(tmp_path: Path) -> None:
    db_path = tmp_path / "tickets.db"
    with TicketStore(db_path) as store:
        store.insert_ticket(_make_ticket("TCK-0004"))

    with TicketStore(db_path) as store:
        assert store.count() == 1
        assert store.get_ticket("TCK-0004") is not None


def test_creates_parent_directories_if_missing(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "dir" / "tickets.db"
    with TicketStore(db_path) as store:
        store.insert_ticket(_make_ticket("TCK-0005"))
    assert db_path.exists()
