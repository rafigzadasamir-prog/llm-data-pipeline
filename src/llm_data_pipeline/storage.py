"""SQLite storage layer for validated tickets, built on the stdlib `sqlite3`.

A hand-rolled module over `sqlite3` (rather than SQLAlchemy) keeps the
dependency footprint small - the schema is one table and the access
patterns are "insert validated rows" and "read them back for the quality
report", which don't need an ORM.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from llm_data_pipeline.schema import ExtractedTicket

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    sentiment TEXT NOT NULL,
    urgency TEXT NOT NULL,
    mentioned_product TEXT,
    summary TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
)
"""

_UPSERT_SQL = """
INSERT INTO tickets (ticket_id, category, sentiment, urgency, mentioned_product, summary)
VALUES (:ticket_id, :category, :sentiment, :urgency, :mentioned_product, :summary)
ON CONFLICT(ticket_id) DO UPDATE SET
    category = excluded.category,
    sentiment = excluded.sentiment,
    urgency = excluded.urgency,
    mentioned_product = excluded.mentioned_product,
    summary = excluded.summary
"""


class TicketStore:
    """Thin wrapper over a SQLite connection for the `tickets` table.

    Usable as a context manager, which closes the underlying connection on
    exit::

        with TicketStore(db_path) as store:
            store.insert_many(tickets)
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.commit()

    def insert_ticket(self, ticket: ExtractedTicket) -> None:
        self._conn.execute(_UPSERT_SQL, _ticket_to_row(ticket))
        self._conn.commit()

    def insert_many(self, tickets: list[ExtractedTicket]) -> int:
        """Insert (or replace) many tickets in one transaction. Returns the count."""
        rows = [_ticket_to_row(t) for t in tickets]
        if rows:
            self._conn.executemany(_UPSERT_SQL, rows)
            self._conn.commit()
        return len(rows)

    def get_ticket(self, ticket_id: str) -> ExtractedTicket | None:
        cursor = self._conn.execute(
            "SELECT ticket_id, category, sentiment, urgency, mentioned_product, summary "
            "FROM tickets WHERE ticket_id = ?",
            (ticket_id,),
        )
        row = cursor.fetchone()
        return _row_to_ticket(row) if row is not None else None

    def all_tickets(self) -> list[ExtractedTicket]:
        cursor = self._conn.execute(
            "SELECT ticket_id, category, sentiment, urgency, mentioned_product, summary "
            "FROM tickets ORDER BY ticket_id"
        )
        return [_row_to_ticket(row) for row in cursor.fetchall()]

    def count(self) -> int:
        cursor = self._conn.execute("SELECT COUNT(*) FROM tickets")
        return int(cursor.fetchone()[0])

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> TicketStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _ticket_to_row(ticket: ExtractedTicket) -> dict[str, Any]:
    return {
        "ticket_id": ticket.ticket_id,
        "category": ticket.category.value,
        "sentiment": ticket.sentiment.value,
        "urgency": ticket.urgency.value,
        "mentioned_product": ticket.mentioned_product,
        "summary": ticket.summary,
    }


def _row_to_ticket(row: sqlite3.Row) -> ExtractedTicket:
    return ExtractedTicket(
        ticket_id=row["ticket_id"],
        category=row["category"],
        sentiment=row["sentiment"],
        urgency=row["urgency"],
        mentioned_product=row["mentioned_product"],
        summary=row["summary"],
    )
