"""Post-pipeline data-quality checks over whatever is currently in storage.

Deliberately separate from the pipeline itself: this module only reads
already-stored :class:`ExtractedTicket` rows and reports on them, so it can
be run independently (e.g. against a database from a previous run) without
re-ingesting or re-extracting anything.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_data_pipeline.schema import ExtractedTicket, Sentiment, TicketCategory, Urgency

# Fields that are required on every ticket (never expected to be null).
# mentioned_product is intentionally excluded - it's a legitimately
# optional field, not a completeness problem.
_REQUIRED_FIELDS = ("category", "sentiment", "urgency", "summary")
_ALL_FIELDS = (*_REQUIRED_FIELDS, "mentioned_product")


@dataclass
class QualityReport:
    total_records: int
    completeness: dict[str, float]  # field -> share of records with a non-null value
    category_distribution: dict[str, int]
    sentiment_distribution: dict[str, int]
    urgency_distribution: dict[str, int]
    sanity_violations: list[str]

    @property
    def passed(self) -> bool:
        return not self.sanity_violations

    def format(self) -> str:
        lines = [
            "Data Quality Report",
            "====================",
            f"total records: {self.total_records}",
            "",
            "Completeness by field:",
        ]
        for field_name, rate in self.completeness.items():
            lines.append(f"  {field_name:<20} {rate:.1%}")

        lines += ["", "Category distribution:"]
        for category, count in sorted(self.category_distribution.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {category:<20} {count}")

        lines += ["", "Sentiment distribution:"]
        for sentiment, count in sorted(self.sentiment_distribution.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {sentiment:<20} {count}")

        lines += ["", "Urgency distribution:"]
        for urgency, count in sorted(self.urgency_distribution.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {urgency:<20} {count}")

        lines += ["", "Sanity checks:"]
        if self.sanity_violations:
            for violation in self.sanity_violations:
                lines.append(f"  FAIL: {violation}")
        else:
            lines.append("  all checks passed")

        return "\n".join(lines)


def compute_completeness(tickets: list[ExtractedTicket]) -> dict[str, float]:
    total = len(tickets)
    if total == 0:
        return {field_name: 0.0 for field_name in _ALL_FIELDS}
    result: dict[str, float] = {}
    for field_name in _ALL_FIELDS:
        non_null = sum(1 for t in tickets if getattr(t, field_name) not in (None, ""))
        result[field_name] = non_null / total
    return result


def _count_by(tickets: list[ExtractedTicket], field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ticket in tickets:
        value = getattr(ticket, field_name).value
        counts[value] = counts.get(value, 0) + 1
    return counts


def run_sanity_checks(tickets: list[ExtractedTicket]) -> list[str]:
    """Defense-in-depth checks. These should never fire given Pydantic
    validation already happened before storage, but a quality report that
    merely trusts its input silently stops being useful the day that
    assumption breaks - so it re-derives the allowed value sets from the
    schema enums and checks every stored row against them directly.
    """
    violations: list[str] = []
    allowed_categories = {c.value for c in TicketCategory}
    allowed_sentiments = {s.value for s in Sentiment}
    allowed_urgencies = {u.value for u in Urgency}

    for ticket in tickets:
        if ticket.category.value not in allowed_categories:
            violations.append(f"{ticket.ticket_id}: category {ticket.category.value!r} not in allowed set")
        if ticket.sentiment.value not in allowed_sentiments:
            violations.append(f"{ticket.ticket_id}: sentiment {ticket.sentiment.value!r} not in allowed set")
        if ticket.urgency.value not in allowed_urgencies:
            violations.append(f"{ticket.ticket_id}: urgency {ticket.urgency.value!r} not in allowed set")
        if not ticket.summary.strip():
            violations.append(f"{ticket.ticket_id}: summary is blank")

    return violations


def build_quality_report(tickets: list[ExtractedTicket]) -> QualityReport:
    return QualityReport(
        total_records=len(tickets),
        completeness=compute_completeness(tickets),
        category_distribution=_count_by(tickets, "category"),
        sentiment_distribution=_count_by(tickets, "sentiment"),
        urgency_distribution=_count_by(tickets, "urgency"),
        sanity_violations=run_sanity_checks(tickets),
    )
