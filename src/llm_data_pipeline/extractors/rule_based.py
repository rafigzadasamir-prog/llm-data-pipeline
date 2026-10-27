"""Deterministic, offline extractor used as the default and in tests.

No network calls, no API key, no randomness: the same input text always
produces the same output fields. That determinism is exactly what lets the
test suite exercise the full pipeline (ingest -> extract -> validate ->
store -> report) without ever touching the Anthropic API.
"""

from __future__ import annotations

import re
from typing import Any

from llm_data_pipeline.extractors.base import Extractor
from llm_data_pipeline.schema import KNOWN_PRODUCTS, RawRecord

# Keyword lists are checked in this fixed order per category group. Order
# matters because tickets often contain overlapping vocabulary (e.g. a
# cancellation request mentions "subscription", which is also a billing
# word) - the first category whose keywords match wins, so the list below
# is ordered from most-specific intent to most-generic fallback.
_CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    (
        "cancellation",
        (
            "cancel my",
            "cancel the",
            "cancel this subscription",
            "unsubscribe",
            "terminate my",
            "close my account",
            "stop billing me",
        ),
    ),
    (
        "account_access",
        (
            "log in",
            "login",
            "locked out",
            "can't access",
            "cannot access",
            "two-factor",
            "2fa",
            "reset my password",
            "forgot my password",
            "sign in",
        ),
    ),
    (
        "technical_issue",
        (
            "bug",
            "crash",
            "crashed",
            "crashing",
            "error message",
            "not working",
            "stopped working",
            "broken",
            "freezes",
            "freezing",
            "glitch",
            "malfunction",
        ),
    ),
    (
        "shipping",
        (
            "delivery",
            "shipment",
            "package",
            "tracking number",
            "hasn't arrived",
            "has not arrived",
            "shipped",
            "courier",
            "delayed delivery",
        ),
    ),
    (
        "feature_request",
        (
            "feature request",
            "would be great if",
            "could you add",
            "please add",
            "suggestion",
            "add support for",
            "wish list",
            "feature idea",
        ),
    ),
    (
        "billing",
        (
            "invoice",
            "billed",
            "billing",
            "overcharged",
            "charge",
            "charged",
            "payment failed",
            "subscription price",
            "refund",
        ),
    ),
]

_NEGATIVE_WORDS: tuple[str, ...] = (
    "terrible",
    "awful",
    "frustrated",
    "frustrating",
    "angry",
    "disappointed",
    "worst",
    "horrible",
    "unacceptable",
    "annoyed",
    "annoying",
    "furious",
    "ridiculous",
    "upset",
)

_POSITIVE_WORDS: tuple[str, ...] = (
    "great",
    "love",
    "thanks",
    "thank you",
    "awesome",
    "excellent",
    "happy",
    "appreciate",
    "fantastic",
    "pleased",
    "wonderful",
)

# Checked in this order: an explicit "not urgent" phrase should win over a
# generic "soon", and an explicit emergency phrase should win over both.
_CRITICAL_WORDS: tuple[str, ...] = (
    "immediately",
    "critical",
    "urgent",
    "right now",
    "asap",
    "emergency",
    "as soon as possible",
)

_LOW_URGENCY_WORDS: tuple[str, ...] = (
    "no rush",
    "no hurry",
    "whenever you can",
    "whenever you get a chance",
    "low priority",
    "not urgent",
    "take your time",
)

_HIGH_URGENCY_WORDS: tuple[str, ...] = (
    "soon",
    "quickly",
    "important",
    "promptly",
    "time-sensitive",
)


def _contains_any(haystack_lower: str, needles: tuple[str, ...]) -> bool:
    return any(needle in haystack_lower for needle in needles)


def _classify_category(text_lower: str) -> str:
    for category, keywords in _CATEGORY_KEYWORDS:
        if _contains_any(text_lower, keywords):
            return category
    return "general_inquiry"


def _classify_sentiment(text_lower: str) -> str:
    negative_hits = sum(1 for word in _NEGATIVE_WORDS if word in text_lower)
    positive_hits = sum(1 for word in _POSITIVE_WORDS if word in text_lower)
    if negative_hits > positive_hits:
        return "negative"
    if positive_hits > negative_hits:
        return "positive"
    return "neutral"


def _classify_urgency(text_lower: str) -> str:
    if _contains_any(text_lower, _CRITICAL_WORDS):
        return "critical"
    if _contains_any(text_lower, _LOW_URGENCY_WORDS):
        return "low"
    if _contains_any(text_lower, _HIGH_URGENCY_WORDS):
        return "high"
    return "medium"


def _find_mentioned_product(text: str) -> str | None:
    text_lower = text.lower()
    for product in KNOWN_PRODUCTS:
        if product.lower() in text_lower:
            return product
    return None


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_SUMMARY_MAX_LEN = 117  # leaves room for the "..." suffix within 120 chars


def _summarize(text: str) -> str:
    stripped = text.strip()
    sentences = _SENTENCE_SPLIT_RE.split(stripped)
    first_sentence = sentences[0].strip() if sentences else ""
    candidate = first_sentence if first_sentence else stripped
    if len(candidate) <= _SUMMARY_MAX_LEN:
        return candidate
    return candidate[:_SUMMARY_MAX_LEN].rstrip() + "..."


class RuleBasedExtractor(Extractor):
    """Keyword/regex heuristic extractor - deterministic, offline, no API key."""

    def extract(self, record: RawRecord) -> dict[str, Any]:
        text_lower = record.text.lower()
        return {
            "category": _classify_category(text_lower),
            "sentiment": _classify_sentiment(text_lower),
            "urgency": _classify_urgency(text_lower),
            "mentioned_product": _find_mentioned_product(record.text),
            "summary": _summarize(record.text),
        }

    def extract_many(
        self, records: list[RawRecord]
    ) -> list[tuple[RawRecord, dict[str, Any] | None, str | None]]:
        # Pure and side-effect free, so a failure in one record would be a
        # bug rather than an expected runtime condition - no per-record
        # try/except needed here, unlike the LLM extractor's batch path.
        return [(record, self.extract(record), None) for record in records]
