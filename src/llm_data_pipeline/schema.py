"""Pydantic schema for the support-ticket extraction pipeline.

Every extractor (LLM-backed or rule-based) must ultimately produce data that
validates against :class:`ExtractedTicket`. Keeping the schema in one place
means the validation stage, the storage layer, and both extractors all agree
on exactly what "structured" means for this pipeline.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TicketCategory(StrEnum):
    """The support-ticket taxonomy this pipeline recognizes."""

    BILLING = "billing"
    TECHNICAL_ISSUE = "technical_issue"
    ACCOUNT_ACCESS = "account_access"
    FEATURE_REQUEST = "feature_request"
    SHIPPING = "shipping"
    CANCELLATION = "cancellation"
    GENERAL_INQUIRY = "general_inquiry"


class Sentiment(StrEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class Urgency(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# The product catalogue mentioned in the bundled sample dataset. Both
# extractors use this list to recognize `mentioned_product`: the rule-based
# one scans for it directly, and the LLM one is told about it in-prompt.
KNOWN_PRODUCTS: tuple[str, ...] = (
    "CloudSync Pro",
    "DataVault",
    "PayEase",
    "StreamBox",
    "InvoiceMate",
    "TaskFlow",
)


class RawRecord(BaseModel):
    """One unstructured input record before extraction."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class ExtractedTicket(BaseModel):
    """The structured output every extractor must produce for a ticket.

    `ticket_id` links back to the originating `RawRecord.id`; the remaining
    fields are the extracted structured data itself.
    """

    model_config = ConfigDict(extra="forbid")

    ticket_id: str = Field(min_length=1)
    category: TicketCategory
    sentiment: Sentiment
    urgency: Urgency
    mentioned_product: str | None = Field(default=None, max_length=100)
    summary: str = Field(min_length=1, max_length=300)

    @field_validator("mentioned_product")
    @classmethod
    def _blank_product_is_none(cls, value: str | None) -> str | None:
        # The LLM extractor's tool schema uses "" as a sentinel for "no
        # product mentioned" because strict JSON-schema tool use does not
        # support nullable string types cleanly. Normalize that here so
        # every downstream consumer only ever sees a real value or None.
        if value is None:
            return None
        stripped = value.strip()
        if not stripped or stripped.lower() == "none":
            return None
        return stripped

    @field_validator("summary")
    @classmethod
    def _summary_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("summary must not be blank")
        return stripped
