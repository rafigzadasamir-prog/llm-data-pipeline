"""LLM-backed extractor using the `anthropic` Python SDK.

Structured output is obtained via forced tool use: Claude must call a single
tool whose JSON-schema input matches the extracted-fields shape. A
strict-mode schema (`additionalProperties: false`, every property
`required`) keeps the response shape honest, but schema conformance alone
does not guarantee the response is semantically valid against our Pydantic
model (for example `summary` has a max length the JSON schema does not
enforce). So every tool call is still re-validated with Pydantic, and a
validation failure triggers "repair": the validation error is handed back
to Claude as a `tool_result` with `is_error=True`, in the same shape Claude
already expects for a failed tool call, and we ask it to call the tool
again with corrected arguments. This mirrors how a human reviewer would
bounce back bad input, and in practice resolves transient issues (an
out-of-taxonomy guess, a too-long summary) within one or two retries
without needing to re-send the whole prompt from scratch.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from pydantic import ValidationError

from llm_data_pipeline.extractors.base import ExtractionError, Extractor
from llm_data_pipeline.schema import (
    KNOWN_PRODUCTS,
    ExtractedTicket,
    RawRecord,
    Sentiment,
    TicketCategory,
    Urgency,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5"
_TOOL_NAME = "extract_ticket_fields"

# "" is used as the sentinel for "no product mentioned" rather than a
# nullable type, because strict-mode JSON schema (required for reliable
# structured tool-use output) does not support `type: ["string", "null"]`
# unions as cleanly as a single `string` type. ExtractedTicket's validator
# maps "" (and "none") back to `None`.
_TOOL_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": [c.value for c in TicketCategory],
            "description": "The single best-matching support ticket category.",
        },
        "sentiment": {
            "type": "string",
            "enum": [s.value for s in Sentiment],
            "description": "The customer's overall emotional tone.",
        },
        "urgency": {
            "type": "string",
            "enum": [u.value for u in Urgency],
            "description": "How urgently the ticket needs a response.",
        },
        "mentioned_product": {
            "type": "string",
            "description": (
                "The exact product name mentioned in the ticket, if any, "
                f"from this list: {', '.join(KNOWN_PRODUCTS)}. "
                "Use an empty string if no product from the list is mentioned."
            ),
        },
        "summary": {
            "type": "string",
            "description": "A one-sentence summary of the ticket, under 300 characters.",
        },
    },
    "required": ["category", "sentiment", "urgency", "mentioned_product", "summary"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = (
    "You are a support-ticket triage assistant. Read the customer message "
    "you are given and call the extract_ticket_fields tool exactly once "
    "with the structured fields that best describe it. Be decisive: pick "
    "the single closest category, sentiment, and urgency even if the "
    "message is ambiguous."
)


def _build_tool_definition() -> dict[str, Any]:
    return {
        "name": _TOOL_NAME,
        "description": "Record the structured fields extracted from a support ticket.",
        "input_schema": _TOOL_INPUT_SCHEMA,
        "strict": True,
    }


def _extract_tool_use_block(content: list[Any]) -> Any:
    for block in content:
        if getattr(block, "type", None) == "tool_use":
            return block
    return None


def _sanitize_tool_input(raw_input: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(raw_input)
    # The sentinel-to-None mapping also happens in ExtractedTicket's
    # validator, but doing it here too means the dict this extractor
    # returns (the public contract described in Extractor.extract) is
    # already clean, not just the pipeline's later validated model.
    product = sanitized.get("mentioned_product")
    if isinstance(product, str) and (not product.strip() or product.strip().lower() == "none"):
        sanitized["mentioned_product"] = None
    return sanitized


class LLMExtractor(Extractor):
    """Extracts ticket fields via the Anthropic Messages API (forced tool use)."""

    def __init__(
        self,
        client: Any = None,
        model: str | None = None,
        max_retries: int = 2,
    ) -> None:
        """Create an LLM-backed extractor.

        Args:
            client: An `anthropic.Anthropic`-compatible client. Built lazily
                from `ANTHROPIC_API_KEY` if not given - injected explicitly
                in tests so the retry/repair loop can be exercised without
                any network access.
            model: Model ID to call. Defaults to `ANTHROPIC_MODEL` env var,
                falling back to `DEFAULT_MODEL`.
            max_retries: Number of repair attempts after an initial
                validation failure (so at most `1 + max_retries` API calls
                per record).
        """
        if client is None:
            # Imported lazily: rule-based-only usage of this package never
            # needs to construct a real Anthropic client.
            import anthropic

            client = anthropic.Anthropic()
        self._client = client
        self._model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
        self._max_retries = max_retries
        self._tool = _build_tool_definition()

    def extract(self, record: RawRecord) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": f"Support ticket:\n\n{record.text}"}
        ]

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=1024,
                    system=_SYSTEM_PROMPT,
                    tools=[self._tool],
                    tool_choice={"type": "tool", "name": _TOOL_NAME},
                    messages=messages,
                )
            except Exception as exc:  # noqa: BLE001 - any SDK/network failure is non-retryable here
                raise ExtractionError(record.id, f"Anthropic API call failed: {exc}") from exc

            tool_use = _extract_tool_use_block(response.content)
            if tool_use is None:
                last_error = ExtractionError(record.id, "model response contained no tool_use block")
                break

            try:
                fields = _sanitize_tool_input(dict(tool_use.input))
                ExtractedTicket(ticket_id=record.id, **fields)  # validate; raises on bad data
            except (ValidationError, TypeError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning(
                    "record %s: attempt %d/%d produced invalid fields: %s",
                    record.id,
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                )
                if attempt >= self._max_retries:
                    break
                messages.append({"role": "assistant", "content": response.content})
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use.id,
                                "content": (
                                    "Your previous call produced invalid fields: "
                                    f"{exc}. Call extract_ticket_fields again with "
                                    "corrected values that satisfy the schema."
                                ),
                                "is_error": True,
                            }
                        ],
                    }
                )
                continue
            else:
                return fields

        raise ExtractionError(
            record.id,
            f"exhausted {self._max_retries + 1} attempt(s) without valid output: {last_error}",
        )
