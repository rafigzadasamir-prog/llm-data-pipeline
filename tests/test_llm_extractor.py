"""LLMExtractor's retry-and-repair loop, exercised entirely against a fake client.

No real `anthropic.Anthropic()` client is ever constructed here (the
client is injected), so this test makes zero network calls - it verifies
the repair-loop *logic* only: on a validation failure, the error is fed
back to the model as a `tool_result` with `is_error=True`, and the loop
gives up after `max_retries` repair attempts.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from llm_data_pipeline.extractors.base import ExtractionError
from llm_data_pipeline.extractors.llm import LLMExtractor
from llm_data_pipeline.schema import RawRecord


def _tool_use_response(tool_input: dict[str, Any], tool_use_id: str = "toolu_1") -> SimpleNamespace:
    block = SimpleNamespace(type="tool_use", input=tool_input, id=tool_use_id)
    return SimpleNamespace(content=[block])


class FakeMessagesAPI:
    """Stands in for `client.messages`, returning queued canned responses."""

    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("FakeMessagesAPI ran out of queued responses")
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.messages = FakeMessagesAPI(responses)


VALID_FIELDS = {
    "category": "billing",
    "sentiment": "negative",
    "urgency": "high",
    "mentioned_product": "PayEase",
    "summary": "Customer reports a billing discrepancy.",
}


def test_valid_first_response_returns_fields_without_retry() -> None:
    client = FakeClient([_tool_use_response(VALID_FIELDS)])
    extractor = LLMExtractor(client=client, model="claude-test-model")
    record = RawRecord(id="TCK-0001", text="I was overcharged on my PayEase invoice.")

    fields = extractor.extract(record)

    assert fields["category"] == "billing"
    assert fields["mentioned_product"] == "PayEase"
    assert len(client.messages.calls) == 1
    assert client.messages.calls[0]["model"] == "claude-test-model"
    assert client.messages.calls[0]["tool_choice"] == {"type": "tool", "name": "extract_ticket_fields"}


def test_empty_product_sentinel_normalizes_to_none() -> None:
    fields_with_sentinel = {**VALID_FIELDS, "mentioned_product": ""}
    client = FakeClient([_tool_use_response(fields_with_sentinel)])
    extractor = LLMExtractor(client=client)
    record = RawRecord(id="TCK-0002", text="General question, no product mentioned.")

    fields = extractor.extract(record)

    assert fields["mentioned_product"] is None


def test_invalid_category_triggers_repair_then_succeeds() -> None:
    invalid_response = _tool_use_response({**VALID_FIELDS, "category": "not_a_real_category"})
    valid_response = _tool_use_response(VALID_FIELDS)
    client = FakeClient([invalid_response, valid_response])
    extractor = LLMExtractor(client=client, max_retries=2)
    record = RawRecord(id="TCK-0003", text="Billing issue.")

    fields = extractor.extract(record)

    assert fields == VALID_FIELDS
    assert len(client.messages.calls) == 2

    # The repair call must carry the failed assistant turn plus an
    # is_error tool_result describing what went wrong, so the model has
    # enough context to self-correct.
    repair_messages = client.messages.calls[1]["messages"]
    assert repair_messages[-1]["role"] == "user"
    tool_result = repair_messages[-1]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["is_error"] is True
    assert tool_result["tool_use_id"] == "toolu_1"
    assert "invalid fields" in tool_result["content"]


def test_exhausting_retries_raises_extraction_error() -> None:
    always_invalid = {**VALID_FIELDS, "urgency": "not_a_real_urgency"}
    client = FakeClient([_tool_use_response(always_invalid) for _ in range(2)])
    extractor = LLMExtractor(client=client, max_retries=1)
    record = RawRecord(id="TCK-0004", text="Some ticket.")

    with pytest.raises(ExtractionError) as exc_info:
        extractor.extract(record)

    assert exc_info.value.record_id == "TCK-0004"
    assert len(client.messages.calls) == 2  # initial attempt + 1 retry, then give up


def test_no_tool_use_block_raises_extraction_error() -> None:
    text_only_response = SimpleNamespace(content=[SimpleNamespace(type="text", text="I decline.")])
    client = FakeClient([text_only_response])
    extractor = LLMExtractor(client=client, max_retries=2)
    record = RawRecord(id="TCK-0005", text="Some ticket.")

    with pytest.raises(ExtractionError):
        extractor.extract(record)


def test_api_failure_raises_extraction_error_without_retry() -> None:
    class ExplodingMessagesAPI:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs: Any) -> SimpleNamespace:
            self.calls += 1
            raise RuntimeError("simulated network failure")

    class ExplodingClient:
        def __init__(self) -> None:
            self.messages = ExplodingMessagesAPI()

    client = ExplodingClient()
    extractor = LLMExtractor(client=client, max_retries=2)
    record = RawRecord(id="TCK-0006", text="Some ticket.")

    with pytest.raises(ExtractionError):
        extractor.extract(record)

    assert client.messages.calls == 1


def test_model_defaults_to_env_var_when_not_specified(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-env-model")
    client = FakeClient([_tool_use_response(VALID_FIELDS)])
    extractor = LLMExtractor(client=client)
    record = RawRecord(id="TCK-0007", text="Some ticket.")

    extractor.extract(record)

    assert client.messages.calls[0]["model"] == "claude-env-model"
