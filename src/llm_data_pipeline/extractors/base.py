"""Common interface shared by every extractor implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from llm_data_pipeline.schema import RawRecord


class ExtractionError(RuntimeError):
    """Raised when an extractor cannot produce structured fields for a record."""

    def __init__(self, record_id: str, message: str) -> None:
        self.record_id = record_id
        super().__init__(f"record {record_id!r}: {message}")


class Extractor(ABC):
    """Turns one unstructured :class:`RawRecord` into a field dict.

    Implementations return a plain ``dict`` (not a validated
    :class:`~llm_data_pipeline.schema.ExtractedTicket`) with keys matching
    the schema's extracted fields (``category``, ``sentiment``, ``urgency``,
    ``mentioned_product``, ``summary``). The pipeline's validate stage is
    responsible for constructing and validating the Pydantic model -
    extractors only promise to produce their best-effort structured guess,
    not a guaranteed-valid one.
    """

    @abstractmethod
    def extract(self, record: RawRecord) -> dict[str, Any]:
        """Extract structured fields from a single record.

        Raises:
            ExtractionError: if the extractor cannot produce any output at
                all (e.g. the LLM extractor exhausted its retry budget).
        """
        raise NotImplementedError

    def extract_many(
        self, records: list[RawRecord]
    ) -> list[tuple[RawRecord, dict[str, Any] | None, str | None]]:
        """Extract fields for a batch of records.

        Returns a list of ``(record, fields_or_none, error_or_none)`` so
        that one record's extraction failure never aborts the rest of the
        batch - the pipeline's validate/report stages need to see every
        attempted record, not just the successful ones.
        """
        results: list[tuple[RawRecord, dict[str, Any] | None, str | None]] = []
        for record in records:
            try:
                fields = self.extract(record)
            except ExtractionError as exc:
                results.append((record, None, str(exc)))
            else:
                results.append((record, fields, None))
        return results
