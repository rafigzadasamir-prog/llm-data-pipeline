"""Loads the bundled unstructured dataset from disk (no runtime download)."""

from __future__ import annotations

import json
from pathlib import Path

from llm_data_pipeline.schema import RawRecord


def load_raw_records(path: str | Path) -> list[RawRecord]:
    """Load raw text records from a JSON file.

    Expects a JSON array of objects with ``id`` and ``text`` keys, e.g.
    ``[{"id": "TCK-0001", "text": "..."}]`` - the format the bundled
    ``data/raw_tickets.json`` sample dataset uses.
    """
    raw_path = Path(path)
    with raw_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"{raw_path}: expected a JSON array of records")
    return [RawRecord(**item) for item in payload]
