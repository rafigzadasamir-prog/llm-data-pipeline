"""Loading the bundled raw-record dataset from disk."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_data_pipeline.ingestion import load_raw_records


def test_loads_valid_json_array(tmp_path: Path) -> None:
    data_path = tmp_path / "records.json"
    data_path.write_text(
        json.dumps([{"id": "TCK-0001", "text": "Hello"}, {"id": "TCK-0002", "text": "World"}]),
        encoding="utf-8",
    )

    records = load_raw_records(data_path)

    assert len(records) == 2
    assert records[0].id == "TCK-0001"
    assert records[1].text == "World"


def test_rejects_non_array_json(tmp_path: Path) -> None:
    data_path = tmp_path / "records.json"
    data_path.write_text(json.dumps({"id": "TCK-0001", "text": "Hello"}), encoding="utf-8")

    with pytest.raises(ValueError, match="expected a JSON array"):
        load_raw_records(data_path)


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_raw_records(Path("/no/such/file.json"))


def test_bundled_sample_dataset_loads_and_has_expected_size() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    records = load_raw_records(repo_root / "data" / "raw_tickets.json")
    assert 30 <= len(records) <= 50
    assert len({r.id for r in records}) == len(records)  # ids are unique
