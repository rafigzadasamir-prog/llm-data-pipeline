"""RuleBasedExtractor output on known fixture inputs.

Every case here pins down a specific keyword path so a future change to
the keyword lists in `rule_based.py` has to consciously decide whether it
is allowed to change these outputs.
"""

from __future__ import annotations

from llm_data_pipeline.extractors.rule_based import RuleBasedExtractor
from llm_data_pipeline.schema import KNOWN_PRODUCTS, RawRecord

extractor = RuleBasedExtractor()


def test_cancellation_negative_critical_with_product() -> None:
    record = RawRecord(
        id="FIX-01",
        text="This is absolutely terrible - I want to cancel my DataVault subscription immediately.",
    )
    fields = extractor.extract(record)
    assert fields["category"] == "cancellation"
    assert fields["sentiment"] == "negative"
    assert fields["urgency"] == "critical"
    assert fields["mentioned_product"] == "DataVault"
    assert fields["summary"] == record.text


def test_account_access_neutral_low_no_product() -> None:
    record = RawRecord(
        id="FIX-02",
        text="I can't access my dashboard this morning, no rush, just figured I'd mention it.",
    )
    fields = extractor.extract(record)
    assert fields["category"] == "account_access"
    assert fields["sentiment"] == "neutral"
    assert fields["urgency"] == "low"
    assert fields["mentioned_product"] is None


def test_technical_issue_positive_high_with_product() -> None:
    record = RawRecord(
        id="FIX-03",
        text=(
            "The TaskFlow app keeps crashing, but thanks for looking into it so "
            "quickly, I really appreciate it."
        ),
    )
    fields = extractor.extract(record)
    assert fields["category"] == "technical_issue"
    assert fields["sentiment"] == "positive"
    assert fields["urgency"] == "high"
    assert fields["mentioned_product"] == "TaskFlow"


def test_shipping_category_detected() -> None:
    record = RawRecord(
        id="FIX-04",
        text="My package was marked as shipped but the tracking number has not updated in a week.",
    )
    fields = extractor.extract(record)
    assert fields["category"] == "shipping"


def test_feature_request_category_detected() -> None:
    record = RawRecord(
        id="FIX-05",
        text="Feature request: please add support for exporting reports as PDF.",
    )
    fields = extractor.extract(record)
    assert fields["category"] == "feature_request"


def test_billing_category_detected() -> None:
    record = RawRecord(
        id="FIX-06",
        text="My invoice this month looks wrong and I was charged for a plan I never selected.",
    )
    fields = extractor.extract(record)
    assert fields["category"] == "billing"


def test_fallback_to_general_inquiry_with_long_text_summary_truncated() -> None:
    long_text = (
        "I have been trying to get help with an issue for the past three days and no "
        "one from your support team has responded despite multiple emails and follow "
        "ups sent through every channel available to me which is honestly quite frustrating"
    )
    record = RawRecord(id="FIX-07", text=long_text)
    fields = extractor.extract(record)

    assert fields["category"] == "general_inquiry"
    assert fields["sentiment"] == "negative"  # "frustrating"
    assert fields["urgency"] == "medium"  # no urgency keywords present
    assert fields["mentioned_product"] is None

    summary = fields["summary"]
    assert len(summary) <= 120
    assert summary.endswith("...")
    assert long_text.startswith(summary[:-3])


def test_product_detection_follows_known_products_order_not_text_order() -> None:
    # TaskFlow appears first in the text, but DataVault comes first in
    # KNOWN_PRODUCTS, so DataVault should win.
    assert KNOWN_PRODUCTS.index("DataVault") < KNOWN_PRODUCTS.index("TaskFlow")
    record = RawRecord(id="FIX-08", text="Both TaskFlow and DataVault seem to be having issues today.")
    fields = extractor.extract(record)
    assert fields["mentioned_product"] == "DataVault"


def test_extraction_is_deterministic_across_calls() -> None:
    record = RawRecord(id="FIX-09", text="Please cancel my PayEase account, this is urgent.")
    first = extractor.extract(record)
    second = extractor.extract(record)
    assert first == second
