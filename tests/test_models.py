import pytest
from pydantic import ValidationError

from comparison.models import MAX_ASPECT_CHARS, CompareRequest, ComparisonRow


def test_valid_request():
    request = CompareRequest(product1="iPhone 15", product2="Galaxy S24", num_urls=2, num_highlights=3)
    assert request.product1 == "iPhone 15"
    assert request.num_urls == 2


def test_identical_products_rejected():
    with pytest.raises(ValidationError):
        CompareRequest(product1="A", product2="A", num_urls=2, num_highlights=2)


def test_identical_products_case_insensitive_rejected():
    with pytest.raises(ValidationError):
        CompareRequest(product1="widget", product2="Widget", num_urls=2, num_highlights=2)


def test_empty_product_name_rejected():
    with pytest.raises(ValidationError):
        CompareRequest(product1="", product2="B", num_urls=2, num_highlights=2)


def test_whitespace_only_product_name_rejected():
    with pytest.raises(ValidationError):
        CompareRequest(product1="   ", product2="B", num_urls=2, num_highlights=2)


@pytest.mark.parametrize("num_urls", [0, 6, -1])
def test_num_urls_out_of_range_rejected(num_urls):
    with pytest.raises(ValidationError):
        CompareRequest(product1="A", product2="B", num_urls=num_urls, num_highlights=2)


@pytest.mark.parametrize("num_highlights", [0, 11, -1])
def test_num_highlights_out_of_range_rejected(num_highlights):
    with pytest.raises(ValidationError):
        CompareRequest(product1="A", product2="B", num_urls=2, num_highlights=num_highlights)


def test_comparison_row_aspect_truncated_to_max_length():
    """Regression for issue 3: without a length cap, a model could smuggle
    a full, unverified claim into the aspect label itself (e.g. "Battery:
    Alpha lasts 48 hours, Beta only 10"), which would render right next to
    grounded a_text/b_text and read as if it were equally verified even
    though nothing ever checked it against a source. Truncating to
    MAX_ASPECT_CHARS keeps aspect a short label, structurally unable to
    carry a sentence-length claim."""
    adversarial = (
        "Battery: Alpha lasts a full 48 hours on a single charge while Beta "
        "barely makes it past 10, according to independent lab testing"
    )
    row = ComparisonRow(aspect=adversarial)
    assert len(row.aspect) <= MAX_ASPECT_CHARS
    assert row.aspect == adversarial[:MAX_ASPECT_CHARS]


def test_comparison_row_short_aspect_unchanged():
    row = ComparisonRow(aspect="Battery life")
    assert row.aspect == "Battery life"


def test_comparison_row_aspect_whitespace_collapsed_before_truncation():
    row = ComparisonRow(aspect="  Price   and   Value  ")
    assert row.aspect == "Price and Value"
