import pytest
from pydantic import ValidationError

from comparison.models import CompareRequest


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
