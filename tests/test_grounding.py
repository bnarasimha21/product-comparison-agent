from comparison.grounding import ground_highlights, ground_rate, is_grounded
from comparison.models import Highlight, Page


def _page(url: str, text: str) -> Page:
    return Page(url=url, text=text, ok=True)


def test_exact_match_grounded():
    page = _page("https://a.example/", "WidgetPro costs $19 per month.")
    highlight = Highlight(text="costs $19/mo", source_url=page.url, quote="costs $19 per month")
    assert is_grounded(highlight, {page.url: page})


def test_case_difference_still_grounded():
    page = _page("https://a.example/", "WidgetPro Costs $19 Per Month.")
    highlight = Highlight(text="x", source_url=page.url, quote="costs $19 per month")
    assert is_grounded(highlight, {page.url: page})


def test_whitespace_difference_still_grounded():
    page = _page("https://a.example/", "WidgetPro costs\n$19   per month.")
    highlight = Highlight(text="x", source_url=page.url, quote="costs $19 per month")
    assert is_grounded(highlight, {page.url: page})


def test_punctuation_difference_not_grounded():
    """Punctuation is not normalized (only whitespace/case): a quote that
    adds or drops punctuation the page does not have is correctly rejected
    as an inexact match, not silently accepted."""
    page = _page("https://a.example/", "WidgetPro costs $19 per month")
    highlight = Highlight(text="x", source_url=page.url, quote="costs $19, per month.")
    assert not is_grounded(highlight, {page.url: page})


def test_quote_spanning_two_pages_not_grounded():
    page_a = _page("https://a.example/", "WidgetPro costs $19 per month.")
    page_b = _page("https://b.example/", "WidgetPro supports dark mode.")
    highlight = Highlight(
        text="x", source_url=page_a.url, quote="costs $19 per month. WidgetPro supports dark mode"
    )
    assert not is_grounded(highlight, {page_a.url: page_a, page_b.url: page_b})


def test_unfetched_source_not_grounded():
    highlight = Highlight(text="x", source_url="https://never-fetched.example/", quote="anything")
    assert not is_grounded(highlight, {})


def test_failed_fetch_source_not_grounded():
    page = Page(url="https://a.example/", text="", ok=False, error="HTTP 500")
    highlight = Highlight(text="x", source_url=page.url, quote="anything")
    assert not is_grounded(highlight, {page.url: page})


def test_empty_quote_not_grounded():
    page = _page("https://a.example/", "some text")
    highlight = Highlight(text="x", source_url=page.url, quote="   ")
    assert not is_grounded(highlight, {page.url: page})


def test_ground_highlights_splits_grounded_and_dropped():
    page = _page("https://a.example/", "WidgetPro costs $19 per month.")
    real = Highlight(text="price", source_url=page.url, quote="costs $19 per month")
    fabricated = Highlight(text="fake", source_url=page.url, quote="made up claim not on the page")
    grounded, dropped = ground_highlights([real, fabricated], {page.url: page})
    assert grounded == [real]
    assert len(dropped) == 1
    assert dropped[0]["text"] == "fake"


def test_ground_rate():
    assert ground_rate(0, 0) is None
    assert ground_rate(2, 1) == 0.5
    assert ground_rate(4, 4) == 1.0


def test_single_common_word_quote_not_grounded():
    """The exact attack scenario from the adversarial review: a model
    proposes a plausible-sounding `text` paraphrase but cites a quote that
    is just one common word, which is a substring of nearly any page. This
    must be rejected on quote length alone, regardless of what `text` says,
    and regardless of whether `text` happens to appear on the page too."""
    page = _page("https://a.example/", "Fully waterproof to 50m, the manual says so.")
    highlight = Highlight(
        text="Fully waterproof to 50m", source_url=page.url, quote="the"
    )
    assert not is_grounded(highlight, {page.url: page})


def test_short_two_word_quote_below_char_minimum_not_grounded():
    page = _page("https://a.example/", "It is on sale.")
    highlight = Highlight(text="x", source_url=page.url, quote="on sale")
    assert not is_grounded(highlight, {page.url: page})
