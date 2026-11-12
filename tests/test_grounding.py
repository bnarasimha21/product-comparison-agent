from comparison.grounding import ground_highlights, ground_rate, is_grounded
from comparison.models import Highlight, Page


def _page(url: str, text: str) -> Page:
    return Page(url=url, text=text, ok=True)


def test_exact_match_grounded():
    page = _page("https://a.example/", "WidgetPro costs $19 per month.")
    highlight = Highlight(text="costs $19/mo", source_url=page.url, quote="WidgetPro costs $19 per month")
    assert is_grounded(highlight, {page.url: page})


def test_case_difference_still_grounded():
    page = _page("https://a.example/", "WidgetPro Costs $19 Per Month.")
    highlight = Highlight(text="x", source_url=page.url, quote="widgetpro costs $19 per month")
    assert is_grounded(highlight, {page.url: page})


def test_whitespace_difference_still_grounded():
    page = _page("https://a.example/", "WidgetPro costs\n$19   per month.")
    highlight = Highlight(text="x", source_url=page.url, quote="WidgetPro costs $19 per month")
    assert is_grounded(highlight, {page.url: page})


def test_punctuation_difference_not_grounded():
    """Punctuation is not normalized (only whitespace/case): a quote that
    adds or drops punctuation the page does not have is correctly rejected
    as an inexact match, not silently accepted."""
    page = _page("https://a.example/", "WidgetPro costs $19 per month")
    highlight = Highlight(text="x", source_url=page.url, quote="WidgetPro costs $19, per month.")
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
    real = Highlight(text="price", source_url=page.url, quote="WidgetPro costs $19 per month")
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


def test_reviewer_cited_connector_phrases_not_grounded():
    """The exact bypass confirmed in the second adversarial review: a bare
    connector phrase that carries no actual claim passes as "evidence"
    because it is a substring of almost any product page. "the phone is"
    and "is one of the" must be rejected because each has at most one word
    that isn't filler/stopword, below the substantive-word floor."""
    page = _page(
        "https://a.example/",
        "The phone is one of the best devices on the market this year.",
    )
    for quote in ("the phone is", "is one of the"):
        highlight = Highlight(text="x", source_url=page.url, quote=quote)
        assert not is_grounded(highlight, {page.url: page}), quote


def test_short_genuine_factual_quotes_are_grounded():
    """THE REGRESSION (third adversarial review): the previous pure
    length-based floor (3 words / 20 chars) overshot and rejected these
    genuine short factual quotes even though they are real, substantive
    facts and each is well above the connector-phrase bar in substance.
    This caused fake-mode eval fact_recall/ground_rate to drop from 1.00 to
    well below that. All four must be grounded."""
    cases = [
        ("https://a.example/", "The iPhone weighs 171 grams.", "weighs 171 grams"),
        ("https://b.example/", "It has a 6.1-inch display.", "6.1-inch display"),
        ("https://c.example/", "It ships with USB-C.", "ships with usb-c"),
        ("https://d.example/", "WidgetPro costs $19 per month.", "costs $19 per month"),
    ]
    for url, page_text, quote in cases:
        page = _page(url, page_text)
        highlight = Highlight(text="x", source_url=url, quote=quote)
        assert is_grounded(highlight, {url: page}), quote


def test_marketing_filler_quotes_still_not_grounded():
    """The floor must keep rejecting real junk that happens to be a
    substring of a page, even though it's of similar or greater length
    than the genuine short quotes above."""
    cases = [
        (
            "https://a.example/",
            "This is on the market this year, available everywhere.",
            "on the market this year",
        ),
        (
            "https://b.example/",
            "By using this site, terms and conditions apply to you.",
            "terms and conditions apply",
        ),
        (
            "https://c.example/",
            "For more info, click here to learn more about pricing.",
            "click here to learn more",
        ),
        (
            "https://d.example/",
            "Copyright 2026, all rights reserved.",
            "all rights reserved.",
        ),
        (
            "https://e.example/",
            "Ratings: 1 2 3 4 5 6 7 8 9 10 out of 10 reviewers liked it.",
            "1 2 3 4 5 6 7 8 9 10",
        ),
    ]
    for url, page_text, quote in cases:
        page = _page(url, page_text)
        highlight = Highlight(text="x", source_url=url, quote=quote)
        assert not is_grounded(highlight, {url: page}), quote


def test_mid_word_fragment_not_grounded():
    page = _page("https://a.example/", "The phone is one of the best on the market.")
    highlight = Highlight(text="x", source_url=page.url, quote="e phone is one of the")
    assert not is_grounded(highlight, {page.url: page})


def test_negated_source_rejects_quote_that_flips_meaning():
    """The exact negation-flip bypass cited in the third adversarial
    review: the quote is a real, verbatim substring of the source, but the
    source negates it just outside the quoted span, so the quote asserts
    the OPPOSITE of what the source actually says. Both reviewer-cited
    examples must be rejected as NOT grounded."""
    page1 = _page(
        "https://a.example/", "It does not support wireless charging at all."
    )
    highlight1 = Highlight(
        text="x", source_url=page1.url, quote="support wireless charging at all"
    )
    assert not is_grounded(highlight1, {page1.url: page1})

    page2 = _page(
        "https://b.example/", "It is not the cheapest phone on the market."
    )
    highlight2 = Highlight(
        text="x", source_url=page2.url, quote="the cheapest phone on the market"
    )
    assert not is_grounded(highlight2, {page2.url: page2})


def test_non_negated_quote_still_grounded():
    """Sanity check that the negation-proximity check doesn't start
    rejecting ordinary, non-negated quotes."""
    page = _page("https://a.example/", "It does support wireless charging fully.")
    highlight = Highlight(
        text="x", source_url=page.url, quote="support wireless charging fully"
    )
    assert is_grounded(highlight, {page.url: page})
