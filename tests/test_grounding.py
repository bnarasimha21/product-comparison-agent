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


# --- Round-4 review fixtures: clause/sentence-boundary negation rework ---


def test_curly_apostrophe_negation_rejected():
    """A curly right single quotation mark (U+2019) in "doesn't" must be
    normalized to a straight apostrophe before the negation regex runs, so
    this is caught exactly like the straight-quote version."""
    page = _page("https://a.example/", "It doesn’t support wireless charging.")
    highlight = Highlight(text="x", source_url=page.url, quote="support wireless charging")
    assert not is_grounded(highlight, {page.url: page})


def test_cannot_negation_rejected():
    """"cannot" must be caught as its own negation word -- `\\bnot\\b`
    alone does not match inside "cannot" since there's no word boundary
    before "not" there."""
    page = _page("https://a.example/", "This device cannot connect to 5GHz networks.")
    highlight = Highlight(text="x", source_url=page.url, quote="connect to 5ghz networks")
    assert not is_grounded(highlight, {page.url: page})


def test_abbreviation_period_quote_not_falsely_rejected():
    """A quote containing an abbreviation period ("U.S.") with no actual
    negation nearby must still ground -- the clause splitter must not
    choke on the period inside the quote itself."""
    page = _page("https://a.example/", "It ships with the U.S. charger adapter.")
    highlight = Highlight(text="x", source_url=page.url, quote="ships with the u.s. charger adapter")
    assert is_grounded(highlight, {page.url: page})


def test_decimal_number_quote_not_falsely_rejected():
    """A quote containing a decimal ("v2.0") with no actual negation
    nearby must still ground."""
    page = _page("https://a.example/", "It supports v2.0 noise cancellation on both earbuds.")
    highlight = Highlight(text="x", source_url=page.url, quote="supports v2.0 noise cancellation")
    assert is_grounded(highlight, {page.url: page})


def test_comma_separated_true_quote_still_grounded():
    """The exact "commas aren't a cut point" bypass from the round-4
    review: "No hidden fees" is a separate, true claim ahead of a comma,
    not a negation of the quote that follows it. The clause boundary must
    stop at the comma so "No hidden fees, it costs $19 per month" doesn't
    get the real "costs $19 per month" quote rejected just because an
    unrelated "No" sits a few words further back in the same sentence."""
    page = _page("https://a.example/", "No hidden fees, it costs $19 per month.")
    highlight = Highlight(text="x", source_url=page.url, quote="costs $19 per month")
    assert is_grounded(highlight, {page.url: page})


def test_trailing_negation_after_quote_rejected():
    """The round-4 reviewer's example: the negation follows the quote,
    separated by commas, rather than preceding it. "supports wireless
    charging" is a real substring and nothing immediately before it is
    negated, but the rest of the sentence after it ("...which is not
    true") negates the claim, so it must still be rejected."""
    page = _page(
        "https://a.example/",
        "It supports wireless charging, or so the ad claims, which is not true.",
    )
    highlight = Highlight(text="x", source_url=page.url, quote="supports wireless charging")
    assert not is_grounded(highlight, {page.url: page})


def test_negation_flip_cases_still_rejected_after_rework():
    """The two original round-3 negation-flip cases must still be rejected
    after the clause-boundary rework (not just under the old fixed-window
    approach)."""
    page1 = _page("https://a.example/", "It does not support wireless charging at all.")
    highlight1 = Highlight(text="x", source_url=page1.url, quote="support wireless charging at all")
    assert not is_grounded(highlight1, {page1.url: page1})

    page2 = _page("https://b.example/", "It is not the cheapest phone on the market.")
    highlight2 = Highlight(text="x", source_url=page2.url, quote="the cheapest phone on the market")
    assert not is_grounded(highlight2, {page2.url: page2})


def test_page_number_boilerplate_with_digit_still_not_grounded():
    """Round-4 review: a digit next to a stopword/filler word must not
    pass the floor just because a digit is present. "weighs 171 grams" /
    "6.1-inch display" must still pass (covered elsewhere) since
    "grams"/"inch" are substantive, not filler."""
    page = _page("https://a.example/", "Page 1 of 2. All rights reserved. 2024.")
    for quote in ("page 1 of 2", "all rights reserved. 2024"):
        highlight = Highlight(text="x", source_url=page.url, quote=quote)
        assert not is_grounded(highlight, {page.url: page}), quote


def test_distinct_drop_reasons():
    """Round-4 review: floor-rejected, negation-rejected and
    not-found-verbatim must each get their own accurate reason string, not
    all be lumped under "not found verbatim in fetched page"."""
    from comparison.grounding import ground_highlights

    page = _page("https://a.example/", "It does not support wireless charging at all.")
    floor_rejected = Highlight(text="x", source_url=page.url, quote="on sale")
    negation_rejected = Highlight(
        text="x", source_url=page.url, quote="support wireless charging at all"
    )
    not_verbatim = Highlight(text="x", source_url=page.url, quote="made up claim entirely")

    _, dropped = ground_highlights([floor_rejected, negation_rejected, not_verbatim], {page.url: page})
    reasons = {d["quote"]: d["reason"] for d in dropped}

    assert "floor" in reasons["on sale"]
    assert "negat" in reasons["support wireless charging at all"]
    assert "verbatim" in reasons["made up claim entirely"]
    # All three reasons must be distinct from each other.
    assert len(set(reasons.values())) == 3
