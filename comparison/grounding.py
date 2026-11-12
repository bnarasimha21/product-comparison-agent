"""Deterministic grounding check for highlights.

The pipeline never trusts an LLM's claim that a highlight came from a page;
is_grounded re-checks it in code. A highlight is grounded only if:

1. its quote is substantive enough to be a meaningful claim, not filler or
   a bare connector phrase (see ``_quote_long_enough`` below),
2. its quote is a (whitespace/case-normalized) substring of the text fetched
   from its own source_url -- never a concatenation of multiple pages, so a
   quote that spans two pages is correctly rejected, and
3. the source text immediately before the quote does not negate it (see
   ``_negated_before`` below) -- a quote that is a real substring of the
   page can still assert the OPPOSITE of what the page says if the page's
   negation ("does not support...") sits just outside the quoted span.

Grounding only ever checks ``quote`` against the page, never ``quote``
against the model's separate, human-readable ``text`` field -- ``text`` is
never checked against anything. That means ``text`` must never be what the
final report renders as the claim: comparison/flow.py renders
``highlight.quote`` itself for a_text/b_text, so the thing shown to a human
is always exactly the thing that was verified, not a free-floating paraphrase
the model could have fabricated independently of its own cited quote.
"""

from __future__ import annotations

import re

from .models import Highlight, Page

_WHITESPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")

# A previous version of this check used a pure length/word-count floor (3
# words / 20 chars). That overshot: it started rejecting genuine short
# factual quotes like "weighs 171 grams" (17 chars) or "6.1-inch display"
# (16 chars) just as readily as it rejected filler. A length floor alone
# can never tell "weighs 171 grams" (substantive) apart from "on the
# market this year" (filler) of similar length.
#
# Instead: a quote counts as evidence if EITHER
#   (a) it contains at least one digit token alongside at least one
#       alphabetic token (a number paired with a unit/word is almost always
#       a real spec -- "171 grams", "$19 per month", "6.1-inch" -- while a
#       bare sequence of numbers with no words, like a stray "1 2 3 4 5 6 7
#       8 9 10" list, is not), or
#   (b) it contains at least MIN_NONSTOPWORD_TOKENS distinct words that are
#       not in the generic filler/stopword list below and are not
#       single-letter fragments.
#
# This still rejects the reviewer-cited connector phrases ("the phone is",
# "is one of the": each has at most one non-filler word) and the marketing
# filler examples ("terms and conditions apply", "click here to learn
# more", "all rights reserved.", "on the market this year": every word in
# each is either filler or, for "market", a single lone substantive word,
# which is below the 2-word bar) while accepting "weighs 171 grams",
# "6.1-inch display", "ships with usb-c", and "costs $19 per month".
MIN_NONSTOPWORD_TOKENS = 2
MIN_SUBSTANTIVE_WORD_LEN = 2  # single-letter fragments ("c" in "usb-c") don't count alone

_STOPWORDS = frozenset(
    {
        # articles / pronouns / determiners
        "a", "an", "the", "this", "that", "these", "those", "it", "its",
        "one", "all", "any", "each", "every", "both", "such",
        # linking / auxiliary verbs
        "is", "are", "was", "were", "be", "been", "being", "has", "have",
        "had", "do", "does", "did",
        # prepositions / connectives
        "on", "at", "in", "of", "to", "for", "with", "by", "as", "from",
        "or", "and", "but", "if", "than", "then", "so", "up", "out",
        "about", "into", "over", "under",
        # generic marketing / boilerplate filler called out by review. Kept
        # deliberately narrow (e.g. no "view"/"read"/"go") so a real
        # feature name is never accidentally treated as filler -- "view
        # filters" (a genuine SaaS feature quote) must still pass, while
        # "on the market this year" must still fail.
        "year", "years", "click", "here", "learn", "more", "rights",
        "reserved", "terms", "conditions", "apply", "market",
    }
)

# A quote that is a substring of the source but sits just past a negation
# in the source ("does not support...", "...is not the cheapest...") would
# otherwise pass grounding while asserting the opposite of the source. We
# look a short window back from the quote's start (within the same
# clause/sentence) for a negation word that is not itself part of the
# quote.
_NEGATION_RE = re.compile(r"\b(?:not|no|never|without)\b|n't")
_NEGATION_WINDOW_CHARS = 60


def normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip().casefold()


def _is_bare_number_sequence(tokens: list[str]) -> bool:
    """True for a quote that is nothing but a sequence of numbers (e.g. a
    stray "1 2 3 4 5 6 7 8 9 10" list), which has digits but no actual
    claim content."""
    return bool(tokens) and all(token.isdigit() for token in tokens)


def _quote_long_enough(quote: str) -> bool:
    normalized = normalize(quote)
    if not normalized:
        return False
    tokens = _WORD_RE.findall(normalized)
    if not tokens or _is_bare_number_sequence(tokens):
        return False

    has_digit_token = any(token.isdigit() for token in tokens)
    has_alpha_token = any(not token.isdigit() for token in tokens)
    if has_digit_token and has_alpha_token:
        return True

    substantive = {
        token
        for token in tokens
        if not token.isdigit()
        and token not in _STOPWORDS
        and len(token) >= MIN_SUBSTANTIVE_WORD_LEN
    }
    return len(substantive) >= MIN_NONSTOPWORD_TOKENS


def _negated_before(normalized_quote: str, normalized_page: str, start_idx: int) -> bool:
    """True if a negation word sits in the source text just before the
    quote's start position, within the same clause, and is not part of the
    quote itself."""
    window_start = max(0, start_idx - _NEGATION_WINDOW_CHARS)
    window = normalized_page[window_start:start_idx]
    # Don't look past the start of the current clause/sentence -- a
    # negation word in an unrelated preceding sentence shouldn't flag an
    # unrelated later quote.
    for sep in (".", "!", "?", ";"):
        sep_idx = window.rfind(sep)
        if sep_idx != -1:
            window = window[sep_idx + 1 :]
    return bool(_NEGATION_RE.search(window))


def is_grounded(highlight: Highlight, pages: dict[str, Page]) -> bool:
    """True iff highlight.quote is substantive evidence AND is a substring
    of the text at its own source_url (which must have been fetched
    successfully) AND that substring is not negated by text immediately
    preceding it in the source."""
    quote = normalize(highlight.quote)
    if not quote:
        return False
    if not _quote_long_enough(highlight.quote):
        return False
    page = pages.get(highlight.source_url)
    if page is None or not page.ok:
        return False
    normalized_page = normalize(page.text)
    start_idx = normalized_page.find(quote)
    if start_idx == -1:
        return False
    if _negated_before(quote, normalized_page, start_idx):
        return False
    return True


def ground_highlights(
    highlights: list[Highlight], pages: dict[str, Page]
) -> tuple[list[Highlight], list[dict]]:
    """Split proposed highlights into (grounded, dropped-with-reason)."""
    grounded: list[Highlight] = []
    dropped: list[dict] = []
    for highlight in highlights:
        if is_grounded(highlight, pages):
            grounded.append(highlight)
        else:
            dropped.append(
                {
                    "text": highlight.text,
                    "source_url": highlight.source_url,
                    "quote": highlight.quote,
                    "reason": "not found verbatim in fetched page"
                    if highlight.source_url in pages
                    else "source_url was not fetched",
                }
            )
    return grounded, dropped


def ground_rate(proposed_count: int, grounded_count: int) -> float | None:
    """The fraction of proposed highlights that survived grounding, or None
    when nothing was ever proposed.

    Returning 1.0 for "zero proposed" (the old behaviour) misleadingly
    displayed "100% grounded" for a product with no evidence at all --
    indistinguishable from a product where everything genuinely checked
    out. None is a distinct "n/a: nothing to ground" signal that callers
    (comparison/render.py, evals/metrics.py) must render/aggregate
    differently from an actual 0.0 ("everything proposed was rejected") or
    1.0 ("everything proposed was grounded").
    """
    if proposed_count <= 0:
        return None
    return grounded_count / proposed_count
