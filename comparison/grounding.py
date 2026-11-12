"""Deterministic grounding check for highlights.

The pipeline never trusts an LLM's claim that a highlight came from a page;
is_grounded re-checks it in code. A highlight is grounded only if:

1. its quote is long enough to be a meaningful claim, not a single common
   word (see MIN_QUOTE_WORDS / MIN_QUOTE_CHARS below), and
2. its quote is a (whitespace/case-normalized) substring of the text fetched
   from its own source_url -- never a concatenation of multiple pages, so a
   quote that spans two pages is correctly rejected.

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

# A quote must be at least this many characters and contain at least this
# many distinct words to count as evidence. Without this, a single common
# word ("the", "is") is trivially a substring of almost any page, so
# `is_grounded` would accept a quote that proves nothing.
#
# The previous bar (2 words / 12 characters) was still trivial to clear
# with a bare connector phrase that carries no actual claim: "the phone
# is" (12 chars, 3 words) and "is one of the" (13 chars, 4 words) both
# passed it and are substrings of almost any product page. Raising the
# character floor to 20 (and requiring 3 distinct words, not 2) rejects
# both of those outright while still accepting genuine short factual
# quotes, such as "WidgetPro costs $19 per month" or "weighs 1.24
# kilograms".
MIN_QUOTE_WORDS = 3
MIN_QUOTE_CHARS = 20


def normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip().casefold()


def _quote_long_enough(quote: str) -> bool:
    normalized = normalize(quote)
    if len(normalized) < MIN_QUOTE_CHARS:
        return False
    return len(set(_WORD_RE.findall(normalized))) >= MIN_QUOTE_WORDS


def is_grounded(highlight: Highlight, pages: dict[str, Page]) -> bool:
    """True iff highlight.quote is long enough to be meaningful evidence AND
    is a substring of the text at its own source_url, which must have been
    fetched successfully."""
    quote = normalize(highlight.quote)
    if not quote:
        return False
    if not _quote_long_enough(highlight.quote):
        return False
    page = pages.get(highlight.source_url)
    if page is None or not page.ok:
        return False
    return quote in normalize(page.text)


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
