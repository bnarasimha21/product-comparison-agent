"""Deterministic grounding check for highlights.

The pipeline never trusts an LLM's claim that a highlight came from a page;
is_grounded re-checks it in code. A highlight is grounded only if its quote
is a (whitespace/case-normalized) substring of the text fetched from its
own source_url -- never a concatenation of multiple pages, so a quote that
spans two pages is correctly rejected.
"""

from __future__ import annotations

import re

from .models import Highlight, Page

_WHITESPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip().casefold()


def is_grounded(highlight: Highlight, pages: dict[str, Page]) -> bool:
    """True iff highlight.quote is a substring of the text at its own
    source_url, and that URL was actually fetched successfully."""
    quote = normalize(highlight.quote)
    if not quote:
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


def ground_rate(proposed_count: int, grounded_count: int) -> float:
    if proposed_count <= 0:
        return 1.0
    return grounded_count / proposed_count
