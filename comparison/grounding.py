"""Deterministic grounding check for highlights.

The pipeline never trusts an LLM's claim that a highlight came from a page;
is_grounded re-checks it in code. A highlight is grounded only if:

1. its quote is substantive enough to be a meaningful claim, not filler or
   a bare connector phrase (see ``_quote_long_enough`` below),
2. its quote is a (whitespace/case/curly-quote-normalized) substring of the
   text fetched from its own source_url -- never a concatenation of
   multiple pages, so a quote that spans two pages is correctly rejected,
   and
3. the CLAUSE of source text immediately before the quote, and the
   remainder of the SENTENCE immediately after the quote, do not negate it
   (see ``_negated`` below) -- a quote that is a real substring of the
   page can still assert the OPPOSITE of what the page says if the page's
   negation ("does not support...", "...which is not true") sits just
   outside the quoted span, possibly several commas away rather than
   within some fixed character count.

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
# Curly single quotes/apostrophes normalized to a straight apostrophe before
# any matching happens, so "doesn't" (curly U+2019) matches the same as
# "doesn't" (straight) in both the substring match and the negation regex.
_CURLY_QUOTE_RE = re.compile("[‘’]")

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
        # boilerplate words that pair with a stray digit (page numbers,
        # copyright years, step numbers) but carry no claim of their own.
        # Added per round-4 review: "page 1 of 2" and "all rights
        # reserved. 2024" must still fail the floor even though each has a
        # digit next to a word, because neither adjacent word is
        # substantive. "weighs 171 grams" / "6.1-inch display" must still
        # pass since "grams"/"inch" are not in this set.
        "page", "step",
    }
)

# A quote that is a real substring of the source can still assert the
# OPPOSITE of what the source says, if a negation sits just outside the
# quoted span -- either in the clause right before it ("does not
# support...") or in a later clause of the same sentence that negates it
# after the fact ("..., which is not true"). See ``_negated`` below for how
# the window searched is bounded (by real clause/sentence edges, not a
# fixed character count).
#
# "cannot" is listed explicitly because `\bnot\b` does not match inside it
# (there is no word boundary before "not" in "cannot"). "neither"/"nor"/
# "lacks"/"hardly"/"rarely" and the two-word "fails to" / "fail to" were
# added per round-4 review as additional real-world negation phrasing.
_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|without|cannot|neither|nor|lacks|hardly|rarely)\b"
    r"|n't"
    r"|\bfails?\s+to\b"
)

# Hard sentence-ending punctuation. A period only counts as one of these
# when it is a genuine sentence end, not a period embedded in an
# abbreviation or decimal number (see ``_is_real_sentence_boundary``).
_HARD_ENDERS = ".!?;"
# Softer clause separators. These don't end a sentence, but they do bound
# how far back/forward the negation check looks -- "No hidden fees, it
# costs $19 per month" must not have "No hidden fees" dragged into the
# window just because it's within some fixed number of characters.
_SOFT_ENDERS = ",:"


def normalize(text: str) -> str:
    text = _CURLY_QUOTE_RE.sub("'", text)
    return _WHITESPACE_RE.sub(" ", text).strip().casefold()


def _is_bare_number_sequence(tokens: list[str]) -> bool:
    """True for a quote that is nothing but a sequence of numbers (e.g. a
    stray "1 2 3 4 5 6 7 8 9 10" list), which has digits but no actual
    claim content."""
    return bool(tokens) and all(token.isdigit() for token in tokens)


def _is_substantive_word(token: str) -> bool:
    return (
        not token.isdigit()
        and token not in _STOPWORDS
        and len(token) >= MIN_SUBSTANTIVE_WORD_LEN
    )


def _digit_adjacent_word_is_substantive(tokens: list[str]) -> bool:
    """True if some digit token in `tokens` has at least one immediately
    adjacent word token that is itself substantive (not a stopword/filler).

    A bare digit next to a stopword -- "page 1 of 2", "all rights
    reserved. 2024" -- must NOT pass the floor just because a digit is
    present; the digit has to be paired with an actual unit/spec word like
    "grams", "inch", "per" is too weak alone but "171 grams"/"6.1-inch"
    qualify because "grams"/"inch" survive the stopword filter.
    """
    for i, token in enumerate(tokens):
        if not token.isdigit():
            continue
        if i > 0 and _is_substantive_word(tokens[i - 1]):
            return True
        if i + 1 < len(tokens) and _is_substantive_word(tokens[i + 1]):
            return True
    return False


def _quote_long_enough(quote: str) -> bool:
    normalized = normalize(quote)
    if not normalized:
        return False
    tokens = _WORD_RE.findall(normalized)
    if not tokens or _is_bare_number_sequence(tokens):
        return False

    has_digit_token = any(token.isdigit() for token in tokens)
    has_alpha_token = any(not token.isdigit() for token in tokens)
    if has_digit_token and has_alpha_token and _digit_adjacent_word_is_substantive(tokens):
        return True

    substantive = {token for token in tokens if _is_substantive_word(token)}
    return len(substantive) >= MIN_NONSTOPWORD_TOKENS


def _is_real_sentence_boundary(text: str, period_idx: int) -> bool:
    """True if the '.' at period_idx is a genuine sentence-ending period,
    not one embedded in an abbreviation or decimal number.

    Heuristic: a period surrounded by non-whitespace on BOTH sides ("U.S.",
    "v2.0", "2.4GHz") is not a real sentence end -- there's no space on
    either side of it. A period followed by whitespace (or at the very end
    of the text) is treated as a real sentence end. This deliberately
    favors NOT splitting on ambiguous abbreviation periods, which only
    widens the clause window we search for negation -- it can never cause
    a false rejection by itself, only let the search look further back/
    forward than a naive splitter would.
    """
    before = text[period_idx - 1] if period_idx > 0 else " "
    after = text[period_idx + 1] if period_idx + 1 < len(text) else " "
    if not before.isspace() and not after.isspace():
        return False
    return True


def _is_hard_ender_at(text: str, idx: int) -> bool:
    ch = text[idx]
    if ch not in _HARD_ENDERS:
        return False
    if ch == ".":
        return _is_real_sentence_boundary(text, idx)
    return True


def _clause_start(text: str, pos: int) -> int:
    """Start of the clause containing `pos`: the character after the
    nearest sentence-ending punctuation OR comma/colon looking backward,
    whichever is closer -- not a fixed character count."""
    i = pos - 1
    while i >= 0:
        ch = text[i]
        if _is_hard_ender_at(text, i) or ch in _SOFT_ENDERS:
            return i + 1
        i -= 1
    return 0


def _sentence_end(text: str, pos: int) -> int:
    """End of the full SENTENCE containing `pos`, bounded only by a real
    hard sentence-ending punctuation mark (not a comma) -- a comma-joined
    trailing clause ("..., which is not true") can still refer back to an
    earlier quoted claim in the same sentence, so commas don't cut this
    scan short the way they do for the backward clause check."""
    i = pos
    while i < len(text):
        if _is_hard_ender_at(text, i):
            return i
        i += 1
    return len(text)


def _negated(normalized_quote: str, normalized_page: str, start_idx: int) -> bool:
    """True if the source text negates the quote, checked in two
    directions:

    1. Backward: a negation word in the CLAUSE immediately preceding the
       quote (bounded by the nearer of a sentence end or a comma/colon) --
       "does not support...", "is not the cheapest...".
    2. Forward: a negation word anywhere in the rest of the SENTENCE after
       the quote (bounded by a real sentence end, not a comma, since
       comma-joined trailing clauses commonly refer back to the same
       claim) -- "it supports wireless charging, or so the ad claims,
       which is not true". This direction is deliberately broad: we don't
       attempt to detect whether the trailing clause introduces an
       unrelated subject, so it can reject a quote over an unrelated
       negation later in an unusually long, multi-claim sentence. That is
       an accepted tradeoff -- a false claim slipping through as
       "grounded" is worse than an occasional true quote being dropped.
    """
    end_idx = start_idx + len(normalized_quote)

    clause_start = _clause_start(normalized_page, start_idx)
    before = normalized_page[clause_start:start_idx]
    if _NEGATION_RE.search(before):
        return True

    sentence_end = _sentence_end(normalized_page, end_idx)
    after = normalized_page[end_idx:sentence_end]
    if _NEGATION_RE.search(after):
        return True

    return False


# Reason strings used by `_ungrounded_reason` / `ground_highlights`. Each
# rejection cause gets its own distinct string -- a round-4 review found
# that floor-rejected and negation-rejected highlights were both logged
# with the misleading "not found verbatim in fetched page" reason, even
# though they WERE found verbatim and were rejected for an entirely
# different cause.
REASON_EMPTY_QUOTE = "quote is empty"
REASON_NOT_SUBSTANTIVE = "quote is too short or not substantive (floor-rejected)"
REASON_SOURCE_NOT_FETCHED = "source_url was not fetched"
REASON_NOT_VERBATIM = "quote not found verbatim in fetched page"
REASON_NEGATED = "quote is negated by nearby source text (negation-rejected)"


def _ungrounded_reason(highlight: Highlight, pages: dict[str, Page]) -> str | None:
    """Return None if `highlight` is grounded, otherwise a specific,
    accurate reason it was rejected."""
    quote = normalize(highlight.quote)
    if not quote:
        return REASON_EMPTY_QUOTE
    if not _quote_long_enough(highlight.quote):
        return REASON_NOT_SUBSTANTIVE
    page = pages.get(highlight.source_url)
    if page is None or not page.ok:
        return REASON_SOURCE_NOT_FETCHED
    normalized_page = normalize(page.text)
    start_idx = normalized_page.find(quote)
    if start_idx == -1:
        return REASON_NOT_VERBATIM
    if _negated(quote, normalized_page, start_idx):
        return REASON_NEGATED
    return None


def is_grounded(highlight: Highlight, pages: dict[str, Page]) -> bool:
    """True iff highlight.quote is substantive evidence AND is a substring
    of the text at its own source_url (which must have been fetched
    successfully) AND that substring is not negated by the source, either
    by a negation in the clause just before it or a negation in the rest
    of the sentence just after it."""
    return _ungrounded_reason(highlight, pages) is None


def ground_highlights(
    highlights: list[Highlight], pages: dict[str, Page]
) -> tuple[list[Highlight], list[dict]]:
    """Split proposed highlights into (grounded, dropped-with-reason)."""
    grounded: list[Highlight] = []
    dropped: list[dict] = []
    for highlight in highlights:
        reason = _ungrounded_reason(highlight, pages)
        if reason is None:
            grounded.append(highlight)
        else:
            dropped.append(
                {
                    "text": highlight.text,
                    "source_url": highlight.source_url,
                    "quote": highlight.quote,
                    "reason": reason,
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
