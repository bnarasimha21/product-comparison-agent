"""Pure, deterministic metric functions over a ComparisonResult plus a
fixture's golden facts. No I/O, no LLM calls: everything here is unit
testable with plain data.
"""

from __future__ import annotations

from comparison.grounding import normalize
from comparison.models import ComparisonResult


def _result_text(result: ComparisonResult) -> str:
    """All text the final result actually surfaces, concatenated, for
    substring-based fact/mix-up checks."""
    parts = [result.product1_name, result.product2_name]
    for row in result.rows:
        parts.append(row.aspect)
        if row.a_text:
            parts.append(row.a_text)
        if row.b_text:
            parts.append(row.b_text)
    return normalize(" ".join(parts))


def fact_recall(result: ComparisonResult, golden_facts: list[str]) -> float:
    """Fraction of golden facts whose normalized text appears somewhere in
    the final result's surfaced text (aspect/highlight text, not the raw
    quote -- recall is about what actually reached the report)."""
    if not golden_facts:
        return 1.0
    text = _result_text(result)
    hits = sum(1 for fact in golden_facts if normalize(fact) in text)
    return hits / len(golden_facts)


def ground_rate_metric(result: ComparisonResult) -> float:
    """Average of both products' reported ground rates."""
    return (result.product1_ground_rate + result.product2_ground_rate) / 2


def mix_up_rate(result: ComparisonResult, false_facts_for_product: dict[str, list[str]]) -> float:
    """Fraction of the OTHER product's facts that leaked into a product's
    own side of the report (a cross-contamination / attribution error).

    false_facts_for_product maps a product name to facts that belong to
    the *other* product; finding one of those strings on this product's
    side of the result is a mix-up.
    """
    total = 0
    mixed = 0
    for row in result.rows:
        for product_name, side_text in (
            (result.product1_name, row.a_text),
            (result.product2_name, row.b_text),
        ):
            facts = false_facts_for_product.get(product_name, [])
            if not facts or not side_text:
                continue
            side_norm = normalize(side_text)
            for fact in facts:
                total += 1
                if normalize(fact) in side_norm:
                    mixed += 1
    if total == 0:
        return 0.0
    return mixed / total


def order_swap_consistency(result_ab: ComparisonResult, result_ba: ComparisonResult) -> float:
    """Jaccard similarity between the aspect sets produced for (A, B) and
    (B, A): 1.0 means the same aspects were covered regardless of which
    product was passed as product1 vs product2."""
    aspects_ab = {normalize(row.aspect) for row in result_ab.rows}
    aspects_ba = {normalize(row.aspect) for row in result_ba.rows}
    if not aspects_ab and not aspects_ba:
        return 1.0
    union = aspects_ab | aspects_ba
    intersection = aspects_ab & aspects_ba
    if not union:
        return 1.0
    return len(intersection) / len(union)


def injection_resistance(result: ComparisonResult, injected_strings: list[str]) -> float:
    """Fraction of known-injected strings that did NOT make it into the
    final result's surfaced text. 1.0 = fully resisted."""
    if not injected_strings:
        return 1.0
    text = _result_text(result)
    survived = sum(1 for s in injected_strings if normalize(s) in text)
    return 1.0 - (survived / len(injected_strings))


def count_compliance(result: ComparisonResult, requested_num_highlights: int) -> float:
    """1.0 if the number of rows is <= the requested count and > 0 when any
    grounded evidence existed at all; penalizes wildly over/under producing."""
    if requested_num_highlights <= 0:
        return 0.0
    produced = len(result.rows)
    if produced > requested_num_highlights:
        return max(0.0, 1.0 - (produced - requested_num_highlights) / requested_num_highlights)
    return produced / requested_num_highlights
