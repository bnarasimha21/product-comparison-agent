"""Eval runner (issue 5): `python -m evals.run --llm fake` runs all fixtures
fully offline with zero network calls and zero API keys.

For each fixture, runs the pipeline twice: once as (product1, product2) and
once swapped as (product2, product1), to compute order-swap consistency.
Writes a markdown report to evals/results/<date>-<model>.md and prints a
summary table.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from comparison.models import CompareRequest
from comparison.pipeline import compare

from .llm_backends import Fixture, build_fetch_fn, build_llm, build_search_provider, load_all_fixtures
from .metrics import (
    count_compliance,
    fact_recall,
    ground_rate_metric,
    injection_resistance,
    mix_up_rate,
    order_swap_consistency,
)

RESULTS_DIR = Path(__file__).parent / "results"


def _avg_or_none(*values: float | None) -> float | None:
    """Average the non-None values; None if every value given is None.

    Used for ground_rate, which is itself None ("n/a") whenever a run had
    nothing proposed to ground -- that must not silently become a 0 or a 1
    when averaged across the two order-swap runs.
    """
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


def run_fixture(fixture: Fixture, llm_name: str) -> dict:
    search_provider = build_search_provider(fixture)
    fetch_fn = build_fetch_fn(fixture)

    request_ab = CompareRequest(
        product1=fixture.product1,
        product2=fixture.product2,
        num_urls=fixture.num_urls,
        num_highlights=fixture.num_highlights,
    )
    request_ba = CompareRequest(
        product1=fixture.product2,
        product2=fixture.product1,
        num_urls=fixture.num_urls,
        num_highlights=fixture.num_highlights,
    )

    llm_ab = build_llm(llm_name, fixture)
    result_ab = compare(request_ab, llm=llm_ab, search_provider=search_provider, fetch_fn=fetch_fn)

    llm_ba = build_llm(llm_name, fixture)
    result_ba = compare(request_ba, llm=llm_ba, search_provider=search_provider, fetch_fn=fetch_fn)

    all_golden = fixture.golden.get(fixture.product1, []) + fixture.golden.get(fixture.product2, [])
    recall_ab = fact_recall(result_ab, all_golden)
    recall_ba = fact_recall(result_ba, all_golden)

    metrics = {
        "fixture": fixture.id,
        "fact_recall": (recall_ab + recall_ba) / 2,
        "ground_rate": _avg_or_none(ground_rate_metric(result_ab), ground_rate_metric(result_ba)),
        "mix_up_rate": (
            mix_up_rate(result_ab, fixture.false_facts) + mix_up_rate(result_ba, fixture.false_facts)
        )
        / 2,
        "order_swap_consistency": order_swap_consistency(result_ab, result_ba),
        "injection_resistance": (
            injection_resistance(result_ab, fixture.all_injected_strings())
            + injection_resistance(result_ba, fixture.all_injected_strings())
        )
        / 2,
        "count_compliance": (
            count_compliance(result_ab, fixture.num_highlights)
            + count_compliance(result_ba, fixture.num_highlights)
        )
        / 2,
    }
    return metrics


def run_all(llm_name: str, fixtures: list[Fixture] | None = None) -> list[dict]:
    fixtures = fixtures if fixtures is not None else load_all_fixtures()
    return [run_fixture(fixture, llm_name) for fixture in fixtures]


def _write_report(llm_name: str, rows: list[dict]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    date = dt.date.today().isoformat()
    path = RESULTS_DIR / f"{date}-{llm_name}.md"

    def _fmt(value: float | None) -> str:
        return f"{value:.2f}" if value is not None else "n/a"

    lines = [f"# Eval results: {llm_name} ({date})", ""]
    lines.append("| fixture | fact_recall | ground_rate | mix_up_rate | order_swap | injection_resist | count_compliance |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| {row['fixture']} | {row['fact_recall']:.2f} | {_fmt(row['ground_rate'])} | "
            f"{row['mix_up_rate']:.2f} | {row['order_swap_consistency']:.2f} | "
            f"{row['injection_resistance']:.2f} | {row['count_compliance']:.2f} |"
        )

    def _avg(key: str) -> float | None:
        if not rows:
            return None
        if key == "ground_rate":
            return _avg_or_none(*(r[key] for r in rows))
        return sum(r[key] for r in rows) / len(rows)

    lines.append("")
    lines.append(
        f"Averages: fact_recall={_avg('fact_recall'):.2f}, ground_rate={_fmt(_avg('ground_rate'))}, "
        f"mix_up_rate={_avg('mix_up_rate'):.2f}, order_swap_consistency={_avg('order_swap_consistency'):.2f}, "
        f"injection_resistance={_avg('injection_resistance'):.2f}, count_compliance={_avg('count_compliance'):.2f}"
    )
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the comparison pipeline eval suite.")
    parser.add_argument("--llm", choices=["fake", "do_serverless", "claude_cli"], default="fake")
    args = parser.parse_args(argv)

    rows = run_all(args.llm)
    report_path = _write_report(args.llm, rows)
    print(f"Wrote eval report to {report_path}")
    for row in rows:
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
