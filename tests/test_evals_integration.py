"""Integration test for the eval runner itself (issue 5): runs the real
fixtures in fake mode end to end, asserts every metric is in [0, 1], that
the order-swap run actually executed both orders, and that a results file
is written. Fully offline, no network, no API key.
"""

import pytest

from evals.llm_backends import load_all_fixtures
from evals.run import _write_report, run_all, run_fixture

pytestmark = pytest.mark.integration

_METRIC_KEYS = [
    "fact_recall",
    "ground_rate",
    "mix_up_rate",
    "order_swap_consistency",
    "injection_resistance",
    "count_compliance",
]


def test_all_fixtures_load_and_have_required_fields():
    fixtures = load_all_fixtures()
    assert len(fixtures) == 7
    ids = {f.id for f in fixtures}
    assert ids == {
        "phones",
        "laptops",
        "saas_tools",
        "dev_tools",
        "near_identical",
        "sparse_pair",
        "injection_attempt",
    }
    for fixture in fixtures:
        assert fixture.product1 != fixture.product2
        assert fixture.num_urls >= 1
        assert fixture.num_highlights >= 1


def test_run_all_fake_produces_metrics_in_unit_range():
    rows = run_all("fake")
    assert len(rows) == 7
    for row in rows:
        for key in _METRIC_KEYS:
            assert 0.0 <= row[key] <= 1.0, f"{row['fixture']}.{key} = {row[key]} out of [0,1]"


def test_injection_attempt_fixture_moves_injection_resistance_below_one():
    """Before this fix, run_fixture always called injection_resistance(...,
    []) for every fixture, so the metric was hardcoded to 1.0 no matter
    what the pipeline actually surfaced. injection_attempt.yaml plants a
    real injected string and scripts the fake extractor to fall for it
    (see llm_backends._highlight_extractor_reply), so this must now show
    up as a real, non-trivial resistance score below 1.0."""
    fixtures = load_all_fixtures()
    fixture = next(f for f in fixtures if f.id == "injection_attempt")
    assert fixture.all_injected_strings(), "fixture must actually carry injected content"
    metrics = run_fixture(fixture, "fake")
    assert metrics["injection_resistance"] < 1.0


def test_other_fixtures_have_no_injected_strings_and_stay_fully_resistant():
    """Fixtures that plant nothing legitimately score 1.0 -- that is no
    longer a vacuous "always 1.0 regardless of input" result, since the
    metric is now fed each fixture's own (possibly empty) injected_strings
    rather than a literal [] constant for every fixture."""
    fixtures = load_all_fixtures()
    for fixture in fixtures:
        if fixture.id == "injection_attempt":
            continue
        assert fixture.all_injected_strings() == []


def test_order_swap_run_actually_executes_both_orders(monkeypatch):
    """Patches comparison.pipeline.compare to record which (product1,
    product2) order it was called with, confirming run_fixture truly
    issues one call per order rather than reusing a single result."""
    import evals.run as run_module

    calls = []
    original_compare = run_module.compare

    def spy_compare(request, **kwargs):
        calls.append((request.product1, request.product2))
        return original_compare(request, **kwargs)

    monkeypatch.setattr(run_module, "compare", spy_compare)

    fixtures = load_all_fixtures()
    phones = next(f for f in fixtures if f.id == "phones")
    run_fixture(phones, "fake")

    assert len(calls) == 2
    assert calls[0] == (phones.product1, phones.product2)
    assert calls[1] == (phones.product2, phones.product1)


def test_fully_grounded_fixture_has_zero_mix_up_rate():
    """The fake backend only ever quotes verbatim from the page text it was
    given its own product, so it should never attribute the other
    product's facts to this product's side."""
    fixtures = load_all_fixtures()
    phones = next(f for f in fixtures if f.id == "phones")
    metrics = run_fixture(phones, "fake")
    assert metrics["mix_up_rate"] == 0.0


def test_sparse_pair_still_produces_a_result_without_crashing():
    fixtures = load_all_fixtures()
    sparse = next(f for f in fixtures if f.id == "sparse_pair")
    metrics = run_fixture(sparse, "fake")
    for key in _METRIC_KEYS:
        assert 0.0 <= metrics[key] <= 1.0


def test_results_file_is_written(tmp_path, monkeypatch):
    import evals.run as run_module

    monkeypatch.setattr(run_module, "RESULTS_DIR", tmp_path)
    rows = run_all("fake")
    path = _write_report("fake", rows)
    assert path.exists()
    assert path.parent == tmp_path
    content = path.read_text()
    assert "fact_recall" in content
    for row in rows:
        assert row["fixture"] in content
