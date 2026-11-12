"""Trace + cache integration: full pipeline twice over a recorded-page
server with a call counter, confirming the second run makes zero
network/LLM calls; the usage_metrics regression test for finding 3 in
comparison/REGRESSION_NOTES.md; a trace JSON round-trip; and a partial
result when one page fails. Fully offline.
"""

import json

import pytest

from comparison.cache import DiskCache
from comparison.fake_llm import FakeLLM
from comparison.fetch import fetch_page
from comparison.flow import ProductComparisonFlow
from comparison.models import CompareRequest, SearchResult
from comparison.pipeline import compare
from comparison.search import FixtureSearch, query_templates
from comparison.trace import Trace

pytestmark = pytest.mark.integration


def _counting_search(url_a: str, url_b: str, counter: dict):
    table = {
        **{q: [SearchResult(title="a", url=url_a, snippet="")] for q in query_templates("WidgetPro")},
        **{q: [SearchResult(title="b", url=url_b, snippet="")] for q in query_templates("GadgetMax")},
    }
    inner = FixtureSearch(table)

    class CountingSearch:
        def search(self, query, n):
            counter["search_calls"] += 1
            return inner.search(query, n)

    return CountingSearch()


def _counting_fetch(counter: dict):
    def fetch(url, **kwargs):
        counter["fetch_calls"] += 1
        return fetch_page(url, allow_http=True, allow_private=True)

    return fetch


@pytest.mark.asyncio
async def test_usage_metrics_reflect_fake_llm_calls(local_server):
    """Regression test for finding 3 in comparison/REGRESSION_NOTES.md:
    flow.usage_metrics only reflects a custom BaseLLM's calls because
    FakeLLM explicitly emits LLMCallStartedEvent/LLMCallCompletedEvent, not
    merely because it calls _track_token_usage_internal."""
    url_a = local_server.add_page("/a.html", "<html><body>WidgetPro costs $19 per month.</body></html>")
    url_b = local_server.add_page("/b.html", "<html><body>GadgetMax costs $25 per month.</body></html>")

    search = FixtureSearch(
        {
            **{q: [SearchResult(title="a", url=url_a, snippet="")] for q in query_templates("WidgetPro")},
            **{q: [SearchResult(title="b", url=url_b, snippet="")] for q in query_templates("GadgetMax")},
        }
    )

    def fetch(url, **kwargs):
        return fetch_page(url, allow_http=True, allow_private=True)

    llm = FakeLLM(
        scripts={
            "URL Selector for WidgetPro": [json.dumps({"urls": [url_a]})],
            "URL Selector for GadgetMax": [json.dumps({"urls": [url_b]})],
            "Highlight Extractor for WidgetPro": [
                json.dumps({"highlights": [{"text": "price", "quote": "WidgetPro costs $19 per month"}]})
            ],
            "Highlight Extractor for GadgetMax": [
                json.dumps({"highlights": [{"text": "price", "quote": "GadgetMax costs $25 per month"}]})
            ],
            "Comparison Agent": [json.dumps({"rows": [{"aspect": "Price", "a_index": 1, "b_index": 1}]})],
        }
    )

    flow = ProductComparisonFlow(llm=llm, search_provider=search, fetch_fn=fetch)
    flow.state.request = CompareRequest(
        product1="WidgetPro", product2="GadgetMax", num_urls=1, num_highlights=1
    )
    await flow.kickoff_async()

    metrics = flow.usage_metrics
    assert metrics.total_tokens > 0
    assert metrics.prompt_tokens > 0
    assert metrics.completion_tokens > 0
    assert metrics.successful_requests == len(llm.calls)


def test_second_run_over_same_request_hits_cache_with_zero_network_or_llm_calls(tmp_path, local_server):
    url_a = local_server.add_page("/a.html", "<html><body>WidgetPro costs $19 per month.</body></html>")
    url_b = local_server.add_page("/b.html", "<html><body>GadgetMax costs $25 per month.</body></html>")

    counter = {"search_calls": 0, "fetch_calls": 0}
    search_provider = _counting_search(url_a, url_b, counter)
    fetch_fn = _counting_fetch(counter)

    def make_llm() -> FakeLLM:
        return FakeLLM(
            scripts={
                "URL Selector for WidgetPro": [json.dumps({"urls": [url_a]})],
                "URL Selector for GadgetMax": [json.dumps({"urls": [url_b]})],
                "Highlight Extractor for WidgetPro": [
                    json.dumps(
                        {"highlights": [{"text": "price", "quote": "WidgetPro costs $19 per month"}]}
                    )
                ],
                "Highlight Extractor for GadgetMax": [
                    json.dumps(
                        {"highlights": [{"text": "price", "quote": "GadgetMax costs $25 per month"}]}
                    )
                ],
                "Comparison Agent": [
                    json.dumps({"rows": [{"aspect": "Price", "a_index": 1, "b_index": 1}]})
                ],
            }
        )

    cache = DiskCache(tmp_path / "cache")
    request = CompareRequest(product1="WidgetPro", product2="GadgetMax", num_urls=1, num_highlights=1)

    first_llm = make_llm()
    first_result = compare(request, llm=first_llm, search_provider=search_provider, fetch_fn=fetch_fn, cache=cache)
    assert counter["search_calls"] > 0
    assert counter["fetch_calls"] > 0
    assert len(first_result.rows) == 1

    counter["search_calls"] = 0
    counter["fetch_calls"] = 0

    class ExplodingLLM:
        # Same model identity as first_llm (FakeLLM defaults to
        # "fake/fake-1"): result_cache_key now includes model_id (issue 5a),
        # so a *different* model identity here would legitimately be a
        # fresh cache miss. This keeps the model identity constant, which
        # isolates what this test actually checks: an identical
        # request+model is served from cache without ever calling the LLM.
        model = "fake/fake-1"

        def call(self, *args, **kwargs):
            raise AssertionError("second run must not call the LLM: result should come from cache")

    second_result = compare(
        request, llm=ExplodingLLM(), search_provider=search_provider, fetch_fn=fetch_fn, cache=cache
    )

    assert counter["search_calls"] == 0
    assert counter["fetch_calls"] == 0
    assert second_result.rows == first_result.rows
    assert second_result.product1_name == first_result.product1_name


def test_cache_hit_miss_counters_in_trace_are_scoped_to_one_request(tmp_path, local_server):
    """DiskCache.hits/.misses are plain instance attributes that accumulate
    for as long as the DiskCache instance lives. A real server (main.py)
    builds exactly one DiskCache and reuses it for every request for the
    life of the process, so a naive `cache.hits`/`cache.misses` read inside
    compare() would report the running total since the server started, not
    this one request's own activity. Simulate that long-lived-instance
    scenario here: reuse a single DiskCache across two DIFFERENT requests
    and confirm the second request's own trace only reflects what the
    second request itself did, not an ever-growing cumulative count
    carried over from the first.
    """
    url_a = local_server.add_page("/a.html", "<html><body>WidgetPro costs $19 per month.</body></html>")
    url_b = local_server.add_page("/b.html", "<html><body>GadgetMax costs $25 per month.</body></html>")
    url_c = local_server.add_page("/c.html", "<html><body>ThirdThing costs $9 per month.</body></html>")

    def fetch(url, **kwargs):
        return fetch_page(url, allow_http=True, allow_private=True)

    def make_llm(product1_quote, product2_quote):
        return FakeLLM(
            scripts={
                "URL Selector for WidgetPro": [json.dumps({"urls": [url_a]})],
                "URL Selector for GadgetMax": [json.dumps({"urls": [url_b]})],
                "URL Selector for ThirdThing": [json.dumps({"urls": [url_c]})],
                "Highlight Extractor for WidgetPro": [
                    json.dumps({"highlights": [{"text": "price", "quote": product1_quote}]})
                ],
                "Highlight Extractor for GadgetMax": [
                    json.dumps({"highlights": [{"text": "price", "quote": product2_quote}]})
                ],
                "Highlight Extractor for ThirdThing": [
                    json.dumps({"highlights": [{"text": "price", "quote": "ThirdThing costs $9 per month"}]})
                ],
                "Comparison Agent": [
                    json.dumps({"rows": [{"aspect": "Price", "a_index": 1, "b_index": 1}]})
                ],
            }
        )

    cache = DiskCache(tmp_path / "cache")  # one long-lived instance, like main.py's module-level _cache

    search_ab = FixtureSearch(
        {
            **{q: [SearchResult(title="a", url=url_a, snippet="")] for q in query_templates("WidgetPro")},
            **{q: [SearchResult(title="b", url=url_b, snippet="")] for q in query_templates("GadgetMax")},
        }
    )
    request_ab = CompareRequest(product1="WidgetPro", product2="GadgetMax", num_urls=1, num_highlights=1)
    compare(
        request_ab,
        llm=make_llm("WidgetPro costs $19 per month", "GadgetMax costs $25 per month"),
        search_provider=search_ab,
        fetch_fn=fetch,
        cache=cache,
    )
    # cache now has nonzero hits/misses accumulated from the first request.
    assert cache.misses > 0

    search_cb = FixtureSearch(
        {
            **{q: [SearchResult(title="c", url=url_c, snippet="")] for q in query_templates("ThirdThing")},
            **{q: [SearchResult(title="b", url=url_b, snippet="")] for q in query_templates("GadgetMax")},
        }
    )
    request_cb = CompareRequest(product1="ThirdThing", product2="GadgetMax", num_urls=1, num_highlights=1)
    second_result = compare(
        request_cb,
        llm=make_llm("ThirdThing costs $9 per month", "GadgetMax costs $25 per month"),
        search_provider=search_cb,
        fetch_fn=fetch,
        cache=cache,
    )

    # The second request shares GadgetMax with the first (its search/fetch
    # entries are already cached) but is a brand-new product pairing, so
    # its own result was never cached: a fresh result-cache miss, plus at
    # least one fresh search/fetch miss for ThirdThing, is expected. What
    # must NOT happen is this request's reported counters including the
    # first request's own activity.
    reported_misses = second_result.trace["cache_misses"]
    reported_hits = second_result.trace["cache_hits"]
    assert reported_misses < cache.misses, (
        "trace.cache_misses must be this request's own count, not the "
        "DiskCache instance's running total since process start"
    )
    assert reported_hits <= cache.hits


def test_cache_hit_trace_reflects_this_request_not_the_original_run(tmp_path, local_server):
    """Regression for issue 4: on a cache hit, compare() used to return the
    ORIGINAL run's stored trace wholesale (its llm_calls, token counts,
    cache hit/miss counts), misrepresenting a request that made zero LLM
    calls at all. The cache-hit trace must report its OWN, fresh metrics:
    zero LLM calls, zero tokens, zero estimated cost, zero pages fetched/
    failed (nothing was fetched this request), and a cache_hits count that
    reflects this request, not the first one."""
    url_a = local_server.add_page("/a.html", "<html><body>WidgetPro costs $19 per month.</body></html>")
    url_b = local_server.add_page("/b.html", "<html><body>GadgetMax costs $25 per month.</body></html>")

    search = FixtureSearch(
        {
            **{q: [SearchResult(title="a", url=url_a, snippet="")] for q in query_templates("WidgetPro")},
            **{q: [SearchResult(title="b", url=url_b, snippet="")] for q in query_templates("GadgetMax")},
        }
    )

    def fetch(url, **kwargs):
        return fetch_page(url, allow_http=True, allow_private=True)

    def make_llm():
        return FakeLLM(
            scripts={
                "URL Selector for WidgetPro": [json.dumps({"urls": [url_a]})],
                "URL Selector for GadgetMax": [json.dumps({"urls": [url_b]})],
                "Highlight Extractor for WidgetPro": [
                    json.dumps(
                        {"highlights": [{"text": "price", "quote": "WidgetPro costs $19 per month"}]}
                    )
                ],
                "Highlight Extractor for GadgetMax": [
                    json.dumps(
                        {"highlights": [{"text": "price", "quote": "GadgetMax costs $25 per month"}]}
                    )
                ],
                "Comparison Agent": [
                    json.dumps({"rows": [{"aspect": "Price", "a_index": 1, "b_index": 1}]})
                ],
            }
        )

    cache = DiskCache(tmp_path / "cache")
    request = CompareRequest(product1="WidgetPro", product2="GadgetMax", num_urls=1, num_highlights=1)

    first_result = compare(request, llm=make_llm(), search_provider=search, fetch_fn=fetch, cache=cache)
    assert first_result.trace is not None
    assert first_result.trace["llm_calls"] > 0  # the first run genuinely called the LLM

    class ExplodingLLM:
        model = "fake/fake-1"

        def call(self, *args, **kwargs):
            raise AssertionError("second run must not call the LLM: result should come from cache")

    second_result = compare(
        request, llm=ExplodingLLM(), search_provider=search, fetch_fn=fetch, cache=cache
    )
    assert second_result.trace is not None
    assert second_result.trace["llm_calls"] == 0
    assert second_result.trace["prompt_tokens"] == 0
    assert second_result.trace["completion_tokens"] == 0
    assert second_result.trace["total_tokens"] == 0
    assert second_result.trace["estimated_cost_usd"] == 0.0
    assert second_result.trace["pages_fetched"] == 0
    assert second_result.trace["pages_failed"] == 0
    # The trace is not simply the first run's trace copied verbatim.
    assert second_result.trace != first_result.trace


def test_trace_json_round_trips(tmp_path, local_server):
    url_a = local_server.add_page("/a.html", "<html><body>WidgetPro costs $19 per month.</body></html>")
    url_b = local_server.add_page("/b.html", "<html><body>GadgetMax costs $25 per month.</body></html>")

    search = FixtureSearch(
        {
            **{q: [SearchResult(title="a", url=url_a, snippet="")] for q in query_templates("WidgetPro")},
            **{q: [SearchResult(title="b", url=url_b, snippet="")] for q in query_templates("GadgetMax")},
        }
    )

    def fetch(url, **kwargs):
        return fetch_page(url, allow_http=True, allow_private=True)

    llm = FakeLLM(
        scripts={
            "URL Selector for WidgetPro": [json.dumps({"urls": [url_a]})],
            "URL Selector for GadgetMax": [json.dumps({"urls": [url_b]})],
            "Highlight Extractor for WidgetPro": [
                json.dumps({"highlights": [{"text": "price", "quote": "WidgetPro costs $19 per month"}]})
            ],
            "Highlight Extractor for GadgetMax": [
                json.dumps({"highlights": [{"text": "price", "quote": "GadgetMax costs $25 per month"}]})
            ],
            "Comparison Agent": [json.dumps({"rows": [{"aspect": "Price", "a_index": 1, "b_index": 1}]})],
        }
    )

    request = CompareRequest(product1="WidgetPro", product2="GadgetMax", num_urls=1, num_highlights=1)
    result = compare(request, llm=llm, search_provider=search, fetch_fn=fetch)

    assert result.trace is not None
    dumped = json.dumps(result.trace)
    reloaded = Trace.model_validate(json.loads(dumped))
    assert reloaded.llm_calls == result.trace["llm_calls"]
    assert reloaded.total_tokens == result.trace["total_tokens"]
    assert reloaded.pages_fetched == 2
    assert reloaded.pages_failed == 0


@pytest.mark.asyncio
async def test_partial_result_when_one_product_page_fails(local_server):
    url_a = local_server.add_page("/a.html", "<html><body>WidgetPro costs $19 per month.</body></html>")
    failing_url = local_server.add_status("/missing.html", 500)

    search = FixtureSearch(
        {
            **{q: [SearchResult(title="a", url=url_a, snippet="")] for q in query_templates("WidgetPro")},
            **{q: [SearchResult(title="b", url=failing_url, snippet="")] for q in query_templates("GadgetMax")},
        }
    )

    def fetch(url, **kwargs):
        return fetch_page(url, allow_http=True, allow_private=True)

    llm = FakeLLM(
        scripts={
            "URL Selector for WidgetPro": [json.dumps({"urls": [url_a]})],
            "URL Selector for GadgetMax": [json.dumps({"urls": [failing_url]})],
            "Highlight Extractor for WidgetPro": [
                json.dumps({"highlights": [{"text": "price", "quote": "WidgetPro costs $19 per month"}]})
            ],
            "Comparison Agent": [
                json.dumps({"rows": [{"aspect": "Price", "a_index": 1, "b_index": None}]})
            ],
        }
    )

    flow = ProductComparisonFlow(llm=llm, search_provider=search, fetch_fn=fetch)
    flow.state.request = CompareRequest(
        product1="WidgetPro", product2="GadgetMax", num_urls=1, num_highlights=1
    )
    await flow.kickoff_async()

    result = flow.state.result
    assert result is not None
    assert result.product1_error is None
    assert result.product2_error == "no page could be fetched"
    assert len(result.rows) == 1
    assert result.rows[0].a_text == "WidgetPro costs $19 per month"
    assert result.rows[0].b_text == "no evidence found"
