"""Full Flow, driven with FakeLLM + FixtureSearch + a local HTTP server.

Fully offline: no network call ever leaves localhost, no API key is used.
"""

import json

import pytest

from comparison.fake_llm import FakeLLM
from comparison.flow import ProductComparisonFlow
from comparison.models import CompareRequest, SearchResult
from comparison.search import FixtureSearch

pytestmark = pytest.mark.integration


def _reply(payload: dict) -> str:
    return json.dumps(payload)


def _build_search(product_a: str, product_b: str, url_a: str, url_b: str) -> FixtureSearch:
    from comparison.search import query_templates

    table: dict[str, list[SearchResult]] = {}
    for query in query_templates(product_a):
        table[query] = [SearchResult(title=product_a, url=url_a, snippet="about " + product_a)]
    for query in query_templates(product_b):
        table[query] = [SearchResult(title=product_b, url=url_b, snippet="about " + product_b)]
    return FixtureSearch(table)


@pytest.mark.asyncio
async def test_flow_produces_exact_num_highlights_rows(local_server):
    url_a = local_server.add_page(
        "/a.html", "<html><body>WidgetPro costs $19 per month and ships worldwide.</body></html>"
    )
    url_b = local_server.add_page(
        "/b.html", "<html><body>GadgetMax costs $25 per month and has a mobile app.</body></html>"
    )

    def fetch(url, **kwargs):
        from comparison.fetch import fetch_page

        return fetch_page(url, allow_http=True, allow_private=True)

    llm = FakeLLM(
        scripts={
            "URL Selector for WidgetPro": [_reply({"urls": [url_a]})],
            "URL Selector for GadgetMax": [_reply({"urls": [url_b]})],
            "Highlight Extractor for WidgetPro": [
                _reply(
                    {
                        "highlights": [
                            {"text": "price", "quote": "WidgetPro costs $19 per month"},
                            {"text": "shipping", "quote": "ships worldwide"},
                        ]
                    }
                )
            ],
            "Highlight Extractor for GadgetMax": [
                _reply(
                    {
                        "highlights": [
                            {"text": "price", "quote": "GadgetMax costs $25 per month"},
                            {"text": "app", "quote": "has a mobile app"},
                        ]
                    }
                )
            ],
            "Comparison Agent": [
                _reply(
                    {
                        "rows": [
                            {"aspect": "Price", "a_index": 1, "b_index": 1},
                            {"aspect": "Extras", "a_index": 2, "b_index": 2},
                        ]
                    }
                )
            ],
        }
    )

    flow = ProductComparisonFlow(llm=llm, search_provider=_build_search("WidgetPro", "GadgetMax", url_a, url_b), fetch_fn=fetch)
    flow.state.request = CompareRequest(
        product1="WidgetPro", product2="GadgetMax", num_urls=1, num_highlights=2
    )
    await flow.kickoff_async()

    result = flow.state.result
    assert result is not None
    assert len(result.rows) == 2
    assert result.rows[0].a_text == "WidgetPro costs $19 per month"
    assert result.rows[0].b_text == "GadgetMax costs $25 per month"


@pytest.mark.asyncio
async def test_comparison_prompt_contains_both_products_highlights(local_server):
    """Asserts the comparison task was actually given both products'
    extracted highlights, by inspecting the recorded LLM call messages."""
    url_a = local_server.add_page("/a.html", "<html><body>WidgetPro is fast.</body></html>")
    url_b = local_server.add_page("/b.html", "<html><body>GadgetMax is light.</body></html>")

    def fetch(url, **kwargs):
        from comparison.fetch import fetch_page

        return fetch_page(url, allow_http=True, allow_private=True)

    llm = FakeLLM(
        scripts={
            "URL Selector for WidgetPro": [_reply({"urls": [url_a]})],
            "URL Selector for GadgetMax": [_reply({"urls": [url_b]})],
            "Highlight Extractor for WidgetPro": [
                _reply({"highlights": [{"text": "speed", "quote": "WidgetPro is fast"}]})
            ],
            "Highlight Extractor for GadgetMax": [
                _reply({"highlights": [{"text": "weight", "quote": "GadgetMax is light"}]})
            ],
            "Comparison Agent": [_reply({"rows": [{"aspect": "Feel", "a_index": 1, "b_index": 1}]})],
        }
    )

    flow = ProductComparisonFlow(
        llm=llm, search_provider=_build_search("WidgetPro", "GadgetMax", url_a, url_b), fetch_fn=fetch
    )
    flow.state.request = CompareRequest(
        product1="WidgetPro", product2="GadgetMax", num_urls=1, num_highlights=1
    )
    await flow.kickoff_async()

    compare_calls = [c for c in llm.calls if c["role"] == "Comparison Agent"]
    assert len(compare_calls) == 1
    prompt_text = str(compare_calls[0]["messages"])
    assert "speed" in prompt_text
    assert "weight" in prompt_text


@pytest.mark.asyncio
async def test_product_name_with_braces_does_not_crash_agent_interpolation(local_server):
    """A product name containing `{word}` must not crash crewai's Agent
    role/goal template interpolation with an unhandled KeyError. crewai
    scans an agent's already-built role/goal string for any `{identifier}`
    pattern at kickoff and raises KeyError if `identifier` is not one of
    the kickoff inputs -- a product name like "Widget{data}Pro" baked into
    the role via an f-string becomes exactly such a bogus placeholder.
    comparison/agents.py neutralizes braces (to parentheses) before they
    reach role/goal, so the sanitized role is what FakeLLM's scripts key
    off below."""
    product1 = "Widget{data}Pro"
    product2 = "Gadget}Max{"
    url_a = local_server.add_page("/a.html", "<html><body>WidgetPro is fast.</body></html>")
    url_b = local_server.add_page("/b.html", "<html><body>GadgetMax is light.</body></html>")

    def fetch(url, **kwargs):
        from comparison.fetch import fetch_page

        return fetch_page(url, allow_http=True, allow_private=True)

    llm = FakeLLM(
        scripts={
            "URL Selector for Widget(data)Pro": [_reply({"urls": [url_a]})],
            "URL Selector for Gadget)Max(": [_reply({"urls": [url_b]})],
            "Highlight Extractor for Widget(data)Pro": [
                _reply({"highlights": [{"text": "speed", "quote": "WidgetPro is fast"}]})
            ],
            "Highlight Extractor for Gadget)Max(": [
                _reply({"highlights": [{"text": "weight", "quote": "GadgetMax is light"}]})
            ],
            "Comparison Agent": [_reply({"rows": [{"aspect": "Feel", "a_index": 1, "b_index": 1}]})],
        }
    )

    flow = ProductComparisonFlow(
        llm=llm, search_provider=_build_search(product1, product2, url_a, url_b), fetch_fn=fetch
    )
    flow.state.request = CompareRequest(product1=product1, product2=product2, num_urls=1, num_highlights=1)
    await flow.kickoff_async()

    result = flow.state.result
    assert result is not None
    assert result.product1_error is None
    assert result.product2_error is None
    assert len(result.rows) == 1


@pytest.mark.asyncio
async def test_one_product_crashing_does_not_abort_the_other(local_server):
    """asyncio.gather without return_exceptions=True lets ANY exception in
    one product's pipeline cancel/abort the sibling task too, defeating
    issue 6's graceful-degradation goal. Simulates a non-timeout crash
    (e.g. a library bug) in product1's search stage and asserts product2's
    research still completes normally, with product1 degraded to an
    errored report instead of blowing up the whole flow."""

    class _CrashingThenWorkingSearch:
        def __init__(self, crash_queries: set[str], working: FixtureSearch) -> None:
            self._crash_queries = crash_queries
            self._working = working

        def search(self, query, n):
            if query in self._crash_queries:
                raise RuntimeError("simulated search provider bug")
            return self._working.search(query, n)

    url_b = local_server.add_page("/b.html", "<html><body>GadgetMax is light.</body></html>")

    working = _build_search("WidgetPro", "GadgetMax", "https://unused.example/a.html", url_b)
    from comparison.search import query_templates

    crash_queries = set(query_templates("WidgetPro"))
    search_provider = _CrashingThenWorkingSearch(crash_queries, working)

    def fetch(url, **kwargs):
        from comparison.fetch import fetch_page

        return fetch_page(url, allow_http=True, allow_private=True)

    llm = FakeLLM(
        scripts={
            "URL Selector for GadgetMax": [_reply({"urls": [url_b]})],
            "Highlight Extractor for GadgetMax": [
                _reply({"highlights": [{"text": "weight", "quote": "GadgetMax is light"}]})
            ],
            "Comparison Agent": [_reply({"rows": [{"aspect": "Feel", "a_index": None, "b_index": 1}]})],
        }
    )

    flow = ProductComparisonFlow(llm=llm, search_provider=search_provider, fetch_fn=fetch)
    flow.state.request = CompareRequest(
        product1="WidgetPro", product2="GadgetMax", num_urls=1, num_highlights=1
    )
    await flow.kickoff_async()

    result = flow.state.result
    assert result is not None
    assert result.product1_error is not None
    assert "simulated search provider bug" in result.product1_error
    assert result.product2_error is None
    assert len(result.rows) == 1
    assert result.rows[0].b_text == "GadgetMax is light"


@pytest.mark.asyncio
async def test_fake_llm_reply_shape_without_tools(local_server):
    """Regression test for finding 2 in comparison/REGRESSION_NOTES.md: no
    agent in this app has a tool attached, so FakeLLM only ever needs to
    return JSON text parseable by the task's output_pydantic model -- no
    ReAct Thought/Action/Observation loop is required. This test asserts
    the 'tools' argument the FakeLLM call recorded was always empty/falsy
    for every call this flow makes."""
    url_a = local_server.add_page("/a.html", "<html><body>WidgetPro is solid.</body></html>")
    url_b = local_server.add_page("/b.html", "<html><body>GadgetMax is sleek.</body></html>")

    def fetch(url, **kwargs):
        from comparison.fetch import fetch_page

        return fetch_page(url, allow_http=True, allow_private=True)

    llm = FakeLLM(
        scripts={
            "URL Selector for WidgetPro": [_reply({"urls": [url_a]})],
            "URL Selector for GadgetMax": [_reply({"urls": [url_b]})],
            "Highlight Extractor for WidgetPro": [
                _reply({"highlights": [{"text": "solid", "quote": "WidgetPro is solid"}]})
            ],
            "Highlight Extractor for GadgetMax": [
                _reply({"highlights": [{"text": "sleek", "quote": "GadgetMax is sleek"}]})
            ],
            "Comparison Agent": [_reply({"rows": [{"aspect": "Build", "a_index": 1, "b_index": 1}]})],
        }
    )

    flow = ProductComparisonFlow(
        llm=llm, search_provider=_build_search("WidgetPro", "GadgetMax", url_a, url_b), fetch_fn=fetch
    )
    flow.state.request = CompareRequest(
        product1="WidgetPro", product2="GadgetMax", num_urls=1, num_highlights=1
    )
    await flow.kickoff_async()

    assert len(llm.calls) >= 5
    for call in llm.calls:
        assert not call["tools"]


@pytest.mark.asyncio
async def test_invented_url_rejected_and_retry_succeeds(local_server):
    url_a = local_server.add_page("/real-a.html", "<html><body>WidgetPro details.</body></html>")

    search = FixtureSearch(
        {
            query: [SearchResult(title="WidgetPro", url=url_a, snippet="real result")]
            for query in __import__("comparison.search", fromlist=["query_templates"]).query_templates(
                "WidgetPro"
            )
        }
    )
    search_b = FixtureSearch({})  # no results at all for product B

    class CombinedSearch:
        def search(self, query, n):
            a_results = search.search(query, n)
            if a_results:
                return a_results
            return search_b.search(query, n)

    def fetch(url, **kwargs):
        from comparison.fetch import fetch_page

        return fetch_page(url, allow_http=True, allow_private=True)

    invented_url = "https://not-a-real-candidate.example/fake"
    llm = FakeLLM(
        scripts={
            "URL Selector for WidgetPro": [
                _reply({"urls": [invented_url]}),  # first attempt: invented URL, rejected
                _reply({"urls": [url_a]}),  # retry: valid URL
            ],
            "Highlight Extractor for WidgetPro": [
                _reply({"highlights": [{"text": "x", "quote": "WidgetPro details"}]})
            ],
            "Comparison Agent": [_reply({"rows": [{"aspect": "Overview", "a_index": 1, "b_index": None}]})],
        }
    )

    flow = ProductComparisonFlow(llm=llm, search_provider=CombinedSearch(), fetch_fn=fetch)
    flow.state.request = CompareRequest(
        product1="WidgetPro", product2="OtherGadget", num_urls=1, num_highlights=1
    )
    await flow.kickoff_async()

    product1_report = flow.state.product1_report
    assert product1_report is not None
    assert invented_url in product1_report.rejected_urls
    assert product1_report.selected_urls == [url_a]

    product2_report = flow.state.product2_report
    assert product2_report is not None
    assert product2_report.error == "no search results found"
