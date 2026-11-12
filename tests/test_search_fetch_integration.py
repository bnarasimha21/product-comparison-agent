"""Search + fetch integration tests: a local HTTP server, a FixtureSearch,
a scripted FakeLLM driving URL selection, and the confirmed-broken-constructor
regression test. Fully offline.
"""

import json

import pytest

from comparison.fake_llm import FakeLLM
from comparison.fetch import fetch_page
from comparison.flow import ProductComparisonFlow
from comparison.models import CompareRequest, SearchResult
from comparison.search import FixtureSearch, query_templates

pytestmark = pytest.mark.integration


def test_old_constructor_calls_are_silently_ignored():
    """Regression test for finding 1 in comparison/REGRESSION_NOTES.md.

    The original code called SerperDevTool(query=...) and
    ScrapeWebsiteTool(urls=...), passing kwargs neither tool accepts. On the
    pinned crewai-tools version this does not raise; it silently drops the
    value and the tool falls back to its real defaults. This test pins that
    exact confirmed behaviour so a future crewai-tools upgrade that changes
    it (e.g. starts raising, or starts honoring the kwarg) is caught here
    rather than discovered as a live bug.
    """
    crewai_tools = pytest.importorskip("crewai_tools")

    scrape_tool = crewai_tools.ScrapeWebsiteTool(urls="https://example.com/ignored")
    dumped = scrape_tool.model_dump()
    assert "urls" not in dumped
    assert dumped.get("website_url") is None

    serper_tool = crewai_tools.SerperDevTool(query="ignored product name")
    serper_dumped = serper_tool.model_dump()
    assert "query" not in serper_dumped
    assert serper_dumped.get("n_results") == 10


def test_fixture_search_returns_configured_results_per_query():
    results = {"widget": [SearchResult(title="t", url="https://a.example/", snippet="s")]}
    provider = FixtureSearch(results)
    assert provider.search("widget", 5) == results["widget"]
    assert provider.search("unknown query", 5) == []


def test_query_templates_deterministic_and_product_specific():
    templates_a = query_templates("WidgetPro")
    templates_b = query_templates("GadgetMax")
    assert templates_a != templates_b
    assert all("WidgetPro" in q for q in templates_a)
    assert query_templates("WidgetPro") == templates_a  # deterministic


def test_fetch_against_local_server_matches_real_response(local_server):
    url = local_server.add_page("/p.html", "<html><body><p>Battery lasts 10 hours.</p></body></html>")
    page = fetch_page(url, allow_http=True, allow_private=True)
    assert page.ok
    assert "Battery lasts 10 hours" in page.text


@pytest.mark.asyncio
async def test_flow_selects_fetched_set_and_rejects_invalid_selection(local_server):
    url_real = local_server.add_page("/real.html", "<html><body>WidgetPro is reliable.</body></html>")
    url_unselected = local_server.add_page("/unselected.html", "<html><body>Unused page.</body></html>")

    table = {
        q: [
            SearchResult(title="real", url=url_real, snippet="real"),
            SearchResult(title="unselected", url=url_unselected, snippet="unselected"),
        ]
        for q in query_templates("WidgetPro")
    }
    search = FixtureSearch(table)

    def fetch(url, **kwargs):
        return fetch_page(url, allow_http=True, allow_private=True)

    invented = "https://never-a-candidate.example/made-up"
    llm = FakeLLM(
        scripts={
            "URL Selector for WidgetPro": [json.dumps({"urls": [url_real, invented]})],
            "Highlight Extractor for WidgetPro": [
                json.dumps({"highlights": [{"text": "reliability", "quote": "WidgetPro is reliable"}]})
            ],
            "URL Selector for OtherGadget": [json.dumps({"urls": []})],
            "Comparison Agent": [json.dumps({"rows": [{"aspect": "Overall", "a_index": 1, "b_index": None}]})],
        }
    )

    flow = ProductComparisonFlow(llm=llm, search_provider=search, fetch_fn=fetch)
    flow.state.request = CompareRequest(
        product1="WidgetPro", product2="OtherGadget", num_urls=2, num_highlights=1
    )
    await flow.kickoff_async()

    report1 = flow.state.product1_report
    assert report1 is not None
    # the invented URL was rejected, the real one was selected and fetched
    assert invented in report1.rejected_urls
    assert report1.selected_urls == [url_real]
    assert url_unselected not in report1.pages
    assert url_real in report1.pages
