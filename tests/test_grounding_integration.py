"""Full Flow over recorded pages with a scripted FakeLLM that includes one
fabricated (ungrounded) highlight per product; asserts the fabrication is
dropped and ground rates are reported correctly. Fully offline.
"""

import json

import pytest

from comparison.fake_llm import FakeLLM
from comparison.fetch import fetch_page
from comparison.flow import ProductComparisonFlow
from comparison.models import CompareRequest, SearchResult
from comparison.search import FixtureSearch, query_templates

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_fabricated_highlights_are_dropped_and_ground_rate_reported(local_server):
    url_a = local_server.add_page(
        "/a.html", "<html><body>WidgetPro costs $19 per month and weighs 200 grams.</body></html>"
    )
    url_b = local_server.add_page(
        "/b.html", "<html><body>GadgetMax costs $25 per month and weighs 180 grams.</body></html>"
    )

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
                json.dumps(
                    {
                        "highlights": [
                            {"text": "price", "quote": "WidgetPro costs $19 per month"},
                            {"text": "fabricated", "quote": "WidgetPro cures the common cold"},
                        ]
                    }
                )
            ],
            "Highlight Extractor for GadgetMax": [
                json.dumps(
                    {
                        "highlights": [
                            {"text": "weight", "quote": "weighs 180 grams"},
                            {"text": "fabricated", "quote": "GadgetMax can fly unaided"},
                        ]
                    }
                )
            ],
            "Comparison Agent": [
                json.dumps({"rows": [{"aspect": "Overview", "a_index": 1, "b_index": 1}]})
            ],
        }
    )

    flow = ProductComparisonFlow(llm=llm, search_provider=search, fetch_fn=fetch)
    flow.state.request = CompareRequest(
        product1="WidgetPro", product2="GadgetMax", num_urls=1, num_highlights=2
    )
    await flow.kickoff_async()

    report1 = flow.state.product1_report
    report2 = flow.state.product2_report
    assert report1 is not None and report2 is not None

    grounded_quotes_1 = [h.quote for h in report1.highlights]
    grounded_quotes_2 = [h.quote for h in report2.highlights]
    assert "WidgetPro cures the common cold" not in grounded_quotes_1
    assert "GadgetMax can fly unaided" not in grounded_quotes_2
    assert "WidgetPro costs $19 per month" in grounded_quotes_1
    assert "weighs 180 grams" in grounded_quotes_2

    # 1 of 2 proposed highlights survived grounding on each side.
    assert report1.ground_rate == 0.5
    assert report2.ground_rate == 0.5
    assert len(report1.dropped) == 1
    assert len(report2.dropped) == 1

    result = flow.state.result
    assert result.product1_ground_rate == 0.5
    assert result.product2_ground_rate == 0.5
    assert result.product1_dropped == 1
    assert result.product2_dropped == 1


@pytest.mark.asyncio
async def test_fully_grounded_run_reports_ground_rate_one(local_server):
    url_a = local_server.add_page("/a.html", "<html><body>WidgetPro ships worldwide.</body></html>")
    url_b = local_server.add_page("/b.html", "<html><body>GadgetMax has a mobile app.</body></html>")

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
                json.dumps({"highlights": [{"text": "shipping", "quote": "WidgetPro ships worldwide"}]})
            ],
            "Highlight Extractor for GadgetMax": [
                json.dumps({"highlights": [{"text": "app", "quote": "GadgetMax has a mobile app"}]})
            ],
            "Comparison Agent": [
                json.dumps({"rows": [{"aspect": "Overview", "a_index": 1, "b_index": 1}]})
            ],
        }
    )

    flow = ProductComparisonFlow(llm=llm, search_provider=search, fetch_fn=fetch)
    flow.state.request = CompareRequest(
        product1="WidgetPro", product2="GadgetMax", num_urls=1, num_highlights=1
    )
    await flow.kickoff_async()

    result = flow.state.result
    assert result.product1_ground_rate == 1.0
    assert result.product2_ground_rate == 1.0
    assert result.product1_dropped == 0
    assert result.product2_dropped == 0
