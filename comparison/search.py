"""Search providers.

A SearchProvider turns a query into a list of SearchResult. The pipeline
never lets the LLM build search queries or search-tool call arguments
itself (see the module docstring in comparison/fetch.py for why); it always
calls a SearchProvider directly in code and only asks the LLM to pick among
the URLs that came back.
"""

from __future__ import annotations

from typing import Protocol

from .models import SearchResult


class SearchProvider(Protocol):
    def search(self, query: str, n: int) -> list[SearchResult]:
        """Return up to n results for query."""
        ...


class SerperSearch:
    """Calls crewai_tools.SerperDevTool directly, with a real search_query.

    This bypasses the broken pattern in the original product_details.py,
    where SerperDevTool(query="{product_name}") passed an argument the tool
    does not have (see comparison/REGRESSION_NOTES.md and
    tests/test_search_fetch_integration.py::test_old_constructor_calls_are_silently_ignored
    for the confirmed behaviour on crewai-tools 1.15.34: unknown constructor
    kwargs are dropped silently, no error, no value stored).
    """

    def __init__(self, api_key: str | None = None, n_results: int = 10) -> None:
        from crewai_tools import SerperDevTool

        kwargs: dict = {"n_results": n_results}
        if api_key is not None:
            kwargs["api_key"] = api_key
        self._tool = SerperDevTool(**kwargs)

    def search(self, query: str, n: int) -> list[SearchResult]:
        raw = self._tool.run(search_query=query)
        return _parse_serper_output(raw)[:n]


def _parse_serper_output(raw) -> list[SearchResult]:
    """Normalize SerperDevTool's return value into SearchResult objects.

    The tool can return a dict (organic results under "organic"), a JSON
    string, or plain text depending on version/settings; handle the shapes
    we can, and fail closed (empty list) rather than guess at the rest.
    """
    import json

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return []

    if isinstance(raw, dict):
        organic = raw.get("organic") or raw.get("results") or []
        results = []
        for item in organic:
            if not isinstance(item, dict):
                continue
            url = item.get("link") or item.get("url")
            if not url:
                continue
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=url,
                    snippet=item.get("snippet", ""),
                )
            )
        return results

    if isinstance(raw, list):
        results = []
        for item in raw:
            if isinstance(item, dict) and item.get("url"):
                results.append(
                    SearchResult(
                        title=item.get("title", ""),
                        url=item["url"],
                        snippet=item.get("snippet", ""),
                    )
                )
        return results

    return []


class FixtureSearch:
    """A deterministic SearchProvider for tests and evals.

    results_by_query maps a query string to the canned list of results it
    should return; an unknown query returns an empty list rather than
    raising, matching how a real provider might legitimately find nothing.
    """

    def __init__(self, results_by_query: dict[str, list[SearchResult]]) -> None:
        self._results = results_by_query

    def search(self, query: str, n: int) -> list[SearchResult]:
        return list(self._results.get(query, []))[:n]


def query_templates(product_name: str) -> list[str]:
    """The fixed set of queries issued per product.

    Kept in code (not model-authored) so results are reproducible and the
    same templates can be used to build FixtureSearch tables in tests.
    """
    return [
        f"{product_name} official site",
        f"{product_name} pricing documentation",
        f"{product_name} review",
    ]
