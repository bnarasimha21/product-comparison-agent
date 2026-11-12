"""The single, Gradio-free entry point: compare(request) -> ComparisonResult.

Importing this module makes no network call and does not import Gradio;
main.py is a thin wrapper around compare().
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .cache import CachingSearchProvider, DiskCache, caching_fetch, result_cache_key
from .fetch import fetch_page
from .flow import ProductComparisonFlow
from .models import CompareRequest, ComparisonResult, Page
from .search import SearchProvider, SerperSearch
from .trace import build_trace

ProgressCallback = Callable[[float, str], None]


def compare(
    request: CompareRequest,
    *,
    llm: Any = None,
    search_provider: SearchProvider | None = None,
    fetch_fn: Callable[..., Page] = fetch_page,
    progress: ProgressCallback | None = None,
    cache: DiskCache | None = None,
    pricing_path: Path | None = None,
) -> ComparisonResult:
    """Run one comparison and return a populated ComparisonResult.

    request is validated by CompareRequest's own validators before this is
    ever called (construct it with CompareRequest(...); invalid input raises
    there, before any LLM or network call).

    When cache is given, a second call with an equivalent request (same
    products in the same order, same num_urls/num_highlights) returns the
    cached result with zero search, fetch, or LLM calls. search_provider and
    fetch_fn are wrapped to use the same cache for their own sub-results, so
    a *different* request that shares a search query or a fetched URL with a
    prior run still benefits even on a result-cache miss.
    """
    provider = search_provider or SerperSearch()
    fetcher = fetch_fn

    if cache is not None:
        provider = CachingSearchProvider(provider, cache)
        fetcher = caching_fetch(cache, fetcher)

        cache_key = result_cache_key(request)
        cached_payload = cache.get(cache_key)
        if cached_payload is not None:
            return ComparisonResult.model_validate(cached_payload)

    result = asyncio.run(
        _run_flow(request, llm=llm, search_provider=provider, fetch_fn=fetcher, progress=progress)
    )

    flow_trace, usage_metrics, model_id = result
    comparison_result = flow_trace.pop("_result")
    trace = build_trace(
        flow_trace,
        usage_metrics,
        cache_hits=cache.hits if cache is not None else 0,
        cache_misses=cache.misses if cache is not None else 0,
        model_id=model_id,
        pricing_path=pricing_path,
    )
    comparison_result.trace = trace.model_dump()

    if cache is not None:
        cache.set(result_cache_key(request), comparison_result.model_dump())

    return comparison_result


async def _run_flow(
    request: CompareRequest,
    *,
    llm: Any,
    search_provider: SearchProvider,
    fetch_fn: Callable[..., Page],
    progress: ProgressCallback | None,
) -> tuple[dict, Any, str | None]:
    flow = ProductComparisonFlow(
        llm=llm,
        search_provider=search_provider,
        fetch_fn=fetch_fn,
        progress=progress,
    )
    flow.state.request = request
    await flow.kickoff_async()

    flow_trace = dict(flow.trace)
    flow_trace["_result"] = flow.state.result
    model_id = getattr(llm, "model", None)
    return flow_trace, flow.usage_metrics, model_id
