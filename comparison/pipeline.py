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
    # Known before the flow even runs: llm is given to us directly, so the
    # cache key (and later, this request's own hit/miss delta) can use it
    # from the start rather than waiting for the flow to finish.
    model_id = _model_id_for_pricing(llm)

    hits_before = misses_before = 0
    if cache is not None:
        provider = CachingSearchProvider(provider, cache)
        fetcher = caching_fetch(cache, fetcher)

        cache_key = result_cache_key(request, model_id)
        # Snapshot before the lookup below so that lookup's own hit/miss is
        # still attributed to this request (see hits/misses diff below).
        hits_before, misses_before = cache.hits, cache.misses
        cached_payload = cache.get(cache_key)
        if cached_payload is not None:
            return ComparisonResult.model_validate(cached_payload)

    result = asyncio.run(
        _run_flow(request, llm=llm, search_provider=provider, fetch_fn=fetcher, progress=progress)
    )

    flow_trace, usage_metrics = result
    comparison_result = flow_trace.pop("_result")
    # DiskCache.hits/.misses are instance attributes that accumulate for as
    # long as the DiskCache instance lives -- main.py constructs exactly one
    # DiskCache and reuses it for every Gradio callback invocation for the
    # life of the process, so reading cache.hits/cache.misses directly here
    # would report a running total across every request the server has ever
    # served, not this one. Diffing against the snapshot taken just before
    # this request's own cache activity started isolates this request's own
    # count, which is what trace.cache_hits/cache_misses is meant to show.
    cache_hits = cache.hits - hits_before if cache is not None else 0
    cache_misses = cache.misses - misses_before if cache is not None else 0
    trace = build_trace(
        flow_trace,
        usage_metrics,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        model_id=model_id,
        pricing_path=pricing_path,
    )
    comparison_result.trace = trace.model_dump()

    if cache is not None:
        cache.set(result_cache_key(request, model_id), comparison_result.model_dump())

    return comparison_result


async def _run_flow(
    request: CompareRequest,
    *,
    llm: Any,
    search_provider: SearchProvider,
    fetch_fn: Callable[..., Page],
    progress: ProgressCallback | None,
) -> tuple[dict, Any]:
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
    return flow_trace, flow.usage_metrics


def _model_id_for_pricing(llm: Any) -> str | None:
    """The pricing.yaml key for llm, e.g. "openai/gpt-4o-mini".

    crewai's LLM() factory auto-detects a known provider prefix (e.g.
    "openai/gpt-4o-mini") and returns a provider-specific subclass whose
    own `.model` attribute has that prefix stripped back off (just
    "gpt-4o-mini", with the provider name on a separate `.provider`
    attribute) -- `getattr(llm, "model", None)` alone would then never
    match a pricing.yaml key, silently leaving cost blank again even
    though a real, priced model answered every call. Re-join provider and
    model when both are present and the model string does not already
    carry a "/"; FakeLLM and the evals/llm_backends.py BaseLLM subclasses
    already set a self-contained model string (e.g. "fake/fake-1") with no
    separate `.provider`, so they pass through unchanged.
    """
    model_id = getattr(llm, "model", None)
    if not model_id:
        return None
    provider = getattr(llm, "provider", None)
    if provider and "/" not in model_id:
        return f"{provider}/{model_id}"
    return model_id
