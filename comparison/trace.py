"""Per-run timing, token, and cost reporting.

Flow.usage_metrics only reflects reality for a custom BaseLLM if that LLM
calls BaseLLM._emit_call_started_event / _emit_call_completed_event itself
(confirmed in comparison/fake_llm.py's module docstring and
tests/test_trace_cache_integration.py); FakeLLM does this, so flow.usage_metrics
is trustworthy for fake runs too, not just real-provider ones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class Trace(BaseModel):
    stages: list[dict] = Field(default_factory=list)
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    pages_fetched: int = 0
    pages_failed: int = 0
    estimated_cost_usd: float | None = None
    model_id: str | None = None

    def total_seconds(self) -> float:
        return sum(stage.get("seconds", 0.0) for stage in self.stages)


def _load_pricing(pricing_path: Path | None) -> dict[str, Any]:
    if pricing_path is None or not pricing_path.exists():
        return {}
    with open(pricing_path) as handle:
        data = yaml.safe_load(handle) or {}
    return data.get("models", {})


def estimate_cost_usd(
    model_id: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    pricing_path: Path | None = None,
) -> float | None:
    """Return an estimated USD cost, or None when the model's price is unknown."""
    if not model_id:
        return None
    pricing = _load_pricing(pricing_path)
    entry = pricing.get(model_id)
    if not entry:
        return None
    input_price = entry.get("input_per_million_usd")
    output_price = entry.get("output_per_million_usd")
    if input_price is None or output_price is None:
        return None
    return (prompt_tokens / 1_000_000) * input_price + (completion_tokens / 1_000_000) * output_price


def build_trace(
    flow_trace: dict,
    usage_metrics: Any,
    *,
    cache_hits: int = 0,
    cache_misses: int = 0,
    model_id: str | None = None,
    pricing_path: Path | None = None,
) -> Trace:
    pages_fetched = 0
    pages_failed = 0
    for stage in flow_trace.get("stages", []):
        if stage.get("name") == "fetch":
            pages_fetched += stage.get("fetched", 0)
            pages_failed += len(stage.get("failed", []))

    prompt_tokens = getattr(usage_metrics, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage_metrics, "completion_tokens", 0) or 0
    total_tokens = getattr(usage_metrics, "total_tokens", 0) or (prompt_tokens + completion_tokens)

    return Trace(
        stages=flow_trace.get("stages", []),
        llm_calls=flow_trace.get("llm_calls", 0),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        pages_fetched=pages_fetched,
        pages_failed=pages_failed,
        estimated_cost_usd=estimate_cost_usd(model_id, prompt_tokens, completion_tokens, pricing_path),
        model_id=model_id,
    )
