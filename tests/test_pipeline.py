"""Unit tests for pure helpers in comparison/pipeline.py (no I/O)."""

from types import SimpleNamespace

from comparison.pipeline import _model_id_for_pricing


def test_model_id_for_pricing_joins_provider_and_model():
    """crewai's LLM() factory for a known provider (e.g. "openai/gpt-4o-mini")
    returns a provider-specific object whose own `.model` has the provider
    prefix stripped back off, with the provider on a separate attribute.
    main.py relies on this rejoining to produce a pricing.yaml-matching key
    like "openai/gpt-4o-mini", not a bare "gpt-4o-mini" that would never
    match anything and silently leave cost blank."""
    llm = SimpleNamespace(model="gpt-4o-mini", provider="openai")
    assert _model_id_for_pricing(llm) == "openai/gpt-4o-mini"


def test_model_id_for_pricing_leaves_self_contained_model_unchanged():
    """FakeLLM and the evals/llm_backends.py BaseLLM subclasses already set
    a self-contained model string (e.g. "fake/fake-1") with no provider
    attribute; they must pass through unchanged, not get double-prefixed."""
    llm = SimpleNamespace(model="fake/fake-1")
    assert _model_id_for_pricing(llm) == "fake/fake-1"


def test_model_id_for_pricing_handles_none_llm():
    assert _model_id_for_pricing(None) is None


def test_model_id_for_pricing_handles_missing_model_attribute():
    assert _model_id_for_pricing(SimpleNamespace()) is None
