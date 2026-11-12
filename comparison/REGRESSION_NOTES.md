# Confirmed framework facts (crewai 1.15.34 / crewai-tools 1.15.34)

These were not safe to assume from reading the old code or the docs alone;
each was checked empirically against the pinned version before being relied
on anywhere in this package. Each has a regression test named below.

## 1. The two broken constructor calls in the original code were silently ignored

The original `product_details.py` called `SerperDevTool(query="{product_name}")`
and the original `product_scraper.py` called `ScrapeWebsiteTool(urls="{urls}")`.
Neither tool has a `query` or `urls` constructor parameter
(`SerperDevTool` takes `search_query`, `ScrapeWebsiteTool` takes
`website_url`). On the pinned version, passing an unknown kwarg to either
constructor raises no error and silently drops the value: `tool.model_dump()`
shows no `query`/`urls` key afterwards, and the tool falls back to its real
defaults (`website_url=None` for `ScrapeWebsiteTool`,
`search_type='search'`/`n_results=10` for `SerperDevTool`, ignoring the
templated per-product string entirely). This means the original app's
"per-product search" and "per-product scrape" were never actually
parameterized by product name at all.

Regression test:
`tests/test_search_fetch_integration.py::test_old_constructor_calls_are_silently_ignored`.

Fix: `comparison/search.py::SerperSearch` and `comparison/fetch.py::fetch_page`
never construct a tool with an unsupported kwarg; search queries and fetch
URLs are built and passed in code, not via a broken tool constructor.

## 2. Reply shape needed from a custom BaseLLM when no tool is attached

None of this app's agents has a tool attached (see `comparison/agents.py`).
Confirmed: in that shape, `crewai.BaseLLM.call()` only needs to return text
that parses into the task's `output_pydantic` model. There is no ReAct loop
to drive and no `Thought/Action/Action Input/Observation` format required.
That text format is only needed when a tool *is* attached to an agent whose
LLM does not implement `supports_function_calling()` (crewai falls back to
a text tool-calling prompt in that case,
`crewai/utilities/agent_utils.py::check_native_tool_support` ->
`crewai/translations/en.json`'s `"tools"` slice). This app avoids that path
entirely by design (issue 2), so `comparison/fake_llm.py::FakeLLM` never
needs to emit it.

Regression test: `tests/test_flow_integration.py::test_fake_llm_reply_shape_without_tools`.

## 3. `flow.usage_metrics` vs a custom BaseLLM's own tracked usage

`Flow.usage_metrics` is populated by the flow runtime listening for
`LLMCallCompletedEvent` on the shared `crewai_event_bus`
(`crewai/flow/runtime/__init__.py::_attach_usage_aggregation_listener`), not
by polling `llm.get_token_usage_summary()`. A custom `BaseLLM` subclass that
only calls `self._track_token_usage_internal(usage)` leaves `flow.usage_metrics`
at all zeros even though real calls happened; it must also call
`self._emit_call_started_event(...)` and
`self._emit_call_completed_event(..., call_type=LLMCallType.LLM_CALL, usage=usage)`
itself. `comparison/fake_llm.py::FakeLLM.call` does both, which is why
`flow.usage_metrics` is trustworthy for fake (`--llm fake`) eval runs, not
just real-provider ones.

Regression test: `tests/test_trace_cache_integration.py::test_usage_metrics_reflect_fake_llm_calls`.
