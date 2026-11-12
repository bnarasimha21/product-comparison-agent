"""A scripted crewai.BaseLLM for offline tests and `evals/run.py --llm fake`.

Confirmed on the pinned crewai 1.15.34 (recorded here and in
tests/test_flow_integration.py::test_fake_llm_reply_shape_without_tools):
none of this app's agents ever has a tool attached (search and fetch are
plain code in comparison/search.py and comparison/fetch.py; the LLM only
ever produces a structured output_pydantic response). In that shape, a
custom BaseLLM.call simply needs to return text that parses into the
task's output_pydantic model -- there is no ReAct loop to drive and no
Thought/Action/Action Input/Observation format to produce.

That ReAct text format *would* be required if a tool were attached to an
agent whose LLM does not implement `supports_function_calling()` (the
default for a bare BaseLLM subclass): crewai's
`utilities.agent_utils.check_native_tool_support` falls back to a text
tool-calling prompt (crewai/translations/en.json, the "tools" slice) that
expects exactly:

    Thought: ...
    Action: <tool name>
    Action Input: {"key": "value"}
    Observation: ...

    Thought: I now know the final answer
    Final Answer: ...

This app deliberately avoids that path entirely (issue 2), which is why
FakeLLM below never needs to emit it.

Second confirmed finding (tests/test_trace_cache_integration.py and
comparison/trace.py): ``Flow.usage_metrics`` is NOT populated just because a
custom BaseLLM tracks its own usage via ``_track_token_usage_internal``.
crewai's flow runtime aggregates usage by listening for
``LLMCallCompletedEvent`` on the shared event bus
(crewai/flow/runtime/__init__.py, ``_attach_usage_aggregation_listener``),
and that event is only emitted by calling the ``BaseLLM`` helper methods
``_emit_call_started_event`` / ``_emit_call_completed_event`` -- built-in
provider classes (OpenAI, Anthropic, ...) call these themselves inside
their own ``call()``; a bare ``BaseLLM`` subclass must call them explicitly
or ``flow.usage_metrics`` stays at all zeros even though real LLM calls
happened. FakeLLM.call below does this, with a deterministic
four-characters-per-token estimate (the same fallback BaseLLM itself uses
in ``get_input_tokens_count``) standing in for real token counts.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from crewai import BaseLLM
from crewai.events.types.llm_events import LLMCallType
from pydantic import PrivateAttr

ScriptEntry = str | Callable[[dict], str]


def _estimate_tokens(text: Any) -> int:
    if not isinstance(text, str):
        text = str(text)
    return max(1, (len(text) + 3) // 4)


class FakeLLM(BaseLLM):
    """Replies are scripted per agent role, consumed in order (a queue).

    scripts: {role_name: [reply_or_callable, ...]}. A callable is called
    with the call record ({"role", "messages"}) and must return the reply
    text. When a role's queue is empty, `default` is used if given,
    otherwise FakeLLM raises -- a missing script entry is a test bug, not
    something to paper over with a generic reply.
    """

    _scripts: dict[str, list[ScriptEntry]] = PrivateAttr(default_factory=dict)
    _calls: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _default: ScriptEntry | None = PrivateAttr(default=None)

    def __init__(
        self,
        scripts: dict[str, list[ScriptEntry]] | None = None,
        default: ScriptEntry | None = None,
        **data: Any,
    ) -> None:
        data.setdefault("model", "fake/fake-1")
        super().__init__(**data)
        self._scripts = {role: list(entries) for role, entries in (scripts or {}).items()}
        self._calls = []
        self._default = default

    @property
    def calls(self) -> list[dict[str, Any]]:
        return list(self._calls)

    def add_script(self, role: str, entries: list[ScriptEntry]) -> None:
        self._scripts.setdefault(role, []).extend(entries)

    def call(
        self,
        messages: Any,
        tools: Any = None,
        callbacks: Any = None,
        available_functions: Any = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> str:
        role = getattr(from_agent, "role", None) or "__default__"
        record = {"role": role, "messages": messages, "tools": tools}
        self._calls.append(record)

        self._emit_call_started_event(
            messages=messages,
            tools=tools,
            callbacks=callbacks,
            available_functions=available_functions,
            from_task=from_task,
            from_agent=from_agent,
        )

        queue = self._scripts.get(role)
        if queue:
            entry = queue.pop(0)
        elif self._default is not None:
            entry = self._default
        else:
            raise AssertionError(
                f"FakeLLM has no scripted response left for role {role!r}. "
                f"Recorded roles so far: {[c['role'] for c in self._calls]}"
            )

        reply = entry(record) if callable(entry) else entry

        prompt_tokens = _estimate_tokens(messages)
        completion_tokens = _estimate_tokens(reply)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "successful_requests": 1,
        }
        self._track_token_usage_internal(usage)
        self._emit_call_completed_event(
            response=reply,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            messages=messages,
            usage=usage,
        )
        return reply
