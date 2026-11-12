"""Fixture loading plus the three LLM backends the eval runner supports.

``--llm fake`` needs nothing: it builds a deterministic, scripted FakeLLM
per fixture that only ever quotes verbatim from the page text it is given
(see ``build_fake_llm``), so `python -m evals.run --llm fake` is fully
offline and needs zero API keys. It exists to validate the pipeline and
the metrics in evals/metrics.py, not to measure real model quality.

``--llm do_serverless`` and ``--llm claude_cli`` are real backends for
measuring an actual model, and both need credentials this session does not
have:

- do_serverless calls DigitalOcean's serverless inference endpoint
  (https://inference.do-ai.run/v1) and needs ``DIGITALOCEAN_INFERENCE_API_KEY``
  set as an environment variable. It raises RuntimeError with that message
  if the variable is unset; never prompts for it.
- claude_cli shells out to the ``claude`` CLI already on PATH (the
  subscription route, no API key) and raises RuntimeError if the binary is
  not found.

Neither has been exercised end-to-end in this session (no key configured,
no interactive CLI session available here); the code paths are real, not
stubs, but are unverified beyond their own construction.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from crewai import BaseLLM

from comparison.fake_llm import FakeLLM
from comparison.fetch import _extract_text
from comparison.models import Page, SearchResult
from comparison.search import FixtureSearch, query_templates

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@dataclass
class Fixture:
    id: str
    product1: str
    product2: str
    num_urls: int
    num_highlights: int
    pages: dict[str, dict[str, str]]  # {"product1": {url: html}, "product2": {...}}
    golden: dict[str, list[str]]
    false_facts: dict[str, list[str]] = field(default_factory=dict)
    # Known prompt-injection strings planted in this fixture's page text
    # (e.g. "IGNORE ALL PREVIOUS INSTRUCTIONS..."), keyed by product name.
    # evals/run.py feeds these into metrics.injection_resistance so the
    # metric is computed against real injected content instead of an
    # always-empty list (see injection_attempt.yaml for the one fixture
    # that actually plants one).
    injected_strings: dict[str, list[str]] = field(default_factory=dict)

    def injected_strings_for(self, product_name: str) -> list[str]:
        return self.injected_strings.get(product_name, [])

    def all_injected_strings(self) -> list[str]:
        return self.injected_strings_for(self.product1) + self.injected_strings_for(self.product2)

    def pages_for(self, product_name: str) -> dict[str, str]:
        if product_name == self.product1:
            return self.pages.get("product1", {})
        if product_name == self.product2:
            return self.pages.get("product2", {})
        return {}

    def all_pages(self) -> dict[str, str]:
        merged: dict[str, str] = {}
        merged.update(self.pages.get("product1", {}))
        merged.update(self.pages.get("product2", {}))
        return merged


def load_fixture(path: Path) -> Fixture:
    data = yaml.safe_load(path.read_text())
    return Fixture(
        id=data["id"],
        product1=data["product1"],
        product2=data["product2"],
        num_urls=data.get("num_urls", 2),
        num_highlights=data.get("num_highlights", 3),
        pages=data.get("pages", {}),
        golden=data.get("golden", {}),
        false_facts=data.get("false_facts", {}) or {},
        injected_strings=data.get("injected_strings", {}) or {},
    )


def load_all_fixtures() -> list[Fixture]:
    return [load_fixture(p) for p in sorted(FIXTURES_DIR.glob("*.yaml"))]


def build_search_provider(fixture: Fixture) -> FixtureSearch:
    table: dict[str, list[SearchResult]] = {}
    for product_name in (fixture.product1, fixture.product2):
        urls = list(fixture.pages_for(product_name).keys())
        results = [SearchResult(title=product_name, url=url, snippet="") for url in urls]
        for query in query_templates(product_name):
            table[query] = results
    return FixtureSearch(table)


def build_fetch_fn(fixture: Fixture):
    pages = fixture.all_pages()

    def fetch(url: str, **kwargs: Any) -> Page:
        html = pages.get(url)
        if html is None:
            return Page(url=url, text="", ok=False, error="fixture has no page for this url")
        return Page(url=url, text=_extract_text(html), ok=True, error=None)

    return fetch


def _message_text(messages: Any) -> str:
    """Join the real `content` text of each message, with real newlines.

    `str(messages)` on crewai's message list produces a Python repr, where
    every real newline inside a message's content is escaped as the two
    characters backslash-n rather than an actual line break; regexes
    anchored on `\\n` would silently never match against that. This pulls
    the actual content strings out instead.
    """
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list):
        parts = []
        for message in messages:
            if isinstance(message, dict):
                parts.append(str(message.get("content", "")))
            else:
                parts.append(str(message))
        return "\n".join(parts)
    return str(messages)


_NUMBERED_LINE = re.compile(r"^\s*\d+\.\s*(.+?)\s*$", re.MULTILINE)


def _parse_candidates(prompt_text: str) -> list[str]:
    """Pull URLs back out of the "N. <url> -- title: snippet" candidate
    list format comparison/agents.py::format_candidates builds."""
    urls = []
    for line in prompt_text.splitlines():
        match = re.match(r"\s*\d+\.\s*(\S+)\s*--", line)
        if match:
            urls.append(match.group(1))
    return urls


def _url_selector_reply(record: dict, fixture: Fixture, product_name: str) -> str:
    prompt_text = _message_text(record["messages"])
    candidates = _parse_candidates(prompt_text)
    known = list(fixture.pages_for(product_name).keys())
    chosen = [url for url in candidates if url in known] or known
    return json.dumps({"urls": chosen[: fixture.num_urls]})


def _highlight_extractor_reply(record: dict, fixture: Fixture, product_name: str) -> str:
    """Scripted "model" reply for the Highlight Extractor agent.

    By default this is deliberately well-behaved: it only ever emits a
    highlight whose quote is a literal, verbatim substring of the
    page_text it was given (never of the other product's page, never of
    an injected instruction it was told to ignore).

    For a fixture that sets `injected_strings` (currently only
    injection_attempt.yaml), it ALSO echoes back any injected string that
    is present in the prompt text verbatim, simulating a model that fell
    for a planted "ignore previous instructions" style injection. This is
    what lets evals/run.py's injection_resistance metric be exercised
    against a real violation instead of staying hardcoded at 1.0 because
    it was never given anything to detect (see evals/run.py and
    evals/metrics.py for the consuming side).
    """
    prompt_text = _message_text(record["messages"])
    facts = fixture.golden.get(product_name, [])
    highlights = []
    for fact in facts:
        if fact.lower() in prompt_text.lower():
            highlights.append({"text": fact, "quote": fact})
    for injected in fixture.injected_strings_for(product_name):
        if injected.lower() in prompt_text.lower():
            highlights.append({"text": injected, "quote": injected})
    return json.dumps({"highlights": highlights})


_SECTION_RE = re.compile(
    r"Grounded highlights for A \(1-based\):\n(?P<a>.*?)\n\nProduct B",
    re.DOTALL,
)
_SECTION_B_RE = re.compile(
    r"Grounded highlights for B \(1-based\):\n(?P<b>.*?)\n\nBuild up to",
    re.DOTALL,
)


def _comparison_reply(record: dict, fixture: Fixture) -> str:
    prompt_text = _message_text(record["messages"])
    a_match = _SECTION_RE.search(prompt_text)
    b_match = _SECTION_B_RE.search(prompt_text)
    a_lines = _NUMBERED_LINE.findall(a_match.group("a")) if a_match else []
    b_lines = _NUMBERED_LINE.findall(b_match.group("b")) if b_match else []
    a_count = len(a_lines) if a_lines and a_lines[0] != "(no grounded highlights)" else 0
    b_count = len(b_lines) if b_lines and b_lines[0] != "(no grounded highlights)" else 0

    rows = []
    for i in range(max(a_count, b_count, 1) if (a_count or b_count) else 0):
        rows.append(
            {
                "aspect": f"Highlight {i + 1}",
                "a_index": i + 1 if i < a_count else None,
                "b_index": i + 1 if i < b_count else None,
            }
        )
    return json.dumps({"rows": rows[: fixture.num_highlights]})


def build_fake_llm(fixture: Fixture) -> FakeLLM:
    llm = FakeLLM()
    for product_name in (fixture.product1, fixture.product2):
        llm.add_script(
            f"URL Selector for {product_name}",
            [lambda record, p=product_name: _url_selector_reply(record, fixture, p)] * 3,
        )
        llm.add_script(
            f"Highlight Extractor for {product_name}",
            [lambda record, p=product_name: _highlight_extractor_reply(record, fixture, p)] * 5,
        )
    llm.add_script(
        "Comparison Agent",
        [lambda record: _comparison_reply(record, fixture)] * 3,
    )
    return llm


class DoServerlessLLM(BaseLLM):
    """Real DigitalOcean serverless inference backend.

    Needs DIGITALOCEAN_INFERENCE_API_KEY as an environment variable; never
    read from chat or asked for interactively. Untested in this session
    (no key configured here) but not a stub: it makes a real HTTP call.
    """

    def __init__(self, model: str = "llama3.3-70b-instruct", api_key: str | None = None, **data: Any) -> None:
        import os

        super().__init__(model=model, **data)
        self._api_key = api_key or os.environ.get("DIGITALOCEAN_INFERENCE_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "DoServerlessLLM needs DIGITALOCEAN_INFERENCE_API_KEY set as an "
                "environment variable (never pass it in chat)."
            )

    def call(self, messages: Any, tools: Any = None, callbacks: Any = None, **kwargs: Any) -> str:
        import requests

        self._emit_call_started_event(messages=messages, tools=tools, callbacks=callbacks)
        response = requests.post(
            "https://inference.do-ai.run/v1/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self.model, "messages": messages},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        reply = payload["choices"][0]["message"]["content"]
        usage_raw = payload.get("usage", {})
        usage = {
            "prompt_tokens": usage_raw.get("prompt_tokens", 0),
            "completion_tokens": usage_raw.get("completion_tokens", 0),
            "total_tokens": usage_raw.get("total_tokens", 0),
            "successful_requests": 1,
        }
        self._track_token_usage_internal(usage)
        from crewai.events.types.llm_events import LLMCallType

        self._emit_call_completed_event(
            response=reply, call_type=LLMCallType.LLM_CALL, messages=messages, usage=usage
        )
        return reply


class ClaudeCliLLM(BaseLLM):
    """Shells out to the `claude` CLI (subscription auth, no API key).

    Raises RuntimeError if the binary is not on PATH. Untested in this
    session (no interactive CLI available here) but not a stub.
    """

    def __init__(self, model: str = "claude-cli", binary: str | None = None, **data: Any) -> None:
        super().__init__(model=model, **data)
        self._binary = binary or shutil.which("claude")
        if not self._binary:
            raise RuntimeError("ClaudeCliLLM needs the `claude` CLI on PATH.")

    def call(self, messages: Any, tools: Any = None, callbacks: Any = None, **kwargs: Any) -> str:
        self._emit_call_started_event(messages=messages, tools=tools, callbacks=callbacks)
        prompt = "\n".join(
            m.get("content", "") if isinstance(m, dict) else str(m) for m in (messages or [])
        )
        completed = subprocess.run(
            [self._binary, "-p", prompt],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        reply = completed.stdout.strip()
        usage = {
            "prompt_tokens": max(1, len(prompt) // 4),
            "completion_tokens": max(1, len(reply) // 4),
            "total_tokens": max(1, (len(prompt) + len(reply)) // 4),
            "successful_requests": 1,
        }
        self._track_token_usage_internal(usage)
        from crewai.events.types.llm_events import LLMCallType

        self._emit_call_completed_event(
            response=reply, call_type=LLMCallType.LLM_CALL, messages=messages, usage=usage
        )
        return reply


def build_llm(name: str, fixture: Fixture | None = None) -> Any:
    if name == "fake":
        if fixture is None:
            raise ValueError("fake backend needs a fixture to script replies from")
        return build_fake_llm(fixture)
    if name == "do_serverless":
        return DoServerlessLLM()
    if name == "claude_cli":
        return ClaudeCliLLM()
    raise ValueError(f"unknown --llm backend: {name!r}")
