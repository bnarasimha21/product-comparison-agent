# Product Comparison Agent

An AI-powered agent that compares two products side by side: it searches the web, fetches the pages itself, extracts grounded highlights with an LLM, and renders a safe HTML comparison report. Built with CrewAI Flows for orchestration and Gradio for the UI.

## Architecture

All real logic lives in `comparison/`, which has no Gradio dependency and can be imported and tested without starting a UI. `main.py` is a thin Gradio wrapper around `comparison.pipeline.compare`.

- **`comparison/flow.py`** -- a typed `crewai.Flow[FlowState]` with `@start()`/`@listen()` stages: research each product, then compare. State (search results, fetched pages, highlights, rows) is a Pydantic model throughout, not free-form dict passing.
- **`comparison/search.py`** -- deterministic query templates and a `SearchProvider` protocol (a `FixtureSearch` for tests/evals; a real provider can be swapped in). No LLM tool-wrapper agent sits in front of search.
- **`comparison/fetch.py`** -- fetches a page directly with `requests`, with SSRF protections: https-only by default, private/loopback/link-local/reserved/multicast addresses rejected, bounded redirects, a response-size cap, and hidden-HTML stripping (`display:none`, `visibility:hidden`, the `hidden` attribute) before text extraction. The LLM only ever selects from URLs that code, not the LLM, fetched.
- **`comparison/agents.py`** -- CrewAI agents and tasks with `output_pydantic`, parsed straight from the LLM's JSON text. No agent in this app has a tool attached, so there is no ReAct tool-calling loop to defend or go wrong (see `comparison/REGRESSION_NOTES.md`).
- **`comparison/grounding.py`** -- every highlight's `quote` must be an exact, whitespace/case-normalized substring of the one page it claims to come from (`comparison.grounding.normalize`). Anything that doesn't match is dropped before it reaches the Comparison Agent or the rendered report; the ground rate is reported alongside the result.
- **`comparison/render.py`** -- Jinja2 with `autoescape` on, plus a separate URL-scheme allow-list for links. Prompt-injection defenses: hidden text is stripped before extraction, fetched page text is wrapped in `<page>` delimiters with a "data not instructions" system note, and nothing in a fetched page can inject raw HTML or a `javascript:`/`data:` link into the report.
- **`comparison/cache.py`** -- a disk-backed JSON cache with TTL for both search and fetch, keyed in an order-sensitive way (`(A, B)` and `(B, A)` are different cache entries, deliberately, since the eval harness needs to run both orders without one hiding the other).
- **`comparison/trace.py`** -- per-stage tracing (cache hits/misses, pages fetched, estimated cost from `pricing.yaml`) attached to the result and shown in the Gradio UI as JSON.
- **`comparison/fake_llm.py`** -- a scripted `crewai.BaseLLM` used by every integration test and by `evals/run.py --llm fake`; needs no network and no API key.

### What was and wasn't tested for prompt injection

Tested (see `tests/test_render_integration.py` and `tests/test_grounding_integration.py`): a fetched page containing hidden text with an instruction aimed at the LLM (for example a `display:none` block telling the model to claim a fabricated price) does not reach the rendered report, for two independent reasons: the hidden text is stripped out of the extracted page text before the LLM ever sees it, and even if an instruction were visible, the grounding check would reject any claim that isn't a verbatim substring of the real page text.

Not tested: injection via the product *name* itself, injection through a redirect chain's final URL differing from the one the LLM selected, and anything that would require a real LLM backend (the fake backend is deliberately well-behaved, so it cannot show how an actual model might comply with a visible, non-hidden instruction embedded in page text). The `injection_resistance` metric in `evals/metrics.py` exists so that running `--llm do_serverless` or `--llm claude_cli` against the fixtures (once credentials are configured) would surface that gap against a real model.

### Framework facts confirmed empirically, not assumed

Recorded in `comparison/REGRESSION_NOTES.md`, each backed by a named regression test:

1. `SerperDevTool(query=...)` / `ScrapeWebsiteTool(urls=...)` (the old constructor calls this app used to make) silently drop unknown kwargs instead of raising -- `tests/test_search_fetch_integration.py::test_old_constructor_calls_are_silently_ignored`.
2. No agent in this app has a tool attached, so `FakeLLM` only needs to return JSON text matching `output_pydantic`; there is no ReAct loop to script -- `tests/test_flow_integration.py::test_fake_llm_reply_shape_without_tools`.
3. `flow.usage_metrics` only reflects a custom `BaseLLM`'s calls if that LLM explicitly calls `_emit_call_started_event`/`_emit_call_completed_event` itself -- `tests/test_trace_cache_integration.py::test_usage_metrics_reflect_fake_llm_calls`.

## Offline eval harness

`evals/` runs the real pipeline against six fixed fixtures (`evals/fixtures/*.yaml`: phones, laptops, SaaS tools, dev tools, a near-identical pair, and a sparse pair where one product has no page at all) and reports six metrics per fixture (`evals/metrics.py`): fact recall, ground rate, mix-up rate, order-swap consistency, injection resistance, and count compliance.

```bash
python -m evals.run --llm fake
```

runs fully offline, with zero network calls and zero API keys, using the scripted `FakeLLM`. It writes a markdown report to `evals/results/<date>-fake.md` and prints the per-fixture metrics.

Two real-model backends are implemented in `evals/llm_backends.py` but have not been exercised in this repo because no credentials are configured here, and this assistant never asks for API keys in chat:

- `--llm do_serverless` calls DigitalOcean's serverless inference endpoint and needs `DIGITALOCEAN_INFERENCE_API_KEY` set as an environment variable.
- `--llm claude_cli` shells out to the `claude` CLI already on `PATH` (subscription auth, no API key).

Both raise a clear `RuntimeError` if their prerequisite isn't present, rather than silently falling back to the fake backend.

## Testing

```bash
pip install -r requirements-dev.txt
pytest                    # unit + integration, all offline
ruff check .               # lint
```

Tests are split into plain unit tests (`tests/test_*.py` without the marker) and `@pytest.mark.integration` tests that exercise the real Flow, tasks, and extraction end to end against local fixtures and a local HTTP test server -- still fully offline, no real network calls.

## Setup / Installation

Requires Python 3.11 or newer.

```bash
git clone https://github.com/bnarasimha21/product-comparison-agent.git
cd product-comparison-agent
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root with your API keys (`OPENAI_API_KEY` drives the agents' default LLM; search/fetch no longer depend on a third-party search API key by default since `comparison/search.py` and `comparison/fetch.py` are plain code):

```env
OPENAI_API_KEY=your_openai_api_key
```

## Usage

```bash
python main.py
```

1. Open the Gradio interface in your browser (default: `http://localhost:7860`)
2. Enter two product names to compare (e.g., "iPhone 15 Pro" and "Samsung Galaxy S24 Ultra")
3. Select the number of URLs to search per product (1-5)
4. Select the number of highlights to compare (1-10)
5. Click **Submit** and wait for the agents to research and generate the comparison report

The output is an HTML table comparing the two products across the selected number of feature highlights, plus a trace panel showing cache hits, pages fetched, and estimated cost.

## License

MIT
