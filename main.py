"""Thin Gradio wrapper around comparison.pipeline.compare.

All real logic lives in the comparison/ package, which has no Gradio
dependency and can be imported and tested without starting a UI.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr
from crewai import LLM
from dotenv import load_dotenv
from pydantic import ValidationError

from comparison.cache import DiskCache
from comparison.models import CompareRequest
from comparison.pipeline import compare
from comparison.render import render_html

load_dotenv()

_PRICING_PATH = Path(__file__).parent / "pricing.yaml"
_CACHE_DIR = Path(__file__).parent / ".cache" / "comparison"
_cache = DiskCache(_CACHE_DIR)

# comparison/pipeline.py reads model_id off of `llm.model` (see
# _run_flow's `getattr(llm, "model", None)`) to look up a per-token price
# in pricing.yaml and compute a run's cost. Passing llm=None all the way
# through (the old behaviour) left model_id -- and therefore cost --
# always blank, even though a real model answered every call: crewai's
# own Agent(llm=None) default is resolved deep inside each Agent, not
# visible back here. Building one real, named LLM here and passing it
# through is what makes cost reporting work at all.
_MODEL_ID = os.environ.get("MODEL_ID", "openai/gpt-4o-mini")
_llm = LLM(model=_MODEL_ID)


def product_comparison(
    product_name1: str,
    product_name2: str,
    num_urls: str,
    num_highlights: str,
    progress: gr.Progress = gr.Progress(track_tqdm=True),  # noqa: B008 (Gradio's own documented pattern: it detects this default to wire up progress tracking)
) -> tuple[str, str]:
    try:
        request = CompareRequest(
            product1=product_name1,
            product2=product_name2,
            num_urls=int(num_urls),
            num_highlights=int(num_highlights),
        )
    except (ValidationError, ValueError) as exc:
        error_html = f"<p class='error'>Invalid input: {exc}</p>"
        return error_html, "{}"

    def report_progress(fraction: float, message: str) -> None:
        progress(fraction, desc=message)

    try:
        result = compare(
            request, llm=_llm, progress=report_progress, cache=_cache, pricing_path=_PRICING_PATH
        )
    except Exception as exc:  # noqa: BLE001 -- last-resort guard, see comment below
        # comparison/flow.py already isolates one product's crash from the
        # other (asyncio.gather(..., return_exceptions=True)) and bounds
        # each stage with a timeout, but neither covers every failure mode:
        # a bug in the comparison stage itself, a cache/disk error, or a
        # crewai/library exception raised before the flow even starts. Any
        # of those must render a visible error in the UI, not crash this
        # Gradio callback and take the whole running server down with it.
        error_html = f"<p class='error'>Comparison failed: {exc}</p>"
        return error_html, "{}"
    return render_html(result), json.dumps(result.trace or {}, indent=2)


iface = gr.Interface(
    fn=product_comparison,
    inputs=[
        gr.Textbox(label="Product Name 1"),
        gr.Textbox(label="Product Name 2"),
        gr.Dropdown(label="Number of URLs to Search", choices=["1", "2", "3", "4", "5"], value="2"),
        gr.Dropdown(
            label="Number of Highlights to compare",
            choices=[str(n) for n in range(1, 11)],
            value="2",
        ),
    ],
    outputs=[
        gr.HTML(label="Product Comparison Report"),
        gr.Code(label="Run details (timing, tokens, cost, cache)", language="json"),
    ],
    title="Product Comparison",
    description="Enter two product names to compare their details.",
)

if __name__ == "__main__":
    iface.launch()
