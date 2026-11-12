"""Thin Gradio wrapper around comparison.pipeline.compare.

All real logic lives in the comparison/ package, which has no Gradio
dependency and can be imported and tested without starting a UI.
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
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

    result = compare(request, progress=report_progress, cache=_cache, pricing_path=_PRICING_PATH)
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
