"""Unit tests for main.py's Gradio callback error handling.

Importing main.py builds a real (uncalled) crewai LLM and a DiskCache under
the repo's own .cache/comparison directory -- the same side effect the app
itself already has every time it starts; no network call happens here.
"""

from unittest.mock import patch

import main


def test_comparison_failure_renders_error_instead_of_crashing():
    """comparison/flow.py already isolates one product's crash from the
    other and bounds each stage with a timeout, but neither covers every
    failure mode (a bug in the comparison stage itself, a cache/disk
    error, a crewai exception raised before the flow even starts). main.py
    must catch any such exception from compare() and render a visible
    error, not propagate it and crash the whole Gradio callback."""
    with patch("main.compare", side_effect=RuntimeError("boom")):
        html, trace_json = main.product_comparison("WidgetPro", "GadgetMax", "1", "1")

    assert "Comparison failed" in html
    assert "boom" in html
    assert trace_json == "{}"


def test_valid_input_still_reaches_compare():
    """The try/except must wrap only the compare() call, not swallow the
    existing CompareRequest validation path (invalid input still returns
    its own distinct 'Invalid input' message, never reaching compare())."""
    with patch("main.compare") as mock_compare:
        mock_compare.side_effect = AssertionError("compare should not be called for invalid input")
        html, _ = main.product_comparison("WidgetPro", "WidgetPro", "1", "1")

    assert "Invalid input" in html
    mock_compare.assert_not_called()
