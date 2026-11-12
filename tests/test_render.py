from comparison.models import ComparisonResult, ComparisonRow
from comparison.render import render_html


def _result(**overrides) -> ComparisonResult:
    defaults = dict(
        product1_name="WidgetPro",
        product2_name="GadgetMax",
        rows=[
            ComparisonRow(
                aspect="Price",
                a_text="$19/mo",
                a_source="https://a.example/pricing",
                b_text="$25/mo",
                b_source="https://b.example/pricing",
            )
        ],
    )
    defaults.update(overrides)
    return ComparisonResult(**defaults)


def test_basic_render_contains_rows_and_names():
    html = render_html(_result())
    assert "WidgetPro" in html
    assert "GadgetMax" in html
    assert "$19/mo" in html
    assert "$25/mo" in html
    assert 'href="https://a.example/pricing"' in html
    assert 'href="https://b.example/pricing"' in html


def test_script_tag_in_aspect_text_is_escaped():
    row = ComparisonRow(aspect='<script>alert("xss")</script>', a_text="x", b_text="y")
    html = render_html(_result(rows=[row]))
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_onerror_attribute_text_is_escaped():
    """The word "onerror=" surviving as escaped text is fine; what matters is
    that it never becomes a live HTML attribute, i.e. no bare `<img` tag and
    the quote character around it is escaped so it cannot break out of a
    real attribute context."""
    row = ComparisonRow(aspect="Price", a_text='<img src=x onerror="alert(1)">', b_text="y")
    html = render_html(_result(rows=[row]))
    assert "<img" not in html
    assert "&lt;img" in html
    assert "&#34;alert(1)&#34;" in html


def test_javascript_scheme_source_not_emitted_as_link():
    row = ComparisonRow(aspect="Price", a_text="x", a_source="javascript:alert(1)", b_text="y")
    html = render_html(_result(rows=[row]))
    assert "javascript:" not in html


def test_data_scheme_source_not_emitted_as_link():
    row = ComparisonRow(aspect="Price", a_text="x", a_source="data:text/html,hi", b_text="y")
    html = render_html(_result(rows=[row]))
    assert "data:text/html" not in html


def test_missing_text_renders_no_evidence_found():
    row = ComparisonRow(aspect="Battery life")
    html = render_html(_result(rows=[row]))
    assert "no evidence found" in html


def test_ground_rate_and_dropped_counts_shown():
    html = render_html(_result(product1_ground_rate=0.5, product1_dropped=1, product2_ground_rate=1.0))
    assert "50%" in html
    assert "1 dropped as ungrounded" in html


def test_error_notes_shown_when_present():
    html = render_html(_result(product1_error="research stage timed out after 120.0s"))
    assert "research stage timed out after 120.0s" in html


def test_no_rows_renders_fallback_row():
    html = render_html(_result(rows=[]))
    assert "No comparison rows could be produced." in html


def test_identical_input_produces_byte_identical_output():
    result = _result()
    assert render_html(result) == render_html(result)
