"""Render a ComparisonResult to HTML with a template, not a model.

Replaces output_pydantic=HtmlReport in the original product_comparison.py,
where the LLM authored the entire HTML table and main.py rendered it raw
via gr.HTML. Scraped web text is attacker-controlled; letting a model turn
it directly into markup is a prompt-injection-to-XSS path. Jinja2's
autoescape handles ordinary escaping; the URL scheme allow-list below is a
second, independent guard specifically for the `href` attribute, since
autoescape alone does not stop a `javascript:` URL from becoming a live
link.
"""

from __future__ import annotations

from jinja2 import Environment, select_autoescape

from .models import ComparisonResult

_ALLOWED_URL_SCHEMES = ("http://", "https://")

_TEMPLATE_SOURCE = """
<div class="product-comparison-report">
  <table>
    <thead>
      <tr>
        <th>Aspect</th>
        <th>{{ result.product1_name }}</th>
        <th>{{ result.product2_name }}</th>
      </tr>
    </thead>
    <tbody>
      {% for row in result.rows %}
      <tr>
        <td>{{ row.aspect }}</td>
        <td>
          {{ row.a_text if row.a_text else "no evidence found" }}
          {% set a_href = safe_href(row.a_source) %}
          {% if a_href %}
          <br><a href="{{ a_href }}" rel="noopener noreferrer" target="_blank">source</a>
          {% endif %}
        </td>
        <td>
          {{ row.b_text if row.b_text else "no evidence found" }}
          {% set b_href = safe_href(row.b_source) %}
          {% if b_href %}
          <br><a href="{{ b_href }}" rel="noopener noreferrer" target="_blank">source</a>
          {% endif %}
        </td>
      </tr>
      {% else %}
      <tr><td colspan="3">No comparison rows could be produced.</td></tr>
      {% endfor %}
    </tbody>
  </table>
  <p class="report-footer">
    {{ result.product1_name }} ground rate: {{ "%.0f"|format(result.product1_ground_rate * 100) }}%
    ({{ result.product1_dropped }} dropped as ungrounded).
    {{ result.product2_name }} ground rate: {{ "%.0f"|format(result.product2_ground_rate * 100) }}%
    ({{ result.product2_dropped }} dropped as ungrounded).
    {% if result.product1_error %}<br>{{ result.product1_name }}: {{ result.product1_error }}{% endif %}
    {% if result.product2_error %}<br>{{ result.product2_name }}: {{ result.product2_error }}{% endif %}
  </p>
</div>
""".strip()

_env = Environment(autoescape=select_autoescape(enabled_extensions=(), default_for_string=True))


def _safe_href(url: str | None) -> str | None:
    """Return url unchanged if it starts with an allowed scheme, else None.

    This is the only place a Highlight's source_url becomes a clickable
    link; a `javascript:` (or any other) scheme never reaches an href.
    """
    if not url:
        return None
    if url.lower().startswith(_ALLOWED_URL_SCHEMES):
        return url
    return None


_template = _env.from_string(_TEMPLATE_SOURCE)


def render_html(result: ComparisonResult) -> str:
    """Render result to a byte-identical-for-identical-input HTML string."""
    return _template.render(result=result, safe_href=_safe_href)
