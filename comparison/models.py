"""Typed data structures shared across the pipeline.

Nothing in this module imports Gradio, crewai, or a network library, so it
can be imported anywhere (pipeline code, tests, evals) without side effects.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


class CompareRequest(BaseModel):
    """Validated input for one comparison run.

    Validation runs before any LLM or network call is made, so a bad
    request (empty name, identical products, an out-of-range count) fails
    fast and cheaply.
    """

    product1: str
    product2: str
    num_urls: int = Field(ge=1, le=5)
    num_highlights: int = Field(ge=1, le=10)

    @field_validator("product1", "product2")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("product name must not be empty")
        return value

    @model_validator(mode="after")
    def _distinct_products(self) -> CompareRequest:
        if self.product1.strip().casefold() == self.product2.strip().casefold():
            raise ValueError("product1 and product2 must be distinct")
        return self


class SearchResult(BaseModel):
    """One result returned by a SearchProvider."""

    title: str
    url: str
    snippet: str = ""


class Page(BaseModel):
    """The outcome of fetching one URL, success or failure.

    ``ok`` is false for every rejected or failed fetch (bad scheme, private
    address, timeout, non-2xx, oversize body); ``error`` then carries a short
    human-readable reason. ``text`` is the empty string whenever ``ok`` is
    false.
    """

    url: str
    text: str = ""
    ok: bool = True
    error: str | None = None


class Highlight(BaseModel):
    """One claim about a product, with the evidence it came from.

    ``quote`` must be a verbatim, substring-matchable excerpt of the page at
    ``source_url`` (see comparison.grounding.is_grounded). ``text`` is the
    human-readable highlight derived from that quote.
    """

    text: str
    source_url: str
    quote: str


class ProductReport(BaseModel):
    """What the research stage produced for a single product."""

    name: str
    highlights: list[Highlight] = Field(default_factory=list)
    pages: dict[str, Page] = Field(default_factory=dict)
    selected_urls: list[str] = Field(default_factory=list)
    rejected_urls: list[str] = Field(default_factory=list)
    proposed_count: int = 0
    ground_rate: float = 1.0
    dropped: list[dict] = Field(default_factory=list)
    error: str | None = None


class ComparisonRow(BaseModel):
    """One row of the final report, referencing grounded evidence by index."""

    aspect: str
    a_text: str | None = None
    a_source: str | None = None
    b_text: str | None = None
    b_source: str | None = None


class ComparisonResult(BaseModel):
    """The full output of comparison.pipeline.compare."""

    product1_name: str
    product2_name: str
    rows: list[ComparisonRow] = Field(default_factory=list)
    product1_ground_rate: float = 1.0
    product2_ground_rate: float = 1.0
    product1_dropped: int = 0
    product2_dropped: int = 0
    product1_error: str | None = None
    product2_error: str | None = None
    trace: dict | None = None
