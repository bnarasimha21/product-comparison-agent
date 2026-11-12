"""The typed-state CrewAI Flow that drives one comparison run.

Replaces the untyped `self.state` dict in the original main.py
(ProductComparisonFlow read `self.state["product1"]` etc, and stages
unpacked results by list position and string key). Agents/Crews are built
fresh per call (comparison/agents.py), never shared at import time.

The two products are researched concurrently inside research_products via
asyncio.gather over kickoff_async, keeping each product's own stage order
(search -> select URLs -> fetch -> extract) intact; the comparison stage
runs once, after both reports are in.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel

from . import agents as agent_factory
from .fetch import fetch_page
from .grounding import ground_highlights, ground_rate
from .models import (
    CompareRequest,
    ComparisonResult,
    ComparisonRow,
    Highlight,
    Page,
    ProductReport,
    SearchResult,
)
from .search import SearchProvider, query_templates

ProgressCallback = Callable[[float, str], None]
FetchFn = Callable[..., Page]


def _noop_progress(fraction: float, message: str) -> None:
    return None


class CompareState(BaseModel):
    request: CompareRequest | None = None
    product1_report: ProductReport | None = None
    product2_report: ProductReport | None = None
    result: ComparisonResult | None = None


class ProductComparisonFlow(Flow[CompareState]):
    """One comparison run.

    Dependencies are set as plain attributes after construction (the same
    pattern the original main.py used for `flow.progress`); crewai's Flow
    permits attribute assignment outside the declared state model.
    """

    def __init__(
        self,
        *,
        llm: Any = None,
        search_provider: SearchProvider | None = None,
        fetch_fn: FetchFn = fetch_page,
        progress: ProgressCallback | None = None,
        trace: dict | None = None,
        stage_timeout_seconds: float = 120.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.llm = llm
        self.search_provider = search_provider
        self.fetch_fn = fetch_fn
        self.progress = progress or _noop_progress
        self.stage_timeout_seconds = stage_timeout_seconds
        # issue 6: per-stage timings and counters, filled in as the run
        # proceeds; comparison/trace.py turns this into a Trace object.
        self.trace = trace if trace is not None else {"stages": [], "llm_calls": 0}

    def _record_stage(self, name: str, started_at: float, **extra: Any) -> None:
        self.trace["stages"].append(
            {"name": name, "seconds": time.monotonic() - started_at, **extra}
        )

    @start()
    async def research_products(self) -> tuple[ProductReport, ProductReport]:
        request = self.state.request
        if request is None:
            raise ValueError("CompareState.request must be set before kickoff")
        if self.search_provider is None:
            raise ValueError("ProductComparisonFlow.search_provider must be set before kickoff")

        self.progress(0.0, "Starting research for both products...")
        gathered = await asyncio.gather(
            self._research_one_with_timeout(request.product1, request.num_urls, request.num_highlights),
            self._research_one_with_timeout(request.product2, request.num_urls, request.num_highlights),
            return_exceptions=True,
        )
        report1 = self._report_or_errored(gathered[0], request.product1)
        report2 = self._report_or_errored(gathered[1], request.product2)
        self.state.product1_report = report1
        self.state.product2_report = report2
        self.progress(0.7, "Research complete for both products.")
        return report1, report2

    @staticmethod
    def _report_or_errored(outcome: ProductReport | BaseException, product_name: str) -> ProductReport:
        """Turn a gather() outcome into a ProductReport, never an exception.

        asyncio.gather(..., return_exceptions=True) hands back whatever
        exception a product's pipeline raised instead of raising it itself,
        so one product's unexpected error (a bug, a bad response shape, a
        library exception that is not a timeout) can be turned into *that
        product's own* partial/errored report, degrading gracefully instead
        of aborting the other, unrelated product's already-successful
        research (issue 6's goal) via an unhandled exception.
        """
        if isinstance(outcome, BaseException):
            return ProductReport(name=product_name, error=f"research stage failed: {outcome}")
        return outcome

    async def _research_one_with_timeout(
        self, product_name: str, num_urls: int, num_highlights: int
    ) -> ProductReport:
        """Run _research_one, degrading to a partial (errored) report
        rather than letting one slow/stuck product take the whole run down.
        """
        try:
            return await asyncio.wait_for(
                self._research_one(product_name, num_urls, num_highlights),
                timeout=self.stage_timeout_seconds,
            )
        except TimeoutError:
            return ProductReport(
                name=product_name,
                error=f"research stage timed out after {self.stage_timeout_seconds:.0f}s",
            )

    async def _research_one(
        self, product_name: str, num_urls: int, num_highlights: int
    ) -> ProductReport:
        started = time.monotonic()
        results: list[SearchResult] = []
        for query in query_templates(product_name):
            # A real SearchProvider (SerperSearch) makes a blocking HTTP
            # call. Calling it directly inside this coroutine blocks the
            # whole event loop for its duration, serializing the two
            # products' searches even though research_products gathers
            # them concurrently. asyncio.to_thread moves it off the loop,
            # the same way the fetch stage below already does for
            # fetch_fn.
            page_results = await asyncio.to_thread(
                self.search_provider.search, query, n=max(num_urls, 3)
            )
            results.extend(page_results)

        seen: set[str] = set()
        deduped: list[SearchResult] = []
        for result in results:
            if result.url not in seen:
                seen.add(result.url)
                deduped.append(result)
        results = deduped
        self._record_stage("search", started, product=product_name, result_count=len(results))

        if not results:
            return ProductReport(name=product_name, error="no search results found")

        select_started = time.monotonic()
        selected, rejected = await self._select_urls(product_name, results, num_urls)
        self._record_stage(
            "select", select_started, product=product_name, selected=len(selected), rejected=len(rejected)
        )

        if not selected:
            return ProductReport(
                name=product_name, selected_urls=[], rejected_urls=rejected, error="no valid URLs selected"
            )

        fetch_started = time.monotonic()
        pages_list = await asyncio.gather(*[asyncio.to_thread(self.fetch_fn, url) for url in selected])
        pages = {page.url: page for page in pages_list}
        usable_pages = {url: page for url, page in pages.items() if page.ok}
        self._record_stage(
            "fetch",
            fetch_started,
            product=product_name,
            fetched=len(pages),
            usable=len(usable_pages),
            failed=[url for url, page in pages.items() if not page.ok],
        )

        if not usable_pages:
            return ProductReport(
                name=product_name,
                pages=pages,
                selected_urls=selected,
                rejected_urls=rejected,
                error="no page could be fetched",
            )

        extract_started = time.monotonic()
        proposed: list[Highlight] = []
        for url, page in usable_pages.items():
            extracted = await self._extract_highlights(product_name, page, num_highlights)
            for item in extracted:
                proposed.append(Highlight(text=item.text, source_url=url, quote=item.quote))
        self._record_stage("extract", extract_started, product=product_name, proposed=len(proposed))

        grounded, dropped = ground_highlights(proposed, pages)
        grounded = grounded[:num_highlights]

        return ProductReport(
            name=product_name,
            highlights=grounded,
            pages=pages,
            selected_urls=selected,
            rejected_urls=rejected,
            proposed_count=len(proposed),
            ground_rate=ground_rate(len(proposed), len(grounded)),
            dropped=dropped,
        )

    async def _select_urls(
        self, product_name: str, results: list[SearchResult], num_urls: int
    ) -> tuple[list[str], list[str]]:
        candidate_urls = {result.url for result in results}
        candidates_text = agent_factory.format_candidates(results)
        rejected: list[str] = []

        for _attempt in range(2):
            crew = agent_factory.build_url_selection_crew(self.llm, product_name)
            output = await crew.kickoff_async(
                inputs={
                    "product_name": product_name,
                    "candidates": candidates_text,
                    "num_urls": num_urls,
                }
            )
            self.trace["llm_calls"] += 1
            selection = output.pydantic
            urls = selection.urls if selection else []
            valid = [url for url in urls if url in candidate_urls]
            invalid = [url for url in urls if url not in candidate_urls]
            rejected.extend(invalid)
            if valid or not invalid:
                return valid[:num_urls], rejected
            candidates_text += (
                f"\n\nNote: you previously invented URLs not in this list: {invalid}. "
                "Only choose URLs copied exactly, character for character, from the list above."
            )

        return [], rejected

    async def _extract_highlights(
        self, product_name: str, page: Page, num_highlights: int
    ) -> list:
        crew = agent_factory.build_highlight_extraction_crew(self.llm, product_name, num_highlights)
        output = await crew.kickoff_async(
            inputs={
                "product_name": product_name,
                "page_url": page.url,
                # Neutralize a literal "</page>"/"<page ...>" inside the
                # page's own visible text so it cannot close this prompt's
                # <page> delimiter early (see agents.escape_page_delimiters).
                "page_text": agent_factory.escape_page_delimiters(page.text),
            }
        )
        self.trace["llm_calls"] += 1
        extracted = output.pydantic
        return extracted.highlights if extracted else []

    @listen(research_products)
    async def compare_products(self, reports: tuple[ProductReport, ProductReport]) -> ComparisonResult:
        request = self.state.request
        assert request is not None
        report1, report2 = reports

        self.progress(0.8, "Comparing products...")
        compare_started = time.monotonic()
        crew = agent_factory.build_comparison_crew(
            self.llm, report1.name, report2.name, request.num_highlights
        )
        timed_out = False
        draft = None
        try:
            output = await asyncio.wait_for(
                crew.kickoff_async(
                    inputs={
                        "product1_name": report1.name,
                        "product1_highlights": agent_factory.format_indexed_highlights(report1.highlights),
                        "product2_name": report2.name,
                        "product2_highlights": agent_factory.format_indexed_highlights(report2.highlights),
                        "num_highlights": request.num_highlights,
                    }
                ),
                timeout=self.stage_timeout_seconds,
            )
            self.trace["llm_calls"] += 1
            draft = output.pydantic
        except TimeoutError:
            timed_out = True

        rows: list[ComparisonRow] = []
        if draft:
            for row in draft.rows[: request.num_highlights]:
                a_text: str | None = "no evidence found"
                a_source: str | None = None
                b_text: str | None = "no evidence found"
                b_source: str | None = None

                if row.a_index is not None and 1 <= row.a_index <= len(report1.highlights):
                    highlight = report1.highlights[row.a_index - 1]
                    # Render the verified quote, never the model's separate
                    # `text` paraphrase: `text` is never checked against the
                    # source page (see comparison/grounding.py), so showing
                    # it would let an ungrounded claim reach the report even
                    # though its quote passed grounding.
                    a_text, a_source = highlight.quote, highlight.source_url
                if row.b_index is not None and 1 <= row.b_index <= len(report2.highlights):
                    highlight = report2.highlights[row.b_index - 1]
                    b_text, b_source = highlight.quote, highlight.source_url

                rows.append(
                    ComparisonRow(aspect=row.aspect, a_text=a_text, a_source=a_source, b_text=b_text, b_source=b_source)
                )

        self._record_stage("compare", compare_started, rows=len(rows), timed_out=timed_out)

        compare_note = "comparison stage timed out; showing research results without rows" if timed_out else None
        result = ComparisonResult(
            product1_name=report1.name,
            product2_name=report2.name,
            rows=rows,
            product1_ground_rate=report1.ground_rate,
            product2_ground_rate=report2.ground_rate,
            product1_dropped=len(report1.dropped),
            product2_dropped=len(report2.dropped),
            product1_error=report1.error or (compare_note if timed_out else None),
            product2_error=report2.error,
        )
        self.state.result = result
        self.progress(1.0, "Done.")
        return result
