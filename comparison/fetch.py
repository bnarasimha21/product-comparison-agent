"""Deterministic page fetching.

Replaces the "Web Scraper" LLM agent from the original product_scraper.py
(which wrapped ScrapeWebsiteTool(urls="{urls}") -- a constructor argument the
tool does not have, see tests/test_search_fetch_integration.py). Fetching is
now plain code: https-only by default, a size cap, a timeout, a bounded
number of redirects, and rejection of any hop (initial URL or redirect
target) that resolves to a private, loopback, link-local, or otherwise
non-public address. This is the only layer that should ever see this app
make an outbound request to an attacker-influenced URL.

allow_http and allow_private exist only so tests can point this function at
a local fixture HTTP server. Production call sites (comparison/flow.py)
never set them.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .models import Page

DEFAULT_TIMEOUT = 15
DEFAULT_MAX_BYTES = 2_000_000
DEFAULT_MAX_REDIRECTS = 5
DEFAULT_MAX_EXCERPT_CHARS = 20_000
_USER_AGENT = "product-comparison-agent/1.0 (+https://github.com/bnarasimha21/product-comparison-agent)"


class _Rejected(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _is_private_host(host: str) -> bool:
    """True if host has no address, or every address it resolves to is
    private/loopback/link-local/reserved/multicast."""
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return True
    if not infos:
        return True
    for info in infos:
        sockaddr = info[4]
        ip_str = sockaddr[0]
        try:
            addr = ipaddress.ip_address(ip_str)
        except ValueError:
            return True
        if not (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_reserved
            or addr.is_multicast
            or addr.is_unspecified
        ):
            return False
    return True


def _check_hop(url: str, *, allow_http: bool, allow_private: bool) -> None:
    parsed = urlparse(url)
    allowed_schemes = {"https"} | ({"http"} if allow_http else set())
    if parsed.scheme not in allowed_schemes:
        raise _Rejected(f"scheme not allowed: {parsed.scheme!r}")
    if not parsed.hostname:
        raise _Rejected("no host in URL")
    if not allow_private and _is_private_host(parsed.hostname):
        raise _Rejected("private or unresolvable address rejected")


def _extract_text(html: str) -> str:
    """Readable text, with hidden elements dropped.

    Elements with display:none, visibility:hidden, or the `hidden`
    attribute are removed before extracting text, so content a page author
    hid from a human reader cannot still reach the model as "page text".
    This is a deliberate, documented choice (see README) rather than an
    oversight: keeping hidden text would let a page smuggle claims a
    reader never saw.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for tag in soup.find_all(True):
        style = (tag.get("style") or "").lower()
        if "display:none" in style.replace(" ", "") or "visibility:hidden" in style.replace(" ", ""):
            tag.decompose()
            continue
        if tag.has_attr("hidden"):
            tag.decompose()
    text = soup.get_text(separator=" ")
    return " ".join(text.split())


def fetch_page(
    url: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    max_excerpt_chars: int = DEFAULT_MAX_EXCERPT_CHARS,
    allow_http: bool = False,
    allow_private: bool = False,
    session: requests.Session | None = None,
) -> Page:
    """Fetch url and return readable text, or a structured failure.

    Never raises for an ordinary failure (bad scheme, private address,
    timeout, non-2xx, oversize body, too many redirects); those all come
    back as Page(ok=False, error=...).
    """
    owns_session = session is None
    session = session or requests.Session()
    current = url
    try:
        for _hop in range(max_redirects + 1):
            try:
                _check_hop(current, allow_http=allow_http, allow_private=allow_private)
            except _Rejected as exc:
                return Page(url=url, text="", ok=False, error=exc.reason)

            try:
                response = session.get(
                    current,
                    timeout=timeout,
                    allow_redirects=False,
                    stream=True,
                    headers={"User-Agent": _USER_AGENT},
                )
            except requests.RequestException as exc:
                return Page(url=url, text="", ok=False, error=f"request failed: {exc}")

            if response.is_redirect or response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("Location")
                response.close()
                if not location:
                    return Page(url=url, text="", ok=False, error="redirect with no Location header")
                current = urljoin(current, location)
                continue

            if response.status_code != 200:
                status = response.status_code
                response.close()
                return Page(url=url, text="", ok=False, error=f"HTTP {status}")

            body = bytearray()
            try:
                for chunk in response.iter_content(chunk_size=8192):
                    body.extend(chunk)
                    if len(body) > max_bytes:
                        response.close()
                        return Page(url=url, text="", ok=False, error="response exceeded size cap")
            finally:
                response.close()

            try:
                html = body.decode(response.encoding or "utf-8", errors="replace")
            except (LookupError, TypeError):
                html = body.decode("utf-8", errors="replace")

            text = _extract_text(html)[:max_excerpt_chars]
            return Page(url=url, text=text, ok=True, error=None)

        return Page(url=url, text="", ok=False, error="too many redirects")
    finally:
        if owns_session:
            session.close()
