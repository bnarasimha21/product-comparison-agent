"""Deterministic page fetching.

Replaces the "Web Scraper" LLM agent from the original product_scraper.py
(which wrapped ScrapeWebsiteTool(urls="{urls}") -- a constructor argument the
tool does not have, see tests/test_search_fetch_integration.py). Fetching is
now plain code: https-only by default, a size cap, a timeout, a bounded
number of redirects, and rejection of any hop (initial URL or redirect
target) whose hostname resolves to anything other than an all-public set of
addresses.

allow_http and allow_private exist only so tests can point this function at
a local fixture HTTP server. Production call sites (comparison/flow.py)
never set them.

SSRF defense, and why it is shaped this way:

- Every resolved address for a hostname must be global (``ipaddress``'s
  ``is_global``), not just one. A hostname can have multiple A/AAAA
  records, and nothing stops one of them from being a private address a
  misbehaving or malicious DNS answer could later hand back as *the*
  address actually used; accepting a hostname because *any* resolved
  address was public would leave that door open.
- ``is_global`` is used instead of ``is_private`` because ``is_private``
  does not cover the shared address space ``100.64.0.0/10``. That range is
  used by Tailscale, by CGNAT, and by some cloud providers for metadata
  endpoints (Alibaba Cloud's ``100.100.100.200`` is in it) -- exactly the
  kind of address an SSRF would want to reach. ``is_global`` rejects it
  along with everything ``is_private`` already covered.
- The hostname is resolved exactly ONCE per hop, in ``_check_hop``, and the
  validated IP address it returns is the literal address the connection is
  opened to (an IP-pinned ``urllib3`` connection pool, with the real
  hostname kept as the TLS SNI name / cert-verification name and as the
  HTTP ``Host`` header). There is deliberately no second DNS lookup for an
  attacker controlling the DNS record to race: the address checked is the
  address connected to, for the initial URL and for every redirect hop.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import certifi
import urllib3
from bs4 import BeautifulSoup
from requests.utils import get_encoding_from_headers

from .models import Page

DEFAULT_TIMEOUT = 15
DEFAULT_MAX_BYTES = 2_000_000
DEFAULT_MAX_REDIRECTS = 5
DEFAULT_MAX_EXCERPT_CHARS = 20_000
_USER_AGENT = "product-comparison-agent/1.0 (+https://github.com/bnarasimha21/product-comparison-agent)"
_DEFAULT_PORTS = {"http": 80, "https": 443}


class _Rejected(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _resolve_addresses(hostname: str) -> list[str]:
    """Resolve hostname to every distinct IP address it has, exactly once.

    The caller must connect directly to one of these addresses and must
    never call getaddrinfo on this hostname again for this hop: a second,
    later lookup is exactly what a DNS-rebinding attack needs (answer
    public at check time, private at connect time).
    """
    try:
        infos = socket.getaddrinfo(hostname, None)
    except OSError:
        return []
    addrs: list[str] = []
    seen: set[str] = set()
    for info in infos:
        ip_str = info[4][0]
        if ip_str not in seen:
            seen.add(ip_str)
            addrs.append(ip_str)
    return addrs


def _all_global(addrs: list[str]) -> bool:
    """True iff addrs is non-empty and every address in it is global.

    See the module docstring for why this checks EVERY address (multi-record
    DNS) and uses is_global rather than is_private (100.64.0.0/10).
    """
    if not addrs:
        return False
    for ip_str in addrs:
        ip_str = ip_str.split("%", 1)[0]  # strip an IPv6 zone id, if present
        try:
            addr = ipaddress.ip_address(ip_str)
        except ValueError:
            return False
        if not addr.is_global:
            return False
    return True


def _check_hop(url: str, *, allow_http: bool, allow_private: bool) -> tuple[str, str, int, str]:
    """Validate one hop and return (pinned_ip, hostname, port, path_and_query).

    Resolution happens exactly once, here, and the returned IP is the exact
    address the caller must connect to.
    """
    # urlparse() itself can raise ValueError on a malformed URL (e.g. an
    # unterminated IPv6 literal like "https://[::1/"), and parsed.port
    # raises ValueError on an out-of-range port (e.g. ":99999"). Both must
    # become an ordinary _Rejected failure -- the same kind of structured
    # result every other failure mode in this function already returns --
    # not an unhandled exception that escapes fetch_page.
    try:
        parsed = urlparse(url)
        port = parsed.port
        hostname = parsed.hostname
    except ValueError as exc:
        raise _Rejected(f"malformed URL: {exc}") from exc

    allowed_schemes = {"https"} | ({"http"} if allow_http else set())
    if parsed.scheme not in allowed_schemes:
        raise _Rejected(f"scheme not allowed: {parsed.scheme!r}")
    if not hostname:
        raise _Rejected("no host in URL")

    addrs = _resolve_addresses(hostname)
    if not addrs:
        raise _Rejected("private or unresolvable address rejected")
    if not allow_private and not _all_global(addrs):
        raise _Rejected("private or unresolvable address rejected")

    port = port or _DEFAULT_PORTS[parsed.scheme]
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    return addrs[0], hostname, port, path


def _host_header(hostname: str, port: int, scheme: str) -> str:
    if port == _DEFAULT_PORTS.get(scheme):
        return hostname
    return f"{hostname}:{port}"


def _build_pool(scheme: str, ip: str, port: int, hostname: str, timeout: int) -> urllib3.HTTPConnectionPool:
    """Build a connection pool pinned to ip, the specific address _check_hop
    already validated. For https, TLS SNI and certificate verification still
    use the real hostname (server_hostname / assert_hostname), so a page is
    still only accepted if its certificate is valid for the hostname the
    user asked for -- only the socket-level connect target is pinned.
    """
    if scheme == "https":
        return urllib3.HTTPSConnectionPool(
            ip,
            port=port,
            timeout=timeout,
            maxsize=1,
            retries=False,
            server_hostname=hostname,
            assert_hostname=hostname,
            ca_certs=certifi.where(),
        )
    return urllib3.HTTPConnectionPool(ip, port=port, timeout=timeout, maxsize=1, retries=False)


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
    for tag in list(soup(["script", "style"])):
        tag.decompose()

    # Collect every element to remove BEFORE removing any of them. Calling
    # .decompose() on a tag while still iterating the very find_all(True)
    # result that produced it is what crashes: decompose() clears
    # __dict__ on every descendant too (see bs4's Tag.decompose), so once a
    # hidden parent is decomposed, a nested child tag still waiting in this
    # loop's results list has had its own .attrs wiped out from under it --
    # the next .get(...)/.has_attr(...) call on that child then blows up
    # with AttributeError, because the attribute the method needs no
    # longer exists on the object at all. Separating "find what to remove"
    # from "remove it" means every .get()/.has_attr() call below always
    # runs against a still-intact tree.
    to_remove = []
    for tag in soup.find_all(True):
        style = (tag.get("style") or "").lower()
        if "display:none" in style.replace(" ", "") or "visibility:hidden" in style.replace(" ", ""):
            to_remove.append(tag)
            continue
        if tag.has_attr("hidden"):
            to_remove.append(tag)

    for tag in to_remove:
        # A tag whose own ancestor was already decomposed earlier in this
        # same pass (e.g. <div hidden><span hidden>...) has already been
        # removed from the tree; skip it rather than decompose a tag with
        # no parent left to detach from.
        if tag.parent is not None:
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
) -> Page:
    """Fetch url and return readable text, or a structured failure.

    Never raises for an ordinary failure (bad scheme, private/non-global
    address, timeout, non-2xx, oversize body, too many redirects); those all
    come back as Page(ok=False, error=...).
    """
    current = url
    for _hop in range(max_redirects + 1):
        try:
            ip, hostname, port, path = _check_hop(current, allow_http=allow_http, allow_private=allow_private)
        except _Rejected as exc:
            return Page(url=url, text="", ok=False, error=exc.reason)

        scheme = urlparse(current).scheme
        pool = _build_pool(scheme, ip, port, hostname, timeout)
        try:
            try:
                response = pool.urlopen(
                    "GET",
                    path,
                    headers={
                        "Host": _host_header(hostname, port, scheme),
                        "User-Agent": _USER_AGENT,
                    },
                    timeout=timeout,
                    redirect=False,
                    retries=False,
                    preload_content=False,
                )
            except Exception as exc:  # noqa: BLE001 - any connect/TLS/timeout failure is an ordinary fetch failure
                return Page(url=url, text="", ok=False, error=f"request failed: {exc}")

            try:
                if response.status in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location")
                    if not location:
                        return Page(url=url, text="", ok=False, error="redirect with no Location header")
                    current = urljoin(current, location)
                    continue

                if response.status != 200:
                    return Page(url=url, text="", ok=False, error=f"HTTP {response.status}")

                body = bytearray()
                try:
                    for chunk in response.stream(8192):
                        body.extend(chunk)
                        if len(body) > max_bytes:
                            return Page(url=url, text="", ok=False, error="response exceeded size cap")
                except Exception as exc:  # noqa: BLE001 - a mid-read failure is an ordinary fetch failure
                    return Page(url=url, text="", ok=False, error=f"request failed: {exc}")

                encoding = get_encoding_from_headers(response.headers) or "utf-8"
                try:
                    html = body.decode(encoding, errors="replace")
                except (LookupError, TypeError):
                    html = body.decode("utf-8", errors="replace")

                text = _extract_text(html)[:max_excerpt_chars]
                return Page(url=url, text=text, ok=True, error=None)
            finally:
                response.release_conn()
        finally:
            pool.close()

    return Page(url=url, text="", ok=False, error="too many redirects")
