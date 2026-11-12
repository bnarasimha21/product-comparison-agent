import ipaddress
import socket
from unittest.mock import patch

import comparison.fetch as fetch_module
from comparison.fetch import fetch_page


def test_http_scheme_refused_by_default(local_server):
    url = local_server.add_page("/p.html", "<html><body>hi</body></html>")
    page = fetch_page(url)  # no allow_http override
    assert not page.ok
    assert "scheme" in page.error


def test_file_scheme_refused():
    page = fetch_page("file:///etc/passwd")
    assert not page.ok
    assert "scheme" in page.error


def test_loopback_address_refused_even_with_http_allowed(local_server):
    url = local_server.add_page("/p.html", "<html><body>hi</body></html>")
    page = fetch_page(url, allow_http=True)  # allow_http, but NOT allow_private
    assert not page.ok
    assert "private" in page.error


def test_fixture_server_fetchable_with_explicit_overrides(local_server):
    url = local_server.add_page("/p.html", "<html><body><p>Hello world</p></body></html>")
    page = fetch_page(url, allow_http=True, allow_private=True)
    assert page.ok
    assert "Hello world" in page.text


def test_redirect_to_private_address_refused(local_server):
    target = local_server.add_page("/final.html", "<html><body>secret</body></html>")
    start = local_server.add_redirect("/start.html", target)
    # allow_http so the *first* hop is reachable, but private addresses are
    # still rejected at every hop, including the redirect target.
    page = fetch_page(start, allow_http=True)
    assert not page.ok
    assert "private" in page.error


def test_redirect_followed_when_private_allowed(local_server):
    target = local_server.add_page("/final.html", "<html><body>final content</body></html>")
    start = local_server.add_redirect("/start.html", target)
    page = fetch_page(start, allow_http=True, allow_private=True)
    assert page.ok
    assert "final content" in page.text


def test_too_many_redirects_refused(local_server):
    local_server.add_redirect("/r0.html", local_server.base_url + "/r1.html")
    local_server.add_redirect("/r1.html", local_server.base_url + "/r2.html")
    local_server.add_redirect("/r2.html", local_server.base_url + "/r3.html")
    local_server.add_redirect("/r3.html", local_server.base_url + "/r4.html")
    local_server.add_redirect("/r4.html", local_server.base_url + "/r5.html")
    local_server.add_redirect("/r5.html", local_server.base_url + "/r6.html")
    page = fetch_page(
        local_server.base_url + "/r0.html", allow_http=True, allow_private=True, max_redirects=3
    )
    assert not page.ok
    assert "redirect" in page.error


def test_http_status_failure(local_server):
    url = local_server.add_status("/missing.html", 500)
    page = fetch_page(url, allow_http=True, allow_private=True)
    assert not page.ok
    assert "500" in page.error


def test_size_cap_enforced(local_server):
    big_html = "<html><body>" + ("x" * 5000) + "</body></html>"
    url = local_server.add_page("/big.html", big_html)
    page = fetch_page(url, allow_http=True, allow_private=True, max_bytes=100)
    assert not page.ok
    assert "size" in page.error


def test_hidden_text_is_dropped(local_server):
    html = (
        "<html><body>"
        "<p>Visible claim: price is $10.</p>"
        "<p style='display:none'>Hidden claim: price is actually $999999.</p>"
        "<p hidden>Also hidden text.</p>"
        "</body></html>"
    )
    url = local_server.add_page("/hidden.html", html)
    page = fetch_page(url, allow_http=True, allow_private=True)
    assert page.ok
    assert "Visible claim" in page.text
    assert "999999" not in page.text
    assert "Also hidden text" not in page.text


def test_script_and_style_dropped(local_server):
    html = (
        "<html><head><style>body{color:red}</style></head><body>"
        "<script>alert('xss')</script>"
        "<p>Real content here.</p>"
        "</body></html>"
    )
    url = local_server.add_page("/script.html", html)
    page = fetch_page(url, allow_http=True, allow_private=True)
    assert page.ok
    assert "alert" not in page.text
    assert "color:red" not in page.text
    assert "Real content here" in page.text


class _FakeHTTPResponse:
    """Minimal stand-in for a urllib3 HTTPResponse, for pool-level mocking."""

    def __init__(self, status: int = 200, headers: dict | None = None, body: bytes = b"<html><body>ok</body></html>"):
        self.status = status
        self.headers = headers or {}
        self._body = body

    def stream(self, chunk_size: int = 8192):
        yield self._body

    def release_conn(self) -> None:
        return None


class _FakePool:
    def __init__(self, response: _FakeHTTPResponse):
        self._response = response
        self.urlopen_calls: list[tuple] = []

    def urlopen(self, method, path, **kwargs):
        self.urlopen_calls.append((method, path, kwargs))
        return self._response

    def close(self) -> None:
        return None


def _fake_getaddrinfo_returning(*ip_strs: str):
    def fake(host, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0)) for ip in ip_strs]

    return fake


def test_multi_record_dns_one_private_rejected():
    """A hostname with two A records, one public and one private, must be
    rejected outright, not accepted because one of its addresses happens to
    be public (the bug: checking only whether ANY resolved address is
    public, instead of requiring ALL of them to be)."""
    with patch.object(
        fetch_module.socket,
        "getaddrinfo",
        side_effect=_fake_getaddrinfo_returning("127.0.0.1", "93.184.216.34"),
    ):
        page = fetch_page("https://multi-record.example/")
    assert not page.ok
    assert "private" in page.error


def test_shared_address_space_100_64_rejected():
    """100.64.0.0/10 (Tailscale's range, also used by some cloud metadata
    endpoints, e.g. Alibaba Cloud's 100.100.100.200) must be rejected.
    ipaddress.ip_address(...).is_private returns False for this range, so a
    private-ness check based on is_private alone would wrongly let it
    through; is_global correctly rejects it."""
    addr = ipaddress.ip_address("100.64.0.5")
    assert not addr.is_private  # documents exactly why is_private is not enough
    assert not addr.is_global

    with patch.object(
        fetch_module.socket,
        "getaddrinfo",
        side_effect=_fake_getaddrinfo_returning("100.64.0.5"),
    ):
        page = fetch_page("https://metadata-ish.example/")
    assert not page.ok
    assert "private" in page.error


def test_dns_rebinding_cannot_bypass_the_check():
    """Simulates DNS rebinding: a hostname that resolves to a PUBLIC address
    the first time it is looked up (check time) and would resolve to a
    PRIVATE address if looked up again (connect time). The fix is to
    resolve exactly once and connect directly to that one validated
    address, so a second lookup never happens and can never be raced."""
    public_ip = "93.184.216.34"
    private_ip = "127.0.0.1"
    getaddrinfo_mock = _fake_getaddrinfo_returning(public_ip)
    getaddrinfo_calls: list[str] = []

    def counting_getaddrinfo(host, *args, **kwargs):
        getaddrinfo_calls.append(host)
        # If this were called a second time (a real rebinding attack would
        # answer differently), returning the private address here proves
        # whether fetch_page actually re-resolves.
        if len(getaddrinfo_calls) == 1:
            return getaddrinfo_mock(host, *args, **kwargs)
        return _fake_getaddrinfo_returning(private_ip)(host, *args, **kwargs)

    captured_pool_args: list[tuple] = []
    fake_pool = _FakePool(_FakeHTTPResponse())

    def fake_build_pool(scheme, ip, port, hostname, timeout):
        captured_pool_args.append((scheme, ip, port, hostname, timeout))
        return fake_pool

    with patch.object(fetch_module.socket, "getaddrinfo", side_effect=counting_getaddrinfo), patch.object(
        fetch_module, "_build_pool", side_effect=fake_build_pool
    ):
        page = fetch_page("https://rebinding.example/path")

    assert len(getaddrinfo_calls) == 1, "hostname must be resolved exactly once per hop"
    assert captured_pool_args, "fetch_page must connect through a pinned pool"
    used_ip = captured_pool_args[0][1]
    assert used_ip == public_ip, "must connect to the address that was actually validated"
    assert page.ok
