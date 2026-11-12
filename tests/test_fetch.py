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
