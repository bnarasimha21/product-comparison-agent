"""Shared pytest fixtures: FakeLLM, a local HTTP fixture server, helpers.

Nothing here makes a real network call or needs an API key; see
comparison/fake_llm.py for the scripted LLM and comparison/search.py's
FixtureSearch for the scripted search provider.
"""

from __future__ import annotations

import http.server
import threading

import pytest

from comparison.fake_llm import FakeLLM  # re-exported for convenience

__all__ = ["FakeLLM", "local_server"]


class _FixtureHandler(http.server.BaseHTTPRequestHandler):
    pages: dict = {}
    redirects: dict = {}
    statuses: dict = {}

    def do_GET(self) -> None:
        if self.path in self.redirects:
            self.send_response(302)
            self.send_header("Location", self.redirects[self.path])
            self.end_headers()
            return
        if self.path in self.statuses:
            self.send_response(self.statuses[self.path])
            self.end_headers()
            return
        body = self.pages.get(self.path)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:  # silence test output
        pass


class LocalServer:
    def __init__(self) -> None:
        handler_cls = type("_Handler", (_FixtureHandler,), {"pages": {}, "redirects": {}, "statuses": {}})
        self._handler_cls = handler_cls
        self._httpd = http.server.HTTPServer(("127.0.0.1", 0), handler_cls)
        self.port = self._httpd.server_port
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def add_page(self, path: str, html: str) -> str:
        self._handler_cls.pages[path] = html.encode("utf-8")
        return self.base_url + path

    def add_redirect(self, path: str, target: str) -> str:
        self._handler_cls.redirects[path] = target
        return self.base_url + path

    def add_status(self, path: str, status: int) -> str:
        self._handler_cls.statuses[path] = status
        return self.base_url + path

    def shutdown(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


@pytest.fixture
def local_server():
    server = LocalServer()
    yield server
    server.shutdown()
