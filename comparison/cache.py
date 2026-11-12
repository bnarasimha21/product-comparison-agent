"""A simple disk cache for search results, fetched pages, and final results.

One JSON file per key under a directory, keyed by a sha256 of a caller-built
string key so the filesystem never sees raw URLs/queries. TTL defaults to 24
hours; an expired or missing entry behaves like a miss.

No third-party cache library is used; the access pattern here (get, set,
get-or-compute, by string key, JSON-serializable values) does not need one.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

DEFAULT_TTL_SECONDS = 24 * 60 * 60


class DiskCache:
    def __init__(self, directory: str | Path, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.directory / f"{digest}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists():
            self.misses += 1
            return None
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            self.misses += 1
            return None
        if time.time() - payload.get("stored_at", 0) > self.ttl_seconds:
            self.misses += 1
            return None
        self.hits += 1
        return payload["value"]

    def set(self, key: str, value: Any) -> None:
        path = self._path(key)
        payload = {"stored_at": time.time(), "value": value}
        path.write_text(json.dumps(payload))

    def get_or_set(self, key: str, compute: Callable[[], Any]) -> Any:
        cached = self.get(key)
        if cached is not None:
            return cached
        value = compute()
        self.set(key, value)
        return value

    def clear(self) -> None:
        for path in self.directory.glob("*.json"):
            path.unlink(missing_ok=True)
        self.hits = 0
        self.misses = 0


class CachingSearchProvider:
    """Wraps a SearchProvider; identical (query, n) pairs hit the cache."""

    def __init__(self, inner: Any, cache: DiskCache) -> None:
        self._inner = inner
        self._cache = cache

    def search(self, query: str, n: int) -> list:
        from .models import SearchResult

        key = f"search:{query}:{n}"
        cached = self._cache.get(key)
        if cached is not None:
            return [SearchResult(**item) for item in cached]
        results = self._inner.search(query, n)
        self._cache.set(key, [r.model_dump() for r in results])
        return results


def caching_fetch(cache: DiskCache, fetch_fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wraps a fetch_page-shaped callable; identical URLs hit the cache."""
    from .models import Page

    def wrapped(url: str, **kwargs: Any) -> Any:
        key = f"fetch:{url}"
        cached = cache.get(key)
        if cached is not None:
            return Page(**cached)
        page = fetch_fn(url, **kwargs)
        cache.set(key, page.model_dump())
        return page

    return wrapped


def result_cache_key(request: Any) -> str:
    """A stable key for a final ComparisonResult, from the normalized request.

    Deliberately order-sensitive: (A, B) and (B, A) are different cache
    entries, not the same one. An order-insensitive key would make the
    order-swap consistency eval (issue 5) read a cached (A, B) result back
    out for a (B, A) request, which would hide exactly the inconsistency
    that eval exists to catch.
    """
    from .models import CompareRequest

    assert isinstance(request, CompareRequest)
    product1 = request.product1.strip().casefold()
    product2 = request.product2.strip().casefold()
    return f"result:{product1}|{product2}|{request.num_urls}|{request.num_highlights}"
