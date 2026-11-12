import time

from comparison.cache import CachingSearchProvider, DiskCache, caching_fetch, result_cache_key
from comparison.models import CompareRequest, Page, SearchResult
from comparison.search import FixtureSearch


def test_get_miss_then_set_then_hit(tmp_path):
    cache = DiskCache(tmp_path)
    assert cache.get("k") is None
    assert cache.misses == 1
    cache.set("k", {"a": 1})
    assert cache.get("k") == {"a": 1}
    assert cache.hits == 1


def test_ttl_expiry(tmp_path):
    cache = DiskCache(tmp_path, ttl_seconds=0.05)
    cache.set("k", "value")
    assert cache.get("k") == "value"
    time.sleep(0.1)
    assert cache.get("k") is None


def test_get_or_set_computes_once(tmp_path):
    cache = DiskCache(tmp_path)
    calls = []

    def compute():
        calls.append(1)
        return "computed"

    assert cache.get_or_set("k", compute) == "computed"
    assert cache.get_or_set("k", compute) == "computed"
    assert len(calls) == 1


def test_clear_resets_counts_and_files(tmp_path):
    cache = DiskCache(tmp_path)
    cache.set("k", "v")
    cache.get("k")
    cache.clear()
    assert cache.hits == 0
    assert cache.misses == 0
    assert cache.get("k") is None


def test_caching_search_provider_hits_cache_on_repeat_query(tmp_path):
    cache = DiskCache(tmp_path)
    calls = []

    class CountingProvider:
        def search(self, query, n):
            calls.append((query, n))
            return [SearchResult(title="t", url="https://a.example/", snippet="s")]

    provider = CachingSearchProvider(CountingProvider(), cache)
    first = provider.search("widget", 3)
    second = provider.search("widget", 3)
    assert first == second
    assert len(calls) == 1


def test_caching_fetch_hits_cache_on_repeat_url(tmp_path):
    cache = DiskCache(tmp_path)
    calls = []

    def fetch_fn(url, **kwargs):
        calls.append(url)
        return Page(url=url, text="hello", ok=True)

    wrapped = caching_fetch(cache, fetch_fn)
    first = wrapped("https://a.example/")
    second = wrapped("https://a.example/")
    assert first == second
    assert len(calls) == 1


def test_result_cache_key_is_order_sensitive():
    req_ab = CompareRequest(product1="A", product2="B", num_urls=2, num_highlights=2)
    req_ba = CompareRequest(product1="B", product2="A", num_urls=2, num_highlights=2)
    assert result_cache_key(req_ab) != result_cache_key(req_ba)


def test_result_cache_key_stable_for_same_request():
    req1 = CompareRequest(product1="Widget", product2="Gadget", num_urls=2, num_highlights=2)
    req2 = CompareRequest(product1="widget", product2="gadget", num_urls=2, num_highlights=2)
    assert result_cache_key(req1) == result_cache_key(req2)


def test_fixture_search_unaffected_by_caching_layer_contract(tmp_path):
    # Sanity check that FixtureSearch (used across the test suite) satisfies
    # the same interface the caching wrapper expects.
    provider = FixtureSearch({"q": [SearchResult(title="t", url="https://a.example/", snippet="")]})
    cache = DiskCache(tmp_path)
    wrapped = CachingSearchProvider(provider, cache)
    results = wrapped.search("q", 1)
    assert results[0].url == "https://a.example/"
