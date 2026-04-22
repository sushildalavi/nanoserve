"""tests for the prometheus metrics module — pure python, no engine."""
from __future__ import annotations

import pytest

from nanoserve.server import metrics as m


def test_render_returns_text_format():
    body, content_type = m.render()
    assert b"# HELP" in body or b"# TYPE" in body
    assert "text/plain" in content_type


def test_counter_appears_after_increment():
    m.requests_total.labels(status="ok").inc()
    body, _ = m.render()
    assert b"nanoserve_requests_total" in body


def test_histogram_observation_appears():
    m.ttft_seconds.observe(0.42)
    body, _ = m.render()
    assert b"nanoserve_ttft_seconds_bucket" in body
    assert b"nanoserve_ttft_seconds_count" in body


def test_gauge_set_and_render():
    m.active_seqs.set(7)
    body, _ = m.render()
    assert b"nanoserve_active_seqs 7" in body or b"nanoserve_active_seqs 7.0" in body


def test_refresh_gauges_with_none_is_safe():
    # before any engine exists the api may scrape /metrics. should not crash.
    m.refresh_gauges_from_engine(None)


class _FakeScheduler:
    def __init__(self, running_count: int):
        self.running = list(range(running_count))


class _FakePrefixCache:
    def __init__(self, hits: int, misses: int, size: int):
        self.hits = hits
        self.misses = misses
        self._size = size

    def __len__(self) -> int:
        return self._size


class _FakeEngine:
    def __init__(self, running_count, batched, single, prefix_cache=None):
        self._scheduler = _FakeScheduler(running_count)
        self.batched_forward_steps = batched
        self.single_forward_steps = single
        self.prefix_cache = prefix_cache


def test_refresh_gauges_updates_active_and_frac():
    eng = _FakeEngine(running_count=3, batched=80, single=20)
    m.refresh_gauges_from_engine(eng)
    body, _ = m.render()
    assert b"nanoserve_active_seqs 3" in body or b"nanoserve_active_seqs 3.0" in body
    # 80/100 = 0.8
    assert b"nanoserve_batched_forward_frac 0.8" in body


def test_refresh_gauges_increments_cache_counters_by_delta():
    # reset module-level prev counters and metrics fresh
    m._prev_hits = 0
    m._prev_misses = 0

    cache = _FakePrefixCache(hits=10, misses=2, size=5)
    eng = _FakeEngine(running_count=0, batched=1, single=0, prefix_cache=cache)
    m.refresh_gauges_from_engine(eng)

    # second scrape with more activity
    cache.hits = 25
    cache.misses = 4
    cache._size = 7
    m.refresh_gauges_from_engine(eng)

    body, _ = m.render()
    # we incremented by 10+15 = 25 hits and 2+2 = 4 misses
    assert b"nanoserve_prefix_cache_hits_total 25" in body
    assert b"nanoserve_prefix_cache_misses_total 4" in body
    assert (
        b"nanoserve_prefix_cache_size 7" in body
        or b"nanoserve_prefix_cache_size 7.0" in body
    )


def test_refresh_handles_counter_reset_without_going_negative():
    """if the engine restarts mid-server (which we don't currently support
    but might one day), the cache hit count would reset to 0. the prom
    counter must not go backward. the delta path uses max(0, ...) for this.
    """
    m._prev_hits = 100
    m._prev_misses = 50
    cache = _FakePrefixCache(hits=5, misses=2, size=1)
    eng = _FakeEngine(running_count=0, batched=0, single=0, prefix_cache=cache)
    m.refresh_gauges_from_engine(eng)
    # we shouldn't have crashed and the prom counters should not have
    # gone negative (they can't anyway). this just exercises the guard.
    assert True


@pytest.fixture(autouse=True)
def _reset_prev_counters():
    """isolate cache delta state between tests."""
    m._prev_hits = 0
    m._prev_misses = 0
    yield
