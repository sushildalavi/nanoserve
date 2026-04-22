"""tests for the prefix cache.

these are pure-python tests — no model, no MPS. they verify the lookup,
LRU eviction, hit-rate accounting, and the "longest prefix wins" rule.
the L5-prefix correctness gate (cached vs from-scratch produce same
tokens) is a separate test that does need the model.
"""
from __future__ import annotations

import pytest

from nanoserve.engine.prefix_cache import PrefixCache


class _FakeCache:
    """stand-in for DynamicCache; deepcopy-able and equality-checkable."""

    def __init__(self, payload: int):
        self.payload = payload

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FakeCache) and self.payload == other.payload


# every test ID stream is long enough to clear MIN_LCP_FOR_HIT (=8). we use
# range(20)[:k] for matching prefixes so token overlap is unambiguous.
LONG = list(range(20))


def test_empty_cache_returns_none():
    c = PrefixCache(capacity=4)
    assert c.lookup(LONG) is None
    assert c.misses == 1
    assert c.hits == 0


def test_store_and_hit_returns_entry():
    c = PrefixCache(capacity=4)
    c.store(LONG[:10], _FakeCache(42))
    res = c.lookup(LONG[:15])
    assert res is not None
    entry, lcp = res
    assert entry.prefix_ids == LONG[:10]
    assert entry.cached_kv == _FakeCache(42)
    assert lcp == 10
    assert c.hits == 1


def test_exact_match_caps_lcp_to_len_minus_one():
    """there must be at least one suffix token to prefill, so when the
    lookup probe equals the cached prefix we return lcp = len-1.
    """
    c = PrefixCache(capacity=4)
    c.store(LONG[:10], _FakeCache(7))
    res = c.lookup(LONG[:10])
    # only 9 suffix-able overlap available; need 1 token to prefill
    assert res is not None
    entry, lcp = res
    assert lcp == 9


def test_longest_prefix_wins():
    c = PrefixCache(capacity=4)
    c.store(LONG[:9], _FakeCache(1))
    c.store(LONG[:11], _FakeCache(2))
    c.store(LONG[:14], _FakeCache(3))
    res = c.lookup(LONG[:15])
    assert res is not None
    entry, lcp = res
    assert entry.cached_kv == _FakeCache(3)
    assert lcp == 14


def test_lcp_below_threshold_misses():
    """only 5 tokens shared — under MIN_LCP_FOR_HIT, treated as miss."""
    c = PrefixCache(capacity=4)
    c.store([1, 2, 3, 4, 5, 99, 99, 99, 99, 99], _FakeCache(0))
    assert c.lookup([1, 2, 3, 4, 5, 100, 100, 100, 100, 100]) is None
    assert c.misses == 1


def test_lru_eviction():
    c = PrefixCache(capacity=2)
    c.store(LONG[:9], _FakeCache(1))
    c.store(LONG[:10], _FakeCache(2))
    c.store(LONG[:11], _FakeCache(3))  # evicts the oldest (the 9-len entry)
    assert len(c) == 2
    # cache no longer contains the 9-length entry; longest hit on LONG[:15]
    # should be the 11-length entry now.
    res = c.lookup(LONG[:15])
    assert res is not None
    entry, lcp = res
    assert lcp == 11


def test_lookup_promotes_recent():
    c = PrefixCache(capacity=2)
    a_ids = LONG[:10]
    b_ids = list(range(100, 110))  # disjoint
    c.store(a_ids, _FakeCache(1))
    c.store(b_ids, _FakeCache(2))
    # touch a
    res = c.lookup(a_ids + [99])
    assert res is not None and res[0].cached_kv == _FakeCache(1)
    # storing a third entry should evict b (LRU), not a
    c.store(list(range(200, 210)), _FakeCache(3))
    res2 = c.lookup(a_ids + [99])
    assert res2 is not None and res2[0].cached_kv == _FakeCache(1)


def test_hit_rate_tracking():
    c = PrefixCache(capacity=4)
    c.store(LONG[:10], _FakeCache(1))
    c.lookup(LONG[:15])  # hit
    c.lookup(list(range(100, 115)))  # miss (disjoint)
    c.lookup(LONG[:14])  # hit
    assert c.hits == 2
    assert c.misses == 1
    assert c.hit_rate == pytest.approx(2 / 3)


def test_reset_stats():
    c = PrefixCache(capacity=4)
    c.store(LONG[:10], _FakeCache(1))
    c.lookup(LONG[:12])
    c.reset_stats()
    assert c.hits == 0
    assert c.misses == 0


def test_store_empty_is_noop():
    c = PrefixCache(capacity=4)
    c.store([], _FakeCache(0))
    assert len(c) == 0


def test_bad_capacity():
    with pytest.raises(ValueError):
        PrefixCache(capacity=0)


def test_redundant_store_just_promotes():
    c = PrefixCache(capacity=2)
    c.store(LONG[:10], _FakeCache(1))
    c.store(list(range(100, 110)), _FakeCache(2))
    c.store(LONG[:10], _FakeCache(99))  # same key, ignored payload
    res = c.lookup(LONG[:15])
    assert res is not None
    assert res[0].cached_kv == _FakeCache(1)
