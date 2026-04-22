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


def test_empty_cache_returns_none():
    c = PrefixCache(capacity=4)
    assert c.lookup([1, 2, 3]) is None
    assert c.misses == 1
    assert c.hits == 0


def test_store_and_hit_returns_entry():
    c = PrefixCache(capacity=4)
    c.store([1, 2], _FakeCache(42))
    entry = c.lookup([1, 2, 3, 4])
    assert entry is not None
    assert entry.prefix_ids == [1, 2]
    assert entry.cached_kv == _FakeCache(42)
    assert c.hits == 1


def test_exact_match_does_not_hit():
    """no point returning a hit when there's nothing left to prefill."""
    c = PrefixCache(capacity=4)
    c.store([1, 2, 3], _FakeCache(7))
    assert c.lookup([1, 2, 3]) is None


def test_longest_prefix_wins():
    c = PrefixCache(capacity=4)
    c.store([1, 2], _FakeCache(1))
    c.store([1, 2, 3], _FakeCache(2))
    c.store([1, 2, 3, 4], _FakeCache(3))
    entry = c.lookup([1, 2, 3, 4, 5])
    assert entry is not None
    assert entry.prefix_ids == [1, 2, 3, 4]
    assert entry.cached_kv == _FakeCache(3)


def test_no_partial_match():
    """[1,9,2] is not a prefix-match for stored [1,2,3]."""
    c = PrefixCache(capacity=4)
    c.store([1, 2, 3], _FakeCache(99))
    assert c.lookup([1, 9, 2, 3]) is None


def test_lru_eviction():
    c = PrefixCache(capacity=2)
    c.store([1], _FakeCache(1))
    c.store([1, 2], _FakeCache(2))
    c.store([1, 2, 3], _FakeCache(3))  # evicts [1]
    assert len(c) == 2
    # [1] should be gone — querying [1, 9] now misses
    miss = c.lookup([1, 9])
    assert miss is None or miss.prefix_ids != [1]


def test_lookup_promotes_recent():
    c = PrefixCache(capacity=2)
    c.store([1, 2], _FakeCache(1))
    c.store([3, 4], _FakeCache(2))
    # touch [1, 2]
    hit = c.lookup([1, 2, 99])
    assert hit is not None
    # storing a third entry should evict [3, 4], not [1, 2]
    c.store([5, 6], _FakeCache(3))
    again = c.lookup([1, 2, 99])
    assert again is not None
    assert again.cached_kv == _FakeCache(1)


def test_hit_rate_tracking():
    c = PrefixCache(capacity=4)
    c.store([1, 2], _FakeCache(1))
    c.lookup([1, 2, 3])  # hit
    c.lookup([4, 5, 6])  # miss
    c.lookup([1, 2, 9])  # hit
    assert c.hits == 2
    assert c.misses == 1
    assert c.hit_rate == pytest.approx(2 / 3)


def test_reset_stats():
    c = PrefixCache(capacity=4)
    c.store([1], _FakeCache(1))
    c.lookup([1, 2])
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
    c.store([1, 2], _FakeCache(1))
    c.store([3, 4], _FakeCache(2))
    c.store([1, 2], _FakeCache(99))  # same key, ignored payload
    # cached payload should still be the original
    hit = c.lookup([1, 2, 9])
    assert hit is not None
    assert hit.cached_kv == _FakeCache(1)
