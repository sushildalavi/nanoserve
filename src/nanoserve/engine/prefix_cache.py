"""prefix cache for nanoserve.

idea: when many requests share a common prompt prefix (system prompt,
multi-turn chat history, RAG snippet), we can avoid re-running prefill
across that prefix every time. on a cache hit we deep-copy the cached
DynamicCache and only prefill the suffix.

scope: exact-token-prefix match, bounded LRU eviction. no fancy radix tree
yet — this proves the mechanic and gives a measurable TTFT win for
shared-prefix workloads. true paged KV with cross-seq page sharing is a
separate phase that needs custom attention kernels (vLLM territory).

correctness: see L5-prefix parity gate. greedy outputs from a cache-hit
sequence must match the same prompt run from scratch, token-exact.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


def _hash_ids(ids: list[int]) -> str:
    h = hashlib.blake2b(digest_size=16)
    # use bytes for speed; ids fit in 4 bytes each for vocab < 2^32
    h.update(b"".join(i.to_bytes(4, "big", signed=False) for i in ids))
    return h.hexdigest()


@dataclass
class PrefixCacheEntry:
    """one cached prefix. holds the snapshot of past_kv after prefilling
    `prefix_ids`, plus the token id that the model would have emitted next
    so callers can decide whether to use it as the first generated token
    (chat templates make this fiddly — see engine integration).
    """
    prefix_ids: list[int]
    cached_kv: Any  # DynamicCache snapshot
    pad_len: int = 0  # if the cached prefill used left-padding


class PrefixCache:
    """bounded LRU keyed by token-prefix hash.

    semantics: lookup(prompt) returns the longest cached prefix that
    matches the start of `prompt`, or None. for now we only check exact
    full-prefix matches (no longest-common-prefix on partial overlap) — a
    radix tree would do better but explodes in complexity.
    """

    def __init__(self, capacity: int = 32):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._entries: OrderedDict[str, PrefixCacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    def reset_stats(self) -> None:
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def lookup(self, prompt_ids: list[int]) -> PrefixCacheEntry | None:
        """try every stored prefix; return the longest one that's a prefix
        of prompt_ids. prompt_ids must strictly extend the prefix (we don't
        return a hit when prompt_ids equals the cached prefix exactly,
        because there'd be nothing left to prefill).
        """
        best: PrefixCacheEntry | None = None
        best_len = 0
        for entry in self._entries.values():
            plen = len(entry.prefix_ids)
            if plen <= best_len or plen >= len(prompt_ids):
                continue
            if prompt_ids[:plen] == entry.prefix_ids:
                best = entry
                best_len = plen
        if best is not None:
            self.hits += 1
            # mark recently used
            self._entries.move_to_end(_hash_ids(best.prefix_ids))
        else:
            self.misses += 1
        return best

    def store(self, prefix_ids: list[int], cached_kv: Any, pad_len: int = 0) -> None:
        if not prefix_ids:
            return
        key = _hash_ids(prefix_ids)
        if key in self._entries:
            self._entries.move_to_end(key)
            return
        # snapshot the cache so subsequent decode steps don't mutate the
        # cached version. deepcopy is heavy but the cache holds tensors
        # that reference shared memory otherwise.
        snap = deepcopy(cached_kv)
        self._entries[key] = PrefixCacheEntry(
            prefix_ids=list(prefix_ids), cached_kv=snap, pad_len=pad_len
        )
        if len(self._entries) > self.capacity:
            self._entries.popitem(last=False)
