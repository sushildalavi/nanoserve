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

    semantics: lookup(prompt) does a longest-common-prefix scan across
    every cached entry. returns (entry, lcp_length) for the entry that
    shares the longest token-prefix with `prompt`, or None if no overlap
    exceeds MIN_LCP_FOR_HIT. callers slice entry.cached_kv to lcp_length
    before reusing it.
    """

    # don't bother returning a hit if the overlap is shorter than this — the
    # deepcopy + slice cost is only worth it for meaningfully long shared
    # prefixes (system prompts, RAG context, etc).
    MIN_LCP_FOR_HIT = 8

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

    def lookup(self, prompt_ids: list[int]) -> tuple[PrefixCacheEntry, int] | None:
        """find the cached entry that shares the longest token prefix with
        prompt_ids. returns (entry, lcp_length) where lcp_length <=
        min(len(entry.prefix_ids), len(prompt_ids) - 1). callers slice
        entry.cached_kv to lcp_length before reusing it.

        miss conditions: no overlap, or every overlap is less than
        MIN_LCP_FOR_HIT tokens (small overlaps aren't worth the deepcopy).
        """
        best_entry: PrefixCacheEntry | None = None
        best_lcp = 0
        for entry in self._entries.values():
            # longest common prefix of entry.prefix_ids and prompt_ids,
            # capped at len(prompt_ids) - 1 so there's at least 1 suffix
            # token to actually run prefill on.
            cap = min(len(entry.prefix_ids), len(prompt_ids) - 1)
            lcp = 0
            while lcp < cap and entry.prefix_ids[lcp] == prompt_ids[lcp]:
                lcp += 1
            if lcp > best_lcp:
                best_lcp = lcp
                best_entry = entry
        if best_entry is None or best_lcp < self.MIN_LCP_FOR_HIT:
            self.misses += 1
            return None
        self.hits += 1
        self._entries.move_to_end(_hash_ids(best_entry.prefix_ids))
        return best_entry, best_lcp

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
