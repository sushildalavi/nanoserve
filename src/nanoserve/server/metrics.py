"""prometheus metrics for the nanoserve api server.

design: one module-level Registry holds all metrics so the /metrics endpoint
serializes them with a single call. counters and histograms are updated by
the api layer (request lifecycle); gauges are refreshed on each /metrics
scrape from the engine's live state.

label discipline: low-cardinality labels only. backend (nanoserve|hf_mps|
llama_cpp), quant (fp16|int8|torchao_int8), batching (off|on). no per-
request labels — those would explode the cardinality and break
prometheus's scrape model.
"""
from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# all metrics share this registry so /metrics emits exactly nanoserve's
# numbers and nothing from the global default registry. avoids duplicate
# python_gc_objects_collected_total and friends from polluting the scrape.
REGISTRY = CollectorRegistry()


# ---- request lifecycle counters / histograms ----

requests_total = Counter(
    "nanoserve_requests_total",
    "completed requests, partitioned by terminal status",
    labelnames=("status",),  # ok | error
    registry=REGISTRY,
)

input_tokens_total = Counter(
    "nanoserve_input_tokens_total",
    "tokens fed into prefill across all requests",
    registry=REGISTRY,
)

output_tokens_total = Counter(
    "nanoserve_output_tokens_total",
    "tokens generated across all requests",
    registry=REGISTRY,
)

# histograms: buckets chosen from the empirical distribution we've seen in
# the phase 1-3 sweeps. ttft ranges 100ms (idle) → 60s (heavy poisson).
# tpot 20ms (single seq fp16) → 700ms (int8 batch of 4).
ttft_seconds = Histogram(
    "nanoserve_ttft_seconds",
    "time from request submit to first generated token",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
    registry=REGISTRY,
)

tpot_seconds = Histogram(
    "nanoserve_tpot_seconds",
    "time per output token after the first (steady-state decode)",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0),
    registry=REGISTRY,
)

e2e_seconds = Histogram(
    "nanoserve_e2e_seconds",
    "wall time from submit to last generated token",
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
    registry=REGISTRY,
)


# ---- engine state gauges (refreshed on /metrics scrape) ----

active_seqs = Gauge(
    "nanoserve_active_seqs",
    "sequences currently in the scheduler's running set",
    registry=REGISTRY,
)

batched_forward_frac = Gauge(
    "nanoserve_batched_forward_frac",
    "fraction of forward steps that used the batched path (0.0-1.0)",
    registry=REGISTRY,
)

prefix_cache_hits_total = Counter(
    "nanoserve_prefix_cache_hits_total",
    "prefix cache hits since server start",
    registry=REGISTRY,
)

prefix_cache_misses_total = Counter(
    "nanoserve_prefix_cache_misses_total",
    "prefix cache misses since server start",
    registry=REGISTRY,
)

prefix_cache_size = Gauge(
    "nanoserve_prefix_cache_size",
    "number of entries currently held by the prefix cache",
    registry=REGISTRY,
)


def render() -> tuple[bytes, str]:
    """serialize the whole nanoserve registry. returns (body, content_type)
    suitable for a fastapi Response.
    """
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


def refresh_gauges_from_engine(engine) -> None:
    """call before serializing /metrics. updates state gauges + cache
    hit/miss counters from whatever the engine has accumulated since the
    last scrape. counter deltas are computed locally to avoid going
    backwards.
    """
    if engine is None:
        return
    sched = engine._scheduler
    active_seqs.set(len(sched.running))
    total_fwd = engine.batched_forward_steps + engine.single_forward_steps
    if total_fwd > 0:
        batched_forward_frac.set(engine.batched_forward_steps / total_fwd)
    if engine.prefix_cache is not None:
        # use _prev_* attrs on this module to compute deltas — counter
        # values must monotonically increase, so we add the delta since
        # last scrape.
        global _prev_hits, _prev_misses
        cur_hits = engine.prefix_cache.hits
        cur_misses = engine.prefix_cache.misses
        prefix_cache_hits_total.inc(max(0, cur_hits - _prev_hits))
        prefix_cache_misses_total.inc(max(0, cur_misses - _prev_misses))
        _prev_hits = cur_hits
        _prev_misses = cur_misses
        prefix_cache_size.set(len(engine.prefix_cache))


_prev_hits: int = 0
_prev_misses: int = 0
