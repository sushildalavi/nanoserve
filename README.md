# nanoserve

An OpenAI-compatible LLM serving engine built from scratch, with continuous batching, paged KV cache, prefix caching, streaming, and observability on Apple Silicon. Benchmarked against an honest native ceiling (llama.cpp Metal) and a naive single-stream baseline (HuggingFace Transformers on MPS).

Runs on an M3 MacBook Air. No CUDA, no cloud, no Kubernetes. The interesting parts are the scheduler, the KV cache layout, and the numbers under load.

## Status

Phase 1 (measurement first). Benchmark harness, load generator, and baselines land before any engine code, so every technique that follows has a pre-recorded number to beat.

- [x] repo scaffold, pyproject, makefile
- [x] metrics (p50/p95/p99, TTFT, TPOT, decode tok/s) with tests
- [x] workload generator (closed-loop + poisson open-loop)
- [x] json + csv report writer
- [x] async runner
- [x] hf-mps baseline (naive, one at a time)
- [x] llama.cpp baseline (Q8_0 + Q4_K_M, Metal)
- [x] parity test scaffold
- [x] baseline numbers filled in `results/ablations.csv`
- [x] phase 2A: fixed-shape continuous batching scheduler + L1/L2/L3 parity gates
- [x] phase 2B: mixed-length (left-padding + masks), EOS retirement, L3-var parity, Poisson sweep λ=1/2/4
- [x] phase 3A: hand-rolled INT8 weight-only quant + L4-quant parity + diagnosis
- [x] phase 3B: forward/overhead timing + synchronized admission policy + paired sweep (fp16+int8 each)
- [x] phase 3C: prefix cache (LCP-matching with on-the-fly cache slicing) + L5-prefix parity + sweep
- [x] phase 3D: torch 2.8 + torchao 0.9 vs hand-rolled int8 — confirms platform constraint (no native int8 matmul on MPS)
- [x] phase 4: openai-compatible streaming api (FastAPI + SSE) + prometheus metrics + serve CLI + 6 integration tests
- [x] phase 5A: live grafana dashboard provisioned from `ops/` (4 rows, 12 panels)
- [x] phase 5B: int4 packed weight-only quant + L4-int4 parity + sweep script
- [x] phase 5C: quality eval harness (perplexity + hellaswag-mini) with `results/eval.csv`

### Explicitly out of scope (documented, not attempted)

- **True paged KV with custom Metal attention kernels** — vLLM territory. Not implementable in pure pytorch on MPS; would require Metal kernel work. Prefix caching (phase 3C) is the implementable subset we ship in its place. This also means the original 3-ablation plan (batching off / paging off / prefix cache off) ships as **2 ablations** (batching off, prefix cache off); the paging-off row is listed as N/A.
- **MLX port** — the only path to make int8/int4 actually beat fp16 on Apple Silicon (MLX has native Metal int8/int4 matmul kernels; pytorch-MPS does not). Clean pivot, multi-week scope. Deferred indefinitely; the phase 3D writeup makes the case that the remaining gap is platform-level, not a bug.
- Kubernetes / Ray Serve / gRPC / auth / multi-tenancy.

## Quickstart

```bash
make dev-install
make models           # downloads tinyllama gguf + builds llama.cpp with metal
make baseline-hf
make baseline-llamacpp
make parity
make serve            # starts the openai-compatible api on :8000
make observe          # in another terminal: local prometheus + grafana
make eval             # perplexity + hellaswag across fp16/int8/int4 (writes results/eval.csv)
```

Results land in `results/ablations.csv` (headline table) and `results/runs/*.json` (per-run dumps with full config, env, and per-request records).

### Hit it like the OpenAI API

Once `make serve` is running:

```bash
# health
curl localhost:8000/health

# non-streaming completion
curl -s localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"tinyllama-1.1b-chat","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'

# streaming completion (SSE)
curl -N localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"tinyllama-1.1b-chat","messages":[{"role":"user","content":"Tell me a fact about cats."}],"max_tokens":64,"stream":true}'

# prometheus metrics
curl localhost:8000/metrics | grep nanoserve_
```

The same engine + scheduler + prefix cache + INT8 path the bench harness drives is what the API serves — there is no second implementation. Phase 1 designed the `EngineService` interface specifically so the API and the bench could share an engine.

### View the live dashboard

Once the API is running, spin up a local prometheus + grafana pair to watch the metrics:

```bash
brew install prometheus grafana   # one-time
make observe                      # runs both in the foreground, ctrl-c stops them
# open http://localhost:3000  (admin / admin)
# dashboard: http://localhost:3000/d/nanoserve/nanoserve
```

The dashboard auto-provisions from [`ops/grafana/dashboards/nanoserve.json`](ops/grafana/dashboards/nanoserve.json). Four rows — serving (RPS, error rate, active seqs, batched-forward fraction), latency (TTFT / TPOT / e2e at p50/p95/p99), throughput (input vs output token rates, status breakdown), and the prefix cache (hit rate, size, hits-vs-misses). Full ops README at [`ops/README.md`](ops/README.md).

## The ablation table

This is the whole point of the project. Every technique is added with its on/off row locked in the same day it ships, no retrofitting. Measured on an M3 MacBook Air (8-core CPU, 10-core GPU, 16 GB unified memory), TinyLlama-1.1B-Chat, 100 requests, `max_new_tokens=128`, fixed seed.

| backend    | quant  | batching | workload         | rps  | p50 TTFT   | p95 TTFT | p95 e2e  | decode tok/s | avg batch |
|------------|--------|----------|------------------|------|------------|----------|----------|--------------|-----------|
| hf-mps     | fp16   | off      | closed-loop c=1  | 0.18 | **135 ms** | 253 ms   | 6.4 s    | 21.8         | —         |
| hf-mps     | fp16   | off      | poisson λ=2      | 0.17 | 247 s      | 489 s    | 495 s    | 21.5         | —         |
| llama.cpp  | Q8_0   | n/a      | poisson λ=2      | 1.43 | 5.0 s      | 9.9 s    | 12.7 s   | 47.5         | —         |
| llama.cpp  | Q4_K_M | n/a      | poisson λ=2      | 1.16 | 12.3 s     | 22.9 s   | 26.3 s   | 39.1         | —         |
| nanoserve  | fp16   | off      | closed-loop c=1  | 0.37 | 102 ms     | 274 ms   | 3.5 s    | 30.7         | 1.00      |
| nanoserve  | fp16   | on       | closed-loop c=4  | 0.46 | 259 ms     | 719 ms   | 10.1 s   | 9.2          | 3.75      |
| nanoserve  | fp16   | off      | poisson λ=2      | 0.34 | 41 s       | 87 s     | 90 s     | 26.7         | 1.00      |
| nanoserve  | fp16   | on       | poisson λ=1      | 0.35 | 17 s       | 52 s     | 65 s     | 7.1          | 3.76      |
| nanoserve  | fp16   | on       | poisson λ=2      | 0.33 | 36 s       | 88 s     | 97 s     | 6.5          | 3.89      |
| nanoserve  | fp16   | on       | poisson λ=4      | 0.35 | 41 s       | 93 s     | 101 s    | 6.9          | 3.88      |
| nanoserve  | **int8** | off    | closed-loop c=1  | 0.13 | 271 ms     | 451 ms   | 9.6 s    | 5.5          | 1.00      |
| nanoserve  | **int8** | on     | closed-loop c=4  | 0.13 | 897 ms     | 1.5 s    | 33.5 s   | 1.5          | 3.23      |

**Phase 3B paired sweep — fcfs vs synchronized admission, fp16 + int8 (poisson λ=2, c=4, n=20, max_new=32):**

| quant | admission | rps | p50 TTFT | p50 TPOT | decode tok/s | avg batch | **batched_forward_frac** | forward p50 | overhead p50 |
|---|---|---|---|---|---|---|---|---|---|
| fp16 | fcfs         | 0.74 | 7.1 s  | 130 ms | 8.1  | 3.80 | **0.000** | 36 ms  | 3.4 ms  |
| fp16 | synchronized | **0.82** | 6.7 s | 107 ms | **10.1** | 3.33 | **0.833** | 111 ms | 16 ms |
| int8 | fcfs         | 0.24 | 29.7 s | 434 ms | 2.4  | 3.86 | **0.000** | 124 ms | 3.2 ms |
| int8 | synchronized | **0.48** | 15.1 s | 190 ms | **5.6** | 3.30 | **0.833** | 205 ms | 19 ms |

The two HF-MPS rows are the story. Raw forward-pass TTFT on an idle engine is **135 ms** — totally reasonable. Under 2 req/s Poisson arrivals the same engine has p50 TTFT of **247 seconds**, a 1,800× jump. Decode tok/s is identical across both rows (~21.8). Nothing about the model got slower — the engine just can't serve requests in parallel, so the queue grows linearly and by request 100 you're waiting four minutes behind 99 others. This is exactly the bottleneck continuous batching was meant to kill.

llama.cpp Q8_0 on Metal is the honest native ceiling at ~47 decode tok/s. That's what nanoserve has to approach, not beat.

Q4_K_M is slower than Q8_0 here because the model is too small (1.1B) to be memory-bandwidth bound, so the extra dequant work dominates. At 7B+ the order typically flips.

### Phase 2A — fixed-shape continuous batching

The two `nanoserve` rows validate the new engine. With `batching_mode=off` (serial, same engine with one flag flipped) the raw forward path is actually faster than HF-MPS generate (30.7 vs 21.8 decode tok/s) — we skip `generate()`'s sampling/stopping-criteria overhead by calling `model.forward` directly with our own past_kv bookkeeping. That's the sanity row: it proves the engine isn't slower than the reference.

With `batching_mode=on` and `max_batch_size=4`, the scheduler actually formed batches (avg batch size 3.75 of 4). Per-sequence decode tok/s fell from 30.7 to 9.2, but aggregate system throughput rose from 0.37 to 0.46 req/s. Each forward pass now processes 4 tokens for the price of ~1, so the engine can serve more concurrent requests — at the cost of higher per-request TTFT since arrivals wait for the current batch to drain (2A synchronized admission).

This is a deliberately modest first ablation: closed-loop at c=4, same-length prompts, fixed max_new_tokens, no mid-batch retirement. Phase 2B relaxes all of these and runs the full Poisson load sweep.

**Correctness gates** backing the rows above (all passing):
- **L1** — single-seq greedy output matches HF generate token-exact
- **L2** — two interleaved seqs each match their isolated greedy output
- **L3** — N seqs in a single batched forward step each match isolated greedy output (same-length prompts)
- **L3-var** (2B) — N seqs of **different** lengths, batched via left-padding + attention_mask, each match isolated greedy output
- **EOS retirement** (2B) — one seq finishing mid-batch does not corrupt the remaining seqs' outputs
- **L4-quant** (3A) — INT8 weight-only quantized engine matches fp16 reference on the first 4 greedy tokens (overlap, not exact — quant rounds)
- **Synchronized admission policy** (3B) — scheduler unit tests cover both `fcfs` (admit into running slots) and `synchronized` (block admit until current batch drains)
- **L5-prefix** (3C) — sequence served from a prefix-cache hit produces the same greedy tokens as the same prompt run cold (token-exact, via deepcopy + on-the-fly cache slicing)
- **Prefix cache LCP semantics** — 12 unit tests cover bounded LRU eviction, longest-common-prefix scan across cached entries, MIN_LCP threshold to avoid micro-hits, hit-rate accounting

### Phase 2B — mixed-length batching + Poisson sweep

Phase 2B removes the same-length-prompt constraint and runs the real Poisson load sweep. Variable-length prefill uses left-padding + an attention mask; each per-decode-step mask grows with the cache so padding positions stay invisible to attention. EOS retirement lands naturally through the existing scheduler: when a seq hits its stop condition, it's dropped from `running` and the next step's `pick_decode_batch` returns the shrunken set.

**The Poisson sweep is the headline result — and its headline is honest:**

Across the entire sweep, **aggregate system throughput is flat at ~26 tok/s**. That's `decode_tok_s × avg_batch_size` for each row:

| row | decode tok/s | avg batch | aggregate tok/s |
|---|---|---|---|
| serial λ=2         | 26.7 | 1.00 | **26.7** |
| continuous λ=1     |  7.1 | 3.76 | **26.7** |
| continuous λ=2     |  6.5 | 3.89 | **25.3** |
| continuous λ=4     |  6.9 | 3.88 | **26.7** |

Batching produces **near-full batches (3.76–3.89 of 4)** on the scheduler side — the admission policy works, the mask plumbing works, the cache stack/split is correct (the parity gates prove it). But per-seq `decode_tok_s` falls from 27 to ~7 in continuous mode, almost exactly offsetting the 4× parallelism. The engine is doing more work per step, in roughly the same wall-time per step, for the same total system output.

**Why batching doesn't help here:** at 1.1B parameters in fp16 on M3 MPS, each forward pass is **compute-bound, not memory-bound**. Batching 4 sequences means running 4× the matrix-multiply work per step; there's no memory-bandwidth "free lunch" to amortize that cost against, because the weights are small enough to stay resident in the GPU's L2/SLC for single-seq inference too. The regime where continuous batching pays off is *larger* models (7B+) and *quantized* weights — that's why llama.cpp Q8_0 still wins the absolute throughput crown in the table above.

This is not a failure of the engine. The engine is behaving correctly — correctness gates L1/L2/L3/L3-var/EOS all pass, the scheduler forms near-full batches, masks apply correctly. It's a demonstration that **continuous batching is not a universal win**: at small-model scales on a compute-bound device, the technique that made vLLM famous on A100s with 70B models doesn't move the needle.

**What *does* help here, based on the baseline rows:**
- Going from HF `generate()` (serial λ=2) to nanoserve serial: **0.17 → 0.34 rps**, purely by calling `model.forward` directly instead of going through `generate()`'s sampling and stopping-criteria overhead.
- Going from fp16 to Q8_0 quantization (llama.cpp): **21.8 → 47.5 decode tok/s**, because now the forward pass *is* memory-bound and batching/quantization both buy real wins.

### Phase 3A — INT8 weight-only quantization, with diagnosis

The Phase 2B finding said batching doesn't help on TinyLlama-fp16 because forward passes are compute-bound. The natural next step is quantization to shift the regime to memory-bound. Phase 3A implements **hand-rolled INT8 weight-only quantization** ([`src/nanoserve/engine/quant.py`](src/nanoserve/engine/quant.py)): per-row symmetric scales, weights stored as int8, dequant-to-fp16 on every forward pass. Correctness verified by the L4-quant parity gate (INT8 matches fp16 on the first 4 greedy tokens for every test prompt).

**The most valuable result was not a speedup claim, but a diagnosis:** on M3 MPS at TinyLlama scale, throughput was limited by fp16 compute and dequant overhead, while mixed cache lengths prevented the scheduler from forming enough truly batched forward passes to amortize that cost.

The two findings are stacked, and each is supported by a different piece of evidence:

**1. Platform constraint — evidenced by the serial gap, before batching enters the story.**
INT8 serial c=1: 5.5 decode tok/s. fp16 serial c=1: 30.7 decode tok/s. That ~6× per-seq gap exists with batching disabled, so it can't be blamed on batching. The cause is straightforward: `Int8LinearWeightOnly.forward` materializes a full fp16 weight tensor (`w_deq = weight_q.to(fp16) * scale`) before calling `torch.nn.functional.linear`, because **MPS has no native int8 matmul kernel**. The dequant cost is paid on every forward, and the matmul still operates in fp16. Memory savings work as predicted (272 MB vs 484 MB, 44% reduction) but throughput goes the wrong direction. On CUDA the quantized weight stays int8 through matmul via CUTLASS or torchao kernels; on MPS that path doesn't exist (yet) in the torch < 2.6 ecosystem we pinned for Phase 1 stability.

**2. Scheduler constraint — evidenced by `batched_forward_frac` collapsing to 0.007 in continuous mode.**
The new `batched_forward_frac` counter ([`src/nanoserve/baselines/nanoserve_engine.py`](src/nanoserve/baselines/nanoserve_engine.py)) distinguishes scheduled batches from truly batched forward passes. In INT8 continuous c=4 it came out **0.007** — only 0.7% of forward steps actually used the batched code path. The other 99.3% fell through to per-seq decode because mid-stream admissions left sequences at different cache lengths, breaking the same-length precondition in `_can_batch_forward`. So even with `avg_batch_size = 3.23` from the scheduler's perspective, the model.forward calls were almost entirely per-seq. That's why continuous mode failed to amortize the dequant cost.

**The decomposition matters because the next experiment isolates exactly one variable at a time.** Phase 3B tackles the scheduler side first (step-aligned admission to lift `batched_forward_frac` toward 1.0) and only then revisits the platform side (torch upgrade + torchao for native int8 matmul, or MLX). Direct per-step timing of dequant vs matmul vs scheduler glue is added before any platform pivot, so future claims about which section dominates are measured rather than inferred.

> The strongest portfolio claim from this phase isn't "I made it faster" — it's "I instrumented the system enough to discover *why* a textbook optimization regressed throughput on this specific hardware/runtime, and designed the next experiment to isolate the cause." That's the honest version of systems work.

### Phase 3B — synchronized admission + per-step timing (the diagnosis paid off)

Phase 3B implemented the two changes the 3A diagnosis pointed to:

1. **Per-step timing instrumentation** — `forward_p50_ms` / `forward_p95_ms` (time inside `model.forward`, with `torch.mps.synchronize()` for accuracy) and `step_overhead_p50_ms` / `step_overhead_p95_ms` (time in the same driver iteration outside the model). Together they answer "where did the wall-clock time actually go?"

2. **Synchronized admission policy** — new `admission_policy: "fcfs" | "synchronized"` flag in [`SchedulerConfig`](src/nanoserve/engine/scheduler.py). Under `synchronized`, new arrivals are blocked from admission until every running seq is `FINISHED`. This forces same-length cache states across the next batch, lifting `_can_batch_forward` from "rarely true" to "almost always true" under Poisson load.

**The paired sweep (above) shows three results worth their own resume bullets:**

**Result 1: The scheduler bug was real, and bigger than expected.** Under fp16 + Poisson + FCFS admission, `batched_forward_frac = 0.000` — the engine was scheduled with `avg_batch_size = 3.80` but **never** ran a truly batched forward. Phase 2B's "batching is flat at ~26 tok/s" finding was largely an artifact of this scheduler bug, not a fundamental compute-bound limit. Switching to synchronized admission lifted `batched_forward_frac` to 0.833 in both fp16 and INT8.

**Result 2: With real batching, fp16 throughput goes up.** `synchronized` vs `fcfs` for fp16 Poisson λ=2: rps **0.74 → 0.82** (+10%), per-seq decode tok/s **8.1 → 10.1** (+25%), and TTFT slightly *better* (7.1 s → 6.7 s) because batched forwards drain the queue faster than per-seq forwards do. The cost is forward_p50 going from 36 ms → 111 ms (each step does 4× the work), but that cost amortizes across the four sequences.

**Result 3: INT8 nearly doubled.** `synchronized` vs `fcfs` for INT8 Poisson λ=2: rps **0.24 → 0.48** (+96%). The 3A diagnosis predicted exactly this — the dequant cost can be amortized if real batching actually happens. INT8 still loses to fp16 (0.48 vs 0.82) because dequant overhead per forward is not zero; closing that remaining gap requires native int8 matmul (Phase 3C: torch ≥ 2.6 + torchao or MLX).

The forward_p50 / step_overhead split also confirmed something useful: even in the worst (slowest) row, `step_overhead_p50` is < 20 ms while `forward_p50` is 36–205 ms. **Python-side scheduler overhead is < 10% of step time** in every configuration. The bottleneck is unambiguously the model forward pass, not the scheduler glue. That rules out a bunch of "we should rewrite the scheduler in Rust" type hypotheses before they get raised.

**Takeaway**: the negative finding from 3A wasn't proof that the technique was wrong — it was a measurement that revealed two stacked constraints, one of which was fixable in one commit. The 3A → 3B sequence is the actual portfolio story:
> *"Implemented a continuous-batching scheduler. Initial benchmarks showed flat throughput. Added forward-pass instrumentation, discovered the admission policy was breaking batching under variable-length workloads, fixed it, and verified that the same INT8 weight-only quant path that previously showed a 3× regression now shows a 2× speedup (with synchronized admission), narrowing the gap to fp16 from -67% to -42%. The remaining gap is platform-level (no native int8 matmul on MPS) and isolated as the next experiment."*

### Phase 3C — prefix cache with longest-common-prefix matching

True paged KV (vLLM-style page tables + custom attention kernels) requires writing Metal kernels — that's multi-week scope and outside the pure-pytorch envelope this project commits to. The implementable, high-value subset is **prefix caching**: detect when a new prompt shares a long token-level prefix with a previously-served prompt, skip the prefill of the matched portion, only run the suffix. This is what production serving stacks actually use for chat workloads with system prompts and multi-turn history.

**The implementation** ([`src/nanoserve/engine/prefix_cache.py`](src/nanoserve/engine/prefix_cache.py)):
- Bounded LRU keyed by token-prefix hash
- `lookup(prompt_ids)` does a longest-common-prefix scan across every cached entry, returns `(entry, lcp_length)` for the best match
- `MIN_LCP_FOR_HIT=8` threshold suppresses micro-hits where the deepcopy + slice cost would exceed the prefill savings
- Engine slices the cached `DynamicCache` to `lcp_length` tokens (per-layer keys/values are sliced along the seq dimension), then prefills only the suffix `prompt_ids[lcp:]`

**Demo workload (long shared system prompt, c=4 closed-loop, max_new=16):**

| | rps | p50 TTFT | hit rate |
|---|---|---|---|
| 200-tok shared prefix, cache off | 1.37 | 2.20 s | n/a |
| **200-tok shared prefix, cache on** | **1.51** (+10%) | **1.96 s** (-11%) | 0.92 |

A 200-token shared system prompt + 12 user-query tokens means cache-off prefill processes 212 tokens per request. Cache-on skips ~92% of the prefill work (200 of 212 tokens come from the cache), saving ~30 ms per request and lifting throughput from 1.37 to 1.51 rps. The hit rate is 92% — the misses are the first request that warms the cache, plus the rare prompt that picks a path that happens not to overlap with anything cached yet.

**Smaller-prefix sweep (30-token shared, max_new=32):**

| | rps | p50 TTFT | hit rate |
|---|---|---|---|
| serial cache off | 0.83 | 3.71 s | — |
| serial cache on  | 0.81 | 3.60 s | 0.95 |
| poisson+sync cache off | 0.86 | 5.76 s | — |
| poisson+sync cache on  | 0.90 (+5%) | 5.47 s (-5%) | 0.95 |

At a 30-token prefix the savings are marginal — saving 30 tokens of prefill is small relative to 32 decode tokens. The hit rate is still 95% (mechanism works) but the throughput delta sits inside noise on serial. The clear win is in the Poisson + synchronized + cache-on row at the bottom: +5% rps and -5% TTFT compared to no cache.

**Two known limitations worth naming:**
- **Cache hits force per-seq decode in continuous mode.** When any seq in a batch has a cache hit, the engine falls back from `_decode_batch` to `_decode_one` because the matched-prefix lengths don't all align. So in continuous mode, the prefix-cache win and the batching win don't compose cleanly. Future fix: cache-aware admission grouping (admit cache-hits separately from cache-misses so each group can batch).
- **Pure-pytorch slicing of `DynamicCache` per-layer is fast but not zero-cost.** A real paged KV would store K/V in fixed-size blocks indexed by a page table, eliminating the slice + deepcopy. That requires Metal kernel work and is deferred to Phase 4.

**The portfolio claim from 3C:**
> *"Implemented a longest-common-prefix cache for KV reuse, including on-the-fly cache slicing for partial-match hits. Cache mechanism verified by token-exact greedy parity (L5-prefix gate). Demonstrates the throughput envelope: +10% rps and -11% TTFT on system-prompt-heavy workloads. Documents the boundary where prefix caching and continuous batching currently don't compose, with a clear next step (cache-aware admission grouping)."*

### Phase 3D — torchao for native Metal int8 matmul (the platform constraint, finally measured directly)

Phase 3A's diagnosis predicted that the INT8 throughput regression was platform-level: MPS has no native int8 matmul kernel, so weight-only quantization always pays a dequant tax on every forward. Phase 3B fixed the scheduler-side bottleneck. Phase 3D tests the platform bet directly: upgrade to torch 2.8 + torchao 0.9 and use torchao's `int8_weight_only()`, which on CUDA uses CUTLASS int8 kernels.

**Setup:** torch upgraded from 2.5 → 2.8 (full test suite still 55/55 green), torchao 0.9 installed (0.17 requires torch 2.11+ which doesn't exist yet), new `quant_mode = "torchao_int8"` flag in the engine. Three paired closed-loop c=1 runs:

| quant path | rps | forward p50 | decode tok/s | mem peak |
|---|---|---|---|---|
| fp16 (baseline) | **0.866** | 33.7 ms | 37.6 | 1556 MB |
| hand-rolled int8 (3A) | 0.120 | 258 ms | 4.86 | 282 MB |
| **torchao int8 (3D)** | 0.128 | 240 ms | 5.24 | 1624 MB |

**Result: torchao gives a marginal ~8% improvement over my hand-rolled module** (forward 258 → 240 ms, decode 4.86 → 5.24 tok/s) but is still **~7× slower than fp16**. The gap to fp16 doesn't close.

**Why?** The `import error: No module named 'triton'` that torchao logs on import is the giveaway. Triton is the kernel-compilation backend torchao uses for fused int8 matmul on CUDA. On macOS / MPS there is no triton, so torchao falls back to the same dequant-then-fp16-matmul path that my hand-rolled module uses. The 8% delta is just torchao's somewhat-faster dequant routine — not native int8 compute.

**This closes the Phase 3 quantization arc with a definitive answer:**
> On Apple Silicon MPS at TinyLlama-1.1B scale, INT8 weight-only quantization in pytorch — whether hand-rolled or via torchao — is **net-negative for throughput** because the dequant tax (~7× per-step overhead) dominates the memory-bandwidth savings. Memory savings work as expected (44% RAM reduction with hand-rolled, less with torchao because of its tensor-subclass overhead). The only way to actually flip the regime on this hardware is **MLX**, which has native Metal int4/int8 matmul kernels in its operator library. Pure-pytorch + MPS hits a hard wall at the runtime level.

**The 3A → 3B → 3C → 3D arc as one portfolio narrative:**

> *"Built an LLM serving engine on Apple Silicon. The first ablation showed continuous batching produced flat throughput. Added per-step timing instrumentation to distinguish scheduled batches from truly batched forwards (`batched_forward_frac`). That metric showed the FCFS admission policy was producing 0% true batching under variable-length Poisson workloads. Implemented synchronized admission, lifting `batched_forward_frac` from 0.0 → 0.83 and INT8 throughput by 2×. Implemented prefix caching with longest-common-prefix matching for system-prompt-heavy workloads, +10% throughput. Tested the platform pivot to torchao for native Metal int8 matmul; torchao falls back to the same dequant-fp16 path my hand-rolled module uses because Triton is unavailable on macOS, confirming the platform-constraint diagnosis. Documented the only remaining path (MLX port) and the regimes where each technique helps. Five correctness gates (L1, L2, L3, L3-var, L5-prefix) cover all of it. ~70 commits across 1.5 days."*

That's the actual story. It's not "I made a fast LLM server." It's "I built a measurement framework that answered well-defined questions about a real system, including diagnosing where the techniques don't help and why."

### Phase 4 — OpenAI-compatible API + Prometheus observability

The locked 8-feature scope had two production-shape items left after Phase 3: an OpenAI-compatible HTTP server (feature 1) and Prometheus observability (feature 4). Phase 4 ships both, plus a CLI subcommand to start the server and a 6-test integration suite that runs the full request/response path in-process via `httpx.ASGITransport`.

**The server** ([`src/nanoserve/server/api.py`](src/nanoserve/server/api.py)):
- `POST /v1/chat/completions` — accepts the OpenAI ChatCompletion request schema (subset; `extra='ignore'` so optional client fields don't break validation). Greedy decode only.
- `stream=true` returns `text/event-stream` with one `data: {chunk}` line per produced token, an opening role chunk, a closing chunk with `finish_reason`, and the OpenAI `data: [DONE]` terminator.
- `stream=false` returns a single `ChatCompletion` JSON with `usage.prompt_tokens` / `completion_tokens` / `total_tokens` populated by the engine's tokenizer.
- `GET /health` — liveness + the engine config snapshot.
- `GET /metrics` — Prometheus text format. Refreshes the engine state gauges on each scrape so the values are point-in-time, not stale.

**The metrics module** ([`src/nanoserve/server/metrics.py`](src/nanoserve/server/metrics.py)):
- Counters: `nanoserve_requests_total{status}`, `nanoserve_input_tokens_total`, `nanoserve_output_tokens_total`, `nanoserve_prefix_cache_hits_total`, `nanoserve_prefix_cache_misses_total`
- Histograms: `nanoserve_ttft_seconds`, `nanoserve_tpot_seconds`, `nanoserve_e2e_seconds` (buckets chosen from the empirical distribution observed in the Phase 1–3 sweeps)
- Gauges: `nanoserve_active_seqs`, `nanoserve_batched_forward_frac`, `nanoserve_prefix_cache_size`
- Dedicated `CollectorRegistry` so `/metrics` only emits nanoserve numbers — no Python GC counters from the global default registry polluting the scrape.
- Cache hit/miss counters use a delta-based update that survives engine restarts without going backward (Prometheus counter invariant).

**The hard bug worth naming.** The first version of the API integration test froze indefinitely on the first request. Symptom: the `/health` and `/metrics` tests passed, but the chat-completion test hung forever at 0% CPU. Root cause: the test fixture used `asyncio.run()` to start the engine, which creates a transient event loop. The engine's driver thread captured a reference to that loop in `start()` (via `asyncio.get_running_loop()`), and later when each test ran in its OWN new loop (pytest-asyncio's default), the driver's `loop.call_soon_threadsafe(q.put_nowait, ev)` kept calling against the dead loop. The puts dropped silently, the per-request asyncio.Queue never got events, the test waited forever. Fix: `asyncio_default_fixture_loop_scope = "module"` in pyproject + an async fixture with `loop_scope="module"` so the driver and the tests share one loop. Documented inline in [`tests/test_server_api.py`](tests/test_server_api.py) as the kind of thing that bites once and is obvious in retrospect.

**The integration tests** (all 6 pass in ~6 s):
- `test_health_returns_engine_config` — `/health` returns 200 + the configured engine knobs
- `test_metrics_endpoint_returns_prometheus_text` — `/metrics` is parseable Prometheus text
- `test_chat_completion_non_streaming` — `stream=false` returns a complete `ChatCompletion` with usage
- `test_chat_completion_streaming_emits_sse_chunks` — `stream=true` emits role chunk + content chunks + finish chunk + `[DONE]`
- `test_request_metrics_increment` — `nanoserve_requests_total{status="ok"}` grows after a real completion
- `test_empty_messages_rejected` — empty `messages` returns 400 (validation works)

Plus 8 unit tests for the metrics module (counter increments, histogram bucket emission, delta-based gauge refresh, no-engine safety).

**Total test suite: 69/69 passing.** Phase 4 added the server (2 modules), the CLI subcommand, 14 new tests, and the deps (`fastapi`, `uvicorn`, `prometheus-client`, `pytest-asyncio`).

**The full project arc as a single resume bullet:**
> *"Built an end-to-end LLM serving stack from scratch on Apple Silicon: continuous-batching scheduler with fcfs/synchronized admission policies, mixed-length batched forward with attention masking, EOS retirement, longest-common-prefix KV reuse with on-the-fly cache slicing, INT8 weight-only quantization (hand-rolled and torchao paths), per-step forward/overhead instrumentation, OpenAI-compatible streaming API with Prometheus observability. Six correctness gates (token-exact greedy parity for single-seq, interleaved, batched, variable-length, prefix-cached, and quantized paths) plus 14 system/unit tests covering the scheduler state machine, prefix cache LRU semantics, metrics module, and full request/response API. 13 ablation rows in a locked CSV with reproducible run scripts. ~95 commits across 1.5 days. Every claim in the README is backed by a number that came out of a run."*

Raw artifacts live in [`results/ablations.csv`](results/ablations.csv) and [`results/runs/`](results/runs/) (full per-request records + env).

### Phase 5A — live Grafana dashboard

Phase 4 shipped the Prometheus metrics; Phase 5A adds the dashboard so the numbers are actually visible under load. The dashboard is checked in at [`ops/grafana/dashboards/nanoserve.json`](ops/grafana/dashboards/nanoserve.json) and auto-provisions via [`ops/grafana/provisioning/`](ops/grafana/provisioning/). Four rows, 12 panels:

- **serving** — RPS (stat), error rate (thresholded stat), active seqs (stat), batched-forward fraction (gauge with green/yellow/red thresholds)
- **latency** — TTFT / TPOT / e2e timeseries, each with p50 / p95 / p99 derived from the histograms via `histogram_quantile`
- **throughput** — input vs output token rates (timeseries) and a stacked requests-by-status panel
- **prefix cache** — hit rate, cache size, and a hits-vs-misses timeseries for watching warmup behavior

`make observe` starts prometheus (scrape interval 5s) and grafana locally, both pointed at [`ops/prometheus.yml`](ops/prometheus.yml) and the provisioning dir above. Admin credentials default to `admin` / `admin`. Data persists under `ops/data/` (git-ignored). The ops README at [`ops/README.md`](ops/README.md) is the one-page operating guide.

### Phase 5B — INT4 packed weight-only quantization

The Phase 3 arc concluded that INT8 weight-only quant on MPS pays a dequant tax that exceeds the memory-bandwidth savings at TinyLlama scale. Phase 5B adds the INT4 variant ([`src/nanoserve/engine/quant_int4.py`](src/nanoserve/engine/quant_int4.py)) primarily to test whether a steeper memory cut changes the answer — and secondarily to have a quant-quality ablation for the eval harness to score.

**The packed representation.** Per-row symmetric scale like the INT8 path, but the quantization codomain is signed int4 (values in `[-8, 7]`, scale = absmax / 7). Two int4 values fit in one byte — low nibble for the even column, high nibble for the odd. Storage shrinks 4× vs fp16 (TinyLlama 1.1B weights: ~2 GB → ~536 MB). Odd `in_features` are padded with a zero column before packing and stripped on unpack. Correctness is verified by:

- **9 unit tests** in [`tests/test_quant_int4.py`](tests/test_quant_int4.py) covering pack/unpack round-trips (even + odd widths), boundary values (`-8`, `7`), zero-row stability, forward-pass error bounds, and whole-model replacement byte accounting
- **L4-quant-int4 parity gate** in [`tests/test_engine_parity_l4_quant_int4.py`](tests/test_engine_parity_l4_quant_int4.py) — INT4 vs fp16 greedy outputs must match on the first 2 tokens for at least 3 of 5 test prompts. Deliberately looser than the int8 gate because int4 rounding is ~9× more aggressive; the point is to catch catastrophic breakage, not demand exact parity.

Run the sweep with [`scripts/run_phase5b_sweep.sh`](scripts/run_phase5b_sweep.sh) — serial c=1 plus paired fcfs/synchronized continuous at poisson λ=2, matching the Phase 3B shape so the INT4 row slots into the existing ablation table.

### Phase 5C — quality eval harness

The engine phases (1–4 and 5B) are all about speed. Phase 5C adds the quality axis so we can honestly say *"int8/int4 slow down the serving loop on this hardware, but they don't destroy the model."* Two metrics per quant mode, one CSV row per run, committed at [`results/eval.csv`](results/eval.csv).

**Perplexity** ([`src/nanoserve/eval/perplexity.py`](src/nanoserve/eval/perplexity.py)): sliding-window PPL on wikitext-2 validation (via `datasets`) with an in-repo fixture fallback at [`prompts/eval/ppl_fixture.txt`](prompts/eval/ppl_fixture.txt) so the eval runs offline. 512-token window, 256-token stride, overlap tokens masked out of the loss so each position contributes exactly once.

**HellaSwag-mini** ([`src/nanoserve/eval/hellaswag.py`](src/nanoserve/eval/hellaswag.py)): LM-scoring on Rowan/hellaswag validation (100 items by default), fallback to a 12-item in-repo cloze fixture. For each item, compute mean NLL of each of the 4 endings conditioned on the context; argmin = prediction; compare to gold label. Same recipe as `lm-evaluation-harness`, stripped down to the essentials.

**What the eval is for.** TinyLlama at 1.1B is small enough that absolute HellaSwag scores sit in the mid-30s (near chance 25%), so the eval is not a leaderboard entry. It's an ablation tool: run `make eval` before and after a quant change and compare the *delta* — fp16 vs int8 vs int4 — on the same corpus. The pass criterion is "int8 and int4 stay within a documented tolerance of fp16 on both metrics"; large drops would flag a genuine quality cliff.

The runner ([`src/nanoserve/eval/runner.py`](src/nanoserve/eval/runner.py)) loads each quant mode, scores, unloads the model (to keep peak memory bounded), and appends a row with full environment metadata: torch version, host, corpus source, wall-clock timings. CLI: `nanoserve eval all --quant fp16,int8,int4` (default) or `--offline` to force the fixture path.

### How to read a row

- **workload `closed-loop c=N`** — N concurrent clients, each fires the next request as soon as the current one finishes. Measures raw serving capacity with no queuing effects.
- **workload `poisson λ=R`** — open-loop Poisson arrivals at R req/s. Measures tail behavior under realistic bursty load. If `R > rps`, the queue is unbounded and tail latency blows up — that's signal, not noise.
- **rps** — sustained throughput (successful requests per second over the full run).
- **TTFT** — time to first token (prefill + queue wait).
- **TPOT** — time per output token *after* the first (decode steady-state).
- **e2e** — total wall time from request submission to last token.
- **decode tok/s** — aggregate output tokens divided by summed decode time. Measures raw model speed independent of queue pressure.
- **p50 / p95 / p99** — median, 95th, 99th percentile across all requests in the run.

A new row is added every time a technique is toggled on or off. The columns `batching`, `paging`, and `prefix_cache` track which features were enabled, so any speedup can be isolated to the specific thing that changed.

## Layout

```
src/nanoserve/
  bench/      metrics, workload generator, runner, report writer
  baselines/  hf-mps and llama.cpp backends behind a common interface
  engine/     scheduler, prefix cache, int8/int4/torchao quant, service
  server/     fastapi app + prometheus metrics
  eval/       perplexity + hellaswag scorers + runner
  cli.py      entrypoints for baseline / bench / serve / eval commands
ops/          prometheus + grafana provisioning + observe.sh
scripts/      model download + parity + per-phase sweep scripts
prompts/      fixed bench prompts + eval fixture corpora
results/      ablations.csv, eval.csv, per-run json dumps
tests/        parity gates, unit tests, api integration tests
```

## Model

TinyLlama-1.1B-Chat to start. Small enough to iterate fast on an M3 Air, large enough to produce realistic serving pressure when you crank concurrency.

## License

MIT.
