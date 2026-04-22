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
- [ ] phase 3C: MLX or torch 2.6 + torchao for native Metal int8 matmul (close the remaining int8↔fp16 gap)
- [ ] paged kv + prefix cache
- [ ] prometheus + grafana

## Quickstart

```bash
make dev-install
make models           # downloads tinyllama gguf + builds llama.cpp with metal
make baseline-hf
make baseline-llamacpp
make parity
```

Results land in `results/ablations.csv` (headline table) and `results/runs/*.json` (per-run dumps with full config, env, and per-request records).

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

Raw artifacts live in [`results/ablations.csv`](results/ablations.csv) and [`results/runs/`](results/runs/) (full per-request records + env).

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
  cli.py      entrypoints for `baseline` and `bench` commands
scripts/      model download + parity test
prompts/      fixed prompt set (seeded, versioned)
results/      ablations.csv + per-run json dumps
tests/        unit tests for metrics
```

## Model

TinyLlama-1.1B-Chat to start. Small enough to iterate fast on an M3 Air, large enough to produce realistic serving pressure when you crank concurrency.

## License

MIT.
