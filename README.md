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
- [ ] continuous batching scheduler
- [ ] paged kv + prefix cache
- [ ] prometheus + grafana
- [ ] int8 / int4 quant + eval harness

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

| backend   | quant  | batching | workload         | rps  | p50 TTFT   | p95 TTFT | p95 e2e  | decode tok/s |
|-----------|--------|----------|------------------|------|------------|----------|----------|--------------|
| hf-mps    | fp16   | off      | closed-loop c=1  | 0.18 | **135 ms** | 253 ms   | 6.4 s    | 21.8         |
| hf-mps    | fp16   | off      | poisson λ=2      | 0.17 | 247 s      | 489 s    | 495 s    | 21.5         |
| llama.cpp | Q8_0   | n/a      | poisson λ=2      | 1.43 | 5.0 s      | 9.9 s    | 12.7 s   | 47.5         |
| llama.cpp | Q4_K_M | n/a      | poisson λ=2      | 1.16 | 12.3 s     | 22.9 s   | 26.3 s   | 39.1         |

The two HF-MPS rows are the story. Raw forward-pass TTFT on an idle engine is **135 ms** — totally reasonable. Under 2 req/s Poisson arrivals the same engine has p50 TTFT of **247 seconds**, a 1,800× jump. Decode tok/s is identical across both rows (~21.8). Nothing about the model got slower — the engine just can't serve requests in parallel, so the queue grows linearly and by request 100 you're waiting four minutes behind 99 others. This is exactly the bottleneck continuous batching kills in Phase 2.

llama.cpp Q8_0 on Metal is the honest native ceiling at ~47 decode tok/s. That's what nanoserve has to approach, not beat.

Q4_K_M is slower than Q8_0 here because the model is too small (1.1B) to be memory-bandwidth bound, so the extra dequant work dominates. At 7B+ the order typically flips.

Raw artifacts live in [`results/ablations.csv`](results/ablations.csv) and [`results/runs/`](results/runs/) (full per-request records + env).

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
