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
- [ ] baseline numbers filled in `results/ablations.csv`
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

This is the whole point of the project. Every technique is added with its on/off row locked in the same day it ships, no retrofitting.

| backend   | quant  | batching | paging | prefix | workload   | rps | p50 ttft | p95 ttft | p99 e2e | tok/s | mem mb |
|-----------|--------|----------|--------|--------|------------|-----|----------|----------|---------|-------|--------|
| hf-mps    | fp16   | off      | off    | off    | poisson λ=2 | —   | —        | —        | —       | —     | —      |
| llama.cpp | Q8_0   | n/a      | n/a    | n/a    | poisson λ=2 | —   | —        | —        | —       | —     | —      |
| llama.cpp | Q4_K_M | n/a      | n/a    | n/a    | poisson λ=2 | —   | —        | —        | —       | —     | —      |

(Filled once Phase 1 runs on the M3.)

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
