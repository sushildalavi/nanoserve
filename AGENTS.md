# NanoServe — Project AGENTS.md

From-scratch LLM inference server for Apple Silicon (M-series MPS).
Portfolio project targeting SWE/SDE/ML generalist roles.

---

## What This Is

OpenAI-compatible LLM serving engine built from scratch with:
- Continuous batching (iteration-level)
- Prefix KV-cache
- INT8/INT4 quantization (PyTorch torchao + packed int4)
- MLX backend (mlx-lm) for Apple Silicon
- Prometheus + Grafana observability
- Realistic Poisson load generator with p50/p95/p99 latency reporting

Model: TinyLlama-1.1B on MPS. Baseline: llama.cpp via subprocess.

---

## Architecture

```
src/nanoserve/
  engine/        — core decode loop, scheduler, prefix cache, quant
  server/        — FastAPI app, SSE streaming, Prometheus metrics
  bench/         — workload gen, load runner, metrics aggregation
  baselines/     — hf_mps, llama_cpp_bin, nanoserve_engine adapters
  eval/          — perplexity + hellaswag-mini scorers, mlx eval
  mlx_engine/    — MLX inference backend
  cli.py         — typer CLI (nanoserve bench / serve / eval)
  config.py      — model paths, device config
```

---

## Key Commands

```bash
# install
pip install -e ".[dev,eval,mlx]"

# tests
pytest                           # all unit tests
pytest tests/test_engine_parity_l3.py  # specific level

# benchmark sweep (runs require local model files)
scripts/run_phase6a_sweep.sh     # mlx fp16 + int4 vs pytorch
make bench                       # see Makefile for targets

# observability stack (brew prometheus + grafana)
make observe

# eval (perplexity + hellaswag)
make eval                        # writes results/eval.csv

# lint
ruff check src tests
```

---

## Scope — What's In vs Out

**In scope (shipped):**
- Continuous batching, prefix cache, INT8/INT4 quant
- OpenAI-compatible `/v1/completions` + `/v1/chat/completions` SSE
- Prometheus metrics + Grafana dashboard (`make observe`)
- llama.cpp baseline comparison
- MLX fp16 + int4 backend (Apple Silicon only)

**Out of scope (explicitly):**
- True paged KV / custom Metal attention kernels
- Kubernetes / Ray Serve / gRPC / auth / multi-tenancy
- vLLM benchmarking (CUDA-only)
- General model support beyond TinyLlama-1.1B for benchmarking

---

## Results Files

- `results/eval.csv` — perplexity + hellaswag by quant level (source of truth)
- `results/ablations.csv` — throughput/latency ablation rows
- `results/runs/` — raw per-run JSONs (gitignored; regenerate via sweep scripts)

---

## Dev Notes

- All benchmark runs need local model files in `models/` (gitignored).
- MLX eval requires `pip install -e ".[mlx]"` and Apple Silicon.
- `scripts/parity_test.py` is a manual sanity check for output correctness.
- Phase sweep scripts (`scripts/run_phase*.sh`) document experiment methodology; don't delete them.
- Tests use `pytest-asyncio`; `asyncio_mode = "auto"` is set in pyproject.toml.
