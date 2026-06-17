# Portfolio Proof

## What the project does

Nanoserve is an Apple Silicon focused local LLM serving engine with continuous batching, prefix caching, quantization paths, and an OpenAI-compatible API.

## Why it is technically impressive

- Multiple backends and quantization modes are benchmarked in repo artifacts.
- The project targets local Mac inference rather than cloud-only serving.
- It includes evaluation artifacts for both throughput and task quality.

## Architecture summary

- Request queue -> scheduler -> batch builder -> model engine -> streaming response.
- Metrics and eval outputs are recorded under `results/`.

## How to run locally

- `make dev-install`
- `make models`
- `make baseline-hf`
- `make parity`
- `make serve`
- `make observe`
- `make eval`

## How to test

- `pytest`
- `python -m compileall src`
- Any parser or client smoke tests added under `tests/`

## How to benchmark or evaluate

- Review `results/ablations.csv`
- Review `results/eval.csv`

## Verified metrics only

- MLX int4 throughput: 136.26 tok/s
- MLX fp16 throughput: 36.92 tok/s
- HF MPS fp16 throughput: 21.82 tok/s
- fp16 MPS perplexity: 6.8695
- fp16 MPS HellaSwag accuracy: 0.5800

## Current limitations

- Backend availability varies by hardware and installed Apple Silicon runtimes.
- Benchmark matrix tooling now parses repo artifacts and can append an optional live row when the server is already running.

## Future improvements

- Add a tiny OpenAI-compatible client example and smoke test.
- Add scheduler and batching documentation for portfolio review.

## Resume bullets

- Built a local Apple Silicon LLM serving engine with continuous batching and prefix cache support.
- Benchmarked quantized and non-quantized inference paths with verified throughput and quality metrics.
- Exposed an OpenAI-compatible serving interface for local developer workflows.

## Verification Log

- `python3 -m pytest /Users/sushildalavi/Desktop/Github/nanoserve/tests/test_benchmark_matrix.py` - pass - 2026-06-16 - Verified benchmark matrix parser and Markdown renderer.
- `python3 -m compileall /Users/sushildalavi/Desktop/Github/nanoserve/scripts/benchmark_matrix.py` - pass - 2026-06-16 - Verified script syntax.
- `python3 /Users/sushildalavi/Desktop/Github/nanoserve/scripts/benchmark_matrix.py --output /tmp/nanoserve_benchmark_matrix.md` - pass - 2026-06-16 - Rendered the matrix to a Markdown artifact.
- `python3 /Users/sushildalavi/Desktop/Github/nanoserve/scripts/benchmark_matrix.py --output /tmp/nanoserve_benchmark_matrix.md --output-json /tmp/nanoserve_benchmark_matrix.json` - pass - 2026-06-17 - Verified the new JSON + Markdown artifact path.
