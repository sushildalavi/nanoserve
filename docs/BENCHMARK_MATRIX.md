# Benchmark Matrix

## Verified results from repo artifacts

| backend | quantization | concurrency | metric | value |
| --- | --- | --- | --- | --- |
| MLX | int4 | 1 | tokens/sec | 136.26 |
| MLX | fp16 | 1 | tokens/sec | 36.92 |
| HF MPS | fp16 | 1 | tokens/sec | 21.82 |
| nanoserve | fp16 | 1 | perplexity | 6.8695 |
| nanoserve | fp16 | 1 | HellaSwag accuracy | 0.5800 |

## Pending matrix rows

- HF MPS int8
- MLX int8
- concurrency 2, 4, 8
- llama.cpp if and when supported locally

## Notes

- The repo already contains `results/ablations.csv` and `results/eval.csv`.
- Any future benchmark row must come from parsed artifacts or an actual local run.
- `scripts/benchmark_matrix.py` can now append a live row from a running local server with `--live-url`.
