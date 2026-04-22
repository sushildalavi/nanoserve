#!/usr/bin/env bash
# phase 3 int8 sweep: re-run the phase 2b workload with int8 weight-only
# quantization to test whether memory savings let batching pay off.
# paired rows: fp16 serial, fp16 continuous, int8 serial, int8 continuous
# at two load points.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

N=${N:-30}
MAX_NEW=${MAX_NEW:-64}
PY=${PY:-.venv/bin/python}

run() {
  echo "==> $*"
  $PY -m nanoserve.cli baseline nanoserve "$@"
}

# closed-loop pair at c=4 (same as phase 2a headline row, +int8 variant)
run --batching-mode serial --max-batch-size 1 --quant-mode int8 \
    --workload closed-loop --rate 2.0 \
    --concurrency 1 --num-requests "$N" --max-new-tokens "$MAX_NEW"

run --batching-mode continuous --max-batch-size 4 --quant-mode int8 \
    --workload closed-loop --rate 2.0 \
    --concurrency 4 --num-requests "$N" --max-new-tokens "$MAX_NEW"

# poisson at lambda=2 with int8, serial + continuous
run --batching-mode serial --max-batch-size 1 --quant-mode int8 \
    --workload poisson --rate 2.0 \
    --concurrency 4 --num-requests "$N" --max-new-tokens "$MAX_NEW"

run --batching-mode continuous --max-batch-size 4 --quant-mode int8 \
    --workload poisson --rate 2.0 \
    --concurrency 4 --num-requests "$N" --max-new-tokens "$MAX_NEW"

echo "done. see results/ablations.csv"
