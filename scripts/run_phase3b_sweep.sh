#!/usr/bin/env bash
# phase 3b sweep: test whether synchronized admission lifts batched_forward_frac
# enough to make continuous batching pay off, with new forward_ms / step_overhead_ms
# timing rows so we can see where time actually went.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

N=${N:-20}
MAX_NEW=${MAX_NEW:-32}
PY=${PY:-.venv/bin/python}

run() {
  echo "==> $*"
  $PY -m nanoserve.cli baseline nanoserve "$@"
}

# fp16 paired comparison: fcfs vs synchronized
run --batching-mode continuous --max-batch-size 4 \
    --admission-policy fcfs \
    --workload poisson --rate 2.0 \
    --concurrency 4 --num-requests "$N" --max-new-tokens "$MAX_NEW"

run --batching-mode continuous --max-batch-size 4 \
    --admission-policy synchronized \
    --workload poisson --rate 2.0 \
    --concurrency 4 --num-requests "$N" --max-new-tokens "$MAX_NEW"

# int8 paired comparison: fcfs vs synchronized
run --batching-mode continuous --max-batch-size 4 --quant-mode int8 \
    --admission-policy fcfs \
    --workload poisson --rate 2.0 \
    --concurrency 4 --num-requests "$N" --max-new-tokens "$MAX_NEW"

run --batching-mode continuous --max-batch-size 4 --quant-mode int8 \
    --admission-policy synchronized \
    --workload poisson --rate 2.0 \
    --concurrency 4 --num-requests "$N" --max-new-tokens "$MAX_NEW"

echo "done. see results/ablations.csv"
