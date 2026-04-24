#!/usr/bin/env bash
# phase 5b int4 sweep: paired closed-loop and poisson runs with int4
# weight-only quant. same harness as phase 3, new quant_mode flag.
# expected outcome on MPS: same dequant-tax story as int8 but more
# memory saved, throughput still < fp16. we want the numbers on paper.
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

# serial baseline — the "platform tax" row, no batching story yet.
run --batching-mode serial --max-batch-size 1 --quant-mode int4 \
    --workload closed-loop --rate 2.0 \
    --concurrency 1 --num-requests "$N" --max-new-tokens "$MAX_NEW"

# continuous c=4 fcfs vs synchronized, same pair shape as phase 3b.
run --batching-mode continuous --max-batch-size 4 --quant-mode int4 \
    --admission-policy fcfs \
    --workload poisson --rate 2.0 \
    --concurrency 4 --num-requests "$N" --max-new-tokens "$MAX_NEW"

run --batching-mode continuous --max-batch-size 4 --quant-mode int4 \
    --admission-policy synchronized \
    --workload poisson --rate 2.0 \
    --concurrency 4 --num-requests "$N" --max-new-tokens "$MAX_NEW"

echo "done. see results/ablations.csv"
