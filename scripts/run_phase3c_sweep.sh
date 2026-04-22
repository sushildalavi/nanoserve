#!/usr/bin/env bash
# phase 3c prefix-cache sweep: same workload, varying shared prefix length and
# cache on/off. measures the TTFT/throughput improvement from skipping
# repeated prefill across many requests that share a system-prompt-style prefix.
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

# baseline pair: short shared prefix, cache off vs on
run --batching-mode serial --max-batch-size 1 \
    --prefix-cache-capacity 0 \
    --workload closed-loop --concurrency 4 \
    --num-requests "$N" --max-new-tokens "$MAX_NEW" \
    --shared-prefix-tokens 30

run --batching-mode serial --max-batch-size 1 \
    --prefix-cache-capacity 8 \
    --workload closed-loop --concurrency 4 \
    --num-requests "$N" --max-new-tokens "$MAX_NEW" \
    --shared-prefix-tokens 30

# pair under continuous + synchronized + heavier shared prefix
run --batching-mode continuous --max-batch-size 4 \
    --admission-policy synchronized \
    --prefix-cache-capacity 0 \
    --workload poisson --rate 2.0 --concurrency 4 \
    --num-requests "$N" --max-new-tokens "$MAX_NEW" \
    --shared-prefix-tokens 30

run --batching-mode continuous --max-batch-size 4 \
    --admission-policy synchronized \
    --prefix-cache-capacity 8 \
    --workload poisson --rate 2.0 --concurrency 4 \
    --num-requests "$N" --max-new-tokens "$MAX_NEW" \
    --shared-prefix-tokens 30

echo "done. see results/ablations.csv"
