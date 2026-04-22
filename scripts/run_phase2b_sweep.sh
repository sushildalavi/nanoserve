#!/usr/bin/env bash
# phase 2b poisson sweep: run nanoserve across rates to fill the ablation
# table. outputs go to results/ablations.csv (appended) and the full per-run
# json dumps land in results/runs/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

N=${N:-40}
MAX_NEW=${MAX_NEW:-64}
PY=${PY:-.venv/bin/python}

run() {
  echo "==> $*"
  $PY -m nanoserve.cli baseline nanoserve "$@"
}

# serial baseline at lambda=2
run --batching-mode serial --max-batch-size 1 \
    --workload poisson --rate 2.0 \
    --concurrency 4 --num-requests "$N" --max-new-tokens "$MAX_NEW"

# continuous sweep
for RATE in 1.0 2.0 4.0; do
  run --batching-mode continuous --max-batch-size 4 \
      --workload poisson --rate "$RATE" \
      --concurrency 4 --num-requests "$N" --max-new-tokens "$MAX_NEW"
done

echo "done. see results/ablations.csv"
