#!/usr/bin/env bash
# phase 6a mlx sweep: fp16 + int4 on apple's native ml framework.
# the int4 row is what the sweep is really for — MLX has native Metal
# int4 matmul, which pytorch-MPS does not. this is the same model at
# the same quant level through a different runtime, so the delta vs the
# pytorch int4 row in ablations.csv is the platform tax.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

N=${N:-30}
MAX_NEW=${MAX_NEW:-64}
PY=${PY:-.venv/bin/python}

run() {
  echo "==> $*"
  $PY -m nanoserve.cli baseline mlx "$@"
}

run --quant-mode fp16 \
    --workload closed-loop --concurrency 1 \
    --num-requests "$N" --max-new-tokens "$MAX_NEW"

run --quant-mode int4 \
    --workload closed-loop --concurrency 1 \
    --num-requests "$N" --max-new-tokens "$MAX_NEW"

echo "done. see results/ablations.csv"
