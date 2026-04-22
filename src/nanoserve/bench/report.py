import csv
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from nanoserve.bench.metrics import AggregateMetrics, RequestRecord
from nanoserve.config import RESULTS_DIR

ABLATION_HEADERS = [
    "ts",
    "commit",
    "backend",
    "model",
    "quant",
    "batching",
    "paging",
    "prefix_cache",
    "workload",
    "rate",
    "concurrency",
    "n",
    "rps",
    "p50_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "p50_tpot_ms",
    "p95_tpot_ms",
    "p50_e2e_ms",
    "p95_e2e_ms",
    "p99_e2e_ms",
    "decode_tok_s",
    "mem_peak_mb",
]


def git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def env_snapshot() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def ensure_dirs() -> None:
    (RESULTS_DIR / "runs").mkdir(parents=True, exist_ok=True)


def dump_run(
    tag: str,
    config: dict,
    records: list[RequestRecord],
    agg: AggregateMetrics,
    mem_peak_mb: float,
) -> Path:
    ensure_dirs()
    ts = now_iso()
    run_path = RESULTS_DIR / "runs" / f"{ts}_{tag}.json"
    payload = {
        "ts": ts,
        "commit": git_commit(),
        "env": env_snapshot(),
        "config": config,
        "aggregate": agg.as_dict(),
        "mem_peak_mb": mem_peak_mb,
        "records": [asdict(r) for r in records],
    }
    with run_path.open("w") as f:
        json.dump(payload, f, indent=2)
    return run_path


def append_ablation_row(row: dict) -> Path:
    ensure_dirs()
    csv_path = RESULTS_DIR / "ablations.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ABLATION_HEADERS)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in ABLATION_HEADERS})
    return csv_path
