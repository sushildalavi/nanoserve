# ops

local prometheus + grafana wired to the nanoserve api server.

## one-time setup

```bash
brew install prometheus grafana
```

## start the stack

in three terminals:

```bash
# terminal 1 — the api server
make serve

# terminal 2 — prometheus + grafana (foreground, ctrl-c kills both)
make observe

# terminal 3 — fire some requests so the panels light up
curl -s localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"tinyllama-1.1b-chat","messages":[{"role":"user","content":"hi"}],"max_tokens":32}'
```

then open:

- grafana: <http://localhost:3000> (login: `admin` / `admin`)
- dashboard: <http://localhost:3000/d/nanoserve/nanoserve>
- prometheus ui: <http://localhost:9090>

the dashboard auto-provisions from [`grafana/dashboards/nanoserve.json`](grafana/dashboards/nanoserve.json). edits you make in the UI are saved back to grafana's internal db, not to this file — export JSON and overwrite the file if you want them tracked in git.

## what's on the dashboard

- **serving** — rps, error rate, active seqs, batched-forward fraction
- **latency** — TTFT / TPOT / e2e as p50 / p95 / p99 timeseries
- **throughput** — input vs output token rates, requests by status
- **prefix cache** — hit rate, cache size, hits-vs-misses timeseries

## notes

- prometheus scrapes `127.0.0.1:8000/metrics` every 5s. if `make serve` isn't running, the nanoserve job goes red in the prom targets page — that's expected.
- the gauges (`active_seqs`, `batched_forward_frac`, `prefix_cache_size`) refresh on each scrape via [`refresh_gauges_from_engine`](../src/nanoserve/server/metrics.py). they show point-in-time state, not accumulated values.
- grafana + prometheus data live in `ops/data/` (git-ignored). delete that dir to reset.
- admin credentials are set via env in `observe.sh` and default to `admin` / `admin`. rotate if you expose the port.
