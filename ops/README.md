# ops

Local Prometheus and Grafana wiring for nanoserve.

## Setup

```bash
brew install prometheus grafana
```

## Start

```bash
make serve
make observe
```

Open Grafana at `http://localhost:3000` and the nanoserve dashboard from there.

## Dashboard sections

- Serving metrics
- Latency metrics
- Throughput metrics
- Prefix cache metrics

## Notes

- Prometheus scrapes the API every 5 seconds.
- Grafana and Prometheus data live in the local ops data directory.
- Dashboard edits stay in Grafana until you export them.
