#!/usr/bin/env bash
# start a local prometheus + grafana pair wired to nanoserve's /metrics.
# both processes run in the foreground; ctrl-c kills both.
#
# assumes: `brew install prometheus grafana`.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data="$here/data"
mkdir -p "$data/prom" "$data/grafana" "$data/grafana/logs" "$data/grafana/plugins"

prom_bin="$(command -v prometheus)"
grafana_bin="$(command -v grafana-server || command -v grafana || true)"
if [[ -z "$grafana_bin" ]]; then
  echo "grafana not found on PATH. did you run: brew install grafana ?" >&2
  exit 1
fi

grafana_homepath="$(brew --prefix grafana)/share/grafana"
if [[ ! -d "$grafana_homepath" ]]; then
  echo "could not locate grafana homepath at $grafana_homepath" >&2
  exit 1
fi

echo "starting prometheus on :9090 ..."
"$prom_bin" \
  --config.file="$here/prometheus.yml" \
  --storage.tsdb.path="$data/prom" \
  --web.listen-address=127.0.0.1:9090 \
  --web.enable-lifecycle \
  >"$data/prom.log" 2>&1 &
prom_pid=$!

echo "starting grafana on :3000 ..."
export GF_PATHS_PROVISIONING="$here/grafana/provisioning"
export GF_PATHS_DATA="$data/grafana"
export GF_PATHS_LOGS="$data/grafana/logs"
export GF_PATHS_PLUGINS="$data/grafana/plugins"
export GF_SECURITY_ADMIN_USER=admin
export GF_SECURITY_ADMIN_PASSWORD=admin
export GF_SERVER_HTTP_PORT=3000
export GF_ANALYTICS_REPORTING_ENABLED=false
export GF_ANALYTICS_CHECK_FOR_UPDATES=false

if [[ "$(basename "$grafana_bin")" == "grafana" ]]; then
  "$grafana_bin" server --homepath="$grafana_homepath" >"$data/grafana.log" 2>&1 &
else
  "$grafana_bin" --homepath="$grafana_homepath" >"$data/grafana.log" 2>&1 &
fi
grafana_pid=$!

cleanup() {
  echo
  echo "stopping..."
  kill "$prom_pid" "$grafana_pid" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 2
echo
echo "  prometheus  http://127.0.0.1:9090"
echo "  grafana     http://127.0.0.1:3000  (admin / admin)"
echo "  dashboard   http://127.0.0.1:3000/d/nanoserve/nanoserve"
echo
echo "logs: $data/prom.log  $data/grafana.log"
echo "ctrl-c to stop."
wait
