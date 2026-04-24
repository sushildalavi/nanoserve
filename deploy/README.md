# deploy/

containerization + kubernetes scaffolding.

nanoserve is an Apple-Silicon-native serving engine — the primary deployment target is your laptop via `make serve`. The artifacts in this directory exist so a reviewer can see the production-shape layer, not because the project expects to run in a GPU cluster. The container runs CPU-only (MPS is macOS-only; Linux containers can't reach it), so expect ~1–2 tok/s on TinyLlama inside the container. Use it for CI / portability / interview demos, not performance.

## local stack via docker compose

```bash
docker compose up --build
# api:        http://localhost:8000
# prometheus: http://localhost:9090
# grafana:    http://localhost:3000   (admin / admin → dashboards → nanoserve)
```

This brings up three containers: `nanoserve` (the CPU-mode api), `prometheus` (scraping the container on port 8000), and `grafana` (provisioning the same dashboard you get from `make observe`). First request takes ~30s because the pod is downloading TinyLlama weights into the `hf-cache` volume; subsequent calls stay in-cache.

Stop with `docker compose down`; add `-v` to also delete the weight cache and Grafana db.

## kubernetes

```bash
# build and load into your local cluster (kind/minikube/Docker-Desktop)
docker build -t nanoserve:latest .

# render the full manifest set
kubectl kustomize deploy/k8s

# apply (with a live cluster)
kubectl apply -k deploy/k8s
kubectl -n nanoserve get all
```

### what's here

| file                   | what it is                                                                     |
|------------------------|--------------------------------------------------------------------------------|
| `namespace.yaml`       | `nanoserve` namespace                                                          |
| `configmap.yaml`       | `NANOSERVE_*` env vars the engine reads on startup                             |
| `deployment.yaml`      | 1 replica, non-root uid 10001, startup + liveness + readiness probes on `/health`, resource requests/limits, emptyDir for the HF weight cache, pod annotations for annotation-based Prometheus scraping |
| `service.yaml`         | ClusterIP exposing `http` (80→8000) and a second `metrics` port for Prometheus |
| `hpa.yaml`             | v2 HPA targeting 70% CPU, 1–4 replicas, conservative scale-down window so a scale-down doesn't kill a pod that's still loading weights |
| `servicemonitor.yaml`  | optional — Prometheus-Operator CRD. skip if the operator isn't installed (the Deployment's `prometheus.io/scrape` annotations handle static-config scraping on their own) |
| `kustomization.yaml`   | kustomize entrypoint; does not include the ServiceMonitor by default          |

### where the scaffolding is deliberately thin

- **No Ingress / service mesh**. Production would add an Ingress + TLS termination or wire into Istio; that's orthogonal to the engine and cluster-dependent.
- **No secrets management**. TinyLlama weights are public, so nothing sensitive; in a real deployment, `HF_TOKEN` would live in a Secret mounted as an env var.
- **No NetworkPolicy**. Sensible default would be egress-only to HuggingFace + cluster-internal ingress; again, cluster-policy dependent.
- **Custom-metrics HPA (RPS per pod) commented as a follow-on**. Needs Prometheus-Adapter in the cluster; the CPU-based HPA works without any additional operators.

### things that intentionally do not appear here

- **GPU node selectors / tolerations.** The container is CPU-only by design (see top of this doc). On a CUDA cluster, a separate Dockerfile + node selector would be added; the engine already speaks `cuda` in principle (HF transformers handles the device string), but that's unexercised.
- **Ray Serve / gRPC wrappers.** The OpenAI-compatible HTTP API is the contract. Ray Serve would be a routing layer on top; gRPC would be a second protocol. Either adds surface area without changing what the engine does. Portfolio-wise, the engine is the artifact.
