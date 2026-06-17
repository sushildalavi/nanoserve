# deploy/

Container and Kubernetes scaffolding for nanoserve.

The main deployment target is still a local Apple Silicon machine. This directory exists so the project has a production-shaped deployment story and a portable container path for CI or demos.

## Docker Compose

```bash
docker compose up --build
```

## Kubernetes

```bash
docker build -t nanoserve:latest .
kubectl kustomize deploy/k8s
kubectl apply -k deploy/k8s
```

## What’s included

- Namespace and config
- Deployment and service
- Autoscaling
- Optional monitoring integration

## Deliberately omitted

- Ingress and service mesh
- Secrets management
- Network policy
- GPU-specific scheduling
- Ray Serve and gRPC wrappers
