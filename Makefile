.PHONY: help install dev-install fmt lint test clean models baseline-hf baseline-llamacpp bench parity serve observe eval docker-build docker-up docker-down k8s-render k8s-apply

PY ?= python3.11
VENV ?= .venv
PIP = $(VENV)/bin/pip
PYV = $(VENV)/bin/python

help:
	@echo "targets:"
	@echo "  install            create venv and install runtime deps"
	@echo "  dev-install        runtime + dev deps"
	@echo "  fmt                ruff format"
	@echo "  lint               ruff check"
	@echo "  test               pytest"
	@echo "  models             download tinyllama hf + gguf"
	@echo "  baseline-hf        run hf-mps baseline"
	@echo "  baseline-llamacpp  run llama.cpp baseline"
	@echo "  parity             run parity test across backends"
	@echo "  bench              run full baseline sweep"
	@echo "  serve              start the openai-compatible api server on :8000"
	@echo "  observe            start local prometheus + grafana (brew) pointed at :8000/metrics"
	@echo "  eval               run the quality eval harness (perplexity + hellaswag-mini)"
	@echo "  docker-build       build the cpu-only nanoserve container image"
	@echo "  docker-up          docker compose up --build (api + prom + grafana stack)"
	@echo "  docker-down        stop and remove the compose stack"
	@echo "  k8s-render         kubectl kustomize deploy/k8s (no cluster required)"
	@echo "  k8s-apply          kubectl apply -k deploy/k8s (requires a live cluster)"

$(VENV):
	$(PY) -m venv $(VENV)

install: $(VENV)
	$(PIP) install -U pip
	$(PIP) install -e .

dev-install: $(VENV)
	$(PIP) install -U pip
	$(PIP) install -e ".[dev]"

fmt:
	$(VENV)/bin/ruff format src tests

lint:
	$(VENV)/bin/ruff check src tests

test:
	$(VENV)/bin/pytest

models:
	bash scripts/download_models.sh

baseline-hf:
	$(PYV) -m nanoserve.cli baseline hf

baseline-llamacpp:
	$(PYV) -m nanoserve.cli baseline llamacpp

parity:
	$(PYV) scripts/parity_test.py

bench:
	$(PYV) -m nanoserve.cli bench sweep

serve:
	$(PYV) -m nanoserve.cli serve --host 127.0.0.1 --port 8000

observe:
	bash ops/observe.sh

eval:
	$(PYV) -m nanoserve.cli eval all

docker-build:
	docker build -t nanoserve:latest .

docker-up:
	docker compose up --build

docker-down:
	docker compose down

k8s-render:
	kubectl kustomize deploy/k8s

k8s-apply:
	kubectl apply -k deploy/k8s

clean:
	rm -rf build dist *.egg-info .ruff_cache .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
