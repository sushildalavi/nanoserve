# syntax=docker/dockerfile:1.7

# nanoserve container image. multi-stage, CPU-only torch (MPS is
# macOS-native; linux containers can't reach it). the resulting image
# is primarily a *portability* target — it proves the full stack can
# be built and served without Apple silicon — not a performance target.
# Expect ~1–2 tok/s on TinyLlama in this container.

FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# install build tooling only in the builder stage
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential git && \
    rm -rf /var/lib/apt/lists/*

# install torch (cpu wheels) first so the heavy download is cached
# independently of project source changes
RUN pip install --prefix=/install \
      --index-url https://download.pytorch.org/whl/cpu \
      "torch>=2.6,<2.9"

# project deps that don't need a gpu wheel index
COPY pyproject.toml ./
RUN pip install --prefix=/install \
      "transformers>=4.44" \
      "tokenizers>=0.19" \
      "sentencepiece>=0.2" \
      "accelerate>=0.33" \
      "numpy>=1.26" \
      "pydantic>=2.8" \
      "typer>=0.12" \
      "rich>=13.7" \
      "psutil>=6.0" \
      "httpx>=0.27" \
      "fastapi>=0.115" \
      "uvicorn>=0.32" \
      "prometheus-client>=0.21"


FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    NANOSERVE_BATCHING_MODE=serial \
    NANOSERVE_MAX_BATCH_SIZE=1 \
    NANOSERVE_QUANT_MODE=none

WORKDIR /app

# copy pre-built site-packages from the builder stage (no compilers in runtime)
COPY --from=builder /install /usr/local

# copy application source last so code-only changes don't bust the dep cache
COPY src/ ./src/
COPY pyproject.toml README.md ./
RUN pip install --no-deps --no-cache-dir -e .

# unprivileged user for prod hygiene
RUN useradd --system --uid 10001 --create-home --home-dir /home/serve nanoserve && \
    mkdir -p /app/.cache && \
    chown -R nanoserve:nanoserve /app /home/serve
USER nanoserve

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request as u, sys; \
                 sys.exit(0 if u.urlopen('http://127.0.0.1:8000/health', timeout=3).status==200 else 1)"

ENTRYPOINT ["python", "-m", "nanoserve.cli"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
