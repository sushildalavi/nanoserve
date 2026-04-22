"""FastAPI app exposing nanoserve over an OpenAI-compatible endpoint.

design:
- single shared NanoServeEngine, started at app startup, stopped at shutdown
- POST /v1/chat/completions: openai-compatible. supports stream=true (SSE)
  and stream=false (single JSON response). greedy decode only.
- GET /health: liveness + engine-ready probe
- GET /metrics: prometheus text format

the engine's submit/stream interface (from EngineService, designed in
phase 2 day 1) is exactly what the api needs — no adapter logic, the
engine was built for this layering from the start.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse

from nanoserve.config import tinyllama_nanoserve
from nanoserve.engine.engine import NanoServeEngine
from nanoserve.engine.service import SubmitRequest
from nanoserve.server import metrics
from nanoserve.server.schemas import (
    ChatCompletion,
    ChatCompletionRequest,
    CompletionChoice,
    CompletionMessage,
    Delta,
    StreamChoice,
    StreamChunk,
    Usage,
    new_id,
    now_ts,
)


def _engine_from_env() -> NanoServeEngine:
    """build the engine from env vars so deployments can flag-flip without
    code changes. defaults match the phase-3 best config: continuous +
    synchronized + fp16 + max_batch=4 + prefix cache 16. all knobs the
    bench harness exposes are exposed here too.
    """
    spec = tinyllama_nanoserve(
        batching_mode=os.environ.get("NANOSERVE_BATCHING_MODE", "continuous"),
        max_batch_size=int(os.environ.get("NANOSERVE_MAX_BATCH_SIZE", "4")),
        quant_mode=os.environ.get("NANOSERVE_QUANT_MODE", "none"),
        admission_policy=os.environ.get(
            "NANOSERVE_ADMISSION_POLICY", "synchronized"
        ),
        prefix_cache_capacity=int(
            os.environ.get("NANOSERVE_PREFIX_CACHE_CAPACITY", "16")
        ),
    )
    return NanoServeEngine(
        spec,
        batching_mode=spec.batching_mode,
        max_batch_size=spec.max_batch_size,
        quant_mode=spec.quant_mode,
        admission_policy=spec.admission_policy,
        prefix_cache_capacity=spec.prefix_cache_capacity,
    )


# the live engine. lifespan-managed. tests can patch this.
ENGINE: NanoServeEngine | None = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global ENGINE
    ENGINE = _engine_from_env()
    await ENGINE.start()
    try:
        yield
    finally:
        if ENGINE is not None:
            await ENGINE.stop()
            ENGINE = None


app = FastAPI(title="nanoserve", lifespan=_lifespan)


@app.get("/health")
async def health():
    if ENGINE is None or ENGINE._tokenizer is None:
        raise HTTPException(status_code=503, detail="engine not ready")
    return {
        "status": "ok",
        "engine": {
            "batching_mode": ENGINE._scheduler.cfg.batching_mode,
            "max_batch_size": ENGINE._scheduler.cfg.max_batch_size,
            "admission_policy": ENGINE._scheduler.cfg.admission_policy,
            "quant_mode": ENGINE.quant_mode,
            "prefix_cache_enabled": ENGINE.prefix_cache is not None,
        },
    }


@app.get("/metrics")
async def get_metrics():
    metrics.refresh_gauges_from_engine(ENGINE)
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)


def _build_prompt(messages) -> str:
    """concat openai-style messages into one prompt the chat template can
    handle. for nanoserve we collapse system + user into a single user
    message because TinyLlama's chat template does its own role wrapping
    and we want the prefix cache to see a stable token sequence.
    """
    parts = []
    for m in messages:
        if m.role == "system":
            parts.append(m.content.strip())
        elif m.role == "user":
            parts.append(m.content.strip())
        elif m.role == "assistant":
            # in a real multi-turn impl we'd wrap with <|assistant|>, but
            # for now we just append the content as if it were context.
            parts.append(m.content.strip())
    return "\n\n".join(p for p in parts if p)


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if ENGINE is None or ENGINE._tokenizer is None:
        raise HTTPException(status_code=503, detail="engine not ready")
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    prompt = _build_prompt(req.messages)
    submit_ts = time.time()
    seq_id = await ENGINE.submit(
        SubmitRequest(prompt=prompt, max_new_tokens=req.max_tokens)
    )

    if req.stream:
        return StreamingResponse(
            _stream_response(seq_id, req.model, submit_ts),
            media_type="text/event-stream",
        )
    return await _full_response(seq_id, req.model, submit_ts, prompt)


async def _stream_response(seq_id: int, model: str, submit_ts: float) -> AsyncIterator[bytes]:
    """SSE generator. emits openai-shaped chunks one per produced token."""
    cmpl_id = new_id()
    created = now_ts()
    first_token_ts: float | None = None
    last_token_ts: float | None = None
    n_out = 0
    stop_reason: str | None = None

    # opening chunk with the role (some clients require this).
    open_chunk = StreamChunk(
        id=cmpl_id,
        created=created,
        model=model,
        choices=[StreamChoice(delta=Delta(role="assistant"))],
    )
    yield _sse(open_chunk)

    try:
        async for ev in ENGINE.stream(seq_id):
            if first_token_ts is None and ev.token_text:
                first_token_ts = time.time()
                metrics.ttft_seconds.observe(first_token_ts - submit_ts)
            if ev.token_text:
                last_token_ts = time.time()
                n_out += 1
                chunk = StreamChunk(
                    id=cmpl_id,
                    created=created,
                    model=model,
                    choices=[StreamChoice(delta=Delta(content=ev.token_text))],
                )
                yield _sse(chunk)
            if ev.done:
                stop_reason = ev.stop_reason
                break

        finish_chunk = StreamChunk(
            id=cmpl_id,
            created=created,
            model=model,
            choices=[
                StreamChoice(
                    delta=Delta(),
                    finish_reason=_map_finish(stop_reason),
                )
            ],
        )
        yield _sse(finish_chunk)
        yield b"data: [DONE]\n\n"
    finally:
        ENGINE.retire(seq_id)
        _record_request_metrics(
            submit_ts, first_token_ts, last_token_ts, n_out, ok=stop_reason is not None
        )


async def _full_response(
    seq_id: int, model: str, submit_ts: float, prompt: str
) -> ChatCompletion:
    pieces: list[str] = []
    first_token_ts: float | None = None
    last_token_ts: float | None = None
    stop_reason: str | None = None
    try:
        async for ev in ENGINE.stream(seq_id):
            if ev.token_text and first_token_ts is None:
                first_token_ts = time.time()
                metrics.ttft_seconds.observe(first_token_ts - submit_ts)
            if ev.token_text:
                pieces.append(ev.token_text)
                last_token_ts = time.time()
            if ev.done:
                stop_reason = ev.stop_reason
                break
        text = "".join(pieces)
        prompt_tokens = ENGINE.count_tokens(prompt)
        completion_tokens = ENGINE.count_tokens(text)
        return ChatCompletion(
            id=new_id(),
            created=now_ts(),
            model=model,
            choices=[
                CompletionChoice(
                    index=0,
                    message=CompletionMessage(content=text),
                    finish_reason=_map_finish(stop_reason),
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
    finally:
        ENGINE.retire(seq_id)
        _record_request_metrics(
            submit_ts, first_token_ts, last_token_ts, len(pieces), ok=stop_reason is not None
        )


def _record_request_metrics(
    submit_ts: float,
    first_token_ts: float | None,
    last_token_ts: float | None,
    n_out: int,
    ok: bool,
) -> None:
    end_ts = last_token_ts or first_token_ts or time.time()
    metrics.requests_total.labels(status="ok" if ok else "error").inc()
    metrics.output_tokens_total.inc(n_out)
    metrics.e2e_seconds.observe(end_ts - submit_ts)
    if first_token_ts is not None and last_token_ts is not None and n_out > 1:
        # tpot = (last - first) / (n_out - 1)
        metrics.tpot_seconds.observe((last_token_ts - first_token_ts) / (n_out - 1))


def _sse(chunk: StreamChunk) -> bytes:
    return ("data: " + json.dumps(chunk.model_dump(exclude_none=True)) + "\n\n").encode()


def _map_finish(reason: str | None) -> str | None:
    if reason is None:
        return None
    if reason == "max_tokens":
        return "length"
    if reason == "eos":
        return "stop"
    return reason


# tests construct the engine directly without going through the lifespan.
# this hook lets them swap in a pre-built engine for in-process testing.
async def _test_inject_engine(engine: NanoServeEngine) -> None:
    global ENGINE
    ENGINE = engine


# also re-export asyncio so tests can `from nanoserve.server.api import asyncio`
__all__ = ["app", "asyncio"]
