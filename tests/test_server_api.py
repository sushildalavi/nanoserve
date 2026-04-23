"""end-to-end integration test for the FastAPI server.

we don't spin up uvicorn in the test loop — that's flaky. instead we use
httpx.AsyncClient against the FastAPI app directly via ASGITransport.
this exercises the full request/response path including json parsing,
validation, the engine.submit/stream wiring, and the SSE chunk format,
without the network layer.

the engine fixture is module-scoped so model load happens once.
"""
from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from nanoserve.config import tinyllama_nanoserve
from nanoserve.engine.engine import NanoServeEngine
from nanoserve.server import api as api_module


def _mps_available() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except ImportError:
        return False


import pytest_asyncio


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def shared_engine_and_client():
    """async fixture so the engine's driver thread captures the SAME loop
    that runs the test bodies. mismatched loops break call_soon_threadsafe
    silently — we hit that on the first attempt of this test module.
    """
    if not _mps_available():
        pytest.skip("mps not available on this host")

    spec = tinyllama_nanoserve(
        batching_mode="serial",
        max_batch_size=1,
        quant_mode="none",
        admission_policy="fcfs",
        prefix_cache_capacity=0,
    )
    engine = NanoServeEngine(
        spec,
        batching_mode="serial",
        max_batch_size=1,
        quant_mode="none",
        admission_policy="fcfs",
        prefix_cache_capacity=0,
    )
    await engine.start()
    api_module.ENGINE = engine
    try:
        yield engine
    finally:
        await engine.stop()
        api_module.ENGINE = None


@pytest.mark.asyncio(loop_scope="module")
async def test_health_returns_engine_config(shared_engine_and_client):
    transport = ASGITransport(app=api_module.app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["engine"]["batching_mode"] == "serial"


@pytest.mark.asyncio(loop_scope="module")
async def test_metrics_endpoint_returns_prometheus_text(shared_engine_and_client):
    transport = ASGITransport(app=api_module.app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers.get("content-type", "")
    assert b"nanoserve_active_seqs" in r.content


@pytest.mark.asyncio(loop_scope="module")
async def test_chat_completion_non_streaming(shared_engine_and_client):
    transport = ASGITransport(app=api_module.app)
    async with AsyncClient(
        transport=transport, base_url="http://test", timeout=60.0
    ) as ac:
        r = await ac.post(
            "/v1/chat/completions",
            json={
                "model": "tinyllama-1.1b-chat",
                "messages": [{"role": "user", "content": "Say hi briefly."}],
                "max_tokens": 8,
                "stream": False,
            },
        )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"]
    assert body["usage"]["completion_tokens"] >= 1
    assert body["choices"][0]["finish_reason"] in ("length", "stop")


@pytest.mark.asyncio(loop_scope="module")
async def test_chat_completion_streaming_emits_sse_chunks(shared_engine_and_client):
    transport = ASGITransport(app=api_module.app)
    async with AsyncClient(
        transport=transport, base_url="http://test", timeout=60.0
    ) as ac:
        async with ac.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "tinyllama-1.1b-chat",
                "messages": [{"role": "user", "content": "Say hi."}],
                "max_tokens": 6,
                "stream": True,
            },
        ) as r:
            assert r.status_code == 200
            assert "text/event-stream" in r.headers.get("content-type", "")
            chunks_seen: list[dict] = []
            saw_done = False
            async for line in r.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: [DONE]"):
                    saw_done = True
                    continue
                assert line.startswith("data: "), line
                chunks_seen.append(json.loads(line[len("data: ") :]))

    assert saw_done
    # opening role chunk + at least one content chunk + closing finish chunk
    assert len(chunks_seen) >= 3
    assert chunks_seen[0]["choices"][0]["delta"].get("role") == "assistant"
    assert chunks_seen[-1]["choices"][0].get("finish_reason") in ("length", "stop")


@pytest.mark.asyncio(loop_scope="module")
async def test_request_metrics_increment(shared_engine_and_client):
    transport = ASGITransport(app=api_module.app)
    async with AsyncClient(
        transport=transport, base_url="http://test", timeout=60.0
    ) as ac:
        # baseline scrape
        before = await ac.get("/metrics")
        # one tiny completion
        await ac.post(
            "/v1/chat/completions",
            json={
                "model": "tinyllama-1.1b-chat",
                "messages": [{"role": "user", "content": "Hi."}],
                "max_tokens": 4,
                "stream": False,
            },
        )
        after = await ac.get("/metrics")

    def _count(body: bytes) -> int:
        for line in body.decode().splitlines():
            if line.startswith("nanoserve_requests_total{") and 'status="ok"' in line:
                return int(float(line.rsplit(" ", 1)[-1]))
        return 0

    assert _count(after.content) > _count(before.content)


@pytest.mark.asyncio(loop_scope="module")
async def test_empty_messages_rejected(shared_engine_and_client):
    transport = ASGITransport(app=api_module.app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post(
            "/v1/chat/completions",
            json={
                "model": "tinyllama-1.1b-chat",
                "messages": [],
                "max_tokens": 4,
            },
        )
    assert r.status_code == 400
