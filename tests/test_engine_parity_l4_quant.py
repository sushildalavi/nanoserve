"""L4-quant parity: INT8 weight-only quantized engine must produce outputs
close to the fp16 reference — not token-exact (quantization drifts), but
with high token overlap in the first K positions.

the metric: exact-match on the first N tokens of greedy decode. the
threshold is deliberately loose because even small per-row rounding
differences compound over 16 decode steps. a real production eval would
use perplexity or task accuracy. for a portfolio project the simpler
overlap test is enough to flag catastrophic breakage without requiring a
full eval harness.
"""
from __future__ import annotations

import asyncio

import pytest

from nanoserve.config import TINYLLAMA_HF, tinyllama_nanoserve
from nanoserve.engine.engine import NanoServeEngine
from nanoserve.engine.service import SubmitRequest

PROMPTS = [
    "What is 2 + 2?",
    "List three primary colors.",
    "Say the word cat.",
]
MAX_NEW = 16
FIRST_K = 4  # minimum number of leading tokens that must match


def _mps_available() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except ImportError:
        return False


@pytest.fixture(scope="module")
def guard():
    if not _mps_available():
        pytest.skip("mps not available on this host")


async def _fp16_ids(prompt: str) -> list[int]:
    engine = NanoServeEngine(
        TINYLLAMA_HF, max_batch_size=1, batching_mode="serial"
    )
    await engine.start()
    try:
        sid = await engine.submit(SubmitRequest(prompt=prompt, max_new_tokens=MAX_NEW))
        async for _ in engine.stream(sid):
            pass
        seq = engine.get_finished_seq(sid)
        assert seq is not None
        return seq.output_ids
    finally:
        await engine.stop()


async def _int8_ids(prompt: str) -> list[int]:
    spec = tinyllama_nanoserve(
        batching_mode="serial", max_batch_size=1, quant_mode="int8"
    )
    engine = NanoServeEngine(
        spec, max_batch_size=1, batching_mode="serial", quant_mode="int8"
    )
    await engine.start()
    try:
        sid = await engine.submit(SubmitRequest(prompt=prompt, max_new_tokens=MAX_NEW))
        async for _ in engine.stream(sid):
            pass
        seq = engine.get_finished_seq(sid)
        assert seq is not None
        return seq.output_ids
    finally:
        await engine.stop()


def test_l4_quant_close_to_fp16(guard):
    """int8 quantized greedy must agree with fp16 on the first FIRST_K tokens
    for every prompt. this is a sanity check — it catches major quant
    regressions without demanding exact parity (which int8 can't provide).
    """

    async def run():
        results = []
        for p in PROMPTS:
            fp = await _fp16_ids(p)
            qu = await _int8_ids(p)
            results.append((p, fp, qu))
        return results

    results = asyncio.run(run())

    failures = []
    for p, fp, qu in results:
        head_fp = fp[:FIRST_K]
        head_qu = qu[:FIRST_K]
        if head_fp != head_qu:
            failures.append((p, head_fp, head_qu))

    assert not failures, (
        f"int8 quant diverged from fp16 on first-{FIRST_K} tokens:\n"
        + "\n".join(
            f"  {p!r}\n    fp16: {a}\n    int8: {b}" for p, a, b in failures
        )
    )
