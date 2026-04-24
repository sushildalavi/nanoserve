"""L4-quant-int4 parity: INT4 weight-only quantized engine vs fp16 reference.

int4 is lossier than int8 — per-weight quantization error is ~9x larger
(scale / 2 vs scale / 2 with a scale that's 18x bigger since we pack into
[-7, 7] instead of [-127, 127]). we relax the overlap budget accordingly:
the first FIRST_K tokens must match on at least MIN_MATCH of the test
prompts. exact-first-K-tokens on every prompt is too tight for int4.
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
    "Name a common pet.",
    "What color is the sky?",
]
MAX_NEW = 16
FIRST_K = 2
MIN_MATCH = 3  # at least 3 of 5 prompts must agree on the first K tokens


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


async def _int4_ids(prompt: str) -> list[int]:
    spec = tinyllama_nanoserve(
        batching_mode="serial", max_batch_size=1, quant_mode="int4"
    )
    engine = NanoServeEngine(
        spec, max_batch_size=1, batching_mode="serial", quant_mode="int4"
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


def test_l4_quant_int4_close_to_fp16(guard):
    """int4 quantized greedy should agree with fp16 on the first FIRST_K
    tokens for at least MIN_MATCH of the test prompts. this is the int4
    counterpart of L4-quant — it catches catastrophic degradation (random
    tokens everywhere) while acknowledging that int4 won't be exact.
    """

    async def run():
        results = []
        for p in PROMPTS:
            fp = await _fp16_ids(p)
            qu = await _int4_ids(p)
            results.append((p, fp, qu))
        return results

    results = asyncio.run(run())

    matches = 0
    report = []
    for p, fp, qu in results:
        head_fp = fp[:FIRST_K]
        head_qu = qu[:FIRST_K]
        hit = head_fp == head_qu
        if hit:
            matches += 1
        report.append((p, head_fp, head_qu, hit))

    assert matches >= MIN_MATCH, (
        f"int4 quant diverged from fp16 on too many prompts "
        f"({matches}/{len(PROMPTS)} agreed on first {FIRST_K} tokens):\n"
        + "\n".join(
            f"  {'ok' if hit else 'MISS'}  {p!r}\n    fp16: {a}\n    int4: {b}"
            for p, a, b, hit in report
        )
    )
