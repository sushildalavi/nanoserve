"""L5-prefix parity: a sequence served from a prefix-cache hit must produce
the same greedy tokens as the same prompt served without the cache.

we use pre-tokenized prompt_ids (bypassing the chat template) to construct
an unambiguous strict-prefix relationship, since chat templates wrap
content with markers that break naive text-prefix → token-prefix
correspondence.

methodology:
1. baseline: run prompt P with cache OFF, record greedy output O_ref.
2. warm path: in a fresh engine with cache ON, first run a "prefix
   primer" P[:k] (which populates the cache for prefix P[:k]), then run
   the full P (which should hit the cache and only prefill P[k:]).
3. assert greedy output of (2) matches O_ref token-exact.
"""
from __future__ import annotations

import asyncio

import pytest

from nanoserve.config import TINYLLAMA_HF, tinyllama_nanoserve
from nanoserve.engine.engine import NanoServeEngine
from nanoserve.engine.service import SubmitRequest

MAX_NEW = 12


def _mps_available() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except ImportError:
        return False


@pytest.fixture(scope="module")
def model_and_tokenizer():
    if not _mps_available():
        pytest.skip("mps not available on this host")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TINYLLAMA_HF.path)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        TINYLLAMA_HF.path,
        dtype=torch.float16,
        attn_implementation="eager",
    ).to("mps")
    model.eval()
    model.generation_config.max_length = None
    return model, tok


def _tokenize(tok, text: str) -> list[int]:
    """raw tokenize, no chat template — used to build strict-prefix pairs."""
    return tok.encode(text, add_special_tokens=False)


async def _run_with_ids(
    prompt_ids: list[int],
    prefix_cache_capacity: int = 0,
    primer_ids: list[int] | None = None,
) -> tuple[list[int], dict]:
    spec = tinyllama_nanoserve(
        batching_mode="serial",
        max_batch_size=1,
        prefix_cache_capacity=prefix_cache_capacity,
    )
    engine = NanoServeEngine(
        spec,
        batching_mode="serial",
        max_batch_size=1,
        prefix_cache_capacity=prefix_cache_capacity,
    )
    await engine.start()
    try:
        # pre-warm with the primer (must be a strict prefix of prompt_ids).
        if primer_ids is not None:
            sid_p = await engine.submit(
                SubmitRequest(
                    prompt="", max_new_tokens=1, prompt_ids=primer_ids
                )
            )
            async for _ in engine.stream(sid_p):
                pass
            engine.retire(sid_p)

        sid = await engine.submit(
            SubmitRequest(prompt="", max_new_tokens=MAX_NEW, prompt_ids=prompt_ids)
        )
        async for _ in engine.stream(sid):
            pass
        seq = engine.get_finished_seq(sid)
        assert seq is not None
        engine.retire(sid)

        cache_meta = {}
        if engine.prefix_cache is not None:
            cache_meta = {
                "hits": engine.prefix_cache.hits,
                "misses": engine.prefix_cache.misses,
                "size": len(engine.prefix_cache),
            }
        return seq.output_ids, cache_meta
    finally:
        await engine.stop()


def test_l5_prefix_cache_hit_preserves_outputs(model_and_tokenizer):
    """run prompt P; then in a fresh engine, prime the cache with P[:k] and
    run P. the warm-path output must match the cold-path output token-exact.
    """
    _, tok = model_and_tokenizer

    # construct an unambiguous strict-prefix pair by tokenizing the full
    # text once and slicing. defeats trailing-space / merge-token weirdness
    # in the tokenizer.
    full_text = (
        "The quick brown fox jumps over the lazy dog. "
        "Tell me a fact about dogs."
    )
    prompt_ids = _tokenize(tok, full_text)
    primer_ids = prompt_ids[: max(1, len(prompt_ids) // 2)]
    assert prompt_ids[: len(primer_ids)] == primer_ids
    assert len(primer_ids) < len(prompt_ids)

    async def run_all():
        # cold baseline
        cold_out, _ = await _run_with_ids(prompt_ids, prefix_cache_capacity=0)
        # warm path: pre-warm cache with primer, then run full prompt
        warm_out, meta = await _run_with_ids(
            prompt_ids, prefix_cache_capacity=8, primer_ids=primer_ids
        )
        return cold_out, warm_out, meta

    cold_out, warm_out, meta = asyncio.run(run_all())

    assert meta["hits"] >= 1, (
        f"expected the warm prompt to hit the primer's cached prefix, "
        f"got hits={meta['hits']} misses={meta['misses']}"
    )
    assert warm_out == cold_out, (
        "L5-prefix parity: warm-path output diverged from cold baseline\n"
        f"  cold ({len(cold_out)}): {cold_out}\n"
        f"  warm ({len(warm_out)}): {warm_out}"
    )


def test_l5_prefix_cache_disabled_does_not_corrupt(model_and_tokenizer):
    """sanity: with cache disabled, behavior must be identical to a fresh
    engine — i.e. the cache wiring path doesn't accidentally affect
    cache-off runs.
    """
    _, tok = model_and_tokenizer
    prompt_ids = _tokenize(tok, "List two prime numbers.")

    async def run_all():
        a, _ = await _run_with_ids(prompt_ids, prefix_cache_capacity=0)
        b, _ = await _run_with_ids(prompt_ids, prefix_cache_capacity=0)
        return a, b

    a, b = asyncio.run(run_all())
    assert a == b, f"non-deterministic without cache: {a} vs {b}"
