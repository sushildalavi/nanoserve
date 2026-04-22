"""L2 parity: two sequences submitted concurrently into one engine must each
produce the same tokens as if they had been run in isolation.

this is the actual scheduler correctness test. L1 validates the model path;
L2 validates that interleaving scheduler state across multiple live
sequences doesn't corrupt any of them. past_kv must stay pinned to its
sequence, token emission must route to the right queue, and no sequence's
output must depend on what else is happening in the engine.

uses continuous batching_mode with max_batch_size=2 so the scheduler admits
both seqs simultaneously. forward pass is still one-at-a-time per step in
day 3 — that's the point: if L2 passes here, the scheduler state machine is
correct. L3 (day 4) will flip in true batched forward.
"""
from __future__ import annotations

import asyncio

import pytest

from nanoserve.config import TINYLLAMA_HF
from nanoserve.engine.engine import NanoServeEngine
from nanoserve.engine.service import SubmitRequest

PROMPTS = [
    "What is 2 + 2?",
    "List three primary colors.",
]
MAX_NEW = 16


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


async def _run_prompt_isolated(model, tok, prompt: str) -> list[int]:
    engine = NanoServeEngine(TINYLLAMA_HF, max_batch_size=1, batching_mode="serial")
    engine._loop = asyncio.get_running_loop()
    engine._tokenizer = tok
    engine._model = model
    engine._start_driver()
    try:
        seq_id = await engine.submit(
            SubmitRequest(prompt=prompt, max_new_tokens=MAX_NEW)
        )
        async for _ in engine.stream(seq_id):
            pass
        seq = engine.get_finished_seq(seq_id)
        assert seq is not None
        engine.retire(seq_id)
        return seq.output_ids
    finally:
        await engine.stop()


async def _run_both_interleaved(model, tok) -> dict[str, list[int]]:
    engine = NanoServeEngine(TINYLLAMA_HF, max_batch_size=2, batching_mode="continuous")
    engine._loop = asyncio.get_running_loop()
    engine._tokenizer = tok
    engine._model = model
    engine._start_driver()
    try:
        ids: dict[int, str] = {}
        for p in PROMPTS:
            sid = await engine.submit(
                SubmitRequest(prompt=p, max_new_tokens=MAX_NEW)
            )
            ids[sid] = p

        async def drain(seq_id: int):
            async for _ in engine.stream(seq_id):
                pass

        await asyncio.gather(*(drain(sid) for sid in ids))

        out: dict[str, list[int]] = {}
        for sid, p in ids.items():
            seq = engine.get_finished_seq(sid)
            assert seq is not None, f"seq {sid} did not finish"
            out[p] = seq.output_ids
            engine.retire(sid)
        return out
    finally:
        await engine.stop()


def test_l2_parity_two_interleaved_seqs(model_and_tokenizer):
    model, tok = model_and_tokenizer

    async def run():
        isolated = {}
        for p in PROMPTS:
            isolated[p] = await _run_prompt_isolated(model, tok, p)
        interleaved = await _run_both_interleaved(model, tok)
        return isolated, interleaved

    isolated, interleaved = asyncio.run(run())

    for p in PROMPTS:
        assert interleaved[p] == isolated[p], (
            f"L2 parity failed for prompt {p!r}\n"
            f"  isolated   ({len(isolated[p])} toks): {isolated[p]}\n"
            f"  interleaved({len(interleaved[p])} toks): {interleaved[p]}"
        )
