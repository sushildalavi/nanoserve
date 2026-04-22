"""EOS retirement test: when one sequence in a batch stops before others,
the remaining sequences must still produce their correct isolated outputs
— the scheduler shrinks the batch and carries on.

this is the last 2B correctness gate before mid-batch admission. if this
passes, heterogeneous batches with different decode lengths work end-to-
end.

technique: submit two sequences concurrently with different max_new_tokens
(e.g. 4 vs 16). the short one hits max_tokens after 4 decode steps; the
long one must still produce the same tokens it would have in isolation.
"""
from __future__ import annotations

import asyncio

import pytest

from nanoserve.config import TINYLLAMA_HF
from nanoserve.engine.engine import NanoServeEngine
from nanoserve.engine.service import SubmitRequest


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


async def _isolated(model, tok, prompt: str, max_new: int) -> list[int]:
    engine = NanoServeEngine(TINYLLAMA_HF, max_batch_size=1, batching_mode="serial")
    engine._loop = asyncio.get_running_loop()
    engine._tokenizer = tok
    engine._model = model
    engine._start_driver()
    try:
        sid = await engine.submit(SubmitRequest(prompt=prompt, max_new_tokens=max_new))
        async for _ in engine.stream(sid):
            pass
        seq = engine.get_finished_seq(sid)
        assert seq is not None
        engine.retire(sid)
        return seq.output_ids
    finally:
        await engine.stop()


async def _batched_mixed(model, tok, specs: list[tuple[str, int]]) -> list[list[int]]:
    engine = NanoServeEngine(
        TINYLLAMA_HF, max_batch_size=len(specs), batching_mode="continuous"
    )
    engine._loop = asyncio.get_running_loop()
    engine._tokenizer = tok
    engine._model = model
    engine._start_driver()
    try:
        ids = await engine.submit_many(
            [SubmitRequest(prompt=p, max_new_tokens=n) for p, n in specs]
        )

        async def drain(sid: int):
            async for _ in engine.stream(sid):
                pass

        await asyncio.gather(*(drain(sid) for sid in ids))
        outs = []
        for sid in ids:
            seq = engine.get_finished_seq(sid)
            assert seq is not None
            outs.append(seq.output_ids)
            engine.retire(sid)
        return outs
    finally:
        await engine.stop()


def test_eos_retirement_short_seq_does_not_corrupt_long(model_and_tokenizer):
    """short_max=4, long_max=16. the short one finishes at step 4; the long
    one keeps decoding through step 16. its output must match isolated.
    """
    model, tok = model_and_tokenizer
    short_prompt = "Say hi."
    long_prompt = "Count from one to ten."

    async def run():
        long_ref = await _isolated(model, tok, long_prompt, max_new=16)
        short_ref = await _isolated(model, tok, short_prompt, max_new=4)
        short_out, long_out = await _batched_mixed(
            model, tok, [(short_prompt, 4), (long_prompt, 16)]
        )
        return short_ref, short_out, long_ref, long_out

    short_ref, short_out, long_ref, long_out = asyncio.run(run())

    assert short_out == short_ref, (
        f"short seq corrupted by batch\n  ref: {short_ref}\n  out: {short_out}"
    )
    assert long_out == long_ref, (
        f"long seq corrupted after short retired\n  ref: {long_ref}\n  out: {long_out}"
    )
